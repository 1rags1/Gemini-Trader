"""Paper execution server: TradingView alert -> Gemini review -> CSV trade log.

The Pine strategy decides *when* to trade; this server decides *whether* to act
on it. An alert arrives, the agent re-examines live market data, and the trade
is either CONFIRMED (written to data/paper_trades.csv) or REJECTED (written to
data/rejected_alerts.csv, so the veto rate stays measurable).

    python -m core.webhook_server

Point the TradingView alert webhook at http://<host>:5000/webhook and set the
message body to {{strategy.order.alert_message}}.
"""

from __future__ import annotations

import csv
import logging
import secrets
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

if __package__ in (None, ""):  # allow `python core/webhook_server.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from core.config import get_settings
from core.exposure import assert_secret_for_public_bind, request_from_localhost
from core.gemini_agent import GeminiAgent
from core.market_data import add_indicators, fetch_ohlcv, latest_snapshot
from core.net import enable_os_trust_store
from strategies import list_strategies, load_strategy

log = logging.getLogger("webhook")

#: EMA 200 is the longest lookback in the indicator set, so anything less than
#: this leaves the macro-trend gate unresolved and forces a HOLD.
SNAPSHOT_CANDLES = 400

#: Longest-suffix-first, so BTCUSDT resolves to BTC/USDT rather than BTCU/SDT.
QUOTE_CURRENCIES = ("USDT", "USDC", "TUSD", "USD", "EUR", "GBP", "BTC", "ETH")

#: TradingView's {{interval}} token uses minute counts and letter codes.
TRADINGVIEW_INTERVALS = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "45": "45m", "60": "1h", "120": "2h", "180": "3h", "240": "4h",
    "D": "1d", "1D": "1d", "W": "1w", "1W": "1w", "M": "1M", "1M": "1M",
}

TRADE_LOG_FIELDS = [
    "timestamp", "symbol", "action", "entry_price", "stop_loss", "take_profit",
    "verdict", "confidence", "agent_action", "model", "alert_reason",
    "adx", "macro_ema", "loss_streak", "rationale",
]


class TradingViewAlert(BaseModel):
    """The JSON emitted by the Pine strategy's alert_message.

    Extra fields are allowed so adding a key in Pine cannot start returning 422
    to TradingView, which does not retry failed webhook deliveries.
    """

    model_config = ConfigDict(extra="allow")

    strategy: str
    action: Literal["BUY", "SELL", "CLOSE"]
    ticker: str
    timeframe: str | None = None
    price: float | None = None
    atr: float | None = None
    stop: float | None = None
    target: float | None = None
    reason: str | None = None
    adx: float | None = None
    macro_ema: float | None = None
    loss_streak: int | None = None
    breaker_active: bool | None = None
    position_size: float | None = None
    bar_time: str | None = None


class ExecutionResult(BaseModel):
    verdict: Literal["CONFIRMED", "REJECTED"]
    reason: str
    symbol: str
    action: str
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    agent_action: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    model: str | None = None
    logged_to: str | None = None
    regime: dict[str, Any] | None = Field(default=None)


def to_ccxt_symbol(ticker: str) -> str:
    """`BINANCE:BTCUSDT` -> `BTC/USDT`."""
    raw = ticker.split(":")[-1].upper().replace("PERP", "")
    if "/" in raw:
        return raw
    for quote in QUOTE_CURRENCIES:
        if raw.endswith(quote) and len(raw) > len(quote):
            return f"{raw[: -len(quote)]}/{quote}"
    return raw


def to_ccxt_timeframe(interval: str | None) -> str | None:
    if not interval:
        return None
    return TRADINGVIEW_INTERVALS.get(str(interval).strip().upper())


# A single agent is reused across requests; building one per alert would rebuild
# the HTTP client and its TLS context every time.
_agent: GeminiAgent | None = None
_agent_lock = threading.Lock()
_csv_lock = threading.Lock()


def get_agent() -> GeminiAgent:
    global _agent
    with _agent_lock:
        if _agent is None:
            enable_os_trust_store()
            _agent = GeminiAgent()
        return _agent


def trade_log_path() -> Path:
    return get_settings().paths["data"] / "paper_trades.csv"


def reject_log_path() -> Path:
    return get_settings().paths["data"] / "rejected_alerts.csv"


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one row, writing the header when the file is new."""
    with _csv_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in TRADE_LOG_FIELDS})


def build_snapshot(alert: TradingViewAlert) -> dict[str, Any]:
    """Fresh market snapshot for the alert's symbol.

    Falls back to the alert's own fields when the exchange is unreachable, so a
    data outage degrades the agent's context instead of dropping the alert.
    """
    symbol = to_ccxt_symbol(alert.ticker)
    try:
        frame = add_indicators(
            fetch_ohlcv(
                symbol=symbol,
                timeframe=to_ccxt_timeframe(alert.timeframe),
                limit=SNAPSHOT_CANDLES,
            )
        )
        snapshot = latest_snapshot(frame)
        snapshot["data_source"] = "exchange"
    except Exception as exc:  # noqa: BLE001 - a data outage must not drop the alert
        log.warning("Market data unavailable for %s (%s); using alert fields", symbol, exc)
        snapshot = {
            "symbol": symbol,
            "timeframe": alert.timeframe,
            "as_of": alert.bar_time,
            "last_close": alert.price,
            "regime": {
                "macro_ema": alert.macro_ema,
                "macro_trend": (
                    "unknown"
                    if alert.macro_ema is None or alert.price is None
                    else ("bull" if alert.price > alert.macro_ema else "bear")
                ),
                "adx": alert.adx,
                "adx_min": 20.0,
                "trend_strength": (
                    "unknown" if alert.adx is None
                    else ("strong" if alert.adx > 20.0 else "weak")
                ),
                "tradeable_direction": "unknown",
            },
            "indicators": {"atr": alert.atr},
            "data_source": "alert_payload",
        }

    snapshot["tradingview_alert"] = {
        "action": alert.action,
        "reason": alert.reason,
        "strategy_stop": alert.stop,
        "strategy_target": alert.target,
        "loss_streak": alert.loss_streak,
        "breaker_active": alert.breaker_active,
        "position_size": alert.position_size,
    }
    return snapshot


def strategy_context() -> str:
    names = list_strategies()
    return load_strategy(names[0]).prompt_context() if names else ""


def evaluate(alert: TradingViewAlert) -> ExecutionResult:
    """Decide whether to execute `alert`, and record the outcome."""
    cfg = get_settings()
    symbol = to_ccxt_symbol(alert.ticker)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    base = {
        "timestamp": now,
        "symbol": symbol,
        "action": alert.action,
        "entry_price": alert.price,
        "stop_loss": alert.stop,
        "take_profit": alert.target,
        "alert_reason": alert.reason,
        "adx": alert.adx,
        "macro_ema": alert.macro_ema,
        "loss_streak": alert.loss_streak,
    }

    # An exit is risk-reducing, so it is never put to a vote. Letting the model
    # veto a close could strand an open position.
    if alert.action == "CLOSE":
        result = ExecutionResult(
            verdict="CONFIRMED",
            reason="exit_not_subject_to_review",
            symbol=symbol,
            action=alert.action,
            entry_price=alert.price,
            stop_loss=alert.stop,
            take_profit=alert.target,
            logged_to=str(trade_log_path()),
        )
        append_row(trade_log_path(), {**base, "verdict": result.verdict, "rationale": result.reason})
        return result

    # The strategy's own circuit breaker outranks the model.
    if alert.breaker_active:
        result = ExecutionResult(
            verdict="REJECTED",
            reason="circuit_breaker_active",
            symbol=symbol,
            action=alert.action,
            logged_to=str(reject_log_path()),
        )
        append_row(reject_log_path(), {**base, "verdict": result.verdict, "rationale": result.reason})
        return result

    snapshot = build_snapshot(alert)
    agent = get_agent()

    # Fail closed. If the agent cannot be reached we decline the entry rather
    # than executing unreviewed, and we answer 200 so TradingView - which does
    # not retry webhooks - does not silently drop the alert.
    try:
        decision = agent.decide(snapshot, strategy_context())
    except Exception as exc:  # noqa: BLE001
        log.error("agent unavailable, declining entry: %s", exc)
        result = ExecutionResult(
            verdict="REJECTED",
            reason="agent_unavailable",
            symbol=symbol,
            action=alert.action,
            rationale=str(exc)[:300],
            logged_to=str(reject_log_path()),
            regime=snapshot.get("regime"),
        )
        append_row(
            reject_log_path(),
            {**base, "verdict": result.verdict, "rationale": result.reason},
        )
        return result

    agrees = decision.action == alert.action
    confident = decision.confidence >= cfg.min_confidence

    if agrees and confident:
        verdict, reason = "CONFIRMED", "agent_agrees"
    elif not agrees:
        verdict, reason = "REJECTED", f"agent_returned_{decision.action}"
    else:
        verdict = "REJECTED"
        reason = f"confidence_{decision.confidence:.2f}_below_{cfg.min_confidence:.2f}"

    # Prefer the agent's own levels when it confirms; they are ATR-derived from
    # the same rules the Pine strategy uses.
    entry = alert.price
    stop = decision.stop_loss if decision.stop_loss is not None else alert.stop
    target = decision.take_profit if decision.take_profit is not None else alert.target

    result = ExecutionResult(
        verdict=verdict,
        reason=reason,
        symbol=symbol,
        action=alert.action,
        entry_price=entry,
        stop_loss=stop if verdict == "CONFIRMED" else None,
        take_profit=target if verdict == "CONFIRMED" else None,
        agent_action=decision.action,
        confidence=decision.confidence,
        rationale=decision.rationale,
        model=agent.last_model_used,
        logged_to=str(trade_log_path() if verdict == "CONFIRMED" else reject_log_path()),
        regime=snapshot.get("regime"),
    )

    append_row(
        trade_log_path() if verdict == "CONFIRMED" else reject_log_path(),
        {
            **base,
            "stop_loss": stop,
            "take_profit": target,
            "verdict": verdict,
            "confidence": round(decision.confidence, 4),
            "agent_action": decision.action,
            "model": agent.last_model_used,
            "rationale": decision.rationale,
        },
    )
    return result


app = FastAPI(
    title="Gemini Trader paper execution server",
    description="Reviews TradingView strategy alerts with Gemini before simulating a fill.",
    version="1.0.0",
)


def check_token(request: Request, token: str | None) -> None:
    """Empty WEBHOOK_SECRET is allowed only for a localhost client.

    A tunnel header or any other peer must present the secret. When a secret
    is configured, every caller must present it, including localhost.
    """
    secret = get_settings().webhook_secret
    if secret:
        if not token or not secrets.compare_digest(token, secret):
            raise HTTPException(status_code=401, detail="invalid or missing token")
        return
    if request_from_localhost(request):
        return
    raise HTTPException(
        status_code=401,
        detail=(
            "WEBHOOK_SECRET is required for requests that are not from localhost. "
            "Set WEBHOOK_SECRET before exposing this endpoint through a tunnel or a public bind."
        ),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = get_settings()
    return {
        "status": "ok",
        "model": cfg.gemini_model,
        "min_confidence": cfg.min_confidence,
        "auth_required": bool(cfg.webhook_secret),
        "trade_log": str(trade_log_path()),
    }


@app.post("/webhook", response_model=ExecutionResult)
def webhook(
    alert: TradingViewAlert,
    request: Request,
    token: str | None = Query(default=None),
) -> ExecutionResult:
    check_token(request, token)
    log.info("alert: %s %s (%s)", alert.action, alert.ticker, alert.reason)
    result = evaluate(alert)
    log.info("verdict: %s (%s)", result.verdict, result.reason)
    return result


@app.get("/trades")
def trades(
    request: Request,
    limit: int = Query(default=20, ge=1, le=500),
    token: str | None = Query(default=None),
) -> dict[str, Any]:
    check_token(request, token)
    path = trade_log_path()
    if not path.exists():
        return {"count": 0, "trades": []}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {"count": len(rows), "trades": rows[-limit:]}


def main() -> int:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = get_settings()
    try:
        assert_secret_for_public_bind(
            host=cfg.webhook_host,
            secret=cfg.webhook_secret,
            service="the TradingView webhook",
            secret_name="WEBHOOK_SECRET",
        )
    except RuntimeError as exc:
        print(f"Refusing to start: {exc}", file=sys.stderr)
        return 1
    if cfg.webhook_secret:
        auth = "token required"
    else:
        auth = "open on localhost only (set WEBHOOK_SECRET before a tunnel or public bind)"
    print(f"Paper execution server on http://{cfg.webhook_host}:{cfg.webhook_port}/webhook")
    print(f"  model          : {cfg.gemini_model}")
    print(f"  min confidence : {cfg.min_confidence}")
    print(f"  auth           : {auth}")
    print(f"  trade log      : {trade_log_path()}")
    uvicorn.run(app, host=cfg.webhook_host, port=cfg.webhook_port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
