"""Native paper-trading loop: candles -> ema_atr_trend rules -> Gemini -> CSV.

TradingView is not required. Each cycle fetches live OHLCV, evaluates the same
regime-gated EMA crossover the Pine strategy uses, and asks Gemini to confirm
entries. Open positions are managed locally against the ATR stop, target, and
armed trailing stop.

    python -m core.strategy_runner --once
    python -m core.strategy_runner --poll-interval 60
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import (
    MAX_OPEN_POSITIONS,
    POSITION_SIZE_FRACTION,
    TRADING_PAIRS,
    empty_position_book,
    get_settings,
    legacy_open_slot,
    migrate_position_book,
)
from core.gemini_agent import Decision, GeminiAgent
from core.market_data import add_indicators, fetch_ohlcv, latest_snapshot
from strategies import load_strategy

log = logging.getLogger("runner")

STRATEGY_NAME = "ema_atr_trend"
CANDLE_LIMIT = 400

TRADE_LOG_FIELDS = [
    "timestamp", "symbol", "action", "entry_price", "stop_loss", "take_profit",
    "verdict", "confidence", "agent_action", "model", "alert_reason",
    "adx", "macro_ema", "loss_streak", "rationale",
]

TIMEFRAME_DELTAS = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),
}


@dataclass
class Signal:
    action: Literal["BUY", "SELL"]
    reason: str
    price: float
    atr: float
    stop: float
    target: float
    adx: float
    macro_ema: float


@dataclass
class Position:
    side: Literal["LONG", "SHORT"]
    entry_price: float
    entry_atr: float
    qty: float
    stop: float
    target: float
    trail_stop: float
    trail_armed: bool
    opened_bar: str
    confidence: float
    model: str | None
    rationale: str
    reason: str


@dataclass
class Exit:
    price: float
    reason: str
    hit: Literal["stop", "target"]


@dataclass
class RunnerState:
    equity: float
    equity_history: list[float] = field(default_factory=list)
    loss_streak: int = 0
    breaker_active: bool = False
    breaker_bars: int = 0
    last_bar: str | None = None
    #: Per-symbol book. Each value is at least {"status": "FLAT"|"LONG"|"SHORT"}.
    positions: dict[str, dict[str, Any]] = field(default_factory=empty_position_book)
    #: Closed-bar timestamp last processed per symbol (ISO).
    last_bars: dict[str, str] = field(default_factory=dict)
    #: Legacy single-slot mirror. None means the book is flat.
    position: dict[str, Any] | None = None


@dataclass
class CycleReport:
    symbol: str
    timeframe: str
    exchange: str
    bar: str | None
    close: float | None
    new_bar: bool
    signal: str | None
    signal_reason: str | None
    regime: dict[str, Any]
    breaker_active: bool
    loss_streak: int
    position: str
    verdict: str | None
    reason: str
    gemini_action: str | None = None
    gemini_confidence: float | None = None
    gemini_rationale: str | None = None
    model: str | None = None
    stop: float | None = None
    target: float | None = None
    logged_to: str | None = None
    equity: float | None = None
    legs: list["CycleReport"] = field(default_factory=list)
    book: dict[str, str] = field(default_factory=dict)


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    try:
        return TIMEFRAME_DELTAS[timeframe]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe {timeframe!r}") from exc


def last_closed_ts(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the open-time of the most recently *closed* candle.

    ccxt includes the in-progress bar as the last row. Signals must not fire on
    it: the Pine strategy evaluates `ta.crossover` on confirmed closes.
    """
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    delta = timeframe_delta(timeframe)
    last_open = df.index[-1]
    if last_open + delta <= now:
        return last_open
    if len(df) < 2:
        raise RuntimeError("Need at least two candles to identify a closed bar")
    return df.index[-2]


def _finite(series: pd.Series, *names: str) -> bool:
    return all(name in series.index and pd.notna(series[name]) for name in names)


def detect_signal(closed: pd.DataFrame, params: dict[str, Any], breaker_active: bool) -> Signal | None:
    """EMA 21/55 crossover on the last closed bar, gated by RSI, EMA 200, ADX."""
    if breaker_active or len(closed) < 2:
        return None

    prev, curr = closed.iloc[-2], closed.iloc[-1]
    needed = ("ema_fast", "ema_slow", "rsi", "adx", "ema_macro", "atr", "close")
    if not _finite(prev, "ema_fast", "ema_slow") or not _finite(curr, *needed):
        return None

    close = float(curr["close"])
    atr = float(curr["atr"])
    adx = float(curr["adx"])
    macro = float(curr["ema_macro"])
    rsi = float(curr["rsi"])

    use_macro = bool(params.get("use_macro_filter", True))
    use_adx = bool(params.get("use_adx_filter", True))
    adx_min = float(params.get("adx_min", 20))
    rsi_floor = float(params.get("rsi_floor", 50))
    rsi_ceil = float(params.get("rsi_ceiling", 50))
    stop_mult = float(params["atr_stop_mult"])
    target_mult = float(params["atr_target_mult"])

    macro_bull = (not use_macro) or close > macro
    macro_bear = (not use_macro) or close < macro
    trend_strong = (not use_adx) or adx > adx_min

    crossed_up = (
        float(prev["ema_fast"]) <= float(prev["ema_slow"])
        and float(curr["ema_fast"]) > float(curr["ema_slow"])
    )
    crossed_down = (
        float(prev["ema_fast"]) >= float(prev["ema_slow"])
        and float(curr["ema_fast"]) < float(curr["ema_slow"])
    )

    if crossed_up and rsi > rsi_floor and macro_bull and trend_strong:
        return Signal(
            action="BUY",
            reason="macro_aligned_trend_long",
            price=close,
            atr=atr,
            stop=close - atr * stop_mult,
            target=close + atr * target_mult,
            adx=adx,
            macro_ema=macro,
        )

    allow_shorts = bool(params.get("allow_shorts", True))
    if allow_shorts and crossed_down and rsi < rsi_ceil and macro_bear and trend_strong:
        return Signal(
            action="SELL",
            reason="macro_aligned_trend_short",
            price=close,
            atr=atr,
            stop=close + atr * stop_mult,
            target=close - atr * target_mult,
            adx=adx,
            macro_ema=macro,
        )
    return None


def update_trail(position: Position, close: float, atr: float, params: dict[str, Any]) -> Position:
    """Arm and ratchet the trailing stop the same way the Pine script does.

    ATR is frozen at entry for the initial stop and the hard target. Once price
    is `trail_activate_mult` ATR in profit, the stop trails `atr_trail_mult`
    of *live* ATR and only ever ratchets.
    """
    arm_mult = float(params["trail_activate_mult"])
    trail_mult = float(params["atr_trail_mult"])

    if position.side == "LONG":
        if close >= position.entry_price + position.entry_atr * arm_mult:
            position.trail_armed = True
            position.trail_stop = max(position.trail_stop, close - atr * trail_mult)
    else:
        if close <= position.entry_price - position.entry_atr * arm_mult:
            position.trail_armed = True
            position.trail_stop = min(position.trail_stop, close + atr * trail_mult)
    return position


def maybe_exit(
    position: Position,
    high: float,
    low: float,
    close: float,
    atr: float,
    params: dict[str, Any],
) -> Exit | None:
    """Check stop then target. If both print on the same bar, assume the stop.

    Pessimistic same-bar resolution is the conservative paper-trading choice;
    TradingView's fill when both are touched is not knowable from OHLC alone.
    """
    update_trail(position, close, atr, params)

    if position.side == "LONG":
        if low <= position.trail_stop:
            reason = "atr_trail_long" if position.trail_armed else "initial_stop_or_target_long"
            return Exit(price=position.trail_stop, reason=reason, hit="stop")
        if high >= position.target:
            return Exit(price=position.target, reason="initial_stop_or_target_long", hit="target")
    else:
        if high >= position.trail_stop:
            reason = "atr_trail_short" if position.trail_armed else "initial_stop_or_target_short"
            return Exit(price=position.trail_stop, reason=reason, hit="stop")
        if low <= position.target:
            return Exit(price=position.target, reason="initial_stop_or_target_short", hit="target")
    return None


def position_qty(equity: float, price: float, atr: float, params: dict[str, Any]) -> float:
    """Risk-based size with a notional ceiling, matching the Pine inputs."""
    if price <= 0:
        return 0.0
    if not params.get("use_risk_sizing", True):
        return equity * float(params.get("position_pct", 25)) / 100.0 / price

    stop_distance = atr * float(params["atr_stop_mult"])
    risk_capital = equity * float(params["risk_per_trade_pct"]) / 100.0
    qty_by_risk = risk_capital / stop_distance if stop_distance > 0 else 0.0
    qty_ceiling = equity * float(params["max_notional_pct"]) / 100.0 / price
    return min(qty_by_risk, qty_ceiling)


def close_pnl(position: Position, exit_price: float, commission_pct: float) -> float:
    direction = 1.0 if position.side == "LONG" else -1.0
    gross = position.qty * (exit_price - position.entry_price) * direction
    fee = (position.qty * position.entry_price + position.qty * exit_price) * commission_pct / 100.0
    return gross - fee


def rolling_drawdown_pct(history: list[float], lookback: int) -> float:
    window = history[-lookback:] if lookback > 0 else history
    if not window:
        return 0.0
    peak = max(window)
    equity = window[-1]
    if peak <= 0:
        return 0.0
    return (peak - equity) / peak * 100.0


def apply_breaker(
    state: RunnerState,
    params: dict[str, Any],
    trend_strong: bool,
    new_bar: bool,
) -> RunnerState:
    if not params.get("use_circuit_breaker", True):
        state.breaker_active = False
        state.breaker_bars = 0
        return state
    if not new_bar:
        return state

    lookback = int(params.get("equity_lookback_bars", 500))
    max_dd = float(params.get("max_rolling_drawdown_pct", 15.0))
    max_streak = int(params.get("max_loss_streak", 4))
    cooldown = int(params.get("breaker_cooldown_bars", 24))
    equity_breach = rolling_drawdown_pct(state.equity_history, lookback) > max_dd

    if not state.breaker_active and (state.loss_streak >= max_streak or equity_breach):
        state.breaker_active = True
        state.breaker_bars = 0
    elif state.breaker_active:
        state.breaker_bars += 1
        if state.breaker_bars >= cooldown and trend_strong and not equity_breach:
            state.breaker_active = False
            state.loss_streak = 0
            state.breaker_bars = 0
    return state


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in TRADE_LOG_FIELDS})


def _position_from_dict(raw: dict[str, Any] | None) -> Position | None:
    """Rehydrate a Position, ignoring book-only keys like status/symbol."""
    if not raw:
        return None
    status = str(raw.get("status") or raw.get("side") or "FLAT").upper()
    if status in {"", "FLAT"}:
        return None
    payload = dict(raw)
    payload.setdefault("side", status)
    allowed = {item.name for item in fields(Position)}
    try:
        return Position(**{key: payload[key] for key in allowed if key in payload})
    except TypeError:
        return None


def count_open_positions(positions: dict[str, dict[str, Any]]) -> int:
    n = 0
    for slot in positions.values():
        status = str(slot.get("status") or slot.get("side") or "FLAT").upper()
        if status not in {"", "FLAT"}:
            n += 1
    return n


def fraction_qty(equity: float, price: float, fraction: float) -> float:
    """Notional = fraction * equity. Used instead of ATR risk sizing for the book."""
    if price <= 0 or fraction <= 0 or equity <= 0:
        return 0.0
    return equity * fraction / price


def _book_status(positions: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        symbol: str(slot.get("status") or slot.get("side") or "FLAT").upper()
        for symbol, slot in positions.items()
    }


def _bar_ts(bar: pd.Series) -> str:
    ts = bar.name
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return str(ts)


class StrategyRunner:
    def __init__(
        self,
        *,
        agent: GeminiAgent | None = None,
        fetch_fn: Callable[..., pd.DataFrame] | None = None,
        indicate_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        params: dict[str, Any] | None = None,
        strategy_context: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        exchange_id: str | None = None,
        min_confidence: float | None = None,
        starting_equity: float | None = None,
        candle_limit: int = CANDLE_LIMIT,
        trade_log: Path | None = None,
        reject_log: Path | None = None,
        state_path: Path | None = None,
        pairs: tuple[str, ...] | None = None,
        fetch_delay: float | None = None,
    ) -> None:
        self._settings = None
        self.fetch_fn = fetch_fn or fetch_ohlcv
        self.indicate_fn = indicate_fn or add_indicators
        self.candle_limit = candle_limit
        self._agent = agent
        self._params = params
        self._strategy_context = strategy_context

        self.symbol = symbol if symbol is not None else self.settings.symbol
        if pairs is not None:
            self.pairs = tuple(pairs)
            self.max_open_positions = MAX_OPEN_POSITIONS
            self.position_size_fraction = POSITION_SIZE_FRACTION
        else:
            self.pairs = tuple(self.settings.trading_pairs or TRADING_PAIRS)
            self.max_open_positions = int(self.settings.max_open_positions or MAX_OPEN_POSITIONS)
            self.position_size_fraction = float(
                self.settings.position_size_fraction or POSITION_SIZE_FRACTION
            )
        self.fetch_delay = 1.0 if fetch_delay is None else max(0.0, float(fetch_delay))
        self.timeframe = timeframe if timeframe is not None else self.settings.timeframe
        self.exchange_id = exchange_id if exchange_id is not None else self.settings.exchange_id
        self.min_confidence = (
            min_confidence if min_confidence is not None else self.settings.min_confidence
        )
        self.starting_equity = (
            starting_equity
            if starting_equity is not None
            else self.settings.paper_starting_balance
        )
        data_dir = self.settings.paths["data"] if trade_log is None else trade_log.parent
        self.trade_log = trade_log or (data_dir / "paper_trades.csv")
        self.reject_log = reject_log or (data_dir / "rejected_alerts.csv")
        self.state_path = state_path or (self.settings.paths["root"] / "state" / "runner.json")
        self.state = self._load_state()

    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def params(self) -> dict[str, Any]:
        if self._params is None:
            self._params = load_strategy(STRATEGY_NAME).params
        return self._params

    def strategy_context(self) -> str:
        if self._strategy_context is None:
            self._strategy_context = load_strategy(STRATEGY_NAME).prompt_context()
        return self._strategy_context

    def get_agent(self) -> GeminiAgent:
        if self._agent is None:
            self._agent = GeminiAgent()
        return self._agent

    def _fetch_pair(self, symbol: str) -> pd.DataFrame:
        return self.indicate_fn(
            self.fetch_fn(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=self.candle_limit,
                exchange_id=self.exchange_id,
            )
        )

    def _load_state(self) -> RunnerState:
        pairs = self.pairs
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            positions = migrate_position_book(raw, pairs)
            last_bars = dict(raw.get("last_bars") or {})
            if raw.get("last_bar") and not last_bars:
                last_bars = {symbol: raw["last_bar"] for symbol in pairs}
            return RunnerState(
                equity=float(raw.get("equity", self.starting_equity)),
                equity_history=list(raw.get("equity_history") or []),
                loss_streak=int(raw.get("loss_streak") or 0),
                breaker_active=bool(raw.get("breaker_active") or False),
                breaker_bars=int(raw.get("breaker_bars") or 0),
                last_bar=raw.get("last_bar"),
                last_bars=last_bars,
                positions=positions,
                position=legacy_open_slot(positions),
            )
        return RunnerState(equity=self.starting_equity, positions=empty_position_book(pairs))

    def save_state(self) -> None:
        if not self.state.positions:
            self.state.positions = empty_position_book(self.pairs)
        self.state.position = legacy_open_slot(self.state.positions)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _base_row(
        self,
        action: str,
        price: float | None,
        signal: Signal | None,
        extra: dict[str, Any],
        symbol: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "timestamp": now,
            "symbol": symbol or self.symbol,
            "action": action,
            "entry_price": price,
            "stop_loss": extra.get("stop_loss", signal.stop if signal else ""),
            "take_profit": extra.get("take_profit", signal.target if signal else ""),
            "verdict": extra.get("verdict", ""),
            "confidence": extra.get("confidence", ""),
            "agent_action": extra.get("agent_action", ""),
            "model": extra.get("model", ""),
            "alert_reason": extra.get("alert_reason", signal.reason if signal else ""),
            "adx": extra.get("adx", signal.adx if signal else ""),
            "macro_ema": extra.get("macro_ema", signal.macro_ema if signal else ""),
            "loss_streak": self.state.loss_streak,
            "rationale": extra.get("rationale", ""),
        }

    def _close_position(self, symbol: str, position: Position, exit_event: Exit, bar: pd.Series) -> str:
        pnl = close_pnl(position, exit_event.price, float(self.params.get("commission_pct", 0.075)))
        self.state.equity += pnl
        self.state.loss_streak = self.state.loss_streak + 1 if pnl < 0 else 0
        self.state.positions[symbol] = {"status": "FLAT"}
        self.state.position = legacy_open_slot(self.state.positions)
        adx = bar["adx"] if "adx" in bar.index else None
        macro = bar["ema_macro"] if "ema_macro" in bar.index else None
        append_row(
            self.trade_log,
            self._base_row(
                "CLOSE",
                exit_event.price,
                None,
                {
                    "verdict": "CONFIRMED",
                    "stop_loss": position.trail_stop,
                    "take_profit": position.target,
                    "alert_reason": exit_event.reason,
                    "adx": None if adx is None or pd.isna(adx) else float(adx),
                    "macro_ema": None if macro is None or pd.isna(macro) else float(macro),
                    "rationale": (
                        f"exit {position.side} entry={position.entry_price:.4f} "
                        f"fill={exit_event.price:.4f} pnl={pnl:.2f} hit={exit_event.hit}"
                    ),
                },
                symbol=symbol,
            ),
        )
        return str(self.trade_log)

    def _open_position(self, symbol: str, signal: Signal, decision: Decision, opened_bar: str) -> Position:
        stop = decision.stop_loss if decision.stop_loss is not None else signal.stop
        target = decision.take_profit if decision.take_profit is not None else signal.target
        qty = fraction_qty(self.state.equity, signal.price, self.position_size_fraction)
        side: Literal["LONG", "SHORT"] = "LONG" if signal.action == "BUY" else "SHORT"
        position = Position(
            side=side,
            entry_price=signal.price,
            entry_atr=signal.atr,
            qty=qty,
            stop=float(stop),
            target=float(target),
            trail_stop=float(stop),
            trail_armed=False,
            opened_bar=opened_bar,
            confidence=decision.confidence,
            model=getattr(self.get_agent(), "last_model_used", None),
            rationale=decision.rationale,
            reason=signal.reason,
        )
        self.state.positions[symbol] = {**asdict(position), "status": side, "symbol": symbol}
        self.state.position = legacy_open_slot(self.state.positions)
        return position

    def _confirm(self, signal: Signal, snapshot: dict[str, Any]) -> tuple[str, str, Decision | None]:
        """Return (verdict, reason, decision). Fail closed if the agent is down."""
        try:
            decision = self.get_agent().decide(snapshot, self.strategy_context())
        except Exception as exc:  # noqa: BLE001
            log.error("agent unavailable, declining entry: %s", exc)
            return "REJECTED", "agent_unavailable", None

        agrees = decision.action == signal.action
        confident = decision.confidence >= self.min_confidence
        if agrees and confident:
            return "CONFIRMED", "agent_agrees", decision
        if not agrees:
            return "REJECTED", f"agent_returned_{decision.action}", decision
        return (
            "REJECTED",
            f"confidence_{decision.confidence:.2f}_below_{self.min_confidence:.2f}",
            decision,
        )

    def _exit_bars(self, position: Position, last: pd.Series, live: pd.Series) -> list[pd.Series]:
        """Bars whose range can close `position`, excluding the bar it opened on."""
        bars: list[pd.Series] = []
        opened = position.opened_bar
        if _bar_ts(last) != opened:
            bars.append(last)
        if live.name != last.name and _bar_ts(live) != opened:
            bars.append(live)
        return bars

    def cycle(self, *, consult_always: bool = False, now: pd.Timestamp | None = None) -> CycleReport:
        """Evaluate every pair, manage exits, then open new slots up to the cap."""
        if not self.state.positions:
            self.state.positions = empty_position_book(self.pairs)
        for symbol in self.pairs:
            self.state.positions.setdefault(symbol, {"status": "FLAT"})

        prepared: list[dict[str, Any]] = []
        for index, symbol in enumerate(self.pairs):
            if index > 0 and self.fetch_delay > 0:
                time.sleep(self.fetch_delay)
            frame = self._fetch_pair(symbol)
            closed_ts = last_closed_ts(frame, self.timeframe, now=now)
            closed = frame.loc[:closed_ts]
            live = frame.iloc[-1]
            last = closed.iloc[-1]
            snapshot = latest_snapshot(closed)
            snapshot["tradingview_alert"] = None
            bar_iso = closed_ts.isoformat()
            use_adx = bool(self.params.get("use_adx_filter", True))
            adx_min = float(self.params.get("adx_min", 20))
            prepared.append(
                {
                    "symbol": symbol,
                    "closed_ts": closed_ts,
                    "closed": closed,
                    "live": live,
                    "last": last,
                    "snapshot": snapshot,
                    "bar_iso": bar_iso,
                    "new_bar": self.state.last_bars.get(symbol) != bar_iso,
                    "trend_strong": (not use_adx)
                    or (pd.notna(last.get("adx")) and float(last["adx"]) > adx_min),
                    "signal": None,
                    "position": _position_from_dict(self.state.positions.get(symbol)),
                    "verdict": None,
                    "reason": "flat_no_signal",
                    "gemini": None,
                    "logged": None,
                }
            )

        for item in prepared:
            position = item["position"]
            if position is None:
                continue
            exit_event = None
            exit_bar = item["live"]
            for bar in self._exit_bars(position, item["last"], item["live"]):
                atr = float(bar["atr"]) if pd.notna(bar.get("atr")) else position.entry_atr
                exit_event = maybe_exit(
                    position,
                    high=float(bar["high"]),
                    low=float(bar["low"]),
                    close=float(bar["close"]),
                    atr=atr,
                    params=self.params,
                )
                self.state.positions[item["symbol"]] = {
                    **asdict(position),
                    "status": position.side,
                    "symbol": item["symbol"],
                }
                if exit_event is not None:
                    exit_bar = bar
                    break
            if exit_event is not None:
                item["logged"] = self._close_position(item["symbol"], position, exit_event, exit_bar)
                item["position"] = None
                item["verdict"] = "CONFIRMED"
                item["reason"] = exit_event.reason
            else:
                item["reason"] = "holding"

        any_new_bar = any(item["new_bar"] for item in prepared)
        trend_strong = any(item["trend_strong"] for item in prepared)
        if any_new_bar:
            self.state.equity_history.append(self.state.equity)
            lookback = int(self.params.get("equity_lookback_bars", 500))
            if len(self.state.equity_history) > lookback:
                self.state.equity_history = self.state.equity_history[-lookback:]
        apply_breaker(self.state, self.params, trend_strong, any_new_bar)

        consulted = False
        for item in prepared:
            if item["position"] is not None:
                continue
            signal = detect_signal(item["closed"], self.params, self.state.breaker_active)
            item["signal"] = signal
            if (
                signal is None
                and self.state.breaker_active
                and detect_signal(item["closed"], self.params, False) is not None
            ):
                item["reason"] = "circuit_breaker_active"

            if signal is not None and item["new_bar"]:
                if count_open_positions(self.state.positions) >= self.max_open_positions:
                    item["reason"] = "max_open_positions"
                    continue
                snapshot = item["snapshot"]
                snapshot["tradingview_alert"] = {
                    "action": signal.action,
                    "reason": signal.reason,
                    "strategy_stop": signal.stop,
                    "strategy_target": signal.target,
                    "loss_streak": self.state.loss_streak,
                    "breaker_active": self.state.breaker_active,
                }
                verdict, reason, gemini = self._confirm(signal, snapshot)
                consulted = True
                item["verdict"] = verdict
                item["reason"] = reason
                item["gemini"] = gemini
                extra = {
                    "verdict": verdict,
                    "confidence": None if gemini is None else round(gemini.confidence, 4),
                    "agent_action": None if gemini is None else gemini.action,
                    "model": getattr(self.get_agent(), "last_model_used", None),
                    "rationale": reason if gemini is None else gemini.rationale,
                    "alert_reason": signal.reason,
                    "adx": signal.adx,
                    "macro_ema": signal.macro_ema,
                    "stop_loss": signal.stop if gemini is None else (gemini.stop_loss or signal.stop),
                    "take_profit": signal.target if gemini is None else (gemini.take_profit or signal.target),
                }
                if verdict == "CONFIRMED" and gemini is not None:
                    position = self._open_position(item["symbol"], signal, gemini, item["bar_iso"])
                    item["position"] = position
                    extra["stop_loss"] = position.stop
                    extra["take_profit"] = position.target
                    append_row(
                        self.trade_log,
                        self._base_row(
                            signal.action, signal.price, signal, extra, symbol=item["symbol"]
                        ),
                    )
                    item["logged"] = str(self.trade_log)
                else:
                    append_row(
                        self.reject_log,
                        self._base_row(
                            signal.action, signal.price, signal, extra, symbol=item["symbol"]
                        ),
                    )
                    item["logged"] = str(self.reject_log)
            elif signal is not None:
                item["reason"] = "signal_already_processed"

        consult_gemini: Decision | None = next(
            (item["gemini"] for item in prepared if item["gemini"] is not None), None
        )
        if consult_always and not consulted:
            snapshot = prepared[0]["snapshot"] if prepared else {}
            try:
                consult_gemini = self.get_agent().decide(snapshot, self.strategy_context())
                for item in prepared:
                    if item["gemini"] is None:
                        item["gemini"] = consult_gemini
                    if item["verdict"] is None:
                        item["verdict"] = "HOLD"
                    if item["reason"] == "flat_no_signal":
                        item["reason"] = "no_entry_signal"
            except Exception as exc:  # noqa: BLE001
                log.error("agent unavailable on dry-run consult: %s", exc)
                for item in prepared:
                    if item["verdict"] is None:
                        item["verdict"] = "REJECTED"
                        item["reason"] = "agent_unavailable"

        latest_closed: pd.Timestamp | None = None
        for item in prepared:
            if item["new_bar"]:
                self.state.last_bars[item["symbol"]] = item["bar_iso"]
            if latest_closed is None or item["closed_ts"] > latest_closed:
                latest_closed = item["closed_ts"]
        if latest_closed is not None:
            self.state.last_bar = latest_closed.isoformat()
        self.save_state()

        legs = [self._leg_report(item) for item in prepared]
        primary = next((leg for leg in legs if leg.symbol == self.symbol), legs[0] if legs else None)
        if primary is None:
            raise RuntimeError("cycle produced no pair reports")
        primary.legs = legs
        primary.book = _book_status(self.state.positions)
        primary.equity = self.state.equity
        primary.breaker_active = self.state.breaker_active
        primary.loss_streak = self.state.loss_streak
        return primary

    def _leg_report(self, item: dict[str, Any]) -> CycleReport:
        signal: Signal | None = item["signal"]
        position: Position | None = item["position"]
        gemini: Decision | None = item["gemini"]
        last: pd.Series = item["last"]
        snapshot: dict[str, Any] = item["snapshot"]
        return CycleReport(
            symbol=item["symbol"],
            timeframe=self.timeframe,
            exchange=self.exchange_id,
            bar=item["bar_iso"],
            close=float(last["close"]),
            new_bar=item["new_bar"],
            signal=None if signal is None else signal.action,
            signal_reason=None if signal is None else signal.reason,
            regime=snapshot.get("regime") or {},
            breaker_active=self.state.breaker_active,
            loss_streak=self.state.loss_streak,
            position="FLAT" if position is None else position.side,
            verdict=item["verdict"],
            reason=item["reason"],
            gemini_action=None if gemini is None else gemini.action,
            gemini_confidence=None if gemini is None else gemini.confidence,
            gemini_rationale=None if gemini is None else gemini.rationale,
            model=getattr(self._agent, "last_model_used", None),
            stop=None if position is None else position.trail_stop,
            target=None if position is None else position.target,
            logged_to=item["logged"],
            equity=self.state.equity,
        )


def _format_leg(report: CycleReport) -> list[str]:
    regime = report.regime
    conf = "" if report.gemini_confidence is None else f"{report.gemini_confidence:.2f} "
    rationale = (report.gemini_rationale or "")[:180]
    return [
        f"bar        : {report.bar} ({'new close' if report.new_bar else 'already processed'})",
        f"close      : {report.close}",
        (
            f"regime     : {regime.get('macro_trend')} / {regime.get('trend_strength')}"
            f" (ADX {regime.get('adx')}, EMA200 {regime.get('macro_ema')},"
            f" dir={regime.get('tradeable_direction')})"
        ),
        f"signal     : {report.signal or 'none'}"
        + (f" ({report.signal_reason})" if report.signal_reason else ""),
        f"position   : {report.position}"
        + (f"  stop={report.stop} target={report.target}" if report.stop is not None else ""),
        f"gemini     : {report.gemini_action or '—'} {conf}{rationale}".rstrip(),
        f"result     : {report.verdict or '—'} ({report.reason})",
        f"logged     : {report.logged_to or '(nothing written)'}",
    ]


def format_report(report: CycleReport) -> str:
    equity = "n/a" if report.equity is None else f"{report.equity:.2f}"
    universe = " ".join(report.book) if report.book else report.symbol
    lines = [
        f"=== Gemini Trend Guard | {universe} {report.timeframe} @ {report.exchange} ===",
        f"breaker    : {'ON' if report.breaker_active else 'off'} (streak {report.loss_streak})",
        f"equity     : {equity}",
        f"model      : {report.model or '—'}",
    ]
    if report.book:
        book = " ".join(f"{symbol}={status}" for symbol, status in report.book.items())
        lines.append(f"book       : {book}")
    if len(report.legs) > 1:
        for leg in report.legs:
            lines.append(f"--- {leg.symbol} ---")
            lines.extend(_format_leg(leg))
    else:
        lines.extend(_format_leg(report))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Gemini Trend Guard paper-trading runner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single cycle against the latest closed candle, then exit. Always consults Gemini.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        metavar="SECONDS",
        help=(
            "Seconds between cycles in continuous mode (default: 60). Entries fire "
            "only on a new closed 1h bar; polls also catch intra-bar stop/target hits."
        ),
    )
    parser.add_argument("--symbol", default=None, help="Override SYMBOL (default from .env, BTC/USDT)")
    parser.add_argument("--timeframe", default=None, help="Override TIMEFRAME (default 1h)")
    parser.add_argument("--exchange", default=None, help="Override EXCHANGE_ID (default kraken)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    runner = StrategyRunner(
        symbol=args.symbol,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        pairs=(args.symbol,) if args.symbol else None,
    )

    def run_once(consult_always: bool) -> CycleReport:
        report = runner.cycle(consult_always=consult_always)
        print(format_report(report))
        return report

    if args.once:
        run_once(consult_always=True)
        return 0

    print(
        f"Polling every {args.poll_interval}s | {' '.join(runner.pairs)} {runner.timeframe} "
        f"@ {runner.exchange_id} | cap={runner.max_open_positions} | Ctrl+C to stop"
    )
    while True:
        try:
            run_once(consult_always=False)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0
        except Exception as exc:  # noqa: BLE001
            log.exception("cycle failed: %s", exc)
            print(f"[error] {type(exc).__name__}: {exc}")
        time.sleep(max(1, args.poll_interval))


if __name__ == "__main__":
    sys.exit(main())
