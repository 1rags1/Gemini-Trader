"""Read-only dashboard for the paper / live Kraken runner.

Serves a single Tailwind page on 127.0.0.1:8050 and a JSON snapshot the browser
polls every 10 seconds. Binding another host requires DASHBOARD_SECRET (or
WEBHOOK_SECRET). File reads are short-lived and shared, so this process never
locks `state/runner.json` or the trade CSV away from the runner.

    python -m core.dashboard
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from core.config import (
    ALLOW_LIVE_TRADING,
    MACRO_TIMEFRAME,
    MAX_OPEN_POSITIONS,
    PAPER_TRADING,
    PROJECT_ROOT,
    TRADING_PAIRS,
    TRIGGER_TIMEFRAME,
    classify_book,
    get_settings,
    migrate_circuit_breaker,
    migrate_position_book,
)
from core.order_lifecycle import live_orders_permitted
from core.market_data import (
    add_macro_indicators,
    add_trigger_indicators,
    classify_macro_regime,
    fetch_ohlcv,
)
from core.exposure import assert_secret_for_public_bind, request_from_localhost
from core.net import enable_os_trust_store

log = logging.getLogger("dashboard")

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
TRADE_LIMIT = 15
MARKET_CACHE_TTL = 20.0
MACRO_CANDLES = 250
TRIGGER_CANDLES = 100
MARKET_FETCH_DELAY = 1.0

_FILL_RE = re.compile(r"fill=([-\d.]+)")
_ENTRY_RE = re.compile(r"entry=([-\d.]+)")
_PNL_RE = re.compile(r"pnl=([-\d.]+)")
_HIT_RE = re.compile(r"hit=(\w+)")

_market_lock = threading.Lock()
_market_cache: dict[str, Any] = {"fetched_at": 0.0, "payload": None, "error": None}


def _read_shared(path: Path) -> str:
    """Read a file and drop the handle immediately. No exclusive lock."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read()


def _cfg():
    try:
        return get_settings()
    except Exception:
        return None


def is_paper_trading() -> bool:
    cfg = _cfg()
    if cfg is not None:
        return bool(cfg.paper_trading)
    return bool(PAPER_TRADING)


def paper_starting_balance() -> float:
    cfg = _cfg()
    if cfg is not None:
        try:
            return float(cfg.paper_starting_balance)
        except (TypeError, ValueError):
            pass
    return 10_000.0


def allow_live_trading() -> bool:
    """Second live gate. Default false. Does not override paper mode."""
    cfg = _cfg()
    if cfg is None:
        return bool(ALLOW_LIVE_TRADING)
    return bool(getattr(cfg, "allow_live_trading", ALLOW_LIVE_TRADING))


def live_trading_active() -> bool:
    """Real Kraken orders are armed only when both gates are open."""
    return live_orders_permitted(is_paper_trading(), allow_live_trading())


def desk_labels(*, paper: bool | None = None) -> dict[str, Any]:
    """Header copy. Live is active only when both trading gates are open."""
    paper_mode = (not live_trading_active()) if paper is None else bool(paper)
    if paper_mode:
        return {
            "paper_trading": True,
            "active_mode": "paper",
            "active_banner": "ACTIVE: PAPER TRADING",
            "desk_title": "Paper desk",
            "equity_label": "PAPER EQUITY",
            "live_badge": None,
            "trade_log_empty": "No paper trades yet",
        }
    return {
        "paper_trading": False,
        "active_mode": "live",
        "active_banner": "ACTIVE: LIVE TRADING",
        "desk_title": "Live Execution Desk",
        "equity_label": "LIVE KRAKEN EQUITY",
        "live_badge": "● LIVE KRAKEN (USD)",
        "trade_log_empty": "No live trades yet",
    }


def dashboard_secret() -> str:
    """Dashboard token, falling back to the webhook secret.

    If settings cannot load (for example a missing Gemini key), still honor
    DASHBOARD_SECRET / WEBHOOK_SECRET from the environment so a public bind
    does not lose its lock.
    """
    cfg = _cfg()
    if cfg is not None:
        return (getattr(cfg, "dashboard_secret", None) or cfg.webhook_secret or "").strip()
    return (os.getenv("DASHBOARD_SECRET") or os.getenv("WEBHOOK_SECRET") or "").strip()


def check_token(request: Request, token: str | None) -> None:
    """Localhost is open. A tunnel or any other peer needs a dashboard secret."""
    if request_from_localhost(request):
        return
    secret = dashboard_secret()
    if not secret:
        raise HTTPException(
            status_code=401,
            detail=(
                "DASHBOARD_SECRET or WEBHOOK_SECRET is required off localhost. "
                "Set one before binding 0.0.0.0 or opening a tunnel."
            ),
        )
    if not token or not secrets.compare_digest(token, secret):
        raise HTTPException(status_code=401, detail="invalid or missing token")


def _trade_log_name(*, paper: bool | None = None) -> str:
    paper_mode = (not live_trading_active()) if paper is None else bool(paper)
    return "paper_trades.csv" if paper_mode else "live_trades.csv"


def _paths() -> dict[str, Path]:
    cfg = _cfg()
    trade_name = _trade_log_name()
    if cfg is not None:
        return {
            "state": cfg.paths["root"] / "state" / "runner.json",
            "trades": cfg.paths["data"] / trade_name,
            "rejects": cfg.paths["data"] / "rejected_alerts.csv",
        }
    return {
        "state": PROJECT_ROOT / "state" / "runner.json",
        "trades": PROJECT_ROOT / "data" / trade_name,
        "rejects": PROJECT_ROOT / "data" / "rejected_alerts.csv",
    }


def _pairs() -> tuple[str, ...]:
    cfg = _cfg()
    if cfg is not None and getattr(cfg, "trading_pairs", None):
        return tuple(cfg.trading_pairs)
    return TRADING_PAIRS


def _max_open() -> int:
    cfg = _cfg()
    if cfg is not None and getattr(cfg, "max_open_positions", None):
        return int(cfg.max_open_positions)
    return MAX_OPEN_POSITIONS


def _slot_status(slot: dict[str, Any] | None) -> str:
    if not isinstance(slot, dict):
        return "FLAT"
    return str(slot.get("status") or slot.get("side") or "FLAT").upper() or "FLAT"


def _position_cards(
    book: dict[str, dict[str, Any]],
    last_bars: dict[str, Any],
    pairs: tuple[str, ...],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for symbol in pairs:
        slot = book.get(symbol) if isinstance(book, dict) else None
        status = _slot_status(slot)
        open_slot = dict(slot) if isinstance(slot, dict) and status not in {"", "FLAT"} else None
        if open_slot is not None:
            open_slot.setdefault("side", status)
        cards.append(
            {
                "symbol": symbol,
                "status": status if status not in {"", "FLAT"} else "FLAT",
                "position": open_slot,
                "last_bar": last_bars.get(symbol),
                "macro_regime": str((slot or {}).get("macro_regime") or "UNKNOWN").upper(),
            }
        )
    return cards


def _resolve_start_equity(raw: dict[str, Any], equity_f: float | None, paper: bool) -> float | None:
    """Prefer persisted live deposit baseline over PAPER_STARTING_BALANCE."""
    if raw.get("start_equity") is not None:
        try:
            return float(raw["start_equity"])
        except (TypeError, ValueError):
            pass
    history = raw.get("equity_history")
    if isinstance(history, list) and history:
        try:
            return float(history[0])
        except (TypeError, ValueError):
            pass
    if not paper and equity_f is not None:
        return equity_f
    cfg = _cfg()
    if cfg is not None:
        return float(cfg.paper_starting_balance)
    return 10_000.0


def _book_paths() -> dict[str, Path]:
    paths = _paths()
    state = paths["state"]
    parent = state.parent
    trades = paths.get("trades")
    data_dir = trades.parent if isinstance(trades, Path) else parent
    return {
        "runner": state,
        "paper": paths.get("paper_state") or (parent / "paper.json"),
        "live": paths.get("live_state") or (parent / "live.json"),
        "paper_trades": paths.get("paper_trades") or (data_dir / "paper_trades.csv"),
        "live_trades": paths.get("live_trades") or (data_dir / "live_trades.csv"),
        "rejects": paths.get("rejects") or (data_dir / "rejected_alerts.csv"),
    }


def _read_raw(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(_read_shared(path))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("state unreadable %s: %s", path, exc)
        return None
    return raw if isinstance(raw, dict) else None


def _book_metrics(raw: dict[str, Any] | None) -> dict[str, Any]:
    pairs = _pairs()
    if not isinstance(raw, dict):
        return {
            "exists": False,
            "equity": None,
            "starting_equity": None,
            "pnl": None,
            "pnl_pct": None,
            "positions": [],
            "open_count": 0,
            "position": None,
            "position_side": "FLAT",
            "loss_streak": 0,
            "breaker_active": False,
            "breaker_bars": 0,
            "last_bar": None,
            "last_bars": {},
            "max_open_positions": _max_open(),
        }
    try:
        equity_f = float(raw["equity"]) if raw.get("equity") is not None else None
    except (TypeError, ValueError):
        equity_f = None
    try:
        starting = float(raw["start_equity"]) if raw.get("start_equity") is not None else None
    except (TypeError, ValueError):
        starting = None
    if starting is None:
        starting = _resolve_start_equity(raw, equity_f, True)
    pnl = None if equity_f is None or starting is None else equity_f - starting
    last_bars = dict(raw.get("last_bars") or {})
    cards = _position_cards(migrate_position_book(raw, pairs), last_bars, pairs)
    first_open = next((card["position"] for card in cards if card["position"] is not None), None)
    side = "FLAT" if first_open is None else str(first_open.get("side") or first_open.get("status") or "FLAT").upper()
    breaker = migrate_circuit_breaker(raw)
    return {
        "exists": True,
        "equity": equity_f,
        "starting_equity": starting,
        "pnl": pnl,
        "pnl_pct": None if pnl is None or not starting else (pnl / starting) * 100.0,
        "positions": cards,
        "open_count": sum(1 for card in cards if card["status"] != "FLAT"),
        "position": first_open,
        "position_side": side,
        "loss_streak": int(breaker["loss_streak"]),
        "breaker_active": bool(breaker["tripped"]),
        "breaker_bars": int(breaker.get("cooldown_bars") or raw.get("breaker_bars") or 0),
        "last_bar": raw.get("last_bar"),
        "last_bars": last_bars,
        "max_open_positions": _max_open(),
    }


def _select_books(baseline: float, *, active_live: bool) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Paper raw and live raw. A live deposit is never returned as the paper book."""
    paths = _book_paths()
    runner = _read_raw(paths["runner"])
    paper_file = _read_raw(paths["paper"])
    live_file = _read_raw(paths["live"])
    runner_kind = classify_book(runner, baseline)

    paper_raw = None
    if not active_live and runner is not None and runner_kind != "live":
        paper_raw = runner
    elif paper_file is not None and classify_book(paper_file, baseline) != "live":
        paper_raw = paper_file
    elif runner is not None and runner_kind == "paper":
        paper_raw = runner

    live_raw = None
    if runner is not None and runner_kind == "live":
        live_raw = runner
    elif live_file is not None and classify_book(live_file, baseline) != "paper":
        live_raw = live_file
    elif active_live and runner is not None and runner_kind != "paper":
        live_raw = runner
    return paper_raw, live_raw


_BALANCE_TTL = 30.0
_balance_cache: dict[str, Any] = {"at": 0.0, "payload": None}


def peek_live_balance() -> dict[str, Any]:
    """Read-only Kraken USD balance. Never places an order."""
    now = time.monotonic()
    cached = _balance_cache.get("payload")
    if isinstance(cached, dict) and now - float(_balance_cache.get("at") or 0) < _BALANCE_TTL:
        return cached
    payload: dict[str, Any] = {"ok": False, "equity": None}
    cfg = _cfg()
    key = (getattr(cfg, "exchange_api_key", "") or "").strip() if cfg is not None else ""
    secret = (getattr(cfg, "exchange_api_secret", "") or "").strip() if cfg is not None else ""
    if cfg is not None and key and secret:
        try:
            from core.broker import fetch_quote_balance

            payload = {
                "ok": True,
                "equity": float(fetch_quote_balance(cfg.exchange_id, key, secret)),
            }
        except Exception as exc:  # noqa: BLE001
            log.info("live balance unavailable: %s", type(exc).__name__)
            payload = {"ok": False, "equity": None}
    _balance_cache["at"] = now
    _balance_cache["payload"] = payload
    return payload


def _paper_panel(raw: dict[str, Any] | None, *, in_use: bool, baseline: float) -> dict[str, Any]:
    metrics = _book_metrics(raw)
    if not metrics["exists"]:
        metrics = _book_metrics(None)
        metrics["positions"] = _position_cards({}, {}, _pairs())
        metrics["equity"] = baseline
        metrics["starting_equity"] = baseline
        metrics["pnl"] = 0.0
        metrics["pnl_pct"] = 0.0
        note = f"Configured practice balance ${baseline:,.2f}. No paper fills yet."
        balance_state = "configured"
    else:
        note = "Practice book · fake money"
        balance_state = "book"
    metrics.update(
        {
            "title": "PRACTICE / PAPER",
            "money": "Fake money",
            "in_use": in_use,
            "use_label": "IN USE" if in_use else "NOT IN USE",
            "idle": not in_use,
            "equity_label": "PAPER EQUITY",
            "balance_note": note,
            "balance_state": balance_state,
            "placeholder": None,
            "status_line": (
                ("Practice book is running. " if in_use else "Practice book is idle. ")
                + (
                    f"Circuit breaker ON · loss streak {metrics['loss_streak']}."
                    if metrics.get("breaker_active")
                    else f"Circuit breaker off · loss streak {metrics.get('loss_streak', 0)}."
                )
            ),
        }
    )
    return metrics


def _live_panel(
    raw: dict[str, Any] | None,
    *,
    in_use: bool,
    peeked: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = _book_metrics(raw)
    peeked_equity = None
    if isinstance(peeked, dict) and peeked.get("ok") and peeked.get("equity") is not None:
        try:
            peeked_equity = float(peeked["equity"])
        except (TypeError, ValueError):
            peeked_equity = None

    if in_use and metrics["exists"]:
        note = "Live Kraken equity. Real money."
        balance_state = "book"
        placeholder = None
    elif in_use and peeked_equity is not None:
        metrics["equity"] = peeked_equity
        metrics["starting_equity"] = peeked_equity
        metrics["pnl"] = 0.0
        metrics["pnl_pct"] = 0.0
        note = "Kraken free USD. Real money."
        balance_state = "exchange"
        placeholder = None
    elif not in_use and peeked_equity is not None:
        start = metrics["starting_equity"]
        metrics["equity"] = peeked_equity
        if start is not None:
            metrics["pnl"] = peeked_equity - start
            metrics["pnl_pct"] = None if not start else ((peeked_equity - start) / start) * 100.0
        note = "Kraken free USD, read-only. Idle — no live orders."
        balance_state = "exchange"
        placeholder = None
    elif metrics["exists"] and metrics["equity"] is not None:
        note = "Last known Kraken USD. Idle — no live orders are firing."
        balance_state = "last_known"
        placeholder = None
    else:
        metrics = _book_metrics(None)
        note = "Live not active"
        balance_state = "unavailable"
        placeholder = "Live not active"
    if not in_use and balance_state == "unavailable":
        metrics["positions"] = []
    metrics.update(
        {
            "title": "LIVE / KRAKEN",
            "money": "Real money",
            "in_use": in_use,
            "use_label": "IN USE" if in_use else "NOT IN USE",
            "idle": not in_use,
            "equity_label": "LIVE KRAKEN EQUITY",
            "balance_note": note,
            "balance_state": balance_state,
            "placeholder": placeholder,
            "status_line": (
                "Real money is in use. " if in_use else "IDLE — live orders are not firing. "
            )
            + (
                f"Circuit breaker ON · loss streak {metrics.get('loss_streak', 0)}."
                if metrics.get("exists") and metrics.get("breaker_active")
                else (
                    f"Circuit breaker off · loss streak {metrics.get('loss_streak', 0)}."
                    if metrics.get("exists")
                    else ""
                )
            ),
        }
    )
    return metrics


def build_account_panels(*, peek_balance: bool = False) -> dict[str, Any]:
    """Paper practice book and live Kraken book, side by side."""
    baseline = paper_starting_balance()
    armed = live_trading_active()
    paper_flag = is_paper_trading()
    paper_raw, live_raw = _select_books(baseline, active_live=armed)
    peeked = peek_live_balance() if peek_balance and not armed else None
    if armed:
        banner = "ACTIVE: LIVE TRADING"
        detail = (
            "Real Kraken money is in use. The practice book is idle. "
            "This dashboard does not place orders."
        )
    elif paper_flag:
        banner = "ACTIVE: PAPER TRADING"
        detail = (
            f"Practice book is running from ${baseline:,.2f} fake money. "
            "Live Kraken is idle — no live orders are firing."
        )
    else:
        banner = "ACTIVE: PAPER TRADING"
        detail = (
            "Live orders are not armed. PAPER_TRADING is off and "
            "ALLOW_LIVE_TRADING is off, so nothing is sent to Kraken."
        )
    return {
        "active_mode": "live" if armed else "paper",
        "active_banner": banner,
        "active_detail": detail,
        "paper_trading": not armed,
        "allow_live_trading": allow_live_trading(),
        "live_armed": armed,
        "paper": _paper_panel(paper_raw, in_use=bool(paper_flag) and not armed, baseline=baseline),
        "live": _live_panel(live_raw, in_use=armed, peeked=peeked),
    }


def _flatten_active(panels: dict[str, Any]) -> dict[str, Any]:
    active = panels["live"] if panels["active_mode"] == "live" else panels["paper"]
    labels = desk_labels(paper=panels["active_mode"] != "live")
    return {
        "exists": active.get("exists"),
        "equity": active.get("equity"),
        "starting_equity": active.get("starting_equity"),
        "pnl": active.get("pnl"),
        "pnl_pct": active.get("pnl_pct"),
        "loss_streak": active.get("loss_streak", 0),
        "breaker_active": active.get("breaker_active", False),
        "breaker_bars": active.get("breaker_bars", 0),
        "last_bar": active.get("last_bar"),
        "last_bars": active.get("last_bars") or {},
        "position_side": active.get("position_side", "FLAT"),
        "position": active.get("position"),
        "positions": active.get("positions") or [],
        "open_count": active.get("open_count", 0),
        "max_open_positions": active.get("max_open_positions", _max_open()),
        **labels,
        "active_mode": panels["active_mode"],
        "active_banner": panels["active_banner"],
        "active_detail": panels["active_detail"],
        "live_armed": panels["live_armed"],
        "paper": panels["paper"],
        "live": panels["live"],
    }


def _empty_state(*, error: str | None = None) -> dict[str, Any]:
    pairs = _pairs()
    labels = desk_labels()
    empty: dict[str, Any] = {
        "exists": False,
        "equity": None,
        "starting_equity": None,
        "pnl": None,
        "pnl_pct": None,
        "loss_streak": 0,
        "breaker_active": False,
        "breaker_bars": 0,
        "last_bar": None,
        "last_bars": {},
        "position_side": "FLAT",
        "position": None,
        "positions": _position_cards({}, {}, pairs),
        "open_count": 0,
        "max_open_positions": _max_open(),
        **labels,
    }
    if error:
        empty["error"] = error
    return empty


def read_runner_state() -> dict[str, Any]:
    """Active book, plus separate paper and live panels. No exchange orders."""
    return _flatten_active(build_account_panels(peek_balance=False))


def _f(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify(row: dict[str, str]) -> str:
    """Map a log row to a badge tone: green | red | amber | slate."""
    action = (row.get("action") or "").upper()
    verdict = (row.get("verdict") or "").upper()
    reason = f"{row.get('alert_reason') or ''} {row.get('rationale') or ''}".lower()
    hit_match = _HIT_RE.search(reason)
    hit = hit_match.group(1) if hit_match else ""

    if verdict == "REJECTED" or hit == "stop" or "stop_loss" in reason:
        return "red"
    if action == "SELL" and verdict != "CONFIRMED":
        return "red"
    if action == "SELL":
        return "red"
    if action == "BUY" or hit == "target" or "profit" in reason:
        return "green"
    if verdict == "CONFIRMED":
        return "green"
    return "slate"


def _enrich_trade(row: dict[str, str]) -> dict[str, Any]:
    rationale = row.get("rationale") or ""
    fill_m = _FILL_RE.search(rationale)
    entry_m = _ENTRY_RE.search(rationale)
    pnl_m = _PNL_RE.search(rationale)
    hit_m = _HIT_RE.search(rationale)
    action = (row.get("action") or "").upper()
    logged_price = _f(row.get("entry_price"))
    entry = _f(entry_m.group(1)) if entry_m else (logged_price if action != "CLOSE" else None)
    exit_price = _f(fill_m.group(1)) if fill_m else (logged_price if action == "CLOSE" else None)
    return {
        "timestamp": row.get("timestamp") or "",
        "symbol": row.get("symbol") or "",
        "action": action,
        "verdict": (row.get("verdict") or "").upper(),
        "tone": _classify(row),
        "entry_price": entry,
        "exit_price": exit_price,
        "stop_loss": _f(row.get("stop_loss")),
        "take_profit": _f(row.get("take_profit")),
        "confidence": _f(row.get("confidence")),
        "agent_action": row.get("agent_action") or "",
        "model": row.get("model") or "",
        "alert_reason": row.get("alert_reason") or "",
        "rationale": rationale,
        "pnl": _f(pnl_m.group(1)) if pnl_m else None,
        "hit": hit_m.group(1) if hit_m else "",
        "adx": _f(row.get("adx")),
        "loss_streak": row.get("loss_streak") or "",
    }


def read_trades(limit: int = TRADE_LIMIT, path: Path | None = None) -> dict[str, Any]:
    path = path or _paths()["trades"]
    if not path.exists() or path.stat().st_size == 0:
        return {"count": 0, "trades": []}
    try:
        text = _read_shared(path)
        rows = list(csv.DictReader(io.StringIO(text)))
    except OSError as exc:
        log.warning("trade log unreadable: %s", exc)
        return {"count": 0, "trades": [], "error": str(exc)}
    recent = rows[-limit:]
    recent.reverse()
    return {"count": len(rows), "trades": [_enrich_trade(r) for r in recent]}


def read_reject_count() -> int:
    path = _paths()["rejects"]
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        text = _read_shared(path)
        return max(0, text.count("\n") - 1)
    except OSError:
        return 0


def _num(bar: Any, name: str) -> float | None:
    try:
        value = float(bar[name])
    except Exception:
        return None
    if value != value:
        return None
    return value


def _empty_pair_metrics(symbol: str, error: str | None = None) -> dict[str, Any]:
    return {
        "ok": error is None,
        "symbol": symbol,
        "macro_regime": "CHOP",
        "as_of": None,
        "last_close": None,
        "ema_fast": None,
        "ema_slow": None,
        "atr": None,
        "adx": None,
        "macro_close": None,
        "macro_ema_fast": None,
        "macro_ema_slow": None,
        "macro_ema_trend": None,
        "error": error,
    }


def _pair_metrics(symbol: str, exchange: str, *, paced: bool) -> dict[str, Any]:
    try:
        enable_os_trust_store()
        if paced and MARKET_FETCH_DELAY > 0:
            time.sleep(MARKET_FETCH_DELAY)
        macro = add_macro_indicators(
            fetch_ohlcv(
                symbol=symbol,
                timeframe=MACRO_TIMEFRAME,
                limit=MACRO_CANDLES,
                exchange_id=exchange,
            )
        )
        if MARKET_FETCH_DELAY > 0:
            time.sleep(MARKET_FETCH_DELAY)
        trigger = add_trigger_indicators(
            fetch_ohlcv(
                symbol=symbol,
                timeframe=TRIGGER_TIMEFRAME,
                limit=TRIGGER_CANDLES,
                exchange_id=exchange,
            )
        )
        macro_last = macro.iloc[-2] if len(macro) >= 2 else macro.iloc[-1]
        trigger_last = trigger.iloc[-2] if len(trigger) >= 2 else trigger.iloc[-1]
        regime = classify_macro_regime(macro_last)
        badge = "CHOP" if regime == "NEUTRAL" else regime
        return {
            "ok": True,
            "symbol": symbol,
            "macro_regime": badge,
            "as_of": trigger_last.name.isoformat() if hasattr(trigger_last.name, "isoformat") else str(trigger_last.name),
            "last_close": _num(trigger_last, "close"),
            "ema_fast": _num(trigger_last, "ema_fast"),
            "ema_slow": _num(trigger_last, "ema_slow"),
            "atr": _num(trigger_last, "atr"),
            "adx": _num(macro_last, "adx"),
            "macro_close": _num(macro_last, "close"),
            "macro_ema_fast": _num(macro_last, "ema_fast"),
            "macro_ema_slow": _num(macro_last, "ema_slow"),
            "macro_ema_trend": _num(macro_last, "ema_macro"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("market fetch failed for %s: %s", symbol, exc)
        return _empty_pair_metrics(symbol, error=f"{type(exc).__name__}: {exc}")


def fetch_market() -> dict[str, Any]:
    """Latest 1h + 15m snapshot per pair, cached so polls do not hammer Kraken."""
    now = time.monotonic()
    with _market_lock:
        age = now - float(_market_cache["fetched_at"])
        if _market_cache["payload"] is not None and age < MARKET_CACHE_TTL:
            return _market_cache["payload"]

    cfg = _cfg()
    pairs = _pairs()
    exchange = cfg.exchange_id if cfg is not None else "kraken"

    rows: list[dict[str, Any]] = []
    for index, symbol in enumerate(pairs):
        rows.append(_pair_metrics(symbol, exchange, paced=index > 0))

    errors = [row["error"] for row in rows if row.get("error")]
    payload = {
        "ok": any(row.get("ok") for row in rows),
        "timeframe": f"{MACRO_TIMEFRAME}/{TRIGGER_TIMEFRAME}",
        "exchange": exchange,
        "as_of": next((row.get("as_of") for row in rows if row.get("as_of")), None),
        "pairs": rows,
        "error": None if not errors else errors[0],
    }

    with _market_lock:
        _market_cache["fetched_at"] = time.monotonic()
        _market_cache["payload"] = payload
    return payload


def build_snapshot() -> dict[str, Any]:
    panels = build_account_panels(peek_balance=True)
    state = _flatten_active(panels)
    market = fetch_market()
    paths = _book_paths()
    paper_trades = read_trades(path=paths["paper_trades"])
    live_trades = read_trades(path=paths["live_trades"])
    active_trades = live_trades if panels["active_mode"] == "live" else paper_trades
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "poll_seconds": 10,
        "active_mode": panels["active_mode"],
        "active_banner": panels["active_banner"],
        "active_detail": panels["active_detail"],
        "paper": panels["paper"],
        "live": panels["live"],
        "state": state,
        "market": market,
        "trades": active_trades,
        "paper_trades": paper_trades,
        "live_trades": live_trades,
        "rejects": read_reject_count(),
    }


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gemini Trader · Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui"] },
          colors: { ink: "#0b1014", panel: "#121a21" }
        }
      }
    }
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    body { font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; }
    .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; }
  </style>
</head>
<body class="bg-ink text-slate-200 min-h-screen">
  <div class="max-w-6xl mx-auto px-5 py-6">
    <header class="flex flex-wrap items-end justify-between gap-3 mb-4">
      <div>
        <p class="text-xs uppercase tracking-[0.2em] text-teal-400/80">Gemini Trend Guard</p>
        <div class="flex flex-wrap items-center gap-3 mt-1">
          <h1 id="desk-title" class="text-2xl font-semibold text-white">Paper desk</h1>
          <span id="live-badge" class="hidden inline-flex items-center gap-1.5 text-xs font-medium text-emerald-300 bg-emerald-500/10 ring-1 ring-emerald-500/30 px-2.5 py-1 rounded-md">● LIVE KRAKEN (USD)</span>
        </div>
        <p class="text-sm text-slate-400">Read-only monitor · does not place orders · does not lock the runner</p>
      </div>
      <div class="text-right text-sm text-slate-400">
        <div>Updated <span id="updated" class="mono text-slate-200">—</span></div>
        <div>Next refresh in <span id="countdown" class="mono text-teal-300">10</span>s</div>
        <div id="status" class="text-xs mt-1 text-slate-500">connecting…</div>
      </div>
    </header>

    <section id="mode-banner" class="mb-4 rounded-xl border border-amber-300/50 bg-amber-400/10 px-4 py-3">
      <p id="mode-banner-label" class="text-lg sm:text-xl font-semibold tracking-wide text-amber-100">ACTIVE: PAPER TRADING</p>
      <p id="mode-detail" class="text-sm text-amber-50/80 mt-1">Practice book is running from $10,000.00 fake money. Live Kraken is idle — no live orders are firing.</p>
    </section>

    <section id="books" class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
      <article id="paper-panel" class="rounded-xl border border-amber-300/80 ring-2 ring-amber-400/40 bg-panel p-4">
        <div class="flex items-start justify-between gap-3">
          <div>
            <p class="text-xs uppercase tracking-[0.16em] text-amber-200/90">PRACTICE / PAPER</p>
            <p class="text-sm text-slate-300">Fake money</p>
          </div>
          <span id="paper-use" class="inline-flex px-2.5 py-1 rounded-md text-xs font-semibold bg-amber-400/20 text-amber-100 ring-1 ring-amber-300/50">IN USE</span>
        </div>
        <p id="equity-label" class="text-xs uppercase tracking-wider text-slate-400 mt-4">PAPER EQUITY</p>
        <p id="paper-equity" class="text-3xl font-semibold mt-1 mono text-white">$10,000.00</p>
        <p id="paper-pnl" class="text-sm mt-2 text-slate-400">—</p>
        <p id="paper-note" class="text-xs text-slate-400 mt-2">Configured practice balance. No paper fills yet.</p>
        <p id="paper-status" class="text-sm mt-3 text-amber-100">Practice book is running.</p>
        <div id="paper-positions" class="mt-3 space-y-2"></div>
      </article>
      <article id="live-panel" class="rounded-xl border border-white/10 bg-panel p-4 opacity-60">
        <div class="flex items-start justify-between gap-3">
          <div>
            <p class="text-xs uppercase tracking-[0.16em] text-rose-200/80">LIVE / KRAKEN</p>
            <p class="text-sm text-slate-400">Real money</p>
          </div>
          <span id="live-use" class="inline-flex px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-500/20 text-slate-300 ring-1 ring-white/10">NOT IN USE</span>
        </div>
        <p id="live-equity-label" class="text-xs uppercase tracking-wider text-slate-500 mt-4">LIVE KRAKEN EQUITY</p>
        <p id="live-equity" class="text-3xl font-semibold mt-1 mono text-slate-300">Live not active</p>
        <p id="live-pnl" class="text-sm mt-2 text-slate-500">—</p>
        <p id="live-note" class="text-xs text-slate-500 mt-2">IDLE — live orders are not firing.</p>
        <p id="live-status" class="text-sm mt-3 text-slate-400">IDLE — live orders are not firing.</p>
        <div id="live-positions" class="mt-3 space-y-2"></div>
      </article>
    </section>

    <section id="pair-cards" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4"></section>

    <section class="bg-panel rounded-xl border border-white/5 p-4 mb-4 overflow-hidden">
      <div class="flex items-baseline justify-between mb-3">
        <h2 class="text-sm uppercase tracking-wider text-slate-400">Live metrics</h2>
        <p id="mkt-meta" class="text-xs text-slate-500 mono"></p>
      </div>
      <p id="mkt-error" class="hidden text-sm text-rose-300 mb-3"></p>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-left text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th class="pr-4 py-2 font-medium">Pair</th>
              <th class="px-3 py-2 font-medium">1h Regime</th>
              <th class="px-3 py-2 font-medium text-right">ADX</th>
              <th class="px-3 py-2 font-medium text-right">15m Close</th>
              <th class="px-3 py-2 font-medium text-right">EMA 9</th>
              <th class="px-3 py-2 font-medium text-right">EMA 21</th>
              <th class="px-3 py-2 font-medium text-right">15m ATR</th>
            </tr>
          </thead>
          <tbody id="metrics" class="divide-y divide-white/5"></tbody>
        </table>
      </div>
    </section>

    <section class="bg-panel rounded-xl border border-white/5 overflow-hidden">
      <div class="px-4 py-3 flex items-baseline justify-between border-b border-white/5">
        <h2 id="trade-heading" class="text-sm uppercase tracking-wider text-slate-400">Active book trade log</h2>
        <p id="trade-meta" class="text-xs text-slate-500">—</p>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-left text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th class="px-4 py-2 font-medium">Time</th>
              <th class="px-4 py-2 font-medium">Symbol</th>
              <th class="px-4 py-2 font-medium">Action</th>
              <th class="px-4 py-2 font-medium">Verdict</th>
              <th class="px-4 py-2 font-medium text-right">Entry</th>
              <th class="px-4 py-2 font-medium text-right">Exit</th>
              <th class="px-4 py-2 font-medium">Rationale</th>
            </tr>
          </thead>
          <tbody id="trades" class="divide-y divide-white/5"></tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    const POLL_MS = 10000;
    let remaining = 10;

    const fmt = (n, d=2) => n == null || Number.isNaN(n) ? "—" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
    const tonePos = (side) => ({LONG:"text-teal-300", SHORT:"text-rose-300", FLAT:"text-slate-200"}[side] || "text-slate-200");
    const badge = (tone, label) => {
      const map = {
        green: "bg-teal-500/15 text-teal-300 ring-teal-500/30",
        red: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
        amber: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
        slate: "bg-slate-500/15 text-slate-300 ring-slate-500/30",
      };
      return `<span class="inline-flex px-2 py-0.5 rounded-full text-[11px] font-medium ring-1 ${map[tone]||map.slate}">${label}</span>`;
    };

    const px = (n) => n == null || Number.isNaN(n) ? "—" : fmt(n, Number(n) >= 1000 ? 1 : Number(n) >= 100 ? 2 : 3);

    const money = (n) => n == null || Number.isNaN(Number(n)) ? "—" : "$" + fmt(n, 2);

    function positionBlock(panel, idleLive) {
      const cards = panel.positions || [];
      if (!cards.length) {
        const msg = idleLive ? "No live positions. Nothing here is a fill." : "No open positions";
        return `<p class="text-xs text-slate-500">${msg}</p>`;
      }
      return cards.map(card => {
        const side = card.status || "FLAT";
        const p = card.position;
        const detail = p
          ? `Entry ${px(p.entry_price)} · stop ${px(p.stop_loss || p.trail_stop || p.stop)} · target ${px(p.take_profit || p.target)}`
          : "Flat";
        return `<div class="flex items-baseline justify-between gap-2 text-sm border-t border-white/5 pt-2">
          <span class="text-slate-300">${card.symbol || "—"}</span>
          <span class="font-medium ${tonePos(side)}">${side}</span>
          <span class="text-xs text-slate-500">${detail}</span>
        </div>`;
      }).join("");
    }

    function paintBook(prefix, panel, kind) {
      const shell = document.getElementById(prefix + "-panel");
      const inUse = !!panel.in_use;
      shell.className = "rounded-xl border bg-panel p-4 " + (
        inUse
          ? (kind === "live"
              ? "border-rose-400/80 ring-2 ring-rose-500/50"
              : "border-amber-300/80 ring-2 ring-amber-400/40")
          : "border-white/10 opacity-60"
      );
      const useEl = document.getElementById(prefix + "-use");
      useEl.textContent = panel.use_label || (inUse ? "IN USE" : "NOT IN USE");
      useEl.className = "inline-flex px-2.5 py-1 rounded-md text-xs font-semibold ring-1 " + (
        inUse
          ? (kind === "live"
              ? "bg-rose-500/20 text-rose-100 ring-rose-400/60"
              : "bg-amber-400/20 text-amber-100 ring-amber-300/50")
          : "bg-slate-500/20 text-slate-300 ring-white/10"
      );
      const equityEl = document.getElementById(prefix === "paper" ? "paper-equity" : "live-equity");
      if (panel.placeholder && panel.equity == null) equityEl.textContent = panel.placeholder;
      else equityEl.textContent = money(panel.equity);
      const pnlEl = document.getElementById(prefix + "-pnl");
      if (panel.pnl == null) {
        pnlEl.textContent = panel.placeholder ? "" : "P&L unavailable";
        pnlEl.className = "text-sm mt-2 text-slate-500";
      } else {
        const sign = panel.pnl >= 0 ? "+" : "";
        pnlEl.textContent = `${sign}${money(panel.pnl)}  (${sign}${fmt(panel.pnl_pct, 2)}%) vs ${money(panel.starting_equity)} start`;
        pnlEl.className = "text-sm mt-2 " + (panel.pnl >= 0 ? "text-teal-300" : "text-rose-300");
      }
      document.getElementById(prefix + "-note").textContent = panel.balance_note || "";
      document.getElementById(prefix + "-status").textContent = panel.status_line || "";
      document.getElementById(prefix + "-positions").innerHTML = positionBlock(panel, kind === "live" && !inUse);
    }

    function render(data) {
      const s = data.state || {};
      const paper = data.paper || s.paper || {};
      const live = data.live || s.live || {};
      const m = data.market || {};
      const banner = data.active_banner || s.active_banner || (s.paper_trading === false ? "ACTIVE: LIVE TRADING" : "ACTIVE: PAPER TRADING");
      const liveActive = banner.indexOf("LIVE TRADING") !== -1 && banner.indexOf("PAPER") === -1;
      const bannerEl = document.getElementById("mode-banner");
      document.getElementById("mode-banner-label").textContent = banner;
      document.getElementById("mode-detail").textContent = data.active_detail || s.active_detail || "";
      bannerEl.className = "mb-4 rounded-xl border px-4 py-3 " + (
        liveActive
          ? "border-rose-400/70 bg-rose-500/15"
          : "border-amber-300/50 bg-amber-400/10"
      );
      document.getElementById("mode-banner-label").className = "text-lg sm:text-xl font-semibold tracking-wide " + (liveActive ? "text-rose-100" : "text-amber-100");
      if (paper.title) paintBook("paper", paper, "paper");
      if (live.title) paintBook("live", live, "live");
      const cards = (liveActive ? live.positions : paper.positions) || s.positions || [];
      const marketBySymbol = Object.fromEntries((m.pairs || []).map(row => [row.symbol, row]));
      const toneRegime = (reg) => ({BULL:"green", BEAR:"red", CHOP:"amber", NEUTRAL:"amber"}[reg] || "slate");
      document.getElementById("pair-cards").innerHTML = cards.map(card => {
        const side = card.status || "FLAT";
        const p = card.position;
        const live = marketBySymbol[card.symbol] || {};
        const regime = live.macro_regime || card.macro_regime || "CHOP";
        const detail = p
          ? `Entry ${px(p.entry_price)} · stop ${px(p.stop_loss || p.trail_stop || p.stop)} · target ${px(p.take_profit || p.target)}`
          : (card.last_bar ? `Last 15m bar ${card.last_bar}` : "No open position");
        return `<article class="bg-panel rounded-xl border border-white/5 p-4">
          <div class="flex items-center justify-between gap-2">
            <p class="text-xs uppercase tracking-wider text-slate-400">${card.symbol || "—"}</p>
            ${badge(toneRegime(regime), `${regime}${live.adx != null ? " ADX " + fmt(live.adx, 1) : ""}`)}
          </div>
          <p class="text-3xl font-semibold mt-1 ${tonePos(side)}">${side}</p>
          <p class="text-xs text-slate-400 mt-2 leading-relaxed">${detail}</p>
        </article>`;
      }).join("") || `<article class="bg-panel rounded-xl border border-white/5 p-4 md:col-span-3"><p class="text-sm text-slate-500">No position book</p></article>`;

      document.getElementById("desk-title").textContent = liveActive ? "Live Execution Desk" : "Paper desk";
      const badgeEl = document.getElementById("live-badge");
      if (liveActive) {
        badgeEl.textContent = "● LIVE KRAKEN (USD)";
        badgeEl.classList.remove("hidden");
      } else {
        badgeEl.classList.add("hidden");
      }
      document.getElementById("trade-heading").textContent = liveActive ? "Live trade log · real fills only" : "Paper trade log · fake money";

      const pairRows = m.pairs || [];
      document.getElementById("mkt-meta").textContent = [m.exchange, m.timeframe, m.as_of || ""].filter(Boolean).join(" · ");
      const err = document.getElementById("mkt-error");
      if (m.error) { err.textContent = m.error; err.classList.remove("hidden"); }
      else { err.classList.add("hidden"); }
      const metrics = document.getElementById("metrics");
      if (!pairRows.length) {
        metrics.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-slate-500">Waiting for candles</td></tr>`;
      } else {
        metrics.innerHTML = pairRows.map(row => {
          const errNote = row.error ? `<div class="text-[11px] text-rose-300 mt-1">${row.error}</div>` : "";
          const regime = row.macro_regime || "CHOP";
          const regimeTone = {BULL:"green", BEAR:"red", CHOP:"amber", NEUTRAL:"amber"}[regime] || "slate";
          return `<tr class="hover:bg-white/[0.02]">
            <td class="pr-4 py-2.5 font-medium text-slate-200 whitespace-nowrap">${row.symbol || "—"}${errNote}</td>
            <td class="px-3 py-2.5">${badge(regimeTone, regime)}</td>
            <td class="px-3 py-2.5 mono text-right">${fmt(row.adx, 1)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.last_close)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.ema_fast)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.ema_slow)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.atr)}</td>
          </tr>`;
        }).join("");
      }

      const body = document.getElementById("trades");
      const rows = (data.trades && data.trades.trades) || [];
      document.getElementById("trade-meta").textContent =
        `${data.trades && data.trades.count || 0} fills · ${data.rejects || 0} rejects · showing ${rows.length}`;
      if (!rows.length) {
        body.innerHTML = `<tr><td colspan="7" class="px-4 py-8 text-center text-slate-500">${s.trade_log_empty || "No trades yet"}</td></tr>`;
      } else {
        body.innerHTML = rows.map(t => {
          const verdictTone = t.verdict === "REJECTED" || t.hit === "stop" ? "red" : (t.tone || "slate");
          const actionTone = t.action === "BUY" ? "green" : t.action === "SELL" ? "red" : (t.hit === "target" ? "green" : t.hit === "stop" ? "red" : "slate");
          const rationale = (t.rationale || t.alert_reason || "—").slice(0, 160);
          return `<tr class="hover:bg-white/[0.02]">
            <td class="px-4 py-2.5 mono text-xs text-slate-400 whitespace-nowrap">${(t.timestamp||"").replace("T"," ").replace("+00:00"," UTC")}</td>
            <td class="px-4 py-2.5 mono text-xs text-slate-200 whitespace-nowrap">${t.symbol || "—"}</td>
            <td class="px-4 py-2.5">${badge(actionTone, t.action || "—")}</td>
            <td class="px-4 py-2.5">${badge(verdictTone, t.verdict || "—")}${t.hit ? " " + badge(t.hit === "stop" ? "red" : "green", t.hit === "stop" ? "STOP_LOSS" : "PROFIT") : ""}</td>
            <td class="px-4 py-2.5 mono text-right">${fmt(t.entry_price, 1)}</td>
            <td class="px-4 py-2.5 mono text-right">${fmt(t.exit_price, 1)}</td>
            <td class="px-4 py-2.5 text-slate-400 max-w-xl">${rationale}</td>
          </tr>`;
        }).join("");
      }

      document.getElementById("updated").textContent = (data.generated_at || "").replace("T", " ").replace("+00:00", " UTC");
    }

    async function tick() {
      try {
        const token = new URLSearchParams(location.search).get("token") || "";
        const qs = token ? ("?token=" + encodeURIComponent(token)) : "";
        const res = await fetch("/api/snapshot" + qs, { cache: "no-store" });
        if (res.status === 401) throw new Error("add ?token= from DASHBOARD_SECRET or WEBHOOK_SECRET");
        if (!res.ok) throw new Error("HTTP " + res.status);
        render(await res.json());
        document.getElementById("status").textContent = "connected";
        document.getElementById("status").className = "text-xs mt-1 text-teal-400";
      } catch (e) {
        document.getElementById("status").textContent = "poll failed: " + e.message;
        document.getElementById("status").className = "text-xs mt-1 text-rose-300";
      }
      remaining = 10;
    }

    setInterval(() => {
      remaining = Math.max(0, remaining - 1);
      document.getElementById("countdown").textContent = String(remaining);
    }, 1000);
    setInterval(tick, POLL_MS);
    tick();
  </script>
</body>
</html>
"""

app = FastAPI(title="Gemini Trader dashboard", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, token: str | None = Query(default=None)) -> HTMLResponse:
    check_token(request, token)
    return HTMLResponse(PAGE)


@app.get("/api/snapshot")
def snapshot(request: Request, token: str | None = Query(default=None)) -> dict[str, Any]:
    check_token(request, token)
    return build_snapshot()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Read-only paper/live dashboard (localhost by default)")
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address (default 127.0.0.1, or DASHBOARD_HOST). 0.0.0.0 requires a secret.",
    )
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    args = parser.parse_args(argv)
    _cfg()  # load .env so DASHBOARD_HOST / secrets are visible
    host = args.host or os.getenv("DASHBOARD_HOST") or DASHBOARD_HOST

    import uvicorn

    enable_os_trust_store()
    secret = dashboard_secret()
    try:
        assert_secret_for_public_bind(
            host=host,
            secret=secret,
            service="the dashboard",
            secret_name="DASHBOARD_SECRET",
        )
    except RuntimeError as exc:
        print(f"Refusing to start: {exc}", file=sys.stderr)
        return 1
    if secret:
        auth = "token required off localhost (?token=)"
    else:
        auth = "open on localhost only (set DASHBOARD_SECRET before a public bind or tunnel)"
    print(f"Dashboard on http://{host}:{args.port}  (read-only, {auth})")
    if secret:
        print(f"  local bookmark : http://127.0.0.1:{args.port}/?token=<DASHBOARD_SECRET or WEBHOOK_SECRET>")
        print(f"  phone          : cloudflared tunnel --url http://localhost:{args.port}")
        print("                  then open https://<printed-host>/?token=<same secret>")
    uvicorn.run(app, host=host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
