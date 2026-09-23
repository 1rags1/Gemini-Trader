"""Centralised configuration loaded from the root `.env` file.

Every module reads settings through `get_settings()` so the API key is resolved
in exactly one place and never hard-coded or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


#: The gemini-2.5-flash line is closed to new API keys; this is the replacement
#: Google's own 404 response points at. Override with GEMINI_MODEL in .env.
DEFAULT_MODEL = "gemini-3.6-flash"

#: Tried in order when the primary model is overloaded. The 3.6-flash endpoint
#: returns 503/504 under load often enough to stall a loop on its own.
DEFAULT_FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-flash-latest")

#: Binance answers 451 from several regions, so the default is an exchange that
#: serves public market data more broadly. coinbase, kucoin and okx also work.
DEFAULT_EXCHANGE = "kraken"

#: Per-request HTTP timeout for the Gemini API, in milliseconds (the unit the
#: SDK expects). Bounds a stalled request so a trading loop cannot hang.
DEFAULT_TIMEOUT_MS = 20_000

#: Universe the runner scans each cycle. Override with TRADING_PAIRS in .env
#: as a comma-separated list.
TRADING_PAIRS = ("BTC/USD", "ETH/USD", "SOL/USD")

#: 1h candles gate the tradeable direction. 15m candles fire the pullback.
MACRO_TIMEFRAME = "1h"
TRIGGER_TIMEFRAME = "15m"

#: Poll often enough to see a new 15m close within one bar.
POLL_INTERVAL_SECONDS = 30

#: Macro (1h) trend filter.
MACRO_EMA_FAST = 21
MACRO_EMA_SLOW = 55
MACRO_EMA_TREND = 200
MACRO_ADX_PERIOD = 14
MACRO_ADX_THRESHOLD = 20.0

#: Trigger (15m) pullback / ATR risk.
TRIGGER_EMA_FAST = 9
TRIGGER_EMA_SLOW = 21
TRIGGER_ATR_PERIOD = 14

#: Hard cap on concurrent paper positions across the whole book.
MAX_OPEN_POSITIONS = 2

#: Fraction of current equity allocated to each new fill. With
#: MAX_OPEN_POSITIONS = 2, 0.25 uses about half of equity and leaves the
#: rest as a cash buffer for fees and slippage.
POSITION_SIZE_FRACTION = 0.25

#: 15m ATR multiples. 1.5 / 3.5 is about 1 : 2.33 risk-to-reward.
ATR_STOP_MULTIPLIER = 1.5
ATR_PROFIT_MULTIPLIER = 3.5

#: Spot accounts cannot sell short. The runner skips SELL entries when True.
SPOT_LONG_ONLY = True

#: Practice mode is the default. Copying `.env.example` stays on the paper book.
PAPER_TRADING = True

#: Second gate for real orders. Ignored while paper mode is on. Live orders
#: are sent only when PAPER_TRADING is false AND this flag is true. Paper off
#: without this flag fails closed and places nothing.
ALLOW_LIVE_TRADING = False

#: Live entries are post-only limits at the 15m close (maker), not market takes.
USE_POST_ONLY = True


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str = DEFAULT_MODEL
    gemini_timeout_ms: int = DEFAULT_TIMEOUT_MS
    gemini_fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    exchange_id: str = DEFAULT_EXCHANGE
    symbol: str = "BTC/USD"
    trading_pairs: tuple[str, ...] = TRADING_PAIRS
    macro_timeframe: str = MACRO_TIMEFRAME
    trigger_timeframe: str = TRIGGER_TIMEFRAME
    poll_interval_seconds: int = POLL_INTERVAL_SECONDS
    macro_ema_fast: int = MACRO_EMA_FAST
    macro_ema_slow: int = MACRO_EMA_SLOW
    macro_ema_trend: int = MACRO_EMA_TREND
    macro_adx_period: int = MACRO_ADX_PERIOD
    macro_adx_threshold: float = MACRO_ADX_THRESHOLD
    trigger_ema_fast: int = TRIGGER_EMA_FAST
    trigger_ema_slow: int = TRIGGER_EMA_SLOW
    trigger_atr_period: int = TRIGGER_ATR_PERIOD
    max_open_positions: int = MAX_OPEN_POSITIONS
    position_size_fraction: float = POSITION_SIZE_FRACTION
    atr_stop_multiplier: float = ATR_STOP_MULTIPLIER
    atr_profit_multiplier: float = ATR_PROFIT_MULTIPLIER
    spot_long_only: bool = SPOT_LONG_ONLY
    paper_trading: bool = PAPER_TRADING
    allow_live_trading: bool = ALLOW_LIVE_TRADING
    use_post_only: bool = USE_POST_ONLY
    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    timeframe: str = TRIGGER_TIMEFRAME
    candle_limit: int = 500
    paper_starting_balance: float = 10_000.0

    webhook_host: str = "127.0.0.1"
    webhook_port: int = 5000
    #: Shared secret for the webhook endpoint. TradingView can only vary the URL
    #: and body, so it travels as a `?token=` query parameter. Empty disables
    #: the check, which is only safe on a host that is not publicly reachable.
    webhook_secret: str = ""
    #: Token for the read-only dashboard when it is exposed through a tunnel.
    #: Falls back to WEBHOOK_SECRET when DASHBOARD_SECRET is unset.
    dashboard_secret: str = ""
    #: Minimum Gemini confidence required to confirm a TradingView entry.
    min_confidence: float = 0.6

    paths: dict[str, Path] = field(default_factory=dict)

    def masked_key(self) -> str:
        """A safe-to-print fingerprint of the key for diagnostics."""
        if len(self.gemini_api_key) <= 8:
            return "*" * len(self.gemini_api_key)
        return f"{self.gemini_api_key[:4]}...{self.gemini_api_key[-4:]}"


def empty_circuit_breaker() -> dict[str, Any]:
    return {"loss_streak": 0, "tripped": False, "cooldown_bars": 0}


def empty_position_slot(macro_regime: str = "UNKNOWN") -> dict[str, Any]:
    """Canonical FLAT slot for the multi-timeframe book."""
    return {
        "status": "FLAT",
        "entry_price": 0.0,
        "size": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "entry_time": None,
        "macro_regime": str(macro_regime or "UNKNOWN").upper(),
    }


def empty_position_book(pairs: tuple[str, ...] | None = None) -> dict[str, dict]:
    """FLAT slot for every configured pair."""
    return {symbol: empty_position_slot() for symbol in (pairs or TRADING_PAIRS)}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def canonical_position_slot(slot: dict[str, Any] | None) -> dict[str, Any]:
    """Project any slot onto the MTF book fields (drops runner-only extras)."""
    base = empty_position_slot()
    if not isinstance(slot, dict):
        return base
    status = str(slot.get("status") or slot.get("side") or "FLAT").upper() or "FLAT"
    size = slot.get("size", slot.get("qty", 0.0))
    stop = slot.get("stop_loss", slot.get("trail_stop", slot.get("stop", 0.0)))
    target = slot.get("take_profit", slot.get("target", 0.0))
    entry_time = slot.get("entry_time", slot.get("opened_bar"))
    if status in {"", "FLAT"}:
        return empty_position_slot(str(slot.get("macro_regime") or "UNKNOWN"))
    return {
        "status": status,
        "entry_price": _as_float(slot.get("entry_price")),
        "size": _as_float(size),
        "stop_loss": _as_float(stop),
        "take_profit": _as_float(target),
        "entry_time": entry_time or None,
        "macro_regime": str(slot.get("macro_regime") or ("BULL" if status == "LONG" else "BEAR")).upper(),
    }


def hydrate_position_slot(slot: dict[str, Any] | None) -> dict[str, Any]:
    """Canonical MTF fields plus aliases the current runner still reads."""
    canonical = canonical_position_slot(slot)
    if not isinstance(slot, dict):
        extras: dict[str, Any] = {}
    else:
        extras = {
            key: value
            for key, value in slot.items()
            if key not in canonical
        }
    hydrated = {**extras, **canonical}
    hydrated.setdefault("side", canonical["status"])
    hydrated.setdefault("qty", canonical["size"])
    hydrated.setdefault("stop", canonical["stop_loss"])
    hydrated.setdefault("target", canonical["take_profit"])
    hydrated.setdefault("trail_stop", canonical["stop_loss"])
    hydrated.setdefault("opened_bar", canonical["entry_time"])
    return hydrated


def migrate_circuit_breaker(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Accept nested `circuit_breaker` or the older top-level breaker keys."""
    raw = raw or {}
    nested = raw.get("circuit_breaker")
    if isinstance(nested, dict):
        streak = nested.get("loss_streak", raw.get("loss_streak", 0))
        if "tripped" in nested:
            tripped = nested.get("tripped")
        else:
            tripped = nested.get("active", raw.get("breaker_active", False))
        cooldown = nested.get(
            "cooldown_bars",
            nested.get("breaker_bars", raw.get("breaker_bars", 0)),
        )
        return {
            "loss_streak": int(streak or 0),
            "tripped": bool(tripped),
            "cooldown_bars": int(cooldown or 0),
        }
    return {
        "loss_streak": int(raw.get("loss_streak") or 0),
        "tripped": bool(raw.get("breaker_active") or raw.get("tripped") or False),
        "cooldown_bars": int(raw.get("breaker_bars") or raw.get("cooldown_bars") or 0),
    }


def migrate_position_book(raw: dict, pairs: tuple[str, ...] | None = None) -> dict[str, dict]:
    """Build a per-symbol MTF book from either the new or the legacy state file."""
    pairs = pairs or TRADING_PAIRS
    book = empty_position_book(pairs)
    incoming = raw.get("positions")
    if isinstance(incoming, dict):
        for symbol, slot in incoming.items():
            if isinstance(slot, dict):
                book[symbol] = hydrate_position_slot(slot)

    legacy = raw.get("position")
    if isinstance(legacy, dict):
        status = str(legacy.get("status") or legacy.get("side") or "FLAT").upper()
        if status not in {"", "FLAT"}:
            symbol = legacy.get("symbol") or (pairs[0] if pairs else "BTC/USD")
            book[symbol] = hydrate_position_slot({**legacy, "status": status, "symbol": symbol})
    return book


def dump_runner_state(
    *,
    equity: float,
    positions: dict[str, dict],
    circuit_breaker: dict[str, Any] | None = None,
    equity_history: list[float] | None = None,
    last_bar: str | None = None,
    last_bars: dict[str, str] | None = None,
    breaker_bars: int = 0,
    start_equity: float | None = None,
    pairs: tuple[str, ...] | None = None,
    pending_orders: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Serialize runner state in the MTF schema.

    `last_bars` / `equity_history` stay on disk so the current cycle can still
    skip a bar it already processed. They are not part of the public book.
    """
    pairs = pairs or TRADING_PAIRS
    book = empty_position_book(pairs)
    for symbol, slot in (positions or {}).items():
        book[symbol] = canonical_position_slot(slot)
    breaker = migrate_circuit_breaker(circuit_breaker)
    breaker["cooldown_bars"] = int(breaker_bars)
    payload: dict[str, Any] = {
        "equity": float(equity),
        "circuit_breaker": breaker,
        "positions": book,
    }
    if start_equity is not None:
        payload["start_equity"] = float(start_equity)
    if equity_history is not None:
        payload["equity_history"] = list(equity_history)
    if last_bar is not None:
        payload["last_bar"] = last_bar
    if last_bars is not None:
        payload["last_bars"] = dict(last_bars)
    if pending_orders is not None:
        payload["pending_orders"] = {
            symbol: dict(record)
            for symbol, record in pending_orders.items()
            if isinstance(record, dict)
        }
    return payload


def legacy_open_slot(positions: dict[str, dict]) -> dict | None:
    """First non-FLAT slot, for code that still reads a single `position`."""
    for slot in positions.values():
        status = str(slot.get("status") or slot.get("side") or "FLAT").upper()
        if status not in {"", "FLAT"}:
            return slot
    return None


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ENV_PATH, override=False)

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            f"GEMINI_API_KEY is not set. Add it to {ENV_PATH} as:\n"
            "    GEMINI_API_KEY=your_key_here"
        )

    trigger = os.getenv("TRIGGER_TIMEFRAME", TRIGGER_TIMEFRAME)
    return Settings(
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        gemini_timeout_ms=int(os.getenv("GEMINI_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS))),
        gemini_fallback_models=_csv_env("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS),
        exchange_id=os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE),
        symbol=os.getenv("SYMBOL", "BTC/USD"),
        trading_pairs=_csv_env("TRADING_PAIRS", TRADING_PAIRS),
        macro_timeframe=os.getenv("MACRO_TIMEFRAME", MACRO_TIMEFRAME),
        trigger_timeframe=trigger,
        poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", str(POLL_INTERVAL_SECONDS))),
        macro_ema_fast=int(os.getenv("MACRO_EMA_FAST", str(MACRO_EMA_FAST))),
        macro_ema_slow=int(os.getenv("MACRO_EMA_SLOW", str(MACRO_EMA_SLOW))),
        macro_ema_trend=int(os.getenv("MACRO_EMA_TREND", str(MACRO_EMA_TREND))),
        macro_adx_period=int(os.getenv("MACRO_ADX_PERIOD", str(MACRO_ADX_PERIOD))),
        macro_adx_threshold=float(os.getenv("MACRO_ADX_THRESHOLD", str(MACRO_ADX_THRESHOLD))),
        trigger_ema_fast=int(os.getenv("TRIGGER_EMA_FAST", str(TRIGGER_EMA_FAST))),
        trigger_ema_slow=int(os.getenv("TRIGGER_EMA_SLOW", str(TRIGGER_EMA_SLOW))),
        trigger_atr_period=int(os.getenv("TRIGGER_ATR_PERIOD", str(TRIGGER_ATR_PERIOD))),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", str(MAX_OPEN_POSITIONS))),
        position_size_fraction=float(
            os.getenv("POSITION_SIZE_FRACTION", str(POSITION_SIZE_FRACTION))
        ),
        atr_stop_multiplier=float(os.getenv("ATR_STOP_MULTIPLIER", str(ATR_STOP_MULTIPLIER))),
        atr_profit_multiplier=float(
            os.getenv("ATR_PROFIT_MULTIPLIER", str(ATR_PROFIT_MULTIPLIER))
        ),
        spot_long_only=_env_bool("SPOT_LONG_ONLY", SPOT_LONG_ONLY),
        paper_trading=_env_bool("PAPER_TRADING", PAPER_TRADING),
        allow_live_trading=_env_bool("ALLOW_LIVE_TRADING", ALLOW_LIVE_TRADING),
        use_post_only=_env_bool("USE_POST_ONLY", USE_POST_ONLY),
        exchange_api_key=(os.getenv("EXCHANGE_API_KEY") or "").strip(),
        exchange_api_secret=(os.getenv("EXCHANGE_API_SECRET") or "").strip(),
        timeframe=os.getenv("TIMEFRAME", trigger),
        candle_limit=int(os.getenv("CANDLE_LIMIT", "500")),
        paper_starting_balance=float(os.getenv("PAPER_STARTING_BALANCE", "10000")),
        webhook_host=os.getenv("WEBHOOK_HOST", "127.0.0.1"),
        webhook_port=int(os.getenv("WEBHOOK_PORT", "5000")),
        webhook_secret=(os.getenv("WEBHOOK_SECRET") or "").strip(),
        dashboard_secret=(
            os.getenv("DASHBOARD_SECRET") or os.getenv("WEBHOOK_SECRET") or ""
        ).strip(),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.6")),
        paths={
            "root": PROJECT_ROOT,
            "data": Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))),
            "strategies": PROJECT_ROOT / "strategies",
            "templates": PROJECT_ROOT / "strategies" / "templates",
            "params": PROJECT_ROOT / "strategies" / "params",
            "rendered": PROJECT_ROOT / "strategies" / "rendered",
        },
    )
