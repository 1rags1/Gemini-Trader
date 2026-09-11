"""Centralised configuration loaded from the root `.env` file.

Every module reads settings through `get_settings()` so the API key is resolved
in exactly one place and never hard-coded or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

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


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str = DEFAULT_MODEL
    gemini_timeout_ms: int = DEFAULT_TIMEOUT_MS
    gemini_fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    exchange_id: str = DEFAULT_EXCHANGE
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
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


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ENV_PATH, override=False)

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            f"GEMINI_API_KEY is not set. Add it to {ENV_PATH} as:\n"
            "    GEMINI_API_KEY=your_key_here"
        )

    return Settings(
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        gemini_timeout_ms=int(os.getenv("GEMINI_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS))),
        gemini_fallback_models=_csv_env("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS),
        exchange_id=os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE),
        symbol=os.getenv("SYMBOL", "BTC/USDT"),
        timeframe=os.getenv("TIMEFRAME", "1h"),
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
