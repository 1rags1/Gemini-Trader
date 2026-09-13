"""Market data access: OHLCV retrieval via ccxt, indicators via pandas_ta.

Read-only public endpoints only — no exchange credentials are ever used, which
keeps the paper-trading loop incapable of placing a real order.
"""

from __future__ import annotations

import pandas as pd

from core.config import (
    MACRO_ADX_PERIOD,
    MACRO_ADX_THRESHOLD,
    MACRO_EMA_FAST,
    MACRO_EMA_SLOW,
    MACRO_EMA_TREND,
    TRIGGER_ATR_PERIOD,
    TRIGGER_EMA_FAST,
    TRIGGER_EMA_SLOW,
    get_settings,
)
from core.net import enable_os_trust_store

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

#: Indicator lengths, kept in step with strategies/params/ema_atr_trend.json so
#: the Gemini agent reasons over the same regime the Pine strategy trades.
EMA_FAST = 21
EMA_SLOW = 55
EMA_MACRO = 200
RSI_LENGTH = 14
ATR_LENGTH = 14
ADX_LENGTH = 14

#: ADX below this is treated as "no tradeable trend", matching adx_min in Pine.
ADX_MIN = 20.0


class MarketDataError(RuntimeError):
    """Raised when candles cannot be retrieved or are unusable."""


def _build_exchange(exchange_id: str):
    enable_os_trust_store()  # ccxt uses requests, which also needs the OS CA store

    import ccxt

    if not hasattr(ccxt, exchange_id):
        raise MarketDataError(f"Unknown ccxt exchange id: {exchange_id!r}")
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def fetch_ohlcv(
    symbol: str | None = None,
    timeframe: str | None = None,
    limit: int | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame, oldest candle first."""
    cfg = get_settings()
    symbol = symbol or cfg.symbol
    timeframe = timeframe or cfg.timeframe
    limit = limit or cfg.candle_limit

    exchange = _build_exchange(exchange_id or cfg.exchange_id)
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not raw:
        raise MarketDataError(f"No candles returned for {symbol} @ {timeframe}")

    df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").astype(float).sort_index()
    df.attrs.update(symbol=symbol, timeframe=timeframe, exchange=exchange.id)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the indicator set the Gemini agent reasons over.

    Mirrors the Pine strategy: fast/slow EMAs for the signal, EMA 200 for the
    macro trend gate, ADX for trend strength, ATR for risk, RSI/MACD for
    momentum context.
    """
    import pandas_ta as ta  # noqa: F401  (registers the .ta DataFrame accessor)

    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], length=EMA_FAST)
    out["ema_slow"] = ta.ema(out["close"], length=EMA_SLOW)
    out["ema_macro"] = ta.ema(out["close"], length=EMA_MACRO)
    out["rsi"] = ta.rsi(out["close"], length=RSI_LENGTH)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=ATR_LENGTH)

    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        out = out.join(macd)

    # ta.adx returns ADX plus ADXR and the directional pair; only ADX and the
    # DI legs carry signal for the regime gate.
    adx = ta.adx(out["high"], out["low"], out["close"], length=ADX_LENGTH)
    if adx is not None:
        out["adx"] = adx[f"ADX_{ADX_LENGTH}"]
        out["di_plus"] = adx[f"DMP_{ADX_LENGTH}"]
        out["di_minus"] = adx[f"DMN_{ADX_LENGTH}"]

    out.attrs.update(df.attrs)
    return out


def add_macro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """1h trend filter: EMA 21/55/200 and ADX 14."""
    import pandas_ta as ta  # noqa: F401

    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], length=MACRO_EMA_FAST)
    out["ema_slow"] = ta.ema(out["close"], length=MACRO_EMA_SLOW)
    out["ema_macro"] = ta.ema(out["close"], length=MACRO_EMA_TREND)
    adx = ta.adx(out["high"], out["low"], out["close"], length=MACRO_ADX_PERIOD)
    if adx is not None:
        out["adx"] = adx[f"ADX_{MACRO_ADX_PERIOD}"]
        out["di_plus"] = adx[f"DMP_{MACRO_ADX_PERIOD}"]
        out["di_minus"] = adx[f"DMN_{MACRO_ADX_PERIOD}"]
    out.attrs.update(df.attrs)
    return out


def add_trigger_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """15m pullback trigger: EMA 9/21 and ATR 14."""
    import pandas_ta as ta  # noqa: F401

    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], length=TRIGGER_EMA_FAST)
    out["ema_slow"] = ta.ema(out["close"], length=TRIGGER_EMA_SLOW)
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=TRIGGER_ATR_PERIOD)
    out.attrs.update(df.attrs)
    return out


def classify_macro_regime(bar: pd.Series, adx_threshold: float = MACRO_ADX_THRESHOLD) -> str:
    """BULL only when 1h close > EMA 200, EMA 21 > EMA 55, and ADX is strong.

    Otherwise NEUTRAL (weak ADX / incomplete indicators) or BEAR.
    """
    needed = ("close", "ema_fast", "ema_slow", "ema_macro", "adx")
    if any(name not in bar.index or pd.isna(bar[name]) for name in needed):
        return "NEUTRAL"
    close = float(bar["close"])
    ema21 = float(bar["ema_fast"])
    ema55 = float(bar["ema_slow"])
    ema200 = float(bar["ema_macro"])
    adx = float(bar["adx"])
    if close > ema200 and ema21 > ema55 and adx >= adx_threshold:
        return "BULL"
    if adx < adx_threshold:
        return "NEUTRAL"
    return "BEAR"


def _regime(last: pd.Series, adx_min: float) -> dict:
    """Summarise the same gate the Pine strategy applies before entering.

    EMA 200 sets the permitted direction and ADX decides whether any trend is
    worth trading. `tradeable_direction` is what the agent should obey.
    """
    close = float(last["close"])
    macro = last.get("ema_macro")
    adx = last.get("adx")

    above_macro = None if pd.isna(macro) else close > float(macro)
    trend_strong = None if pd.isna(adx) else float(adx) > adx_min

    if above_macro is None or trend_strong is None:
        direction = "unknown"
    elif not trend_strong:
        direction = "none"
    else:
        direction = "long_only" if above_macro else "short_only"

    return {
        "macro_ema": None if pd.isna(macro) else round(float(macro), 4),
        "macro_trend": "unknown" if above_macro is None else ("bull" if above_macro else "bear"),
        "price_vs_macro_ema_pct": (
            None if above_macro is None else round((close / float(macro) - 1) * 100, 3)
        ),
        "adx": None if pd.isna(adx) else round(float(adx), 3),
        "adx_min": adx_min,
        "trend_strength": (
            "unknown" if trend_strong is None else ("strong" if trend_strong else "weak")
        ),
        "tradeable_direction": direction,
    }


def latest_snapshot(
    df: pd.DataFrame, lookback: int = 5, adx_min: float = ADX_MIN
) -> dict:
    """Compact, token-efficient summary of recent state for the LLM prompt."""
    if df.empty:
        raise MarketDataError("Cannot summarise an empty frame")

    tail = df.tail(lookback).round(4)
    last = tail.iloc[-1]
    return {
        "symbol": df.attrs.get("symbol"),
        "timeframe": df.attrs.get("timeframe"),
        "as_of": tail.index[-1].isoformat(),
        "last_close": float(last["close"]),
        "regime": _regime(last, adx_min),
        "indicators": {
            col: (None if pd.isna(last[col]) else float(last[col]))
            for col in df.columns
            if col not in OHLCV_COLUMNS
        },
        "recent_candles": [
            {"t": ts.isoformat(), **{c: float(row[c]) for c in ["open", "high", "low", "close", "volume"]}}
            for ts, row in tail.iterrows()
        ],
    }
