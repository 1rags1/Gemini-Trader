"""Kraken/ccxt helpers for pair limits, quote balance, and live orders.

Public market metadata does not need API keys. Private balance and order
calls are only used when PAPER_TRADING is False.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from core.net import enable_os_trust_store

log = logging.getLogger("broker")

#: Prefer USD funding. Include Kraken's legacy ZUSD code, then USDT.
QUOTE_CANDIDATES = ("USD", "ZUSD", "USDT")


@dataclass(frozen=True)
class PairLimits:
    symbol: str
    ordermin: float
    lot_decimals: int
    pair_decimals: int


def default_pair_limits(symbol: str) -> PairLimits:
    """Paper/offline fallback: no minimum, 8 lot decimals."""
    return PairLimits(symbol=symbol, ordermin=0.0, lot_decimals=8, pair_decimals=5)


def truncate_decimal(value: float, decimals: int) -> float:
    """Floor `value` to `decimals` places so Kraken cannot reject roundoff."""
    if value <= 0 or decimals < 0:
        return 0.0
    quant = Decimal("1").scaleb(-int(decimals))
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_DOWN))


def truncate_qty(qty: float, lot_decimals: int) -> float:
    return truncate_decimal(qty, lot_decimals)


def truncate_price(price: float, pair_decimals: int) -> float:
    return truncate_decimal(price, pair_decimals)


def prepare_order_size(qty: float, limits: PairLimits) -> float | None:
    """Truncate to lot decimals and refuse anything below `ordermin`."""
    sized = truncate_qty(qty, limits.lot_decimals)
    if sized <= 0 or sized + 1e-15 < float(limits.ordermin):
        log.warning(
            "skipping %s: calculated size %s < ordermin %s (would raise EOrder:Order minimum not met)",
            limits.symbol,
            sized,
            limits.ordermin,
        )
        return None
    return sized


def _as_decimals(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number >= 1 or number == 0:
        return max(0, int(number))
    if number > 0:
        return max(0, -int(math.floor(math.log10(number))))
    return default


def limits_from_market(symbol: str, market: dict[str, Any]) -> PairLimits:
    info = market.get("info") if isinstance(market.get("info"), dict) else {}
    amount_limits = (market.get("limits") or {}).get("amount") or {}
    precision = market.get("precision") or {}
    ordermin = info.get("ordermin", amount_limits.get("min") or 0.0)
    try:
        ordermin_f = float(ordermin or 0.0)
    except (TypeError, ValueError):
        ordermin_f = 0.0
    lot_decimals = _as_decimals(info.get("lot_decimals", precision.get("amount")), 8)
    pair_decimals = _as_decimals(info.get("pair_decimals", precision.get("price")), 5)
    return PairLimits(
        symbol=symbol,
        ordermin=max(0.0, ordermin_f),
        lot_decimals=lot_decimals,
        pair_decimals=pair_decimals,
    )


def load_pair_limits(
    symbols: tuple[str, ...] | list[str],
    exchange_id: str,
    *,
    markets: dict[str, Any] | None = None,
) -> dict[str, PairLimits]:
    """Read `ordermin` / `lot_decimals` from Kraken AssetPairs (via ccxt markets)."""
    if markets is None:
        markets = _public_exchange(exchange_id).load_markets()
    book: dict[str, PairLimits] = {}
    for symbol in symbols:
        market = markets.get(symbol) if isinstance(markets, dict) else None
        if isinstance(market, dict):
            book[symbol] = limits_from_market(symbol, market)
        else:
            log.warning("no AssetPairs row for %s; using paper defaults", symbol)
            book[symbol] = default_pair_limits(symbol)
    return book


def free_quote_balance(balance: dict[str, Any]) -> float:
    """Prefer free USD/ZUSD, then USDT, from a ccxt `fetch_balance()` payload.

    ccxt usually normalizes Kraken `ZUSD` → `USD`. We still scan the raw
    `info.result` map so a legacy key cannot be missed on live startup.
    """
    free = balance.get("free") if isinstance(balance.get("free"), dict) else {}
    total = balance.get("total") if isinstance(balance.get("total"), dict) else {}
    info = balance.get("info") if isinstance(balance.get("info"), dict) else {}
    raw_result = info.get("result") if isinstance(info.get("result"), dict) else {}

    for currency in QUOTE_CANDIDATES:
        for mapping in (free, total, balance, raw_result):
            if not isinstance(mapping, dict):
                continue
            raw = mapping.get(currency)
            if isinstance(raw, dict):
                raw = raw.get("free", raw.get("total"))
            try:
                amount = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if amount > 0:
                return amount
    return 0.0


def fetch_quote_balance(exchange_id: str, api_key: str, secret: str) -> float:
    exchange = _private_exchange(exchange_id, api_key, secret)
    return free_quote_balance(exchange.fetch_balance())


def place_limit_entry(
    exchange_id: str,
    api_key: str,
    secret: str,
    symbol: str,
    amount: float,
    price: float,
    *,
    post_only: bool = True,
) -> dict[str, Any]:
    exchange = _private_exchange(exchange_id, api_key, secret)
    params: dict[str, Any] = {}
    if post_only:
        params["postOnly"] = True
    return exchange.create_order(symbol, "limit", "buy", amount, price, params)


def fetch_order(
    exchange_id: str,
    api_key: str,
    secret: str,
    order_id: str,
    symbol: str,
) -> dict[str, Any]:
    """Fetch one live order so a resting limit can be reconciled."""
    exchange = _private_exchange(exchange_id, api_key, secret)
    return exchange.fetch_order(order_id, symbol)


def place_market_exit(
    exchange_id: str,
    api_key: str,
    secret: str,
    symbol: str,
    amount: float,
) -> dict[str, Any]:
    exchange = _private_exchange(exchange_id, api_key, secret)
    return exchange.create_order(symbol, "market", "sell", amount)


def _public_exchange(exchange_id: str):
    enable_os_trust_store()
    import ccxt

    if not hasattr(ccxt, exchange_id):
        raise RuntimeError(f"Unknown ccxt exchange id: {exchange_id!r}")
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def _private_exchange(exchange_id: str, api_key: str, secret: str):
    if not api_key or not secret:
        raise RuntimeError("EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required for live trading")
    enable_os_trust_store()
    import ccxt

    if not hasattr(ccxt, exchange_id):
        raise RuntimeError(f"Unknown ccxt exchange id: {exchange_id!r}")
    return getattr(ccxt, exchange_id)(
        {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
        }
    )
