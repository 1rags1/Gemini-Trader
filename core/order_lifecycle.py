"""Live-order gate and limit-entry fill confirmation.

An accepted post-only limit is not a position. The runner keeps the order id
in `pending_orders` until the exchange reports a fill, a cancel, or an expiry.
Market exits stay in the runner: those acknowledgements are already treated as
fills.

Hard risk limits (size, circuit breaker, correlation cap, spot long-only,
post-only entries) stay in the runner. Nothing in this module lets a model
response skip them.
"""

from __future__ import annotations

from typing import Any, Literal

OrderOutcome = Literal["filled", "pending", "canceled"]

_CANCELED = frozenset({"canceled", "cancelled", "expired", "rejected"})
_FILLED = frozenset({"closed", "filled"})


class LiveTradingDisabled(RuntimeError):
    """Raised when paper mode is off but the explicit live gate is closed."""


def assert_live_trading_allowed(paper_trading: bool, allow_live_trading: bool) -> None:
    """Fail closed unless live orders are explicitly armed.

    Paper mode returns without error. Callers must still skip the exchange
    while `paper_trading` is true. Live orders are permitted only when paper
    mode is off and `allow_live_trading` is true.
    """
    if paper_trading:
        return
    if allow_live_trading:
        return
    raise LiveTradingDisabled(
        "Refusing to place live orders: PAPER_TRADING is false but "
        "ALLOW_LIVE_TRADING is not true. Live orders require both "
        "PAPER_TRADING=false and ALLOW_LIVE_TRADING=true. "
        "Set PAPER_TRADING=true to stay on the practice book."
    )


def live_orders_permitted(paper_trading: bool, allow_live_trading: bool) -> bool:
    """True only when both gates are open for real exchange orders."""
    return (not paper_trading) and bool(allow_live_trading)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_order(order: dict[str, Any] | None) -> OrderOutcome:
    """Map a ccxt-style order payload to filled, pending, or canceled.

    Missing payloads are not fills. A cancel or expiry that already executed
    some size is `filled` so the local book tracks coins the account holds.
    """
    if not isinstance(order, dict):
        return "canceled"
    status = str(order.get("status") or "").strip().lower()
    filled = _as_float(order.get("filled"))
    amount = _as_float(order.get("amount"))
    if status in _CANCELED:
        return "filled" if filled > 0 else "canceled"
    if status in _FILLED:
        if filled > 0 or amount > 0:
            return "filled"
        return "canceled"
    if amount > 0 and filled + 1e-12 >= amount:
        return "filled"
    return "pending"


def order_id_of(order: dict[str, Any]) -> str:
    """Exchange order id, including Kraken's `info.txid` shape."""
    for key in ("id", "order_id", "orderId"):
        value = order.get(key)
        if value not in (None, ""):
            return str(value)
    info = order.get("info")
    if isinstance(info, dict):
        txid = info.get("txid")
        if isinstance(txid, list) and txid:
            return str(txid[0])
        if isinstance(txid, str) and txid:
            return txid
    return ""


def filled_qty(order: dict[str, Any], fallback: float) -> float:
    """Size that actually executed, else the requested size on a full close."""
    filled = _as_float(order.get("filled"))
    if filled > 0:
        return filled
    amount = _as_float(order.get("amount"))
    status = str(order.get("status") or "").strip().lower()
    if amount > 0 and status in _FILLED:
        return amount
    return fallback


def fill_price(order: dict[str, Any], fallback: float) -> float:
    """Average fill when the exchange reports one, else the limit price."""
    for key in ("average", "avgPrice", "price"):
        value = _as_float(order.get(key))
        if value > 0:
            return value
    return fallback


def build_pending_entry(
    *,
    order_id: str,
    qty: float,
    price: float,
    stop: float,
    target: float,
    atr: float,
    action: str,
    reason: str,
    opened_bar: str,
    macro_regime: str,
    confidence: float,
    model: str | None,
    rationale: str,
    adx: float,
    macro_ema: float,
) -> dict[str, Any]:
    """Serializable rest-order record. This is not an open position."""
    return {
        "order_id": str(order_id),
        "qty": float(qty),
        "price": float(price),
        "stop": float(stop),
        "target": float(target),
        "atr": float(atr),
        "action": str(action),
        "reason": str(reason),
        "opened_bar": opened_bar,
        "macro_regime": str(macro_regime),
        "confidence": float(confidence),
        "model": model,
        "rationale": str(rationale),
        "adx": float(adx),
        "macro_ema": float(macro_ema),
    }
