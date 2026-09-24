"""One Kraken USD book for BTC, ETH, and SOL.

OHLCV scans use `BTC/USD`, `ETH/USD`, and `SOL/USD`. A second quote
(`ETH/USDT`, `XBT/USD`, `ZUSD`) is the same asset, not a second slot.
Leaving those keys in `runner.json` made a ghost position that the USD
scan never managed.
"""

from __future__ import annotations

from typing import Any

from core.config import TRADING_PAIRS, empty_position_book, hydrate_position_slot

#: Bases the runner is allowed to track. Kraken's legacy XBT code maps to BTC.
BASE_ALIASES = {"XBT": "BTC", "XXBT": "BTC"}
UNIVERSE = ("BTC", "ETH", "SOL")

#: Same Kraken USD market under another code. Safe to rename the key.
SAME_MARKET_QUOTES = frozenset({"USD", "ZUSD"})

#: Different market. Paper mode may fold the slot onto USD. Live mode must not.
CROSS_QUOTES = frozenset({"USDT", "USDC"})

OFFICIAL_QUOTE = "USD"


def _clean(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("-", "/").replace("_", "/")
    if ":" in raw:
        raw = raw.split(":")[-1]
    return raw.replace("PERP", "")


def parse_pair(symbol: str) -> tuple[str, str] | None:
    """Split `ETH/USDT` or `ETHUSDT` into `(ETH, USDT)` without rewriting the quote."""
    raw = _clean(symbol)
    if not raw:
        return None
    if "/" not in raw:
        for quote in ("USDT", "USDC", "ZUSD", "USD", "EUR", "GBP"):
            if raw.endswith(quote) and len(raw) > len(quote):
                raw = f"{raw[: -len(quote)]}/{quote}"
                break
    if "/" not in raw:
        return None
    base, quote = raw.split("/", 1)
    base = BASE_ALIASES.get(base, base)
    if not base or not quote:
        return None
    return base, quote


def normalize_trading_symbol(symbol: str) -> str | None:
    """Map a configured or stored symbol onto `BASE/USD`, or None if it is outside the book."""
    parsed = parse_pair(symbol)
    if parsed is None:
        return None
    base, quote = parsed
    if base not in UNIVERSE:
        return None
    if quote in SAME_MARKET_QUOTES or quote in CROSS_QUOTES:
        return f"{base}/{OFFICIAL_QUOTE}"
    return None


def canonical_pairs(pairs: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Collapse aliases onto the official USD universe, preserving order."""
    chosen: list[str] = []
    for symbol in pairs or ():
        norm = normalize_trading_symbol(symbol)
        if norm and norm not in chosen:
            chosen.append(norm)
    if not chosen:
        return tuple(TRADING_PAIRS)
    return tuple(chosen)


def _status(slot: dict[str, Any] | None) -> str:
    if not isinstance(slot, dict):
        return "FLAT"
    return str(slot.get("status") or slot.get("side") or "FLAT").upper() or "FLAT"


def _open(slot: dict[str, Any] | None) -> bool:
    return _status(slot) not in {"", "FLAT"}


def _note(
    action: str,
    symbol: str,
    *,
    canonical: str | None = None,
    reason: str,
    slot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "symbol": symbol,
        "canonical": canonical,
        "reason": reason,
    }
    if isinstance(slot, dict) and _open(slot):
        payload["status"] = _status(slot)
        payload["entry_price"] = slot.get("entry_price")
        payload["size"] = slot.get("size", slot.get("qty"))
    return payload


def reconcile_book(
    positions: dict[str, Any] | None,
    pairs: tuple[str, ...],
    *,
    paper: bool,
) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    """Return a USD-only book and a list of ghost actions.

    Paper mode moves a cross-quote open (ETH/USDT) onto the USD slot when
    that slot is flat, so the runner can still manage the stop. If the USD
    slot is already open, the alias is dropped. Live mode renames same-market
    codes (XBT, ZUSD) and quarantines a different quote instead of selling
    the wrong pair.
    """
    book = empty_position_book(pairs)
    notes: list[dict[str, Any]] = []
    incoming = []
    if isinstance(positions, dict):
        for symbol, slot in positions.items():
            if isinstance(slot, dict):
                incoming.append((str(symbol), slot))

    exact: dict[str, dict[str, Any]] = {}
    aliases: list[tuple[str, str, str, dict[str, Any]]] = []
    for symbol, slot in incoming:
        parsed = parse_pair(symbol)
        canonical = normalize_trading_symbol(symbol)
        if parsed is None or canonical is None or canonical not in pairs:
            if _open(slot):
                notes.append(
                    _note(
                        "quarantined",
                        symbol,
                        canonical=canonical,
                        reason="outside the BTC/ETH/SOL USD book",
                        slot=slot,
                    )
                )
            continue
        base, quote = parsed
        displayed = f"{base}/{quote}"
        if quote in SAME_MARKET_QUOTES and displayed == canonical:
            exact[canonical] = slot
            continue
        if quote in SAME_MARKET_QUOTES:
            aliases.append((symbol, canonical, "same-market alias", slot))
            continue
        aliases.append((symbol, canonical, "cross-quote alias", slot))

    for symbol, slot in exact.items():
        book[symbol] = hydrate_position_slot(slot)

    for symbol, canonical, kind, slot in aliases:
        if not _open(slot):
            continue
        if kind == "same-market alias" or (paper and not _open(book.get(canonical))):
            if _open(book.get(canonical)):
                notes.append(
                    _note(
                        "quarantined",
                        symbol,
                        canonical=canonical,
                        reason=f"{kind} conflicts with an open {canonical} slot",
                        slot=slot,
                    )
                )
                continue
            book[canonical] = hydrate_position_slot(slot)
            notes.append(
                _note(
                    "migrated",
                    symbol,
                    canonical=canonical,
                    reason=f"{kind} folded onto the USD book",
                    slot=slot,
                )
            )
            continue
        notes.append(
            _note(
                "quarantined",
                symbol,
                canonical=canonical,
                reason=(
                    "live mode will not retarget a different quote"
                    if not paper
                    else f"{kind} conflicts with an open {canonical} slot"
                ),
                slot=slot,
            )
        )
    return book, notes


def reconcile_pending(
    pending: dict[str, Any] | None,
    pairs: tuple[str, ...],
    *,
    paper: bool,
) -> tuple[dict[str, dict], list[dict[str, Any]]]:
    """Same alias rules for resting entry orders. Live cross-quote orders are not retargeted."""
    kept: dict[str, dict] = {}
    notes: list[dict[str, Any]] = []
    if not isinstance(pending, dict):
        return kept, notes
    for symbol, record in pending.items():
        if not isinstance(record, dict):
            continue
        parsed = parse_pair(str(symbol))
        canonical = normalize_trading_symbol(str(symbol))
        if parsed is None or canonical is None or canonical not in pairs:
            notes.append(
                _note(
                    "quarantined",
                    str(symbol),
                    canonical=canonical,
                    reason="pending order outside the USD book",
                    slot=record,
                )
            )
            continue
        _base, quote = parsed
        same_market = quote in SAME_MARKET_QUOTES
        if canonical in kept:
            notes.append(
                _note(
                    "quarantined",
                    str(symbol),
                    canonical=canonical,
                    reason="pending order duplicates a canonical slot",
                    slot=record,
                )
            )
            continue
        if same_market or paper:
            kept[canonical] = dict(record)
            if str(symbol).upper().replace("-", "/") != canonical and f"{_base}/{quote}" != canonical:
                notes.append(
                    _note(
                        "migrated",
                        str(symbol),
                        canonical=canonical,
                        reason="pending order folded onto the USD book",
                        slot=record,
                    )
                )
            continue
        notes.append(
            _note(
                "quarantined",
                str(symbol),
                canonical=canonical,
                reason="live mode will not retarget a pending order onto a different quote",
                slot=record,
            )
        )
    return kept, notes
