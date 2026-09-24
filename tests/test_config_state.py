"""Offline checks for MTF config defaults and runner.json migration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    ATR_PROFIT_MULTIPLIER,
    ATR_STOP_MULTIPLIER,
    MACRO_ADX_THRESHOLD,
    MACRO_EMA_FAST,
    MACRO_EMA_SLOW,
    MACRO_EMA_TREND,
    MACRO_TIMEFRAME,
    MAX_OPEN_POSITIONS,
    POLL_INTERVAL_SECONDS,
    ALLOW_LIVE_TRADING,
    POSITION_SIZE_FRACTION,
    PAPER_TRADING,
    SPOT_LONG_ONLY,
    TRADING_PAIRS,
    TRIGGER_EMA_FAST,
    TRIGGER_EMA_SLOW,
    TRIGGER_TIMEFRAME,
    USE_POST_ONLY,
    canonical_position_slot,
    dump_runner_state,
    empty_position_book,
    migrate_circuit_breaker,
    migrate_position_book,
)


def test_mtf_defaults() -> None:
    assert TRADING_PAIRS == ("BTC/USD", "ETH/USD", "SOL/USD")
    assert MACRO_TIMEFRAME == "1h"
    assert TRIGGER_TIMEFRAME == "15m"
    assert POLL_INTERVAL_SECONDS == 30
    assert MACRO_EMA_FAST == 21
    assert MACRO_EMA_SLOW == 55
    assert MACRO_EMA_TREND == 200
    assert MACRO_ADX_THRESHOLD == 20.0
    assert TRIGGER_EMA_FAST == 9
    assert TRIGGER_EMA_SLOW == 21
    assert MAX_OPEN_POSITIONS == 2
    assert POSITION_SIZE_FRACTION == 0.25
    assert ATR_STOP_MULTIPLIER == 1.5
    assert ATR_PROFIT_MULTIPLIER == 3.5
    assert SPOT_LONG_ONLY is True
    assert PAPER_TRADING is True
    assert ALLOW_LIVE_TRADING is False
    assert USE_POST_ONLY is True
    print("    MTF constants")


def test_free_quote_balance_prefers_usd_then_zusd() -> None:
    from core.broker import free_quote_balance

    assert free_quote_balance({"free": {"USD": 512.5, "USDT": 99.0}}) == 512.5
    assert (
        free_quote_balance(
            {
                "free": {},
                "total": {},
                "info": {"result": {"ZUSD": "498.25", "XXBT": "0.01"}},
            }
        )
        == 498.25
    )
    assert free_quote_balance({"free": {"USDT": 100.0}}) == 100.0
    assert free_quote_balance({"free": {}}) == 0.0
    print("    USD/ZUSD quote balance parse")


def test_empty_book_schema() -> None:
    book = empty_position_book()
    assert list(book) == list(TRADING_PAIRS)
    for slot in book.values():
        assert slot == {
            "status": "FLAT",
            "entry_price": 0.0,
            "size": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "entry_time": None,
            "macro_regime": "UNKNOWN",
        }
    print("    empty book slots")


def test_migrate_legacy_book_and_breaker() -> None:
    raw = {
        "equity": 9900,
        "loss_streak": 2,
        "breaker_active": True,
        "position": {
            "side": "LONG",
            "symbol": "BTC/USD",
            "entry_price": 77000,
            "qty": 0.1,
            "stop": 76000,
            "target": 80000,
            "opened_bar": "2026-09-11T16:00:00+00:00",
        },
    }
    book = migrate_position_book(raw)
    btc = book["BTC/USD"]
    assert btc["status"] == "LONG"
    assert btc["size"] == 0.1
    assert btc["stop_loss"] == 76000
    assert btc["take_profit"] == 80000
    assert btc["entry_time"] == "2026-09-11T16:00:00+00:00"
    assert btc["macro_regime"] == "BULL"
    assert book["ETH/USD"]["status"] == "FLAT"
    breaker = migrate_circuit_breaker(raw)
    assert breaker == {"loss_streak": 2, "tripped": True, "cooldown_bars": 0}
    print("    legacy position + breaker migrated")


def test_dump_canonical_state() -> None:
    payload = dump_runner_state(
        equity=10000,
        positions={
            "ETH/USD": {
                "status": "LONG",
                "qty": 1.5,
                "entry_price": 2500,
                "trail_stop": 2400,
                "target": 2800,
                "opened_bar": "bar-1",
            }
        },
        circuit_breaker={"loss_streak": 0, "tripped": False},
        pairs=TRADING_PAIRS,
    )
    assert set(payload["positions"]) == set(TRADING_PAIRS)
    eth = payload["positions"]["ETH/USD"]
    assert eth == {
        "status": "LONG",
        "entry_price": 2500.0,
        "size": 1.5,
        "stop_loss": 2400.0,
        "take_profit": 2800.0,
        "entry_time": "bar-1",
        "macro_regime": "BULL",
    }
    assert payload["circuit_breaker"] == {"loss_streak": 0, "tripped": False, "cooldown_bars": 0}
    assert "qty" not in eth
    print("    dump uses MTF slot fields")


def test_on_disk_runner_json() -> None:
    """Local runtime state is optional. When present, it must not hold secrets."""
    path = PROJECT_ROOT / "state" / "runner.json"
    if not path.exists():
        print("    no local state/runner.json (runtime state stays gitignored)")
        return
    text = path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" not in text
    assert "EXCHANGE_API_SECRET" not in text
    raw = json.loads(text)
    assert "circuit_breaker" in raw
    assert isinstance(raw.get("positions"), dict)
    for symbol, slot in raw["positions"].items():
        canonical = canonical_position_slot(slot)
        assert set(canonical) == {
            "status",
            "entry_price",
            "size",
            "stop_loss",
            "take_profit",
            "entry_time",
            "macro_regime",
        }
        print(f"    {symbol} {canonical['status']}")


def main() -> int:
    checks = [
        ("MTF defaults", test_mtf_defaults),
        ("empty book", test_empty_book_schema),
        ("legacy migrate", test_migrate_legacy_book_and_breaker),
        ("canonical dump", test_dump_canonical_state),
        ("on-disk runner.json", test_on_disk_runner_json),
    ]
    failures = 0
    for label, check in checks:
        try:
            check()
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"[ OK ] {label}")
    print("\n" + ("[PASS] config/state." if not failures else f"[FAIL] {failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
