"""Offline checks for the multi-pair dashboard snapshot."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dashboard import PAGE, DASHBOARD_HOST, DASHBOARD_PORT, _enrich_trade, read_runner_state


def test_page_has_multi_pair_surfaces() -> None:
    assert 'id="pair-cards"' in PAGE
    assert ">Symbol</th>" in PAGE
    assert ">1h Regime</th>" in PAGE
    assert ">EMA 9</th>" in PAGE
    assert ">EMA 21</th>" in PAGE
    assert ">15m ATR</th>" in PAGE
    assert DASHBOARD_HOST == "0.0.0.0"
    assert DASHBOARD_PORT == 8050
    print("    page + bind defaults")


def test_enrich_trade_keeps_symbol() -> None:
    row = _enrich_trade(
        {
            "timestamp": "2026-09-11T00:00:00+00:00",
            "symbol": "ETH/USD",
            "action": "BUY",
            "verdict": "CONFIRMED",
            "entry_price": "2500",
            "rationale": "entry=2500 fill=2510 pnl=1.2 hit=target",
        }
    )
    assert row["symbol"] == "ETH/USD"
    assert row["action"] == "BUY"
    print("    trade symbol retained")


def test_read_positions_book() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="gemini-dash-"))
    state_path = tmp / "runner.json"
    state_path.write_text(
        json.dumps(
            {
                "equity": 10100.0,
                "circuit_breaker": {"loss_streak": 1, "tripped": False},
                "last_bar": "2026-09-11T16:00:00+00:00",
                "last_bars": {
                    "BTC/USD": "2026-09-11T16:00:00+00:00",
                    "ETH/USD": "2026-09-11T16:00:00+00:00",
                    "SOL/USD": "2026-09-11T15:00:00+00:00",
                },
                "positions": {
                    "BTC/USD": {
                        "status": "LONG",
                        "side": "LONG",
                        "entry_price": 77000,
                        "stop": 76000,
                        "target": 80000,
                        "trail_stop": 76000,
                        "qty": 0.33,
                    },
                    "ETH/USD": {"status": "FLAT"},
                    "SOL/USD": {"status": "SHORT", "side": "SHORT", "entry_price": 100},
                },
            }
        ),
        encoding="utf-8",
    )

    import core.dashboard as dash

    original = dash._paths
    dash._paths = lambda: {"state": state_path, "trades": tmp / "t.csv", "rejects": tmp / "r.csv"}
    try:
        state = read_runner_state()
    finally:
        dash._paths = original

    by_symbol = {card["symbol"]: card for card in state["positions"]}
    assert list(by_symbol) == ["BTC/USD", "ETH/USD", "SOL/USD"]
    assert by_symbol["BTC/USD"]["status"] == "LONG"
    assert by_symbol["ETH/USD"]["status"] == "FLAT"
    assert by_symbol["SOL/USD"]["status"] == "SHORT"
    assert state["open_count"] == 2
    assert state["max_open_positions"] == 2
    assert state["position"]["entry_price"] == 77000
    print("    book cards BTC LONG / ETH FLAT / SOL SHORT")


def main() -> int:
    checks = [
        ("page surfaces", test_page_has_multi_pair_surfaces),
        ("trade symbol", test_enrich_trade_keeps_symbol),
        ("position book", test_read_positions_book),
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
    print("\n" + ("[PASS] dashboard." if not failures else f"[FAIL] {failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
