"""Offline checks for the multi-pair dashboard snapshot."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dashboard import (
    PAGE,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    _enrich_trade,
    desk_labels,
    read_runner_state,
)


def test_page_has_multi_pair_surfaces() -> None:
    assert 'id="pair-cards"' in PAGE
    assert 'id="desk-title"' in PAGE
    assert 'id="live-badge"' in PAGE
    assert 'id="equity-label"' in PAGE
    assert "Live Execution Desk" in PAGE
    assert "LIVE KRAKEN EQUITY" in PAGE
    assert "PAPER EQUITY" in PAGE
    assert "● LIVE KRAKEN (USD)" in PAGE
    assert ">Symbol</th>" in PAGE
    assert ">1h Regime</th>" in PAGE
    assert ">EMA 9</th>" in PAGE
    assert ">EMA 21</th>" in PAGE
    assert ">15m ATR</th>" in PAGE
    assert DASHBOARD_HOST == "127.0.0.1"
    assert DASHBOARD_PORT == 8050
    print("    page + bind defaults")


def test_desk_labels_toggle() -> None:
    paper = desk_labels(paper=True)
    assert paper["desk_title"] == "Paper desk"
    assert paper["equity_label"] == "PAPER EQUITY"
    assert paper["live_badge"] is None
    live = desk_labels(paper=False)
    assert live["desk_title"] == "Live Execution Desk"
    assert live["equity_label"] == "LIVE KRAKEN EQUITY"
    assert live["live_badge"] == "● LIVE KRAKEN (USD)"
    print("    paper/live desk labels")


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
                "equity": 500.0,
                "start_equity": 500.0,
                "circuit_breaker": {"loss_streak": 0, "tripped": False, "cooldown_bars": 0},
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
    assert state["starting_equity"] == 500.0
    assert abs(state["pnl"] or 0) < 1e-9
    assert abs(state["pnl_pct"] or 0) < 1e-9
    print("    book cards BTC LONG / ETH FLAT / SOL SHORT")


def test_live_start_equity_return_math() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="gemini-dash-"))
    state_path = tmp / "runner.json"
    state_path.write_text(
        json.dumps(
            {
                "equity": 500.0,
                "start_equity": 500.0,
                "circuit_breaker": {"loss_streak": 0, "tripped": False, "cooldown_bars": 0},
                "positions": {
                    "BTC/USD": {"status": "FLAT"},
                    "ETH/USD": {"status": "FLAT"},
                    "SOL/USD": {"status": "FLAT"},
                },
            }
        ),
        encoding="utf-8",
    )
    import core.dashboard as dash

    original_paths = dash._paths
    original_paper = dash.is_paper_trading
    dash._paths = lambda: {"state": state_path, "trades": tmp / "live_trades.csv", "rejects": tmp / "r.csv"}
    dash.is_paper_trading = lambda: False
    try:
        state = read_runner_state()
    finally:
        dash._paths = original_paths
        dash.is_paper_trading = original_paper

    assert state["desk_title"] == "Live Execution Desk"
    assert state["equity_label"] == "LIVE KRAKEN EQUITY"
    assert state["live_badge"] == "● LIVE KRAKEN (USD)"
    assert state["starting_equity"] == 500.0
    assert state["pnl"] == 0.0
    assert state["pnl_pct"] == 0.0
    print("    live equity +0.00 vs 500 start")


def test_dashboard_bind_requires_secret_off_localhost() -> None:
    import os

    from fastapi import HTTPException

    from core.config import get_settings
    from core.dashboard import check_token
    from core.exposure import assert_secret_for_public_bind

    assert_secret_for_public_bind(
        host="127.0.0.1",
        secret="",
        service="the dashboard",
        secret_name="DASHBOARD_SECRET",
    )
    try:
        assert_secret_for_public_bind(
            host="0.0.0.0",
            secret="",
            service="the dashboard",
            secret_name="DASHBOARD_SECRET",
        )
    except RuntimeError as exc:
        assert "DASHBOARD_SECRET" in str(exc)
    else:
        raise AssertionError("public dashboard bind must fail closed")
    assert_secret_for_public_bind(
        host="0.0.0.0",
        secret="dash",
        service="the dashboard",
        secret_name="DASHBOARD_SECRET",
    )

    class Req:
        def __init__(self, host: str, headers: dict | None = None) -> None:
            self.client = type("C", (), {"host": host})()
            self.headers = headers or {}

    os.environ["DASHBOARD_SECRET"] = ""
    os.environ["WEBHOOK_SECRET"] = ""
    get_settings.cache_clear()
    try:
        check_token(Req("127.0.0.1"), None)
        try:
            check_token(Req("203.0.113.10"), None)
            raise AssertionError("remote dashboard request should be rejected")
        except HTTPException as exc:
            assert exc.status_code == 401
            assert "DASHBOARD_SECRET" in exc.detail
        try:
            check_token(Req("127.0.0.1", {"cf-ray": "1"}), None)
            raise AssertionError("tunneled dashboard request should be rejected")
        except HTTPException as exc:
            assert exc.status_code == 401

        os.environ["DASHBOARD_SECRET"] = "dash-secret"
        get_settings.cache_clear()
        check_token(Req("127.0.0.1"), None)
        try:
            check_token(Req("203.0.113.10"), "nope")
            raise AssertionError("bad token should be rejected")
        except HTTPException as exc:
            assert exc.status_code == 401
        check_token(Req("203.0.113.10"), "dash-secret")
        print("    dashboard localhost open; public bind and tunnel need a secret")
    finally:
        os.environ.pop("DASHBOARD_SECRET", None)
        os.environ.pop("WEBHOOK_SECRET", None)
        get_settings.cache_clear()


def main() -> int:
    checks = [
        ("page surfaces", test_page_has_multi_pair_surfaces),
        ("desk labels", test_desk_labels_toggle),
        ("trade symbol", test_enrich_trade_keeps_symbol),
        ("position book", test_read_positions_book),
        ("live return math", test_live_start_equity_return_math),
        ("bind requires secret", test_dashboard_bind_requires_secret_off_localhost),
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
