"""Offline checks for the strategy-hardening pass.

Symbol normalize, ghost cleanup, paper $10k reset, equity continuity,
CSV exit columns, dashboard secret gate, and the Gemini reject/stats path.

    python tests/test_hardening.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from core.config import TRADING_PAIRS, dump_runner_state  # noqa: E402
from core.gemini_agent import system_instruction  # noqa: E402
from core.paper_book import sum_runner_closed_pnl  # noqa: E402
from core.practice_summary import build_summary  # noqa: E402
from core.symbols import canonical_pairs, normalize_trading_symbol, reconcile_book  # noqa: E402
from core.trade_log import TRADE_LOG_FIELDS, append_row, iter_rows  # noqa: E402
from test_strategy_runner import (  # noqa: E402
    NOW,
    StubAgent,
    macro_bear_frame,
    macro_bull_frame,
    runner_for,
    trigger_cross_frame,
    trigger_flat_frame,
    tmpdir,
)


def _flat(symbol: str) -> dict:
    return {
        "status": "FLAT",
        "entry_price": 0.0,
        "size": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "entry_time": None,
        "macro_regime": "BEAR",
    }


def test_symbol_normalize_and_dump_drops_aliases() -> None:
    assert normalize_trading_symbol("ETH/USDT") == "ETH/USD"
    assert normalize_trading_symbol("ETHUSDT") == "ETH/USD"
    assert normalize_trading_symbol("SOL/USDC") == "SOL/USD"
    assert normalize_trading_symbol("XBT/USD") == "BTC/USD"
    assert normalize_trading_symbol("BTC/ZUSD") == "BTC/USD"
    assert normalize_trading_symbol("SOL/EUR") is None
    assert canonical_pairs(("BTC/USDT", "ETH/USDT", "SOL/USD", "ETH/USD")) == (
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
    )
    payload = dump_runner_state(
        equity=10_000,
        positions={
            "ETH/USDT": {
                "status": "LONG",
                "entry_price": 2500,
                "size": 1,
                "stop_loss": 2400,
                "take_profit": 2800,
            }
        },
        pairs=TRADING_PAIRS,
    )
    assert "ETH/USDT" not in payload["positions"]
    assert set(payload["positions"]) == set(TRADING_PAIRS)
    assert payload["positions"]["ETH/USD"]["status"] == "FLAT"
    print("    USD normalize; dump does not keep a USDT key")


def test_ghost_usdt_moves_onto_usd_and_can_close() -> None:
    tmp = tmpdir()
    (tmp / "runner.json").write_text(
        json.dumps(
            {
                "equity": 10_000,
                "start_equity": 10_000,
                "book": "paper",
                "positions": {
                    "BTC/USD": _flat("BTC/USD"),
                    "ETH/USD": _flat("ETH/USD"),
                    "SOL/USD": _flat("SOL/USD"),
                    "ETH/USDT": {
                        "status": "LONG",
                        "entry_price": 2500,
                        "size": 1,
                        "stop_loss": 1,
                        "take_profit": 999999,
                        "entry_time": "2026-01-01T00:00:00+00:00",
                        "macro_regime": "BULL",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    def fetch(*, symbol: str, timeframe: str, **_k):
        return (macro_bear_frame() if timeframe == "1h" else trigger_flat_frame()).copy()

    runner = runner_for(
        macro_bear_frame(),
        trigger_flat_frame(),
        StubAgent("BUY", 0.95),
        tmp,
        fetch_fn=fetch,
        pairs=TRADING_PAIRS,
        fetch_delay=0,
    )
    assert "ETH/USDT" not in runner.state.positions
    assert runner.state.positions["ETH/USD"]["status"] == "LONG"
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert set(saved["positions"]) == set(TRADING_PAIRS)
    assert saved["positions"]["ETH/USD"]["status"] == "LONG"
    quarantine = json.loads((tmp / "quarantine_positions.json").read_text(encoding="utf-8"))
    assert any(item.get("action") == "migrated" and item.get("symbol") == "ETH/USDT" for item in quarantine["items"])

    report = runner.cycle(now=NOW)
    eth = next(leg for leg in report.legs if leg.symbol == "ETH/USD")
    assert eth.reason == "macro_regime_exit"
    assert eth.position == "FLAT"
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    close = rows[-1]
    assert close["action"] == "CLOSE"
    assert close["symbol"] == "ETH/USD"
    assert float(close["entry_price"]) == 2500
    assert float(close["exit_price"]) != float(close["entry_price"])
    assert float(close["stop_loss"]) == 1
    assert float(close["stop_loss"]) != float(close["entry_price"])
    assert close["exit_price"]
    assert "exit_price" in close
    assert runner.state.equity != 10_000
    print("    ETH/USDT ghost folded onto ETH/USD and closed with distinct prices")


def test_conflicting_alias_is_not_a_second_position() -> None:
    book, notes = reconcile_book(
        {
            "ETH/USD": {
                "status": "LONG",
                "entry_price": 2500,
                "size": 1,
                "stop_loss": 2400,
                "take_profit": 2800,
            },
            "ETH/USDT": {
                "status": "LONG",
                "entry_price": 2501,
                "size": 2,
                "stop_loss": 2400,
                "take_profit": 2800,
            },
        },
        TRADING_PAIRS,
        paper=True,
    )
    assert book["ETH/USD"]["size"] == 1
    assert "ETH/USDT" not in book
    assert any(note["action"] == "quarantined" and note["symbol"] == "ETH/USDT" for note in notes)
    live_book, live_notes = reconcile_book(
        {"ETH/USDT": {"status": "LONG", "entry_price": 1, "size": 1, "stop_loss": 1, "take_profit": 2}},
        TRADING_PAIRS,
        paper=False,
    )
    assert live_book["ETH/USD"]["status"] == "FLAT"
    assert any(note["action"] == "quarantined" for note in live_notes)
    print("    conflicting and live cross-quote slots stay off the USD book")


def test_paper_reset_to_10k_archives_and_spares_live() -> None:
    tmp = tmpdir()
    (tmp / "runner.json").write_text(
        json.dumps(
            {
                "equity": 500,
                "start_equity": 500,
                "book": "paper",
                "positions": {"BTC/USD": {"status": "LONG", "entry_price": 1, "size": 1}},
            }
        ),
        encoding="utf-8",
    )
    (tmp / "live.json").write_text(
        json.dumps({"equity": 123, "start_equity": 123, "book": "live", "positions": {}}),
        encoding="utf-8",
    )
    runner = runner_for(
        macro_bear_frame(),
        trigger_flat_frame(),
        StubAgent("HOLD", 0.2),
        tmp,
        reset_paper=True,
    )
    assert runner.state.equity == 10_000
    assert runner.state.start_equity == 10_000
    assert runner.state.positions["BTC/USD"]["status"] == "FLAT"
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert saved["book"] == "paper"
    assert saved["equity"] == 10_000
    assert saved["start_equity"] == 10_000
    assert saved.get("book_reset_at")
    archives = list((tmp / "archive").glob("paper-*.json"))
    assert archives, "old paper state should be archived"
    archived = json.loads(archives[0].read_text(encoding="utf-8"))
    assert archived["equity"] == 500
    live = json.loads((tmp / "live.json").read_text(encoding="utf-8"))
    assert live["equity"] == 123
    assert live["book"] == "live"
    print("    paper reset to 10000; live.json left alone")


def test_stuck_equity_absorbs_closed_pnl() -> None:
    tmp = tmpdir()
    (tmp / "runner.json").write_text(
        json.dumps(
            {
                "equity": 10_000,
                "start_equity": 10_000,
                "book": "paper",
                "positions": {"BTC/USD": _flat("BTC/USD")},
            }
        ),
        encoding="utf-8",
    )
    legacy = tmp / "paper_trades.csv"
    old_fields = [
        "timestamp", "symbol", "action", "entry_price", "stop_loss", "take_profit",
        "verdict", "confidence", "agent_action", "model", "alert_reason",
        "adx", "macro_ema", "loss_streak", "rationale",
    ]
    with legacy.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-09-01T00:00:00+00:00",
                "symbol": "BTC/USD",
                "action": "CLOSE",
                "entry_price": "9850",
                "stop_loss": "9850",
                "take_profit": "10350",
                "verdict": "CONFIRMED",
                "alert_reason": "atr_stop_long",
                "rationale": "exit LONG entry=10000.0000 fill=9850.0000 pnl=98.00 hit=stop",
            }
        )
    assert abs(sum_runner_closed_pnl(legacy) - 98.0) < 1e-9
    runner = runner_for(
        macro_bear_frame(),
        trigger_flat_frame(),
        StubAgent("HOLD", 0.2),
        tmp,
    )
    assert abs(runner.state.equity - 10_098) < 1e-6
    assert runner.state.start_equity == 10_000
    upgraded = iter_rows(legacy)
    assert float(upgraded[-1]["entry_price"]) == 10000
    assert float(upgraded[-1]["exit_price"]) == 9850
    assert float(upgraded[-1]["stop_loss"]) != float(upgraded[-1]["entry_price"])
    print("    stuck 10000 equity picked up +98 and the old CLOSE row was split")


def test_csv_close_columns_round_trip() -> None:
    tmp = tmpdir()
    path = tmp / "paper_trades.csv"
    append_row(
        path,
        {
            "timestamp": "2026-09-02T00:00:00+00:00",
            "symbol": "BTC/USD",
            "action": "CLOSE",
            "entry_price": 10000,
            "exit_price": 9850,
            "pnl": -39.5,
            "hit": "stop",
            "stop_loss": 9850,
            "take_profit": 10350,
            "verdict": "CONFIRMED",
            "source": "runner",
            "rationale": "exit LONG entry=10000 fill=9850 pnl=-39.5 hit=stop",
        },
    )
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == TRADE_LOG_FIELDS
    row = iter_rows(path)[-1]
    assert float(row["entry_price"]) == 10000
    assert float(row["exit_price"]) == 9850
    assert float(row["stop_loss"]) == 9850
    assert float(row["entry_price"]) != float(row["stop_loss"])
    print("    CLOSE row keeps entry, exit, stop, and target apart")


def test_gemini_reject_unavailable_and_ignored_stop() -> None:
    from core.gemini_stats import load_stats

    tmp = tmpdir()
    reject = StubAgent("REJECT", 0.92, stop=1, target=2)
    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        reject,
        tmp,
        gemini_retry_sleep=0,
    )
    report = runner.cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert report.reason == "agent_rejected"
    assert report.position == "FLAT"
    stats = load_stats(tmp / "gemini_stats.json")
    assert stats["rejected"] == 1
    assert stats["confirmed"] == 0
    prompt = system_instruction(paper_trading=True).lower()
    assert "reject" in prompt
    assert "high confidence" in prompt
    assert "you cannot" in prompt
    assert "override" in prompt

    down = tmpdir()
    dead = StubAgent(error=RuntimeError("HTTP 503 from Gemini"))
    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        dead,
        down,
        gemini_retry_sleep=0,
    )
    report = runner.cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert report.reason == "agent_unavailable"
    assert report.position == "FLAT"
    assert dead.calls == 2
    stats = load_stats(down / "gemini_stats.json")
    assert stats["agent_unavailable"] == 1
    assert not (down / "paper_trades.csv").exists()

    opened = tmpdir()
    bossy = StubAgent("BUY", 0.99, stop=1, target=9)
    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        bossy,
        opened,
    )
    report = runner.cycle(now=NOW)
    assert report.position == "LONG"
    assert runner.state.positions["BTC/USD"]["stop_loss"] == 9850
    assert runner.state.positions["BTC/USD"]["take_profit"] == 10350
    print("    REJECT and 503 fail closed; Gemini stop does not replace 9850")


def test_practice_summary_is_readable() -> None:
    import os

    tmp = tmpdir()
    (tmp / "state").mkdir()
    (tmp / "data").mkdir()
    (tmp / "state" / "runner.json").write_text(
        json.dumps(
            {
                "equity": 10098,
                "start_equity": 10000,
                "book": "paper",
                "positions": {
                    "BTC/USD": _flat("BTC/USD"),
                    "ETH/USD": {
                        "status": "LONG",
                        "entry_price": 2500,
                        "size": 1,
                        "stop_loss": 2400,
                        "take_profit": 2800,
                    },
                    "SOL/USD": _flat("SOL/USD"),
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp / "state" / "gemini_stats.json").write_text(
        json.dumps({"confirmed": 2, "rejected": 1, "agent_unavailable": 3}),
        encoding="utf-8",
    )
    append_row(
        tmp / "data" / "paper_trades.csv",
        {
            "timestamp": "2026-09-24T01:00:00+00:00",
            "symbol": "BTC/USD",
            "action": "CLOSE",
            "entry_price": 10000,
            "exit_price": 10100,
            "pnl": 98,
            "hit": "target",
            "stop_loss": 9850,
            "take_profit": 10350,
            "verdict": "CONFIRMED",
            "source": "runner",
            "rationale": "exit LONG entry=10000 fill=10100 pnl=98 hit=target",
        },
    )
    append_row(
        tmp / "data" / "rejected_alerts.csv",
        {
            "timestamp": "2026-09-24T02:00:00+00:00",
            "symbol": "SOL/USD",
            "action": "BUY",
            "verdict": "REJECTED",
            "rationale": "agent_unavailable",
            "alert_reason": "agent_unavailable",
            "source": "runner",
        },
    )
    (tmp / "logs").mkdir()
    (tmp / "logs" / "runner.log").write_text(
        "2026-09-24 02:05:00,000 ERROR runner: Gemini unavailable; fail closed\n",
        encoding="utf-8",
    )
    previous = {
        "PAPER_TRADING": os.environ.get("PAPER_TRADING"),
        "ALLOW_LIVE_TRADING": os.environ.get("ALLOW_LIVE_TRADING"),
        "DATA_DIR": os.environ.get("DATA_DIR"),
    }
    os.environ["PAPER_TRADING"] = "true"
    os.environ["ALLOW_LIVE_TRADING"] = "false"
    os.environ["DATA_DIR"] = str(tmp / "data")
    try:
        text = build_summary(
            root=tmp,
            now=datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert "Still paper? Yes" in text
    assert "Wins: 1" in text
    assert "Losses: 0" in text
    assert "ETH/USD LONG" in text
    assert "agent unavailable: 1" in text
    assert "confirmed=2" in text
    assert "ERROR" in text
    assert "dash-secret" not in text
    print("    practice summary lists paper, P&L, and Gemini counts")


def main() -> int:
    checks = [
        ("symbol normalize", test_symbol_normalize_and_dump_drops_aliases),
        ("ghost cleanup", test_ghost_usdt_moves_onto_usd_and_can_close),
        ("alias conflict", test_conflicting_alias_is_not_a_second_position),
        ("paper reset", test_paper_reset_to_10k_archives_and_spares_live),
        ("equity continuity", test_stuck_equity_absorbs_closed_pnl),
        ("csv columns", test_csv_close_columns_round_trip),
        ("gemini reject path", test_gemini_reject_unavailable_and_ignored_stop),
        ("practice summary", test_practice_summary_is_readable),
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
    print("\n" + ("[PASS] hardening." if not failures else f"[FAIL] {failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
