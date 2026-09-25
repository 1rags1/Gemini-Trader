"""Offline checks for the multi-pair dashboard snapshot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dashboard import (
    PAGE,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    RUNNER_STALE_SECONDS,
    SNAPSHOT_STALE_SECONDS,
    _enrich_trade,
    desk_labels,
    read_runner_state,
    read_state_feed,
    snapshot_is_complete,
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
    assert "ACTIVE: PAPER TRADING" in PAGE
    assert "ACTIVE: LIVE TRADING" in PAGE
    assert "PRACTICE / PAPER" in PAGE
    assert "LIVE / KRAKEN" in PAGE
    assert "Fake money" in PAGE
    assert "Real money" in PAGE
    assert "NOT IN USE" in PAGE
    assert "IN USE" in PAGE
    assert 'id="mode-banner"' in PAGE
    assert 'id="conn-banner"' in PAGE
    assert 'id="conn-title"' in PAGE
    assert "Waiting for VPS…" in PAGE
    assert 'title: "Live"' in PAGE
    assert 'title: "Stale"' in PAGE
    assert 'title: "Disconnected"' in PAGE
    assert "Last good snapshot was" in PAGE
    assert "Last updated" in PAGE
    assert "Numbers below are the last good snapshot." in PAGE
    assert "?token=" in PAGE
    assert "DASHBOARD_SECRET" in PAGE
    assert "WEBHOOK_SECRET" in PAGE
    assert "The secret is not shown" in PAGE
    assert "equity or trades missing" in PAGE
    assert "const FETCH_TIMEOUT_MS = 8000;" in PAGE
    assert "AbortController" in PAGE
    assert "function blankBooks()" in PAGE
    assert "if (view.hideNumbers) blankBooks();" in PAGE
    assert f"let staleAfterSeconds = {SNAPSHOT_STALE_SECONDS};" in PAGE
    assert f"let runnerStaleAfterSeconds = {RUNNER_STALE_SECONDS};" in PAGE
    assert SNAPSHOT_STALE_SECONDS == 60
    assert RUNNER_STALE_SECONDS == 120
    html = PAGE.split("/* __CONN_JS_START__ */", 1)[0]
    equity_at = html.index('id="paper-equity"')
    equity_snip = html[equity_at:equity_at + 180]
    assert ">—" in equity_snip
    assert "10,000" not in equity_snip
    assert "10000" not in equity_snip
    assert "$10,000.00" not in html
    assert 'data-feed="disconnected"' in html
    assert 'id="desk"' in html
    assert "connecting…" not in PAGE
    assert "poll failed:" not in PAGE
    assert "Waiting for live data" not in PAGE
    tick = PAGE.split("async function tick()", 1)[1]
    assert tick.index("snapshotComplete") < tick.index("render(data)")
    assert "pollProblem = null" in tick
    assert "ctrl.abort()" in tick
    print("    connection banner hides the $10k placeholder until a snapshot")
    assert 'id="paper-panel"' in PAGE
    assert 'id="live-panel"' in PAGE
    assert "lg:grid-cols-2" in PAGE
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
    assert paper["active_banner"] == "ACTIVE: PAPER TRADING"
    assert paper["active_mode"] == "paper"
    live = desk_labels(paper=False)
    assert live["desk_title"] == "Live Execution Desk"
    assert live["equity_label"] == "LIVE KRAKEN EQUITY"
    assert live["live_badge"] == "● LIVE KRAKEN (USD)"
    assert live["active_banner"] == "ACTIVE: LIVE TRADING"
    assert live["active_mode"] == "live"
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
                "equity": 10000.0,
                "start_equity": 10000.0,
                "book": "paper",
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
    assert state["starting_equity"] == 10000.0
    assert abs(state["pnl"] or 0) < 1e-9
    assert abs(state["pnl_pct"] or 0) < 1e-9
    assert state["active_banner"] == "ACTIVE: PAPER TRADING"
    assert state["paper"]["in_use"] is True
    assert state["paper"]["use_label"] == "IN USE"
    assert state["paper"]["title"] == "PRACTICE / PAPER"
    assert state["live"]["in_use"] is False
    assert state["live"]["use_label"] == "NOT IN USE"
    assert state["live"]["placeholder"] == "Live not active"
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
    original_allow = dash.allow_live_trading
    dash._paths = lambda: {"state": state_path, "trades": tmp / "live_trades.csv", "rejects": tmp / "r.csv"}
    dash.is_paper_trading = lambda: False
    dash.allow_live_trading = lambda: True
    try:
        state = read_runner_state()
    finally:
        dash._paths = original_paths
        dash.is_paper_trading = original_paper
        dash.allow_live_trading = original_allow

    assert state["desk_title"] == "Live Execution Desk"
    assert state["equity_label"] == "LIVE KRAKEN EQUITY"
    assert state["live_badge"] == "● LIVE KRAKEN (USD)"
    assert state["active_banner"] == "ACTIVE: LIVE TRADING"
    assert state["starting_equity"] == 500.0
    assert state["pnl"] == 0.0
    assert state["pnl_pct"] == 0.0
    assert state["live"]["in_use"] is True
    assert state["live"]["use_label"] == "IN USE"
    assert state["live"]["title"] == "LIVE / KRAKEN"
    assert state["paper"]["in_use"] is False
    assert state["paper"]["use_label"] == "NOT IN USE"
    assert state["paper"]["equity"] == 10000.0
    assert state["paper"]["title"] == "PRACTICE / PAPER"
    print("    live equity +0.00 vs 500 start")


def test_idle_live_book_is_not_the_paper_account() -> None:
    """A ~$500 Kraken baseline must not be labeled as the $10k practice book."""
    tmp = Path(tempfile.mkdtemp(prefix="gemini-dash-"))
    state_path = tmp / "runner.json"
    state_path.write_text(
        json.dumps(
            {
                "equity": 500.0,
                "start_equity": 500.0,
                "book": "live",
                "circuit_breaker": {"loss_streak": 1, "tripped": False, "cooldown_bars": 0},
                "positions": {
                    "BTC/USD": {"status": "LONG", "side": "LONG", "entry_price": 77000, "qty": 0.01},
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
    original_allow = dash.allow_live_trading
    dash._paths = lambda: {"state": state_path, "trades": tmp / "paper_trades.csv", "rejects": tmp / "r.csv"}
    dash.is_paper_trading = lambda: True
    dash.allow_live_trading = lambda: False
    try:
        state = read_runner_state()
    finally:
        dash._paths = original_paths
        dash.is_paper_trading = original_paper
        dash.allow_live_trading = original_allow

    assert state["active_banner"] == "ACTIVE: PAPER TRADING"
    assert state["active_mode"] == "paper"
    assert state["paper"]["in_use"] is True
    assert state["paper"]["money"] == "Fake money"
    assert state["paper"]["equity"] == 10000.0
    assert state["paper"]["starting_equity"] == 10000.0
    assert all(card["status"] == "FLAT" for card in state["paper"]["positions"])
    assert state["live"]["in_use"] is False
    assert state["live"]["use_label"] == "NOT IN USE"
    assert state["live"]["idle"] is True
    assert state["live"]["money"] == "Real money"
    assert state["live"]["equity"] == 500.0
    assert state["live"]["balance_state"] == "last_known"
    assert "IDLE" in state["live"]["status_line"]
    live_btc = next(card for card in state["live"]["positions"] if card["symbol"] == "BTC/USD")
    assert live_btc["status"] == "LONG"
    assert state["equity"] == 10000.0
    print("    $10k paper in use, $500 Kraken idle")


def test_closed_live_gate_does_not_mark_kraken_in_use() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="gemini-dash-"))
    state_path = tmp / "runner.json"
    state_path.write_text(
        json.dumps({"equity": 500.0, "start_equity": 500.0, "positions": {}}),
        encoding="utf-8",
    )
    import core.dashboard as dash

    original_paths = dash._paths
    original_paper = dash.is_paper_trading
    original_allow = dash.allow_live_trading
    dash._paths = lambda: {"state": state_path, "trades": tmp / "t.csv", "rejects": tmp / "r.csv"}
    dash.is_paper_trading = lambda: False
    dash.allow_live_trading = lambda: False
    try:
        state = read_runner_state()
    finally:
        dash._paths = original_paths
        dash.is_paper_trading = original_paper
        dash.allow_live_trading = original_allow

    assert state["active_banner"] == "ACTIVE: PAPER TRADING"
    assert state["live_armed"] is False
    assert state["live"]["in_use"] is False
    assert state["live"]["use_label"] == "NOT IN USE"
    assert state["paper"]["in_use"] is False
    print("    live gate closed keeps Kraken idle")


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
        try:
            check_token(Req("127.0.0.1"), None)
            raise AssertionError("a configured secret must lock localhost too")
        except HTTPException as exc:
            assert exc.status_code == 401
            assert exc.detail == "invalid or missing token"
            assert "dash-secret" not in str(exc.detail)
        check_token(Req("127.0.0.1"), "dash-secret")
        check_token(Req("127.0.0.1", {"X-Dashboard-Token": "dash-secret"}), None)
        check_token(Req("127.0.0.1", {"Authorization": "Bearer dash-secret"}), None)
        try:
            check_token(Req("203.0.113.10"), "nope")
            raise AssertionError("bad token should be rejected")
        except HTTPException as excl:
            assert excl.status_code == 401
            assert "dash-secret" not in str(excl.detail)
        check_token(Req("203.0.113.10"), "dash-secret")
        print("    secret locks every client; public bind still fails closed")
    finally:
        os.environ.pop("DASHBOARD_SECRET", None)
        os.environ.pop("WEBHOOK_SECRET", None)
        get_settings.cache_clear()


def test_snapshot_complete_requires_equity_and_trades() -> None:
    assert snapshot_is_complete(None) is False
    assert snapshot_is_complete({}) is False
    assert snapshot_is_complete({"generated_at": "t", "paper": {}, "live": {}, "trades": {"trades": []}}) is False
    assert snapshot_is_complete(
        {
            "generated_at": "t",
            "paper": {"equity": None},
            "live": {},
            "trades": {"trades": []},
        }
    ) is False
    assert snapshot_is_complete(
        {
            "generated_at": "t",
            "paper": {"equity": 10000},
            "live": {},
            "trades": {"count": 0},
        }
    ) is False
    assert snapshot_is_complete(
        {
            "generated_at": "t",
            "paper": {"equity": 10000},
            "live": {"equity": None, "placeholder": "Live not active"},
            "trades": {"trades": []},
        }
    ) is True
    print("    incomplete snapshot is not a book")


def test_state_feed_age() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="gemini-dash-feed-"))
    runner = tmp / "runner.json"
    paper = tmp / "paper.json"
    runner.write_text("{}", encoding="utf-8")
    old = time.time() - 500
    os.utime(runner, (old, old))
    paper.write_text("{}", encoding="utf-8")

    import core.dashboard as dash

    original = dash._paths
    dash._paths = lambda: {"state": runner, "trades": tmp / "paper_trades.csv", "rejects": tmp / "r.csv"}
    try:
        stale = read_state_feed(active_mode="paper")
        missing = None
        runner.unlink()
        paper.unlink()
        missing = read_state_feed(active_mode="paper")
    finally:
        dash._paths = original

    assert stale["state_present"] is True
    assert stale["state_file"] == "paper.json"
    assert stale["state_age_seconds"] is not None and stale["state_age_seconds"] < 5
    assert stale["stale_after_seconds"] == SNAPSHOT_STALE_SECONDS
    assert stale["runner_stale_after_seconds"] == RUNNER_STALE_SECONDS
    assert missing["state_present"] is False
    assert missing["state_age_seconds"] is None
    print("    runner file age is newest of paper.json and runner.json")


def test_connection_classifier_js() -> None:
    """Run the page's own classifier. A failed poll must not look live."""
    start = PAGE.index("/* __CONN_JS_START__ */") + len("/* __CONN_JS_START__ */")
    end = PAGE.index("/* __CONN_JS_END__ */")
    src = PAGE[start:end]
    waiting = {
        "everLoaded": False,
        "problem": None,
        "httpStatus": None,
        "snapshotAgeSeconds": None,
        "runnerPresent": False,
        "runnerAgeSeconds": None,
        "staleAfterSeconds": 60,
        "runnerStaleAfterSeconds": 120,
    }
    program = src + """
const failures = [];
function check(name, input, expect) {
  const got = classifyConnection(input);
  if (got.level !== expect.level) failures.push(name + " level " + got.level);
  if (got.title !== expect.title) failures.push(name + " title " + got.title);
  if (expect.hideNumbers !== undefined && !!got.hideNumbers !== !!expect.hideNumbers) {
    failures.push(name + " hideNumbers " + got.hideNumbers);
  }
  for (const bit of expect.has || []) {
    if (!String(got.detail).includes(bit)) failures.push(name + " missing " + bit + " in " + got.detail);
  }
  for (const bit of expect.lacks || []) {
    if (String(got.detail).includes(bit) || String(got.title).includes(bit)) failures.push(name + " leaked " + bit);
  }
}
const base = """ + json.dumps(waiting) + """;
check("waiting", base, {level: "disconnected", title: "Disconnected", hideNumbers: true, has: ["Waiting for VPS"]});
check("unauthorized", Object.assign({}, base, {problem: "unauthorized", httpStatus: 401}), {
  level: "disconnected",
  title: "Disconnected",
  hideNumbers: true,
  has: ["401", "?token=", "DASHBOARD_SECRET", "WEBHOOK_SECRET", "not shown"],
  lacks: ["dash-secret", "super-secret-value", "10000"]
});
check("offline", Object.assign({}, base, {problem: "network"}), {
  level: "disconnected",
  title: "Disconnected",
  hideNumbers: true,
  has: ["Waiting for VPS", "Cannot reach the VPS"]
});
check("offline-after-load", Object.assign({}, base, {problem: "network", everLoaded: true, snapshotAgeSeconds: 12, lastUpdatedLabel: "2026-09-25 16:00:00 UTC"}), {
  level: "disconnected",
  title: "Disconnected",
  hideNumbers: false,
  has: ["Last updated 12s ago", "2026-09-25 16:00:00 UTC", "last good snapshot"],
  lacks: ["Waiting for VPS"]
});
check("http", Object.assign({}, base, {problem: "http", httpStatus: 500, everLoaded: true, snapshotAgeSeconds: 4}), {
  level: "disconnected",
  title: "Disconnected",
  hideNumbers: false,
  has: ["HTTP 500", "Last updated 4s ago"]
});
check("timeout", Object.assign({}, base, {problem: "timeout"}), {
  level: "disconnected",
  title: "Disconnected",
  hideNumbers: true,
  has: ["Waiting for VPS", "too long"]
});
check("timeout-after-load", Object.assign({}, base, {problem: "timeout", everLoaded: true, snapshotAgeSeconds: 9}), {
  level: "disconnected",
  title: "Disconnected",
  hideNumbers: false,
  has: ["too long", "Last updated 9s ago", "last good snapshot"]
});
check("incomplete", Object.assign({}, base, {problem: "incomplete", httpStatus: 200}), {
  level: "disconnected",
  title: "Disconnected",
  hideNumbers: true,
  has: ["equity or trades missing", "Waiting for VPS"]
});
check("live", Object.assign({}, base, {everLoaded: true, snapshotAgeSeconds: 3, runnerPresent: true, runnerAgeSeconds: 12}), {
  level: "live",
  title: "Live",
  hideNumbers: false,
  has: ["Updated 3s ago", "Runner file 12s old"]
});
check("stale-snapshot", Object.assign({}, base, {everLoaded: true, snapshotAgeSeconds: 61, runnerPresent: true, runnerAgeSeconds: 12}), {
  level: "stale",
  title: "Stale",
  hideNumbers: false,
  has: ["Last good snapshot was 1m 1s ago", "may be old"]
});
check("stale-runner", Object.assign({}, base, {everLoaded: true, snapshotAgeSeconds: 3, runnerPresent: true, runnerAgeSeconds: 121}), {
  level: "stale",
  title: "Stale",
  hideNumbers: false,
  has: ["Last good snapshot was 3s ago", "runner file is 2m 1s old"]
});
check("recover", Object.assign({}, base, {everLoaded: true, snapshotAgeSeconds: 1, runnerPresent: true, runnerAgeSeconds: 4, problem: null}), {
  level: "live",
  title: "Live",
  hideNumbers: false,
  has: ["Updated 1s ago"]
});
if (snapshotComplete(null) !== false) failures.push("null snapshot");
if (snapshotComplete({generated_at: "t", paper: {}, live: {}, trades: {trades: []}}) !== false) failures.push("missing equity");
if (snapshotComplete({generated_at: "t", paper: {equity: 10000}, live: {}, trades: {}}) !== false) failures.push("missing trades");
if (snapshotComplete({generated_at: "t", paper: {equity: 10000}, live: {equity: null}, trades: {trades: []}}) !== true) failures.push("good snapshot");
if (failures.length) {
  console.error(failures.join("\\n"));
  process.exit(1);
}
console.log("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    print("    js classifier: waiting, 401, disconnect, stale, recover")


def main() -> int:
    checks = [
        ("page surfaces", test_page_has_multi_pair_surfaces),
        ("desk labels", test_desk_labels_toggle),
        ("trade symbol", test_enrich_trade_keeps_symbol),
        ("position book", test_read_positions_book),
        ("live return math", test_live_start_equity_return_math),
        ("paper vs idle live", test_idle_live_book_is_not_the_paper_account),
        ("closed live gate", test_closed_live_gate_does_not_mark_kraken_in_use),
        ("bind requires secret", test_dashboard_bind_requires_secret_off_localhost),
        ("snapshot complete", test_snapshot_complete_requires_equity_and_trades),
        ("state feed age", test_state_feed_age),
        ("connection classifier", test_connection_classifier_js),
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
