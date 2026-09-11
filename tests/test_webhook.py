"""End-to-end test of the paper execution server.

Posts synthetic TradingView payloads over HTTP and checks the verdict, the CSV
side effects, and the auth gate.

    python tests/test_webhook.py           # in-process, isolated temp trade log
    python tests/test_webhook.py --live    # against a running server on :5000

In-process mode routes httpx through FastAPI's TestClient, so no port is bound
and the real trade log is never touched. Live mode exercises the actual socket
and will append to data/paper_trades.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Redirect the trade log before core.config is first read, so the test cannot
# write into the real data/ directory.
_TEMP_DATA = Path(tempfile.mkdtemp(prefix="gemini-trader-test-"))
os.environ.setdefault("DATA_DIR", str(_TEMP_DATA))

import httpx  # noqa: E402

from core.config import get_settings  # noqa: E402

LIVE_URL = "http://127.0.0.1:5000"

#: Mirrors the JSON the Pine strategy emits via alert_message.
BASE_PAYLOAD = {
    "strategy": "Gemini Trend Guard",
    "action": "SELL",
    "reason": "macro_aligned_trend_short",
    "ticker": "BINANCE:BTCUSDT",
    "timeframe": "60",
    "price": 78323.7,
    "atr": 414.03,
    "stop": 78944.75,
    "target": 76771.07,
    "adx": 24.09,
    "macro_ema": 79087.73,
    "loss_streak": 1,
    "breaker_active": False,
    "position_size": 0.0,
    "bar_time": "2026-09-09T20:00:00Z",
}


def make_client(live: bool) -> httpx.Client:
    if live:
        return httpx.Client(base_url=LIVE_URL, timeout=120.0)

    from fastapi.testclient import TestClient

    from core.webhook_server import app

    return TestClient(app)


def post(client, payload: dict, **params) -> httpx.Response:
    return client.post("/webhook", json=payload, params=params or None)


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    print(f"    model={body['model']} min_confidence={body['min_confidence']}")


def test_symbol_and_timeframe_mapping() -> None:
    """TradingView tickers/intervals must translate to ccxt equivalents."""
    from core.webhook_server import to_ccxt_symbol, to_ccxt_timeframe

    assert to_ccxt_symbol("BINANCE:BTCUSDT") == "BTC/USDT"
    assert to_ccxt_symbol("COINBASE:ETHUSD") == "ETH/USD"
    assert to_ccxt_symbol("KRAKEN:BTC/USDT") == "BTC/USDT"
    assert to_ccxt_timeframe("60") == "1h"
    assert to_ccxt_timeframe("240") == "4h"
    assert to_ccxt_timeframe("D") == "1d"
    assert to_ccxt_timeframe(None) is None


def test_entry_alert_is_reviewed(client) -> None:
    """A normal entry alert returns a verdict and consults the agent.

    If the Gemini quota is exhausted the server must still answer 200 with a
    REJECTED/agent_unavailable verdict, since TradingView does not retry.
    """
    response = post(client, BASE_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["verdict"] in {"CONFIRMED", "REJECTED"}
    assert body["symbol"] == "BTC/USDT"
    print(f"    verdict={body['verdict']} ({body['reason']})")

    if body["reason"] == "agent_unavailable":
        print("    NOTE: agent unreachable; server failed closed as designed")
        assert body["stop_loss"] is None and body["take_profit"] is None
        return

    assert body["agent_action"] in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["rationale"]
    print(f"    agent={body['agent_action']} confidence={body['confidence']} model={body['model']}")
    if body.get("regime"):
        print(f"    regime={body['regime']['tradeable_direction']} adx={body['regime']['adx']}")


class _StubAgent:
    """Stands in for GeminiAgent so verdict logic is testable without quota."""

    last_model_used = "stub-model"

    def __init__(self, action: str, confidence: float, stop: float, target: float) -> None:
        self._decision_args = (action, confidence, stop, target)

    def decide(self, *_args, **_kwargs):
        from core.gemini_agent import Decision

        action, confidence, stop, target = self._decision_args
        return Decision(
            action=action,
            confidence=confidence,
            rationale="stubbed decision",
            stop_loss=stop,
            take_profit=target,
        )


@contextmanager
def using_agent(agent):
    """Temporarily swap the server's cached agent."""
    import core.webhook_server as server

    original = server._agent
    server._agent = agent
    try:
        yield
    finally:
        server._agent = original


def test_verdict_matrix(client) -> None:
    """Agreement plus confidence confirms; either one missing rejects.

    Driven by a stub so the three branches are exercised deterministically,
    independently of live model availability or quota.
    """
    cases = [
        ("SELL", 0.90, "CONFIRMED", "agent_agrees"),
        ("HOLD", 0.90, "REJECTED", "agent_returned_HOLD"),
        ("BUY", 0.95, "REJECTED", "agent_returned_BUY"),
        ("SELL", 0.20, "REJECTED", "confidence_0.20_below_0.60"),
    ]
    for action, confidence, expected_verdict, expected_reason in cases:
        stub = _StubAgent(action, confidence, stop=78944.75, target=76771.07)
        with using_agent(stub):
            body = post(client, BASE_PAYLOAD).json()

        assert body["verdict"] == expected_verdict, (action, confidence, body)
        assert body["reason"] == expected_reason, (action, confidence, body)

        if expected_verdict == "CONFIRMED":
            # A confirmed trade must carry the levels forward for the fill log.
            assert body["entry_price"] == BASE_PAYLOAD["price"]
            assert body["stop_loss"] == 78944.75
            assert body["take_profit"] == 76771.07
        else:
            assert body["stop_loss"] is None, "declined trade must carry no levels"

        print(f"    agent {action} @ {confidence:.2f} -> {body['verdict']} ({body['reason']})")


def test_confirmed_row_has_full_audit_trail(client) -> None:
    """A confirmed entry writes one complete row to paper_trades.csv."""
    from core.webhook_server import trade_log_path

    path = trade_log_path()
    before = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0

    with using_agent(_StubAgent("SELL", 0.88, stop=78944.75, target=76771.07)):
        assert post(client, BASE_PAYLOAD).json()["verdict"] == "CONFIRMED"

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    after = len(path.read_text(encoding="utf-8").splitlines())
    assert after == before + 1, f"expected exactly one new row, went {before} -> {after}"

    row = rows[-1]
    assert row["symbol"] == "BTC/USDT"
    assert row["action"] == "SELL"
    assert float(row["entry_price"]) == BASE_PAYLOAD["price"]
    assert float(row["stop_loss"]) == 78944.75
    assert float(row["take_profit"]) == 76771.07
    assert row["verdict"] == "CONFIRMED"
    assert row["agent_action"] == "SELL"
    assert row["model"] == "stub-model"
    assert row["timestamp"].startswith("20")

    # The logged levels must preserve the strategy's 2.5:1 structure.
    entry, stop, target = (float(row[k]) for k in ("entry_price", "stop_loss", "take_profit"))
    ratio = (entry - target) / (stop - entry)
    assert 2.4 <= ratio <= 2.6, f"logged reward:risk is {ratio:.2f}, expected ~2.5"
    print(f"    logged R:R {ratio:.2f} from entry {entry} stop {stop} target {target}")


def test_rejected_alerts_are_logged_separately(client) -> None:
    """Vetoes must be recorded so the rejection rate stays measurable."""
    from core.webhook_server import reject_log_path, trade_log_path

    with using_agent(_StubAgent("HOLD", 0.9, stop=0.0, target=0.0)):
        assert post(client, BASE_PAYLOAD).json()["verdict"] == "REJECTED"

    path = reject_log_path()
    assert path.exists(), f"no rejection log at {path}"
    assert path != trade_log_path(), "rejections must not pollute the trade log"

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[-1]["verdict"] == "REJECTED"
    print(f"    {len(rows)} rejection(s) in {path.name}")


def test_agent_failure_fails_closed(client) -> None:
    """A dead agent must produce REJECTED + HTTP 200, never a 500."""
    import core.webhook_server as server

    class DeadAgent:
        last_model_used = None

        def decide(self, *_args, **_kwargs):
            raise RuntimeError("simulated: no Gemini model in the chain answered")

    original = server._agent
    server._agent = DeadAgent()
    try:
        response = post(client, BASE_PAYLOAD)
        assert response.status_code == 200, f"expected 200, got {response.status_code}"
        body = response.json()
        assert body["verdict"] == "REJECTED"
        assert body["reason"] == "agent_unavailable"
        assert body["stop_loss"] is None, "declined trade must carry no levels"
    finally:
        server._agent = original


def test_circuit_breaker_is_rejected_without_llm(client) -> None:
    """A tripped breaker must short-circuit before the model is consulted."""
    response = post(client, {**BASE_PAYLOAD, "breaker_active": True})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["verdict"] == "REJECTED"
    assert body["reason"] == "circuit_breaker_active"
    assert body["agent_action"] is None, "model was consulted despite the breaker"


def test_close_alert_is_always_confirmed(client) -> None:
    """Exits are risk-reducing and must never be vetoed."""
    response = post(client, {**BASE_PAYLOAD, "action": "CLOSE"})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["verdict"] == "CONFIRMED"
    assert body["reason"] == "exit_not_subject_to_review"


def test_unknown_fields_are_tolerated(client) -> None:
    """Adding a key in Pine must not start returning 422 to TradingView."""
    response = post(client, {**BASE_PAYLOAD, "future_field": 123})
    assert response.status_code == 200, response.text


def test_malformed_payload_is_rejected(client) -> None:
    response = post(client, {"strategy": "x", "action": "SIDEWAYS", "ticker": "y"})
    assert response.status_code == 422, f"expected validation error, got {response.status_code}"


def test_auth_gate(client) -> None:
    """With a secret configured, an unauthenticated post must be refused."""
    get_settings.cache_clear()
    os.environ["WEBHOOK_SECRET"] = "test-secret-value"
    try:
        get_settings.cache_clear()
        assert post(client, BASE_PAYLOAD).status_code == 401
        assert post(client, BASE_PAYLOAD, token="wrong").status_code == 401
        assert post(client, {**BASE_PAYLOAD, "action": "CLOSE"},
                    token="test-secret-value").status_code == 200
    finally:
        os.environ.pop("WEBHOOK_SECRET", None)
        get_settings.cache_clear()


def test_confirmed_trades_are_logged() -> None:
    """Every confirmed trade must land in paper_trades.csv with its levels."""
    from core.webhook_server import trade_log_path

    path = trade_log_path()
    assert path.exists(), f"no trade log written at {path}"

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "trade log is empty"
    for column in ["timestamp", "symbol", "action", "entry_price", "stop_loss", "take_profit"]:
        assert column in rows[0], f"trade log missing column '{column}'"

    latest = rows[-1]
    assert latest["symbol"] == "BTC/USDT"
    assert latest["verdict"] == "CONFIRMED"
    print(f"    {len(rows)} row(s) in {path.name}; last action={latest['action']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live",
        action="store_true",
        help=f"post to a running server at {LIVE_URL} instead of in-process",
    )
    args = parser.parse_args()

    if args.live:
        print(f"Mode: live HTTP against {LIVE_URL}")
        try:
            httpx.get(f"{LIVE_URL}/health", timeout=5.0)
        except httpx.HTTPError as exc:
            print(f"[FAIL] no server at {LIVE_URL}: {exc}")
            print("       start it with: .\\.venv\\Scripts\\python.exe -m core.webhook_server")
            return 1
    else:
        print(f"Mode: in-process (trade log redirected to {_TEMP_DATA})")

    client = make_client(args.live)

    checks = [
        ("symbol/timeframe mapping", lambda: test_symbol_and_timeframe_mapping()),
        ("health endpoint", lambda: test_health(client)),
        ("malformed payload -> 422", lambda: test_malformed_payload_is_rejected(client)),
        ("unknown fields tolerated", lambda: test_unknown_fields_are_tolerated(client)),
        ("circuit breaker short-circuits", lambda: test_circuit_breaker_is_rejected_without_llm(client)),
        ("CLOSE always confirmed", lambda: test_close_alert_is_always_confirmed(client)),
        ("agent failure fails closed", lambda: test_agent_failure_fails_closed(client)),
        ("verdict matrix", lambda: test_verdict_matrix(client)),
        ("confirmed row audit trail", lambda: test_confirmed_row_has_full_audit_trail(client)),
        ("rejections logged separately", lambda: test_rejected_alerts_are_logged_separately(client)),
        ("live entry alert reviewed", lambda: test_entry_alert_is_reviewed(client)),
        ("confirmed trades logged to CSV", lambda: test_confirmed_trades_are_logged()),
        ("auth gate", lambda: test_auth_gate(client)),
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

    print("\n" + ("[PASS] Paper execution server is operational."
                  if not failures else f"[FAIL] {failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
