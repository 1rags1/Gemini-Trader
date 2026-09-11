"""Offline tests for the native strategy runner.

No network: market data is a synthetic frame and Gemini is a stub. Run:

    python tests/test_strategy_runner.py
    pytest tests/test_strategy_runner.py -v
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.gemini_agent import Decision
from core.config import TRADING_PAIRS
from core.strategy_runner import (
    Position,
    RunnerState,
    StrategyRunner,
    apply_breaker,
    count_open_positions,
    detect_signal,
    fraction_qty,
    last_closed_ts,
    maybe_exit,
    position_qty,
    update_trail,
)

PARAMS = json.loads(
    (PROJECT_ROOT / "strategies" / "params" / "ema_atr_trend.json").read_text(encoding="utf-8")
)["params"]

START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
NOW = pd.Timestamp("2026-01-01 05:30:00", tz="UTC")


class StubAgent:
    last_model_used = "stub-model"

    def __init__(
        self,
        action: str = "BUY",
        confidence: float = 0.9,
        stop: float | None = None,
        target: float | None = None,
        error: Exception | None = None,
    ) -> None:
        self.action = action
        self.confidence = confidence
        self.stop = stop
        self.target = target
        self.error = error
        self.calls = 0

    def decide(self, snapshot, strategy_context=""):
        self.calls += 1
        self.last_snapshot = snapshot
        if self.error is not None:
            raise self.error
        return Decision(
            action=self.action,
            confidence=self.confidence,
            rationale="stubbed decision",
            stop_loss=self.stop,
            take_profit=self.target,
        )


def _row(
    close: float,
    ema_fast: float,
    ema_slow: float,
    *,
    rsi: float = 55.0,
    adx: float = 25.0,
    atr: float = 100.0,
    ema_macro: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> dict:
    return {
        "open": close,
        "high": close + 10 if high is None else high,
        "low": close - 10 if low is None else low,
        "close": close,
        "volume": 1.0,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "ema_macro": close - 1000 if ema_macro is None else ema_macro,
        "rsi": rsi,
        "atr": atr,
        "adx": adx,
    }


def make_frame(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range(START, periods=len(rows), freq="h", tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    df.attrs.update(symbol="BTC/USDT", timeframe="1h", exchange="kraken")
    return df


def long_crossover_frame() -> pd.DataFrame:
    """Last closed bar (04:00) is a long EMA crossover; 05:00 is still forming."""
    return make_frame(
        [
            _row(10000, 99, 100),
            _row(10000, 99, 100),
            _row(10000, 99.5, 100),
            _row(10000, 99.9, 100),  # 03:00 still below
            _row(10000, 100.1, 100),  # 04:00 cross up = last closed
            _row(10020, 100.3, 100),  # 05:00 forming
        ]
    )


def short_crossover_frame() -> pd.DataFrame:
    return make_frame(
        [
            _row(10000, 101, 100, rsi=40, ema_macro=11000),
            _row(10000, 101, 100, rsi=40, ema_macro=11000),
            _row(10000, 100.5, 100, rsi=40, ema_macro=11000),
            _row(10000, 100.1, 100, rsi=40, ema_macro=11000),
            _row(10000, 99.9, 100, rsi=40, ema_macro=11000),
            _row(9980, 99.7, 100, rsi=40, ema_macro=11000),
        ]
    )


def runner_for(frame: pd.DataFrame, agent: StubAgent, tmp: Path, **kwargs) -> StrategyRunner:
    return StrategyRunner(
        agent=agent,
        fetch_fn=kwargs.pop("fetch_fn", lambda **_k: frame.copy()),
        indicate_fn=lambda df: df,
        params=PARAMS,
        strategy_context="test",
        symbol=kwargs.pop("symbol", "BTC/USDT"),
        timeframe="1h",
        exchange_id="kraken",
        min_confidence=0.6,
        starting_equity=10_000.0,
        trade_log=tmp / "paper_trades.csv",
        reject_log=tmp / "rejected_alerts.csv",
        state_path=tmp / "runner.json",
        pairs=kwargs.pop("pairs", ("BTC/USDT",)),
        fetch_delay=kwargs.pop("fetch_delay", 0),
        **kwargs,
    )


def tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="gemini-runner-"))


def test_last_closed_skips_forming_bar() -> None:
    frame = long_crossover_frame()
    ts = last_closed_ts(frame, "1h", now=NOW)
    assert ts == frame.index[4]
    print(f"    last closed = {ts}")


def test_long_signal_requires_crossover_rsi_and_regime() -> None:
    frame = long_crossover_frame()
    closed = frame.loc[: frame.index[4]]
    signal = detect_signal(closed, PARAMS, breaker_active=False)
    assert signal is not None
    assert signal.action == "BUY"
    assert signal.reason == "macro_aligned_trend_long"
    assert abs(signal.stop - (10000 - 1.5 * 100)) < 1e-9
    assert abs(signal.target - (10000 + 3.75 * 100)) < 1e-9

    blocked = detect_signal(closed, PARAMS, breaker_active=True)
    assert blocked is None

    weak = closed.copy()
    weak.iloc[-1, weak.columns.get_loc("adx")] = 12
    assert detect_signal(weak, PARAMS, False) is None

    bear = closed.copy()
    bear.iloc[-1, bear.columns.get_loc("ema_macro")] = 20000
    assert detect_signal(bear, PARAMS, False) is None

    tired = closed.copy()
    tired.iloc[-1, tired.columns.get_loc("rsi")] = 40
    assert detect_signal(tired, PARAMS, False) is None
    print("    long signal + four gates")


def test_short_signal() -> None:
    frame = short_crossover_frame()
    closed = frame.loc[: frame.index[4]]
    signal = detect_signal(closed, PARAMS, False)
    assert signal is not None and signal.action == "SELL"
    assert abs(signal.stop - (10000 + 1.5 * 100)) < 1e-9
    assert abs(signal.target - (10000 - 3.75 * 100)) < 1e-9
    print("    short signal")


def test_trail_arms_at_2_75_and_locks_0_67r() -> None:
    pos = Position(
        side="LONG",
        entry_price=10000,
        entry_atr=100,
        qty=1,
        stop=9850,
        target=10375,
        trail_stop=9850,
        trail_armed=False,
        opened_bar="x",
        confidence=0.9,
        model="stub",
        rationale="",
        reason="t",
    )
    update_trail(pos, close=10270, atr=100, params=PARAMS)  # +2.70 ATR, not yet 2.75
    assert pos.trail_armed is False
    assert pos.trail_stop == 9850

    update_trail(pos, close=10280, atr=100, params=PARAMS)  # +2.80 ATR
    assert pos.trail_armed is True
    # trail sits 1.75 ATR behind close: 10280 - 175 = 10105. Locked floor = +1.05 ATR = +0.70R? 
    # (2.80 - 1.75) = 1.05 ATR above entry. Formula for arm exactly at 2.75: (2.75-1.75)/1.5 = 0.667R
    assert pos.trail_stop == 10105
    print(f"    armed trail_stop={pos.trail_stop}")


def test_stop_beats_target_on_same_bar() -> None:
    pos = Position(
        side="LONG",
        entry_price=10000,
        entry_atr=100,
        qty=1,
        stop=9850,
        target=10375,
        trail_stop=9850,
        trail_armed=False,
        opened_bar="x",
        confidence=0.9,
        model="stub",
        rationale="",
        reason="t",
    )
    event = maybe_exit(pos, high=10400, low=9800, close=10000, atr=100, params=PARAMS)
    assert event is not None
    assert event.hit == "stop"
    assert event.price == 9850
    print("    same-bar ambiguity -> stop")


def test_circuit_breaker_trips_on_four_losses() -> None:
    state = RunnerState(equity=9000, equity_history=[10000, 9700, 9400, 9000], loss_streak=4)
    apply_breaker(state, PARAMS, trend_strong=True, new_bar=True)
    assert state.breaker_active is True
    assert state.breaker_bars == 0

    for _ in range(23):
        apply_breaker(state, PARAMS, trend_strong=True, new_bar=True)
    assert state.breaker_active is True
    apply_breaker(state, PARAMS, trend_strong=True, new_bar=True)
    assert state.breaker_active is False
    assert state.loss_streak == 0
    print("    breaker trips at 4 and re-arms after 24 strong bars")


def test_risk_sizing_is_capped() -> None:
    qty = position_qty(10_000, price=10000, atr=100, params=PARAMS)
    # 0.25% of 10k = 25 risk / (1.5*100) = 0.1667 BTC; 50% notional cap = 0.5 BTC
    assert abs(qty - 25 / 150) < 1e-9
    assert abs(fraction_qty(10_000, 10000, 0.33) - 0.33) < 1e-9
    print(f"    qty={qty:.4f}")


def test_confirmed_entry_is_logged(tmp: Path | None = None) -> None:
    tmp = tmp or tmpdir()
    agent = StubAgent("BUY", 0.88, stop=9850, target=10375)
    runner = runner_for(long_crossover_frame(), agent, tmp)
    report = runner.cycle(now=NOW)
    assert report.signal == "BUY"
    assert report.verdict == "CONFIRMED"
    assert report.position == "LONG"
    assert agent.calls == 1
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["action"] == "BUY"
    assert rows[0]["verdict"] == "CONFIRMED"
    assert float(rows[0]["stop_loss"]) == 9850
    assert float(rows[0]["take_profit"]) == 10375
    print(f"    logged BUY @ {rows[0]['entry_price']}")


def test_repeat_cycle_does_not_double_enter() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9, stop=9850, target=10375)
    runner = runner_for(long_crossover_frame(), agent, tmp)
    first = runner.cycle(now=NOW)
    second = runner.cycle(now=NOW)
    assert first.verdict == "CONFIRMED"
    assert second.position == "LONG"
    assert second.reason == "holding"
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    print("    second cycle skipped")


def test_reject_and_fail_closed() -> None:
    tmp = tmpdir()
    hold = StubAgent("HOLD", 0.95)
    runner = runner_for(long_crossover_frame(), hold, tmp)
    report = runner.cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert report.reason == "agent_returned_HOLD"
    assert report.position == "FLAT"
    assert (tmp / "rejected_alerts.csv").exists()
    assert not (tmp / "paper_trades.csv").exists()

    tmp2 = tmpdir()
    dead = StubAgent(error=RuntimeError("quota"))
    runner = runner_for(long_crossover_frame(), dead, tmp2)
    report = runner.cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert report.reason == "agent_unavailable"
    print("    HOLD and agent failure both fail closed")


def test_low_confidence_rejected() -> None:
    tmp = tmpdir()
    timid = StubAgent("BUY", 0.2, stop=9850, target=10375)
    report = runner_for(long_crossover_frame(), timid, tmp).cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert "confidence_0.20_below_0.60" in report.reason
    print("    low confidence rejected")


def test_open_position_closes_on_stop() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9, stop=9850, target=10375)
    runner = runner_for(long_crossover_frame(), agent, tmp)
    runner.cycle(now=NOW)
    assert runner.state.position is not None

    # Next hour: forming bar prints the stop. Advance time so 05:00 is closed and 06:00 forms.
    punched = long_crossover_frame()
    punched.loc[punched.index[5], ["low", "close", "high"]] = (9800, 9900, 10000)
    extra = punched.copy()
    extra.loc[pd.Timestamp("2026-01-01 06:00:00", tz="UTC")] = _row(9900, 100.4, 100)
    later = pd.Timestamp("2026-01-01 06:30:00", tz="UTC")
    runner.fetch_fn = lambda **_k: extra
    report = runner.cycle(now=later)
    assert report.verdict == "CONFIRMED"
    assert report.reason == "initial_stop_or_target_long"
    assert report.position == "FLAT"
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    assert rows[-1]["action"] == "CLOSE"
    print(f"    closed via stop, equity={runner.state.equity:.2f}")


def test_once_consults_gemini_without_signal() -> None:
    tmp = tmpdir()
    # No crossover: fast stays above slow.
    frame = make_frame(
        [
            _row(10000, 110, 100),
            _row(10000, 110, 100),
            _row(10000, 110, 100),
            _row(10000, 110, 100),
            _row(10000, 110, 100),
            _row(10020, 110, 100),
        ]
    )
    agent = StubAgent("HOLD", 0.4)
    report = runner_for(frame, agent, tmp).cycle(consult_always=True, now=NOW)
    assert report.signal is None
    assert agent.calls == 1
    assert report.gemini_action == "HOLD"
    assert report.reason == "no_entry_signal"
    assert report.position == "FLAT"
    print("    --once consulted Gemini on a flat bar")


def test_multi_pair_scan_caps_opens_and_sizes() -> None:
    tmp = tmpdir()
    fetched: list[str] = []
    frame = long_crossover_frame()

    def fetch(*, symbol: str, **_k):
        fetched.append(symbol)
        return frame.copy()

    agent = StubAgent("BUY", 0.9, stop=9850, target=10375)
    runner = runner_for(
        frame,
        agent,
        tmp,
        fetch_fn=fetch,
        pairs=TRADING_PAIRS,
        fetch_delay=0,
    )
    report = runner.cycle(now=NOW)
    assert fetched == list(TRADING_PAIRS)
    assert len(report.legs) == 3
    assert report.book["BTC/USDT"] == "LONG"
    assert report.book["ETH/USDT"] == "LONG"
    assert report.book["SOL/USDT"] == "FLAT"
    assert count_open_positions(runner.state.positions) == 2
    sol = next(leg for leg in report.legs if leg.symbol == "SOL/USDT")
    assert sol.reason == "max_open_positions"
    assert sol.signal == "BUY"
    btc_qty = runner.state.positions["BTC/USDT"]["qty"]
    assert abs(btc_qty - 0.33) < 1e-9
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert saved["positions"]["BTC/USDT"]["status"] == "LONG"
    assert saved["positions"]["ETH/USDT"]["status"] == "LONG"
    assert saved["positions"]["SOL/USDT"]["status"] == "FLAT"
    assert agent.calls == 2
    print("    scanned 3 pairs, opened 2, sized 33%")


def test_breaker_blocks_entry() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9, stop=9850, target=10375)
    runner = runner_for(long_crossover_frame(), agent, tmp)
    runner.state.loss_streak = 4
    runner.state.equity_history = [10000, 9000, 8500, 8000]
    report = runner.cycle(now=NOW)
    assert report.breaker_active is True
    assert report.signal is None
    assert report.reason == "circuit_breaker_active"
    assert agent.calls == 0
    print("    breaker suppressed the long")


def main() -> int:
    checks = [
        ("last closed skips forming bar", test_last_closed_skips_forming_bar),
        ("long signal + regime gates", test_long_signal_requires_crossover_rsi_and_regime),
        ("short signal", test_short_signal),
        ("trail arm 2.75 / 1.75", test_trail_arms_at_2_75_and_locks_0_67r),
        ("stop beats target same bar", test_stop_beats_target_on_same_bar),
        ("circuit breaker 4 / 24", test_circuit_breaker_trips_on_four_losses),
        ("risk sizing", test_risk_sizing_is_capped),
        ("confirmed entry logged", test_confirmed_entry_is_logged),
        ("no double entry", test_repeat_cycle_does_not_double_enter),
        ("reject and fail closed", test_reject_and_fail_closed),
        ("low confidence rejected", test_low_confidence_rejected),
        ("stop closes position", test_open_position_closes_on_stop),
        ("--once consults without signal", test_once_consults_gemini_without_signal),
        ("multi-pair scan + cap", test_multi_pair_scan_caps_opens_and_sizes),
        ("breaker blocks entry", test_breaker_blocks_entry),
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
    print("\n" + ("[PASS] strategy runner." if not failures else f"[FAIL] {failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
