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

from core.gemini_agent import Decision, system_instruction
from core.broker import PairLimits, limits_from_market, prepare_order_size, truncate_qty
from core.config import POSITION_SIZE_FRACTION, TRADING_PAIRS
from core.order_lifecycle import LiveTradingDisabled, classify_order
from core.market_data import classify_macro_regime
from core.strategy_runner import (
    Position,
    RunnerState,
    StrategyRunner,
    apply_breaker,
    correlation_block_reason,
    count_open_positions,
    detect_signal,
    detect_trigger,
    fraction_qty,
    last_closed_ts,
    maybe_exit,
    mtf_maybe_exit,
    position_qty,
    select_alt_entry_winner,
    update_trail,
)

PARAMS = json.loads(
    (PROJECT_ROOT / "strategies" / "params" / "ema_atr_trend.json").read_text(encoding="utf-8")
)["params"]

START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
NOW = pd.Timestamp("2026-01-01 05:30:00", tz="UTC")
TRIGGER_START = pd.Timestamp("2026-01-01 04:00:00", tz="UTC")


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


def make_frame(rows: list[dict], *, freq: str = "h", start: pd.Timestamp | None = None, timeframe: str = "1h") -> pd.DataFrame:
    idx = pd.date_range(start or START, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    df.attrs.update(symbol="BTC/USD", timeframe=timeframe, exchange="kraken")
    return df


def make_trigger_frame(rows: list[dict]) -> pd.DataFrame:
    return make_frame(rows, freq="15min", start=TRIGGER_START, timeframe="15m")


def macro_bull_frame(*, adx: float = 25.0) -> pd.DataFrame:
    bull = _row(10000, 100.5, 99.0, adx=adx, ema_macro=9000)
    return make_frame([bull] * 6)


def macro_bear_frame() -> pd.DataFrame:
    bear = _row(10000, 99.0, 100.5, adx=25.0, ema_macro=11000)
    return make_frame([bear] * 6)


def trigger_cross_frame() -> pd.DataFrame:
    """Last closed 15m (05:15) is an EMA 9/21 cross-up."""
    return make_trigger_frame(
        [
            _row(10000, 99, 100, atr=100),
            _row(10000, 99, 100, atr=100),
            _row(10000, 99.5, 100, atr=100),
            _row(10000, 99.9, 100, atr=100),
            _row(10000, 99.95, 100, atr=100),
            _row(10000, 100.1, 100, atr=100),
            _row(10020, 100.3, 100, atr=100),
        ]
    )


def trigger_flat_frame() -> pd.DataFrame:
    return make_trigger_frame([_row(10000, 110, 100, atr=100)] * 7)


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


def runner_for(
    macro: pd.DataFrame,
    trigger: pd.DataFrame,
    agent: StubAgent,
    tmp: Path,
    **kwargs,
) -> StrategyRunner:
    def fetch(*, timeframe: str, **_k):
        return (macro if timeframe == "1h" else trigger).copy()

    return StrategyRunner(
        agent=agent,
        fetch_fn=kwargs.pop("fetch_fn", fetch),
        indicate_fn=lambda df: df,
        params=PARAMS,
        strategy_context="test",
        symbol=kwargs.pop("symbol", "BTC/USD"),
        timeframe="15m",
        exchange_id="kraken",
        min_confidence=0.6,
        starting_equity=10_000.0,
        trade_log=kwargs.pop("trade_log", tmp / "paper_trades.csv"),
        reject_log=tmp / "rejected_alerts.csv",
        state_path=tmp / "runner.json",
        pairs=kwargs.pop("pairs", ("BTC/USD",)),
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


def test_macro_regime_and_trigger() -> None:
    bull = classify_macro_regime(macro_bull_frame().iloc[-2])
    bear = classify_macro_regime(macro_bear_frame().iloc[-2])
    chop_row = _row(10000, 100.5, 99.0, adx=12.0, ema_macro=9000)
    assert bull == "BULL"
    assert bear == "BEAR"
    assert classify_macro_regime(pd.Series(chop_row)) == "NEUTRAL"

    trigger = trigger_cross_frame()
    closed = trigger.loc[: trigger.index[5]]
    signal = detect_trigger(closed)
    assert signal is not None and signal.action == "BUY"
    assert abs(signal.stop - 9850) < 1e-9
    assert abs(signal.target - 10350) < 1e-9
    assert detect_trigger(trigger_flat_frame().loc[: trigger_flat_frame().index[5]]) is None
    print("    BULL/BEAR/CHOP + 15m cross")


def test_mtf_stop_and_regime_exit() -> None:
    pos = Position(
        side="LONG",
        entry_price=10000,
        entry_atr=100,
        qty=1,
        stop=9850,
        target=10350,
        trail_stop=9850,
        trail_armed=False,
        opened_bar="x",
        confidence=0.9,
        model="stub",
        rationale="",
        reason="t",
    )
    stop = mtf_maybe_exit(pos, high=10400, low=9800, close=10000, regime="BULL")
    assert stop is not None and stop.hit == "stop"
    win = mtf_maybe_exit(pos, high=10400, low=9900, close=10100, regime="BULL")
    assert win is not None and win.hit == "target"
    flip = mtf_maybe_exit(pos, high=10100, low=9950, close=10000, regime="BEAR")
    assert flip is not None and flip.hit == "regime"
    print("    MTF stop / target / regime exit")


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
    agent = StubAgent("BUY", 0.88)
    runner = runner_for(macro_bull_frame(), trigger_cross_frame(), agent, tmp)
    report = runner.cycle(now=NOW)
    assert report.signal == "BUY"
    assert report.verdict == "CONFIRMED"
    assert report.position == "LONG"
    assert report.macro_regime == "BULL"
    assert agent.calls == 1
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["action"] == "BUY"
    assert rows[0]["verdict"] == "CONFIRMED"
    assert float(rows[0]["stop_loss"]) == 9850
    assert float(rows[0]["take_profit"]) == 10350
    print(f"    logged BUY @ {rows[0]['entry_price']}")


def test_repeat_cycle_does_not_double_enter() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    runner = runner_for(macro_bull_frame(), trigger_cross_frame(), agent, tmp)
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
    runner = runner_for(macro_bull_frame(), trigger_cross_frame(), hold, tmp)
    report = runner.cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert report.reason == "agent_returned_HOLD"
    assert report.position == "FLAT"
    assert (tmp / "rejected_alerts.csv").exists()
    assert not (tmp / "paper_trades.csv").exists()

    tmp2 = tmpdir()
    dead = StubAgent(error=RuntimeError("quota"))
    runner = runner_for(macro_bull_frame(), trigger_cross_frame(), dead, tmp2)
    report = runner.cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert report.reason == "agent_unavailable"
    print("    HOLD and agent failure both fail closed")


def test_low_confidence_rejected() -> None:
    tmp = tmpdir()
    timid = StubAgent("BUY", 0.2)
    report = runner_for(macro_bull_frame(), trigger_cross_frame(), timid, tmp).cycle(now=NOW)
    assert report.verdict == "REJECTED"
    assert "confidence_0.20_below_0.60" in report.reason
    print("    low confidence rejected")


def test_open_position_closes_on_stop() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    runner = runner_for(macro_bull_frame(), trigger_cross_frame(), agent, tmp)
    runner.cycle(now=NOW)
    assert runner.state.position is not None

    punched = trigger_cross_frame()
    punched.loc[punched.index[6], ["low", "close", "high"]] = (9800, 9900, 10000)
    extra = punched.copy()
    extra.loc[pd.Timestamp("2026-01-01 05:45:00", tz="UTC")] = _row(9900, 100.4, 100, atr=100)
    later = pd.Timestamp("2026-01-01 05:52:00", tz="UTC")
    runner.fetch_fn = lambda *, timeframe, **_k: (
        macro_bull_frame() if timeframe == "1h" else extra
    )
    report = runner.cycle(now=later)
    assert report.verdict == "CONFIRMED"
    assert report.reason == "atr_stop_long"
    assert report.position == "FLAT"
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    assert rows[-1]["action"] == "CLOSE"
    assert float(rows[-1]["entry_price"]) == 10000
    assert float(rows[-1]["exit_price"]) != float(rows[-1]["entry_price"])
    assert float(rows[-1]["stop_loss"]) != float(rows[-1]["entry_price"])
    assert rows[-1]["hit"] == "stop"
    assert float(rows[-1]["pnl"]) < 0
    assert runner.state.equity < 10_000
    print(f"    closed via stop, equity={runner.state.equity:.2f}")


def test_regime_flip_exits_long() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    runner = runner_for(macro_bull_frame(), trigger_cross_frame(), agent, tmp)
    runner.cycle(now=NOW)
    later = pd.Timestamp("2026-01-01 05:52:00", tz="UTC")
    trigger = trigger_cross_frame()
    trigger.loc[pd.Timestamp("2026-01-01 05:45:00", tz="UTC")] = _row(10010, 100.4, 100, atr=100)
    runner.fetch_fn = lambda *, timeframe, **_k: (
        macro_bear_frame() if timeframe == "1h" else trigger
    )
    report = runner.cycle(now=later)
    assert report.reason == "macro_regime_exit"
    assert report.position == "FLAT"
    print("    BEAR flip closed the long")


def test_once_consults_gemini_without_signal() -> None:
    tmp = tmpdir()
    agent = StubAgent("HOLD", 0.4)
    report = runner_for(macro_bull_frame(), trigger_flat_frame(), agent, tmp).cycle(
        consult_always=True, now=NOW
    )
    assert report.signal is None
    assert agent.calls == 1
    assert report.gemini_action == "HOLD"
    assert report.reason == "no_entry_signal"
    assert report.position == "FLAT"
    print("    --once consulted Gemini on a flat bar")


def test_multi_pair_scan_caps_opens_and_sizes() -> None:
    tmp = tmpdir()
    fetched: list[tuple[str, str]] = []
    # Equal ADX: alt winner is SOL (tie-break by symbol). BTC fills first.
    macros = {
        "BTC/USD": macro_bull_frame(adx=25),
        "ETH/USD": macro_bull_frame(adx=25),
        "SOL/USD": macro_bull_frame(adx=25),
    }
    trigger = trigger_cross_frame()

    def fetch(*, symbol: str, timeframe: str, **_k):
        fetched.append((symbol, timeframe))
        return (macros[symbol] if timeframe == "1h" else trigger).copy()

    agent = StubAgent("BUY", 0.9)
    runner = runner_for(
        macros["BTC/USD"],
        trigger,
        agent,
        tmp,
        fetch_fn=fetch,
        pairs=TRADING_PAIRS,
        fetch_delay=0,
    )
    report = runner.cycle(now=NOW)
    assert fetched == [
        (symbol, tf) for symbol in TRADING_PAIRS for tf in ("1h", "15m")
    ]
    assert len(report.legs) == 3
    assert report.book["BTC/USD"] == "LONG"
    assert report.book["SOL/USD"] == "LONG"
    assert report.book["ETH/USD"] == "FLAT"
    assert count_open_positions(runner.state.positions) == 2
    eth = next(leg for leg in report.legs if leg.symbol == "ETH/USD")
    assert eth.reason == "alt_adx_priority"
    assert eth.signal == "BUY"
    btc_qty = runner.state.positions["BTC/USD"]["qty"]
    assert abs(btc_qty - POSITION_SIZE_FRACTION) < 1e-9
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert saved["circuit_breaker"]["tripped"] is False
    assert saved["positions"]["BTC/USD"]["status"] == "LONG"
    assert saved["positions"]["BTC/USD"]["size"] == btc_qty
    assert saved["positions"]["SOL/USD"]["status"] == "LONG"
    assert saved["positions"]["ETH/USD"]["status"] == "FLAT"
    assert agent.calls == 2
    print("    scanned 3 pairs x 2 timeframes, opened BTC+SOL, sized 25%")


def test_alt_correlation_blocks_second_alt() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    macros = {
        "BTC/USD": macro_bull_frame(adx=20),
        "ETH/USD": macro_bull_frame(adx=30),
        "SOL/USD": macro_bull_frame(adx=40),
    }
    trigger = trigger_cross_frame()

    def fetch(*, symbol: str, timeframe: str, **_k):
        return (macros[symbol] if timeframe == "1h" else trigger).copy()

    # Seed an open ETH slot with room left under MAX_OPEN_POSITIONS.
    runner = runner_for(
        macros["BTC/USD"],
        trigger,
        agent,
        tmp,
        fetch_fn=fetch,
        pairs=TRADING_PAIRS,
        fetch_delay=0,
    )
    runner.state.positions["ETH/USD"] = {
        "status": "LONG",
        "side": "LONG",
        "entry_price": 10000,
        "size": 0.1,
        "qty": 0.1,
        "stop_loss": 9850,
        "take_profit": 10350,
        "stop": 9850,
        "target": 10350,
        "entry_time": "seed",
        "opened_bar": "seed",
        "macro_regime": "BULL",
    }
    report = runner.cycle(now=NOW)
    assert report.book["ETH/USD"] == "LONG"
    assert report.book["BTC/USD"] == "LONG"
    assert report.book["SOL/USD"] == "FLAT"
    sol = next(leg for leg in report.legs if leg.symbol == "SOL/USD")
    assert sol.reason == "alt_correlation_cap"
    assert correlation_block_reason("SOL/USD", runner.state.positions) == "alt_correlation_cap"
    print("    ETH open blocked SOL; BTC still filled")


def test_same_bar_alts_prefer_higher_adx() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    # Flat BTC frame so only alts compete for a single open slot.
    flat_btc = make_frame([_row(10000, 110, 100, adx=18, ema_macro=9000)] * 6)
    macros = {
        "BTC/USD": flat_btc,
        "ETH/USD": macro_bull_frame(adx=22),
        "SOL/USD": macro_bull_frame(adx=35),
    }
    trigger = trigger_cross_frame()
    flat_trigger = trigger_flat_frame()

    def fetch(*, symbol: str, timeframe: str, **_k):
        if timeframe == "1h":
            return macros[symbol].copy()
        if symbol == "BTC/USD":
            return flat_trigger.copy()
        return trigger.copy()

    runner = runner_for(
        macros["BTC/USD"],
        trigger,
        agent,
        tmp,
        fetch_fn=fetch,
        pairs=TRADING_PAIRS,
        fetch_delay=0,
    )
    report = runner.cycle(now=NOW)
    assert report.book["SOL/USD"] == "LONG"
    assert report.book["ETH/USD"] == "FLAT"
    assert report.book["BTC/USD"] == "FLAT"
    eth = next(leg for leg in report.legs if leg.symbol == "ETH/USD")
    assert eth.reason == "alt_adx_priority"
    winner = select_alt_entry_winner(
        [
            {"symbol": "ETH/USD", "signal": type("S", (), {"adx": 22})()},
            {"symbol": "SOL/USD", "signal": type("S", (), {"adx": 35})()},
        ]
    )
    assert winner is not None and winner["symbol"] == "SOL/USD"
    print("    same-bar alts: SOL (ADX 35) beat ETH (ADX 22)")


def test_breaker_blocks_entry() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    runner = runner_for(macro_bull_frame(), trigger_cross_frame(), agent, tmp)
    runner.state.loss_streak = 4
    runner.state.equity_history = [10000, 9000, 8500, 8000]
    report = runner.cycle(now=NOW)
    assert report.breaker_active is True
    assert report.signal is None
    assert report.reason == "circuit_breaker_active"
    assert agent.calls == 0
    print("    breaker suppressed the long")


def test_order_size_truncates_and_clamps() -> None:
    assert truncate_qty(0.123456789, 5) == 0.12345
    assert truncate_qty(1.999, 0) == 1.0
    limits = PairLimits("BTC/USD", ordermin=0.0001, lot_decimals=8, pair_decimals=1)
    assert prepare_order_size(0.00012, limits) == 0.00012
    assert prepare_order_size(0.00001, limits) is None
    market = {
        "info": {"ordermin": "0.002", "lot_decimals": "5", "pair_decimals": "2"},
        "limits": {"amount": {"min": 0.01}},
        "precision": {"amount": 8, "price": 1},
    }
    parsed = limits_from_market("ETH/USD", market)
    assert parsed.ordermin == 0.002
    assert parsed.lot_decimals == 5
    assert parsed.pair_decimals == 2
    print("    truncate + AssetPairs parse")


def test_below_ordermin_skips_entry() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    limits = {"BTC/USD": PairLimits("BTC/USD", ordermin=1.0, lot_decimals=8, pair_decimals=1)}
    report = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        agent,
        tmp,
        pair_limits=limits,
    ).cycle(now=NOW)
    assert report.position == "FLAT"
    assert report.reason == "below_ordermin"
    assert agent.calls == 0
    print("    below ordermin skipped")


def _ack(status: str, qty: float, price: float, order_id: str = "ord-1", filled: float | None = None) -> dict:
    if filled is None:
        filled = qty if status in {"closed", "filled"} else 0.0
    return {
        "id": order_id,
        "status": status,
        "filled": filled,
        "amount": qty,
        "average": price if filled else None,
        "price": price,
    }


def test_order_classification() -> None:
    assert classify_order({"id": "1", "status": "open", "filled": 0, "amount": 1}) == "pending"
    assert classify_order({"id": "1", "status": "closed", "filled": 1, "amount": 1}) == "filled"
    assert classify_order({"id": "1", "status": "canceled", "filled": 0, "amount": 1}) == "canceled"
    assert classify_order({"id": "1", "status": "expired", "filled": 0, "amount": 1}) == "canceled"
    assert classify_order(None) == "canceled"
    print("    ack classified pending / filled / canceled")


def test_gemini_prompt_matches_mode() -> None:
    paper = system_instruction(paper_trading=True)
    live = system_instruction(paper_trading=False)
    assert "Operating mode: paper trading" in paper
    assert "Operating mode: live trading" in live
    assert "Operating mode: paper trading" not in live
    assert "enforced in code" in paper and "enforced in code" in live
    assert "cannot" in paper.lower()
    print("    prompt follows paper vs live")


def test_live_without_allow_fails_closed() -> None:
    tmp = tmpdir()
    called = {"balance": 0, "orders": 0}

    def balance() -> float:
        called["balance"] += 1
        return 200.0

    def order_fn(*_args):
        called["orders"] += 1

    try:
        runner_for(
            macro_bull_frame(),
            trigger_cross_frame(),
            StubAgent("BUY", 0.9),
            tmp,
            paper_trading=False,
            allow_live_trading=False,
            balance_fn=balance,
            order_fn=order_fn,
        )
    except LiveTradingDisabled as exc:
        assert "ALLOW_LIVE_TRADING" in str(exc)
        assert called == {"balance": 0, "orders": 0}
        print("    paper off without allow-live placed nothing")
        return
    raise AssertionError("expected LiveTradingDisabled")


def test_allow_live_does_not_override_paper() -> None:
    tmp = tmpdir()
    orders: list[tuple] = []
    report = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        StubAgent("BUY", 0.9),
        tmp,
        paper_trading=True,
        allow_live_trading=True,
        order_fn=lambda *args: orders.append(args),
    ).cycle(now=NOW)
    assert orders == []
    assert report.position == "LONG"
    print("    paper mode ignored ALLOW_LIVE_TRADING")


def test_live_syncs_balance_and_posts_limit() -> None:
    tmp = tmpdir()
    orders: list[tuple] = []
    agent = StubAgent("BUY", 0.9)

    def order_fn(*args):
        orders.append(args)
        return _ack("open", args[3], args[4], order_id="pending-1", filled=0)

    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        agent,
        tmp,
        paper_trading=False,
        allow_live_trading=True,
        use_post_only=True,
        balance_fn=lambda: 200.0,
        order_fn=order_fn,
        pair_limits={
            "BTC/USD": PairLimits("BTC/USD", ordermin=0.0001, lot_decimals=8, pair_decimals=1)
        },
    )
    assert runner.state.equity == 200.0
    assert runner.state.start_equity == 200.0
    report = runner.cycle(now=NOW)
    assert report.position == "FLAT"
    assert report.reason == "entry_pending_fill"
    assert report.pending.get("BTC/USD") == "pending-1"
    assert runner.state.positions["BTC/USD"]["status"] == "FLAT"
    assert runner.state.pending_orders["BTC/USD"]["order_id"] == "pending-1"
    assert orders and orders[0][0] == "limit"
    assert orders[0][1] == "buy"
    assert orders[0][5] == {"postOnly": True}
    assert not (tmp / "paper_trades.csv").exists()
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert saved["pending_orders"]["BTC/USD"]["order_id"] == "pending-1"
    assert saved["positions"]["BTC/USD"]["status"] == "FLAT"
    print("    live ack without fill stays pending")


def test_live_entry_opens_only_after_fill() -> None:
    tmp = tmpdir()
    orders: list[tuple] = []
    placed: dict = {}
    phase = {"fetch": "open"}

    def order_fn(*args):
        orders.append(args)
        placed["qty"] = args[3]
        placed["price"] = args[4]
        return _ack("open", args[3], args[4], order_id="rest-1", filled=0)

    def fetch(_symbol, order_id):
        if phase["fetch"] == "open":
            return _ack("open", placed["qty"], placed["price"], order_id, filled=0)
        return _ack("closed", placed["qty"], placed["price"], order_id)

    agent = StubAgent("BUY", 0.9)
    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        agent,
        tmp,
        paper_trading=False,
        allow_live_trading=True,
        use_post_only=True,
        balance_fn=lambda: 200.0,
        order_fn=order_fn,
        fetch_order_fn=fetch,
        pair_limits={
            "BTC/USD": PairLimits("BTC/USD", ordermin=0.0001, lot_decimals=8, pair_decimals=1)
        },
    )
    first = runner.cycle(now=NOW)
    assert first.position == "FLAT"
    assert "BTC/USD" in runner.state.pending_orders
    phase["fetch"] = "closed"
    second = runner.cycle(now=NOW)
    assert second.position == "LONG"
    assert runner.state.pending_orders == {}
    expected = truncate_qty(200.0 * POSITION_SIZE_FRACTION / 10000.0, 8)
    assert abs(runner.state.positions["BTC/USD"]["size"] - expected) < 1e-12
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["action"] == "BUY"
    assert len(orders) == 1
    print("    position opened only after the resting limit filled")


def test_live_entry_canceled_clears_pending() -> None:
    tmp = tmpdir()
    orders: list[tuple] = []
    placed: dict = {}

    def order_fn(*args):
        orders.append(args)
        placed["qty"] = args[3]
        placed["price"] = args[4]
        return _ack("canceled", args[3], args[4], order_id="dead-1", filled=0)

    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        StubAgent("BUY", 0.9),
        tmp,
        paper_trading=False,
        allow_live_trading=True,
        balance_fn=lambda: 200.0,
        order_fn=order_fn,
        fetch_order_fn=lambda _symbol, order_id: _ack(
            "canceled", placed.get("qty", 0.0), placed.get("price", 0.0), order_id, filled=0
        ),
        pair_limits={
            "BTC/USD": PairLimits("BTC/USD", ordermin=0.0001, lot_decimals=8, pair_decimals=1)
        },
    )
    report = runner.cycle(now=NOW)
    assert report.position == "FLAT"
    assert report.reason == "entry_not_filled"
    assert runner.state.pending_orders == {}
    assert not (tmp / "paper_trades.csv").exists()
    assert len(orders) == 1
    print("    canceled ack never opened a slot")


def test_live_resting_cancel_on_next_poll() -> None:
    tmp = tmpdir()
    orders: list[tuple] = []
    placed: dict = {}

    def order_fn(*args):
        orders.append(args)
        placed["qty"] = args[3]
        placed["price"] = args[4]
        return _ack("open", args[3], args[4], order_id="rest-2", filled=0)

    def fetch(_symbol, order_id):
        return _ack("expired", placed["qty"], placed["price"], order_id, filled=0)

    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        StubAgent("BUY", 0.9),
        tmp,
        paper_trading=False,
        allow_live_trading=True,
        balance_fn=lambda: 200.0,
        order_fn=order_fn,
        fetch_order_fn=fetch,
        pair_limits={
            "BTC/USD": PairLimits("BTC/USD", ordermin=0.0001, lot_decimals=8, pair_decimals=1)
        },
    )
    first = runner.cycle(now=NOW)
    assert first.position == "FLAT"
    assert "rest-2" == runner.state.pending_orders["BTC/USD"]["order_id"]
    second = runner.cycle(now=NOW)
    assert second.position == "FLAT"
    assert second.reason == "entry_canceled"
    assert runner.state.pending_orders == {}
    assert len(orders) == 1
    assert not (tmp / "paper_trades.csv").exists()
    print("    expired resting order cleared without opening")


def test_live_anchors_start_equity_and_trade_log_name() -> None:
    from core.dashboard import _trade_log_name

    assert _trade_log_name(paper=True) == "paper_trades.csv"
    assert _trade_log_name(paper=False) == "live_trades.csv"

    tmp = tmpdir()
    agent = StubAgent("HOLD", 0.0)
    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        agent,
        tmp,
        paper_trading=False,
        allow_live_trading=True,
        balance_fn=lambda: 500.0,
        order_fn=lambda *args: None,
        trade_log=tmp / "live_trades.csv",
        pair_limits={"BTC/USD": PairLimits("BTC/USD", 0.0, 8, 1)},
    )
    assert runner.trade_log.name == "live_trades.csv"
    assert runner.state.equity == 500.0
    assert runner.state.start_equity == 500.0
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert saved["start_equity"] == 500.0
    assert saved["circuit_breaker"]["cooldown_bars"] == 0
    print("    live start_equity + live_trades.csv")


def test_paper_does_not_adopt_live_deposit() -> None:
    tmp = tmpdir()
    (tmp / "runner.json").write_text(
        json.dumps(
            {
                "equity": 500.0,
                "start_equity": 500.0,
                "book": "live",
                "circuit_breaker": {"loss_streak": 0, "tripped": False, "cooldown_bars": 0},
                "positions": {
                    "BTC/USD": {
                        "status": "LONG",
                        "entry_price": 77000,
                        "size": 0.01,
                        "stop_loss": 76000,
                        "take_profit": 80000,
                        "entry_time": None,
                        "macro_regime": "BULL",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        StubAgent("HOLD", 0.0),
        tmp,
        paper_trading=True,
        allow_live_trading=False,
    )
    assert runner.paper_trading is True
    assert runner.state.equity == 10_000.0
    assert runner.state.start_equity == 10_000.0
    assert runner.state.positions["BTC/USD"]["status"] == "FLAT"
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert saved["book"] == "paper"
    assert saved["equity"] == 10_000.0
    archived = json.loads((tmp / "live.json").read_text(encoding="utf-8"))
    assert archived["book"] == "live"
    assert archived["equity"] == 500.0
    assert archived["positions"]["BTC/USD"]["status"] == "LONG"
    print("    paper book stayed at 10000; 500 archived as live")


def test_paper_does_not_place_exchange_orders() -> None:
    tmp = tmpdir()
    orders: list[tuple] = []
    agent = StubAgent("BUY", 0.9)
    runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        agent,
        tmp,
        paper_trading=True,
        order_fn=lambda *args: orders.append(args),
    ).cycle(now=NOW)
    assert orders == []
    print("    paper path placed no exchange order")


def test_restart_resumes_open_position_without_duplicate_entry() -> None:
    """Simulate daemon crash mid-hold: reload state and continue managing stops."""
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    first = runner_for(macro_bull_frame(), trigger_cross_frame(), agent, tmp)
    opened = first.cycle(now=NOW)
    assert opened.position == "LONG"
    assert first.state.positions["BTC/USD"]["status"] == "LONG"
    saved = json.loads((tmp / "runner.json").read_text(encoding="utf-8"))
    assert saved["positions"]["BTC/USD"]["status"] == "LONG"
    assert saved["positions"]["BTC/USD"]["stop_loss"] == 9850
    assert saved["positions"]["BTC/USD"]["take_profit"] == 10350
    assert "BTC/USD" in saved.get("last_bars", {})
    equity_before = saved["equity"]

    agent2 = StubAgent("BUY", 0.9)
    restarted = runner_for(macro_bull_frame(), trigger_cross_frame(), agent2, tmp)
    assert restarted.state.positions["BTC/USD"]["status"] == "LONG"
    assert abs(restarted.state.equity - equity_before) < 1e-9
    assert restarted.state.last_bars.get("BTC/USD") == saved["last_bars"]["BTC/USD"]

    # Same closed bar: must hold, not fire a second BUY.
    again = restarted.cycle(now=NOW)
    assert again.position == "LONG"
    assert again.reason == "holding"
    assert agent2.calls == 0
    rows = list(csv.DictReader((tmp / "paper_trades.csv").open(encoding="utf-8")))
    assert len([r for r in rows if r["action"] == "BUY"]) == 1

    # Next bar hits stop: recovery path still manages the open slot.
    punched = trigger_cross_frame()
    punched.loc[punched.index[6], ["low", "close", "high"]] = (9800, 9900, 10000)
    extra = punched.copy()
    extra.loc[pd.Timestamp("2026-01-01 05:45:00", tz="UTC")] = _row(9900, 100.4, 100, atr=100)
    later = pd.Timestamp("2026-01-01 05:52:00", tz="UTC")
    restarted.fetch_fn = lambda *, timeframe, **_k: (
        macro_bull_frame() if timeframe == "1h" else extra
    )
    closed = restarted.cycle(now=later)
    assert closed.reason == "atr_stop_long"
    assert closed.position == "FLAT"
    print("    restart held open slot, no duplicate entry, stop still worked")


def test_live_missing_keys_fails_closed_on_startup() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)

    def boom():
        raise RuntimeError("EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required for live trading")

    try:
        runner_for(
            macro_bull_frame(),
            trigger_cross_frame(),
            agent,
            tmp,
            paper_trading=False,
            allow_live_trading=True,
            balance_fn=boom,
        )
    except RuntimeError as exc:
        assert "EXCHANGE_API_KEY" in str(exc) or "required" in str(exc).lower()
        print("    live startup failed closed before orders")
        return
    raise AssertionError("expected live startup to raise when balance sync fails")


def test_live_market_exit_on_stop() -> None:
    tmp = tmpdir()
    orders: list[tuple] = []
    agent = StubAgent("BUY", 0.9)

    def order_fn(*args):
        orders.append(args)
        if args[0] == "limit":
            return _ack("closed", args[3], args[4], order_id="entry-1")
        return {"id": "exit-1", "status": "closed", "filled": args[3], "amount": args[3]}

    runner = runner_for(
        macro_bull_frame(),
        trigger_cross_frame(),
        agent,
        tmp,
        paper_trading=False,
        allow_live_trading=True,
        use_post_only=True,
        balance_fn=lambda: 500.0,
        order_fn=order_fn,
        pair_limits={
            "BTC/USD": PairLimits("BTC/USD", ordermin=0.0001, lot_decimals=8, pair_decimals=1)
        },
    )
    runner.cycle(now=NOW)
    assert orders and orders[0][0] == "limit" and orders[0][5].get("postOnly") is True

    punched = trigger_cross_frame()
    punched.loc[punched.index[6], ["low", "close", "high"]] = (9800, 9900, 10000)
    extra = punched.copy()
    extra.loc[pd.Timestamp("2026-01-01 05:45:00", tz="UTC")] = _row(9900, 100.4, 100, atr=100)
    later = pd.Timestamp("2026-01-01 05:52:00", tz="UTC")
    runner.fetch_fn = lambda *, timeframe, **_k: (
        macro_bull_frame() if timeframe == "1h" else extra
    )
    report = runner.cycle(now=later)
    assert report.reason == "atr_stop_long"
    assert any(o[0] == "market" and o[1] == "sell" for o in orders)
    print("    live stop routed as market sell")


def test_bear_macro_blocks_trigger() -> None:
    tmp = tmpdir()
    agent = StubAgent("BUY", 0.9)
    report = runner_for(macro_bear_frame(), trigger_cross_frame(), agent, tmp).cycle(now=NOW)
    assert report.position == "FLAT"
    assert report.reason == "macro_bear"
    assert agent.calls == 0
    print("    BEAR 1h blocked the 15m cross")


def main() -> int:
    checks = [
        ("last closed skips forming bar", test_last_closed_skips_forming_bar),
        ("long signal + regime gates", test_long_signal_requires_crossover_rsi_and_regime),
        ("MTF regime + trigger", test_macro_regime_and_trigger),
        ("MTF exits", test_mtf_stop_and_regime_exit),
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
        ("regime flip exits long", test_regime_flip_exits_long),
        ("--once consults without signal", test_once_consults_gemini_without_signal),
        ("multi-pair scan + cap", test_multi_pair_scan_caps_opens_and_sizes),
        ("alt correlation cap", test_alt_correlation_blocks_second_alt),
        ("same-bar alt ADX priority", test_same_bar_alts_prefer_higher_adx),
        ("breaker blocks entry", test_breaker_blocks_entry),
        ("bear macro blocks trigger", test_bear_macro_blocks_trigger),
        ("order size clamp", test_order_size_truncates_and_clamps),
        ("below ordermin skip", test_below_ordermin_skips_entry),
        ("order classification", test_order_classification),
        ("gemini prompt matches mode", test_gemini_prompt_matches_mode),
        ("live gate fail closed", test_live_without_allow_fails_closed),
        ("paper wins over allow-live", test_allow_live_does_not_override_paper),
        ("live ack stays pending", test_live_syncs_balance_and_posts_limit),
        ("live opens after fill", test_live_entry_opens_only_after_fill),
        ("live canceled ack", test_live_entry_canceled_clears_pending),
        ("live expired on next poll", test_live_resting_cancel_on_next_poll),
        ("live start_equity + trade log", test_live_anchors_start_equity_and_trade_log_name),
        ("paper ignores live deposit", test_paper_does_not_adopt_live_deposit),
        ("paper places no order", test_paper_does_not_place_exchange_orders),
        ("restart recovery", test_restart_resumes_open_position_without_duplicate_entry),
        ("live missing keys fail closed", test_live_missing_keys_fails_closed_on_startup),
        ("live market exit on stop", test_live_market_exit_on_stop),
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
