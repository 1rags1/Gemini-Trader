"""Live end-to-end check: exchange candles -> indicators -> Gemini decision.

Hits the network on both sides (ccxt public endpoints and the Gemini API), so
run it when you want to confirm the whole loop, not on every save.

Run standalone:   python tests/test_pipeline.py
Run under pytest: pytest tests/test_pipeline.py -s
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.gemini_agent import GeminiAgent
from core.market_data import add_indicators, fetch_ohlcv, latest_snapshot
from strategies import load_strategy, list_strategies

#: Must comfortably exceed the longest indicator lookback (EMA 200), or the
#: macro-trend gate resolves to "unknown" and the agent can only HOLD.
CANDLES = 400


def build_snapshot() -> dict:
    df = add_indicators(fetch_ohlcv(limit=CANDLES))
    return latest_snapshot(df, lookback=3)


def test_market_data_has_indicators() -> None:
    snapshot = build_snapshot()
    assert snapshot["last_close"] > 0
    for indicator in ["ema_fast", "ema_slow", "ema_macro", "rsi", "atr", "adx"]:
        assert snapshot["indicators"].get(indicator) is not None, f"{indicator} is null"


def test_regime_block_is_populated() -> None:
    """The macro/ADX gate must resolve, or every decision degrades to HOLD."""
    regime = build_snapshot()["regime"]

    assert regime["macro_trend"] in {"bull", "bear"}, regime
    assert regime["trend_strength"] in {"strong", "weak"}, regime
    assert regime["tradeable_direction"] in {"long_only", "short_only", "none"}, regime
    assert regime["adx"] is not None and regime["macro_ema"] is not None

    # The gate must be internally consistent with the raw values it reports.
    if regime["adx"] <= regime["adx_min"]:
        assert regime["tradeable_direction"] == "none"
    else:
        expected = "long_only" if regime["macro_trend"] == "bull" else "short_only"
        assert regime["tradeable_direction"] == expected


def test_decision_respects_regime_gate() -> None:
    """Gemini must not trade against the direction the regime permits."""
    snapshot = build_snapshot()
    strategy = load_strategy(list_strategies()[0])
    decision = GeminiAgent().decide(snapshot, strategy.prompt_context())

    forbidden = {
        "long_only": "SELL",
        "short_only": "BUY",
        "none": None,  # any entry is forbidden
    }[snapshot["regime"]["tradeable_direction"]]

    if forbidden is None:
        assert decision.action == "HOLD", (
            f"ADX {snapshot['regime']['adx']} is below "
            f"{snapshot['regime']['adx_min']} but agent returned {decision.action}"
        )
    else:
        assert decision.action != forbidden, (
            f"regime is {snapshot['regime']['tradeable_direction']} but agent "
            f"returned {decision.action}"
        )
    print(f"\n{decision}")


def test_agent_returns_valid_decision() -> None:
    strategy = load_strategy(list_strategies()[0])
    decision = GeminiAgent().decide(build_snapshot(), strategy.prompt_context())

    assert decision.action in {"BUY", "SELL", "HOLD"}
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.rationale
    print(f"\n{decision}")


def main() -> int:
    snapshot = build_snapshot()
    regime = snapshot["regime"]

    print(f"[ OK ] {snapshot['symbol']} @ {snapshot['timeframe']} close={snapshot['last_close']}")
    print(f"       as of {snapshot['as_of']} ({CANDLES} candles)")
    print(f"       indicators: {snapshot['indicators']}")
    print(f"[ OK ] regime: macro={regime['macro_trend']} "
          f"(price {regime['price_vs_macro_ema_pct']:+.2f}% vs EMA200 {regime['macro_ema']})")
    print(f"       ADX {regime['adx']} vs min {regime['adx_min']} -> {regime['trend_strength']}")
    print(f"       tradeable direction: {regime['tradeable_direction']}")

    strategy = load_strategy(list_strategies()[0])
    print(f"[ OK ] strategy loaded: {strategy.name}")

    agent = GeminiAgent()
    print(f"[ .. ] model chain: {' -> '.join(agent.model_chain)} ({agent.timeout_ms} ms timeout)")

    decision = agent.decide(snapshot, strategy.prompt_context())
    print(f"[ OK ] served by: {agent.last_model_used}")
    print(f"[ OK ] decision: {decision.action} (confidence {decision.confidence})")
    print(f"       stop {decision.stop_loss} / target {decision.take_profit}")
    print(f"       {decision.rationale}")

    try:
        test_decision_respects_regime_gate()
    except AssertionError as exc:
        print(f"\n[FAIL] regime gate violated: {exc}")
        return 1
    print("[ OK ] decision respects the regime gate")

    print("\n[PASS] Full pipeline is operational.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
