"""Static validation of the rendered Pine Script.

TradingView is the only real compiler, but the failure that costs the most time
is a malformed webhook payload: it compiles fine and only breaks once a live
alert fires. These checks reconstruct every JSON payload in the script and parse
it, so a stray comma or quote fails here instead of in production.

Run standalone:   python tests/test_pine_render.py
Run under pytest: pytest tests/test_pine_render.py -v
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies import render as pine_render

STRATEGY = "ema_atr_trend"

REQUIRED_STRATEGY_ARGS = [
    'strategy("Gemini Trend Guard"',
    "overlay=true",
    "initial_capital=10000",
    "default_qty_type=strategy.percent_of_equity",
    "default_qty_value=25",
    "commission_type=strategy.commission.percent",
    "commission_value=0.075",
]

#: Sample values standing in for Pine runtime expressions.
SAMPLE_STRING = "SAMPLE"
SAMPLE_NUMBER = "123.45"


def rendered() -> str:
    return pine_render.render(STRATEGY)


def _resolve_concatenation(expression: str) -> str:
    """Collapse a Pine `'a' + expr + 'b'` chain into a concrete string.

    String literals are kept verbatim; anything else is replaced with a sample,
    numeric for `str.tostring(...)` since those sit unquoted in the JSON.
    """
    resolved = []
    for token in expression.split("+"):
        token = token.strip()
        if len(token) >= 2 and token.startswith("'") and token.endswith("'"):
            resolved.append(token[1:-1])
        elif token.startswith("str.tostring"):
            resolved.append(SAMPLE_NUMBER)
        elif '"true"' in token and '"false"' in token:
            resolved.append("true")  # ternary emitting a bare JSON boolean
        else:
            resolved.append(SAMPLE_STRING)
    return "".join(resolved)


def webhook_payload(pine: str) -> str:
    """Reconstruct the JSON built by the webhookJson() helper."""
    match = re.search(
        r"^webhookJson\([^)]*\)\s*=>\s*\n((?:^[ \t]+.*\n)+)", pine, re.MULTILINE
    )
    assert match, "webhookJson() helper not found in rendered script"
    return _resolve_concatenation(match.group(1))


def alertcondition_payloads(pine: str) -> dict[str, str]:
    """Map alert title -> message JSON, with {{tokens}} filled in."""
    pattern = re.compile(
        r"alertcondition\(\s*[^,]+,\s*\"([^\"]+)\"\s*,\s*'(.*?)'\s*\)", re.DOTALL
    )
    found = {
        title: re.sub(r"\{\{[a-z._]+\}\}", SAMPLE_STRING, message)
        for title, message in pattern.findall(pine)
    }
    assert found, "no alertcondition() calls found in rendered script"
    return found


def test_rendered_file_is_current() -> None:
    assert pine_render.check(STRATEGY), (
        "strategies/rendered is stale; run: python -m strategies.render"
    )


def test_version_annotation_is_first_line() -> None:
    assert rendered().splitlines()[0] == "//@version=5"


def test_strategy_declaration_matches_spec() -> None:
    pine = rendered()
    for fragment in REQUIRED_STRATEGY_ARGS:
        assert fragment in pine, f"missing from strategy() call: {fragment}"


def test_no_unsubstituted_placeholders() -> None:
    leftover = pine_render.LEFTOVER_PLACEHOLDER.findall(rendered())
    assert not leftover, f"unsubstituted placeholders: {sorted(set(leftover))}"


def test_webhook_payload_is_valid_json() -> None:
    payload = webhook_payload(rendered())
    parsed = json.loads(payload)  # raises if the concatenation is malformed

    assert parsed["strategy"] == "Gemini Trend Guard"
    for key in [
        "action", "reason", "ticker", "timeframe", "price", "atr", "stop",
        "target", "adx", "macro_ema", "loss_streak", "breaker_active",
    ]:
        assert key in parsed, f"webhook payload missing '{key}'"
    # price/atr must be numbers, not strings, so a consumer can do maths on them.
    assert isinstance(parsed["price"], (int, float))
    assert isinstance(parsed["atr"], (int, float))
    assert isinstance(parsed["breaker_active"], bool)


def test_alertcondition_payloads_are_valid_json() -> None:
    for title, message in alertcondition_payloads(rendered()).items():
        parsed = json.loads(message)  # raises if malformed
        assert parsed["strategy"] == "Gemini Trend Guard", title
        assert parsed["action"] in {"BUY", "SELL", "CLOSE"}, title


#: Break-even win rate the configured reward:risk must undercut. Runs 1 and 2
#: landed at 32.1% and 28.24%, so anything needing more than 30% is betting on
#: a win rate this strategy has never achieved.
MAX_ACCEPTABLE_BREAKEVEN_WIN_RATE = 0.30


def test_atr_parameters_match_python_side() -> None:
    from strategies import load_strategy

    params = load_strategy(STRATEGY).params
    pine = rendered()

    assert f'input.int({params["atr_length"]}, "ATR Length"' in pine
    assert f'input.float({params["atr_stop_mult"]}, "Initial Stop (ATR mult)"' in pine
    assert f'input.float({params["atr_target_mult"]}, "Target (ATR mult)"' in pine
    assert f'input.float({params["atr_trail_mult"]}, "Trail Distance (ATR mult)"' in pine
    assert f'input.int({params["ema_fast"]}, "EMA Fast"' in pine
    assert f'input.int({params["ema_slow"]}, "EMA Slow"' in pine
    assert f'input.int({params["ema_macro"]}, "Macro EMA (dominant trend)"' in pine


def test_reward_risk_clears_breakeven() -> None:
    """The configured reward:risk must break even at a plausible win rate.

    Run 1 realised 1.313 against a 2.115 requirement; run 2 realised 1.279
    against 2.54. Both lost. This guard checks the *ratio*, which is the only
    half of the equation the parameters control directly.
    """
    from strategies import load_strategy

    params = load_strategy(STRATEGY).params
    ratio = params["atr_target_mult"] / params["atr_stop_mult"]
    breakeven_win_rate = 1 / (1 + ratio)

    assert breakeven_win_rate <= MAX_ACCEPTABLE_BREAKEVEN_WIN_RATE, (
        f"reward:risk {ratio:.2f} needs a {breakeven_win_rate:.1%} win rate to "
        f"break even, above the {MAX_ACCEPTABLE_BREAKEVEN_WIN_RATE:.0%} ceiling"
    )


def test_trail_arms_above_the_trail_distance() -> None:
    """Arming must exceed the trail distance, or the trail locks in a loss.

    When trail_activate_mult <= atr_trail_mult, the first trail level sits at or
    below the entry price, so an armed winner can still be stopped out flat.
    That mechanic is what held realised reward:risk near 1.3 in both runs.
    """
    from strategies import load_strategy

    params = load_strategy(STRATEGY).params
    arm = params["trail_activate_mult"]
    trail = params["atr_trail_mult"]
    stop = params["atr_stop_mult"]

    assert arm > trail, (
        f"trail arms at {arm}x ATR but trails {trail}x behind price, locking in "
        f"{arm - trail:+.2f}x ATR; raise trail_activate_mult above atr_trail_mult"
    )
    locked_r = (arm - trail) / stop
    print(f"    (armed trail floor: {locked_r:+.2f}R)")


def test_position_sizing_is_bounded() -> None:
    """Risk sizing must be capped so a quiet regime cannot demand leverage."""
    from strategies import load_strategy

    params = load_strategy(STRATEGY).params
    pine = rendered()

    assert params["max_notional_pct"] <= 100, "notional cap implies leverage"
    assert "math.min(qtyByRisk, qtyCeiling)" in pine, "notional ceiling not applied"
    assert "qty=entryQty" in pine, "entries do not use the sized quantity"


def test_regime_filters_gate_entries() -> None:
    """Both entry signals must be gated by macro trend, ADX and the breaker."""
    pine = rendered()
    for signal, macro in [("longSignal", "macroBull"), ("shortSignal", "macroBear")]:
        match = re.search(rf"^{signal}\s*=(.*)$", pine, re.MULTILINE)
        assert match, f"no assignment found for {signal}"
        for gate in [macro, "trendStrong", "not breakerActive"]:
            assert gate in match.group(1), f"{signal} is not gated by {gate}"


def test_circuit_breaker_present() -> None:
    pine = rendered()
    for fragment in [
        "lossStreak >= maxLossStreak",       # consecutive-loss trip
        "equityBreach",                      # rolling equity trip
        "breakerBars >= breakerCooldown",    # cooldown before re-arming
        "strategy.closedtrades.profit(i)",   # streak counted from real results
    ]:
        assert fragment in pine, f"circuit breaker missing: {fragment}"


def main() -> int:
    checks = [
        ("rendered file is current", test_rendered_file_is_current),
        ("version annotation first", test_version_annotation_is_first_line),
        ("strategy() matches spec", test_strategy_declaration_matches_spec),
        ("no leftover placeholders", test_no_unsubstituted_placeholders),
        ("webhook JSON parses", test_webhook_payload_is_valid_json),
        ("alertcondition JSON parses", test_alertcondition_payloads_are_valid_json),
        ("ATR/EMA params match Python", test_atr_parameters_match_python_side),
        ("reward:risk clears breakeven", test_reward_risk_clears_breakeven),
        ("trail arms above trail distance", test_trail_arms_above_the_trail_distance),
        ("position sizing bounded", test_position_sizing_is_bounded),
        ("regime filters gate entries", test_regime_filters_gate_entries),
        ("circuit breaker wired", test_circuit_breaker_present),
    ]

    failures = 0
    for label, check in checks:
        try:
            check()
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {label}: {exc}")
        else:
            print(f"[ OK ] {label}")

    pine = rendered()
    print("\nWebhook payload (order alert_message):")
    print("  " + webhook_payload(pine))
    print("\nSignal alerts (alertcondition):")
    for title in alertcondition_payloads(pine):
        print(f"  - {title}")

    print("\n" + ("[PASS] Rendered Pine Script validated." if not failures else f"[FAIL] {failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
