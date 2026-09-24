"""Gemini agent wrapper built on the official `google-genai` SDK.

Two responsibilities:
  * `ping()`   - cheap liveness/credential check used by the test suite.
  * `decide()` - schema-constrained trading decision for the runner.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from core.config import get_settings
from core.net import enable_os_trust_store

# The SDK warns about automatic function calling on every generate_content call.
# We pass no tools, so the warning is noise.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

SYSTEM_INSTRUCTION = """You are a disciplined quantitative trading analyst.
You receive a market snapshot with OHLCV candles, technical indicators and a
`regime` block. Return a single trading decision as JSON.

The regime gate is not advisory. It mirrors the backtested Pine strategy and
overrides every other signal:
- regime.tradeable_direction == "long_only"  -> BUY or HOLD only, never SELL.
- regime.tradeable_direction == "short_only" -> SELL or HOLD only, never BUY.
- regime.tradeable_direction == "none"       -> HOLD. ADX is below
  regime.adx_min, so there is no trend worth paying spread and fees for.
- regime.tradeable_direction == "unknown"    -> HOLD. Indicators are still
  warming up and the regime cannot be established.
Trading against the EMA 200 macro trend, or inside a weak-ADX chop, is what
produced a 0.56 profit factor and a 34% commission load in backtesting.

Risk rules:
- Entries need trend agreement (fast EMA vs slow EMA) plus momentum
  confirmation (RSI beyond 50 in the trade's direction).
- stop_loss is 1.5x of the 15m ATR from price, take_profit is 3.5x ATR
  (~1:2.33). Keep that ratio. The 1h regime block overrides the 15m trigger.
- For HOLD, stop_loss and take_profit are null.

Veto rules. You are a coach, not a cheerleader:
- Marginal, late, choppy, or only-partly-aligned setups must be REJECT.
- A high confidence number is not a confirm. Prefer REJECT over a weak BUY or SELL.
- BUY or SELL only when the 1h regime and the 15m trigger agree and the setup is clean.
- REJECT and HOLD both mean "do not open a trade". Skipping is better than a thin edge.
- When indicators conflict, return REJECT, not a hopeful confirm.

Output rules:
- Never invent data that is not in the snapshot.
- confidence is 0.0-1.0 and must reflect genuine signal strength.
- rationale is at most two sentences and must cite the regime when it forces
  the decision.

Hard risk limits are enforced in code, not by this response. You cannot
override position size, the circuit breaker, the altcoin correlation cap,
spot long-only, post-only entries, or the stop and target the strategy already
computed. If you return stop_loss or take_profit, the runner ignores them.
Return only the JSON decision."""

PAPER_MODE_NOTE = (
    "Operating mode: paper trading. Entries and exits are simulated locally "
    "and are not sent to the exchange. This is not financial advice."
)

LIVE_MODE_NOTE = (
    "Operating mode: live trading. A confirmed decision may be followed by a "
    "real exchange order placed by the strategy runner, and only when the "
    "runner's own live gate allows it. You still cannot place, cancel, or "
    "resize orders yourself. This is not financial advice."
)


def system_instruction(*, paper_trading: bool) -> str:
    """Prompt body plus the mode the process is actually running in."""
    note = PAPER_MODE_NOTE if paper_trading else LIVE_MODE_NOTE
    return f"{SYSTEM_INSTRUCTION}\n{note}"

#: Transient overload / rate-limit statuses worth retrying.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Attempts per model before falling back to the next in the chain. Kept low
#: because switching models recovers from an overloaded endpoint far faster
#: than retrying it: 2 attempts x 3 models x 20 s caps a total failure near 2 min.
MAX_ATTEMPTS = 2
BASE_BACKOFF = 1.5

DECISION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "action": {"type": "STRING", "enum": ["BUY", "SELL", "HOLD", "REJECT"]},
        "confidence": {"type": "NUMBER"},
        "stop_loss": {"type": "NUMBER", "nullable": True},
        "take_profit": {"type": "NUMBER", "nullable": True},
        "rationale": {"type": "STRING"},
    },
    "required": ["action", "confidence", "rationale"],
}


@dataclass(frozen=True)
class Decision:
    action: str
    confidence: float
    rationale: str
    stop_loss: float | None = None
    take_profit: float | None = None


class GeminiAgent:
    def __init__(
        self,
        model: str | None = None,
        timeout_ms: int | None = None,
        paper_trading: bool | None = None,
    ) -> None:
        enable_os_trust_store()  # before the client builds its SSL context
        cfg = get_settings()
        self.paper_trading = bool(cfg.paper_trading) if paper_trading is None else bool(paper_trading)
        self.model = model or cfg.gemini_model
        self.timeout_ms = timeout_ms or cfg.gemini_timeout_ms
        self.model_chain = (self.model,) + tuple(
            m for m in cfg.gemini_fallback_models if m != self.model
        )
        #: Model that served the most recent successful call.
        self.last_model_used: str | None = None
        self._client = genai.Client(
            api_key=cfg.gemini_api_key,
            # SDK expects milliseconds. Without this the underlying httpx client
            # has no read timeout, so a stalled request blocks forever.
            http_options=types.HttpOptions(timeout=self.timeout_ms),
        )

    @property
    def client(self) -> genai.Client:
        return self._client

    def _generate_on(
        self,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
        attempts: int,
    ) -> types.GenerateContentResponse:
        """Call one model, retrying overload responses and timeouts with backoff.

        Raises the underlying error once `attempts` is exhausted. Non-retryable
        errors (bad key, retired model, invalid argument) propagate immediately.
        """
        last_error: Exception

        for attempt in range(attempts):
            try:
                return self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except httpx.TimeoutException as exc:
                reason = f"timeout after {self.timeout_ms} ms"
                last_error = exc
            except (genai_errors.ServerError, genai_errors.ClientError) as exc:
                if getattr(exc, "code", None) not in RETRYABLE_STATUS:
                    raise
                # A 429 can mean either a per-minute rate limit or an exhausted
                # quota. Backing off seconds cannot fix the latter, so hand over
                # to the next model in the chain immediately.
                if exc.code == 429 and "RESOURCE_EXHAUSTED" in str(exc):
                    raise
                reason = f"HTTP {exc.code}"
                last_error = exc

            if attempt == attempts - 1:
                break
            delay = BASE_BACKOFF * (2**attempt) + random.uniform(0, 0.5)
            logging.warning(
                "Gemini %s on %s; retrying in %.1fs (attempt %d/%d)",
                reason,
                model,
                delay,
                attempt + 1,
                attempts,
            )
            time.sleep(delay)

        raise last_error

    def _generate(
        self,
        contents: str,
        config: types.GenerateContentConfig,
        attempts: int = MAX_ATTEMPTS,
    ) -> types.GenerateContentResponse:
        """Return the first response from the model chain.

        Each model gets `attempts` tries before falling back to the next one, so
        a single overloaded endpoint cannot stall the trading loop. Every attempt
        is bounded by the client timeout, keeping total wall time finite.
        """
        failures: list[str] = []

        for model in self.model_chain:
            try:
                response = self._generate_on(model, contents, config, attempts)
            except Exception as exc:
                failures.append(f"{model}: {type(exc).__name__}: {str(exc)[:120]}")
                logging.warning("Gemini exhausted '%s'; falling back", model)
                continue

            self.last_model_used = model
            return response

        raise RuntimeError(
            "No Gemini model in the chain answered:\n  " + "\n  ".join(failures)
        )

    def ping(self, prompt: str = "Reply with exactly: OK") -> str:
        """Round-trip a trivial prompt to confirm key, network, and model access."""
        response = self._generate(
            prompt,
            types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        return (response.text or "").strip()

    def decide(self, snapshot: dict, strategy_context: str = "") -> Decision:
        payload = json.dumps(snapshot, indent=2, default=str)
        contents = f"{strategy_context}\n\nMarket snapshot:\n{payload}".strip()

        response = self._generate(
            contents,
            types.GenerateContentConfig(
                system_instruction=system_instruction(paper_trading=self.paper_trading),
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                response_mime_type="application/json",
                response_schema=DECISION_SCHEMA,
            ),
        )

        data = json.loads(response.text or "{}")
        return Decision(
            action=str(data.get("action", "HOLD")).upper(),
            confidence=float(data.get("confidence", 0.0)),
            rationale=str(data.get("rationale", "")),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
        )
