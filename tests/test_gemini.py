"""End-to-end verification of the Gemini API connection.

The API call goes straight to the `google-genai` SDK rather than through
`core.gemini_agent`, so a failure here points at the environment (missing key,
no network, model access) and not at project code. The only thing it borrows
from `core/` is the TLS trust-store helper.

Run standalone:   python tests/test_gemini.py
Run under pytest: pytest tests/test_gemini.py -s
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.net import enable_os_trust_store  # noqa: E402  (needs sys.path above)

ENV_PATH = PROJECT_ROOT / ".env"

# The 2.5-flash line is closed to new API keys, so the default is Google's
# stated replacement. Each fallback is tried in order on 404/503.
DEFAULT_MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-flash-latest"]

TEST_PROMPT = (
    "You are the health check for a paper-trading bot. "
    "In one sentence, state what an ATR-based stop loss protects against."
)

# The SDK takes its timeout in milliseconds. Without it, httpx applies no read
# timeout and a stalled request would hang this script indefinitely.
TIMEOUT_MS = 20_000

# Gemini 3 models take `thinking_level`; the older `thinking_budget` is rejected
# with 400 INVALID_ARGUMENT. "low" keeps the health check fast and cheap.
REQUEST_CONFIG = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=512,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)

logging.getLogger("google_genai.models").setLevel(logging.ERROR)


def load_api_key() -> str:
    """Read GEMINI_API_KEY from the root .env file."""
    if not ENV_PATH.exists():
        raise RuntimeError(f"No .env file found at {ENV_PATH}")

    load_dotenv(ENV_PATH, override=True)
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY is empty or missing in {ENV_PATH}.\n"
            "Expected a line of the form: GEMINI_API_KEY=your_key_here"
        )
    return api_key


def mask(api_key: str) -> str:
    return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "*" * len(api_key)


def candidate_models() -> list[str]:
    preferred = (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()
    return [preferred] + [m for m in FALLBACK_MODELS if m != preferred]


def query_gemini(api_key: str, prompt: str = TEST_PROMPT) -> tuple[str, str]:
    """Send `prompt` to the first reachable model. Returns (model, response)."""
    enable_os_trust_store()
    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)
    )

    errors: list[str] = []
    for model in candidate_models():
        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=REQUEST_CONFIG
            )
        except Exception as exc:
            errors.append(f"  {model}: {type(exc).__name__}: {str(exc)[:160]}")
            continue

        text = (response.text or "").strip()
        if text:
            return model, text
        errors.append(f"  {model}: empty response ({response.candidates})")

    raise RuntimeError("No model answered.\n" + "\n".join(errors))


def test_gemini_connection() -> None:
    """pytest entry point: the API key works and a model answers."""
    api_key = load_api_key()
    model, text = query_gemini(api_key)
    assert text, "Gemini returned an empty response"
    print(f"\n[gemini:{model}] {text}")


def main() -> int:
    print("=" * 70)
    print("Gemini API verification")
    print("=" * 70)

    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        print(f"[FAIL] Configuration: {exc}")
        return 1

    trust = "OS store (truststore)" if enable_os_trust_store() else "certifi bundle"
    print(f"[ OK ] .env              : {ENV_PATH}")
    print(f"[ OK ] GEMINI_API_KEY    : {mask(api_key)} ({len(api_key)} chars)")
    print(f"[ OK ] TLS trust store   : {trust}")
    print(f"[ OK ] Request timeout   : {TIMEOUT_MS} ms per attempt")
    print(f"[ .. ] Model preference  : {' -> '.join(candidate_models())}")
    print(f"[ .. ] Prompt            : {TEST_PROMPT}")
    print("[ .. ] Awaiting response ...", flush=True)

    try:
        model, text = query_gemini(api_key)
    except Exception as exc:
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        return 1

    print(f"\n--- Response from {model} " + "-" * max(0, 48 - len(model)))
    print(text)
    print("-" * 70)
    print(f"\n[PASS] Gemini API reachable; key is valid; served by '{model}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
