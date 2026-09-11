"""System checks that run offline: dependencies, secret hygiene, wiring.

Run standalone:   python tests/test_environment.py
Run under pytest: pytest tests/test_environment.py -v
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_MODULES = ["google.genai", "dotenv", "pandas", "pandas_ta", "ccxt"]
SECRET_PATTERNS = [".env", "__pycache__"]


def test_required_packages_importable() -> None:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            missing.append(f"{name} ({exc})")
    assert not missing, "Missing dependencies: " + ", ".join(missing)


def test_gitignore_protects_secrets() -> None:
    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists(), f"No .gitignore at {gitignore}"

    entries = {
        line.strip().rstrip("/")
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for pattern in SECRET_PATTERNS:
        assert pattern in entries, f"'{pattern}' is not ignored by .gitignore"


def test_settings_resolve() -> None:
    from core.config import get_settings

    cfg = get_settings()
    assert cfg.gemini_api_key, "GEMINI_API_KEY did not resolve"
    assert cfg.gemini_model


def test_strategy_template_renders() -> None:
    from strategies import list_strategies, load_strategy

    names = list_strategies()
    assert names, "No Pine templates found under strategies/templates/"

    strategy = load_strategy(names[0])
    pine = strategy.render()
    assert "@version=5" in pine
    assert strategy.prompt_context()


def main() -> int:
    checks = [
        ("required packages importable", test_required_packages_importable),
        ("gitignore protects secrets", test_gitignore_protects_secrets),
        ("settings resolve from .env", test_settings_resolve),
        ("pine template renders", test_strategy_template_renders),
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

    print("\n" + ("[PASS] Environment ready." if not failures else f"[FAIL] {failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
