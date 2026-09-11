"""Render Pine templates into standalone, paste-ready TradingView scripts.

    python -m strategies.render                  # render every strategy
    python -m strategies.render ema_atr_trend    # render one
    python -m strategies.render --check          # verify committed output is current

Output lands in `strategies/rendered/<name>.pine`. Those files are generated
artifacts, but they are committed on purpose: they are what you paste into the
Pine Editor, so being able to diff them across parameter changes is the point.
`--check` is what keeps them from silently drifting from the template.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python strategies/render.py` as well
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings
from strategies.base import StrategyError, list_strategies, load_strategy

#: A `$name` or `${name}` that survived substitution means a missing parameter.
LEFTOVER_PLACEHOLDER = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def output_dir() -> Path:
    return get_settings().paths["rendered"]


def _banner(name: str) -> list[str]:
    """Header inserted below the version annotation.

    Deliberately carries no timestamp: a rendered file should change only when
    the template or parameters change, otherwise every run dirties the diff and
    `--check` could never pass.
    """
    return [
        "// ===========================================================",
        "// GENERATED FILE - do not edit directly.",
        f"//   template : strategies/templates/{name}.pine",
        f"//   params   : strategies/params/{name}.json",
        f"//   re-render: python -m strategies.render {name}",
        "// ===========================================================",
    ]


def render(name: str) -> str:
    """Return the finished Pine source for `name`."""
    pine = load_strategy(name).render()

    leftover = LEFTOVER_PLACEHOLDER.findall(pine)
    if leftover:
        raise StrategyError(
            f"Template '{name}' has unsubstituted placeholders: {sorted(set(leftover))}"
        )

    lines = pine.splitlines()
    if not lines or not lines[0].startswith("//@version="):
        raise StrategyError(
            f"Template '{name}' must start with a //@version= annotation so the "
            "generated header cannot displace it."
        )

    return "\n".join([lines[0], *_banner(name), *lines[1:]]) + "\n"


def write(name: str) -> Path:
    """Render `name` to disk and return the output path."""
    destination = output_dir() / f"{name}.pine"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the file LF-only on Windows; TradingView does not care,
    # but it stops the committed artifact from flipping line endings.
    destination.write_text(render(name), encoding="utf-8", newline="\n")
    return destination


def check(name: str) -> bool:
    """True if the file on disk matches a fresh render."""
    destination = output_dir() / f"{name}.pine"
    if not destination.exists():
        print(f"[MISSING] {destination}")
        return False
    if destination.read_text(encoding="utf-8") != render(name):
        print(f"[STALE  ] {destination}")
        return False
    print(f"[CURRENT] {destination}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names",
        nargs="*",
        help="strategy names to render (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify rendered files are up to date instead of writing them",
    )
    args = parser.parse_args(argv)

    available = list_strategies()
    targets = args.names or available

    unknown = [n for n in targets if n not in available]
    if unknown:
        parser.error(f"unknown strategy {unknown}; available: {available}")

    try:
        if args.check:
            stale = [name for name in targets if not check(name)]
            if stale:
                print(f"\n{len(stale)} file(s) out of date. Run: python -m strategies.render")
                return 1
            print(f"\nAll {len(targets)} rendered file(s) current.")
            return 0

        for name in targets:
            path = write(name)
            lines = path.read_text(encoding="utf-8").count("\n")
            print(f"[ OK ] {path}  ({lines} lines)")
    except StrategyError as exc:
        print(f"[FAIL] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
