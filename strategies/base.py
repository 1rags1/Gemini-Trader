"""Strategy loading and Pine Script template rendering.

A strategy is a pair of files sharing a stem:
    strategies/templates/<name>.pine   - Pine Script v5 source with $placeholders
    strategies/params/<name>.json      - parameter values + prompt context

`Strategy.render()` substitutes parameters into the Pine template so the same
definition drives both the TradingView script and the Gemini prompt context.

Placeholders use `string.Template` syntax (`$name`) rather than `str.format`
because Pine leans on braces heavily: `{{ticker}}` alert tokens and JSON webhook
payloads would both need every literal brace doubled. `$` appears nowhere in
Pine or JSON, so templates stay readable. Use `$$` for a literal dollar sign.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

from core.config import get_settings


class StrategyError(RuntimeError):
    """Raised when a strategy definition is missing or invalid."""


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    params: dict
    template: str

    def pine_params(self) -> dict[str, object]:
        """Parameters coerced to Pine literals.

        JSON booleans arrive as Python `True`/`False`, which are not valid Pine;
        Pine and JSON both spell them lowercase.
        """
        return {
            key: ("true" if value else "false") if isinstance(value, bool) else value
            for key, value in self.params.items()
        }

    def render(self) -> str:
        """Return the Pine Script with `$param` placeholders filled in."""
        try:
            return Template(self.template).substitute(self.pine_params())
        except KeyError as exc:
            raise StrategyError(
                f"Pine template '{self.name}' references undefined parameter {exc}"
            ) from exc
        except ValueError as exc:
            raise StrategyError(
                f"Malformed placeholder in Pine template '{self.name}': {exc}"
            ) from exc

    def prompt_context(self) -> str:
        """Human-readable strategy brief injected into the Gemini prompt."""
        lines = [f"Strategy: {self.name}", self.description, "Parameters:"]
        lines += [f"  - {k} = {v}" for k, v in sorted(self.params.items())]
        return "\n".join(lines)


def _dirs() -> tuple[Path, Path]:
    paths = get_settings().paths
    return paths["templates"], paths["params"]


def list_strategies() -> list[str]:
    templates_dir, _ = _dirs()
    return sorted(p.stem for p in templates_dir.glob("*.pine"))


def load_strategy(name: str) -> Strategy:
    templates_dir, params_dir = _dirs()
    template_path = templates_dir / f"{name}.pine"
    params_path = params_dir / f"{name}.json"

    if not template_path.exists():
        raise StrategyError(
            f"No Pine template at {template_path}. Available: {list_strategies()}"
        )
    if not params_path.exists():
        raise StrategyError(f"No parameter file at {params_path}")

    config = json.loads(params_path.read_text(encoding="utf-8"))
    return Strategy(
        name=name,
        description=config.get("description", ""),
        params=config.get("params", {}),
        template=template_path.read_text(encoding="utf-8"),
    )
