"""Small counters for Gemini confirms, rejects, and outages.

The runner updates `state/gemini_stats.json`. The practice summary reads it.
Hard risk gates still decide before these counters move.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COUNTERS = ("confirmed", "rejected", "agent_unavailable")


def empty_stats() -> dict[str, Any]:
    return {name: 0 for name in COUNTERS}


def load_stats(path: Path) -> dict[str, Any]:
    stats = empty_stats()
    if not path.exists():
        return stats
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return stats
    if not isinstance(raw, dict):
        return stats
    for name in COUNTERS:
        try:
            stats[name] = int(raw.get(name) or 0)
        except (TypeError, ValueError):
            stats[name] = 0
    if raw.get("updated_at"):
        stats["updated_at"] = str(raw["updated_at"])
    return stats


def bump_stat(path: Path, kind: str) -> dict[str, Any]:
    """Increment one counter and write the file. Unknown names are ignored."""
    stats = load_stats(path)
    if kind not in COUNTERS:
        return stats
    stats[kind] = int(stats[kind]) + 1
    stats["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def format_stats(stats: dict[str, Any] | None) -> str:
    stats = stats or empty_stats()
    return (
        f"confirmed={int(stats.get('confirmed') or 0)} "
        f"rejected={int(stats.get('rejected') or 0)} "
        f"unavailable={int(stats.get('agent_unavailable') or 0)}"
    )
