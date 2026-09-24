"""Paper-book reset and stuck-equity repair.

Practice equity starts at `PAPER_STARTING_BALANCE` (default $10,000). A reset
archives the paper state file and starts that balance again. It does not
delete `live.json` or a live trade log. Closed runner P&L is added to equity
on each exit; if a saved book is still sitting on its start balance while the
CSV has P&L, startup applies that sum once.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.trade_log import closed_pnl, iter_rows


def archive_json(path: Path, *, prefix: str = "paper") -> Path | None:
    """Copy `path` into `state/archive/` and return the copy. Missing files are skipped."""
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = path.parent / "archive" / f"{prefix}-{stamp}-{path.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def sum_runner_closed_pnl(path: Path, *, since: str | None = None) -> float:
    """Sum runner CLOSE `pnl` values. Rows at or before `since` are skipped."""
    total = 0.0
    watermark = (since or "").strip()
    for row in iter_rows(path):
        pnl = closed_pnl(row)
        if pnl is None:
            continue
        if watermark:
            stamp = str(row.get("timestamp") or "").strip()
            if not stamp or stamp <= watermark:
                continue
        total += pnl
    return total
