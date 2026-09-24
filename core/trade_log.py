"""Shared trade CSV schema for the runner and the webhook.

CLOSE rows used to write the exit fill into `entry_price`, so a stop fill
looked like stop == entry. `entry_price` is the position entry. `exit_price`
is the fill. `stop_loss` and `take_profit` stay the protective levels.
"""

from __future__ import annotations

import csv
import re
import threading
from pathlib import Path
from typing import Any

TRADE_LOG_FIELDS = [
    "timestamp",
    "symbol",
    "action",
    "entry_price",
    "exit_price",
    "pnl",
    "hit",
    "stop_loss",
    "take_profit",
    "verdict",
    "confidence",
    "agent_action",
    "model",
    "alert_reason",
    "adx",
    "macro_ema",
    "loss_streak",
    "rationale",
    "source",
]

_LOCK = threading.Lock()
_ENTRY_RE = re.compile(r"entry=([-\d.]+)")
_FILL_RE = re.compile(r"fill=([-\d.]+)")
_PNL_RE = re.compile(r"pnl=([-\d.]+)")
_HIT_RE = re.compile(r"hit=(\w+)")


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def upgrade_row(row: dict[str, Any]) -> dict[str, str]:
    """Fill new columns from a legacy row. Does not invent a stop."""
    cleaned = {key: "" if row.get(key) is None else str(row.get(key)) for key in TRADE_LOG_FIELDS}
    for key, value in row.items():
        if key not in cleaned and value is not None:
            cleaned[key] = str(value)
    action = cleaned.get("action", "").upper()
    rationale = cleaned.get("rationale", "")
    entry_m = _ENTRY_RE.search(rationale)
    fill_m = _FILL_RE.search(rationale)
    pnl_m = _PNL_RE.search(rationale)
    hit_m = _HIT_RE.search(rationale)
    if action == "CLOSE":
        if _blank(cleaned.get("exit_price")):
            if fill_m:
                cleaned["exit_price"] = fill_m.group(1)
            elif not _blank(cleaned.get("entry_price")) and entry_m:
                cleaned["exit_price"] = cleaned["entry_price"]
        if entry_m:
            cleaned["entry_price"] = entry_m.group(1)
        if pnl_m and _blank(cleaned.get("pnl")):
            cleaned["pnl"] = pnl_m.group(1)
        if hit_m and _blank(cleaned.get("hit")):
            cleaned["hit"] = hit_m.group(1)
    return {key: cleaned.get(key, "") for key in TRADE_LOG_FIELDS}


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows


def _rewrite(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            upgraded = upgrade_row(row)
            writer.writerow({key: upgraded.get(key, "") for key in TRADE_LOG_FIELDS})


def migrate_file(path: Path) -> None:
    """Rewrite an older header in place so new columns stay aligned."""
    if not path.exists() or path.stat().st_size == 0:
        return
    header, rows = _read_rows(path)
    if header == TRADE_LOG_FIELDS:
        return
    _rewrite(path, rows)


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append one row, upgrading an older header first."""
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            migrate_file(path)
        is_new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_FIELDS, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            upgraded = upgrade_row(row)
            writer.writerow({key: upgraded.get(key, "") for key in TRADE_LOG_FIELDS})


def iter_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    _header, rows = _read_rows(path)
    return [upgrade_row(row) for row in rows]


def closed_pnl(row: dict[str, Any]) -> float | None:
    """P&L for a runner CLOSE row. Webhook log rows are ignored."""
    source = str(row.get("source") or "").strip().lower()
    if source and source != "runner":
        return None
    if str(row.get("action") or "").upper() != "CLOSE":
        return None
    raw = row.get("pnl")
    if raw in (None, ""):
        match = _PNL_RE.search(str(row.get("rationale") or ""))
        raw = match.group(1) if match else None
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
