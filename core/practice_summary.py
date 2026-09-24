"""Plain-language practice check for the paper book.

    python -m core.practice_summary
    python -m core.practice_summary --write

`--write` saves the same note under `state/practice_notes/`. That folder is
runtime state and is not committed. This command does not place orders.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from core.config import PROJECT_ROOT, TRADING_PAIRS
from core.gemini_stats import format_stats, load_stats
from core.symbols import reconcile_book
from core.trade_log import closed_pnl, iter_rows

WINDOW = timedelta(hours=24)


def _env_bool(name: str, default: bool) -> bool:
    import os

    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_ts(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _in_window(stamp: datetime | None, *, start: datetime, now: datetime) -> bool:
    return stamp is not None and start <= stamp <= now


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _money(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:,.2f}"


def _error_lines(log_dir: Path, *, start: datetime) -> int:
    if not log_dir.exists():
        return 0
    count = 0
    for path in log_dir.glob("*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "ERROR" not in line:
                continue
            stamp = None
            head = line[:19]
            try:
                stamp = datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                stamp = _parse_ts(line[:25])
            if stamp is None or stamp >= start:
                count += 1
    return count


def build_summary(
    *,
    now: datetime | None = None,
    root: Path | None = None,
) -> str:
    """Return the note as plain text. `now` is injectable for tests."""
    import os

    root = root or PROJECT_ROOT
    load_dotenv(root / ".env", override=False)
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    start = moment - WINDOW
    day_start = moment.replace(hour=0, minute=0, second=0, microsecond=0)

    paper = _env_bool("PAPER_TRADING", True)
    allow_live = _env_bool("ALLOW_LIVE_TRADING", False)
    try:
        starting_default = float(os.getenv("PAPER_STARTING_BALANCE", "10000"))
    except ValueError:
        starting_default = 10_000.0

    state_path = root / "state" / "runner.json"
    raw = _read_json(state_path) or {}
    positions, _notes = reconcile_book(raw.get("positions") or {}, TRADING_PAIRS, paper=paper)
    try:
        equity = float(raw["equity"]) if raw.get("equity") is not None else None
    except (TypeError, ValueError):
        equity = None
    try:
        start_equity = float(raw["start_equity"]) if raw.get("start_equity") is not None else None
    except (TypeError, ValueError):
        start_equity = None
    if start_equity is None and paper:
        start_equity = starting_default

    data_dir = Path(os.getenv("DATA_DIR", str(root / "data")))
    trade_name = "paper_trades.csv" if paper or not allow_live else "live_trades.csv"
    trades = iter_rows(data_dir / trade_name)
    rejects = iter_rows(data_dir / "rejected_alerts.csv")

    def rows_between(rows: list[dict[str, str]], begin: datetime) -> list[dict[str, str]]:
        picked = []
        for row in rows:
            stamp = _parse_ts(row.get("timestamp", ""))
            if _in_window(stamp, start=begin, now=moment):
                picked.append(row)
        return picked

    last_day = rows_between(trades, start)
    today = rows_between(trades, day_start)
    closes = []
    for row in last_day:
        pnl = closed_pnl(row)
        if pnl is None and str(row.get("action") or "").upper() == "CLOSE" and row.get("pnl"):
            try:
                pnl = float(row["pnl"])
            except (TypeError, ValueError):
                pnl = None
        if pnl is not None:
            closes.append(pnl)
    wins = sum(1 for pnl in closes if pnl > 0)
    losses = sum(1 for pnl in closes if pnl < 0)
    pnl_sum = sum(closes)

    confirmed = 0
    rejected = 0
    unavailable = 0
    for row in trades + rejects:
        stamp = _parse_ts(row.get("timestamp", ""))
        if not _in_window(stamp, start=start, now=moment):
            continue
        verdict = str(row.get("verdict") or "").upper()
        reason = f"{row.get('alert_reason') or ''} {row.get('rationale') or ''}".lower()
        if "agent_unavailable" in reason:
            unavailable += 1
        elif verdict == "REJECTED":
            rejected += 1
        elif verdict == "CONFIRMED":
            confirmed += 1

    stats = load_stats(root / "state" / "gemini_stats.json")
    error_count = _error_lines(root / "logs", start=start)

    if paper:
        mode_line = "Still paper? Yes. This is fake money. No real Kraken order is sent."
    elif allow_live:
        mode_line = "Still paper? No. Paper is off and live trading is on. Real orders can be sent."
    else:
        mode_line = "Still paper? No, but live trading is also off, so nothing is sent to Kraken."

    open_lines: list[str] = []
    for symbol, slot in positions.items():
        status = str(slot.get("status") or "FLAT").upper()
        if status in {"", "FLAT"}:
            continue
        open_lines.append(
            f"- {symbol} {status} entry {_money(_as_float(slot.get('entry_price')))} "
            f"stop {_money(_as_float(slot.get('stop_loss')))} "
            f"target {_money(_as_float(slot.get('take_profit')))}"
        )
    if not open_lines:
        open_lines.append("- none")

    book_pnl = None if equity is None or start_equity is None else equity - start_equity
    lines = [
        "Practice check",
        mode_line,
        f"Paper start setting: {_money(starting_default)}.",
        f"Saved equity: {_money(equity)}. Start equity: {_money(start_equity)}.",
        f"Saved profit since start: {_money(book_pnl)}.",
        "",
        f"Trades in the last 24 hours: {len(last_day)}.",
        f"Trades today (UTC): {len(today)}.",
        f"Closed trades in the last 24 hours: {len(closes)}.",
        f"Wins: {wins}. Losses: {losses}.",
        f"Rough closed P&L in the last 24 hours: {_money(pnl_sum)}.",
        "",
        "Open positions:",
        *open_lines,
        "",
        "Gemini in the last 24 hours (from the trade logs):",
        f"- confirmed: {confirmed}",
        f"- rejected: {rejected}",
        f"- agent unavailable: {unavailable}",
        f"Runner counters so far: {format_stats(stats)}.",
        "",
        "Problems:",
    ]
    if unavailable >= 3:
        lines.append(
            f"- Gemini was down {unavailable} times in the last day. Those entries were skipped."
        )
    elif unavailable:
        lines.append(
            f"- Gemini was down {unavailable} time(s) in the last day. Those entries were skipped."
        )
    else:
        lines.append("- No Gemini outage in the last day.")
    if error_count >= 5:
        lines.append(f"- Error spike: {error_count} ERROR lines in logs over the last day.")
    elif error_count:
        lines.append(f"- {error_count} ERROR line(s) in logs over the last day.")
    else:
        lines.append("- No ERROR lines in the log folder.")
    if (
        paper
        and equity is not None
        and start_equity is not None
        and abs(equity - start_equity) < 0.05
        and abs(pnl_sum) >= 0.05
    ):
        lines.append(
            "- Saved equity is still the start balance, but recent closes have P&L. "
            "Start the runner once so it can add that P&L, or reset the paper book."
        )
    lines.append("")
    lines.append("Webhook alerts are a log only. They do not change this equity.")
    return "\n".join(lines)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a plain-language paper practice check")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Also save the note under state/practice_notes/ (gitignored runtime state).",
    )
    args = parser.parse_args(argv)
    text = build_summary()
    print(text)
    if args.write:
        folder = PROJECT_ROOT / "state" / "practice_notes"
        folder.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = folder / f"{day}.txt"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
