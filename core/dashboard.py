"""Read-only dashboard for the paper-trading runner.

Serves a single Tailwind page on 0.0.0.0:8050 and a JSON snapshot the browser
polls every 10 seconds. File reads are short-lived and shared, so this process
never locks `state/runner.json` or `data/paper_trades.csv` away from the runner.

    python -m core.dashboard
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from core.config import (
    MACRO_TIMEFRAME,
    MAX_OPEN_POSITIONS,
    PROJECT_ROOT,
    TRADING_PAIRS,
    TRIGGER_TIMEFRAME,
    get_settings,
    migrate_circuit_breaker,
    migrate_position_book,
)
from core.market_data import (
    add_macro_indicators,
    add_trigger_indicators,
    classify_macro_regime,
    fetch_ohlcv,
)
from core.net import enable_os_trust_store

log = logging.getLogger("dashboard")

DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8050
TRADE_LIMIT = 15
MARKET_CACHE_TTL = 20.0
MACRO_CANDLES = 250
TRIGGER_CANDLES = 100
MARKET_FETCH_DELAY = 1.0

_FILL_RE = re.compile(r"fill=([-\d.]+)")
_ENTRY_RE = re.compile(r"entry=([-\d.]+)")
_PNL_RE = re.compile(r"pnl=([-\d.]+)")
_HIT_RE = re.compile(r"hit=(\w+)")

_market_lock = threading.Lock()
_market_cache: dict[str, Any] = {"fetched_at": 0.0, "payload": None, "error": None}


def _read_shared(path: Path) -> str:
    """Read a file and drop the handle immediately. No exclusive lock."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read()


def _cfg():
    try:
        return get_settings()
    except Exception:
        return None


def dashboard_secret() -> str:
    cfg = _cfg()
    if cfg is None:
        return ""
    return (getattr(cfg, "dashboard_secret", None) or cfg.webhook_secret or "").strip()


#: Headers Cloudflare/cloudflared attach. A laptop browser talking to
#: 127.0.0.1 will not send these; a phone hitting the tunnel will.
_TUNNEL_HEADERS = ("cf-ray", "cf-connecting-ip", "cdn-loop")


def is_tunneled(request: Request) -> bool:
    return any(request.headers.get(name) for name in _TUNNEL_HEADERS)


def check_token(request: Request, token: str | None) -> None:
    """Require a token only for the public tunnel, not for localhost."""
    if not is_tunneled(request):
        return
    secret = dashboard_secret()
    if not secret:
        raise HTTPException(
            status_code=401,
            detail="set DASHBOARD_SECRET or WEBHOOK_SECRET before tunneling",
        )
    if not token or not secrets.compare_digest(token, secret):
        raise HTTPException(status_code=401, detail="invalid or missing token")


def _paths() -> dict[str, Path]:
    cfg = _cfg()
    if cfg is not None:
        return {
            "state": cfg.paths["root"] / "state" / "runner.json",
            "trades": cfg.paths["data"] / "paper_trades.csv",
            "rejects": cfg.paths["data"] / "rejected_alerts.csv",
        }
    return {
        "state": PROJECT_ROOT / "state" / "runner.json",
        "trades": PROJECT_ROOT / "data" / "paper_trades.csv",
        "rejects": PROJECT_ROOT / "data" / "rejected_alerts.csv",
    }


def _pairs() -> tuple[str, ...]:
    cfg = _cfg()
    if cfg is not None and getattr(cfg, "trading_pairs", None):
        return tuple(cfg.trading_pairs)
    return TRADING_PAIRS


def _max_open() -> int:
    cfg = _cfg()
    if cfg is not None and getattr(cfg, "max_open_positions", None):
        return int(cfg.max_open_positions)
    return MAX_OPEN_POSITIONS


def _slot_status(slot: dict[str, Any] | None) -> str:
    if not isinstance(slot, dict):
        return "FLAT"
    return str(slot.get("status") or slot.get("side") or "FLAT").upper() or "FLAT"


def _position_cards(
    book: dict[str, dict[str, Any]],
    last_bars: dict[str, Any],
    pairs: tuple[str, ...],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for symbol in pairs:
        slot = book.get(symbol) if isinstance(book, dict) else None
        status = _slot_status(slot)
        open_slot = dict(slot) if isinstance(slot, dict) and status not in {"", "FLAT"} else None
        if open_slot is not None:
            open_slot.setdefault("side", status)
        cards.append(
            {
                "symbol": symbol,
                "status": status if status not in {"", "FLAT"} else "FLAT",
                "position": open_slot,
                "last_bar": last_bars.get(symbol),
                "macro_regime": str((slot or {}).get("macro_regime") or "UNKNOWN").upper(),
            }
        )
    return cards


def _empty_state(*, error: str | None = None) -> dict[str, Any]:
    pairs = _pairs()
    empty: dict[str, Any] = {
        "exists": False,
        "equity": None,
        "starting_equity": None,
        "pnl": None,
        "pnl_pct": None,
        "loss_streak": 0,
        "breaker_active": False,
        "breaker_bars": 0,
        "last_bar": None,
        "last_bars": {},
        "position_side": "FLAT",
        "position": None,
        "positions": _position_cards({}, {}, pairs),
        "open_count": 0,
        "max_open_positions": _max_open(),
    }
    if error:
        empty["error"] = error
    return empty


def read_runner_state() -> dict[str, Any]:
    path = _paths()["state"]
    if not path.exists():
        return _empty_state()
    try:
        raw = json.loads(_read_shared(path))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("state unreadable: %s", exc)
        return _empty_state(error=str(exc))

    cfg = _cfg()
    starting = float(cfg.paper_starting_balance) if cfg is not None else 10_000.0
    equity = raw.get("equity")
    try:
        equity_f = float(equity) if equity is not None else None
    except (TypeError, ValueError):
        equity_f = None
    pnl = None if equity_f is None else equity_f - starting

    pairs = _pairs()
    book = migrate_position_book(raw if isinstance(raw, dict) else {}, pairs)
    last_bars = dict(raw.get("last_bars") or {})
    cards = _position_cards(book, last_bars, pairs)
    first_open = next((card["position"] for card in cards if card["position"] is not None), None)
    side = "FLAT" if first_open is None else str(first_open.get("side") or first_open.get("status") or "FLAT").upper()
    breaker = migrate_circuit_breaker(raw if isinstance(raw, dict) else {})

    return {
        "exists": True,
        "equity": equity_f,
        "starting_equity": starting,
        "pnl": pnl,
        "pnl_pct": None if pnl is None or starting == 0 else (pnl / starting) * 100.0,
        "loss_streak": int(breaker["loss_streak"]),
        "breaker_active": bool(breaker["tripped"]),
        "breaker_bars": int(raw.get("breaker_bars") or 0),
        "last_bar": raw.get("last_bar"),
        "last_bars": last_bars,
        "position_side": side,
        "position": first_open,
        "positions": cards,
        "open_count": sum(1 for card in cards if card["status"] != "FLAT"),
        "max_open_positions": _max_open(),
    }


def _f(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify(row: dict[str, str]) -> str:
    """Map a log row to a badge tone: green | red | amber | slate."""
    action = (row.get("action") or "").upper()
    verdict = (row.get("verdict") or "").upper()
    reason = f"{row.get('alert_reason') or ''} {row.get('rationale') or ''}".lower()
    hit_match = _HIT_RE.search(reason)
    hit = hit_match.group(1) if hit_match else ""

    if verdict == "REJECTED" or hit == "stop" or "stop_loss" in reason:
        return "red"
    if action == "SELL" and verdict != "CONFIRMED":
        return "red"
    if action == "SELL":
        return "red"
    if action == "BUY" or hit == "target" or "profit" in reason:
        return "green"
    if verdict == "CONFIRMED":
        return "green"
    return "slate"


def _enrich_trade(row: dict[str, str]) -> dict[str, Any]:
    rationale = row.get("rationale") or ""
    fill_m = _FILL_RE.search(rationale)
    entry_m = _ENTRY_RE.search(rationale)
    pnl_m = _PNL_RE.search(rationale)
    hit_m = _HIT_RE.search(rationale)
    action = (row.get("action") or "").upper()
    logged_price = _f(row.get("entry_price"))
    entry = _f(entry_m.group(1)) if entry_m else (logged_price if action != "CLOSE" else None)
    exit_price = _f(fill_m.group(1)) if fill_m else (logged_price if action == "CLOSE" else None)
    return {
        "timestamp": row.get("timestamp") or "",
        "symbol": row.get("symbol") or "",
        "action": action,
        "verdict": (row.get("verdict") or "").upper(),
        "tone": _classify(row),
        "entry_price": entry,
        "exit_price": exit_price,
        "stop_loss": _f(row.get("stop_loss")),
        "take_profit": _f(row.get("take_profit")),
        "confidence": _f(row.get("confidence")),
        "agent_action": row.get("agent_action") or "",
        "model": row.get("model") or "",
        "alert_reason": row.get("alert_reason") or "",
        "rationale": rationale,
        "pnl": _f(pnl_m.group(1)) if pnl_m else None,
        "hit": hit_m.group(1) if hit_m else "",
        "adx": _f(row.get("adx")),
        "loss_streak": row.get("loss_streak") or "",
    }


def read_trades(limit: int = TRADE_LIMIT) -> dict[str, Any]:
    path = _paths()["trades"]
    if not path.exists() or path.stat().st_size == 0:
        return {"count": 0, "trades": []}
    try:
        text = _read_shared(path)
        rows = list(csv.DictReader(io.StringIO(text)))
    except OSError as exc:
        log.warning("trade log unreadable: %s", exc)
        return {"count": 0, "trades": [], "error": str(exc)}
    recent = rows[-limit:]
    recent.reverse()
    return {"count": len(rows), "trades": [_enrich_trade(r) for r in recent]}


def read_reject_count() -> int:
    path = _paths()["rejects"]
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        text = _read_shared(path)
        return max(0, text.count("\n") - 1)
    except OSError:
        return 0


def _num(bar: Any, name: str) -> float | None:
    try:
        value = float(bar[name])
    except Exception:
        return None
    if value != value:
        return None
    return value


def _empty_pair_metrics(symbol: str, error: str | None = None) -> dict[str, Any]:
    return {
        "ok": error is None,
        "symbol": symbol,
        "macro_regime": "CHOP",
        "as_of": None,
        "last_close": None,
        "ema_fast": None,
        "ema_slow": None,
        "atr": None,
        "adx": None,
        "macro_close": None,
        "macro_ema_fast": None,
        "macro_ema_slow": None,
        "macro_ema_trend": None,
        "error": error,
    }


def _pair_metrics(symbol: str, exchange: str, *, paced: bool) -> dict[str, Any]:
    try:
        enable_os_trust_store()
        if paced and MARKET_FETCH_DELAY > 0:
            time.sleep(MARKET_FETCH_DELAY)
        macro = add_macro_indicators(
            fetch_ohlcv(
                symbol=symbol,
                timeframe=MACRO_TIMEFRAME,
                limit=MACRO_CANDLES,
                exchange_id=exchange,
            )
        )
        if MARKET_FETCH_DELAY > 0:
            time.sleep(MARKET_FETCH_DELAY)
        trigger = add_trigger_indicators(
            fetch_ohlcv(
                symbol=symbol,
                timeframe=TRIGGER_TIMEFRAME,
                limit=TRIGGER_CANDLES,
                exchange_id=exchange,
            )
        )
        macro_last = macro.iloc[-2] if len(macro) >= 2 else macro.iloc[-1]
        trigger_last = trigger.iloc[-2] if len(trigger) >= 2 else trigger.iloc[-1]
        regime = classify_macro_regime(macro_last)
        badge = "CHOP" if regime == "NEUTRAL" else regime
        return {
            "ok": True,
            "symbol": symbol,
            "macro_regime": badge,
            "as_of": trigger_last.name.isoformat() if hasattr(trigger_last.name, "isoformat") else str(trigger_last.name),
            "last_close": _num(trigger_last, "close"),
            "ema_fast": _num(trigger_last, "ema_fast"),
            "ema_slow": _num(trigger_last, "ema_slow"),
            "atr": _num(trigger_last, "atr"),
            "adx": _num(macro_last, "adx"),
            "macro_close": _num(macro_last, "close"),
            "macro_ema_fast": _num(macro_last, "ema_fast"),
            "macro_ema_slow": _num(macro_last, "ema_slow"),
            "macro_ema_trend": _num(macro_last, "ema_macro"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("market fetch failed for %s: %s", symbol, exc)
        return _empty_pair_metrics(symbol, error=f"{type(exc).__name__}: {exc}")


def fetch_market() -> dict[str, Any]:
    """Latest 1h + 15m snapshot per pair, cached so polls do not hammer Kraken."""
    now = time.monotonic()
    with _market_lock:
        age = now - float(_market_cache["fetched_at"])
        if _market_cache["payload"] is not None and age < MARKET_CACHE_TTL:
            return _market_cache["payload"]

    cfg = _cfg()
    pairs = _pairs()
    exchange = cfg.exchange_id if cfg is not None else "kraken"

    rows: list[dict[str, Any]] = []
    for index, symbol in enumerate(pairs):
        rows.append(_pair_metrics(symbol, exchange, paced=index > 0))

    errors = [row["error"] for row in rows if row.get("error")]
    payload = {
        "ok": any(row.get("ok") for row in rows),
        "timeframe": f"{MACRO_TIMEFRAME}/{TRIGGER_TIMEFRAME}",
        "exchange": exchange,
        "as_of": next((row.get("as_of") for row in rows if row.get("as_of")), None),
        "pairs": rows,
        "error": None if not errors else errors[0],
    }

    with _market_lock:
        _market_cache["fetched_at"] = time.monotonic()
        _market_cache["payload"] = payload
    return payload


def build_snapshot() -> dict[str, Any]:
    state = read_runner_state()
    market = fetch_market()
    trades = read_trades()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "poll_seconds": 10,
        "state": state,
        "market": market,
        "trades": trades,
        "rejects": read_reject_count(),
    }


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gemini Trader · Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui"] },
          colors: { ink: "#0b1014", panel: "#121a21" }
        }
      }
    }
  </script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    body { font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; }
    .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; }
  </style>
</head>
<body class="bg-ink text-slate-200 min-h-screen">
  <div class="max-w-6xl mx-auto px-5 py-6">
    <header class="flex flex-wrap items-end justify-between gap-3 mb-6">
      <div>
        <p class="text-xs uppercase tracking-[0.2em] text-teal-400/80">Gemini Trend Guard</p>
        <h1 class="text-2xl font-semibold text-white">Paper desk</h1>
        <p class="text-sm text-slate-400">Read-only monitor · does not lock the runner</p>
      </div>
      <div class="text-right text-sm text-slate-400">
        <div>Updated <span id="updated" class="mono text-slate-200">—</span></div>
        <div>Next refresh in <span id="countdown" class="mono text-teal-300">10</span>s</div>
        <div id="status" class="text-xs mt-1 text-slate-500">connecting…</div>
      </div>
    </header>

    <section id="pair-cards" class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4"></section>

    <section class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
      <article class="bg-panel rounded-xl border border-white/5 p-4">
        <p class="text-xs uppercase tracking-wider text-slate-400">Paper equity</p>
        <p id="equity" class="text-3xl font-semibold mt-1 mono">—</p>
        <p id="pnl" class="text-sm mt-2">—</p>
      </article>
      <article class="bg-panel rounded-xl border border-white/5 p-4">
        <p class="text-xs uppercase tracking-wider text-slate-400">Active slots</p>
        <p id="slots" class="text-3xl font-semibold mt-1 mono">—</p>
        <p id="slots-detail" class="text-xs text-slate-400 mt-2">Waiting for book</p>
      </article>
      <article class="bg-panel rounded-xl border border-white/5 p-4">
        <p class="text-xs uppercase tracking-wider text-slate-400">Circuit breaker</p>
        <p id="breaker" class="text-3xl font-semibold mt-1">—</p>
        <p id="breaker-detail" class="text-xs text-slate-400 mt-2">Loss streak —</p>
      </article>
    </section>

    <section class="bg-panel rounded-xl border border-white/5 p-4 mb-4 overflow-hidden">
      <div class="flex items-baseline justify-between mb-3">
        <h2 class="text-sm uppercase tracking-wider text-slate-400">Live metrics</h2>
        <p id="mkt-meta" class="text-xs text-slate-500 mono"></p>
      </div>
      <p id="mkt-error" class="hidden text-sm text-rose-300 mb-3"></p>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-left text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th class="pr-4 py-2 font-medium">Pair</th>
              <th class="px-3 py-2 font-medium">1h Regime</th>
              <th class="px-3 py-2 font-medium text-right">ADX</th>
              <th class="px-3 py-2 font-medium text-right">15m Close</th>
              <th class="px-3 py-2 font-medium text-right">EMA 9</th>
              <th class="px-3 py-2 font-medium text-right">EMA 21</th>
              <th class="px-3 py-2 font-medium text-right">15m ATR</th>
            </tr>
          </thead>
          <tbody id="metrics" class="divide-y divide-white/5"></tbody>
        </table>
      </div>
    </section>

    <section class="bg-panel rounded-xl border border-white/5 overflow-hidden">
      <div class="px-4 py-3 flex items-baseline justify-between border-b border-white/5">
        <h2 class="text-sm uppercase tracking-wider text-slate-400">Trade log</h2>
        <p id="trade-meta" class="text-xs text-slate-500">—</p>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-left text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th class="px-4 py-2 font-medium">Time</th>
              <th class="px-4 py-2 font-medium">Symbol</th>
              <th class="px-4 py-2 font-medium">Action</th>
              <th class="px-4 py-2 font-medium">Verdict</th>
              <th class="px-4 py-2 font-medium text-right">Entry</th>
              <th class="px-4 py-2 font-medium text-right">Exit</th>
              <th class="px-4 py-2 font-medium">Rationale</th>
            </tr>
          </thead>
          <tbody id="trades" class="divide-y divide-white/5"></tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    const POLL_MS = 10000;
    let remaining = 10;

    const fmt = (n, d=2) => n == null || Number.isNaN(n) ? "—" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
    const tonePos = (side) => ({LONG:"text-teal-300", SHORT:"text-rose-300", FLAT:"text-slate-200"}[side] || "text-slate-200");
    const badge = (tone, label) => {
      const map = {
        green: "bg-teal-500/15 text-teal-300 ring-teal-500/30",
        red: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
        amber: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
        slate: "bg-slate-500/15 text-slate-300 ring-slate-500/30",
      };
      return `<span class="inline-flex px-2 py-0.5 rounded-full text-[11px] font-medium ring-1 ${map[tone]||map.slate}">${label}</span>`;
    };

    const px = (n) => n == null || Number.isNaN(n) ? "—" : fmt(n, Number(n) >= 1000 ? 1 : Number(n) >= 100 ? 2 : 3);

    function render(data) {
      const s = data.state || {};
      const m = data.market || {};
      const cards = s.positions || [];
      const marketBySymbol = Object.fromEntries((m.pairs || []).map(row => [row.symbol, row]));
      const toneRegime = (reg) => ({BULL:"green", BEAR:"red", CHOP:"amber", NEUTRAL:"amber"}[reg] || "slate");
      document.getElementById("pair-cards").innerHTML = cards.map(card => {
        const side = card.status || "FLAT";
        const p = card.position;
        const live = marketBySymbol[card.symbol] || {};
        const regime = live.macro_regime || card.macro_regime || "CHOP";
        const detail = p
          ? `Entry ${px(p.entry_price)} · stop ${px(p.stop_loss || p.trail_stop || p.stop)} · target ${px(p.take_profit || p.target)}`
          : (card.last_bar ? `Last 15m bar ${card.last_bar}` : "No open position");
        return `<article class="bg-panel rounded-xl border border-white/5 p-4">
          <div class="flex items-center justify-between gap-2">
            <p class="text-xs uppercase tracking-wider text-slate-400">${card.symbol || "—"}</p>
            ${badge(toneRegime(regime), `${regime}${live.adx != null ? " ADX " + fmt(live.adx, 1) : ""}`)}
          </div>
          <p class="text-3xl font-semibold mt-1 ${tonePos(side)}">${side}</p>
          <p class="text-xs text-slate-400 mt-2 leading-relaxed">${detail}</p>
        </article>`;
      }).join("") || `<article class="bg-panel rounded-xl border border-white/5 p-4 md:col-span-3"><p class="text-sm text-slate-500">No position book</p></article>`;

      document.getElementById("equity").textContent = s.equity == null ? "—" : fmt(s.equity, 2);
      const pnl = s.pnl;
      const pnlEl = document.getElementById("pnl");
      if (pnl == null) { pnlEl.textContent = "Starting balance unknown"; pnlEl.className = "text-sm mt-2 text-slate-400"; }
      else {
        const sign = pnl >= 0 ? "+" : "";
        pnlEl.textContent = `${sign}${fmt(pnl, 2)}  (${sign}${fmt(s.pnl_pct, 2)}%) vs ${fmt(s.starting_equity, 0)} start`;
        pnlEl.className = "text-sm mt-2 " + (pnl >= 0 ? "text-teal-300" : "text-rose-300");
      }

      const openCount = s.open_count || 0;
      const maxOpen = s.max_open_positions || 0;
      document.getElementById("slots").textContent = maxOpen ? `${openCount} / ${maxOpen}` : String(openCount);
      document.getElementById("slots-detail").textContent = `Active Slots: ${openCount} / ${maxOpen || 2}` + (
        openCount
          ? " · " + cards.filter(c => c.status && c.status !== "FLAT").map(c => `${c.symbol} ${c.status}`).join(" · ")
          : " · all pairs flat"
      );

      document.getElementById("breaker").textContent = s.breaker_active ? "ON" : "off";
      document.getElementById("breaker").className = "text-3xl font-semibold mt-1 " + (s.breaker_active ? "text-rose-300" : "text-teal-300");
      document.getElementById("breaker-detail").textContent = `Loss streak ${s.loss_streak || 0}` + (s.breaker_active ? ` · cooldown ${s.breaker_bars || 0} bars` : "");

      const pairRows = m.pairs || [];
      document.getElementById("mkt-meta").textContent = [m.exchange, m.timeframe, m.as_of || ""].filter(Boolean).join(" · ");
      const err = document.getElementById("mkt-error");
      if (m.error) { err.textContent = m.error; err.classList.remove("hidden"); }
      else { err.classList.add("hidden"); }
      const metrics = document.getElementById("metrics");
      if (!pairRows.length) {
        metrics.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-slate-500">Waiting for candles</td></tr>`;
      } else {
        metrics.innerHTML = pairRows.map(row => {
          const errNote = row.error ? `<div class="text-[11px] text-rose-300 mt-1">${row.error}</div>` : "";
          const regime = row.macro_regime || "CHOP";
          const regimeTone = {BULL:"green", BEAR:"red", CHOP:"amber", NEUTRAL:"amber"}[regime] || "slate";
          return `<tr class="hover:bg-white/[0.02]">
            <td class="pr-4 py-2.5 font-medium text-slate-200 whitespace-nowrap">${row.symbol || "—"}${errNote}</td>
            <td class="px-3 py-2.5">${badge(regimeTone, regime)}</td>
            <td class="px-3 py-2.5 mono text-right">${fmt(row.adx, 1)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.last_close)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.ema_fast)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.ema_slow)}</td>
            <td class="px-3 py-2.5 mono text-right">${px(row.atr)}</td>
          </tr>`;
        }).join("");
      }

      const body = document.getElementById("trades");
      const rows = (data.trades && data.trades.trades) || [];
      document.getElementById("trade-meta").textContent =
        `${data.trades && data.trades.count || 0} fills · ${data.rejects || 0} rejects · showing ${rows.length}`;
      if (!rows.length) {
        body.innerHTML = `<tr><td colspan="7" class="px-4 py-8 text-center text-slate-500">No paper trades yet</td></tr>`;
      } else {
        body.innerHTML = rows.map(t => {
          const verdictTone = t.verdict === "REJECTED" || t.hit === "stop" ? "red" : (t.tone || "slate");
          const actionTone = t.action === "BUY" ? "green" : t.action === "SELL" ? "red" : (t.hit === "target" ? "green" : t.hit === "stop" ? "red" : "slate");
          const rationale = (t.rationale || t.alert_reason || "—").slice(0, 160);
          return `<tr class="hover:bg-white/[0.02]">
            <td class="px-4 py-2.5 mono text-xs text-slate-400 whitespace-nowrap">${(t.timestamp||"").replace("T"," ").replace("+00:00"," UTC")}</td>
            <td class="px-4 py-2.5 mono text-xs text-slate-200 whitespace-nowrap">${t.symbol || "—"}</td>
            <td class="px-4 py-2.5">${badge(actionTone, t.action || "—")}</td>
            <td class="px-4 py-2.5">${badge(verdictTone, t.verdict || "—")}${t.hit ? " " + badge(t.hit === "stop" ? "red" : "green", t.hit === "stop" ? "STOP_LOSS" : "PROFIT") : ""}</td>
            <td class="px-4 py-2.5 mono text-right">${fmt(t.entry_price, 1)}</td>
            <td class="px-4 py-2.5 mono text-right">${fmt(t.exit_price, 1)}</td>
            <td class="px-4 py-2.5 text-slate-400 max-w-xl">${rationale}</td>
          </tr>`;
        }).join("");
      }

      document.getElementById("updated").textContent = (data.generated_at || "").replace("T", " ").replace("+00:00", " UTC");
    }

    async function tick() {
      try {
        const token = new URLSearchParams(location.search).get("token") || "";
        const qs = token ? ("?token=" + encodeURIComponent(token)) : "";
        const res = await fetch("/api/snapshot" + qs, { cache: "no-store" });
        if (res.status === 401) throw new Error("add ?token= from DASHBOARD_SECRET or WEBHOOK_SECRET");
        if (!res.ok) throw new Error("HTTP " + res.status);
        render(await res.json());
        document.getElementById("status").textContent = "live";
        document.getElementById("status").className = "text-xs mt-1 text-teal-400";
      } catch (e) {
        document.getElementById("status").textContent = "poll failed: " + e.message;
        document.getElementById("status").className = "text-xs mt-1 text-rose-300";
      }
      remaining = 10;
    }

    setInterval(() => {
      remaining = Math.max(0, remaining - 1);
      document.getElementById("countdown").textContent = String(remaining);
    }, 1000);
    setInterval(tick, POLL_MS);
    tick();
  </script>
</body>
</html>
"""

app = FastAPI(title="Gemini Trader dashboard", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, token: str | None = Query(default=None)) -> HTMLResponse:
    check_token(request, token)
    return HTMLResponse(PAGE)


@app.get("/api/snapshot")
def snapshot(request: Request, token: str | None = Query(default=None)) -> dict[str, Any]:
    check_token(request, token)
    return build_snapshot()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Read-only paper-trading dashboard")
    parser.add_argument("--host", default=DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    args = parser.parse_args(argv)

    import uvicorn

    enable_os_trust_store()
    secret = dashboard_secret()
    auth = "token required (?token=)" if secret else "OPEN — set DASHBOARD_SECRET before tunneling"
    print(f"Dashboard on http://{args.host}:{args.port}  (read-only, {auth})")
    if secret:
        print(f"  local bookmark : http://127.0.0.1:{args.port}/?token=<DASHBOARD_SECRET or WEBHOOK_SECRET>")
        print(f"  phone          : cloudflared tunnel --url http://localhost:{args.port}")
        print("                  then open https://<printed-host>/?token=<same secret>")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
