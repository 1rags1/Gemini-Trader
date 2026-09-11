"""Read-only localhost dashboard for the paper-trading runner.

Serves a single Tailwind page on :8050 and a JSON snapshot the browser polls
every 10 seconds. File reads are short-lived and shared, so this process never
locks `state/runner.json` or `data/paper_trades.csv` away from the runner.

    python -m core.dashboard
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from core.config import PROJECT_ROOT, get_settings
from core.market_data import add_indicators, fetch_ohlcv, latest_snapshot
from core.net import enable_os_trust_store

log = logging.getLogger("dashboard")

DASHBOARD_PORT = 8050
TRADE_LIMIT = 15
MARKET_CACHE_TTL = 20.0
MARKET_CANDLES = 300

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


def read_runner_state() -> dict[str, Any]:
    path = _paths()["state"]
    empty = {
        "exists": False,
        "equity": None,
        "starting_equity": None,
        "pnl": None,
        "pnl_pct": None,
        "loss_streak": 0,
        "breaker_active": False,
        "breaker_bars": 0,
        "last_bar": None,
        "position_side": "FLAT",
        "position": None,
    }
    if not path.exists():
        return empty
    try:
        raw = json.loads(_read_shared(path))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("state unreadable: %s", exc)
        return {**empty, "error": str(exc)}

    cfg = _cfg()
    starting = float(cfg.paper_starting_balance) if cfg is not None else 10_000.0
    equity = raw.get("equity")
    try:
        equity_f = float(equity) if equity is not None else None
    except (TypeError, ValueError):
        equity_f = None
    pnl = None if equity_f is None else equity_f - starting
    pos = raw.get("position")
    side = "FLAT"
    if isinstance(pos, dict) and pos.get("side"):
        side = str(pos["side"]).upper()

    return {
        "exists": True,
        "equity": equity_f,
        "starting_equity": starting,
        "pnl": pnl,
        "pnl_pct": None if pnl is None or starting == 0 else (pnl / starting) * 100.0,
        "loss_streak": int(raw.get("loss_streak") or 0),
        "breaker_active": bool(raw.get("breaker_active")),
        "breaker_bars": int(raw.get("breaker_bars") or 0),
        "last_bar": raw.get("last_bar"),
        "position_side": side,
        "position": pos if isinstance(pos, dict) else None,
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


def fetch_market() -> dict[str, Any]:
    """Latest BTC snapshot, cached so a 10s browser poll does not hammer Kraken."""
    now = time.monotonic()
    with _market_lock:
        age = now - float(_market_cache["fetched_at"])
        if _market_cache["payload"] is not None and age < MARKET_CACHE_TTL:
            return _market_cache["payload"]

    cfg = _cfg()
    symbol = cfg.symbol if cfg is not None else "BTC/USDT"
    timeframe = cfg.timeframe if cfg is not None else "1h"
    exchange = cfg.exchange_id if cfg is not None else "kraken"

    try:
        enable_os_trust_store()
        frame = add_indicators(
            fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=MARKET_CANDLES, exchange_id=exchange)
        )
        snap = latest_snapshot(frame)
        indicators = snap.get("indicators") or {}
        payload = {
            "ok": True,
            "symbol": snap.get("symbol") or symbol,
            "timeframe": snap.get("timeframe") or timeframe,
            "exchange": exchange,
            "as_of": snap.get("as_of"),
            "last_close": snap.get("last_close"),
            "regime": snap.get("regime") or {},
            "ema_fast": indicators.get("ema_fast"),
            "ema_slow": indicators.get("ema_slow"),
            "ema_macro": indicators.get("ema_macro"),
            "rsi": indicators.get("rsi"),
            "adx": indicators.get("adx"),
            "atr": indicators.get("atr"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("market fetch failed: %s", exc)
        payload = {
            "ok": False,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": exchange,
            "as_of": None,
            "last_close": None,
            "regime": {},
            "ema_fast": None,
            "ema_slow": None,
            "ema_macro": None,
            "rsi": None,
            "adx": None,
            "atr": None,
            "error": f"{type(exc).__name__}: {exc}",
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

    <section class="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
      <article class="bg-panel rounded-xl border border-white/5 p-4">
        <p class="text-xs uppercase tracking-wider text-slate-400">Position</p>
        <p id="pos-side" class="text-3xl font-semibold mt-1">—</p>
        <p id="pos-detail" class="text-xs text-slate-400 mt-2 leading-relaxed">No open position</p>
      </article>
      <article class="bg-panel rounded-xl border border-white/5 p-4">
        <p class="text-xs uppercase tracking-wider text-slate-400">Paper equity</p>
        <p id="equity" class="text-3xl font-semibold mt-1 mono">—</p>
        <p id="pnl" class="text-sm mt-2">—</p>
      </article>
      <article class="bg-panel rounded-xl border border-white/5 p-4">
        <p class="text-xs uppercase tracking-wider text-slate-400">Regime</p>
        <p id="regime" class="text-3xl font-semibold mt-1">—</p>
        <p id="regime-detail" class="text-xs text-slate-400 mt-2 leading-relaxed">Waiting for candles</p>
      </article>
      <article class="bg-panel rounded-xl border border-white/5 p-4">
        <p class="text-xs uppercase tracking-wider text-slate-400">Circuit breaker</p>
        <p id="breaker" class="text-3xl font-semibold mt-1">—</p>
        <p id="breaker-detail" class="text-xs text-slate-400 mt-2">Loss streak —</p>
      </article>
    </section>

    <section class="bg-panel rounded-xl border border-white/5 p-4 mb-4">
      <div class="flex items-baseline justify-between mb-3">
        <h2 class="text-sm uppercase tracking-wider text-slate-400">Live BTC / USDT</h2>
        <p id="mkt-meta" class="text-xs text-slate-500 mono"></p>
      </div>
      <p id="mkt-error" class="hidden text-sm text-rose-300 mb-3"></p>
      <div class="grid grid-cols-2 md:grid-cols-6 gap-3">
        <div><p class="text-xs text-slate-500">Close</p><p id="ind-close" class="mono text-lg">—</p></div>
        <div><p class="text-xs text-slate-500">EMA 21</p><p id="ind-fast" class="mono text-lg">—</p></div>
        <div><p class="text-xs text-slate-500">EMA 55</p><p id="ind-slow" class="mono text-lg">—</p></div>
        <div><p class="text-xs text-slate-500">EMA 200</p><p id="ind-macro" class="mono text-lg">—</p></div>
        <div><p class="text-xs text-slate-500">RSI 14</p><p id="ind-rsi" class="mono text-lg">—</p></div>
        <div><p class="text-xs text-slate-500">ADX 14</p><p id="ind-adx" class="mono text-lg">—</p></div>
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

    function render(data) {
      const s = data.state || {};
      const m = data.market || {};
      const side = s.position_side || "FLAT";
      document.getElementById("pos-side").textContent = side;
      document.getElementById("pos-side").className = "text-3xl font-semibold mt-1 " + tonePos(side);
      const p = s.position;
      document.getElementById("pos-detail").textContent = p
        ? `Entry ${fmt(p.entry_price, 1)} · stop ${fmt(p.trail_stop || p.stop, 1)} · target ${fmt(p.target, 1)}${p.trail_armed ? " · trail armed" : ""}`
        : (s.last_bar ? `Last closed bar ${s.last_bar}` : "No open position");

      document.getElementById("equity").textContent = s.equity == null ? "—" : fmt(s.equity, 2);
      const pnl = s.pnl;
      const pnlEl = document.getElementById("pnl");
      if (pnl == null) { pnlEl.textContent = "Starting balance unknown"; pnlEl.className = "text-sm mt-2 text-slate-400"; }
      else {
        const sign = pnl >= 0 ? "+" : "";
        pnlEl.textContent = `${sign}${fmt(pnl, 2)}  (${sign}${fmt(s.pnl_pct, 2)}%) vs ${fmt(s.starting_equity, 0)} start`;
        pnlEl.className = "text-sm mt-2 " + (pnl >= 0 ? "text-teal-300" : "text-rose-300");
      }

      const reg = m.regime || {};
      const dir = reg.tradeable_direction || "unknown";
      document.getElementById("regime").textContent = (reg.macro_trend || "—").toUpperCase();
      document.getElementById("regime").className = "text-3xl font-semibold mt-1 " + (reg.macro_trend === "bull" ? "text-teal-300" : reg.macro_trend === "bear" ? "text-rose-300" : "text-slate-200");
      document.getElementById("regime-detail").textContent =
        `${reg.trend_strength || "—"} · ADX ${fmt(reg.adx, 1)} · EMA200 ${fmt(reg.macro_ema, 1)} · ${fmt(reg.price_vs_macro_ema_pct, 2)}% vs macro · ${dir.replaceAll("_", " ")}`;

      document.getElementById("breaker").textContent = s.breaker_active ? "ON" : "off";
      document.getElementById("breaker").className = "text-3xl font-semibold mt-1 " + (s.breaker_active ? "text-rose-300" : "text-teal-300");
      document.getElementById("breaker-detail").textContent = `Loss streak ${s.loss_streak || 0}` + (s.breaker_active ? ` · cooldown ${s.breaker_bars || 0} bars` : "");

      document.getElementById("ind-close").textContent = fmt(m.last_close, 1);
      document.getElementById("ind-fast").textContent = fmt(m.ema_fast, 1);
      document.getElementById("ind-slow").textContent = fmt(m.ema_slow, 1);
      document.getElementById("ind-macro").textContent = fmt(m.ema_macro, 1);
      document.getElementById("ind-rsi").textContent = fmt(m.rsi, 1);
      document.getElementById("ind-adx").textContent = fmt(m.adx, 1);
      document.getElementById("mkt-meta").textContent = [m.exchange, m.timeframe, m.as_of || ""].filter(Boolean).join(" · ");
      const err = document.getElementById("mkt-error");
      if (m.error) { err.textContent = m.error; err.classList.remove("hidden"); }
      else { err.classList.add("hidden"); }

      const body = document.getElementById("trades");
      const rows = (data.trades && data.trades.trades) || [];
      document.getElementById("trade-meta").textContent =
        `${data.trades && data.trades.count || 0} fills · ${data.rejects || 0} rejects · showing ${rows.length}`;
      if (!rows.length) {
        body.innerHTML = `<tr><td colspan="6" class="px-4 py-8 text-center text-slate-500">No paper trades yet</td></tr>`;
      } else {
        body.innerHTML = rows.map(t => {
          const verdictTone = t.verdict === "REJECTED" || t.hit === "stop" ? "red" : (t.tone || "slate");
          const actionTone = t.action === "BUY" ? "green" : t.action === "SELL" ? "red" : (t.hit === "target" ? "green" : t.hit === "stop" ? "red" : "slate");
          const rationale = (t.rationale || t.alert_reason || "—").slice(0, 160);
          return `<tr class="hover:bg-white/[0.02]">
            <td class="px-4 py-2.5 mono text-xs text-slate-400 whitespace-nowrap">${(t.timestamp||"").replace("T"," ").replace("+00:00"," UTC")}</td>
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
        const res = await fetch("/api/snapshot", { cache: "no-store" });
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
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


@app.get("/api/snapshot")
def snapshot() -> dict[str, Any]:
    return build_snapshot()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Read-only paper-trading dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    args = parser.parse_args(argv)

    import uvicorn

    enable_os_trust_store()
    print(f"Dashboard on http://{args.host}:{args.port}  (read-only, Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
