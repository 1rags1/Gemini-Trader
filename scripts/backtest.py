"""Full-history backtest of Gemini Trend Guard on Kraken 1h BTC/USDT.

Pulls every 1h candle Kraken will serve, computes the same indicators the live
runner uses, and walks closed bars with the EMA 21/55 crossover gated by EMA 200
and ADX, plus the 1.5x / 3.75x / 2.75-arm / 1.75-trail ATR exits. Gemini is not
consulted.

    python scripts/backtest.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.market_data import OHLCV_COLUMNS, _build_exchange, add_indicators
from core.net import enable_os_trust_store
from core.strategy_runner import (
    Position,
    RunnerState,
    apply_breaker,
    close_pnl,
    detect_signal,
    last_closed_ts,
    maybe_exit,
    position_qty,
)
from strategies import load_strategy

STARTING_EQUITY = 100.0


@dataclass
class ClosedTrade:
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    pnl: float
    hit: str


def fetch_all_ohlcv(symbol: str, timeframe: str, exchange_id: str) -> pd.DataFrame:
    """Download every 1h candle Kraken's public OHLC endpoint will serve.

    That endpoint caps each request at 720 bars and does not offer a deeper
    archive: a `since` far in the past still returns the most recent 720 hours.
    720 1h candles is therefore the maximum available history via ccxt/Kraken.
    """
    enable_os_trust_store()
    exchange = _build_exchange(exchange_id)
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=720)
    if not rows:
        raise RuntimeError(f"No candles returned for {symbol} @ {timeframe} on {exchange_id}")

    df = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").astype(float).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.attrs.update(symbol=symbol, timeframe=timeframe, exchange=exchange.id)
    print(
        f"  {len(df)} candles  {df.index[0]} -> {df.index[-1]}"
        "  (Kraken public OHLC max is 720 1h bars)",
        flush=True,
    )
    return df


def max_drawdown_pct(equity: list[float]) -> float:
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd * 100.0


def first_valid_index(df: pd.DataFrame) -> int:
    """First bar where EMA 200, ADX, and ATR are all defined."""
    needed = ("ema_fast", "ema_slow", "ema_macro", "adx", "atr")
    for i in range(len(df)):
        row = df.iloc[i]
        if all(col in row.index and pd.notna(row[col]) for col in needed):
            return max(i, 1)
    raise RuntimeError("Indicators never became valid — not enough candles for EMA 200")


def run_backtest(df: pd.DataFrame, params: dict, starting_equity: float) -> dict:
    closed_ts = last_closed_ts(df, str(df.attrs.get("timeframe") or "1h"))
    closed = df.loc[:closed_ts]
    start_i = first_valid_index(closed)

    state = RunnerState(equity=starting_equity)
    position: Position | None = None
    trades: list[ClosedTrade] = []
    equity_curve = [starting_equity]
    commission = float(params.get("commission_pct", 0.075))
    adx_min = float(params.get("adx_min", 20))

    def flatten(bar: pd.Series, price: float, hit: str) -> None:
        nonlocal position, state
        assert position is not None
        pnl = close_pnl(position, price, commission)
        state.equity += pnl
        state.loss_streak = state.loss_streak + 1 if pnl < 0 else 0
        trades.append(
            ClosedTrade(
                side=position.side,
                entry_time=position.opened_bar,
                exit_time=bar.name.isoformat() if hasattr(bar.name, "isoformat") else str(bar.name),
                entry=position.entry_price,
                exit=price,
                pnl=pnl,
                hit=hit,
            )
        )
        position = None
        state.position = None

    for i in range(start_i, len(closed)):
        bar = closed.iloc[i]
        hist = closed.iloc[: i + 1]
        atr = float(bar["atr"]) if pd.notna(bar.get("atr")) else None

        if position is not None and position.opened_bar != bar.name.isoformat():
            if atr is not None:
                event = maybe_exit(
                    position,
                    high=float(bar["high"]),
                    low=float(bar["low"]),
                    close=float(bar["close"]),
                    atr=atr,
                    params=params,
                )
                if event is not None:
                    flatten(bar, event.price, event.hit)

        state.equity_history.append(state.equity)
        lookback = int(params.get("equity_lookback_bars", 500))
        if len(state.equity_history) > lookback:
            state.equity_history = state.equity_history[-lookback:]
        trend_strong = pd.notna(bar.get("adx")) and float(bar["adx"]) > adx_min
        apply_breaker(state, params, trend_strong, new_bar=True)

        if position is None:
            signal = detect_signal(hist, params, state.breaker_active)
            if signal is not None:
                qty = position_qty(state.equity, signal.price, signal.atr, params)
                side = "LONG" if signal.action == "BUY" else "SHORT"
                position = Position(
                    side=side,
                    entry_price=signal.price,
                    entry_atr=signal.atr,
                    qty=qty,
                    stop=signal.stop,
                    target=signal.target,
                    trail_stop=signal.stop,
                    trail_armed=False,
                    opened_bar=bar.name.isoformat(),
                    confidence=1.0,
                    model=None,
                    rationale="backtest",
                    reason=signal.reason,
                )

        equity_curve.append(state.equity)

    if position is not None:
        last = closed.iloc[-1]
        flatten(last, float(last["close"]), "eod")
        equity_curve.append(state.equity)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    n = len(trades)
    if n == 0:
        win_rate = 0.0
        profit_factor = 0.0
    else:
        win_rate = len(wins) / n * 100.0
        profit_factor = float("inf") if gross_loss == 0 else gross_win / gross_loss

    return {
        "bars": len(closed) - start_i,
        "from": closed.index[start_i].isoformat(),
        "to": closed.index[-1].isoformat(),
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown_pct(equity_curve),
        "start_equity": starting_equity,
        "end_equity": state.equity,
        "net_pnl": state.equity - starting_equity,
        "closed": trades,
    }


def format_report(result: dict, symbol: str, exchange: str, candles: int) -> str:
    pf = result["profit_factor"]
    pf_text = "inf" if pf == float("inf") else f"{pf:.3f}"
    return "\n".join(
        [
            f"Backtest     {symbol}  1h  @{exchange}",
            f"Candles      {candles} fetched (Kraken public OHLC max), {result['bars']} after EMA 200 warmup",
            f"Window       {result['from']} -> {result['to']}",
            f"Rules        EMA 21/55 cross, close vs EMA 200, ADX>20, 1.5x ATR stop / 3.75x target, trail 2.75/1.75",
            f"Account      ${result['start_equity']:.2f}",
            "",
            f"Total Trades : {result['trades']}  ({result['wins']} wins / {result['losses']} losses)",
            f"Win Rate %   : {result['win_rate']:.2f}%",
            f"Profit Factor: {pf_text}",
            f"Net PnL      : ${result['net_pnl']:+.2f}  on a ${result['start_equity']:.0f} account",
            f"Max Drawdown : {result['max_drawdown']:.2f}%",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-history Kraken 1h backtest of ema_atr_trend")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--exchange", default="kraken")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--equity", type=float, default=STARTING_EQUITY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    enable_os_trust_store()
    params = load_strategy("ema_atr_trend").params
    print(f"Fetching all available {args.timeframe} {args.symbol} candles from {args.exchange} ...", flush=True)
    raw = fetch_all_ohlcv(args.symbol, args.timeframe, args.exchange)
    print(f"Computing EMA 21/55/200, ADX 14, ATR 14 on {len(raw)} bars ...", flush=True)
    frame = add_indicators(raw)
    result = run_backtest(frame, params, args.equity)
    print(format_report(result, args.symbol, args.exchange, len(raw)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
