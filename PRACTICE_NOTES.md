# Practice notes

Paper mode stays on until you choose otherwise. `PAPER_TRADING=true` and `ALLOW_LIVE_TRADING=false` are the defaults. Do not turn live trading on from this note.

## What runs all day

On the VPS, run two processes:

```bash
python -m core.strategy_runner
python -m core.dashboard
```

The runner watches Kraken `BTC/USD`, `ETH/USD`, and `SOL/USD`. The dashboard reads the book. It does not send orders.

The TradingView webhook (`python -m core.webhook_server`) only writes a CSV. It is not the 24/7 trading path.

## USD only

There is one quote currency: USD. A saved `ETH/USDT` position is not a second coin. In paper mode the runner moves it onto `ETH/USD` when that slot is empty, or drops it into `state/quarantine_positions.json` when `ETH/USD` is already open. Live mode does not move a USDT order onto the USD pair.

## $10,000 paper book

The practice book starts at `PAPER_STARTING_BALANCE` (default `$10,000`). Closed paper trades add their P&L to equity.

Reset, after you mean to:

```bash
python -m core.strategy_runner --reset-paper --once
```

That archives `state/runner.json` under `state/archive/` and sets equity and start equity back to the paper balance. It does not delete `live.json`. If you set `RESET_PAPER=true` in `.env`, remove that line after one start.

## Lock the dashboard

Leave the bind at `127.0.0.1`. From your laptop:

```bash
ssh -L 8050:127.0.0.1:8050 user@your-vps
```

Open `http://127.0.0.1:8050`. Binding `0.0.0.0` requires `DASHBOARD_SECRET`. When that secret is set, the page and `/api/snapshot` both require it (`?token=`, `X-Dashboard-Token`, or `Authorization: Bearer`). The server does not print the secret.

## Daily check

```bash
python -m core.practice_summary
python -m core.practice_summary --write
```

The note says whether you are still in paper, how many trades happened today, wins and losses, open positions, and whether Gemini confirmed, rejected, or was unavailable.
