# Gemini Trader

Gemini Trader is a small Kraken spot bot with a Gemini check on each entry and a read-only dashboard.

The native runner watches BTC, ETH, and SOL. It uses the 1 hour trend as a gate and a 15 minute EMA pullback as the trigger. Gemini must agree before a new long is taken. Stops and targets are multiples of the 15 minute ATR. The book is spot and long-only: at most two positions, and never ETH and SOL at the same time.

A separate TradingView webhook can review Pine alerts and write them to a CSV. That path does not send exchange orders.

## Paper mode is the normal path

Practice mode is the default. Copy `.env.example` to `.env` and it stays on the paper book (`PAPER_TRADING=true`).

Real Kraken orders are sent only when **both** of these are set:

- `PAPER_TRADING=false`
- `ALLOW_LIVE_TRADING=true`

If paper mode is turned off and `ALLOW_LIVE_TRADING` is still false, the runner stops at startup and places nothing. `ALLOW_LIVE_TRADING=true` does not override paper mode. While paper mode is on, no exchange order is sent.

Live entries are post-only limit orders. An accepted order is not treated as an open position. The runner stores the order id and opens the local slot only after the exchange reports a fill. A cancel or expiry clears the pending order and leaves the slot flat. Market exits are sent when a stop, target, or regime flip closes a position that is already open.

Each new position uses `POSITION_SIZE_FRACTION` of equity (default `0.25`). With `MAX_OPEN_POSITIONS=2`, two full slots use about half of equity. The rest stays in cash for fees and slippage.

Gemini is told whether the process is in paper or live mode. That text is only context. Position size, the circuit breaker, the altcoin cap, spot long-only, and post-only entries are enforced in code. The model cannot override them.

## How to run

From the repo root, on Windows or Linux:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Linux or macOS:

```bash
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put your Gemini key in `.env`. Leave the live flags as they are until you mean to trade real funds.

Native runner (paper unless you open both live gates):

```bash
python -m core.strategy_runner --once
python -m core.strategy_runner
```

On Windows, if you use the venv interpreter:

```powershell
.\.venv\Scripts\python.exe -m core.strategy_runner --once
```

Dashboard (read-only, localhost):

```bash
python -m core.dashboard
```

Open `http://127.0.0.1:8050`. The page shows two books at once. **PRACTICE / PAPER** is fake money and starts at `PAPER_STARTING_BALANCE` (default $10,000). **LIVE / KRAKEN** is the real Kraken USD balance. With the default flags the banner says **ACTIVE: PAPER TRADING**, the practice panel is marked in use, and the Kraken panel stays idle even if a last-known balance is on disk. The dashboard never sends an order.

To listen on another interface, set `DASHBOARD_SECRET` (or `WEBHOOK_SECRET`) and start with `--host 0.0.0.0` or `DASHBOARD_HOST`. Without that secret the process refuses a public bind. A Cloudflare tunnel also requires the secret, including when the bind itself is localhost.

TradingView webhook (paper log only):

```bash
python -m core.webhook_server
```

`WEBHOOK_HOST` defaults to `127.0.0.1`. An empty `WEBHOOK_SECRET` accepts alerts only from localhost. Any other bind, or a request that arrives through a tunnel, requires `WEBHOOK_SECRET`. The server exits instead of listening publicly without one.

On Windows, `run_bot.ps1` starts that webhook and a Cloudflare quick tunnel, and it generates `WEBHOOK_SECRET` if you do not have one:

```powershell
.\run_bot.ps1
.\run_bot.ps1 -Stop
```

## Which process can spend money

| Process | What it does | Can it send Kraken orders? |
| --- | --- | --- |
| `python -m core.strategy_runner` | Scans Kraken candles, asks Gemini, manages stops | Yes, only when paper mode is off and `ALLOW_LIVE_TRADING=true` |
| `python -m core.webhook_server` and `run_bot.ps1` | Reviews TradingView alerts and appends a CSV | No. Confirmed alerts are paper logs |
| `python -m core.dashboard` | Reads the state file and the trade log | No |

## Secrets

Do not commit any of these:

- `.env`
- `GEMINI_API_KEY`
- `EXCHANGE_API_KEY` and `EXCHANGE_API_SECRET`
- `WEBHOOK_SECRET` and `DASHBOARD_SECRET`
- `state/runner.json` and the trade CSVs (they are runtime state)

`.env.example` is the only env file that belongs in git. `.gitignore` already excludes `.env` and `state/`.
