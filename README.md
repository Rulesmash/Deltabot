# DeltaRL Trader

> 🛡️ **Risk-Mitigation-First RL crypto trading bot for Delta Exchange India**  
> Built with FastAPI + Stable-Baselines3 PPO + Next.js 14  
> v2 — Capital preservation and risk-adjusted returns over pure profit chasing

---

## ⚠️ Safety Warning

- **Always start in DEMO / Simulation mode first**
- **Never risk more than you can afford to lose**
- Crypto perpetual futures carry extreme risk due to leverage
- The RL agent is experimental — past demo performance does NOT guarantee live results

## Why Risk-Mitigation-First?

Most RL trading research optimises for raw P&L. This is dangerous in production because:

- **Leverage amplifies both wins AND losses** — a purely profit-chasing agent will find high-leverage strategies that blow up in real markets.
- **Reward hacking** — naive P&L rewards incentivise overtrading and ignoring fees.
- **Distribution shift** — a model trained on bull markets will fail catastrophically in volatile or bear regimes.

### Our approach (inspired by 2025–2026 RL-for-trading literature)

The agent is explicitly rewarded for *how it makes money*, not just *how much*:

```
reward = (realized_pnl - fees·1.5x - slippage)    ← core P&L
         + 0.15 × Sharpe                            ← rolling risk-adjusted bonus
         + 0.10 × Sortino                           ← downside-only adj. bonus
         + 0.05 × Calmar                            ← return / max_drawdown ratio
         - 3.0  × max(0, trade_dd - 5%)²            ← HEAVY above 5% per-trade DD
         - 5.0  × max(0, global_dd - 15%)²          ← EMERGENCY brake above 15%
         - volatility_penalty                       ← high leverage in high-vol
         - funding_bleed_penalty                    ← adverse 8h funding cost
         - 10.0 × liquidation_penalty               ← near-liq crisis penalty
         - flat_penalty                             ← tiny anti-lazy nudge
```

Key design choices:
- **Quadratic drawdown penalties**: small violations are lightly penalised; large ones are catastrophically penalised, teaching the agent to never approach those thresholds.
- **No penalty for staying flat in high-vol regimes**: the agent is explicitly rewarded for doing nothing when it would be dangerous to trade.
- **Safety policy baseline**: if portfolio drawdown exceeds 10% OR volatility regime > 85%, a conservative rule-based policy overrides PPO.

### Curriculum Learning

The agent starts on the easiest data and progressively learns harder regimes:

| Stage | Data | Volatility Regime | Gate to Advance |
|-------|------|-------------------|------------------|
| 0 | Low-volatility periods only | ATR < 33rd percentile | Sharpe > 0.5  AND  MaxDD < 8% (over 50 eps) |
| 1 | Low + medium volatility | ATR < 66th percentile | Sharpe > 1.0  AND  MaxDD < 12% (over 50 eps) |
| 2 | Full data (all regimes) | All bars | Final evaluation |

---

## Architecture

```
┌─────────────────┐     WebSocket/REST     ┌──────────────────────┐
│  Next.js 14     │◄──────────────────────►│  FastAPI Backend      │
│  Dashboard      │                        │  + RL Engine (SB3)   │
│  :3000          │                        │  :8000               │
└─────────────────┘                        └──────────┬───────────┘
                                                      │
                        ┌─────────────────────────────┤
                        │                             │
               ┌────────▼──────┐          ┌──────────▼──────────┐
               │  SQLite DB    │          │  Delta Exchange      │
               │  (trades,     │          │  India API           │
               │   models,     │          │  Testnet / Prod      │
               │   config)     │          └─────────────────────┘
               └───────────────┘
```

                        action[1]: leverage  (1x – 20x)
                        action[2]: SL %      (0.3% – 5.0%)
                        action[3]: TP %      (0.5% – 10.0%)
                                                    │
Reward = P&L - fees·1.5x - slippage + 0.15·Sharpe - 2.0·Drawdown - flat_penalty
```

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Docker Desktop | 4.x+ |
| Python | 3.11+ (local dev) |
| Node.js | 20+ (local dev) |

### 1. Clone and configure

```bash
git clone <your-repo>
cd DeltaBot

# Copy and edit the environment variables
cp .env.example .env
```

Open `.env` and at minimum set:
```bash
DEMO_API_KEY=your_testnet_key
DEMO_API_SECRET=your_testnet_secret
TRADING_MODE=demo
SIMULATION_ONLY=true   # safest starting mode — no orders placed
```

### 2. Get Demo (Testnet) API Keys

1. Go to **[testnet.delta.exchange](https://testnet.delta.exchange)** (India testnet)
2. Create a free account
3. Navigate to **Profile → API Keys → Create New API Key**
4. Enable **Trading** permissions
5. Copy the key and secret into your `.env` file

### 3. Run with Docker

```bash
docker-compose up --build
```

Services:
| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Redis | localhost:6379 |

### 4. Run without Docker (Development)

**Backend:**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: .\venv\Scripts\activate   # Windows

# Install TA-Lib (Windows: see note below)
pip install -r requirements.txt

# Start the server
python main.py
```

> **Windows TA-Lib install:**  
> Download the pre-built wheel from [github.com/cgohlke/talib-build](https://github.com/cgohlke/talib-build/releases)  
> `pip install TA_Lib-0.4.xx-cp311-cp311-win_amd64.whl`

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## Training Workflow

### Phase 1: Simulation Training (Recommended start)

1. Open **http://localhost:3000**
2. Go to **Training** tab
3. Select **SIMULATION** mode
4. Set steps to **200,000**
5. Click **Start Training**
6. Monitor equity curve and reward chart
7. Export the model when satisfied

In simulation mode:
- No orders are placed with the exchange
- P&L is calculated from real market price movement
- Fastest training (no API latency)
- Completely safe — no funds of any kind involved

### Phase 2: Demo Training (Testnet)

1. Ensure demo API keys are configured
2. Select **DEMO** mode in Training tab
3. Run another 100k–200k steps
4. Real virtual orders are placed on the testnet
5. You get real market feedback with virtual money

### Phase 3: Backtesting

1. Go to **Backtest** tab
2. Select symbol and lookback bars
3. Click **Run Backtest**
4. Analyse metrics:
   - Sharpe Ratio > 1.0 ✅
   - Max Drawdown < 15% ✅
   - Win Rate > 45% ✅
   - Profit Factor > 1.3 ✅

### Phase 4: Live Trading (Only after thorough testing)

> [!WARNING]  
> Only proceed to Live mode after completing all above phases and checking the safety checklist in the Config tab.

1. Set `LIVE_API_KEY` and `LIVE_API_SECRET` in `.env`
2. Set `TRADING_MODE=live` in `.env`
3. Restart the backend
4. In the dashboard header, click the **DEMO** toggle
5. Confirm the risk disclaimer in the modal
6. Start with low position sizing (`MAX_RISK_PER_TRADE=0.002`)

---

## Project Structure

```
DeltaBot/
├── backend/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── config.py                # All settings via pydantic-settings
│   ├── delta/
│   │   ├── client.py            # HMAC-signed REST client
│   │   ├── websocket_client.py  # Async WS with auto-reconnect
│   │   └── endpoints.py         # URL and channel constants
│   ├── data/
│   │   ├── features.py          # 35 TA features (pandas_ta + fallbacks)
│   │   ├── fetcher.py           # Paginated historical OHLCV fetcher
│   │   └── stream.py            # Real-time candle buffer
│   ├── rl/
│   │   ├── env.py               # Gymnasium env (49-dim obs, Box(5) actions, curriculum)
│   │   ├── agent.py             # PPO [256,256] + safety policy baseline
│   │   ├── reward.py            # Risk-first: Sharpe+Sortino+Calmar-DD_penalty-LiqPenalty
│   │   ├── train.py             # Curriculum orchestrator with auto-advancement gates
│   │   └── backtest.py          # Vectorized backtesting engine
│   ├── trading/
│   │   ├── executor.py          # Bracket order execution (SL+TP)
│   │   └── circuit_breaker.py   # Drawdown auto-pause
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM (trades, checkpoints)
│   │   └── database.py          # SQLite async + sync engines
│   ├── api/
│   │   ├── routes_rl.py         # /api/rl/* endpoints
│   │   ├── routes_combined.py   # /api/trading, /api/data, /api/config
│   │   └── websocket_handler.py # Frontend WS broadcast hub
│   ├── alerts/notifier.py       # Telegram + Discord alerts
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx             # Full dashboard (4 tabs)
│   │   ├── layout.tsx           # Root layout + SEO
│   │   └── globals.css          # Premium dark theme CSS
│   ├── tailwind.config.js       # Trading terminal palette
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## API Reference

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + mode info |
| `POST` | `/api/rl/train/start` | Start training (with optional `curriculum_stage`) |
| `POST` | `/api/rl/train/stop` | Stop training |
| `GET` | `/api/rl/status` | Training metrics + Sharpe/Sortino/Calmar/MaxDD |
| `POST` | `/api/rl/backtest` | Run historical backtest |
| `GET` | `/api/rl/model/export/{version}` | Download model ZIP |
| `POST` | `/api/rl/model/import` | Upload model ZIP |
| `POST` | `/api/rl/circuit-breaker/resume` | Resume after pause |
| `POST` | `/api/rl/curriculum/stage` | Set curriculum stage (0/1/2) |
| `GET` | `/api/rl/curriculum/status` | Current stage, gates, avg Sharpe/DD |
| `POST` | `/api/rl/reward/weights` | Hot-swap reward weights at runtime |
| `GET` | `/api/rl/reward/weights` | Read current reward configuration |
| `GET` | `/api/trading/positions` | Open positions |
| `GET` | `/api/trading/balance` | Wallet balance |
| `GET` | `/api/data/candles/{symbol}` | OHLCV candles |
| `GET` | `/api/config/settings` | Current config |
| `WS` | `/ws` | Real-time event stream (equity, risk metrics, curriculum) |

Full interactive docs at **http://localhost:8000/docs**

---

## Configuration Reference

Key variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_MODE` | `demo` | `demo` or `live` |
| `SIMULATION_ONLY` | `true` | No orders placed if true |
| `MAX_LEVERAGE` | `20` | Max leverage (1–20x) |
| `MAX_RISK_PER_TRADE` | `0.01` | 1% equity risk per trade |
| `CIRCUIT_BREAKER_DRAWDOWN` | `0.05` | Pause at 5% drawdown |
| `RL_LEARNING_RATE` | `3e-4` | PPO learning rate |
| `RL_N_STEPS` | `2048` | Steps before PPO update |
| `FINE_TUNE_EVERY_N_TRADES` | `5` | Online update frequency |
| `TRADING_PAIRS` | `BTCUSDT,ETHUSDT` | Comma-separated pairs |
| `TELEGRAM_BOT_TOKEN` | `""` | Optional alerts |
| `DISCORD_WEBHOOK_URL` | `""` | Optional alerts |

---

## Safety Checklist

Before going live, ensure:

- [ ] Completed 200k+ steps in Simulation mode (Stage 0 → Stage 2 curriculum)
- [ ] Rolling Sharpe > 1.0 over last 50 training episodes  
- [ ] Max Drawdown (episode) < 12% during Stage 2 training
- [ ] Calmar ratio > 0.5 (return / max_drawdown)
- [ ] Completed 50k+ steps in Demo (Testnet) mode
- [ ] Backtested on 1000+ bars with Sharpe > 1.0, MaxDD < 15%
- [ ] Live API key has **Trading only** permissions (no withdrawal)
- [ ] IP whitelist enabled on live API key
- [ ] Circuit breaker set to 5% or less
- [ ] Telegram/Discord alerts configured
- [ ] Start with `MAX_RISK_PER_TRADE=0.002` (0.2%) for first live run
- [ ] Monitoring plan in place (check Risk Panel in dashboard regularly)
- [ ] `SAFETY_DD_THRESHOLD` reviewed — default 10% triggers safety policy

### Monitoring Risk During Training

Watch these key indicators in the **Dashboard → Risk Panel**:

| Indicator | Target | Action if Violated |
|-----------|--------|--------------------|
| **Max DD (episode)** | < 5% (Stage 0), < 12% (Stage 2) | Increase `trade_dd_weight` via `/api/rl/reward/weights` |
| **Volatility Regime** | Should not stay at "High" for long | Check if agent is overtrading in volatile windows |
| **Sharpe** | > 0.5 (Stage 0), > 1.0 (eventually) | Curriculum won't advance until this gate is passed |
| **Sortino** | Should be > Sharpe | Agent respects downside asymmetry |
| **Funding Exposure** | < 0.01% per 8h | Reduce leverage weight if funding is consistently adverse |
| **Safety Zone** badge | Should rarely appear | Investigate training data if this shows frequently |

---

## License

MIT — Use at your own risk. This is experimental software. No warranty is provided.
