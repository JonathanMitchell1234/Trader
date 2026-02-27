# Swing Trading Bot (Alpaca API)

A fully automated **swing-trading bot** that uses the Alpaca brokerage API to scan for setups, size positions, execute trades, and manage exits — all while **guaranteeing zero day trades** so you never trigger the Pattern Day Trader rule.

---

## How It Works

```
Watchlist (60+ liquid stocks/ETFs)
        │
        ▼
   ┌──────────┐
   │ Screener │  price, volume, and liquidity filters
   └────┬─────┘
        │  candidates
        ▼
   ┌──────────┐
   │ Strategy │  EMA crossover · RSI pullback · MACD · ADX · Volume
   └────┬─────┘
        │  BUY / SELL signals
        ▼
   ┌──────────┐
   │ PDT Guard│  blocks any same-day round trip
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ Risk Mgr │  ATR-based stops · 2% risk per trade · 12% max position
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ Executor │  bracket orders (SL + TP) via Alpaca API
   └──────────┘
```

### Entry Criteria (ALL must be true)
| # | Condition | Purpose |
|---|-----------|---------|
| 1 | Price > EMA-50 | Confirms uptrend |
| 2 | EMA-9 crosses above EMA-21 **or** RSI pulls back to 30-45 zone | Timing |
| 3 | MACD histogram positive / turning | Momentum |
| 4 | Volume ≥ 1.3× 20-day average | Conviction |
| 5 | ADX > 20 | Trend strength |

### Exit Criteria (ANY triggers sell)
- RSI ≥ 70 (overbought)
- EMA-9 crosses below EMA-21
- MACD histogram flips negative
- Price closes below EMA-50
- Bracket stop-loss / take-profit hit

### PDT Protection
The bot enforces a **zero day-trade policy**:
- A local ledger records every buy with its fill date
- Before any sell, the guard checks the position wasn't opened today
- Minimum hold period: **1 calendar day** (configurable)
- The PDT counter will always stay at **0**

---

## Project Structure

```
Trader/
├── main.py           # Entry point & scheduler
├── config.py         # All tunable parameters
├── broker.py         # Alpaca API wrapper
├── indicators.py     # Technical indicators (EMA, RSI, MACD, ATR, etc.)
├── strategy.py       # Entry & exit signal logic
├── screener.py       # Stock universe filtering
├── pdt_guard.py      # Pattern Day Trader protection
├── risk_manager.py   # Position sizing & risk limits
├── executor.py       # Trade execution orchestrator
├── logger.py         # Logging setup
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
└── logs/             # Runtime logs & PDT ledger
```

---

## Quick Start

### 1. Install dependencies

```bash
cd Trader
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your Alpaca API keys:
```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TRADING_MODE=paper
```

> **Start with `paper` mode** to test with simulated money before going live.

Get keys at: https://app.alpaca.markets/

### 3. Run the bot

```bash
# Continuous mode (runs every 30 min during market hours)
python main.py

# Single cycle (scan once and exit)
python main.py --once

# Check account status
python main.py --status
```

---

## Configuration

All parameters are in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_OPEN_POSITIONS` | 8 | Max simultaneous positions |
| `MAX_POSITION_PCT` | 12% | Max equity per position |
| `MAX_LOSS_PER_TRADE_PCT` | 2% | Risk budget per trade |
| `ATR_STOP_MULTIPLIER` | 2.0 | Stop loss = Entry − 2×ATR |
| `ATR_PROFIT_MULTIPLIER` | 3.0 | Take profit = Entry + 3×ATR |
| `TRAILING_STOP_PCT` | 5% | Trailing stop on winners |
| `MIN_HOLD_CALENDAR_DAYS` | 1 | PDT safety — minimum hold |
| `SCAN_INTERVAL_MINUTES` | 30 | How often to scan for entries |
| `CHECK_EXITS_MINUTES` | 15 | How often to check exits |

### Watchlist
The default watchlist includes ~60 liquid mid/large-cap stocks and ETFs. Edit `WATCHLIST` in `config.py` to customize.

---

## Risk Management

- **Per-trade risk**: sized so a stop-loss hit only costs ≤ 2% of equity
- **Position cap**: no single position exceeds 12% of equity
- **Portfolio cap**: max 8 concurrent positions, max 95% buying-power usage
- **Bracket orders**: every entry has an automatic stop-loss and take-profit
- **Trailing stops**: automatically added once a position is 5%+ profitable
- **PDT lock**: zero same-day round trips — ever

---

## Logs

Logs are written to both the console and `logs/trader.log`. The PDT ledger is stored at `logs/pdt_ledger.json`.

---

## Disclaimer

This bot is for **educational and paper-trading purposes**. Trading stocks involves risk of loss. Past performance of any strategy does not guarantee future results. Use at your own risk.
# Trader
