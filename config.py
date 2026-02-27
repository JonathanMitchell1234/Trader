"""
Configuration for the Swing Trading Bot.
All tunable parameters in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Alpaca API
# ─────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

BASE_URL = PAPER_BASE_URL if TRADING_MODE == "paper" else LIVE_BASE_URL

# ─────────────────────────────────────────────
# Account Size & Fractional Shares
# ─────────────────────────────────────────────
FRACTIONAL_SHARES = True            # use fractional shares (Alpaca supports)
SMALL_ACCOUNT_THRESHOLD = 2_000     # below this $ → small-account rules kick in

# ─────────────────────────────────────────────
# PDT Protection  (Pattern Day Trader Guard)
# ─────────────────────────────────────────────
# Accounts under $25k are limited to 3 day trades per 5 rolling business days.
# We set a hard cap of 0 day trades — the bot NEVER opens and closes
# the same symbol on the same calendar day.
MAX_DAY_TRADES_ALLOWED = 0          # hard-lock: no same-day round trips
PDT_LOOKBACK_DAYS = 5               # rolling window (business days)
MIN_HOLD_CALENDAR_DAYS = 2          # minimum days to hold before selling

# ─────────────────────────────────────────────
# Strategy – Technical Indicators
# ─────────────────────────────────────────────
# Trend EMAs
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50          # long-term trend filter
EMA_LONG = 200          # market regime / super-trend filter

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30       # buy zone  (widened for more entries)
RSI_OVERBOUGHT = 80     # sell zone (raised to let winners run)

# MACD (standard 12-26-9)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ATR (for stops and position sizing)
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 3.0    # stop loss = entry − ATR * mult (wide to survive noise)
ATR_PROFIT_MULTIPLIER = 6.0  # take profit = entry + ATR * mult (let winners run)

# Volume confirmation
VOLUME_SMA_PERIOD = 20
VOLUME_SURGE_FACTOR = 1.0  # 1.0 = normal volume ok (removed as hard gate)

# ── Scoring system – entry signals are scored, not binary ────
ENTRY_SCORE_THRESHOLD = 5      # minimum score (out of ~14) to trigger a buy

# ── Momentum ranking (buy the strongest stocks) ─────────────
MOMENTUM_LOOKBACK = 20          # days to measure momentum (rate of change)
MOMENTUM_TOP_PCT = 0.50         # only consider top 50% by momentum
MOMENTUM_SCORE_WEIGHT = 2       # bonus points for top-quartile momentum

# ── Trend quality (EMA slope must be rising) ────────────────
EMA_SLOPE_PERIOD = 5            # bars to measure EMA-50 slope direction

# ── Dead-money exit: sell stagnant positions ────────────────
DEAD_MONEY_DAYS = 10            # if position flat for this many trading days
DEAD_MONEY_THRESHOLD = 0.02     # has moved less than 2% total

# ── Cooldown: don't re-enter a symbol within N days of exit ──
RE_ENTRY_COOLDOWN_DAYS = 5

# ── Market regime filter (SPY-based) ────────────────────────
MARKET_REGIME_ENABLED = True   # reject new buys in bear markets
MARKET_REGIME_SYMBOL = "SPY"   # index proxy
# Bull = SPY above its 200-EMA; Bear = below

# ─────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────
MAX_OPEN_POSITIONS = 10           # hard cap (overridden for small accounts)
MAX_POSITION_PCT = 0.15           # max 15% of equity per position
MAX_PORTFOLIO_RISK_PCT = 0.06     # max 6% total portfolio at risk
MAX_LOSS_PER_TRADE_PCT = 0.02     # risk at most 2% of equity per trade
MAX_PORTFOLIO_EXPOSURE_PCT = 0.95  # never use more than 95% buying power
TRAILING_STOP_ACTIVATE_PCT = 0.06  # activate trailing stop after 6% gain
TRAILING_STOP_PCT = 0.04          # 4% trailing stop on winners

# ── Adaptive stop: tighten trailing stop as profit grows ─────
TRAILING_STOP_TIGHT_ACTIVATE = 0.15  # above 15% profit, tighten
TRAILING_STOP_TIGHT_PCT = 0.03       # to 3% trailing stop

# ── Small-account overrides (auto-applied when equity < SMALL_ACCOUNT_THRESHOLD)
SMALL_MAX_OPEN_POSITIONS = 3       # concentrate with tiny capital
SMALL_MAX_POSITION_PCT = 0.45      # up to 45% per position (need size)
SMALL_MAX_LOSS_PER_TRADE_PCT = 0.03  # 3% risk (tolerate more to get in)
SMALL_ATR_STOP_MULTIPLIER = 2.0    # tighter stops to limit $ loss
SMALL_ATR_PROFIT_MULTIPLIER = 4.0  # corresponding tighter targets

# ─────────────────────────────────────────────
# Stock Universe / Screener Filters
# ─────────────────────────────────────────────
# Core watchlist – mixed price range for small + large accounts
WATCHLIST = [
    # Tech (mid-high price)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSM",
    "CRM", "ADBE", "NFLX", "QCOM", "INTC", "AVGO", "MU",
    # Tech (affordable < $50)
    "PLTR", "SOFI", "MARA", "HOOD", "SNAP", "U",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP",
    "C", "SCHW",  # affordable financials
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO",
    # Consumer / Retail
    "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "DIS",
    "F", "RIVN",  # affordable consumer
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrial
    "CAT", "DE", "BA", "GE", "HON", "UNP",
    # ETFs (mixed price range)
    "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK",
    "TQQQ", "SOXL", "ARKK", "VTI", "VOO", "DIA",
]

MIN_PRICE = 5.0           # lower floor for affordable stocks
MAX_PRICE = 1_500.0       # skip ultra-high-price names
MIN_AVG_VOLUME = 500_000  # relaxed for broader universe

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
BARS_LOOKBACK = 100       # how many daily bars to fetch for analysis
BAR_TIMEFRAME = "1Day"    # daily bars for swing trading

# ─────────────────────────────────────────────
# Scheduling
# ─────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 30    # re-scan for entries every N minutes
CHECK_EXITS_MINUTES = 15      # check exit signals every N minutes

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = "logs/trader.log"


# ─────────────────────────────────────────────
# Dynamic helpers (adjust for account size)
# ─────────────────────────────────────────────
def get_max_positions(equity: float) -> int:
    """Scale max open positions to account size."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_MAX_OPEN_POSITIONS
    return MAX_OPEN_POSITIONS


def get_position_pct(equity: float) -> float:
    """Max position % scales up for smaller accounts."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_MAX_POSITION_PCT
    return MAX_POSITION_PCT


def get_risk_per_trade(equity: float) -> float:
    """Risk per trade % — slightly higher for small accounts."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_MAX_LOSS_PER_TRADE_PCT
    return MAX_LOSS_PER_TRADE_PCT


def get_atr_stop_mult(equity: float) -> float:
    """ATR stop multiplier — tighter for small accounts to limit $ loss."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_ATR_STOP_MULTIPLIER
    return ATR_STOP_MULTIPLIER


def get_atr_profit_mult(equity: float) -> float:
    """ATR profit multiplier — tighter targets for small accounts."""
    if equity < SMALL_ACCOUNT_THRESHOLD:
        return SMALL_ATR_PROFIT_MULTIPLIER
    return ATR_PROFIT_MULTIPLIER
