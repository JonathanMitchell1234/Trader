"""
Technical-analysis indicator calculations.
All functions take a pandas DataFrame with OHLCV columns and return
a new DataFrame (or Series) with the indicator values appended.
"""

from __future__ import annotations

import pandas as pd
import ta

import config
from logger import get_logger

log = get_logger("indicators")


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute every indicator the strategy needs and return
    the enriched DataFrame.  Expects columns: open, high, low, close, volume.
    """
    df = df.copy()

    # ── Exponential Moving Averages ──────────────────────────
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=config.EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=config.EMA_SLOW)
    df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=config.EMA_TREND)
    df["ema_200"] = ta.trend.ema_indicator(df["close"], window=200)

    # ── RSI ──────────────────────────────────────────────────
    df["rsi"] = ta.momentum.rsi(df["close"], window=config.RSI_PERIOD)

    # ── MACD ─────────────────────────────────────────────────
    macd = ta.trend.MACD(
        df["close"],
        window_slow=config.MACD_SLOW,
        window_fast=config.MACD_FAST,
        window_sign=config.MACD_SIGNAL,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ── ATR (Average True Range) ─────────────────────────────
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=config.ATR_PERIOD
    )

    # ── Volume SMA ───────────────────────────────────────────
    df["vol_sma"] = df["volume"].rolling(window=config.VOLUME_SMA_PERIOD).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"]

    # ── Bollinger Bands (supplementary) ──────────────────────
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    # ── Stochastic Oscillator (supplementary) ────────────────
    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── ADX (trend strength) ─────────────────────────────────
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    log.debug("Indicators computed for %d bars", len(df))
    return df


def latest_row(df: pd.DataFrame) -> pd.Series:
    """Return the most recent fully-formed bar (last row)."""
    return df.iloc[-1]


def prev_row(df: pd.DataFrame) -> pd.Series:
    """Return the second-to-last bar (for crossover detection)."""
    return df.iloc[-2]
