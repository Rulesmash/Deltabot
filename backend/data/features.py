"""
data/features.py — Technical Analysis feature computation.

Computes 30+ features from OHLCV data using pandas_ta and TA-Lib fallbacks.
Features are normalized to a [-1, 1] or [0, 1] range before being fed to the RL agent.

Feature groups:
  1. Trend:       EMA, SMA, MACD, ADX, Aroon
  2. Momentum:    RSI, Stochastic, Williams %R, CCI, ROC, MFI
  3. Volatility:  Bollinger Bands, ATR, Keltner Channels
  4. Volume:      OBV, VWAP, Chaikin MF, Volume ratio
  5. Price:       Rolling returns (1/5/20 bars), volatility
  6. Account:     Injected externally — see rl/env.py
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing pandas_ta (preferred) with TA-Lib as fallback
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas_ta not installed. Using manual implementations.")

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Trend (8)
    "ema_fast",          # EMA(8) normalized to price
    "ema_slow",          # EMA(21) normalized to price
    "ema_trend",         # (EMA8 - EMA21) / EMA21
    "sma_trend",         # (SMA20 - SMA50) / SMA50
    "macd_line",         # MACD(12,26,9) line normalized by price
    "macd_signal",       # MACD signal line normalized by price
    "macd_hist",         # MACD histogram normalized
    "adx",               # ADX(14) / 100

    # Momentum (9)
    "rsi",               # RSI(14) / 100
    "stoch_k",           # Stochastic %K / 100
    "stoch_d",           # Stochastic %D / 100
    "williams_r",        # Williams %R normalized [-1, 0]
    "cci",               # CCI(14) / 200 → approx [-1, 1]
    "roc_5",             # Rate of change (5-bar)
    "roc_20",            # Rate of change (20-bar)
    "mfi",               # Money Flow Index / 100
    "rsi_divergence",    # RSI vs price divergence signal

    # Volatility (6)
    "bb_upper_pct",      # (Price - BB_upper) / BB_width
    "bb_lower_pct",      # (Price - BB_lower) / BB_width
    "bb_position",       # (Price - BB_lower) / BB_width → [0, 1]
    "atr_pct",           # ATR(14) / Close → volatility measure
    "kc_position",       # Position within Keltner Channel
    "hist_volatility",   # 20-day realized volatility (std of returns)

    # Volume (6)
    "obv_delta",         # OBV first difference (direction of flow)
    "vwap_pct",          # (Price - VWAP) / VWAP
    "volume_ratio",      # Volume / SMA(Volume, 20)
    "cmf",               # Chaikin Money Flow
    "volume_trend",      # Volume trend (5-bar)
    "buying_pressure",   # (Close - Low) / (High - Low)

    # Price action (6)
    "return_1",          # 1-bar log return
    "return_5",          # 5-bar log return
    "return_20",         # 20-bar log return
    "high_low_ratio",    # (High - Low) / Close
    "close_open_ratio",  # (Close - Open) / Open
    "upper_shadow",      # Upper shadow as % of range
]

N_FEATURES = len(FEATURE_NAMES)  # 35 tech features (+ N account features added in env)


# ── Main Feature Computation ──────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features for a DataFrame of OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume, time].
            Must have at least 50 rows for all indicators to be valid.

    Returns:
        DataFrame with original columns + all feature columns.
        The first ~50 rows will contain NaN values (warm-up period).
    """
    df = df.copy()

    # Ensure columns are lowercase
    df.columns = [c.lower() for c in df.columns]

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── 1. Trend ─────────────────────────────────────────────────────────────

    if PANDAS_TA_AVAILABLE:
        df["ema8"]  = ta.ema(c, length=8)
        df["ema21"] = ta.ema(c, length=21)
        df["sma20"] = ta.sma(c, length=20)
        df["sma50"] = ta.sma(c, length=50)
        macd_df     = ta.macd(c, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            df["macd_line_raw"]   = macd_df.iloc[:, 0]
            df["macd_signal_raw"] = macd_df.iloc[:, 2]
            df["macd_hist_raw"]   = macd_df.iloc[:, 1]
        adx_df = ta.adx(h, l, c, length=14)
        if adx_df is not None and not adx_df.empty:
            df["adx_raw"] = adx_df.iloc[:, 0]
    else:
        df["ema8"]  = c.ewm(span=8, adjust=False).mean()
        df["ema21"] = c.ewm(span=21, adjust=False).mean()
        df["sma20"] = c.rolling(20).mean()
        df["sma50"] = c.rolling(50).mean()
        df["macd_line_raw"]   = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        df["macd_signal_raw"] = df["macd_line_raw"].ewm(span=9).mean()
        df["macd_hist_raw"]   = df["macd_line_raw"] - df["macd_signal_raw"]
        df["adx_raw"] = _manual_adx(h, l, c, period=14)

    df["ema_fast"]  = (df["ema8"] - c) / c
    df["ema_slow"]  = (df["ema21"] - c) / c
    df["ema_trend"] = (df["ema8"] - df["ema21"]) / df["ema21"].replace(0, np.nan)
    df["sma_trend"] = (df["sma20"] - df["sma50"]) / df["sma50"].replace(0, np.nan)
    df["macd_line"]   = df["macd_line_raw"] / c
    df["macd_signal"] = df["macd_signal_raw"] / c
    df["macd_hist"]   = df["macd_hist_raw"] / c
    df["adx"]         = df["adx_raw"].fillna(50) / 100

    # ── 2. Momentum ──────────────────────────────────────────────────────────

    if PANDAS_TA_AVAILABLE:
        df["rsi_raw"]      = ta.rsi(c, length=14)
        stoch_df           = ta.stoch(h, l, c, k=14, d=3)
        if stoch_df is not None and not stoch_df.empty:
            df["stoch_k_raw"] = stoch_df.iloc[:, 0]
            df["stoch_d_raw"] = stoch_df.iloc[:, 1]
        df["wr_raw"]       = ta.willr(h, l, c, length=14)
        df["cci_raw"]      = ta.cci(h, l, c, length=14)
        df["mfi_raw"]      = ta.mfi(h, l, c, v, length=14)
    else:
        df["rsi_raw"]      = _manual_rsi(c, period=14)
        df["stoch_k_raw"], df["stoch_d_raw"] = _manual_stoch(h, l, c)
        df["wr_raw"]       = _manual_williams_r(h, l, c, period=14)
        df["cci_raw"]      = _manual_cci(h, l, c, period=14)
        df["mfi_raw"]      = _manual_mfi(h, l, c, v, period=14)

    df["rsi"]         = df["rsi_raw"].fillna(50) / 100
    df["stoch_k"]     = df["stoch_k_raw"].fillna(50) / 100
    df["stoch_d"]     = df["stoch_d_raw"].fillna(50) / 100
    df["williams_r"]  = df["wr_raw"].fillna(-50) / 100   # already -100..0
    df["cci"]         = df["cci_raw"].fillna(0).clip(-200, 200) / 200
    df["roc_5"]       = c.pct_change(5).fillna(0).clip(-0.2, 0.2)
    df["roc_20"]      = c.pct_change(20).fillna(0).clip(-0.5, 0.5)
    df["mfi"]         = df["mfi_raw"].fillna(50) / 100

    # RSI divergence: sign difference between rsi change and price change
    rsi_change    = df["rsi_raw"].diff().fillna(0)
    price_change  = c.pct_change().fillna(0)
    df["rsi_divergence"] = np.where(
        np.sign(rsi_change) != np.sign(price_change), 1.0, 0.0
    )

    # ── 3. Volatility ────────────────────────────────────────────────────────

    if PANDAS_TA_AVAILABLE:
        bb_df = ta.bbands(c, length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            df["bb_lower_raw"]  = bb_df.iloc[:, 0]
            df["bb_mid_raw"]    = bb_df.iloc[:, 1]
            df["bb_upper_raw"]  = bb_df.iloc[:, 2]
        df["atr_raw"] = ta.atr(h, l, c, length=14)
        kc_df = ta.kc(h, l, c, length=20, scalar=2)
        if kc_df is not None and not kc_df.empty:
            df["kc_lower"] = kc_df.iloc[:, 0]
            df["kc_upper"] = kc_df.iloc[:, 2]
    else:
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_lower_raw"] = sma20 - 2 * std20
        df["bb_mid_raw"]   = sma20
        df["bb_upper_raw"] = sma20 + 2 * std20
        df["atr_raw"]      = _manual_atr(h, l, c, period=14)
        df["kc_lower"]     = sma20 - 1.5 * df["atr_raw"]
        df["kc_upper"]     = sma20 + 1.5 * df["atr_raw"]

    bb_width = (df["bb_upper_raw"] - df["bb_lower_raw"]).replace(0, np.nan)
    df["bb_upper_pct"] = (c - df["bb_upper_raw"]) / bb_width.fillna(1)
    df["bb_lower_pct"] = (c - df["bb_lower_raw"]) / bb_width.fillna(1)
    df["bb_position"]  = ((c - df["bb_lower_raw"]) / bb_width.fillna(1)).clip(0, 1)
    df["atr_pct"]      = (df["atr_raw"] / c).fillna(0.01)
    kc_range = (df["kc_upper"] - df["kc_lower"]).replace(0, np.nan)
    df["kc_position"]  = ((c - df["kc_lower"]) / kc_range.fillna(1)).clip(0, 1)
    df["hist_volatility"] = c.pct_change().rolling(20).std().fillna(0)

    # ── 4. Volume ────────────────────────────────────────────────────────────

    if PANDAS_TA_AVAILABLE:
        df["obv_raw"] = ta.obv(c, v)
        df["cmf_raw"] = ta.cmf(h, l, c, v, length=20)
    else:
        df["obv_raw"] = _manual_obv(c, v)
        df["cmf_raw"] = _manual_cmf(h, l, c, v, period=20)

    df["obv_delta"]  = df["obv_raw"].diff().fillna(0).apply(np.sign)
    vwap             = _compute_vwap(h, l, c, v)
    df["vwap_pct"]   = ((c - vwap) / vwap.replace(0, np.nan)).fillna(0).clip(-0.05, 0.05)
    vol_sma20        = v.rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = (v / vol_sma20).fillna(1).clip(0, 5) / 5   # normalized [0,1]
    df["cmf"]          = df["cmf_raw"].fillna(0).clip(-1, 1)
    df["volume_trend"] = v.pct_change(5).fillna(0).clip(-2, 2) / 2
    df["buying_pressure"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)

    # ── 5. Price action ──────────────────────────────────────────────────────

    log_ret = np.log(c / c.shift(1)).fillna(0)
    df["return_1"]  = log_ret.clip(-0.1, 0.1)
    df["return_5"]  = log_ret.rolling(5).sum().fillna(0).clip(-0.2, 0.2)
    df["return_20"] = log_ret.rolling(20).sum().fillna(0).clip(-0.5, 0.5)
    range_c         = (h - l).replace(0, np.nan)
    df["high_low_ratio"]   = (range_c / c).fillna(0.01)
    df["close_open_ratio"] = ((c - o) / o.replace(0, np.nan)).fillna(0).clip(-0.05, 0.05)
    df["upper_shadow"]     = ((h - c.clip(upper=h)) / range_c.fillna(1)).fillna(0)

    return df


def get_feature_vector(df: pd.DataFrame, idx: int = -1) -> np.ndarray:
    """
    Extract the feature vector at position `idx` (default: last row).

    Returns:
        np.ndarray of shape (N_FEATURES,) with float32 values.
        All NaNs are replaced with 0.
    """
    row = df.iloc[idx]
    features = np.array(
        [row.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32
    )
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features


def get_feature_matrix(df: pd.DataFrame, lookback: int) -> np.ndarray:
    """
    Return a (lookback, N_FEATURES) matrix of the last `lookback` rows.
    Used for sequence-based models.
    """
    slice_df = df.tail(lookback)
    matrix = np.array(
        [[row.get(name, 0.0) for name in FEATURE_NAMES] for _, row in slice_df.iterrows()],
        dtype=np.float32,
    )
    return np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)


# ── Manual Indicator Implementations (TA-Lib fallbacks) ──────────────────────

def _manual_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _manual_stoch(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k: int = 14, d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    low_min  = low.rolling(k).min()
    high_max = high.rolling(k).max()
    stoch_k  = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    stoch_d  = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _manual_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    high_max = high.rolling(period).max()
    low_min  = low.rolling(period).min()
    return -100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)


def _manual_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tp   = (high + low + close) / 3
    sma  = tp.rolling(period).mean()
    mad  = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _manual_mfi(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, period: int = 14,
) -> pd.Series:
    tp  = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg = rmf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfr = pos / neg.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def _manual_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _manual_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Simplified ADX calculation."""
    atr  = _manual_atr(high, low, close, period)
    up   = high.diff()
    down = -low.diff()
    dm_plus  = up.where((up > down) & (up > 0), 0)
    dm_minus = down.where((down > up) & (down > 0), 0)
    di_plus  = 100 * dm_plus.ewm(com=period - 1).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(com=period - 1).mean() / atr.replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(com=period - 1).mean()


def _manual_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _manual_cmf(
    high: pd.Series, low: pd.Series, close: pd.Series,
    volume: pd.Series, period: int = 20,
) -> pd.Series:
    mf_mult  = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_vol   = mf_mult * volume
    return mf_vol.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Simple rolling VWAP (20 bars)."""
    tp    = (high + low + close) / 3
    cumtp = (tp * volume).rolling(20).sum()
    cumv  = volume.rolling(20).sum()
    return cumtp / cumv.replace(0, np.nan)
