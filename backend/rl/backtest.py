"""
rl/backtest.py — Vectorized backtesting module.

Runs a trained RL agent (or any prediction function) over historical
OHLCV data and computes performance metrics.

Metrics computed:
  • Total return, annualized return
  • Sharpe ratio (annualized)
  • Sortino ratio
  • Maximum drawdown
  • Win rate, profit factor
  • Average trade return, average trade duration
  • Number of trades
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
import pandas as pd

from data.features import compute_features
from rl.env import TradingEnv

logger = logging.getLogger(__name__)


def run_backtest(
    candle_df: pd.DataFrame,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    initial_equity: float = 10_000.0,
    max_leverage: int = 20,
    max_risk_per_trade: float = 0.01,
    symbol: str = "BTCUSDT",
) -> dict:
    """
    Run a full backtest over a historical DataFrame.

    Args:
        candle_df: OHLCV DataFrame (with DatetimeTZDtype index)
        predict_fn: Callable(obs) → action array (e.g. agent.predict)
        initial_equity: Starting equity in USDT
        max_leverage: Maximum allowed leverage
        max_risk_per_trade: Fraction of equity risked per trade
        symbol: Trading pair symbol

    Returns:
        Dictionary of performance metrics + trade log + equity curve
    """
    logger.info("Starting backtest on %d candles...", len(candle_df))

    env = TradingEnv(
        mode="simulation",
        initial_equity=initial_equity,
        candle_df=candle_df,
        symbol=symbol,
        max_leverage=max_leverage,
        max_risk_per_trade=max_risk_per_trade,
    )

    obs, _ = env.reset(seed=42)
    equity_curve: list[float] = [initial_equity]
    reward_curve: list[float] = []

    terminated = truncated = False

    while not terminated and not truncated:
        action = predict_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        equity_curve.append(info.get("equity", env.equity))
        reward_curve.append(reward)

    trade_log = env.trade_log

    metrics = _compute_metrics(
        equity_curve=equity_curve,
        trade_log=trade_log,
        initial_equity=initial_equity,
        timeframe_minutes=5,   # TODO: pass from config
    )

    metrics["equity_curve"] = equity_curve
    metrics["reward_curve"] = reward_curve
    metrics["trade_log"]    = trade_log

    logger.info(
        "Backtest complete: %.2f%% return | Sharpe: %.2f | MaxDD: %.2f%% | Trades: %d",
        metrics["total_return_pct"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown_pct"],
        metrics["num_trades"],
    )
    return metrics


def _compute_metrics(
    equity_curve: list[float],
    trade_log: list[dict],
    initial_equity: float,
    timeframe_minutes: int = 5,
) -> dict:
    """Compute standard trading performance metrics."""
    eq = np.array(equity_curve)

    # ── Basic returns ─────────────────────────────────────────────────────────
    final_equity = eq[-1]
    total_return = (final_equity - initial_equity) / initial_equity
    n_bars = len(eq) - 1
    bars_per_year = 252 * 24 * (60 / timeframe_minutes)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1
    annual_return = (1 + total_return) ** (1 / max(years, 1e-6)) - 1

    # ── Drawdown ──────────────────────────────────────────────────────────────
    running_max = np.maximum.accumulate(eq)
    drawdowns   = (running_max - eq) / running_max
    max_drawdown = float(drawdowns.max())

    # ── Step returns ─────────────────────────────────────────────────────────
    step_returns = np.diff(eq) / eq[:-1]
    mean_return  = step_returns.mean()
    std_return   = step_returns.std() + 1e-10

    # ── Sharpe ratio (annualized) ─────────────────────────────────────────────
    sharpe = (mean_return / std_return) * math.sqrt(bars_per_year)

    # ── Sortino ratio ─────────────────────────────────────────────────────────
    downside = step_returns[step_returns < 0]
    sortino_std = downside.std() + 1e-10 if len(downside) > 0 else 1e-10
    sortino  = (mean_return / sortino_std) * math.sqrt(bars_per_year)

    # ── Trade-level analysis ──────────────────────────────────────────────────
    num_trades = len(trade_log)
    if num_trades > 0:
        pnls          = [t["pnl"] for t in trade_log]
        wins          = [p for p in pnls if p > 0]
        losses        = [p for p in pnls if p <= 0]
        win_rate      = len(wins) / num_trades
        avg_win       = float(np.mean(wins)) if wins else 0.0
        avg_loss      = float(np.mean(losses)) if losses else 0.0
        gross_profit  = float(sum(wins)) if wins else 0.0
        gross_loss    = abs(float(sum(losses))) if losses else 1e-10
        profit_factor = gross_profit / gross_loss
        avg_duration  = float(np.mean([t.get("steps", 0) for t in trade_log]))
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_duration = 0.0

    return {
        "total_return_pct":  round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "final_equity":      round(final_equity, 2),
        "max_drawdown_pct":  round(max_drawdown * 100, 2),
        "sharpe_ratio":      round(float(sharpe), 3),
        "sortino_ratio":     round(float(sortino), 3),
        "num_trades":        num_trades,
        "win_rate_pct":      round(win_rate * 100, 1) if num_trades > 0 else 0.0,
        "profit_factor":     round(profit_factor, 2) if num_trades > 0 else 0.0,
        "avg_win_usdt":      round(avg_win, 2),
        "avg_loss_usdt":     round(avg_loss, 2),
        "avg_trade_duration_bars": round(avg_duration, 1),
        "n_bars":            n_bars,
    }
