"""
rl/env.py — Custom Gymnasium environment for DeltaRL Trader.
           RISK-MITIGATION-FIRST version (v2)

Supports three operating modes:
  • "simulation"  — No orders placed; P&L simulated from live price movement
  • "demo"        — Real orders on Delta Exchange Testnet (virtual money)
  • "live"        — Real orders on Delta Exchange Production (real money)

State space (N_FEATURES + N_ACCOUNT_FEATURES + N_RISK_FEATURES):
  [35 tech features | 8 account features | 6 risk features]

  Account features (8):
    equity_norm, margin_ratio, unrealized_pnl_pct, position_side,
    position_leverage, position_size_pct, drawdown, steps_in_position

  Risk features (6) — NEW in v2:
    portfolio_drawdown_pct   — (peak - equity) / peak, [0, 1]
    volatility_regime        — ATR percentile rank in rolling window, [0, 1]
    funding_rate_exposure    — signed funding cost proxy, [-1, 1]
    open_pnl_exposure_pct    — unrealized P&L / equity (clipped), [-1, 1]
    liquidation_prob_proxy   — distance to liq as fraction of price, [0, 1]
    time_in_position_norm    — tanh(steps / 100), [0, 1]

Action space (Box, 5-dimensional continuous) — v2 adds close_signal:
  action[0]: direction     (-1 to 1)  → <-0.33=Short, ±0.33=Flat, >0.33=Long
  action[1]: leverage      (0 to 1)   → maps to 1x – 20x
  action[2]: sl_pct        (0 to 1)   → maps to 0.3% – 5.0%
  action[3]: tp_pct        (0 to 1)   → maps to 0.5% – 10.0%
  action[4]: close_signal  (0 to 1)   → >0.7 forces immediate position close

Curriculum Learning:
  Stage 0 — Low-volatility bars only  (vol_regime < 0.33)
  Stage 1 — Low + medium volatility   (vol_regime < 0.66)
  Stage 2 — Full data (all regimes, including extreme events)

  The curriculum stage is set externally: env.set_curriculum_stage(n)

Safety Fallback:
  If portfolio drawdown > SAFETY_DD_THRESHOLD or volatility_regime > SAFETY_VOL_THRESHOLD,
  the environment logs a warning. The agent can (optionally) be overridden by the
  safety policy baseline in the orchestrator before calling env.step().

Reward: see rl/reward.py
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from data.features import FEATURE_NAMES, N_FEATURES, compute_features, get_feature_vector
from rl.reward import RewardCalculator

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

N_ACCOUNT_FEATURES = 8    # unchanged from v1
N_RISK_FEATURES    = 6    # NEW in v2
N_TOTAL_FEATURES   = N_FEATURES + N_ACCOUNT_FEATURES + N_RISK_FEATURES  # 35+8+6 = 49

# Action space dimensions
N_ACTION_DIMS = 5   # [direction, leverage, sl, tp, close_signal]

# Action interpretation thresholds
DIRECTION_THRESHOLD  = 0.33   # |action[0]| must be > this for a directional trade
CLOSE_SIGNAL_THRESH  = 0.70   # action[4] > this forces a position close

# Leverage mapping: action[1] ∈ [0, 1] → leverage ∈ [1, 20]
LEVERAGE_MIN = 1
LEVERAGE_MAX = 20

# SL/TP mapping
SL_MIN_PCT = 0.003   # 0.3%
SL_MAX_PCT = 0.050   # 5.0%
TP_MIN_PCT = 0.005   # 0.5%
TP_MAX_PCT = 0.100   # 10.0%

# Simulated fees per side
MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.0005   # 0.05%

# Max steps per training episode
MAX_EPISODE_STEPS = 500

# Minimum candle history before the agent can act
MIN_HISTORY = 60

# ── Safety thresholds (for safety policy in orchestrator) ─────────────────────
SAFETY_DD_THRESHOLD  = 0.10   # 10% portfolio DD → suggest flat
SAFETY_VOL_THRESHOLD = 0.85   # vol_regime > 85% → suggest lower leverage

# ── Curriculum regime thresholds ──────────────────────────────────────────────
CURRICULUM_STAGE_THRESHOLDS = [0.33, 0.66, 1.01]   # max vol_regime for each stage

# ── Simulated funding rate parameters ────────────────────────────────────────
FUNDING_PERIOD_BARS   = 576   # 8 hours at 5-min bars
FUNDING_BASE_RATE     = 0.0001   # 0.01% per 8h (realistic for BTC)


class TradingEnv(gym.Env):
    """
    Gymnasium environment wrapping the DeltaRL risk-first trading logic.

    Modes:
      - "simulation": fastest, no API calls, price-based P&L sim
      - "demo": places real virtual orders on Delta Testnet
      - "live": places real orders on Delta Production

    v2 additions:
      - 6 new risk observation features
      - 5th action dimension (close_signal)
      - Curriculum learning support
      - Safety fallback detection
      - Full reward bridge to the new RewardCalculator v2 signature
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mode: str = "simulation",
        initial_equity: float = 10_000.0,
        candle_df: pd.DataFrame | None = None,
        delta_client=None,
        symbol: str = "BTCUSDT",
        max_leverage: int = LEVERAGE_MAX,
        max_risk_per_trade: float = 0.01,
        curriculum_stage: int = 0,         # 0=low-vol; 1=medium; 2=full
    ) -> None:
        super().__init__()

        assert mode in ("simulation", "demo", "live"), f"Invalid mode: {mode}"

        self.mode               = mode
        self.symbol             = symbol
        self.initial_equity     = initial_equity
        self.max_leverage       = max_leverage
        self.max_risk_per_trade = max_risk_per_trade
        self._client            = delta_client
        self._curriculum_stage  = curriculum_stage

        # ── Observation and action spaces ────────────────────────────────────
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(N_TOTAL_FEATURES,),
            dtype=np.float32,
        )

        # v2: 5D action (add close_signal)
        self.action_space = gym.spaces.Box(
            low=np.array( [-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Internal state ────────────────────────────────────────────────────
        self._candle_df_raw: pd.DataFrame | None = candle_df
        self._candle_df:     pd.DataFrame | None = None
        self._current_idx:   int = MIN_HISTORY
        self._episode_start_idx: int = MIN_HISTORY

        # ATR-based volatility percentile history (computed on reset)
        self._atr_percentiles: np.ndarray | None = None

        self._equity:        float = initial_equity
        self._peak_equity:   float = initial_equity
        self._num_trades:    int   = 0
        self._step_count:    int   = 0

        # Open position
        self._position_side:         int   = 0
        self._position_entry_price:  float = 0.0
        self._position_size:         float = 0.0    # USDT notional
        self._position_leverage:     float = 1.0
        self._position_sl:           float = 0.0
        self._position_tp:           float = 0.0
        self._steps_in_position:     int   = 0

        # Simulated funding state
        self._funding_rate:          float = FUNDING_BASE_RATE
        self._bars_since_funding:    int   = 0

        self._reward_calc       = RewardCalculator()
        self._episode_rewards:  list[float] = []
        self._trade_log:        list[dict]  = []

    # ── Curriculum Learning ───────────────────────────────────────────────────

    def set_curriculum_stage(self, stage: int) -> None:
        """
        Set the curriculum stage (0=low-vol, 1=medium, 2=full).
        Controls which historical bars are used as episode start points.
        """
        self._curriculum_stage = max(0, min(2, stage))
        logger.info("Curriculum stage set to %d", self._curriculum_stage)

    # ── Gymnasium Interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._equity                 = self.initial_equity
        self._peak_equity            = self.initial_equity
        self._position_side          = 0
        self._position_entry_price   = 0.0
        self._position_size          = 0.0
        self._steps_in_position      = 0
        self._num_trades             = 0
        self._step_count             = 0
        self._episode_rewards        = []
        self._trade_log              = []
        self._bars_since_funding     = 0
        self._funding_rate           = FUNDING_BASE_RATE

        self._reward_calc.reset(self.initial_equity)

        if self._candle_df_raw is not None:
            self._candle_df = compute_features(self._candle_df_raw)
            n = len(self._candle_df)
            if n <= MIN_HISTORY + 1:
                raise ValueError(f"Not enough candles ({n}) — need at least {MIN_HISTORY + 2}.")

            # Pre-compute ATR percentile for curriculum filtering
            self._compute_atr_percentiles()

            # Curriculum: pick start index respecting volatility stage
            max_start = n - MAX_EPISODE_STEPS - 1
            self._current_idx = self._sample_curriculum_start(max_start)
            self._episode_start_idx = self._current_idx

        obs  = self._get_observation()
        info = {"mode": self.mode, "symbol": self.symbol, "curriculum_stage": self._curriculum_stage}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: [direction, leverage_norm, sl_norm, tp_norm, close_signal] array

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        direction, leverage, sl_pct, tp_pct, force_close = self._decode_action(action)

        current_price = self._get_current_price()
        prev_equity   = self._equity

        realized_pnl    = 0.0
        fees            = 0.0
        slippage        = 0.0
        position_closed = False
        was_profitable  = False

        # ── Tick simulated funding rate ───────────────────────────────────────
        self._tick_funding()

        # ── Check explicit close signal first ─────────────────────────────────
        if force_close and self._position_side != 0:
            pnl, fee = self._close_position(current_price)
            realized_pnl  += pnl
            fees          += fee
            position_closed = True
            was_profitable = (pnl > 0)

        # ── Check SL/TP hit ───────────────────────────────────────────────────
        if self._position_side != 0 and not position_closed:
            sl_hit = self._check_sl(current_price)
            tp_hit = self._check_tp(current_price)

            if sl_hit or tp_hit:
                exit_price = self._position_sl if sl_hit else self._position_tp
                pnl, fee   = self._close_position(exit_price)
                realized_pnl  += pnl
                fees          += fee
                position_closed = True
                was_profitable  = (pnl > 0)

        # ── Apply RL action ───────────────────────────────────────────────────
        if not position_closed:
            if direction == 0:
                # Flat: close any open position
                if self._position_side != 0:
                    pnl, fee = self._close_position(current_price)
                    realized_pnl  += pnl
                    fees          += fee
                    position_closed = True
                    was_profitable  = (pnl > 0)

            elif direction != self._position_side:
                # Direction change: close old, open new
                if self._position_side != 0:
                    pnl, fee = self._close_position(current_price)
                    realized_pnl  += pnl
                    fees          += fee
                    position_closed = True
                    was_profitable  = (pnl > 0)

                if direction != 0:
                    _, fee = self._open_position(direction, current_price, leverage, sl_pct, tp_pct)
                    fees     += fee
                    slippage += current_price * 0.0001

        if self._position_side != 0:
            self._steps_in_position += 1
        else:
            self._steps_in_position = 0

        self._update_unrealized_equity(current_price)

        # ── Compute risk context for reward ───────────────────────────────────
        vol_regime         = self._get_volatility_regime()
        liq_proximity      = self._get_liquidation_proximity(current_price)
        open_pnl_pct       = self._get_open_pnl_pct(current_price)
        portfolio_dd       = (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)

        # ── Reward ────────────────────────────────────────────────────────────
        reward, breakdown = self._reward_calc.compute(
            realized_pnl=realized_pnl,
            fees=fees,
            slippage=slippage,
            equity=self._equity,
            leverage=leverage,
            action_direction=direction,
            position_closed=position_closed,
            volatility_regime=vol_regime,
            funding_rate=self._funding_rate if self._position_side != 0 else 0.0,
            liquidation_proximity=liq_proximity,
            open_pnl_pct=open_pnl_pct,
            is_profitable_trade=was_profitable,
            num_trades_episode=self._num_trades,
        )
        self._episode_rewards.append(reward)

        if self._candle_df is not None:
            self._current_idx += 1

        obs        = self._get_observation()
        terminated = self._check_terminated()
        truncated  = self._step_count >= MAX_EPISODE_STEPS

        # Detect safety-policy trigger conditions
        in_safety_zone = (
            portfolio_dd > SAFETY_DD_THRESHOLD or
            vol_regime > SAFETY_VOL_THRESHOLD
        )

        info = {
            "equity":           self._equity,
            "realized_pnl":     realized_pnl,
            "direction":        direction,
            "leverage":         leverage,
            "sl_pct":           sl_pct,
            "tp_pct":           tp_pct,
            "num_trades":       self._num_trades,
            "reward_breakdown": breakdown,
            # v2 risk info
            "volatility_regime":    round(vol_regime, 3),
            "liquidation_proximity": round(liq_proximity, 3),
            "portfolio_drawdown":   round(portfolio_dd, 4),
            "funding_rate":         round(self._funding_rate, 6),
            "rolling_sharpe":       round(self._reward_calc.rolling_sharpe, 4),
            "rolling_sortino":      round(self._reward_calc.rolling_sortino, 4),
            "max_dd_episode":       round(self._reward_calc.max_drawdown_episode, 4),
            "in_safety_zone":       in_safety_zone,
            "curriculum_stage":     self._curriculum_stage,
        }
        return obs, reward, terminated, truncated, info

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build the (N_TOTAL_FEATURES,) = 49-dim state vector."""
        tech_features    = self._get_tech_features()
        account_features = self._get_account_features()
        risk_features    = self._get_risk_features()
        obs = np.concatenate([tech_features, account_features, risk_features]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _get_tech_features(self) -> np.ndarray:
        if self._candle_df is None:
            return np.zeros(N_FEATURES, dtype=np.float32)
        return get_feature_vector(self._candle_df, self._current_idx)

    def _get_account_features(self) -> np.ndarray:
        """
        8 account-state features (unchanged from v1):
          0: equity_norm         — equity / initial_equity - 1
          1: margin_ratio        — used_margin / equity
          2: unrealized_pnl_pct  — unrealized P&L / equity
          3: position_side       — -1, 0, 1
          4: position_leverage   — leverage / max_leverage
          5: position_size_pct   — notional / equity
          6: drawdown            — (peak - equity) / peak
          7: steps_in_position   — tanh(steps / 50)
        """
        curr_price  = self._get_current_price()
        equity_norm = self._equity / self.initial_equity - 1.0

        used_margin = (
            self._position_size / self._position_leverage
            if self._position_side != 0 else 0.0
        )
        margin_ratio = min(used_margin / max(self._equity, 1.0), 1.0)

        unrealized = self._get_open_pnl_pct(curr_price)
        drawdown   = (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)

        return np.array([
            np.clip(equity_norm, -2.0, 2.0),
            margin_ratio,
            np.clip(unrealized, -1.0, 1.0),
            float(self._position_side),
            self._position_leverage / self.max_leverage if self._position_side != 0 else 0.0,
            np.clip(self._position_size / max(self._equity, 1.0), 0.0, 5.0),
            np.clip(drawdown, 0.0, 1.0),
            float(np.tanh(self._steps_in_position / 50.0)),
        ], dtype=np.float32)

    def _get_risk_features(self) -> np.ndarray:
        """
        6 NEW risk features for v2 (risk-mitigation-first):
          0: portfolio_drawdown_pct  — (peak - equity) / peak, [0, 1]
          1: volatility_regime       — ATR percentile rank, [0, 1]
          2: funding_rate_exposure   — signed funding × position_side, [-1, 1]
          3: open_pnl_exposure_pct   — unrealized P&L / equity, [-1, 1]
          4: liquidation_prob_proxy  — distance to liq (fraction), [0, 1]
          5: time_in_position_norm   — tanh(steps / 100), [0, 1]
        """
        curr_price = self._get_current_price()

        portfolio_dd      = (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)
        vol_regime        = self._get_volatility_regime()
        # funding_rate: positive if long pays, negative if short pays
        # exposure = fundng_rate * position_side (adverse = positive)
        funding_exposure  = float(np.clip(
            self._funding_rate * self._position_side, -1.0, 1.0
        ))
        open_pnl_pct      = self._get_open_pnl_pct(curr_price)
        liq_proximity     = self._get_liquidation_proximity(curr_price)
        time_in_pos       = float(np.tanh(self._steps_in_position / 100.0))

        return np.array([
            np.clip(portfolio_dd, 0.0, 1.0),
            np.clip(vol_regime, 0.0, 1.0),
            np.clip(funding_exposure, -1.0, 1.0),
            np.clip(open_pnl_pct, -1.0, 1.0),
            np.clip(liq_proximity, 0.0, 1.0),
            time_in_pos,
        ], dtype=np.float32)

    # ── Risk Feature Helpers ──────────────────────────────────────────────────

    def _get_volatility_regime(self) -> float:
        """
        ATR percentile rank at the current bar, normalised to [0, 1].
        0 = lowest historical volatility, 1 = highest.
        Uses the pre-computed percentile array from reset().
        """
        if self._atr_percentiles is None or self._current_idx >= len(self._atr_percentiles):
            return 0.3   # neutral default
        return float(self._atr_percentiles[self._current_idx])

    def _get_liquidation_proximity(self, current_price: float) -> float:
        """
        Proxy for liquidation probability.
        Returns the fraction of the way from current price to the
        estimated liquidation price (0 = far from liq, 1 = at liq).

        Liquidation price (simplified):
          Long:  liq_price = entry_price * (1 - 1/leverage + main_margin_rate)
          Short: liq_price = entry_price * (1 + 1/leverage - main_margin_rate)
        where main_margin_rate ≈ 0.005 (0.5%)
        """
        if self._position_side == 0 or self._position_entry_price <= 0:
            return 0.0

        maint_rate = 0.005
        lev        = max(self._position_leverage, 1.0)

        if self._position_side == 1:   # Long
            liq_price = self._position_entry_price * (1 - 1/lev + maint_rate)
            if liq_price <= 0 or current_price <= liq_price:
                return 1.0
            total_dist = self._position_entry_price - liq_price
            remaining  = current_price - liq_price
            return float(np.clip(1.0 - remaining / max(total_dist, 1e-6), 0.0, 1.0))
        else:   # Short
            liq_price = self._position_entry_price * (1 + 1/lev - maint_rate)
            if current_price >= liq_price:
                return 1.0
            total_dist = liq_price - self._position_entry_price
            remaining  = liq_price - current_price
            return float(np.clip(1.0 - remaining / max(total_dist, 1e-6), 0.0, 1.0))

    def _get_open_pnl_pct(self, current_price: float) -> float:
        """Unrealized P&L as a fraction of equity."""
        if self._position_side == 0 or self._position_entry_price <= 0:
            return 0.0
        price_change = (current_price - self._position_entry_price) / self._position_entry_price
        return float(np.clip(
            price_change * self._position_side * self._position_size / max(self._equity, 1.0),
            -1.0, 1.0
        ))

    def _tick_funding(self) -> None:
        """
        Simulate Delta Exchange's 8-hourly funding rate mechanism.
        Real funding is fetched by CandleBuffer in demo/live modes.
        In simulation mode, we use a simple random-walk model.
        """
        self._bars_since_funding += 1
        if self._bars_since_funding >= FUNDING_PERIOD_BARS:
            # Funding payment event — reset counter and randomise rate
            self._bars_since_funding = 0
            # Funding rates are mean-reverting around FUNDING_BASE_RATE
            noise = np.random.normal(0, 0.0002)
            self._funding_rate = float(np.clip(
                FUNDING_BASE_RATE * 0.8 + noise,
                -0.002, 0.002
            ))

    # ── Curriculum Learning ────────────────────────────────────────────────────

    def _compute_atr_percentiles(self) -> None:
        """
        Pre-compute per-bar ATR percentile ranks for the loaded dataset.
        Used by curriculum to filter episode start points.
        """
        if self._candle_df is None:
            return

        # Use ATR column if available (computed by features.py), else estimate
        if "atr" in self._candle_df.columns:
            atr_series = self._candle_df["atr"].fillna(method="ffill").fillna(0)
        else:
            # Rough ATR proxy: (high - low)
            atr_series = (
                self._candle_df["high"] - self._candle_df["low"]
                if "high" in self._candle_df.columns
                else pd.Series(np.zeros(len(self._candle_df)))
            ).fillna(0)

        atr_arr = atr_series.values.astype(np.float64)

        # Rolling percentile rank (use a 200-bar window)
        window  = min(200, len(atr_arr))
        percentiles = np.zeros(len(atr_arr), dtype=np.float32)
        for i in range(len(atr_arr)):
            start = max(0, i - window)
            window_vals = atr_arr[start:i + 1]
            val         = atr_arr[i]
            pct         = float(np.mean(window_vals <= val))
            percentiles[i] = pct

        self._atr_percentiles = percentiles

    def _sample_curriculum_start(self, max_start: int) -> int:
        """
        Sample a valid episode start index matching the curriculum stage.
        Falls back to random if no valid indices are found.
        """
        if self._atr_percentiles is None or self._curriculum_stage >= 2:
            return random.randint(MIN_HISTORY, max(MIN_HISTORY, max_start))

        vol_limit = CURRICULUM_STAGE_THRESHOLDS[self._curriculum_stage]

        # Collect valid start indices
        valid = [
            i for i in range(MIN_HISTORY, max(MIN_HISTORY + 1, max_start + 1))
            if i < len(self._atr_percentiles) and
               self._atr_percentiles[i] < vol_limit
        ]

        if not valid:
            logger.debug("No curriculum-valid start points for stage %d — using random.", self._curriculum_stage)
            return random.randint(MIN_HISTORY, max(MIN_HISTORY, max_start))

        return random.choice(valid)

    # ── Action Decoding ────────────────────────────────────────────────────────

    def _decode_action(
        self, action: np.ndarray
    ) -> tuple[int, float, float, float, bool]:
        """
        Decode the 5-dim action vector.

        Returns:
            direction:    -1 (Short), 0 (Flat), 1 (Long)
            leverage:     float in [1, max_leverage]
            sl_pct:       float in [SL_MIN_PCT, SL_MAX_PCT]
            tp_pct:       float in [TP_MIN_PCT, TP_MAX_PCT]
            force_close:  bool — True if action[4] > CLOSE_SIGNAL_THRESH
        """
        raw_dir, lev_norm, sl_norm, tp_norm = (
            float(action[0]), float(action[1]), float(action[2]), float(action[3])
        )
        close_signal = float(action[4]) if len(action) > 4 else 0.0

        if raw_dir > DIRECTION_THRESHOLD:
            direction = 1
        elif raw_dir < -DIRECTION_THRESHOLD:
            direction = -1
        else:
            direction = 0

        leverage = LEVERAGE_MIN + lev_norm * (min(self.max_leverage, LEVERAGE_MAX) - LEVERAGE_MIN)
        leverage = float(np.clip(round(leverage), LEVERAGE_MIN, LEVERAGE_MAX))

        sl_pct = SL_MIN_PCT + sl_norm * (SL_MAX_PCT - SL_MIN_PCT)
        tp_pct = TP_MIN_PCT + tp_norm * (TP_MAX_PCT - TP_MIN_PCT)

        force_close = close_signal > CLOSE_SIGNAL_THRESH

        return direction, leverage, sl_pct, tp_pct, force_close

    # ── Position Management ────────────────────────────────────────────────────

    def _open_position(
        self,
        direction: int,
        price: float,
        leverage: float,
        sl_pct: float,
        tp_pct: float,
    ) -> tuple[float, float]:
        """Open a new position. Returns (notional_size, fee)."""
        risk_amount       = self._equity * self.max_risk_per_trade
        position_notional = min(risk_amount / sl_pct * leverage, self._equity * leverage)
        fee               = position_notional * TAKER_FEE

        self._position_side          = direction
        self._position_entry_price   = price
        self._position_size          = position_notional
        self._position_leverage      = leverage
        self._steps_in_position      = 0

        if direction == 1:   # Long
            self._position_sl = price * (1 - sl_pct)
            self._position_tp = price * (1 + tp_pct)
        else:                 # Short
            self._position_sl = price * (1 + sl_pct)
            self._position_tp = price * (1 - tp_pct)

        self._equity -= fee
        self._num_trades += 1

        logger.debug(
            "OPEN %s %s @ %.2f | Size: %.2f | Lev: %.0fx | SL: %.2f | TP: %.2f | VolRegime: %.2f",
            "LONG" if direction == 1 else "SHORT", self.symbol, price,
            position_notional, leverage, self._position_sl, self._position_tp,
            self._get_volatility_regime(),
        )
        return position_notional, fee

    def _close_position(self, close_price: float) -> tuple[float, float]:
        """Close current position. Returns (realized_pnl, fee)."""
        price_change_pct = (close_price - self._position_entry_price) / self._position_entry_price
        realized_pnl     = price_change_pct * self._position_side * self._position_size
        fee              = self._position_size * TAKER_FEE

        self._equity += realized_pnl - fee

        self._trade_log.append({
            "entry_price":  self._position_entry_price,
            "exit_price":   close_price,
            "side":         self._position_side,
            "leverage":     self._position_leverage,
            "pnl":          realized_pnl,
            "fee":          fee,
            "steps":        self._steps_in_position,
            "vol_regime":   self._get_volatility_regime(),
            "liq_proximity": self._get_liquidation_proximity(close_price),
        })

        logger.debug(
            "CLOSE %s @ %.2f | P&L: %.4f USDT | Fee: %.4f",
            self.symbol, close_price, realized_pnl, fee,
        )

        self._position_side          = 0
        self._position_entry_price   = 0.0
        self._position_size          = 0.0
        self._position_leverage      = 1.0
        self._position_sl            = 0.0
        self._position_tp            = 0.0
        self._steps_in_position      = 0

        return realized_pnl, fee

    def _check_sl(self, price: float) -> bool:
        if self._position_side == 1:
            return price <= self._position_sl
        elif self._position_side == -1:
            return price >= self._position_sl
        return False

    def _check_tp(self, price: float) -> bool:
        if self._position_side == 1:
            return price >= self._position_tp
        elif self._position_side == -1:
            return price <= self._position_tp
        return False

    def _update_unrealized_equity(self, current_price: float) -> None:
        if self._position_side != 0 and self._position_entry_price > 0:
            price_change = (current_price - self._position_entry_price) / self._position_entry_price
            unrealized   = price_change * self._position_side * self._position_size
            nav          = self._equity + unrealized
            if nav > self._peak_equity:
                self._peak_equity = nav

    # ── Price & Termination ───────────────────────────────────────────────────

    def _get_current_price(self) -> float:
        if self._candle_df is not None and self._current_idx < len(self._candle_df):
            return float(self._candle_df["close"].iloc[self._current_idx])
        return 1.0

    def _check_terminated(self) -> bool:
        # Ruin: equity fell below 15% of initial (tighter than v1's 20%)
        if self._equity < self.initial_equity * 0.15:
            logger.info("Episode terminated: ruin (equity=%.2f).", self._equity)
            return True
        if self._candle_df is not None and self._current_idx >= len(self._candle_df) - 1:
            return True
        return False

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self) -> None:
        price   = self._get_current_price()
        vol_reg = self._get_volatility_regime()
        liq_prx = self._get_liquidation_proximity(price)
        print(
            f"[{self.mode.upper()}] {self.symbol} | Price: {price:.2f} | "
            f"Equity: {self._equity:.2f} | Pos: {self._position_side} | "
            f"Vol: {vol_reg:.2f} | LiqProx: {liq_prx:.2f} | "
            f"Trades: {self._num_trades} | Stage: {self._curriculum_stage}"
        )

    def close(self) -> None:
        pass

    # ── External Properties ───────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def trade_log(self) -> list[dict]:
        return list(self._trade_log)

    @property
    def episode_reward_sum(self) -> float:
        return sum(self._episode_rewards)

    @property
    def volatility_regime(self) -> float:
        return self._get_volatility_regime()

    @property
    def portfolio_drawdown(self) -> float:
        return (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)

    def set_candle_df(self, df: pd.DataFrame) -> None:
        """Update the candle data and recompute ATR percentiles."""
        self._candle_df_raw  = df
        self._candle_df      = compute_features(df)
        self._current_idx    = len(self._candle_df) - 1
        self._compute_atr_percentiles()
