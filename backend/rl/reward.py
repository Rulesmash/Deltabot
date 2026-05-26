"""
rl/reward.py — Risk-Mitigation-First Reward Function (v2)

Primary goal: Capital preservation and risk-adjusted returns.
Pure P&L chasing is explicitly penalised when it violates risk rules.

Formula (per step):
  reward = (realized_pnl_pct - fees_pct * 1.5 - slippage_pct)  ← core P&L
           + sharpe_bonus                                         ← rolling Sharpe
           + sortino_bonus                                        ← downside-risk adjusted
           + calmar_term                                          ← return/max-DD ratio
           - drawdown_penalty * max(0, trade_dd - 0.05) ** 2     ← heavy above 5%
           - global_dd_penalty * max(0, global_dd - 0.15) ** 2   ← heavy above 15%
           - volatility_penalty                                   ← high leverage in high vol
           - funding_bleed_penalty                                ← adverse funding cost
           - liquidation_risk_penalty * 10.0                      ← near-liquidation crisis
           - flat_penalty                                         ← anti-lazy (small)
           - overtrading_penalty                                  ← discourages churn

References:
  - Kelly criterion position sizing motivation
  - Moody & Saffell (2001) risk-adjusted RL reward
  - 2025 RL-for-trading survey: importance of Sortino/Calmar over Sharpe alone
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


# ── Reward Configuration ──────────────────────────────────────────────────────

class RewardConfig:
    """
    All reward weights. Can be overridden at runtime via the UI
    config panel (broadcasted to the env via the orchestrator).
    """
    # ── Core P&L ──────────────────────────────────────────────────────────────
    pnl_weight:         float = 1.0       # Weight on normalized realized P&L
    fee_penalty:        float = 1.5       # Amplify fees to discourage overtrading

    # ── Risk-adjusted bonuses ─────────────────────────────────────────────────
    sharpe_weight:      float = 0.15      # Rolling Sharpe bonus
    sortino_weight:     float = 0.10      # Sortino bonus (penalises downside vol only)
    calmar_weight:      float = 0.05      # Calmar ratio (return / max drawdown)
    risk_window:        int   = 30        # Rolling window for Sharpe/Sortino (bars)

    # ── Drawdown penalties (risk-mitigation core) ─────────────────────────────
    trade_dd_threshold:  float = 0.05    # Heavy penalty kicks in above 5% per-trade DD
    global_dd_threshold: float = 0.15   # Hard brake above 15% portfolio DD
    trade_dd_weight:     float = 3.0     # Per-unit penalty beyond trade_dd_threshold
    global_dd_weight:    float = 5.0     # Per-unit penalty beyond global_dd_threshold

    # ── Volatility regime penalty ─────────────────────────────────────────────
    # Penalises high leverage in high-volatility conditions
    volatility_penalty_weight: float = 0.002   # Per unit above 5x in high-vol
    high_vol_regime_threshold: float = 0.6     # vol_regime ∈ [0,1]; above this = high vol

    # ── Funding rate bleed ────────────────────────────────────────────────────
    # Holding a leveraged position when funding is adverse costs money
    funding_penalty_weight: float = 0.001

    # ── Near-liquidation crisis ───────────────────────────────────────────────
    # liq_proximity ∈ [0,1]: 1.0 = price AT liquidation level
    liquidation_penalty_weight: float = 10.0
    liquidation_proximity_threshold: float = 0.7  # trigger above 70% of liq distance

    # ── Flat and overtrading ──────────────────────────────────────────────────
    flat_penalty:          float = 0.001    # Small: just enough to prevent total laziness
    flat_allowed_in_high_vol: bool = True   # If True, NO flat penalty in high-vol regime
    overtrading_threshold: int   = 100      # Trades per episode before overtrading kicks in
    overtrading_penalty:   float = 0.002   # Per-trade above threshold

    # ── Profit quality filter ─────────────────────────────────────────────────
    # Only grant full Sharpe/Sortino bonus when trade respected risk limits
    respected_risk_bonus:  float = 0.02    # Small bonus for profitable + risk-respecting trades

    # ── Clipping ──────────────────────────────────────────────────────────────
    max_reward: float = 10.0
    min_reward: float = -10.0


CFG = RewardConfig()


class RewardCalculator:
    """
    Stateful, risk-aware reward calculator.

    Tracks:
      - Rolling return history (for Sharpe/Sortino)
      - Peak equity and max drawdown (for Calmar)
      - Trade count (for overtrading penalty)
      - Per-episode max drawdown

    Update the config at runtime via set_config() for curriculum adjustment.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self._cfg = config or CFG
        self._return_history: deque[float] = deque(maxlen=self._cfg.risk_window)
        self._neg_returns:    deque[float] = deque(maxlen=self._cfg.risk_window)   # Sortino
        self._peak_equity:    float = 1.0
        self._max_drawdown:   float = 0.0    # worst DD this episode
        self._min_equity:     float = 1.0    # for Calmar denominator
        self._trade_count:    int   = 0
        self._step:           int   = 0

    def reset(self, initial_equity: float = 1.0) -> None:
        """Reset all state at the start of a new episode."""
        self._return_history.clear()
        self._neg_returns.clear()
        self._peak_equity  = initial_equity
        self._max_drawdown = 0.0
        self._min_equity   = initial_equity
        self._trade_count  = 0
        self._step         = 0

    def update_peak(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity
        if equity < self._min_equity:
            self._min_equity = equity

    def set_config(self, cfg: RewardConfig) -> None:
        """Hot-swap reward config at runtime (curriculum stage change)."""
        self._cfg = cfg

    # ── Main compute ─────────────────────────────────────────────────────────

    def compute(
        self,
        *,
        realized_pnl:     float,     # Realized P&L this step (USDT)
        fees:             float,     # Fees paid (USDT)
        slippage:         float,     # Slippage cost (USDT)
        equity:           float,     # Current equity
        leverage:         float,     # Leverage used (1–20)
        action_direction: int,       # -1=Short, 0=Flat, 1=Long
        position_closed:  bool,      # Was a position closed this step?
        # ── Risk context (new v2 inputs) ─────────────────────────────────────
        volatility_regime:   float = 0.0,  # 0=low, 1=high vol
        funding_rate:        float = 0.0,  # Adverse funding rate exposure (signed)
        liquidation_proximity: float = 0.0,  # 0=safe, 1=at liquidation
        open_pnl_pct:       float = 0.0,  # Unrealized P&L / equity
        is_profitable_trade: bool = False, # Was the just-closed trade profitable?
        num_trades_episode:  int   = 0,   # Trades opened so far this episode
    ) -> tuple[float, dict]:
        """
        Compute the risk-mitigation-first reward.

        Returns:
            (reward, breakdown_dict)
        """
        self._step += 1
        self.update_peak(equity)

        # ── 1. Core P&L component ─────────────────────────────────────────────
        net_pnl_pct = (
            realized_pnl
            - fees       * self._cfg.fee_penalty
            - slippage
        ) / max(equity, 1.0)

        pnl_reward = net_pnl_pct * self._cfg.pnl_weight

        # Track returns for risk-adjusted stats
        self._return_history.append(net_pnl_pct)
        if net_pnl_pct < 0:
            self._neg_returns.append(net_pnl_pct)

        # ── 2. Sharpe bonus ────────────────────────────────────────────────────
        sharpe_reward = 0.0
        if len(self._return_history) >= 10:
            rets = np.array(self._return_history)
            mean = rets.mean()
            std  = rets.std() + 1e-8
            # Annualise for 5-min bars: 252 trading days × 24h × 12 bars/h
            sharpe = mean / std * math.sqrt(252 * 288)
            sharpe_reward = float(np.tanh(sharpe * 0.1)) * self._cfg.sharpe_weight

        # ── 3. Sortino bonus (penalises downside variance only) ─────────────────
        sortino_reward = 0.0
        if len(self._return_history) >= 10:
            rets         = np.array(self._return_history)
            mean         = rets.mean()
            neg          = np.array(self._neg_returns) if self._neg_returns else np.array([0.0])
            downside_std = math.sqrt((neg ** 2).mean()) + 1e-8
            sortino      = mean / downside_std * math.sqrt(252 * 288)
            sortino_reward = float(np.tanh(sortino * 0.05)) * self._cfg.sortino_weight

        # ── 4. Calmar term (return / max_drawdown — episode-level) ─────────────
        calmar_reward = 0.0
        if self._peak_equity > 0 and self._step > 20:
            dd_so_far = (self._peak_equity - self._min_equity) / max(self._peak_equity, 1.0)
            self._max_drawdown = max(self._max_drawdown, dd_so_far)
            if self._max_drawdown > 1e-4:
                ep_return = (equity / max(self._peak_equity - (self._peak_equity - self._min_equity), 1.0)) - 1.0
                calmar    = ep_return / max(self._max_drawdown, 1e-4)
                calmar_reward = float(np.clip(calmar * 0.1, -0.2, 0.2)) * self._cfg.calmar_weight

        # ── 5. Per-trade drawdown penalty (risk-first core) ────────────────────
        trade_dd_penalty = 0.0
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - equity) / max(self._peak_equity, 1.0)
            excess_dd  = max(0.0, current_dd - self._cfg.trade_dd_threshold)
            # Quadratic: small violations are lightly penalised, large ones harshly
            trade_dd_penalty = (excess_dd ** 2) * self._cfg.trade_dd_weight

        # ── 6. Global drawdown penalty (emergency brake above 15%) ─────────────
        global_dd_penalty = 0.0
        if self._peak_equity > 0:
            global_dd  = (self._peak_equity - equity) / max(self._peak_equity, 1.0)
            excess_gdd = max(0.0, global_dd - self._cfg.global_dd_threshold)
            if excess_gdd > 0:
                # Very steep penalty — agent should strongly avoid this zone
                global_dd_penalty = (excess_gdd ** 2) * self._cfg.global_dd_weight

        # ── 7. Volatility regime penalty ────────────────────────────────────────
        # High leverage in high-volatility is dangerous — penalise it
        volatility_penalty = 0.0
        if volatility_regime >= self._cfg.high_vol_regime_threshold and leverage > 5:
            vol_excess_lev = leverage - 5
            volatility_penalty = vol_excess_lev * self._cfg.volatility_penalty_weight * volatility_regime

        # ── 8. Funding rate bleed penalty ───────────────────────────────────────
        # Holding a leveraged position when funding is adverse
        funding_penalty = 0.0
        if abs(funding_rate) > 0 and action_direction != 0:
            # Positive funding_rate means longs pay shorts; negative means shorts pay longs
            # funding_rate > 0 and long = adverse; funding_rate < 0 and short = adverse
            is_adverse = (funding_rate > 0 and action_direction == 1) or \
                         (funding_rate < 0 and action_direction == -1)
            if is_adverse:
                funding_penalty = abs(funding_rate) * leverage * self._cfg.funding_penalty_weight

        # ── 9. Liquidation risk penalty ─────────────────────────────────────────
        # Very high cost to approaching the liquidation price
        liquidation_penalty = 0.0
        if liquidation_proximity >= self._cfg.liquidation_proximity_threshold:
            # Quadratic above threshold so the agent learns to avoid this zone entirely
            excess_prox = liquidation_proximity - self._cfg.liquidation_proximity_threshold
            liquidation_penalty = (excess_prox ** 2) * self._cfg.liquidation_penalty_weight

        # ── 10. Flat penalty (anti-lazy) ────────────────────────────────────────
        flat_penalty = 0.0
        if action_direction == 0 and not position_closed:
            # In high-vol regimes, flat is the CORRECT action — don't penalise
            if not (self._cfg.flat_allowed_in_high_vol and volatility_regime >= self._cfg.high_vol_regime_threshold):
                flat_penalty = self._cfg.flat_penalty

        # ── 11. Overtrading penalty ──────────────────────────────────────────────
        overtrading_penalty = 0.0
        if position_closed:
            self._trade_count += 1
        if self._trade_count > self._cfg.overtrading_threshold:
            overtrading_penalty = self._cfg.overtrading_penalty

        # ── 12. Profit quality bonus ─────────────────────────────────────────────
        # Small bonus: profitable trade that respected risk limits
        quality_bonus = 0.0
        if is_profitable_trade and position_closed:
            current_dd = (self._peak_equity - equity) / max(self._peak_equity, 1.0)
            if current_dd < self._cfg.trade_dd_threshold:
                quality_bonus = self._cfg.respected_risk_bonus

        # ── Combine ──────────────────────────────────────────────────────────────
        raw_reward = (
            pnl_reward
            + sharpe_reward
            + sortino_reward
            + calmar_reward
            + quality_bonus
            - trade_dd_penalty
            - global_dd_penalty
            - volatility_penalty
            - funding_penalty
            - liquidation_penalty
            - flat_penalty
            - overtrading_penalty
        )

        reward = float(np.clip(raw_reward, self._cfg.min_reward, self._cfg.max_reward))

        breakdown = {
            "pnl_reward":           round(pnl_reward, 6),
            "sharpe_reward":        round(sharpe_reward, 6),
            "sortino_reward":       round(sortino_reward, 6),
            "calmar_reward":        round(calmar_reward, 6),
            "quality_bonus":        round(quality_bonus, 6),
            "trade_dd_penalty":     round(trade_dd_penalty, 6),
            "global_dd_penalty":    round(global_dd_penalty, 6),
            "volatility_penalty":   round(volatility_penalty, 6),
            "funding_penalty":      round(funding_penalty, 6),
            "liquidation_penalty":  round(liquidation_penalty, 6),
            "flat_penalty":         round(flat_penalty, 6),
            "overtrading_penalty":  round(overtrading_penalty, 6),
            "total_reward":         round(reward, 6),
            "net_pnl_pct":          round(net_pnl_pct * 100, 4),
            "max_drawdown_episode": round(self._max_drawdown * 100, 2),
        }
        return reward, breakdown

    # ── Observable stats ─────────────────────────────────────────────────────

    @property
    def max_drawdown_episode(self) -> float:
        """Max drawdown seen in the current episode (fraction)."""
        return self._max_drawdown

    @property
    def rolling_sharpe(self) -> float:
        if len(self._return_history) < 5:
            return 0.0
        rets = np.array(self._return_history)
        return float(rets.mean() / (rets.std() + 1e-8) * math.sqrt(252 * 288))

    @property
    def rolling_sortino(self) -> float:
        if len(self._return_history) < 5:
            return 0.0
        rets  = np.array(self._return_history)
        negs  = np.array(self._neg_returns) if self._neg_returns else np.array([0.0])
        dstd  = math.sqrt((negs ** 2).mean()) + 1e-8
        return float(rets.mean() / dstd * math.sqrt(252 * 288))
