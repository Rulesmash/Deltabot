"""
rl/train.py — Training loop orchestrator for DeltaRL Trader.
              RISK-MITIGATION-FIRST version (v2)

Manages the full training pipeline:
  1. Fetch historical data (configurable lookback)
  2. Configure curriculum learning stage (0=low-vol, 1=medium, 2=full)
  3. Create environment with risk-first reward
  4. Run PPO training episodes in a thread pool executor
  5. Auto-advance curriculum stage based on performance gates
  6. Trigger online fine-tuning after N closed trades
  7. Broadcast risk metrics (Sharpe, Sortino, MaxDD, Calmar) to WebSocket

Curriculum Auto-Advancement Gates:
  Stage 0 → 1: rolling_sharpe > 0.5 AND avg_max_dd < 0.08 over last 50 episodes
  Stage 1 → 2: rolling_sharpe > 1.0 AND avg_max_dd < 0.12 over last 50 episodes
  Manual override available via set_curriculum_stage() API endpoint.

This module is designed to run as a background asyncio task.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from config import Settings
from data.features import compute_features
from data.fetcher import HistoricalFetcher
from data.stream import CandleBuffer
from delta.client import DeltaClient
from rl.agent import RLAgent
from rl.env import TradingEnv
from rl.reward import RewardConfig

logger = logging.getLogger(__name__)

# ── Curriculum advancement gates ──────────────────────────────────────────────

CURRICULUM_GATES = {
    # (stage_from → stage_to): (min_sharpe, max_avg_dd, min_episodes_at_stage)
    0: {"min_sharpe": 0.5,  "max_avg_dd": 0.08, "min_episodes": 50},
    1: {"min_sharpe": 1.0,  "max_avg_dd": 0.12, "min_episodes": 50},
}


class TrainingOrchestrator:
    """
    Manages the background RL training loop with curriculum learning.

    v2 additions:
      - Curriculum stage auto-advancement based on risk-adjusted performance
      - Manual curriculum override via set_curriculum_stage()
      - Risk metrics (Sharpe, Sortino, MaxDD, Calmar) in status dict
      - Reward weight hot-swapping per curriculum stage
    """

    def __init__(
        self,
        settings: Settings,
        agent: RLAgent,
        client: DeltaClient,
        candle_buffer: CandleBuffer | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self._settings          = settings
        self._agent             = agent
        self._client            = client
        self._candle_buffer     = candle_buffer
        self._progress_callback = progress_callback

        self._running           = False
        self._task: asyncio.Task | None = None
        self._symbol            = settings.trading_pairs_list[0]
        self._resolution        = f"{settings.timeframe_minutes}m"
        self._closed_trade_count = 0

        # ── Curriculum state ──────────────────────────────────────────────────
        self._curriculum_stage: int = 0
        self._manual_curriculum_stage: int | None = None  # None = auto
        self._recent_sharpes:   deque[float] = deque(maxlen=50)
        self._recent_max_dds:   deque[float] = deque(maxlen=50)
        self._episodes_at_stage: int = 0

        # ── Status dict (read by /api/rl/status) ─────────────────────────────
        self.status: dict = {
            "state":              "idle",
            "mode":               settings.trading_mode.value,
            "symbol":             self._symbol,
            "epoch":              0,
            "total_steps":        0,
            "mean_reward":        0.0,
            "equity":             10_000.0,
            "num_trades":         0,
            "started_at":         None,
            # v2 risk metrics
            "curriculum_stage":   0,
            "rolling_sharpe":     0.0,
            "rolling_sortino":    0.0,
            "max_dd_episode":     0.0,
            "calmar_ratio":       0.0,
            "volatility_regime":  0.0,
            "portfolio_drawdown": 0.0,
            "safety_triggered":   False,
            "in_safety_zone":     False,
        }

    # ── Control ───────────────────────────────────────────────────────────────

    async def start(
        self,
        mode: str = "simulation",
        total_timesteps: int = 200_000,
        curriculum_stage: int | None = None,
    ) -> None:
        """Start the training loop as a background task."""
        if self._running:
            logger.warning("Training already running.")
            return

        if curriculum_stage is not None:
            self._manual_curriculum_stage = curriculum_stage
            self._curriculum_stage = curriculum_stage

        self._running = True
        self.status["state"]      = "running"
        self.status["started_at"] = datetime.now(timezone.utc).isoformat()
        self.status["mode"]       = mode

        self._task = asyncio.create_task(
            self._training_loop(mode=mode, total_timesteps=total_timesteps)
        )
        logger.info("Training started: mode=%s steps=%d curriculum=%d",
                    mode, total_timesteps, self._curriculum_stage)

    async def stop(self) -> None:
        """Gracefully stop the training loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.status["state"] = "stopped"
        logger.info("Training stopped.")

    def set_curriculum_stage(self, stage: int) -> None:
        """
        Manually override the curriculum stage.
        Effective on the next episode reset.
        """
        self._manual_curriculum_stage = stage
        self._curriculum_stage        = stage
        self._episodes_at_stage       = 0
        self._emit({
            "event":   "curriculum_override",
            "stage":   stage,
            "message": f"Curriculum stage manually set to {stage}",
        })
        logger.info("Curriculum stage manually overridden to %d", stage)

    # ── Main Loop ─────────────────────────────────────────────────────────────

    async def _training_loop(
        self,
        mode: str = "simulation",
        total_timesteps: int = 200_000,
    ) -> None:
        """Background coroutine: fetch data, create env, train with curriculum."""
        try:
            # Step 1: Fetch historical data
            self._emit({"event": "status", "message": "Fetching historical data..."})
            fetcher = HistoricalFetcher(self._client)
            historical_df = await fetcher.fetch(
                symbol=self._symbol,
                resolution=self._resolution,
                lookback_bars=2000,   # more data = better curriculum coverage
            )

            if historical_df.empty:
                self._emit({"event": "error", "message": "Failed to fetch historical data."})
                return

            self._emit({
                "event":   "status",
                "message": f"Loaded {len(historical_df)} candles | Curriculum stage {self._curriculum_stage} | Building environment...",
            })

            # Step 2: Create environment
            env = TradingEnv(
                mode=mode,
                initial_equity=10_000.0,
                candle_df=historical_df,
                delta_client=self._client if mode != "simulation" else None,
                symbol=self._symbol,
                max_leverage=self._settings.max_leverage,
                max_risk_per_trade=self._settings.max_risk_per_trade,
                curriculum_stage=self._curriculum_stage,
            )

            # Step 3: Train with progress callback
            self._emit({
                "event":   "status",
                "message": f"Starting PPO training ({total_timesteps:,} steps, stage {self._curriculum_stage})...",
            })

            def progress_fn(info: dict) -> None:
                """Called after each episode from the SB3 training thread."""
                episode      = info.get("episode", 0)
                sharpe       = info.get("rolling_sharpe", 0.0)
                sortino      = info.get("rolling_sortino", 0.0)
                max_dd       = info.get("max_dd_episode", 0.0)
                equity       = info.get("equity", 0.0)
                in_sz        = info.get("in_safety_zone", False)
                curr_stage   = env._curriculum_stage

                # Track for curriculum gates
                self._recent_sharpes.append(sharpe)
                self._recent_max_dds.append(max_dd)
                self._episodes_at_stage += 1

                # Compute Calmar: equity_return / max_dd
                ep_return     = (equity - 10_000) / 10_000 if equity > 0 else 0.0
                calmar_ratio  = ep_return / max(max_dd, 1e-4) if max_dd > 0 else 0.0

                self.status.update({
                    "epoch":              episode,
                    "total_steps":        info.get("total_steps", 0),
                    "mean_reward":        info.get("rollout_mean", 0.0),
                    "equity":             equity,
                    "num_trades":         info.get("num_trades", 0),
                    "curriculum_stage":   curr_stage,
                    "rolling_sharpe":     round(sharpe, 4),
                    "rolling_sortino":    round(sortino, 4),
                    "max_dd_episode":     round(max_dd * 100, 2),
                    "calmar_ratio":       round(calmar_ratio, 4),
                    "volatility_regime":  round(info.get("volatility_regime", 0.0), 3),
                    "portfolio_drawdown": round(info.get("portfolio_drawdown", 0.0) * 100, 2),
                    "in_safety_zone":     in_sz,
                })
                self._emit({**info, "status": self.status})

                # Auto-advance curriculum (only if not manually locked)
                if self._manual_curriculum_stage is None:
                    self._try_advance_curriculum(env)

            # Run training in a thread pool executor (SB3 is synchronous)
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                None,
                lambda: self._agent.train(
                    env=env,
                    total_timesteps=total_timesteps,
                    progress_callback=progress_fn,
                    checkpoint_every=max(1000, total_timesteps // 20),
                ),
            )

            self.status["state"] = "completed"
            self._emit({"event": "training_complete", "summary": summary})
            logger.info("Training complete: %s", summary)

        except asyncio.CancelledError:
            logger.info("Training loop cancelled.")
        except Exception as exc:
            logger.exception("Training loop error: %s", exc)
            self.status["state"] = "error"
            self._emit({"event": "error", "message": str(exc)})

    # ── Curriculum Auto-Advancement ───────────────────────────────────────────

    def _try_advance_curriculum(self, env: TradingEnv) -> None:
        """
        Check if the agent meets performance gates to advance curriculum stage.
        Called after each episode end.
        """
        current_stage = env._curriculum_stage
        if current_stage >= 2:
            return   # already at max difficulty

        gate = CURRICULUM_GATES.get(current_stage)
        if gate is None:
            return

        enough_episodes = self._episodes_at_stage >= gate["min_episodes"]
        if not enough_episodes:
            return

        avg_sharpe = (
            sum(self._recent_sharpes) / len(self._recent_sharpes)
            if self._recent_sharpes else 0.0
        )
        avg_max_dd = (
            sum(self._recent_max_dds) / len(self._recent_max_dds)
            if self._recent_max_dds else 1.0
        )

        if avg_sharpe >= gate["min_sharpe"] and avg_max_dd <= gate["max_avg_dd"]:
            new_stage = current_stage + 1
            env.set_curriculum_stage(new_stage)
            self._curriculum_stage   = new_stage
            self._episodes_at_stage  = 0
            self._recent_sharpes.clear()
            self._recent_max_dds.clear()

            msg = (
                f"🎓 Curriculum advanced: Stage {current_stage} → {new_stage} | "
                f"Sharpe={avg_sharpe:.2f} ≥ {gate['min_sharpe']} | "
                f"MaxDD={avg_max_dd*100:.1f}% ≤ {gate['max_avg_dd']*100:.0f}%"
            )
            logger.info(msg)
            self._emit({
                "event":      "curriculum_advance",
                "from_stage": current_stage,
                "to_stage":   new_stage,
                "avg_sharpe": round(avg_sharpe, 3),
                "avg_max_dd": round(avg_max_dd, 3),
                "message":    msg,
            })

    def _emit(self, data: dict) -> None:
        """Dispatch event to registered progress callback."""
        if self._progress_callback:
            try:
                self._progress_callback(data)
            except Exception as exc:
                logger.warning("Progress callback error: %s", exc)
