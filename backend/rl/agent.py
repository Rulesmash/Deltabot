"""
rl/agent.py — PPO agent wrapper for DeltaRL Trader.
               RISK-MITIGATION-FIRST version (v2)

Wraps Stable-Baselines3 PPO to provide:
  • Consistent load/save with versioning
  • Prediction with action decoding (5-dim action space)
  • Online fine-tuning from the replay buffer with risk-aware reward
  • Training callbacks for UI progress streaming (Sharpe, Sortino, MaxDD)
  • Conservative safety policy baseline for OOD (out-of-distribution) states

Network architecture: MlpPolicy with net_arch=[256, 256]
  → Tuned for RTX 3060 (12 GB VRAM); fast training, good generalisation

Safety policy:
  When portfolio_drawdown > SAFETY_DD_THRESHOLD or volatility_regime > SAFETY_VOL_THRESHOLD,
  safety_policy_predict() returns a conservative Flat or reduced-leverage action
  instead of the full PPO prediction.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.env import TradingEnv

logger = logging.getLogger(__name__)


# ── Custom Callbacks ──────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    """
    Streams training progress to a callback function for UI updates.
    Reports episode rewards, equity, and trade counts.
    """

    def __init__(
        self,
        progress_fn: Callable[[dict], None] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._progress_fn = progress_fn
        self._episode_count = 0
        self._last_report_time = time.time()

    def _on_step(self) -> bool:
        # Check for episode endings
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done and self._progress_fn:
                self._episode_count += 1
                self._progress_fn({
                    "event":              "episode_end",
                    "episode":            self._episode_count,
                    "total_steps":        self.num_timesteps,
                    "equity":             info.get("equity", 0),
                    "num_trades":         info.get("num_trades", 0),
                    "rollout_mean":       float(self.model.ep_info_buffer[-1]["r"])
                                          if self.model.ep_info_buffer else 0.0,
                    # v2 risk metrics
                    "volatility_regime":  info.get("volatility_regime", 0.0),
                    "portfolio_drawdown": info.get("portfolio_drawdown", 0.0),
                    "rolling_sharpe":     info.get("rolling_sharpe", 0.0),
                    "rolling_sortino":    info.get("rolling_sortino", 0.0),
                    "max_dd_episode":     info.get("max_dd_episode", 0.0),
                    "in_safety_zone":     info.get("in_safety_zone", False),
                    "curriculum_stage":   info.get("curriculum_stage", 0),
                })
        return True  # continue training


class MetricsCallback(BaseCallback):
    """Accumulates training metrics for post-hoc analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int]   = []

    def _on_step(self) -> bool:
        if self.model.ep_info_buffer:
            for ep in self.model.ep_info_buffer:
                if len(self.episode_rewards) == 0 or ep["r"] != self.episode_rewards[-1]:
                    self.episode_rewards.append(ep["r"])
                    self.episode_lengths.append(ep["l"])
        return True


# ── Agent ─────────────────────────────────────────────────────────────────────

class RLAgent:
    """
    Manages the PPO agent lifecycle: create, train, predict, save, load.
    """

    def __init__(
        self,
        model_dir: str = "./models",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self._model_dir    = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model: PPO | None = None

        self._hyperparams = {
            "learning_rate": learning_rate,
            "n_steps":       n_steps,
            "batch_size":    batch_size,
            "n_epochs":      n_epochs,
            "gamma":         gamma,
            "gae_lambda":    gae_lambda,
        }
        self._version: int = 0
        self._total_steps: int = 0
        self._metrics = MetricsCallback()

    # ── Model creation ────────────────────────────────────────────────────────

    def create(self, env: TradingEnv) -> None:
        """Create a new PPO model for the given environment."""
        vec_env = DummyVecEnv([lambda: env])
        self._model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=0,
            tensorboard_log=str(self._model_dir / "tensorboard"),
            policy_kwargs={
                # [256, 256] — RTX 3060 optimised: fast iterations, good generalisation
                # Larger net_arch wastes VRAM on a risk-first task (sparse rewards)
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
                "activation_fn": __import__("torch").nn.Tanh,  # Tanh > ReLU for bounded obs
            },
            **self._hyperparams,
        )
        logger.info("Created new PPO model with hyperparams: %s", self._hyperparams)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        env: TradingEnv,
        total_timesteps: int = 100_000,
        progress_callback: Callable[[dict], None] | None = None,
        checkpoint_every: int = 10_000,
    ) -> dict:
        """
        Train the PPO agent.

        Args:
            env: TradingEnv instance (will be wrapped in DummyVecEnv)
            total_timesteps: Total environment steps to train for
            progress_callback: Called after each episode with metrics
            checkpoint_every: Auto-save checkpoint every N timesteps

        Returns:
            Training summary dict
        """
        if self._model is None:
            self.create(env)

        vec_env = DummyVecEnv([lambda: env])
        self._model.set_env(vec_env)

        callbacks = [self._metrics]

        if progress_callback:
            callbacks.append(ProgressCallback(progress_fn=progress_callback))

        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_every,
            save_path=str(self._model_dir / "checkpoints"),
            name_prefix="ppo_deltarl",
        )
        callbacks.append(checkpoint_cb)

        logger.info("Starting PPO training for %d timesteps...", total_timesteps)
        start_time = time.time()

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,   # continue from previous total_steps
        )

        elapsed = time.time() - start_time
        self._total_steps += total_timesteps

        # Save final model
        self._version += 1
        save_path = self._save_versioned()

        summary = {
            "version":        self._version,
            "total_steps":    self._total_steps,
            "elapsed_seconds": round(elapsed, 1),
            "mean_reward":    float(np.mean(self._metrics.episode_rewards[-100:]))
            if self._metrics.episode_rewards else 0.0,
            "checkpoint":     str(save_path),
        }
        logger.info("Training complete: %s", summary)
        return summary

    def fine_tune(self, env: TradingEnv, timesteps: int = 2048) -> None:
        """
        Online fine-tuning: brief training step after new trade data.
        Called automatically after every N closed trades.
        """
        if self._model is None:
            logger.warning("Cannot fine-tune: no model loaded.")
            return

        vec_env = DummyVecEnv([lambda: env])
        self._model.set_env(vec_env)
        self._model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        logger.debug("Online fine-tune: %d steps.", timesteps)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """
        Get action from the model for a given observation.

        Returns:
            (action_array, extra_info_dict)
        """
        if self._model is None:
            # Return a conservative "Flat" action if no model loaded (5-dim v2)
            return np.array([0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32), {}

        action, _ = self._model.predict(
            observation.reshape(1, -1),
            deterministic=deterministic,
        )
        return action[0], {}

    # ── Safety Policy Baseline ───────────────────────────────────────────────

    def safety_policy_predict(
        self,
        observation: np.ndarray,
        portfolio_drawdown: float = 0.0,
        volatility_regime: float = 0.0,
    ) -> tuple[np.ndarray, bool]:
        """
        Conservative safety policy baseline for out-of-distribution states.

        When the portfolio is in a dangerous state (high drawdown or extreme
        volatility), this policy overrides the PPO action with a safer choice:
          - portfolio_dd > 10% → Flat (close any open position)
          - vol_regime > 85%   → Flat or very low-leverage with tight SL

        Returns:
            (action, was_overridden) — True if safety policy was applied
        """
        from rl.env import SAFETY_DD_THRESHOLD, SAFETY_VOL_THRESHOLD

        # High drawdown: unconditionally go flat
        if portfolio_drawdown > SAFETY_DD_THRESHOLD:
            flat_action = np.array([0.0, 0.0, 0.5, 0.5, 0.9], dtype=np.float32)  # force close
            logger.warning(
                "Safety policy: Flat — portfolio_dd=%.1f%% exceeds threshold %.0f%%",
                portfolio_drawdown * 100, SAFETY_DD_THRESHOLD * 100,
            )
            return flat_action, True

        # Extreme volatility: allow PPO but cap leverage via low lev_norm
        if volatility_regime > SAFETY_VOL_THRESHOLD:
            raw_action, _ = self.predict(observation, deterministic=True)
            # Cap leverage to 3x in extreme vol (lev_norm for 3x = (3-1)/(20-1) ≈ 0.105)
            safe_action = raw_action.copy()
            safe_action[1] = min(safe_action[1], 0.11)   # cap at ~3x
            safe_action[2] = max(safe_action[2], 0.40)   # wider SL (40% of range = 2%)
            logger.debug(
                "Safety policy: Leverage capped at 3x — vol_regime=%.2f", volatility_regime
            )
            return safe_action, True

        # Normal state: use PPO
        action, _ = self.predict(observation, deterministic=True)
        return action, False

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_versioned(self) -> Path:
        """Save model with version number and metadata."""
        path = self._model_dir / f"ppo_v{self._version}"
        self._model.save(str(path))

        meta = {
            "version":     self._version,
            "total_steps": self._total_steps,
            "hyperparams": self._hyperparams,
        }
        (self._model_dir / f"meta_v{self._version}.json").write_text(
            json.dumps(meta, indent=2)
        )
        logger.info("Model saved to %s", path)
        return path

    def save(self, path: str | None = None) -> str:
        """Save model to specified path or auto-versioned path."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        if path:
            self._model.save(path)
            return path
        return str(self._save_versioned())

    def load(self, path: str) -> None:
        """Load a previously trained model."""
        self._model = PPO.load(path)
        logger.info("Model loaded from %s", path)

        # Try to load metadata
        meta_path = Path(path).with_suffix(".json").parent / f"meta_{Path(path).stem.split('_')[-1]}.json"
        if not meta_path.exists():
            meta_path = Path(str(path) + "_meta.json")

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._version     = meta.get("version", 0)
            self._total_steps = meta.get("total_steps", 0)

    def list_checkpoints(self) -> list[dict]:
        """List available model checkpoints."""
        checkpoints = []
        for p in sorted(self._model_dir.glob("ppo_v*.zip")):
            meta_file = self._model_dir / f"meta_{p.stem}.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                checkpoints.append(meta)
            else:
                checkpoints.append({"version": p.stem, "path": str(p)})
        return checkpoints

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> int:
        return self._version

    @property
    def total_steps(self) -> int:
        return self._total_steps
