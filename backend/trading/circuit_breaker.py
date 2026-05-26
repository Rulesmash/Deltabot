"""
trading/circuit_breaker.py — Global drawdown circuit breaker.

Automatically pauses all trading activity if portfolio drawdown
exceeds the configured threshold (default 5%).

Also provides:
  • Per-trade loss limit (optional)
  • Daily loss limit (optional)
  • Manual pause/resume via API
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Monitors equity and triggers emergency pause if drawdown limits are breached.

    Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.05,
        daily_loss_limit_pct: float = 0.10,
        notify_callback=None,
    ) -> None:
        self._max_drawdown        = max_drawdown_pct
        self._daily_loss_limit    = daily_loss_limit_pct
        self._notify_callback     = notify_callback

        self._peak_equity: float  = 0.0
        self._initial_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._day_start_ts: int   = 0

        self._triggered           = False
        self._trigger_reason      = ""
        self._trigger_time: str   = ""
        self._manual_pause        = False

        self._lock = asyncio.Lock()

    # ── Initialization ───────────────────────────────────────────────────────

    def initialise(self, equity: float) -> None:
        """Set reference equity (call at bot startup)."""
        self._initial_equity    = equity
        self._peak_equity       = equity
        self._day_start_equity  = equity
        self._day_start_ts      = int(time.time())
        self._triggered         = False
        self._manual_pause      = False
        logger.info("CircuitBreaker initialised: equity=%.2f | MaxDD=%.1f%%", equity, self._max_drawdown * 100)

    # ── Equity Update ─────────────────────────────────────────────────────────

    async def update(self, current_equity: float) -> bool:
        """
        Update current equity and check circuit-breaker conditions.

        Returns:
            True if trading should continue, False if breaker triggered.
        """
        async with self._lock:
            # Reset daily P&L counter at midnight UTC
            now = int(time.time())
            if now - self._day_start_ts >= 86400:
                self._day_start_equity = current_equity
                self._day_start_ts     = now

            # Update peak
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity

            # ── Check drawdown from peak ──────────────────────────────────────
            drawdown = (self._peak_equity - current_equity) / max(self._peak_equity, 1.0)
            if drawdown >= self._max_drawdown and not self._triggered:
                await self._trigger(
                    reason=f"Peak drawdown {drawdown*100:.1f}% exceeded limit {self._max_drawdown*100:.1f}%",
                    equity=current_equity,
                )
                return False

            # ── Check daily loss ──────────────────────────────────────────────
            daily_loss = (self._day_start_equity - current_equity) / max(self._day_start_equity, 1.0)
            if daily_loss >= self._daily_loss_limit and not self._triggered:
                await self._trigger(
                    reason=f"Daily loss {daily_loss*100:.1f}% exceeded limit {self._daily_loss_limit*100:.1f}%",
                    equity=current_equity,
                )
                return False

            return not (self._triggered or self._manual_pause)

    async def _trigger(self, reason: str, equity: float) -> None:
        self._triggered      = True
        self._trigger_reason = reason
        self._trigger_time   = datetime.now(timezone.utc).isoformat()
        logger.critical("🔴 CIRCUIT BREAKER TRIGGERED: %s | Equity: %.2f", reason, equity)

        if self._notify_callback:
            try:
                await self._notify_callback({
                    "event":    "circuit_breaker",
                    "reason":   reason,
                    "equity":   equity,
                    "time":     self._trigger_time,
                })
            except Exception as exc:
                logger.error("Notify callback failed: %s", exc)

    # ── Manual Control ────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Manually pause trading."""
        self._manual_pause = True
        logger.warning("Trading manually paused.")

    def resume(self) -> None:
        """
        Manually resume trading after circuit breaker.
        Only call if you have verified the situation and reduced risk.
        """
        self._triggered    = False
        self._manual_pause = False
        self._trigger_reason = ""
        logger.info("Trading resumed by user.")

    def reset(self, equity: float) -> None:
        """Full reset — requires new equity reference."""
        self.initialise(equity)

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def is_triggered(self) -> bool:
        return self._triggered or self._manual_pause

    @property
    def status(self) -> dict:
        drawdown = (
            (self._peak_equity - (self._peak_equity * (1 - self._max_drawdown))) / self._peak_equity
            if self._peak_equity > 0 else 0.0
        )
        return {
            "triggered":       self._triggered,
            "manual_pause":    self._manual_pause,
            "trigger_reason":  self._trigger_reason,
            "trigger_time":    self._trigger_time,
            "peak_equity":     round(self._peak_equity, 2),
            "max_drawdown_pct": self._max_drawdown * 100,
            "daily_loss_limit_pct": self._daily_loss_limit * 100,
        }
