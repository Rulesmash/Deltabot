"""
alerts/notifier.py — Telegram and Discord alert integration.

Sends notifications for:
  • Trade opened / closed
  • Circuit breaker triggered
  • Training milestones
  • Errors

Configure with TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, and DISCORD_WEBHOOK_URL
in your .env file. Leave blank to disable.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class AlertNotifier:
    """
    Sends alerts to Telegram and/or Discord.
    Gracefully no-ops if credentials are not configured.
    """

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook_url: str = "",
        trading_mode: str = "demo",
    ) -> None:
        self._telegram_token    = telegram_token
        self._telegram_chat_id  = telegram_chat_id
        self._discord_url       = discord_webhook_url
        self._mode_label        = "🟡 DEMO" if trading_mode == "demo" else "🔴 LIVE"

    async def send(self, level: str, message: str) -> None:
        """
        Send an alert at the given level.

        Args:
            level: "info" | "warning" | "critical"
            message: Human-readable alert message
        """
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(level, "📢")
        full_message = f"{emoji} [{self._mode_label}] DeltaRL Trader\n{message}"

        await self._send_telegram(full_message)
        await self._send_discord(level, message)

    async def send_trade(self, trade: dict) -> None:
        """Format and send a trade notification."""
        side = "LONG 📈" if trade.get("side") == 1 else "SHORT 📉"
        pnl  = trade.get("realized_pnl", 0)
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} Trade closed [{self._mode_label}]\n"
            f"Pair: {trade.get('symbol')}\n"
            f"Side: {side} @ {trade.get('leverage', 1):.0f}x\n"
            f"Entry: {trade.get('entry_price', 0):.2f}\n"
            f"Exit:  {trade.get('exit_price', 0):.2f}\n"
            f"P&L:   {pnl:+.2f} USDT"
        )
        await self.send("info" if pnl > 0 else "warning", msg)

    async def send_circuit_breaker(self, reason: str, equity: float) -> None:
        """Send circuit breaker alert."""
        msg = (
            f"🔴 CIRCUIT BREAKER TRIGGERED\n"
            f"Reason: {reason}\n"
            f"Equity: {equity:.2f} USDT\n"
            f"All trading is PAUSED. Check the dashboard."
        )
        await self.send("critical", msg)

    # ── Private methods ───────────────────────────────────────────────────────

    async def _send_telegram(self, text: str) -> None:
        if not self._telegram_token or not self._telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {"chat_id": self._telegram_chat_id, "text": text, "parse_mode": "HTML"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.warning("Telegram send failed: %s", resp.text[:200])
        except Exception as exc:
            logger.warning("Telegram error: %s", exc)

    async def _send_discord(self, level: str, message: str) -> None:
        if not self._discord_url:
            return
        color = {"info": 0x3498DB, "warning": 0xF39C12, "critical": 0xE74C3C}.get(level, 0x95A5A6)
        payload = {
            "embeds": [{
                "title":       f"DeltaRL Trader | {self._mode_label}",
                "description": message,
                "color":       color,
            }]
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self._discord_url, json=payload)
                if resp.status_code not in (200, 204):
                    logger.warning("Discord send failed: %s", resp.text[:200])
        except Exception as exc:
            logger.warning("Discord error: %s", exc)
