"""
delta/websocket_client.py — Real-time WebSocket client for Delta Exchange.

Manages:
  • Public channels: candlestick, ticker, mark_price
  • Private channels (auth required): orders, positions, fills, balance
  • Automatic reconnection with exponential back-off
  • Message dispatching via callback registry

Usage:
    ws = DeltaWebSocket(api_key="...", api_secret="...", url="wss://...")
    ws.subscribe_public("candlestick_5", "BTCUSDT", on_candle)
    ws.subscribe_private("orders", on_order)
    await ws.connect()
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Coroutine

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

# Type alias for async callback handlers
AsyncCallback = Callable[[dict], Coroutine[Any, Any, None]]


class DeltaWebSocket:
    """
    Async WebSocket client for Delta Exchange real-time data.

    Subscribes to both public market data channels and private
    account channels. Callbacks are dispatched per channel + symbol.
    """

    def __init__(
        self,
        url: str,
        api_key: str = "",
        api_secret: str = "",
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._api_secret = api_secret.encode() if api_secret else b""
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay

        # Registry: channel_key → list of callback functions
        # channel_key = "{channel}:{symbol}" for public, "{channel}" for private
        self._callbacks: dict[str, list[AsyncCallback]] = defaultdict(list)

        # Pending subscriptions to re-send after reconnect
        self._subscriptions: list[dict] = []

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        self._send_queue: asyncio.Queue = asyncio.Queue()

    # ── Subscription API ──────────────────────────────────────────────────────

    def subscribe_public(
        self,
        channel: str,
        symbol: str,
        callback: AsyncCallback,
    ) -> None:
        """
        Subscribe to a public channel for a given symbol.

        Args:
            channel: e.g. "candlestick_5", "ticker", "mark_price"
            symbol: e.g. "BTCUSDT"
            callback: Async function receiving the message dict
        """
        channel_key = f"{channel}:{symbol}"
        self._callbacks[channel_key].append(callback)
        payload = {
            "type": "subscribe",
            "payload": {
                "channels": [{"name": channel, "symbols": [symbol]}]
            },
        }
        self._subscriptions.append(payload)
        if self._ws:
            asyncio.create_task(self._send(payload))

    def subscribe_private(self, channel: str, callback: AsyncCallback) -> None:
        """
        Subscribe to a private account channel (requires authentication).

        Args:
            channel: e.g. "orders", "positions", "user_fills", "user_balance"
            callback: Async function receiving the message dict
        """
        self._callbacks[channel].append(callback)
        # Private channels are auto-subscribed after auth; just register callback

    def on_message(self, channel_key: str, callback: AsyncCallback) -> None:
        """Generic callback registration."""
        self._callbacks[channel_key].append(callback)

    # ── Connection Management ─────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Start the WebSocket connection loop with automatic reconnection.
        This coroutine runs indefinitely until stop() is called.
        """
        self._running = True
        delay = self._reconnect_delay

        while self._running:
            try:
                logger.info("Connecting to Delta WebSocket: %s", self._url)
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                ) as ws:
                    self._ws = ws
                    delay = self._reconnect_delay  # reset on successful connect
                    logger.info("WebSocket connected.")

                    # Authenticate if credentials provided
                    if self._api_key:
                        await self._authenticate(ws)

                    # Re-subscribe to all channels
                    for subscription in self._subscriptions:
                        await self._send_raw(ws, subscription)

                    # Start concurrent send / receive loops
                    await asyncio.gather(
                        self._receive_loop(ws),
                        self._send_loop(ws),
                    )

            except ConnectionClosed as exc:
                logger.warning("WebSocket closed: %s", exc)
            except Exception as exc:
                logger.error("WebSocket error: %s", exc)
            finally:
                self._ws = None

            if not self._running:
                break

            logger.info("Reconnecting in %.1fs...", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_reconnect_delay)

    async def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()

    # ── Authentication ────────────────────────────────────────────────────────

    async def _authenticate(self, ws: Any) -> None:
        """
        Send HMAC authentication frame for private channel access.
        Signature = HMAC-SHA256(secret, "GET" + timestamp + "/live")
        """
        timestamp = str(int(time.time()))
        path = "/live"
        message = "GET" + timestamp + path
        signature = hmac.new(self._api_secret, message.encode(), hashlib.sha256).hexdigest()

        auth_payload = {
            "type": "auth",
            "payload": {
                "api-key": self._api_key,
                "signature": signature,
                "timestamp": timestamp,
            },
        }
        await self._send_raw(ws, auth_payload)
        logger.info("WebSocket auth payload sent.")

    # ── Message Loops ─────────────────────────────────────────────────────────

    async def _receive_loop(self, ws: Any) -> None:
        """Receive messages and dispatch to registered callbacks."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
                await self._dispatch(msg)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message received: %s", raw[:200])
            except Exception as exc:
                logger.error("Dispatch error: %s", exc)

    async def _send_loop(self, ws: Any) -> None:
        """Drain the outgoing queue and send messages."""
        while True:
            payload = await self._send_queue.get()
            await self._send_raw(ws, payload)
            self._send_queue.task_done()

    async def _send(self, payload: dict) -> None:
        """Queue a message for sending (thread-safe)."""
        await self._send_queue.put(payload)

    async def _send_raw(self, ws: Any, payload: dict) -> None:
        """Directly send a JSON message."""
        await ws.send(json.dumps(payload))

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, msg: dict) -> None:
        """
        Route an incoming message to all matching callbacks.

        Delta Exchange messages have a "type" field (e.g. "candlestick_1m"),
        and a "symbol" field for market data messages.
        """
        msg_type = msg.get("type", "")
        symbol = msg.get("symbol", "")

        # Try symbol-scoped key first, then channel-only key
        keys_to_try = []
        if symbol:
            keys_to_try.append(f"{msg_type}:{symbol}")
        keys_to_try.append(msg_type)

        # Also handle auth responses
        if msg_type in ("auth", "subscriptions", "error"):
            if msg_type == "auth" and msg.get("success"):
                logger.info("WebSocket authenticated successfully.")
            elif msg_type == "error":
                logger.error("WebSocket server error: %s", msg)
            return

        dispatched = False
        for key in keys_to_try:
            if key in self._callbacks:
                for cb in self._callbacks[key]:
                    try:
                        await cb(msg)
                    except Exception as exc:
                        logger.error("Callback error for %s: %s", key, exc)
                dispatched = True

        if not dispatched and msg_type not in ("subscriptions", "heartbeat"):
            logger.debug("No handler for message type=%s symbol=%s", msg_type, symbol)
