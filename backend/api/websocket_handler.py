"""
api/websocket_handler.py — FastAPI WebSocket hub for real-time frontend updates.

Manages:
  • Multiple concurrent frontend connections
  • Broadcast of training progress, candle data, trade events
  • Message typing for structured frontend consumption
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages multiple WebSocket client connections."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WebSocket client connected. Total: %d", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info("WebSocket client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected clients."""
        if not self._connections:
            return

        payload = json.dumps(message)
        dead: list[WebSocket] = []

        async with self._lock:
            connections = list(self._connections)

        for ws in connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            await self.disconnect(ws)

    async def send_to(self, ws: WebSocket, message: dict) -> None:
        """Send a message to a single client."""
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            await self.disconnect(ws)

    @property
    def num_clients(self) -> int:
        return len(self._connections)


# Singleton manager shared across all routes
ws_manager = ConnectionManager()


def emit(event_type: str, data: Any) -> None:
    """
    Synchronous wrapper to queue a broadcast message.
    Call this from non-async contexts (e.g. training thread callbacks).
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(
            ws_manager.broadcast({"type": event_type, "data": data, "ts": _now()})
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def ws_endpoint(ws: WebSocket) -> None:
    """
    WebSocket endpoint handler. Mount at /ws in main.py.
    Keeps the connection alive and handles incoming control messages.
    """
    await ws_manager.connect(ws)
    await ws_manager.send_to(ws, {
        "type": "connected",
        "data": {"message": "DeltaRL Trader WebSocket connected"},
        "ts": _now(),
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                cmd = msg.get("cmd", "")
                if cmd == "ping":
                    await ws_manager.send_to(ws, {"type": "pong", "ts": _now()})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await ws_manager.disconnect(ws)
