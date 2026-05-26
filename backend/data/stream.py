"""
data/stream.py — Real-time tick-to-candle aggregator.

Aggregates live WebSocket trade/ticker messages into OHLCV candles
and appends them to the live DataFrame used by the RL agent.

Also maintains a rolling window of the last N candles so the agent
always has fresh feature data without re-fetching from the REST API.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CandleBuffer:
    """
    Maintains a rolling window of OHLCV candles for a single symbol.

    • New candles are appended from the WebSocket stream.
    • The buffer is kept at `max_size` candles (FIFO eviction).
    • Provides thread-safe access via asyncio.Lock.
    """

    def __init__(self, symbol: str, resolution_minutes: int, max_size: int = 1000) -> None:
        self.symbol = symbol
        self.resolution_minutes = resolution_minutes
        self.max_size = max_size

        # Deque of candle dicts {time, open, high, low, close, volume}
        self._candles: deque[dict] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

        # Tracks the in-progress (current, unclosed) candle
        self._current_candle: dict | None = None
        self._current_bar_ts: int | None = None  # start of current bar (unix seconds)

        # Callbacks to notify when a new closed candle is added
        self._on_new_candle: list = []

    def add_on_candle_callback(self, cb) -> None:
        self._on_new_candle.append(cb)

    async def seed(self, historical_df: pd.DataFrame) -> None:
        """
        Pre-fill the buffer with historical candles.

        Args:
            historical_df: DataFrame with DatetimeTZDtype index and
                           [open, high, low, close, volume] columns.
        """
        async with self._lock:
            self._candles.clear()
            for ts, row in historical_df.iterrows():
                self._candles.append({
                    "time":   int(ts.timestamp()),
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row["volume"]),
                })
        logger.info("CandleBuffer seeded with %d historical candles for %s.", len(self._candles), self.symbol)

    async def on_ticker(self, msg: dict) -> None:
        """
        Handle an incoming ticker/mark-price WebSocket message.
        Updates the current open bar, closing it when the resolution period elapses.

        Delta ticker message example:
          {"type": "ticker", "symbol": "BTCUSDT", "mark_price": "67500.5",
           "timestamp": 1714000000000}
        """
        price_str = msg.get("mark_price") or msg.get("close") or msg.get("last_price")
        if price_str is None:
            return

        price = float(price_str)
        # Delta timestamps can be milliseconds or seconds — normalise
        raw_ts = msg.get("timestamp", 0)
        ts_seconds = raw_ts // 1000 if raw_ts > 1e10 else raw_ts

        bar_seconds = self.resolution_minutes * 60
        bar_start   = (ts_seconds // bar_seconds) * bar_seconds

        async with self._lock:
            if self._current_bar_ts != bar_start:
                # New bar started — close the previous one if it exists
                if self._current_candle is not None:
                    self._candles.append(self._current_candle)
                    asyncio.create_task(self._notify_new_candle(self._current_candle))

                # Open fresh bar
                self._current_bar_ts = bar_start
                self._current_candle = {
                    "time":   bar_start,
                    "open":   price,
                    "high":   price,
                    "low":    price,
                    "close":  price,
                    "volume": 0.0,
                }
            else:
                # Update current bar
                c = self._current_candle
                c["high"]  = max(c["high"], price)
                c["low"]   = min(c["low"], price)
                c["close"] = price

    async def on_candlestick(self, msg: dict) -> None:
        """
        Handle a candlestick update message from the Delta WebSocket.
        Delta sends both "open" bars (candle_type=1) and "close" bars (candle_type=0).
        """
        candle_type = msg.get("candle_type", 1)  # 0=closed, 1=open
        data = {
            "time":   int(msg.get("time",   0)),
            "open":   float(msg.get("open",  0)),
            "high":   float(msg.get("high",  0)),
            "low":    float(msg.get("low",   0)),
            "close":  float(msg.get("close", 0)),
            "volume": float(msg.get("volume", 0)),
        }

        async with self._lock:
            if candle_type == 0:
                # Closed candle — append to history
                if self._candles and self._candles[-1]["time"] == data["time"]:
                    self._candles[-1] = data  # replace the open bar
                else:
                    self._candles.append(data)
                asyncio.create_task(self._notify_new_candle(data))
            else:
                # Open (live) bar — update current candle without appending
                self._current_candle = data
                self._current_bar_ts = data["time"]

    async def _notify_new_candle(self, candle: dict) -> None:
        for cb in self._on_new_candle:
            try:
                await cb(self.symbol, candle)
            except Exception as exc:
                logger.error("Candle callback error: %s", exc)

    async def to_dataframe(self, include_current: bool = False) -> pd.DataFrame:
        """
        Return the current buffer as a DataFrame.
        Optionally includes the unfinished current bar.
        """
        async with self._lock:
            candles = list(self._candles)
            if include_current and self._current_candle:
                candles.append(self._current_candle)

        if not candles:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def __len__(self) -> int:
        return len(self._candles)

    @property
    def latest_close(self) -> float | None:
        if self._current_candle:
            return self._current_candle["close"]
        if self._candles:
            return self._candles[-1]["close"]
        return None
