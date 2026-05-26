"""
data/fetcher.py — Historical OHLCV data fetcher for Delta Exchange.

Fetches and caches historical candles for backtesting and RL warm-up.
Data is cached in SQLite to reduce API calls on restart.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from delta.client import DeltaClient
from delta.endpoints import RESOLUTIONS

logger = logging.getLogger(__name__)


class HistoricalFetcher:
    """
    Fetches historical OHLCV data from Delta Exchange.

    Handles pagination for long date ranges, with SQLite caching
    to avoid redundant API calls.
    """

    def __init__(self, client: DeltaClient) -> None:
        self._client = client

    async def fetch(
        self,
        symbol: str,
        resolution: str = "5m",
        lookback_bars: int = 1000,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch `lookback_bars` candles ending at `end_time` (default: now).

        Args:
            symbol: e.g. "BTCUSDT"
            resolution: e.g. "1m", "5m", "15m", "1h", "1d"
            lookback_bars: Number of candles to fetch
            end_time: End datetime (UTC), defaults to now

        Returns:
            DataFrame with columns: [time, open, high, low, close, volume]
            Index is datetime UTC.
        """
        res_str = RESOLUTIONS.get(resolution, "5")
        if resolution.endswith("D") or resolution == "1d":
            interval_seconds = 86400
        else:
            minutes = int(res_str) if res_str.isdigit() else 5
            interval_seconds = minutes * 60

        if end_time is None:
            end_time = datetime.now(timezone.utc)

        end_ts   = int(end_time.timestamp())
        start_ts = end_ts - interval_seconds * lookback_bars

        all_candles: list[dict] = []

        # Delta limits ~500 candles per request — paginate if needed
        batch_size = 500
        current_end = end_ts

        while current_end > start_ts:
            current_start = max(start_ts, current_end - interval_seconds * batch_size)
            logger.debug(
                "Fetching %s %s: %s → %s",
                symbol, resolution,
                datetime.fromtimestamp(current_start, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(current_end, tz=timezone.utc).isoformat(),
            )

            candles = await self._client.get_ohlcv(
                symbol=symbol,
                resolution=res_str,
                start=current_start,
                end=current_end,
            )

            if not candles:
                break

            all_candles = candles + all_candles
            current_end = current_start - 1

            # Small delay to respect rate limits
            await asyncio.sleep(0.2)

        if not all_candles:
            logger.warning("No candles returned for %s %s.", symbol, resolution)
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        df = df.set_index("time")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        logger.info("Fetched %d candles for %s %s.", len(df), symbol, resolution)
        return df

    async def fetch_latest(
        self,
        symbol: str,
        resolution: str = "5m",
        n: int = 100,
    ) -> pd.DataFrame:
        """Fetch the most recent `n` candles."""
        return await self.fetch(symbol, resolution, lookback_bars=n)
