"""
delta/client.py — Authenticated REST client for Delta Exchange.

Handles:
  • HMAC-SHA256 request signing (per Delta Exchange spec)
  • Automatic demo/live endpoint switching
  • Rate-limit aware retries with exponential back-off
  • Clean typed response dicts

Usage:
    from delta.client import DeltaClient
    client = DeltaClient(api_key="...", api_secret="...", base_url="...")
    ticker = await client.get_ticker("BTCUSDT")
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any

import httpx

from delta.endpoints import (
    CANDLES,
    CANCEL_ALL,
    CANCEL_ORDER,
    FILLS,
    ORDER_BY_ID,
    ORDERS,
    POSITIONS,
    PRODUCTS,
    TICKER,
    TICKERS,
    WALLET,
)

logger = logging.getLogger(__name__)


class DeltaAPIError(Exception):
    """Raised when the Delta Exchange API returns an error response."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{status_code}] {error_code}: {message}")


class DeltaClient:
    """
    Async REST client for Delta Exchange.

    All methods are coroutines. Use within an async context.
    Signing follows the Delta Exchange HMAC specification:
        signature = HMAC-SHA256( method + timestamp + path + query_string + body )
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret.encode()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "DeltaClient":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Request Signing ───────────────────────────────────────────────────────

    def _sign(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body: str = "",
    ) -> dict[str, str]:
        """
        Build authentication headers per Delta Exchange signing spec.
        Signature = HMAC-SHA256(secret, method + timestamp + path + query_string + body)
        """
        timestamp = str(int(time.time()))
        # Delta uses: method is uppercase, path includes leading slash
        message = method.upper() + timestamp + path + query_string + body
        signature = hmac.new(self._api_secret, message.encode(), hashlib.sha256).hexdigest()
        return {
            "api-key": self._api_key,
            "timestamp": timestamp,
            "signature": signature,
        }

    # ── HTTP Helpers ──────────────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
        auth: bool = True,
    ) -> Any:
        """
        Execute an HTTP request with optional signing and retry logic.

        Returns the parsed JSON response or raises DeltaAPIError.
        """
        if self._client is None:
            raise RuntimeError("DeltaClient has not been started. Call await client.start() first.")

        import json as _json

        query_string = ""
        if params:
            from urllib.parse import urlencode
            query_string = urlencode(params)

        body_str = _json.dumps(json_body) if json_body else ""

        headers = {}
        if auth and self._api_key:
            headers = self._sign(method, path, query_string, body_str)

        url = path + ("?" + query_string if query_string else "")

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body_str.encode() if body_str else None,
                )

                if response.status_code == 429:
                    # Rate limited — wait and retry
                    retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                    logger.warning("Rate limited. Waiting %.1fs before retry %d.", retry_after, attempt)
                    await asyncio.sleep(retry_after)
                    continue

                data = response.json()

                if response.status_code >= 400:
                    error_code = data.get("error", {}).get("code", "UNKNOWN")
                    message = data.get("error", {}).get("context", str(data))
                    raise DeltaAPIError(response.status_code, error_code, message)

                return data

            except httpx.RequestError as exc:
                if attempt == self._max_retries:
                    raise
                wait = 2 ** attempt
                logger.warning("Request error (attempt %d/%d): %s. Retrying in %ds.", attempt, self._max_retries, exc, wait)
                await asyncio.sleep(wait)

        raise RuntimeError("Max retries exceeded.")

    # ── Public Endpoints ──────────────────────────────────────────────────────

    async def get_products(self) -> list[dict]:
        """List all available products (perpetual futures, etc.)."""
        data = await self._request("GET", PRODUCTS, auth=False)
        return data.get("result", [])

    async def get_ticker(self, symbol: str) -> dict:
        """Get real-time ticker for a single symbol."""
        path = TICKER.format(symbol=symbol)
        data = await self._request("GET", path, auth=False)
        return data.get("result", {})

    async def get_all_tickers(self) -> list[dict]:
        """Get tickers for all products."""
        data = await self._request("GET", TICKERS, auth=False)
        return data.get("result", [])

    async def get_ohlcv(
        self,
        symbol: str,
        resolution: str = "5",
        start: int | None = None,
        end: int | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Fetch historical OHLCV candles.

        Args:
            symbol: e.g. "BTCUSDT"
            resolution: candle interval in minutes: "1","3","5","15","30","60","1D"
            start: Unix epoch seconds (inclusive)
            end: Unix epoch seconds (inclusive)
            limit: max candles to return

        Returns list of dicts with keys: time, open, high, low, close, volume
        """
        params: dict[str, Any] = {"symbol": symbol, "resolution": resolution}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        data = await self._request("GET", CANDLES, params=params, auth=False)
        candles = data.get("result", {})

        # Delta returns parallel arrays; zip them into dicts
        if isinstance(candles, dict) and "t" in candles:
            return [
                {
                    "time":   candles["t"][i],
                    "open":   float(candles["o"][i]),
                    "high":   float(candles["h"][i]),
                    "low":    float(candles["l"][i]),
                    "close":  float(candles["c"][i]),
                    "volume": float(candles["v"][i]),
                }
                for i in range(len(candles["t"]))
            ]
        return []

    # ── Private Endpoints ─────────────────────────────────────────────────────

    async def get_wallet_balance(self, asset: str = "USDT") -> dict:
        """Get wallet balance for the specified asset."""
        data = await self._request("GET", WALLET)
        balances = data.get("result", [])
        for bal in balances:
            if bal.get("currency_symbol") == asset or bal.get("asset_symbol") == asset:
                return bal
        return {}

    async def get_positions(self) -> list[dict]:
        """Get all open positions."""
        data = await self._request("GET", POSITIONS)
        return data.get("result", {}).get("result", [])

    async def get_position(self, symbol: str) -> dict | None:
        """Get open position for a specific symbol."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.get("product_symbol") == symbol:
                return pos
        return None

    async def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market_order",
        limit_price: float | None = None,
        leverage: int = 1,
        stop_price: float | None = None,
        trail_amount: float | None = None,
        bracket_stop_loss_price: float | None = None,
        bracket_stop_loss_limit_price: float | None = None,
        bracket_take_profit_price: float | None = None,
        bracket_take_profit_limit_price: float | None = None,
        time_in_force: str = "gtc",
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> dict:
        """
        Place an order on Delta Exchange.

        Supports market, limit, stop-limit orders, and bracket orders
        (entry + SL + TP submitted in a single request).

        Args:
            symbol: Product symbol, e.g. "BTCUSDT"
            side: "buy" or "sell"
            size: Contract size (number of contracts/lots)
            order_type: "market_order" | "limit_order" | "stop_order"
            limit_price: Required for limit_order
            leverage: 1–20
            stop_price: Trigger price for stop orders
            bracket_stop_loss_price: Stop-loss trigger price (bracket order)
            bracket_stop_loss_limit_price: SL limit price (slightly below trigger)
            bracket_take_profit_price: Take-profit limit price (bracket order)
            bracket_take_profit_limit_price: TP limit (same as TP price for passive fill)
            time_in_force: "gtc" | "ioc" | "fok"
            reduce_only: Only reduce existing position
            client_order_id: Optional idempotency key

        Returns: Order dict from Delta Exchange API
        """
        body: dict[str, Any] = {
            "product_symbol": symbol,
            "side": side,
            "size": str(int(size)),
            "order_type": order_type,
            "time_in_force": time_in_force,
            "leverage": str(leverage),
            "reduce_only": reduce_only,
        }

        if limit_price is not None:
            body["limit_price"] = str(limit_price)
        if stop_price is not None:
            body["stop_price"] = str(stop_price)
        if trail_amount is not None:
            body["trail_amount"] = str(trail_amount)
        if client_order_id:
            body["client_order_id"] = client_order_id

        # Bracket order legs (submitted with the entry order)
        if bracket_stop_loss_price is not None:
            body["bracket_stop_loss_price"] = str(bracket_stop_loss_price)
            body["bracket_stop_loss_limit_price"] = str(
                bracket_stop_loss_limit_price or bracket_stop_loss_price
            )
        if bracket_take_profit_price is not None:
            body["bracket_take_profit_price"] = str(bracket_take_profit_price)
            body["bracket_take_profit_limit_price"] = str(
                bracket_take_profit_limit_price or bracket_take_profit_price
            )

        data = await self._request("POST", ORDERS, json_body=body)
        return data.get("result", {})

    async def cancel_order(self, order_id: int, symbol: str) -> dict:
        """Cancel an open order by ID."""
        body = {"id": order_id, "product_symbol": symbol}
        data = await self._request(
            "DELETE",
            CANCEL_ORDER.format(order_id=order_id),
            json_body=body,
        )
        return data.get("result", {})

    async def cancel_all_orders(self, symbol: str | None = None) -> dict:
        """Cancel all open orders, optionally for a specific symbol."""
        body: dict[str, Any] = {}
        if symbol:
            body["product_symbol"] = symbol
        data = await self._request("DELETE", CANCEL_ALL, json_body=body)
        return data.get("result", {})

    async def get_orders(self, symbol: str | None = None, state: str = "open") -> list[dict]:
        """Fetch orders filtered by symbol and state."""
        params: dict[str, Any] = {"state": state}
        if symbol:
            params["product_symbol"] = symbol
        data = await self._request("GET", ORDERS, params=params)
        return data.get("result", {}).get("result", [])

    async def get_fills(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        """Get recent fills (executed trades)."""
        params: dict[str, Any] = {"page_size": str(limit)}
        if symbol:
            params["product_symbol"] = symbol
        data = await self._request("GET", FILLS, params=params)
        return data.get("result", {}).get("result", [])
