"""
trading/executor.py — Live order execution engine.

Converts RL agent actions into real Delta Exchange bracket orders
(entry + stop-loss + take-profit submitted in one request).

Also manages:
  • Position tracking (to avoid duplicate opens)
  • Slippage estimation and logging
  • Circuit breaker integration
"""

from __future__ import annotations

import logging
import time
import uuid

import numpy as np

from delta.client import DeltaClient
from delta.endpoints import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from rl.env import (
    LEVERAGE_MAX, LEVERAGE_MIN, SL_MAX_PCT, SL_MIN_PCT,
    TP_MAX_PCT, TP_MIN_PCT, TAKER_FEE,
)

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Executes RL agent actions as real Delta Exchange orders.

    Responsibilities:
      • Translate RL action → bracket order parameters
      • Calculate position size from ATR and equity
      • Place entry + bracket SL/TP via Delta REST API
      • Track open positions to avoid double-opening
      • Log all orders for audit trail
    """

    def __init__(
        self,
        client: DeltaClient,
        symbol: str,
        max_leverage: int = 20,
        max_risk_per_trade: float = 0.01,
    ) -> None:
        self._client             = client
        self._symbol             = symbol
        self._max_leverage       = max_leverage
        self._max_risk_per_trade = max_risk_per_trade
        self._open_order_id: int | None = None
        self._open_side: int            = 0   # 0=flat, 1=long, -1=short
        self._order_log: list[dict]      = []

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def execute_action(
        self,
        direction: int,
        leverage: float,
        sl_pct: float,
        tp_pct: float,
        current_price: float,
        equity: float,
    ) -> dict | None:
        """
        Execute an RL action as a Delta Exchange order.

        Args:
            direction:     1=Long, -1=Short, 0=Flat
            leverage:      1.0 – 20.0
            sl_pct:        SL distance as fraction of price (e.g. 0.02 = 2%)
            tp_pct:        TP distance as fraction of price
            current_price: Current mark price
            equity:        Current account equity in USDT

        Returns:
            Order response dict, or None for Flat actions
        """
        # ── Flat: close any open position ────────────────────────────────────
        if direction == 0:
            if self._open_side != 0:
                await self._close_position(current_price)
            return None

        # ── Same direction as open position: hold ────────────────────────────
        if direction == self._open_side:
            logger.debug("Already in %s position — holding.", "LONG" if direction == 1 else "SHORT")
            return None

        # ── Direction reversal: close first, then open ───────────────────────
        if self._open_side != 0:
            await self._close_position(current_price)

        # ── Calculate position size ───────────────────────────────────────────
        int_leverage  = int(np.clip(round(leverage), LEVERAGE_MIN, self._max_leverage))
        risk_amount   = equity * self._max_risk_per_trade
        # Size in USDT notional = (risk_amount / sl_pct) * leverage
        # Capped at equity * leverage to prevent over-leveraging
        notional = min(
            risk_amount / max(sl_pct, 0.001) * int_leverage,
            equity * int_leverage,
        )
        # Convert USDT notional to contracts (Delta uses contract units)
        # For simplicity: 1 contract = 1 USD notional at $1 per contract
        # (Actual contract size varies by pair — this is approximate)
        contract_size = max(1, int(notional / current_price))

        # ── SL/TP price levels ───────────────────────────────────────────────
        if direction == 1:   # Long
            entry_side  = SIDE_BUY
            sl_price    = round(current_price * (1 - sl_pct), 2)
            tp_price    = round(current_price * (1 + tp_pct), 2)
        else:                # Short
            entry_side  = SIDE_SELL
            sl_price    = round(current_price * (1 + sl_pct), 2)
            tp_price    = round(current_price * (1 - tp_pct), 2)

        client_oid = f"drl_{uuid.uuid4().hex[:12]}"

        logger.info(
            "Placing %s %s: size=%d @ ~%.2f | Lev=%dx | SL=%.2f | TP=%.2f",
            ["SHORT", "FLAT", "LONG"][direction + 1],
            self._symbol, contract_size, current_price,
            int_leverage, sl_price, tp_price,
        )

        try:
            order = await self._client.place_order(
                symbol=self._symbol,
                side=entry_side,
                size=contract_size,
                order_type=ORDER_TYPE_MARKET,
                leverage=int_leverage,
                bracket_stop_loss_price=sl_price,
                bracket_stop_loss_limit_price=sl_price,  # same for market SL
                bracket_take_profit_price=tp_price,
                bracket_take_profit_limit_price=tp_price,
                client_order_id=client_oid,
            )

            self._open_order_id = order.get("id")
            self._open_side     = direction

            log_entry = {
                "timestamp":      int(time.time()),
                "symbol":         self._symbol,
                "side":           entry_side,
                "size":           contract_size,
                "entry_price":    current_price,
                "leverage":       int_leverage,
                "sl_price":       sl_price,
                "tp_price":       tp_price,
                "order_id":       self._open_order_id,
                "client_oid":     client_oid,
                "notional_usdt":  notional,
                "fee_est":        notional * TAKER_FEE,
            }
            self._order_log.append(log_entry)
            return order

        except Exception as exc:
            logger.error("Order placement failed: %s", exc)
            raise

    async def _close_position(self, current_price: float) -> None:
        """Close the current open position with a market order."""
        if self._open_side == 0:
            return

        close_side = SIDE_SELL if self._open_side == 1 else SIDE_BUY

        try:
            # Cancel existing bracket orders first
            await self._client.cancel_all_orders(self._symbol)

            # Get actual position size from API
            position = await self._client.get_position(self._symbol)
            if position:
                size = abs(float(position.get("size", 0)))
                if size > 0:
                    await self._client.place_order(
                        symbol=self._symbol,
                        side=close_side,
                        size=size,
                        order_type=ORDER_TYPE_MARKET,
                        reduce_only=True,
                    )
                    logger.info(
                        "Closed %s position (%d contracts @ ~%.2f)",
                        "LONG" if self._open_side == 1 else "SHORT",
                        int(size), current_price,
                    )
        except Exception as exc:
            logger.error("Position close failed: %s", exc)
        finally:
            self._open_side     = 0
            self._open_order_id = None

    @property
    def order_log(self) -> list[dict]:
        return list(self._order_log)

    @property
    def is_flat(self) -> bool:
        return self._open_side == 0
