"""
api/routes_trading.py — Trading status and position management endpoints.
api/routes_data.py    — Market data and feature endpoints.
api/routes_config.py  — Bot configuration endpoints.

All routes use request.app.state for dependency injection
(set during startup in main.py).
"""

# ─── routes_trading.py ───────────────────────────────────────────────────────
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

trading_router = APIRouter(prefix="/api/trading", tags=["Trading"])


@trading_router.get("/positions")
async def get_positions(request=None):
    """Get all open positions from the exchange."""
    client = request.app.state.delta_client
    try:
        positions = await client.get_positions()
        return {"positions": positions}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@trading_router.get("/balance")
async def get_balance(request=None):
    """Get current wallet balance."""
    client = request.app.state.delta_client
    try:
        balance = await client.get_wallet_balance()
        return {"balance": balance}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@trading_router.get("/orders")
async def get_open_orders(symbol: str | None = None, request=None):
    """Get open orders, optionally for a specific symbol."""
    client = request.app.state.delta_client
    try:
        orders = await client.get_orders(symbol=symbol)
        return {"orders": orders}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@trading_router.get("/fills")
async def get_fills(symbol: str | None = None, limit: int = 50, request=None):
    """Get recent fills (executed orders)."""
    client = request.app.state.delta_client
    try:
        fills = await client.get_fills(symbol=symbol, limit=limit)
        return {"fills": fills}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@trading_router.post("/cancel-all")
async def cancel_all_orders(symbol: str | None = None, request=None):
    """Cancel all open orders."""
    client = request.app.state.delta_client
    try:
        result = await client.cancel_all_orders(symbol=symbol)
        return {"result": result}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@trading_router.get("/ticker/{symbol}")
async def get_ticker(symbol: str, request=None):
    """Get real-time ticker for a symbol."""
    client = request.app.state.delta_client
    try:
        ticker = await client.get_ticker(symbol)
        return ticker
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ─── routes_data.py ──────────────────────────────────────────────────────────

data_router = APIRouter(prefix="/api/data", tags=["Market Data"])


@data_router.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    resolution: str = "5m",
    limit: int = 200,
    request=None,
):
    """
    Get historical OHLCV candles for charting.
    Returns the most recent `limit` candles.
    """
    from data.fetcher import HistoricalFetcher
    client = request.app.state.delta_client
    try:
        fetcher = HistoricalFetcher(client)
        df = await fetcher.fetch_latest(symbol=symbol, resolution=resolution, n=limit)
        if df.empty:
            return {"candles": []}
        # Convert to list of dicts with unix timestamps for frontend charts
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time":   int(ts.timestamp()),
                "open":   round(float(row["open"]), 2),
                "high":   round(float(row["high"]), 2),
                "low":    round(float(row["low"]), 2),
                "close":  round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 2),
            })
        return {"candles": candles, "symbol": symbol, "resolution": resolution}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@data_router.get("/features/{symbol}")
async def get_features(symbol: str, request=None):
    """Get the latest computed feature vector for a symbol."""
    from data.features import FEATURE_NAMES, compute_features, get_feature_vector
    from data.fetcher import HistoricalFetcher
    client = request.app.state.delta_client
    try:
        fetcher = HistoricalFetcher(client)
        df = await fetcher.fetch_latest(symbol=symbol, resolution="5m", n=100)
        df_feat = compute_features(df)
        vec = get_feature_vector(df_feat)
        return {
            "features": {name: round(float(val), 6) for name, val in zip(FEATURE_NAMES, vec)}
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@data_router.get("/products")
async def get_products(request=None):
    """List all available trading products."""
    client = request.app.state.delta_client
    try:
        products = await client.get_products()
        # Filter to USDT perpetual futures only
        perps = [
            p for p in products
            if p.get("contract_type") == "perpetual_futures"
            and "USDT" in p.get("symbol", "")
        ]
        return {"products": perps[:50]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ─── routes_config.py ────────────────────────────────────────────────────────

config_router = APIRouter(prefix="/api/config", tags=["Configuration"])


class ModeRequest(BaseModel):
    mode: str  # "demo" or "live"
    confirm_live: bool = False


class RiskSettingsRequest(BaseModel):
    max_risk_per_trade: float = 0.01
    circuit_breaker_drawdown: float = 0.05
    max_leverage: int = 20
    simulation_only: bool = True


@config_router.get("/settings")
async def get_settings(request=None):
    """Return current (non-secret) configuration."""
    settings = request.app.state.settings
    return {
        "mode":                   settings.trading_mode.value,
        "simulation_only":        settings.simulation_only,
        "trading_pairs":          settings.trading_pairs_list,
        "timeframe_minutes":      settings.timeframe_minutes,
        "max_leverage":           settings.max_leverage,
        "max_risk_per_trade":     settings.max_risk_per_trade,
        "circuit_breaker_drawdown": settings.circuit_breaker_drawdown,
        "demo_api_configured":    bool(settings.demo_api_key),
        "live_api_configured":    bool(settings.live_api_key),
    }


@config_router.post("/mode")
async def set_mode(body: ModeRequest, request=None):
    """
    Switch between Demo and Live mode.

    Live mode requires explicit confirmation and a valid live API key.
    """
    settings = request.app.state.settings

    if body.mode == "live":
        if not body.confirm_live:
            raise HTTPException(
                status_code=400,
                detail="You must set confirm_live=true to enable Live mode. This will use REAL FUNDS.",
            )
        if not settings.live_api_key:
            raise HTTPException(
                status_code=400,
                detail="Live API key is not configured. Set LIVE_API_KEY in your .env file.",
            )

    # Note: In production, this would trigger a new DeltaClient with the appropriate keys.
    # For now, we return instructions to update TRADING_MODE in .env and restart.
    return {
        "message": f"To switch to {body.mode} mode, set TRADING_MODE={body.mode} in .env and restart the backend.",
        "current_mode": settings.trading_mode.value,
    }


@config_router.post("/risk")
async def update_risk_settings(body: RiskSettingsRequest, request=None):
    """Update risk management settings (runtime update, not persisted to env)."""
    settings = request.app.state.settings
    settings.max_risk_per_trade         = body.max_risk_per_trade
    settings.circuit_breaker_drawdown   = body.circuit_breaker_drawdown
    settings.max_leverage               = body.max_leverage
    settings.simulation_only            = body.simulation_only

    # Update circuit breaker
    request.app.state.circuit_breaker._max_drawdown = body.circuit_breaker_drawdown

    return {"status": "updated", "settings": body.model_dump()}
