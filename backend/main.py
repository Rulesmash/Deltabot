"""
main.py — FastAPI application entrypoint for DeltaRL Trader.

Startup sequence:
  1. Load settings from .env
  2. Initialise SQLite database
  3. Connect DeltaClient (demo or live endpoint)
  4. Create RL Agent + Training Orchestrator
  5. Connect WebSocket to Delta Exchange
  6. Mount all API routers
  7. Start WebSocket endpoint for frontend
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from alerts.notifier import AlertNotifier
from api.routes_combined import config_router, data_router, trading_router
from api.routes_rl import router as rl_router
from api.websocket_handler import emit, ws_endpoint, ws_manager
from config import TradingMode, get_settings
from data.stream import CandleBuffer
from db.database import init_db
from delta.client import DeltaClient
from delta.websocket_client import DeltaWebSocket
from rl.agent import RLAgent
from rl.train import TrainingOrchestrator
from trading.circuit_breaker import CircuitBreaker

# ── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Application Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application startup and shutdown lifecycle.
    All shared resources are attached to app.state.
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("  DeltaRL Trader Starting Up")
    logger.info("  Mode: %s | Simulation: %s", settings.trading_mode.value.upper(), settings.simulation_only)
    logger.info("  Endpoint: %s", settings.active_rest_url)
    logger.info("=" * 60)

    if settings.trading_mode == TradingMode.DEMO:
        logger.warning("⚠️  DEMO / VIRTUAL TRAINING MODE — No real funds at risk")
    else:
        logger.warning("🔴  LIVE / PRODUCTION MODE — Real funds are at risk!")

    # ── Database ──────────────────────────────────────────────────────────────
    db_factory = await init_db(settings.database_url)
    app.state.db_factory = db_factory

    # ── Delta REST Client ─────────────────────────────────────────────────────
    client = DeltaClient(
        api_key=settings.active_api_key,
        api_secret=settings.active_api_secret,
        base_url=settings.active_rest_url,
    )
    await client.start()
    app.state.delta_client = client
    app.state.settings     = settings

    # ── Candle Buffer (one per pair) ──────────────────────────────────────────
    candle_buffers: dict[str, CandleBuffer] = {}
    for symbol in settings.trading_pairs_list:
        buf = CandleBuffer(
            symbol=symbol,
            resolution_minutes=settings.timeframe_minutes,
            max_size=1000,
        )
        candle_buffers[symbol] = buf
    app.state.candle_buffers = candle_buffers

    # ── RL Agent ──────────────────────────────────────────────────────────────
    agent = RLAgent(
        model_dir=settings.model_dir,
        learning_rate=settings.rl_learning_rate,
        n_steps=settings.rl_n_steps,
        batch_size=settings.rl_batch_size,
        n_epochs=settings.rl_n_epochs,
        gamma=settings.rl_gamma,
        gae_lambda=settings.rl_gae_lambda,
    )
    app.state.agent = agent

    # ── Circuit Breaker ───────────────────────────────────────────────────────
    circuit = CircuitBreaker(
        max_drawdown_pct=settings.circuit_breaker_drawdown,
        notify_callback=lambda d: emit("circuit_breaker", d),
    )
    app.state.circuit_breaker = circuit

    # ── Alert Notifier ────────────────────────────────────────────────────────
    notifier = AlertNotifier(
        telegram_token=settings.telegram_bot_token,
        telegram_chat_id=settings.telegram_chat_id,
        discord_webhook_url=settings.discord_webhook_url,
        trading_mode=settings.trading_mode.value,
    )
    app.state.notifier = notifier

    # ── Training Orchestrator ─────────────────────────────────────────────────
    def _progress(data: dict) -> None:
        emit("training_progress", data)

    orchestrator = TrainingOrchestrator(
        settings=settings,
        agent=agent,
        client=client,
        candle_buffer=candle_buffers.get(settings.trading_pairs_list[0]),
        progress_callback=_progress,
    )
    app.state.orchestrator = orchestrator

    # ── Delta WebSocket (background task) ────────────────────────────────────
    ws_client = DeltaWebSocket(
        url=settings.active_ws_url,
        api_key=settings.active_api_key,
        api_secret=settings.active_api_secret,
    )

    async def _on_candle(msg: dict) -> None:
        symbol = msg.get("symbol", "")
        if symbol in candle_buffers:
            await candle_buffers[symbol].on_candlestick(msg)
            emit("candle", {**msg, "symbol": symbol})

    async def _on_ticker(msg: dict) -> None:
        symbol = msg.get("symbol", "")
        if symbol in candle_buffers:
            await candle_buffers[symbol].on_ticker(msg)
        emit("ticker", msg)

    for symbol in settings.trading_pairs_list:
        res = str(settings.timeframe_minutes)
        ws_client.subscribe_public(f"candlestick_{res}", symbol, _on_candle)
        ws_client.subscribe_public("ticker", symbol, _on_ticker)
        ws_client.subscribe_public("mark_price", symbol, _on_ticker)

    if settings.active_api_key:
        ws_client.subscribe_private("orders",       lambda m: emit("order_update", m))
        ws_client.subscribe_private("positions",    lambda m: emit("position_update", m))
        ws_client.subscribe_private("user_balance", lambda m: emit("balance_update", m))

    ws_task = asyncio.create_task(ws_client.connect())
    app.state.ws_client = ws_client
    app.state.ws_task   = ws_task

    logger.info("🚀 DeltaRL Trader is ready.")
    logger.info("  Dashboard:  http://localhost:3000")
    logger.info("  API docs:   http://localhost:8000/docs")

    # ── Yield (app running) ───────────────────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down DeltaRL Trader...")
    ws_task.cancel()
    await client.close()
    logger.info("Shutdown complete.")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    mode_label = (
        "⚠️ DEMO / VIRTUAL TRAINING MODE — No real funds at risk"
        if settings.trading_mode == TradingMode.DEMO
        else "🔴 LIVE PRODUCTION MODE"
    )

    app = FastAPI(
        title="DeltaRL Trader API",
        description=f"Autonomous RL-powered crypto trading bot for Delta Exchange India.\n\n**{mode_label}**",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS (allow frontend dev server) ──────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Route injection helper (pass request into route handlers) ─────────────
    # FastAPI auto-injects Request if the handler declares it as a parameter.
    # All routes use `request=None` pattern and access request.app.state.

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(rl_router)
    app.include_router(trading_router)
    app.include_router(data_router)
    app.include_router(config_router)

    # ── WebSocket endpoint ────────────────────────────────────────────────────
    app.websocket("/ws")(ws_endpoint)

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health(request: Request):
        settings = request.app.state.settings
        return {
            "status": "ok",
            "mode":   settings.trading_mode.value,
            "simulation_only": settings.simulation_only,
            "ws_clients": ws_manager.num_clients,
        }

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "app":     "DeltaRL Trader",
            "version": "1.0.0",
            "docs":    "/docs",
        }

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level=settings.log_level.value.lower(),
    )
