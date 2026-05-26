"""
api/routes_rl.py — RL training and model management endpoints.
                   RISK-MITIGATION-FIRST version (v2)

Endpoints:
  POST /api/rl/train/start             — Start training loop
  POST /api/rl/train/stop              — Stop training loop
  GET  /api/rl/status                  — Current training status + risk metrics
  GET  /api/rl/checkpoints             — List saved model versions
  GET  /api/rl/model/export/{version}  — Download model as ZIP
  POST /api/rl/model/import            — Upload and load a model
  POST /api/rl/backtest                — Run backtest on historical data
  POST /api/rl/circuit-breaker/resume  — Resume after circuit breaker
  POST /api/rl/circuit-breaker/pause   — Manual pause
  POST /api/rl/curriculum/stage        — Set curriculum stage (0/1/2)
  POST /api/rl/reward/weights          — Update reward weights at runtime
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rl", tags=["RL Training"])


# ── Request / Response Models ─────────────────────────────────────────────────

class TrainRequest(BaseModel):
    mode: str = Field("simulation", pattern="^(simulation|demo|live)$")
    total_timesteps: int = Field(200_000, ge=1000, le=10_000_000)
    symbol: str = "BTCUSDT"
    curriculum_stage: int | None = Field(None, ge=0, le=2,
        description="Curriculum stage override (0=low-vol, 1=medium, 2=full). None=auto.")


class CurriculumRequest(BaseModel):
    stage: int = Field(..., ge=0, le=2,
        description="0=low-volatility only | 1=medium | 2=full (all regimes)")


class RewardWeightsRequest(BaseModel):
    """
    Hot-swap reward function weights at runtime without restarting.
    All fields are optional — only provided fields are updated.
    """
    pnl_weight:            float | None = None
    sharpe_weight:         float | None = None
    sortino_weight:        float | None = None
    calmar_weight:         float | None = None
    trade_dd_weight:       float | None = None
    global_dd_weight:      float | None = None
    liquidation_penalty_weight: float | None = None
    volatility_penalty_weight:  float | None = None
    flat_penalty:          float | None = None
    trade_dd_threshold:    float | None = None
    global_dd_threshold:   float | None = None


class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    resolution: str = "5m"
    lookback_bars: int = Field(1000, ge=100, le=5000)


# ── Dependency: app state ──────────────────────────────────────────────────────
# These are injected from app.state set in main.py

def get_orchestrator(request):
    from fastapi import Request
    return request.app.state.orchestrator


def get_agent(request):
    from fastapi import Request
    return request.app.state.agent


def get_client(request):
    from fastapi import Request
    return request.app.state.delta_client


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/train/start")
async def start_training(body: TrainRequest, request=None):
    """
    Start the RL training loop as a background task.

    In 'simulation' mode (default): No orders are placed; P&L is simulated
    from real price movement. Safest for initial training.

    In 'demo' mode: Places real virtual orders on the Delta Testnet.

    In 'live' mode: Places REAL orders. Requires explicit confirmation.
    """
    from fastapi import Request
    orchestrator = request.app.state.orchestrator

    if orchestrator.status.get("state") == "running":
        raise HTTPException(status_code=409, detail="Training is already running.")

    if body.mode == "live":
        # Extra safety guard — must be explicitly enabled in config
        settings = request.app.state.settings
        if not settings.is_live():
            raise HTTPException(
                status_code=403,
                detail="Live mode is disabled. Set TRADING_MODE=live in your .env file first.",
            )

    await orchestrator.start(
        mode=body.mode,
        total_timesteps=body.total_timesteps,
        curriculum_stage=body.curriculum_stage,
    )
    return {
        "status":           "started",
        "mode":             body.mode,
        "total_timesteps":  body.total_timesteps,
        "curriculum_stage": body.curriculum_stage,
    }


@router.post("/train/stop")
async def stop_training(request=None):
    """Stop the training loop."""
    from fastapi import Request
    orchestrator = request.app.state.orchestrator
    await orchestrator.stop()
    return {"status": "stopped"}


@router.get("/status")
async def get_training_status(request=None):
    """Get current training status, metrics, and circuit breaker state."""
    from fastapi import Request
    orchestrator  = request.app.state.orchestrator
    circuit       = request.app.state.circuit_breaker
    return {
        "training": orchestrator.status,
        "circuit_breaker": circuit.status,
        "model": {
            "is_trained": request.app.state.agent.is_trained,
            "version":    request.app.state.agent.version,
            "total_steps": request.app.state.agent.total_steps,
        },
    }


@router.get("/checkpoints")
async def list_checkpoints(request=None):
    """List all available model checkpoints."""
    from fastapi import Request
    return request.app.state.agent.list_checkpoints()


@router.get("/model/export/{version}")
async def export_model(version: int, request=None):
    """Download a model checkpoint as a ZIP file."""
    from fastapi import Request
    settings   = request.app.state.settings
    model_dir  = Path(settings.model_dir)
    zip_path   = model_dir / f"ppo_v{version}.zip"

    if not zip_path.exists():
        raise HTTPException(status_code=404, detail=f"Model v{version} not found.")

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"deltarl_ppo_v{version}.zip",
    )


@router.post("/model/import")
async def import_model(file: UploadFile = File(...), request=None):
    """Upload and activate a model checkpoint."""
    from fastapi import Request
    settings  = request.app.state.settings
    agent     = request.app.state.agent
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    save_path = model_dir / "imported_model.zip"
    content   = await file.read()
    save_path.write_bytes(content)

    agent.load(str(save_path.with_suffix("")))  # SB3 appends .zip automatically
    return {"status": "loaded", "file": file.filename}


@router.post("/backtest")
async def run_backtest(body: BacktestRequest, request=None):
    """
    Run a backtest using the current model on historical data.
    Returns equity curve, trade log, and performance metrics.
    """
    from fastapi import Request
    from data.fetcher import HistoricalFetcher
    from rl.backtest import run_backtest as _backtest

    agent  = request.app.state.agent
    client = request.app.state.delta_client

    if not agent.is_trained:
        raise HTTPException(status_code=400, detail="No model trained yet. Start training first.")

    fetcher = HistoricalFetcher(client)
    df = await fetcher.fetch(
        symbol=body.symbol,
        resolution=body.resolution,
        lookback_bars=body.lookback_bars,
    )

    if df.empty:
        raise HTTPException(status_code=503, detail="Could not fetch historical data.")

    loop    = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: _backtest(
            candle_df=df,
            predict_fn=lambda obs: agent.predict(obs, deterministic=True)[0],
            initial_equity=10_000.0,
            symbol=body.symbol,
        ),
    )

    # Trim large arrays for JSON response
    results["equity_curve"] = results["equity_curve"][::10]   # downsample
    results["reward_curve"] = results["reward_curve"][::10]
    results["trade_log"]    = results["trade_log"][:200]       # cap at 200

    return results


@router.post("/circuit-breaker/resume")
async def resume_circuit_breaker(request=None):
    """Manually resume trading after circuit breaker or pause."""
    from fastapi import Request
    circuit = request.app.state.circuit_breaker
    circuit.resume()
    return {"status": "resumed"}


@router.post("/circuit-breaker/pause")
async def pause_circuit_breaker(request=None):
    """Manually pause all trading."""
    from fastapi import Request
    circuit = request.app.state.circuit_breaker
    circuit.pause()
    return {"status": "paused"}


# ── Curriculum & Reward Controls (v2) ───────────────────────────────────────────────

@router.post("/curriculum/stage")
async def set_curriculum_stage(body: CurriculumRequest, request=None):
    """
    Manually set the curriculum learning stage:
      0 — Low-volatility data only  (ATR percentile < 33%)
      1 — Low + medium volatility   (ATR percentile < 66%)
      2 — Full data (all regimes, including bear markets and extreme events)

    Curriculum auto-advances based on performance gates.
    Setting a stage here locks it manually until training is restarted.
    """
    from fastapi import Request
    orchestrator = request.app.state.orchestrator
    orchestrator.set_curriculum_stage(body.stage)
    return {
        "status":    "updated",
        "stage":     body.stage,
        "stage_name": ["Low-Volatility", "Medium-Volatility", "Full-Data"][body.stage],
        "message":   f"Curriculum stage set to {body.stage}. Effective from next episode.",
    }


@router.get("/curriculum/status")
async def get_curriculum_status(request=None):
    """Get current curriculum stage and recent performance gate metrics."""
    from fastapi import Request
    orch = request.app.state.orchestrator
    return {
        "current_stage":    orch._curriculum_stage,
        "stage_name":       ["Low-Volatility", "Medium", "Full-Data"][orch._curriculum_stage],
        "episodes_at_stage": orch._episodes_at_stage,
        "avg_sharpe":       round(
            sum(orch._recent_sharpes) / len(orch._recent_sharpes)
            if orch._recent_sharpes else 0.0, 3
        ),
        "avg_max_dd":       round(
            sum(orch._recent_max_dds) / len(orch._recent_max_dds)
            if orch._recent_max_dds else 0.0, 3
        ),
        "gates": [
            {"stage": 0, "min_sharpe": 0.5,  "max_avg_dd_pct": 8,  "min_episodes": 50},
            {"stage": 1, "min_sharpe": 1.0,  "max_avg_dd_pct": 12, "min_episodes": 50},
        ],
    }


@router.post("/reward/weights")
async def update_reward_weights(body: RewardWeightsRequest, request=None):
    """
    Hot-swap reward function weights without restarting the server.

    Only the provided fields are changed. Useful for:
      - Increasing drawdown penalties if the agent is too aggressive
      - Boosting Sharpe/Sortino during curriculum advancement
      - Reducing flat_penalty if training in high-vol regimes
    """
    from fastapi import Request
    from rl.reward import CFG

    updated = {}
    for field, value in body.model_dump(exclude_none=True).items():
        if hasattr(CFG, field):
            setattr(CFG, field, value)
            updated[field] = value

    if not updated:
        return {"status": "no_changes", "updated": {}}

    return {
        "status":  "updated",
        "updated": updated,
        "message": f"Reward weights updated: {list(updated.keys())}. Effective from next step.",
    }


@router.get("/reward/weights")
async def get_reward_weights(request=None):
    """Get the current reward function configuration."""
    from rl.reward import CFG
    return {
        field: getattr(CFG, field)
        for field in vars(CFG)
        if not field.startswith("_")
    }

