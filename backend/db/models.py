"""
db/models.py — SQLAlchemy ORM models for DeltaRL Trader.

Tables:
  • Trade        — complete trade history with P&L, fees, slippage
  • Checkpoint   — RL model version metadata
  • BarData      — cached OHLCV for backtesting
  • BotConfig    — persisted user settings (API keys stored encrypted)
  • AlertLog     — record of sent alerts
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────

class Trade(Base):
    """
    Records every closed trade for audit trail and RL replay.
    """
    __tablename__ = "trades"

    id             = Column(Integer, primary_key=True, index=True)
    symbol         = Column(String(20), nullable=False, index=True)
    mode           = Column(String(10), nullable=False)        # demo / live / simulation
    side           = Column(Integer, nullable=False)           # 1=long, -1=short
    entry_price    = Column(Float, nullable=False)
    exit_price     = Column(Float, nullable=False)
    size_usdt      = Column(Float, nullable=False)             # notional in USDT
    leverage       = Column(Float, nullable=False)
    sl_price       = Column(Float)
    tp_price       = Column(Float)
    realized_pnl   = Column(Float, nullable=False)
    fees           = Column(Float, nullable=False, default=0.0)
    slippage       = Column(Float, nullable=False, default=0.0)
    duration_bars  = Column(Integer)                           # how many bars held
    close_reason   = Column(String(20))                        # "sl", "tp", "signal", "manual"
    reward         = Column(Float)                             # RL reward for this trade
    entry_action   = Column(JSON)                              # raw action vector
    entry_obs      = Column(JSON)                              # observation at entry
    entered_at     = Column(DateTime(timezone=True), default=_utcnow)
    exited_at      = Column(DateTime(timezone=True))
    exchange_order_id = Column(String(50))

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ─────────────────────────────────────────────────────────────────────────────

class Checkpoint(Base):
    """RL model checkpoint metadata."""
    __tablename__ = "checkpoints"

    id           = Column(Integer, primary_key=True)
    version      = Column(Integer, nullable=False, unique=True)
    total_steps  = Column(Integer, nullable=False)
    mean_reward  = Column(Float)
    sharpe       = Column(Float)
    file_path    = Column(String(500), nullable=False)
    hyperparams  = Column(JSON)
    created_at   = Column(DateTime(timezone=True), default=_utcnow)
    is_active    = Column(Boolean, default=False)
    notes        = Column(Text)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ─────────────────────────────────────────────────────────────────────────────

class BarData(Base):
    """
    Cached OHLCV candles for backtesting and RL seeding.
    Avoids repeated API calls for historical data.
    """
    __tablename__ = "bar_data"

    id         = Column(Integer, primary_key=True)
    symbol     = Column(String(20), nullable=False, index=True)
    resolution = Column(String(5), nullable=False)
    bar_time   = Column(DateTime(timezone=True), nullable=False, index=True)
    open       = Column(Float, nullable=False)
    high       = Column(Float, nullable=False)
    low        = Column(Float, nullable=False)
    close      = Column(Float, nullable=False)
    volume     = Column(Float, nullable=False)


# ─────────────────────────────────────────────────────────────────────────────

class BotConfig(Base):
    """
    Persisted bot configuration (non-secret values only).
    API keys are stored in environment variables / .env, not here.
    """
    __tablename__ = "bot_config"

    id                   = Column(Integer, primary_key=True)
    trading_mode         = Column(String(10), default="demo")
    trading_pairs        = Column(String(200), default="BTCUSDT,ETHUSDT")
    timeframe_minutes    = Column(Integer, default=5)
    max_leverage         = Column(Integer, default=20)
    max_risk_per_trade   = Column(Float, default=0.01)
    circuit_breaker_pct  = Column(Float, default=0.05)
    simulation_only      = Column(Boolean, default=True)
    rl_learning_rate     = Column(Float, default=3e-4)
    rl_n_steps           = Column(Integer, default=2048)
    telegram_enabled     = Column(Boolean, default=False)
    discord_enabled      = Column(Boolean, default=False)
    updated_at           = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ─────────────────────────────────────────────────────────────────────────────

class AlertLog(Base):
    """Record of sent Telegram / Discord alerts."""
    __tablename__ = "alert_logs"

    id         = Column(Integer, primary_key=True)
    channel    = Column(String(20))   # "telegram" | "discord"
    level      = Column(String(10))   # "info" | "warning" | "critical"
    message    = Column(Text, nullable=False)
    sent_at    = Column(DateTime(timezone=True), default=_utcnow)
    success    = Column(Boolean, default=True)
