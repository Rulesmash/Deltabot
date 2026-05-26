"""
config.py — Central configuration management for DeltaRL Trader.

All settings are read from environment variables (via .env file).
Two ExchangeConfig objects are provided: demo_config and live_config.
The active config is selected at runtime based on TRADING_MODE.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


# ── Enums ────────────────────────────────────────────────────────────────────

class TradingMode(str, Enum):
    DEMO = "demo"
    LIVE = "live"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ── Main Settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All application settings. Values are loaded from environment variables,
    falling back to defaults where safe. Secrets have no default.
    """

    # ── Operating mode ─────────────────────────────────────────────────────
    trading_mode: TradingMode = TradingMode.DEMO

    # ── Demo / Testnet credentials ─────────────────────────────────────────
    demo_api_key: str = ""
    demo_api_secret: str = ""

    # ── Live / Production credentials ──────────────────────────────────────
    live_api_key: str = ""
    live_api_secret: str = ""

    # ── Exchange endpoints (India) ─────────────────────────────────────────
    demo_rest_url: str = "https://cdn-ind.testnet.deltaex.org"
    demo_ws_url: str = "wss://socket.ind.testnet.deltaex.org/live"
    live_rest_url: str = "https://api.india.delta.exchange"
    live_ws_url: str = "wss://socket.india.delta.exchange/live"

    # ── Trading parameters ─────────────────────────────────────────────────
    trading_pairs: str = "BTCUSDT,ETHUSDT"
    timeframe_minutes: int = 5
    max_leverage: int = 20          # Delta Exchange max for major perps
    max_risk_per_trade: float = 0.01
    circuit_breaker_drawdown: float = 0.05

    # ── RL hyper-parameters ────────────────────────────────────────────────
    model_dir: str = "./models"
    rl_learning_rate: float = 3e-4
    rl_n_steps: int = 2048
    rl_batch_size: int = 64
    rl_n_epochs: int = 10
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95
    fine_tune_every_n_trades: int = 5

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/deltarl.db"

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    replay_buffer_size: int = 10_000

    # ── FastAPI server ─────────────────────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    secret_key: str = "change_me_please"

    # ── Alerts ─────────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""

    # ── Simulation mode ────────────────────────────────────────────────────
    # True  → predictions only, P&L simulated from price (no real orders)
    # False → real orders on the selected endpoint (demo or live)
    simulation_only: bool = True

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: LogLevel = LogLevel.INFO

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}

    # ── Derived helpers ────────────────────────────────────────────────────

    @property
    def active_api_key(self) -> str:
        return self.demo_api_key if self.trading_mode == TradingMode.DEMO else self.live_api_key

    @property
    def active_api_secret(self) -> str:
        return self.demo_api_secret if self.trading_mode == TradingMode.DEMO else self.live_api_secret

    @property
    def active_rest_url(self) -> str:
        return self.demo_rest_url if self.trading_mode == TradingMode.DEMO else self.live_rest_url

    @property
    def active_ws_url(self) -> str:
        return self.demo_ws_url if self.trading_mode == TradingMode.DEMO else self.live_ws_url

    @property
    def trading_pairs_list(self) -> list[str]:
        return [p.strip() for p in self.trading_pairs.split(",") if p.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)

    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    def is_demo(self) -> bool:
        return self.trading_mode == TradingMode.DEMO


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    s = Settings()
    s.ensure_dirs()
    return s
