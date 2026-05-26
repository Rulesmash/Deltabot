"""
db/database.py — SQLite database connection and session management.

Uses SQLAlchemy async engine for non-blocking DB operations.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from db.models import Base

logger = logging.getLogger(__name__)

# ── Async engine (for FastAPI routes) ─────────────────────────────────────────

def _make_async_url(url: str) -> str:
    """Convert sqlite:/// → sqlite+aiosqlite:///"""
    if url.startswith("sqlite:") and "aiosqlite" not in url:
        return url.replace("sqlite:", "sqlite+aiosqlite:", 1)
    return url


async def init_db(database_url: str) -> async_sessionmaker[AsyncSession]:
    """
    Initialise the database, create tables, return session factory.

    Args:
        database_url: SQLAlchemy database URL (e.g. sqlite:///./data/deltarl.db)

    Returns:
        Async session factory
    """
    async_url = _make_async_url(database_url)
    engine = create_async_engine(
        async_url,
        echo=False,
        connect_args={"check_same_thread": False} if "sqlite" in async_url else {},
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    logger.info("Database initialised: %s", database_url)
    return factory


# ── Sync engine (for SB3 training thread) ────────────────────────────────────

def init_sync_db(database_url: str):
    """Create a synchronous DB engine (used in the training thread)."""
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
    )
    Base.metadata.create_all(engine)
    return engine
