"""
smoke_test.py — Quick import and sanity check for DeltaRL Trader backend.
Run this BEFORE starting the server to catch missing packages / broken imports.

Usage:
    cd backend
    python smoke_test.py
"""
from __future__ import annotations

import sys
import traceback

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results: list[tuple[str, bool, str]] = []


def check(label: str, fn):
    try:
        fn()
        results.append((label, True, ""))
        print(f"  {PASS}  {label}")
    except Exception as e:
        results.append((label, False, str(e)))
        print(f"  {FAIL}  {label}")
        print(f"       {e}")


print("\n" + "=" * 55)
print("  DeltaRL Trader — Backend Smoke Test")
print("=" * 55 + "\n")

# ── Core stdlib / framework imports ──────────────────────────────────────────
print("[ Core packages ]")
check("fastapi",           lambda: __import__("fastapi"))
check("uvicorn",           lambda: __import__("uvicorn"))
check("pydantic",          lambda: __import__("pydantic"))
check("pydantic_settings", lambda: __import__("pydantic_settings"))
check("httpx",             lambda: __import__("httpx"))
check("websockets",        lambda: __import__("websockets"))
check("sqlalchemy",        lambda: __import__("sqlalchemy"))
check("aiosqlite",         lambda: __import__("aiosqlite"))

print("\n[ Data science ]")
check("numpy",             lambda: __import__("numpy"))
check("pandas",            lambda: __import__("pandas"))
check("pandas_ta",         lambda: __import__("pandas_ta"))

print("\n[ RL stack ]")
check("gymnasium",         lambda: __import__("gymnasium"))
check("stable_baselines3", lambda: __import__("stable_baselines3"))
check("torch",             lambda: __import__("torch"))

print("\n[ App modules ]")
check("config",            lambda: __import__("config"))
check("rl.env",            lambda: __import__("rl.env"))
check("rl.reward",         lambda: __import__("rl.reward"))
check("rl.agent",          lambda: __import__("rl.agent"))
check("rl.train",          lambda: __import__("rl.train"))

# ── Env sanity check ─────────────────────────────────────────────────────────
print("\n[ Config / .env ]")

def check_settings():
    from config import get_settings
    s = get_settings()
    print(f"       Mode: {s.trading_mode.value.upper()}  |  Simulation: {s.simulation_only}")
    print(f"       Endpoint: {s.active_rest_url}")
    print(f"       Pairs: {s.trading_pairs_list}")

check("Settings load", check_settings)

# ── Gymnasium env sanity check ────────────────────────────────────────────────
print("\n[ Gym environment ]")

def check_env():
    import numpy as np
    import pandas as pd
    from rl.env import TradingEnv

    # Build a minimal synthetic candle DataFrame
    n = 300
    np.random.seed(42)
    prices = 40000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        "open":   prices,
        "high":   prices * 1.005,
        "low":    prices * 0.995,
        "close":  prices,
        "volume": np.random.rand(n) * 1000 + 100,
    }, index=pd.date_range("2024-01-01", periods=n, freq="5min"))

    env = TradingEnv(
        mode="simulation",
        initial_equity=10_000,
        candle_df=df,
        symbol="BTCUSDT",
        max_leverage=20,
        curriculum_stage=0,
    )
    obs, info = env.reset()
    print(f"       Obs shape: {obs.shape}  (expected: ({env.observation_space.shape[0]},))")
    print(f"       Action space: {env.action_space}")
    print(f"       Curriculum stage: {env._curriculum_stage}")

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f"       Step OK | reward={reward:.4f} | obs shape: {obs2.shape}")

check("TradingEnv (49-dim obs, 5-dim action)", check_env)

# ── Reward calculator check ───────────────────────────────────────────────────
print("\n[ Reward function ]")

def check_reward():
    from rl.reward import RewardCalculator, CFG
    rc = RewardCalculator()
    rc.reset(initial_equity=10_000.0)
    reward, breakdown = rc.compute(
        realized_pnl=50.0,
        fees=0.05,
        slippage=0.01,
        equity=10_050.0,
        leverage=5.0,
        action_direction=1,
        position_closed=True,
        volatility_regime=0.3,
        funding_rate=0.0001,
        liquidation_proximity=0.1,
        open_pnl_pct=0.005,
        is_profitable_trade=True,
        num_trades_episode=1,
    )
    print(f"       Reward: {reward:.4f}")
    print(f"       Breakdown keys: {list(breakdown.keys())}")
    print(f"       CFG.sharpe_weight: {CFG.sharpe_weight}")
    print(f"       CFG.global_dd_weight: {CFG.global_dd_weight}")

check("RewardCalculator", check_reward)

# ── Results summary ───────────────────────────────────────────────────────────
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print("\n" + "=" * 55)
if failed == 0:
    print(f"  {PASS}  All {total} checks passed! Ready to start the server.")
    print("\n  Run: python -m uvicorn main:app --reload --port 8000")
else:
    print(f"  {FAIL}  {failed}/{total} checks FAILED.")
    print("\n  Fix the issues above, then re-run this script.")
    print("\n  Common fixes:")
    print("    pip install -r requirements.txt")
    print("    # For TA-Lib on Windows: download pre-built wheel from")
    print("    # https://github.com/cgohlke/talib-build/releases")
print("=" * 55 + "\n")

sys.exit(0 if failed == 0 else 1)
