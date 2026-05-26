# DeltaRL Trader — Backend Launcher (PowerShell)
# Run this from the project root: .\run_backend.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  DeltaRL Trader — Backend Test Runner   " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check Python ─────────────────────────────────────────────────────
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pyver = python --version 2>&1
    Write-Host "      $pyver" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Python not found. Install Python 3.11+ first." -ForegroundColor Red
    exit 1
}

# ── Step 2: Check/create venv ────────────────────────────────────────────────
$venvPath = Join-Path $backend "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvPath
    Write-Host "      venv created at $venvPath" -ForegroundColor Green
} else {
    Write-Host "[2/5] Virtual environment already exists." -ForegroundColor Green
}

$pip    = Join-Path $venvPath "Scripts\pip.exe"
$python = Join-Path $venvPath "Scripts\python.exe"

# ── Step 3: Install dependencies ─────────────────────────────────────────────
Write-Host "[3/5] Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
Write-Host "      Installing PyTorch (CPU) first..." -ForegroundColor Gray

# Install CPU torch first (smaller, faster, no CUDA needed for testing)
& $pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

Write-Host "      Installing remaining requirements..." -ForegroundColor Gray
& $pip install -r (Join-Path $backend "requirements.txt") --quiet

Write-Host "      Dependencies installed." -ForegroundColor Green

# ── Step 4: Create .env if missing ───────────────────────────────────────────
Write-Host "[4/5] Checking .env..." -ForegroundColor Yellow
$envFile = Join-Path $root ".env"
$envExample = Join-Path $root ".env.example"

if (-not (Test-Path $envFile)) {
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Host "      .env created from .env.example (SIMULATION_ONLY mode)" -ForegroundColor Green
        Write-Host "      NOTE: Edit .env and set your Demo API keys for full functionality." -ForegroundColor Yellow
    } else {
        # Create a minimal test .env
        @"
TRADING_MODE=demo
SIMULATION_ONLY=true
DEMO_API_KEY=test_key
DEMO_API_SECRET=test_secret
LIVE_API_KEY=
LIVE_API_SECRET=
SECRET_KEY=deltarl_dev_secret_$(Get-Random)
DEBUG=true
LOG_LEVEL=INFO
MAX_LEVERAGE=20
MAX_RISK_PER_TRADE=0.01
CIRCUIT_BREAKER_DRAWDOWN=0.05
CIRCUIT_BREAKER_DAILY_LOSS=0.10
RL_LEARNING_RATE=0.0003
RL_N_STEPS=2048
RL_BATCH_SIZE=64
RL_N_EPOCHS=10
RL_GAMMA=0.99
TRADING_PAIRS=BTCUSDT
TIMEFRAME_MINUTES=5
MODEL_DIR=./models
DB_PATH=./deltarl.db
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=
"@ | Out-File -FilePath $envFile -Encoding utf8
        Write-Host "      Minimal .env created for testing (SIMULATION_ONLY=true)." -ForegroundColor Green
    }
} else {
    Write-Host "      .env already exists." -ForegroundColor Green
}

# ── Step 5: Start backend ─────────────────────────────────────────────────────
Write-Host "[5/5] Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  API Docs:    http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  Health:      http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "  WebSocket:   ws://localhost:8000/ws" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

Set-Location $backend
& $python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info
