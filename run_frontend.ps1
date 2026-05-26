# DeltaRL Trader — Frontend Launcher (PowerShell)
# Run AFTER run_backend.ps1 is running.

$ErrorActionPreference = "Stop"
$root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $root "frontend"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Magenta
Write-Host "  DeltaRL Trader — Frontend Launcher     " -ForegroundColor Magenta
Write-Host "=========================================" -ForegroundColor Magenta
Write-Host ""

# Check Node.js
Write-Host "[1/3] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodever = node --version 2>&1
    Write-Host "      $nodever" -ForegroundColor Green
} catch {
    Write-Host "      ERROR: Node.js not found. Install Node.js 20+ from https://nodejs.org" -ForegroundColor Red
    exit 1
}

# Install npm packages
Write-Host "[2/3] Installing npm packages..." -ForegroundColor Yellow
Set-Location $frontend
npm install --legacy-peer-deps
Write-Host "      Packages installed." -ForegroundColor Green

# Start dev server
Write-Host "[3/3] Starting Next.js dashboard on http://localhost:3000 ..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Dashboard:   http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

npm run dev
