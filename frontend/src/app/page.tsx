"use client";

import { useEffect, useRef, useState } from "react";
import {
  Activity, AlertTriangle, BarChart3, Bot, ChevronDown,
  ChevronRight, Download, Play, RefreshCw, Settings,
  Shield, Square, TrendingDown, TrendingUp, Upload,
  Zap, Bell, BookOpen, ArrowUpRight, ArrowDownRight,
  Minus, Cpu, Database, Globe, Lock, Flame, Wind, Gauge
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

type Mode = "demo" | "live";
type TrainingState = "idle" | "running" | "stopped" | "completed" | "error";
type Direction = "long" | "short" | "flat";

interface Stats {
  equity: number;
  totalReturn: number;
  totalPnL: number;
  numTrades: number;
  winRate: number;
  drawdown: number;
  sharpe: number;
  currentPrice: number;
}

// v2: Risk-mitigation metrics
interface RiskMetrics {
  rollingSharee: number;
  rollingSortino: number;
  maxDdEpisode: number;
  calmarRatio: number;
  volatilityRegime: number;    // 0=low, 0.5=medium, 1=high
  portfolioDrawdown: number;   // fraction
  fundingRate: number;
  inSafetyZone: boolean;
  liquidationProximity: number;
  curriculumStage: number;     // 0, 1, or 2
  safetyTriggered: boolean;
}

const REGIME_LABELS = ["Low", "Medium", "High"] as const;
const REGIME_COLORS = ["text-signal-long", "text-signal-warning", "text-signal-short"] as const;

function getRegimeIdx(r: number): 0|1|2 {
  if (r < 0.33) return 0;
  if (r < 0.66) return 1;
  return 2;
}

interface TrainingStatus {
  state: TrainingState;
  epoch: number;
  totalSteps: number;
  meanReward: number;
  equity: number;
  numTrades: number;
  mode: string;
  // v2 risk fields
  curriculum_stage?: number;
  rolling_sharpe?: number;
  rolling_sortino?: number;
  max_dd_episode?: number;
  calmar_ratio?: number;
  volatility_regime?: number;
  portfolio_drawdown?: number;
  in_safety_zone?: boolean;
}

interface LogEntry {
  id: number;
  ts: string;
  level: "info" | "success" | "warning" | "error" | "trade";
  msg: string;
}

interface Position {
  symbol: string;
  side: number;
  size: number;
  entryPrice: number;
  markPrice: number;
  pnl: number;
  leverage: number;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL   = process.env.NEXT_PUBLIC_WS_URL  || "ws://localhost:8000";
const INITIAL_EQUITY = 10_000;

// ── Helper Components ─────────────────────────────────────────────────────────

function MetricCard({
  label, value, sub, trend, color = "text-text-primary", icon: Icon
}: {
  label: string;
  value: string;
  sub?: string;
  trend?: "up" | "down" | "flat";
  color?: string;
  icon?: React.ElementType;
}) {
  return (
    <div className="gradient-border p-4 rounded-xl hover:shadow-card-hover transition-all duration-300 group">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-secondary font-medium uppercase tracking-wider">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-text-muted group-hover:text-brand-primary transition-colors" />}
      </div>
      <div className={`text-xl font-bold font-mono ${color} animate-number-tick`}>{value}</div>
      {sub && (
        <div className={`text-xs mt-1 flex items-center gap-1 ${
          trend === "up" ? "text-signal-long" : trend === "down" ? "text-signal-short" : "text-text-muted"
        }`}>
          {trend === "up" && <ArrowUpRight className="w-3 h-3" />}
          {trend === "down" && <ArrowDownRight className="w-3 h-3" />}
          {trend === "flat" && <Minus className="w-3 h-3" />}
          {sub}
        </div>
      )}
    </div>
  );
}

function Badge({ label, variant = "neutral" }: { label: string; variant?: string }) {
  const styles = {
    success: "bg-signal-long/20 text-signal-long border-signal-long/30",
    danger:  "bg-signal-short/20 text-signal-short border-signal-short/30",
    warning: "bg-signal-warning/20 text-signal-warning border-signal-warning/30",
    neutral: "bg-bg-border/50 text-text-secondary border-bg-border",
    primary: "bg-brand-primary/20 text-brand-primary border-brand-primary/30",
  }[variant] || "bg-bg-border/50 text-text-secondary border-bg-border";

  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${styles}`}>
      {label}
    </span>
  );
}

// ── Mode Toggle Modal ─────────────────────────────────────────────────────────

function LiveModeConfirmModal({
  onConfirm, onCancel
}: { onConfirm: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="gradient-border rounded-2xl max-w-md w-full mx-4 p-6 shadow-card">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 rounded-xl bg-signal-short/20">
            <AlertTriangle className="w-6 h-6 text-signal-short" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-text-primary">Enable Live Trading?</h2>
            <p className="text-sm text-text-secondary">This will use REAL funds</p>
          </div>
        </div>

        <div className="bg-signal-short/10 border border-signal-short/30 rounded-xl p-4 mb-6 space-y-2">
          <p className="text-sm text-signal-short font-semibold">⚠️ Risk Acknowledgment Required</p>
          <ul className="text-xs text-text-secondary space-y-1 list-disc list-inside">
            <li>Real funds will be used for all trades</li>
            <li>The RL agent may incur losses — past demo performance does not guarantee live results</li>
            <li>Ensure your Live API key is configured in .env and has trading permissions</li>
            <li>The circuit breaker will pause trading if drawdown exceeds your set limit</li>
            <li>You are solely responsible for any financial outcomes</li>
          </ul>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-2.5 rounded-xl bg-bg-border hover:bg-bg-hover text-text-secondary font-semibold transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 py-2.5 rounded-xl bg-signal-short hover:bg-red-600 text-white font-bold transition-colors shadow-red-glow"
          >
            I Understand — Go Live
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Equity Spark Line ─────────────────────────────────────────────────────────

function SparkLine({ data, color = "#00D4FF" }: { data: number[]; color?: string }) {
  if (data.length < 2) return null;
  const min  = Math.min(...data);
  const max  = Math.max(...data);
  const range = max - min || 1;
  const w = 120, h = 40;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width={w} height={h} className="opacity-80">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [mode, setMode]         = useState<Mode>("demo");
  const [showConfirm, setShowConfirm] = useState(false);
  const [connected, setConnected]     = useState(false);
  const [trainStatus, setTrainStatus] = useState<TrainingStatus>({
    state: "idle", epoch: 0, totalSteps: 0, meanReward: 0, equity: INITIAL_EQUITY, numTrades: 0, mode: "demo",
  });
  const [stats, setStats]     = useState<Stats>({
    equity: INITIAL_EQUITY, totalReturn: 0, totalPnL: 0, numTrades: 0,
    winRate: 0, drawdown: 0, sharpe: 0, currentPrice: 0,
  });
  const [logs, setLogs]         = useState<LogEntry[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [equityCurve, setEquityCurve]   = useState<number[]>([INITIAL_EQUITY]);
  const [rewardCurve, setRewardCurve]   = useState<number[]>([]);
  const [activeTab, setActiveTab] = useState<"dashboard" | "training" | "config" | "backtest">("dashboard");
  const [trainSteps, setTrainSteps]   = useState(200000);
  const [trainMode, setTrainMode]     = useState<"simulation" | "demo" | "live">("simulation");
  const [circuitTriggered, setCircuit] = useState(false);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    rollingSharee: 0, rollingSortino: 0, maxDdEpisode: 0, calmarRatio: 0,
    volatilityRegime: 0, portfolioDrawdown: 0, fundingRate: 0,
    inSafetyZone: false, liquidationProximity: 0, curriculumStage: 0,
    safetyTriggered: false,
  });
  const [curriculumStage, setCurriculumStage] = useState(0);

  const wsRef     = useRef<WebSocket | null>(null);
  const logRef    = useRef<HTMLDivElement>(null);
  const logId     = useRef(0);

  // ── WebSocket connection ────────────────────────────────────────────────────

  function addLog(level: LogEntry["level"], msg: string) {
    const entry: LogEntry = {
      id: logId.current++,
      ts: new Date().toLocaleTimeString(),
      level,
      msg,
    };
    setLogs(prev => [entry, ...prev].slice(0, 200));
  }

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(`${WS_URL}/ws`);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        addLog("success", "Connected to DeltaRL Trader backend");
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          dispatch(msg);
        } catch {}
      };

      ws.onclose = () => {
        setConnected(false);
        addLog("warning", "Connection lost — reconnecting in 3s...");
        setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        addLog("error", "WebSocket connection error");
      };
    }

    connect();
    return () => wsRef.current?.close();
  }, []);

  function dispatch(msg: { type: string; data: any; ts?: string }) {
    switch (msg.type) {
      case "training_progress": {
        const d = msg.data;
        if (d.status) setTrainStatus(d.status);
        if (d.equity)      setEquityCurve(prev => [...prev.slice(-300), d.equity]);
        if (d.rollout_mean !== undefined) setRewardCurve(prev => [...prev.slice(-300), d.rollout_mean]);
        if (d.event === "episode_end") {
          const sharpe = (d.rolling_sharpe ?? 0).toFixed(2);
          const dd     = ((d.max_dd_episode ?? 0) * 100).toFixed(1);
          const stage  = d.curriculum_stage ?? 0;
          addLog("info",
            `Ep ${d.episode} | Steps: ${((d.total_steps||0)/1000).toFixed(1)}k | ` +
            `Equity: $${(d.equity||0).toFixed(0)} | Sharpe: ${sharpe} | MaxDD: ${dd}% | Stage: ${stage}`
          );
          // Update risk metrics from episode info
          setRiskMetrics(prev => ({
            ...prev,
            rollingSharee:     d.rolling_sharpe    ?? prev.rollingSharee,
            rollingSortino:    d.rolling_sortino   ?? prev.rollingSortino,
            maxDdEpisode:      d.max_dd_episode     ?? prev.maxDdEpisode,
            calmarRatio:       d.calmar_ratio       ?? prev.calmarRatio,
            volatilityRegime:  d.volatility_regime  ?? prev.volatilityRegime,
            portfolioDrawdown: (d.portfolio_drawdown ?? 0) / 100,
            inSafetyZone:      d.in_safety_zone     ?? prev.inSafetyZone,
            curriculumStage:   d.curriculum_stage   ?? prev.curriculumStage,
          }));
          setCurriculumStage(d.curriculum_stage ?? stage);
        }
        break;
      }
      case "curriculum_advance": {
        const d = msg.data;
        setCurriculumStage(d.to_stage ?? 0);
        addLog("success", `🎓 ${d.message || `Curriculum advanced to Stage ${d.to_stage}`}`);
        break;
      }
      case "curriculum_override": {
        setCurriculumStage(msg.data.stage ?? 0);
        addLog("info", `Curriculum stage set to ${msg.data.stage}`);
        break;
      }
      case "circuit_breaker": {
        setCircuit(true);
        addLog("error", `🔴 CIRCUIT BREAKER: ${msg.data.reason}`);
        break;
      }
      case "ticker": {
        const price = parseFloat(msg.data.mark_price || msg.data.close || 0);
        if (price > 0) setStats(prev => ({ ...prev, currentPrice: price }));
        break;
      }
      case "position_update": {
        addLog("trade", `Position update: ${JSON.stringify(msg.data).slice(0, 80)}`);
        break;
      }
      case "error": {
        addLog("error", msg.data.message || "Unknown error");
        break;
      }
      case "training_complete": {
        addLog("success", `✅ Training complete! ${JSON.stringify(msg.data.summary || {})}`);
        fetchStatus();
        break;
      }
    }
  }

  // ── API calls ───────────────────────────────────────────────────────────────

  async function fetchStatus() {
    try {
      const r = await fetch(`${API_BASE}/api/rl/status`);
      if (!r.ok) return;
      const d = await r.json();
      setTrainStatus(d.training);
      setCircuit(d.circuit_breaker?.triggered || false);
    } catch {}
  }

  async function startTraining() {
    try {
      addLog("info", `Starting training: mode=${trainMode}, steps=${trainSteps.toLocaleString()}`);
      const r = await fetch(`${API_BASE}/api/rl/train/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: trainMode, total_timesteps: trainSteps }),
      });
      const d = await r.json();
      if (!r.ok) {
        addLog("error", d.detail || "Failed to start training");
      } else {
        addLog("success", `Training started: ${d.mode} mode, ${d.total_timesteps.toLocaleString()} steps`);
      }
    } catch (e: any) {
      addLog("error", `Start failed: ${e.message}`);
    }
  }

  async function stopTraining() {
    try {
      await fetch(`${API_BASE}/api/rl/train/stop`, { method: "POST" });
      addLog("warning", "Training stop requested");
    } catch {}
  }

  async function resumeCircuitBreaker() {
    try {
      await fetch(`${API_BASE}/api/rl/circuit-breaker/resume`, { method: "POST" });
      setCircuit(false);
      addLog("success", "Circuit breaker reset — trading resumed");
    } catch {}
  }

  // Poll status on mount
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10_000);
    return () => clearInterval(interval);
  }, []);

  // ── Derived values ──────────────────────────────────────────────────────────

  const equityVal    = trainStatus.equity || stats.equity;
  const totalReturn  = ((equityVal - INITIAL_EQUITY) / INITIAL_EQUITY) * 100;
  const totalPnL     = equityVal - INITIAL_EQUITY;
  const isRunning    = trainStatus.state === "running";
  const progressPct  = Math.min((trainStatus.totalSteps / trainSteps) * 100, 100);

  // ── Navigation tabs ─────────────────────────────────────────────────────────

  const tabs = [
    { id: "dashboard", label: "Dashboard", icon: Activity },
    { id: "training",  label: "Training",  icon: Cpu },
    { id: "config",    label: "Config",    icon: Settings },
    { id: "backtest",  label: "Backtest",  icon: BarChart3 },
  ] as const;

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-bg-base bg-grid">
      {showConfirm && (
        <LiveModeConfirmModal
          onConfirm={() => { setMode("live"); setShowConfirm(false); addLog("warning", "⚠️ Switched to LIVE mode — real funds at risk!"); }}
          onCancel={() => setShowConfirm(false)}
        />
      )}

      {/* ── Mode Banner ──────────────────────────────────────────────────────── */}
      {mode === "demo" ? (
        <div className="demo-banner">
          ⚠️ DEMO / VIRTUAL TRAINING MODE — No real funds at risk — Delta Exchange India Testnet
        </div>
      ) : (
        <div className="live-banner">
          🔴 LIVE PRODUCTION MODE — REAL FUNDS AT RISK — Delta Exchange India Production
        </div>
      )}

      {/* ── Header ───────────────────────────────────────────────────────────── */}
      <header className="glass border-b border-bg-border sticky top-0 z-40">
        <div className="max-w-screen-2xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-1.5 rounded-lg bg-brand-primary/20">
              <Bot className="w-5 h-5 text-brand-primary" />
            </div>
            <span className="font-bold text-base tracking-tight neon-text-blue">DeltaRL Trader</span>
            <span className="text-xs text-text-muted hidden sm:block">v2 · India</span>
            {/* Risk-Mitigation Mode Badge */}
            <span className="hidden md:flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border bg-brand-primary/10 border-brand-primary/30 text-brand-primary">
              <Shield className="w-3 h-3" /> RISK-MITIGATION MODE
            </span>
          </div>

          <nav className="flex items-center gap-1">
            {tabs.map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  activeTab === t.id
                    ? "bg-brand-primary/20 text-brand-primary"
                    : "text-text-secondary hover:text-text-primary hover:bg-bg-hover"
                }`}
              >
                <t.icon className="w-4 h-4" />
                <span className="hidden md:block">{t.label}</span>
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            {/* Circuit Breaker indicator */}
            {circuitTriggered && (
              <button
                onClick={resumeCircuitBreaker}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-signal-short/20 border border-signal-short/40 text-signal-short text-xs font-bold hover:bg-signal-short/30 transition-colors animate-pulse-slow"
              >
                <Shield className="w-3.5 h-3.5" />
                CIRCUIT BREAKER
              </button>
            )}

            {/* Connection status */}
            <div className="flex items-center gap-1.5 text-xs">
              <span className={`pulse-dot ${connected ? "green" : "red"}`} />
              <span className={connected ? "text-signal-long" : "text-signal-short"}>
                {connected ? "Live" : "Offline"}
              </span>
            </div>

            {/* Mode toggle */}
            <button
              onClick={() => mode === "demo" ? setShowConfirm(true) : setMode("demo")}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-bold transition-all border ${
                mode === "live"
                  ? "bg-signal-short/20 border-signal-short/50 text-signal-short hover:bg-signal-short/30 shadow-red-glow"
                  : "bg-signal-warning/10 border-signal-warning/30 text-signal-warning hover:bg-signal-warning/20"
              }`}
            >
              <Globe className="w-3.5 h-3.5" />
              {mode === "demo" ? "DEMO" : "🔴 LIVE"}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-screen-2xl mx-auto px-4 py-6">

        {/* ── DASHBOARD TAB ──────────────────────────────────────────────────── */}
        {activeTab === "dashboard" && (
          <div className="space-y-6 animate-fade-in">
            {/* Metrics row — v2: 8 cards with risk-adjusted metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
              <MetricCard
                label="Equity"
                value={`$${equityVal.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                sub={`${totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(2)}%`}
                trend={totalPnL >= 0 ? "up" : "down"}
                color={totalPnL >= 0 ? "text-signal-long" : "text-signal-short"}
                icon={Database}
              />
              <MetricCard
                label="Total P&L"
                value={`${totalPnL >= 0 ? "+" : ""}$${Math.abs(totalPnL).toFixed(0)}`}
                color={totalPnL >= 0 ? "pnl-positive" : "pnl-negative"}
                icon={totalPnL >= 0 ? TrendingUp : TrendingDown}
              />
              <MetricCard
                label="Mark Price"
                value={stats.currentPrice > 0 ? `$${stats.currentPrice.toLocaleString()}` : "—"}
                sub="BTC-USDT-PERP"
                icon={Activity}
              />
              <MetricCard
                label="Trades"
                value={String(trainStatus.numTrades || 0)}
                sub={`${stats.winRate > 0 ? `Win ${stats.winRate.toFixed(0)}%` : "—"}`}
                icon={BarChart3}
              />
              {/* v2: risk-adjusted metrics */}
              <MetricCard
                label="Sharpe"
                value={riskMetrics.rollingSharee ? riskMetrics.rollingSharee.toFixed(2) : "—"}
                color={riskMetrics.rollingSharee > 1 ? "text-signal-long" : riskMetrics.rollingSharee > 0 ? "text-signal-warning" : "text-signal-short"}
                sub="Risk-adjusted"
                icon={Zap}
              />
              <MetricCard
                label="Sortino"
                value={riskMetrics.rollingSortino ? riskMetrics.rollingSortino.toFixed(2) : "—"}
                color={riskMetrics.rollingSortino > 1 ? "text-signal-long" : "text-text-secondary"}
                sub="Downside-only"
                icon={TrendingDown}
              />
              <MetricCard
                label="Max DD"
                value={riskMetrics.maxDdEpisode > 0 ? `${(riskMetrics.maxDdEpisode * 100).toFixed(1)}%` : "0%"}
                color={riskMetrics.maxDdEpisode > 0.15 ? "text-signal-short" : riskMetrics.maxDdEpisode > 0.05 ? "text-signal-warning" : "text-signal-long"}
                sub={riskMetrics.maxDdEpisode > 0.05 ? "⚠ Above 5% threshold" : "Within limit"}
                icon={Shield}
              />
              <MetricCard
                label="Calmar"
                value={riskMetrics.calmarRatio ? riskMetrics.calmarRatio.toFixed(2) : "—"}
                color={riskMetrics.calmarRatio > 1 ? "text-signal-long" : "text-text-secondary"}
                sub="Return/MaxDD"
                icon={Gauge}
              />
            </div>

            {/* Risk Panel — NEW in v2 */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              {/* Current Drawdown Gauge */}
              <div className="gradient-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-signal-short" />
                    <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Portfolio Drawdown</span>
                  </div>
                  <span className="text-xs text-text-muted" title="Episode ends if drawdown exceeds 15%">?</span>
                </div>
                <div className="text-2xl font-bold font-mono mb-2" style={{
                  color: riskMetrics.portfolioDrawdown > 0.15 ? "#EF4444" :
                         riskMetrics.portfolioDrawdown > 0.05 ? "#F59E0B" : "#10B981"
                }}>
                  {(riskMetrics.portfolioDrawdown * 100).toFixed(1)}%
                </div>
                <div className="relative h-2 bg-bg-border rounded-full overflow-hidden">
                  <div
                    className="absolute top-0 left-0 h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${Math.min(riskMetrics.portfolioDrawdown / 0.15 * 100, 100)}%`,
                      background: riskMetrics.portfolioDrawdown > 0.15 ? "#EF4444" :
                                  riskMetrics.portfolioDrawdown > 0.05 ? "#F59E0B" : "#10B981"
                    }}
                  />
                </div>
                <div className="flex justify-between text-[9px] text-text-muted mt-1">
                  <span>0%</span><span>5% warn</span><span>15% limit</span>
                </div>
              </div>

              {/* Volatility Regime */}
              <div className="gradient-border rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Wind className="w-4 h-4 text-brand-primary" />
                  <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Volatility Regime</span>
                </div>
                <div className={`text-2xl font-bold mb-2 ${REGIME_COLORS[getRegimeIdx(riskMetrics.volatilityRegime)]}`}>
                  {REGIME_LABELS[getRegimeIdx(riskMetrics.volatilityRegime)]}
                </div>
                <div className="flex gap-1">
                  {[0, 1, 2].map(i => (
                    <div key={i} className={`flex-1 h-2 rounded-full transition-all duration-700 ${
                      getRegimeIdx(riskMetrics.volatilityRegime) >= i
                        ? i === 0 ? "bg-signal-long" : i === 1 ? "bg-signal-warning" : "bg-signal-short"
                        : "bg-bg-border"
                    }`} />
                  ))}
                </div>
                <div className="text-xs text-text-muted mt-2">
                  ATR pct: {(riskMetrics.volatilityRegime * 100).toFixed(0)}%
                  {riskMetrics.volatilityRegime > 0.85 &&
                    <span className="ml-2 text-signal-short font-semibold">⚠ Safety Policy Active</span>
                  }
                </div>
              </div>

              {/* Funding Exposure */}
              <div className="gradient-border rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Flame className="w-4 h-4 text-signal-warning" />
                  <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Funding Exposure</span>
                </div>
                <div className="text-2xl font-bold font-mono mb-2" style={{
                  color: Math.abs(riskMetrics.fundingRate) > 0.001 ? "#F59E0B" : "#10B981"
                }}>
                  {riskMetrics.fundingRate >= 0 ? "+" : ""}{(riskMetrics.fundingRate * 100).toFixed(4)}%
                </div>
                <div className="text-xs text-text-muted">
                  Per 8h funding rate
                  {Math.abs(riskMetrics.fundingRate) > 0.001 &&
                    <div className="text-signal-warning mt-1">⚠ Adverse funding — holding cost high</div>
                  }
                </div>
              </div>

              {/* Curriculum Stage + Risk Score */}
              <div className="gradient-border rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Lock className="w-4 h-4 text-brand-primary" />
                  <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">Curriculum</span>
                </div>
                <div className="flex items-center gap-2 mb-3">
                  {[0, 1, 2].map(i => (
                    <div key={i} className={`flex-1 py-1.5 rounded-lg text-center text-[10px] font-bold transition-all ${
                      curriculumStage === i
                        ? "bg-brand-primary/20 border border-brand-primary text-brand-primary"
                        : curriculumStage > i
                          ? "bg-signal-long/10 border border-signal-long/30 text-signal-long"
                          : "bg-bg-hover border border-bg-border text-text-muted"
                    }`}>
                      {["Low Vol", "Medium", "Full"][i]}
                    </div>
                  ))}
                </div>
                <div className="text-xs text-text-muted">
                  Stage {curriculumStage}/2 — {["Low-volatility only", "Low + medium vol", "All regimes"][curriculumStage]}
                </div>
                {riskMetrics.inSafetyZone && (
                  <div className="mt-2 text-[10px] font-bold text-signal-short bg-signal-short/10 rounded-lg px-2 py-1">
                    ⚠ IN SAFETY ZONE — Conservative policy active
                  </div>
                )}
              </div>
            </div>

            {/* Equity curve + Log panel */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Equity curve (simple sparkline) */}
              <div className="lg:col-span-2 gradient-border rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-brand-primary" /> Equity Curve
                  </h3>
                  <span className="text-xs text-text-muted">{equityCurve.length} points</span>
                </div>
                {equityCurve.length > 2 ? (
                  <div className="w-full relative h-40 flex items-end">
                    <EquityChart data={equityCurve} />
                  </div>
                ) : (
                  <div className="h-40 flex items-center justify-center text-text-muted text-sm">
                    Equity curve will appear once training starts
                  </div>
                )}
              </div>

              {/* Open positions panel */}
              <div className="gradient-border rounded-xl p-5">
                <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-brand-primary" /> Open Positions
                </h3>
                {positions.length === 0 ? (
                  <div className="text-text-muted text-sm text-center py-8">
                    No open positions
                  </div>
                ) : (
                  positions.map((p, i) => (
                    <div key={i} className="flex items-center justify-between py-2 border-b border-bg-border last:border-0">
                      <div>
                        <div className="text-xs font-semibold text-text-primary">{p.symbol}</div>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          <Badge
                            label={p.side > 0 ? "LONG" : "SHORT"}
                            variant={p.side > 0 ? "success" : "danger"}
                          />
                          <span className="text-xs text-text-muted">{p.leverage}x</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-sm font-mono font-bold ${p.pnl >= 0 ? "text-signal-long" : "text-signal-short"}`}>
                          {p.pnl >= 0 ? "+" : ""}{p.pnl.toFixed(2)}
                        </div>
                        <div className="text-xs text-text-muted">${p.markPrice.toFixed(2)}</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Log terminal */}
            <div className="gradient-border rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-brand-primary" /> Live Log
                </h3>
                <button
                  onClick={() => setLogs([])}
                  className="text-xs text-text-muted hover:text-text-secondary transition-colors"
                >
                  Clear
                </button>
              </div>
              <div ref={logRef} className="log-terminal h-48 overflow-y-auto space-y-0.5 pr-1">
                {logs.length === 0 ? (
                  <div className="text-text-muted text-center py-8 text-xs">
                    No logs yet. Connect the backend and start training.
                  </div>
                ) : (
                  logs.map(entry => (
                    <div key={entry.id} className={`flex gap-2 log-${entry.level}`}>
                      <span className="text-text-muted flex-shrink-0">[{entry.ts}]</span>
                      <span>{entry.msg}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── TRAINING TAB ─────────────────────────────────────────────────────── */}
        {activeTab === "training" && (
          <div className="space-y-6 animate-fade-in">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Training controls */}
              <div className="gradient-border rounded-xl p-6 space-y-5">
                <h2 className="text-base font-bold text-text-primary flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-brand-primary" /> Training Controls
                </h2>

                {/* Mode select */}
                <div>
                  <label className="block text-xs text-text-secondary mb-2 font-medium uppercase tracking-wider">
                    Training Mode
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {(["simulation", "demo", "live"] as const).map(m => (
                      <button
                        key={m}
                        onClick={() => setTrainMode(m)}
                        className={`py-2.5 rounded-xl text-xs font-semibold border transition-all ${
                          trainMode === m
                            ? m === "live"
                              ? "bg-signal-short/20 border-signal-short text-signal-short"
                              : "bg-brand-primary/20 border-brand-primary text-brand-primary"
                            : "bg-bg-hover border-bg-border text-text-secondary hover:text-text-primary"
                        }`}
                      >
                        {m.toUpperCase()}
                        {m === "simulation" && (
                          <div className="text-[9px] mt-0.5 opacity-70">No orders</div>
                        )}
                        {m === "demo" && (
                          <div className="text-[9px] mt-0.5 opacity-70">Testnet</div>
                        )}
                        {m === "live" && (
                          <div className="text-[9px] mt-0.5 opacity-70 text-signal-short">⚠ Real $</div>
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Steps slider */}
                <div>
                  <label className="block text-xs text-text-secondary mb-2 font-medium uppercase tracking-wider">
                    Training Steps: <span className="text-brand-primary font-mono">{trainSteps.toLocaleString()}</span>
                  </label>
                  <input
                    type="range"
                    min={10000} max={2000000} step={10000}
                    value={trainSteps}
                    onChange={e => setTrainSteps(Number(e.target.value))}
                    className="w-full accent-brand-primary"
                  />
                  <div className="flex justify-between text-xs text-text-muted mt-1">
                    <span>10k</span><span>500k</span><span>1M</span><span>2M</span>
                  </div>
                </div>

                {/* Start/Stop */}
                <div className="flex gap-3">
                  <button
                    onClick={startTraining}
                    disabled={isRunning}
                    className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-bold text-sm transition-all ${
                      isRunning
                        ? "bg-bg-border text-text-muted cursor-not-allowed"
                        : "bg-brand-primary hover:bg-cyan-400 text-bg-base shadow-glow-md"
                    }`}
                  >
                    <Play className="w-4 h-4" />
                    Start Training
                  </button>
                  <button
                    onClick={stopTraining}
                    disabled={!isRunning}
                    className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl font-bold text-sm transition-all border ${
                      !isRunning
                        ? "border-bg-border text-text-muted cursor-not-allowed"
                        : "border-signal-short/50 text-signal-short hover:bg-signal-short/10"
                    }`}
                  >
                    <Square className="w-4 h-4" />
                    Stop
                  </button>
                </div>

                {/* Progress bar */}
                {isRunning && (
                  <div>
                    <div className="flex justify-between text-xs text-text-muted mb-1.5">
                      <span>Progress</span>
                      <span>{progressPct.toFixed(1)}%</span>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
                    </div>
                  </div>
                )}
              </div>

              {/* Training stats */}
              <div className="gradient-border rounded-xl p-6">
                <h3 className="text-base font-bold text-text-primary mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-brand-primary" /> Training Metrics
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <MetricCard label="Episodes" value={String(trainStatus.epoch)} icon={RefreshCw} />
                  <MetricCard
                    label="Total Steps"
                    value={`${(trainStatus.totalSteps / 1000).toFixed(1)}k`}
                    icon={BarChart3}
                  />
                  <MetricCard
                    label="Mean Reward"
                    value={trainStatus.meanReward.toFixed(4)}
                    color={trainStatus.meanReward >= 0 ? "text-signal-long" : "text-signal-short"}
                    icon={TrendingUp}
                  />
                  <MetricCard
                    label="Equity"
                    value={`$${(trainStatus.equity || INITIAL_EQUITY).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    icon={Database}
                  />
                  {/* v2: risk-adjusted training metrics */}
                  <MetricCard
                    label="Sharpe"
                    value={riskMetrics.rollingSharee ? riskMetrics.rollingSharee.toFixed(2) : "—"}
                    color={riskMetrics.rollingSharee > 1 ? "text-signal-long" : "text-signal-warning"}
                    sub="Target: >1.0"
                    icon={Zap}
                  />
                  <MetricCard
                    label="Sortino"
                    value={riskMetrics.rollingSortino ? riskMetrics.rollingSortino.toFixed(2) : "—"}
                    color={riskMetrics.rollingSortino > 1 ? "text-signal-long" : "text-text-secondary"}
                    sub="Downside adj."
                    icon={TrendingDown}
                  />
                  <MetricCard
                    label="Max DD"
                    value={riskMetrics.maxDdEpisode > 0 ? `${(riskMetrics.maxDdEpisode * 100).toFixed(1)}%` : "0%"}
                    color={riskMetrics.maxDdEpisode > 0.15 ? "text-signal-short" : riskMetrics.maxDdEpisode > 0.05 ? "text-signal-warning" : "text-signal-long"}
                    sub="Hard limit: 15%"
                    icon={Shield}
                  />
                  <MetricCard
                    label="Calmar"
                    value={riskMetrics.calmarRatio ? riskMetrics.calmarRatio.toFixed(2) : "—"}
                    color={riskMetrics.calmarRatio > 1 ? "text-signal-long" : "text-text-secondary"}
                    sub="Return/MaxDD"
                    icon={Gauge}
                  />
                </div>

                {/* Curriculum stage selector */}
                <div className="mt-4 pt-4 border-t border-bg-border">
                  <div className="text-xs text-text-secondary mb-2 font-medium uppercase tracking-wider">Curriculum Stage</div>
                  <div className="flex gap-2">
                    {[{n:0,l:"Stage 0",d:"Low Vol"}, {n:1,l:"Stage 1",d:"Medium"}, {n:2,l:"Stage 2",d:"Full"}].map(s => (
                      <button
                        key={s.n}
                        onClick={async () => {
                          await fetch(`${API_BASE}/api/rl/curriculum/stage`, {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({stage: s.n}),
                          });
                          setCurriculumStage(s.n);
                          addLog("info", `Curriculum set to Stage ${s.n} (${s.d})`);
                        }}
                        className={`flex-1 py-2 rounded-xl text-xs font-semibold border transition-all ${
                          curriculumStage === s.n
                            ? "bg-brand-primary/20 border-brand-primary text-brand-primary"
                            : "bg-bg-hover border-bg-border text-text-muted hover:text-text-primary"
                        }`}
                      >
                        <div>{s.l}</div>
                        <div className="text-[9px] opacity-70">{s.d}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Reward chart sparkline */}
                {rewardCurve.length > 2 && (
                  <div className="mt-4">
                    <div className="text-xs text-text-muted mb-2">Reward History (rolling)</div>
                    <SparkLine
                      data={rewardCurve}
                      color={
                        rewardCurve[rewardCurve.length - 1] >= 0 ? "#10B981" : "#EF4444"
                      }
                    />
                  </div>
                )}

                {/* Model import/export */}
                <div className="flex gap-2 mt-5 pt-4 border-t border-bg-border">
                  <a
                    href={`${API_BASE}/api/rl/model/export/1`}
                    className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-semibold bg-bg-hover hover:bg-bg-border text-text-secondary hover:text-text-primary border border-bg-border transition-all"
                  >
                    <Download className="w-3.5 h-3.5" /> Export Model
                  </a>
                  <label className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-xs font-semibold bg-bg-hover hover:bg-bg-border text-text-secondary hover:text-text-primary border border-bg-border transition-all cursor-pointer">
                    <Upload className="w-3.5 h-3.5" /> Import Model
                    <input type="file" accept=".zip" className="hidden" onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      const fd = new FormData();
                      fd.append("file", file);
                      const r = await fetch(`${API_BASE}/api/rl/model/import`, { method: "POST", body: fd });
                      if (r.ok) addLog("success", `Model imported: ${file.name}`);
                      else addLog("error", "Model import failed");
                    }} />
                  </label>
                </div>
              </div>
            </div>

            {/* Training log */}
            <div className="gradient-border rounded-xl p-5">
              <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-brand-primary" /> Training Log
              </h3>
              <div className="log-terminal h-56 overflow-y-auto space-y-0.5 pr-1">
                {logs.map(entry => (
                  <div key={entry.id} className={`flex gap-2 log-${entry.level}`}>
                    <span className="text-text-muted flex-shrink-0">[{entry.ts}]</span>
                    <span>{entry.msg}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── CONFIG TAB ───────────────────────────────────────────────────────── */}
        {activeTab === "config" && (
          <div className="max-w-2xl space-y-6 animate-fade-in">
            <div className="gradient-border rounded-xl p-6">
              <h2 className="text-base font-bold text-text-primary mb-6 flex items-center gap-2">
                <Settings className="w-5 h-5 text-brand-primary" /> Configuration
              </h2>

              <div className="space-y-5">
                {/* API key status */}
                <div>
                  <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wider">API Keys</h3>
                  <p className="text-xs text-text-muted mb-3">
                    API keys are configured via your <code className="text-brand-primary">.env</code> file. They are never stored in the database or sent to the frontend.
                  </p>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between bg-bg-hover rounded-xl px-4 py-3">
                      <span className="text-sm text-text-secondary">Demo API Key</span>
                      <Badge label="Set via .env" variant="primary" />
                    </div>
                    <div className="flex items-center justify-between bg-bg-hover rounded-xl px-4 py-3">
                      <span className="text-sm text-text-secondary">Live API Key</span>
                      <Badge label="Set via .env" variant="warning" />
                    </div>
                  </div>
                </div>

                {/* Risk settings info */}
                <div>
                  <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wider">Risk Settings</h3>
                  <div className="space-y-2 text-sm text-text-secondary bg-bg-hover rounded-xl p-4">
                    <div className="flex justify-between">
                      <span>Max Risk Per Trade</span>
                      <span className="text-brand-primary font-mono">1%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Circuit Breaker Drawdown</span>
                      <span className="text-signal-warning font-mono">5%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Max Leverage</span>
                      <span className="text-brand-primary font-mono">20x</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Simulation Only</span>
                      <span className="text-signal-long font-mono">true</span>
                    </div>
                    <p className="text-xs text-text-muted pt-2 border-t border-bg-border">
                      Edit these in <code className="text-brand-primary">.env</code> and restart the backend to apply.
                    </p>
                  </div>
                </div>

                {/* Endpoint info */}
                <div>
                  <h3 className="text-sm font-semibold text-text-secondary mb-3 uppercase tracking-wider">Endpoints (India)</h3>
                  <div className="space-y-2 text-xs font-mono text-text-muted bg-bg-hover rounded-xl p-4">
                    <div><span className="text-signal-warning">Demo REST: </span>cdn-ind.testnet.deltaex.org</div>
                    <div><span className="text-signal-warning">Demo WS:   </span>socket.ind.testnet.deltaex.org</div>
                    <div><span className="text-signal-short">Live REST: </span>api.india.delta.exchange</div>
                    <div><span className="text-signal-short">Live WS:   </span>socket.india.delta.exchange</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Safety checklist */}
            <div className="gradient-border rounded-xl p-6">
              <h3 className="text-sm font-bold text-text-primary mb-4 flex items-center gap-2">
                <Shield className="w-4 h-4 text-brand-primary" /> Pre-Live Safety Checklist
              </h3>
              <div className="space-y-2 text-sm">
                {[
                  "Trained for at least 200k steps in Simulation mode",
                  "Backtested on 6+ months of historical data",
                  "Sharpe ratio > 1.0 in backtest",
                  "Max drawdown < 15% in backtest",
                  "Live API key configured with Trading permissions only",
                  "Telegram/Discord alerts configured",
                  "Circuit breaker threshold set (default 5%)",
                  "Start with small position size (reduce max_risk_per_trade to 0.002)",
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <div className="w-4 h-4 rounded border border-bg-border flex-shrink-0 mt-0.5" />
                    <span className="text-text-secondary">{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── BACKTEST TAB ─────────────────────────────────────────────────────── */}
        {activeTab === "backtest" && (
          <BacktestPanel apiBase={API_BASE} onLog={addLog} />
        )}
      </main>
    </div>
  );
}

// ── Equity Chart (SVG-based) ──────────────────────────────────────────────────

function EquityChart({ data }: { data: number[] }) {
  const w = 800, h = 160, pad = 10;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / range) * (h - 2 * pad);
    return `${x},${y}`;
  }).join(" ");

  const area = `M${pad},${h - pad} L${pts.split(" ")[0]} L${pts} L${w - pad},${h - pad} Z`;
  const isPositive = data[data.length - 1] >= data[0];
  const color = isPositive ? "#10B981" : "#EF4444";

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="eq-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill="url(#eq-grad)" />
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ── Backtest Panel ────────────────────────────────────────────────────────────

function BacktestPanel({ apiBase, onLog }: { apiBase: string; onLog: (l: "error" | "info" | "success" | "warning" | "trade", m: string) => void }) {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<any | null>(null);
  const [symbol, setSymbol]   = useState("BTCUSDT");
  const [bars, setBars]       = useState(1000);

  async function run() {
    setRunning(true);
    onLog("info", `Starting backtest: ${symbol}, ${bars} bars...`);
    try {
      const r = await fetch(`${apiBase}/api/rl/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, resolution: "5m", lookback_bars: bars }),
      });
      const d = await r.json();
      if (!r.ok) { onLog("error", d.detail); return; }
      setResults(d);
      onLog("success", `Backtest done: ${d.total_return_pct}% return | Sharpe: ${d.sharpe_ratio}`);
    } catch (e: any) {
      onLog("error", `Backtest failed: ${e.message}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="max-w-3xl space-y-5 animate-fade-in">
      <div className="gradient-border rounded-xl p-6">
        <h2 className="text-base font-bold text-text-primary mb-5 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-brand-primary" /> Backtesting
        </h2>
        <div className="flex gap-3 mb-4">
          <div className="flex-1">
            <label className="text-xs text-text-muted mb-1 block">Symbol</label>
            <select
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
              className="w-full bg-bg-hover border border-bg-border rounded-xl px-3 py-2 text-sm text-text-primary"
            >
              {["BTCUSDT", "ETHUSDT", "SOLUSDT"].map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div className="flex-1">
            <label className="text-xs text-text-muted mb-1 block">Lookback Bars</label>
            <input
              type="number" min={100} max={5000} step={100}
              value={bars}
              onChange={e => setBars(Number(e.target.value))}
              className="w-full bg-bg-hover border border-bg-border rounded-xl px-3 py-2 text-sm text-text-primary font-mono"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={run}
              disabled={running}
              className={`flex items-center gap-2 px-5 py-2 rounded-xl font-bold text-sm transition-all ${
                running
                  ? "bg-bg-border text-text-muted cursor-not-allowed"
                  : "bg-brand-primary hover:bg-cyan-400 text-bg-base shadow-glow-sm"
              }`}
            >
              {running ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {running ? "Running..." : "Run Backtest"}
            </button>
          </div>
        </div>
      </div>

      {results && (
        <div className="gradient-border rounded-xl p-6 animate-fade-in">
          <h3 className="text-sm font-bold text-text-primary mb-4">Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: "Total Return",  value: `${results.total_return_pct > 0 ? "+" : ""}${results.total_return_pct}%`,  color: results.total_return_pct >= 0 ? "text-signal-long" : "text-signal-short" },
              { label: "Annual Return", value: `${results.annual_return_pct > 0 ? "+" : ""}${results.annual_return_pct}%`, color: results.annual_return_pct >= 0 ? "text-signal-long" : "text-signal-short" },
              { label: "Sharpe Ratio",  value: results.sharpe_ratio,  color: results.sharpe_ratio >= 1 ? "text-signal-long" : "text-signal-warning" },
              { label: "Sortino Ratio", value: results.sortino_ratio, color: "text-text-primary" },
              { label: "Max Drawdown",  value: `-${results.max_drawdown_pct}%`, color: results.max_drawdown_pct < 10 ? "text-signal-long" : "text-signal-short" },
              { label: "Win Rate",      value: `${results.win_rate_pct}%`, color: results.win_rate_pct >= 50 ? "text-signal-long" : "text-signal-short" },
              { label: "Profit Factor", value: results.profit_factor, color: results.profit_factor >= 1.5 ? "text-signal-long" : "text-signal-warning" },
              { label: "Num Trades",    value: results.num_trades, color: "text-text-primary" },
            ].map((m, i) => (
              <div key={i} className="bg-bg-hover rounded-xl p-3">
                <div className="text-xs text-text-muted mb-1">{m.label}</div>
                <div className={`text-lg font-bold font-mono ${m.color}`}>{m.value}</div>
              </div>
            ))}
          </div>

          {/* Equity curve of backtest */}
          {results.equity_curve && results.equity_curve.length > 2 && (
            <div className="mt-5">
              <div className="text-xs text-text-muted mb-2">Backtest Equity Curve</div>
              <div className="h-32 w-full">
                <EquityChart data={results.equity_curve} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
