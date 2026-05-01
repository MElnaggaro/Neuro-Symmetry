/**
 * Joypad — Face-Controlled Gamification Module (Phase 3)
 *
 * Polls `mathLandmarksRef` for facial movement events (brow raise, smile,
 * blink) and awards points. A 3-second sustained "Hold" triggers a 2x
 * multiplier with a visual canvas particle burst.
 *
 * Architecture:
 *   - Logic reads from mathLandmarksRef (math-aligned coordinates)
 *   - Canvas effects use a local canvas
 *   - React state updates are throttled to ~4 FPS for the score display
 *   - `useRafLoop` handles the RAF lifecycle and cleanup
 */

import { useRef, useState, useEffect, useCallback } from "react";
import { useFaceTracking } from "@/providers/FaceTrackingProvider";
import { useRafLoop } from "@/hooks/useRafLoop";
import logger from "@/utils/logger";

// ── Landmark indices ──────────────────────────────────────────────────────────

const LM = {
  R_EYE_ANCHOR: 159,
  L_EYE_ANCHOR: 386,
  R_BROW:       70,
  L_BROW:       300,
  MOUTH_L:      61,
  MOUTH_R:      291,
  MOUTH_TOP:    13,
  MOUTH_BOT:    14,
} as const;

// ── Thresholds ────────────────────────────────────────────────────────────────

const BROW_RAISE_THRESHOLD = 0.03;
const SMILE_THRESHOLD      = 0.05;
const HOLD_DURATION_MS     = 3000;
const SCORE_UPDATE_MS      = 250;

// ── Types ─────────────────────────────────────────────────────────────────────

interface GameState {
  score:        number;
  multiplier:   number;
  combo:        string | null;
  holdProgress: number;
}

interface Particle {
  x: number; y: number; vx: number; vy: number; life: number; color: string;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Joypad() {
  const { mathLandmarksRef, status: trackingStatus, start } = useFaceTracking();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [game, setGame] = useState<GameState>({ score: 0, multiplier: 1, combo: null, holdProgress: 0 });

  // ── Mutable game state (no renders) ───────────────────────────────────
  const scoreRef         = useRef(0);
  const multiplierRef    = useRef(1);
  const holdStartRef     = useRef(0);
  const activeActionRef  = useRef<string | null>(null);
  const lastScoreUpdateRef = useRef(0);
  const particlesRef     = useRef<Particle[]>([]);

  useEffect(() => { start(); }, [start]);

  const resetGame = useCallback(() => {
    scoreRef.current        = 0;
    multiplierRef.current   = 1;
    holdStartRef.current    = 0;
    activeActionRef.current = null;
    lastScoreUpdateRef.current = 0;
    particlesRef.current    = [];
    setGame({ score: 0, multiplier: 1, combo: null, holdProgress: 0 });
  }, []);

  // ── RAF loop ──────────────────────────────────────────────────────────
  useRafLoop(() => {
    const now = performance.now();
    const lm  = mathLandmarksRef.current;
    if (!lm || lm.length <= 466) return;

    // Detect actions
    const browLift =
      (lm[LM.R_EYE_ANCHOR].y - lm[LM.R_BROW].y) +
      (lm[LM.L_EYE_ANCHOR].y - lm[LM.L_BROW].y);
    const smileWidth = Math.abs(lm[LM.MOUTH_L].x - lm[LM.MOUTH_R].x);
    const mouthOpen  = Math.abs(lm[LM.MOUTH_TOP].y - lm[LM.MOUTH_BOT].y);

    let currentAction: string | null = null;
    if (browLift > BROW_RAISE_THRESHOLD)  currentAction = "BROW_RAISE";
    else if (smileWidth > SMILE_THRESHOLD) currentAction = "SMILE";
    else if (mouthOpen > 0.04)             currentAction = "MOUTH_OPEN";

    // Hold multiplier
    if (currentAction) {
      if (activeActionRef.current === currentAction) {
        const held = now - holdStartRef.current;
        if (held >= HOLD_DURATION_MS && multiplierRef.current === 1) {
          multiplierRef.current = 2;
          spawnParticles(20, "#fbbf24");
          logger.info("Joypad", "2x multiplier activated", { action: currentAction });
        }
      } else {
        activeActionRef.current = currentAction;
        holdStartRef.current    = now;
        multiplierRef.current   = 1;
      }
      scoreRef.current += 1 * multiplierRef.current;
    } else {
      if (activeActionRef.current) {
        activeActionRef.current = null;
        holdStartRef.current    = 0;
        multiplierRef.current   = 1;
      }
    }

    // Draw particles
    const canvas = canvasRef.current;
    const ctx    = canvas?.getContext("2d");
    if (canvas && ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particlesRef.current = particlesRef.current.filter((p) => {
        p.x += p.vx; p.y += p.vy; p.life -= 0.015;
        if (p.life <= 0) return false;
        ctx.globalAlpha = p.life;
        ctx.fillStyle   = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3 * p.life, 0, Math.PI * 2);
        ctx.fill();
        return true;
      });
      ctx.globalAlpha = 1;
    }

    // Throttled state update
    if (now - lastScoreUpdateRef.current > SCORE_UPDATE_MS) {
      lastScoreUpdateRef.current = now;
      const holdElapsed = activeActionRef.current
        ? Math.min(1, (now - holdStartRef.current) / HOLD_DURATION_MS)
        : 0;
      setGame({
        score:        scoreRef.current,
        multiplier:   multiplierRef.current,
        combo:        activeActionRef.current,
        holdProgress: holdElapsed,
      });
    }
  }, trackingStatus === "ready", "Joypad");

  function spawnParticles(count: number, color: string) {
    for (let i = 0; i < count; i++) {
      particlesRef.current.push({
        x: 120 + Math.random() * 60, y: 60 + Math.random() * 30,
        vx: (Math.random() - 0.5) * 4, vy: (Math.random() - 0.5) * 4 - 2,
        life: 1.0, color,
      });
    }
  }

  // ── Render ────────────────────────────────────────────────────────────
  const holdPct = Math.round(game.holdProgress * 100);

  return (
    <div className="flex flex-col gap-3 p-4 rounded-2xl bg-[#1a1a2e] border border-[#2a2a4a]">
      <div className="flex items-center justify-between">
        <h2 className="text-[13px] font-bold text-slate-200 uppercase tracking-wide">
          Face-Joypad — Rehabilitation Game
        </h2>
        {game.multiplier > 1 && (
          <span className="px-2 py-1 rounded-lg text-[10px] font-bold bg-amber-900/30 border border-amber-500/40 text-amber-400 animate-pulse">
            {game.multiplier}x MULTIPLIER
          </span>
        )}
        <button
          onClick={resetGame}
          className="px-2 py-1 rounded-lg text-[10px] font-bold bg-[#252540] border border-[#2a2a4a] text-slate-500 hover:border-slate-500 hover:text-slate-300 transition-all"
        >
          RESET
        </button>
      </div>

      <div className="relative rounded-xl overflow-hidden bg-[#1e1e35] border border-[#2a2a4a]"
           style={{ height: 120 }}>
        <canvas ref={canvasRef} width={300} height={120} className="w-full h-full" />
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="text-3xl font-mono font-bold text-cyan-400 tabular-nums">{game.score}</div>
            <div className="text-[9px] font-semibold tracking-[0.3em] text-slate-500 uppercase">POINTS</div>
          </div>
        </div>
      </div>

      <div className="flex gap-2">
        <div className={[
          "flex-1 p-2 rounded-lg text-center text-[10px] font-bold tracking-wide border transition-all",
          game.combo
            ? "bg-cyan-900/20 border-cyan-500/30 text-cyan-400"
            : "bg-[#252540] border-[#2a2a4a] text-slate-600",
        ].join(" ")}>
          {game.combo ? game.combo.replace("_", " ") : "WAITING…"}
        </div>
      </div>

      {game.combo && (
        <div className="w-full bg-[#2a2a4a] rounded-full h-2 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-200"
            style={{
              width:      `${holdPct}%`,
              background: holdPct >= 100 ? "#fbbf24" : "linear-gradient(90deg, #06b6d4, #22d3ee)",
            }}
          />
        </div>
      )}

      <p className="text-[10px] text-slate-500 leading-relaxed">
        Raise eyebrows, smile, or open mouth to score points.
        Hold any action for 3 seconds to activate a 2x multiplier!
      </p>
    </div>
  );
}
