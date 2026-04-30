/**
 * Mirror — AR Feathered Mirror Therapy Module (Phase 3)
 *
 * Uses `rawScreenLandmarksRef` (raw screen coordinates) to calculate the
 * facial midline, then mirrors the healthy side over the affected side with
 * a horizontal gradient alpha-blend (feathering) for a seamless edge.
 *
 * Architecture:
 *   - Canvas drawing reads rawScreenLandmarksRef (screen coords — no math alignment)
 *   - This prevents the disjointed visuals during head tilts
 *   - All per-frame rendering is imperative via requestAnimationFrame
 */

import { useEffect, useRef, useState } from "react";
import { useFaceTracking } from "@/providers/FaceTrackingProvider";
import { VIDEO_CONFIG } from "@/config";
import logger from "@/utils/logger";

const { WIDTH, HEIGHT } = VIDEO_CONFIG;
const FEATHER_WIDTH = 40; // pixels — width of the gradient blend zone

// Landmark indices for midline calculation
const NOSE_TIP = 1;
const FOREHEAD = 10;

export type MirrorSide = "LEFT" | "RIGHT";

export default function Mirror() {
  const {
    videoRef,
    rawScreenLandmarksRef,
    status: trackingStatus,
    start,
  } = useFaceTracking();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<OffscreenCanvas | null>(null);
  const featherRef = useRef<OffscreenCanvas | null>(null);
  const lastSourceWRef = useRef(0);
  const [mirrorSide, setMirrorSide] = useState<MirrorSide>("RIGHT");

  // ── Start tracking on mount ───────────────────────────────────────────
  useEffect(() => { start(); }, [start]);

  // ── RAF loop — feathered mirror rendering ─────────────────────────────
  useEffect(() => {
    if (trackingStatus !== "ready") return;

    let running = true;

    const tick = () => {
      if (!running) return;

      const canvas = canvasRef.current;
      const video = videoRef.current;
      const lm = rawScreenLandmarksRef.current;
      const ctx = canvas?.getContext("2d");

      if (canvas && video && ctx && lm && lm.length > FOREHEAD) {
        const w = canvas.width;
        const h = canvas.height;

        // ── Step 1: Draw the live video frame ─────────────────────────
        ctx.save();
        // Mirror the video horizontally (match CSS scaleX(-1))
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, w, h);
        ctx.restore();

        // ── Step 2: Calculate the facial midline from raw screen coords
        // Raw MediaPipe coords are in [0,1] fractional space.
        // After mirroring: midlineX = w - (landmark.x * w) = w * (1 - landmark.x)
        const noseFrac = lm[NOSE_TIP].x;
        const foreheadFrac = lm[FOREHEAD].x;
        // Average for stability; apply mirror transform
        const midlineX = w * (1 - (noseFrac + foreheadFrac) / 2);

        // ── Step 3: Mirror the healthy side over the affected side
        // "Mirror RIGHT" = take the left half of the video, flip it onto the right
        const sourceX = mirrorSide === "RIGHT" ? 0 : midlineX;
        const sourceW = mirrorSide === "RIGHT" ? midlineX : w - midlineX;
        const destX = mirrorSide === "RIGHT" ? midlineX : 0;

        if (sourceW > 10) {
          // Reuse OffscreenCanvas — only recreate when dimensions change
          const roundedW = Math.ceil(sourceW);
          if (roundedW !== lastSourceWRef.current) {
            lastSourceWRef.current = roundedW;
            offscreenRef.current = new OffscreenCanvas(roundedW, h);
            featherRef.current = new OffscreenCanvas(roundedW, h);
          }

          const offscreen = offscreenRef.current!;
          const featherCanvas = featherRef.current!;

          // Draw mirrored half
          const offCtx = offscreen.getContext("2d")!;
          offCtx.clearRect(0, 0, offscreen.width, h);
          offCtx.save();
          offCtx.translate(sourceW, 0);
          offCtx.scale(-1, 1);
          offCtx.drawImage(canvas, sourceX, 0, sourceW, h, 0, 0, sourceW, h);
          offCtx.restore();

          // ── Step 4: Alpha-blending feather mask ──────────────────────
          const featherCtx = featherCanvas.getContext("2d")!;
          featherCtx.clearRect(0, 0, featherCanvas.width, h);
          featherCtx.drawImage(offscreen, 0, 0);

          // Apply the feather gradient as a compositing mask
          featherCtx.globalCompositeOperation = "destination-in";
          const gradient = featherCtx.createLinearGradient(
            mirrorSide === "RIGHT" ? 0 : sourceW,
            0,
            mirrorSide === "RIGHT" ? FEATHER_WIDTH : sourceW - FEATHER_WIDTH,
            0,
          );
          gradient.addColorStop(0, "rgba(255,255,255,0)");
          gradient.addColorStop(1, "rgba(255,255,255,1)");
          featherCtx.fillStyle = gradient;
          featherCtx.fillRect(0, 0, sourceW, h);

          // ── Step 5: Composite the feathered mirror onto the main canvas
          ctx.drawImage(featherCanvas, destX, 0);
        }
      }

      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
    logger.info("Mirror", "RAF loop started");

    return () => {
      running = false;
      logger.info("Mirror", "RAF loop stopped");
    };
  }, [trackingStatus, videoRef, rawScreenLandmarksRef, mirrorSide]);

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-3 p-4 rounded-2xl bg-[#1a1a2e] border border-[#2a2a4a]">
      <div className="flex items-center justify-between">
        <h2 className="text-[13px] font-bold text-slate-200 uppercase tracking-wide">
          AR Mirror — Feathered Blend
        </h2>
        <div className="flex gap-1">
          {(["LEFT", "RIGHT"] as const).map((side) => (
            <button
              key={side}
              onClick={() => setMirrorSide(side)}
              className={[
                "px-3 py-1 rounded-lg text-[10px] font-bold tracking-wide border transition-all",
                mirrorSide === side
                  ? "bg-cyan-900/30 border-cyan-500/40 text-cyan-400"
                  : "bg-[#252540] border-[#2a2a4a] text-slate-500 hover:border-slate-500",
              ].join(" ")}
            >
              Mirror {side}
            </button>
          ))}
        </div>
      </div>

      <div className="relative rounded-xl overflow-hidden border border-[#2a2a4a]"
           style={{ width: WIDTH, height: HEIGHT }}>
        <canvas
          ref={canvasRef}
          width={WIDTH}
          height={HEIGHT}
          className="w-full h-full"
        />
      </div>

      <p className="text-[10px] text-slate-500 leading-relaxed">
        The healthy side is mirrored over the affected side with a feathered
        blend. Use this during mirror therapy exercises.
      </p>
    </div>
  );
}
