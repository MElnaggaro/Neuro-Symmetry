/**
 * Mirror — AR Feathered Mirror Therapy Module (Phase 3)
 *
 * Uses `rawScreenLandmarksRef` (raw screen coordinates) to calculate the
 * facial midline, then mirrors the healthy side over the affected side with
 * a horizontal gradient alpha-blend (feathering) for a seamless edge.
 *
 * Architecture:
 *   - Canvas drawing reads rawScreenLandmarksRef (screen coords — no math alignment)
 *   - `useRafLoop` handles the RAF lifecycle and cleanup
 */

import { useRef, useState, useEffect } from "react";
import { useFaceTracking } from "@/providers/FaceTrackingProvider";
import { VIDEO_CONFIG } from "@/config";
import { useRafLoop } from "@/hooks/useRafLoop";

const { WIDTH, HEIGHT } = VIDEO_CONFIG;
const FEATHER_WIDTH = 40;
const NOSE_TIP  = 1;
const FOREHEAD  = 10;

export type MirrorSide = "LEFT" | "RIGHT";

export default function Mirror() {
  const { videoRef, rawScreenLandmarksRef, status: trackingStatus, start } = useFaceTracking();

  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const offscreenRef  = useRef<OffscreenCanvas | null>(null);
  const featherRef    = useRef<OffscreenCanvas | null>(null);
  const lastSourceWRef = useRef(0);
  const [mirrorSide, setMirrorSide] = useState<MirrorSide>("RIGHT");

  useEffect(() => { start(); }, [start]);

  // ── RAF loop — feathered mirror rendering ─────────────────────────────
  useRafLoop(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    const lm     = rawScreenLandmarksRef.current;
    const ctx    = canvas?.getContext("2d");

    if (!canvas || !video || !ctx || !lm || lm.length <= FOREHEAD) return;

    const w = canvas.width;
    const h = canvas.height;

    // Step 1: Draw the live video frame (mirrored to match CSS scaleX(-1))
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    // Step 2: Calculate the facial midline from raw screen coords
    const midlineX = w * (1 - (lm[NOSE_TIP].x + lm[FOREHEAD].x) / 2);

    // Step 3: Mirror the healthy side over the affected side
    const sourceX = mirrorSide === "RIGHT" ? 0       : midlineX;
    const sourceW = mirrorSide === "RIGHT" ? midlineX : w - midlineX;
    const destX   = mirrorSide === "RIGHT" ? midlineX : 0;

    if (sourceW > 10) {
      const roundedW = Math.ceil(sourceW);
      if (roundedW !== lastSourceWRef.current) {
        lastSourceWRef.current = roundedW;
        offscreenRef.current   = new OffscreenCanvas(roundedW, h);
        featherRef.current     = new OffscreenCanvas(roundedW, h);
      }

      const offscreen     = offscreenRef.current!;
      const featherCanvas = featherRef.current!;

      // Draw mirrored half into offscreen buffer
      const offCtx = offscreen.getContext("2d")!;
      offCtx.clearRect(0, 0, offscreen.width, h);
      offCtx.save();
      offCtx.translate(sourceW, 0);
      offCtx.scale(-1, 1);
      offCtx.drawImage(canvas, sourceX, 0, sourceW, h, 0, 0, sourceW, h);
      offCtx.restore();

      // Step 4: Apply feather gradient as a compositing mask
      const featherCtx = featherCanvas.getContext("2d")!;
      featherCtx.clearRect(0, 0, featherCanvas.width, h);
      featherCtx.drawImage(offscreen, 0, 0);
      featherCtx.globalCompositeOperation = "destination-in";
      const gradient = featherCtx.createLinearGradient(
        mirrorSide === "RIGHT" ? 0       : sourceW,
        0,
        mirrorSide === "RIGHT" ? FEATHER_WIDTH : sourceW - FEATHER_WIDTH,
        0,
      );
      gradient.addColorStop(0, "rgba(255,255,255,0)");
      gradient.addColorStop(1, "rgba(255,255,255,1)");
      featherCtx.fillStyle = gradient;
      featherCtx.fillRect(0, 0, sourceW, h);

      // Step 5: Composite feathered mirror onto main canvas
      ctx.drawImage(featherCanvas, destX, 0);
    }
  }, trackingStatus === "ready", "Mirror");

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
        <canvas ref={canvasRef} width={WIDTH} height={HEIGHT} className="w-full h-full" />
      </div>

      <p className="text-[10px] text-slate-500 leading-relaxed">
        The healthy side is mirrored over the affected side with a feathered
        blend. Use this during mirror therapy exercises.
      </p>
    </div>
  );
}
