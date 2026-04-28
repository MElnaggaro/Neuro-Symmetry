import { useEffect, useRef } from "react";
import type { XAIFeature, XAILevel } from "@/types/analysis";

// Normalised [x, y, w, h] bounding zones for a frontal mirrored-webcam view.
// x = 0 is image-left (= person's right due to mirror flip).
const ZONE_MAP: Record<string, [number, number, number, number]> = {
  ear_right:          [0.27, 0.36, 0.17, 0.15],
  ear_left:           [0.56, 0.36, 0.17, 0.15],
  ear_delta:          [0.24, 0.34, 0.52, 0.18],
  brow_height_right:  [0.26, 0.22, 0.18, 0.12],
  brow_height_left:   [0.56, 0.22, 0.18, 0.12],
  brow_height_delta:  [0.23, 0.19, 0.54, 0.17],
  mouth_y_delta:      [0.35, 0.60, 0.30, 0.13],
  mouth_x_offset:     [0.32, 0.57, 0.36, 0.16],
  texture_score:      [0.18, 0.17, 0.64, 0.66],
};

const FILL:   Record<XAILevel, string> = {
  HIGH:   "rgba(239,68,68,0.20)",
  MEDIUM: "rgba(249,115,22,0.14)",
  LOW:    "rgba(16,185,129,0.09)",
};
const STROKE: Record<XAILevel, string> = {
  HIGH:   "#ef4444cc",
  MEDIUM: "#f97316aa",
  LOW:    "#10b98177",
};
const LABEL:  Record<XAILevel, string> = {
  HIGH:   "#f87171",
  MEDIUM: "#fb923c",
  LOW:    "#34d399",
};

interface HeatmapOverlayProps {
  xai:    XAIFeature[];
  width:  number;
  height: number;
}

export default function HeatmapOverlay({ xai, width, height }: HeatmapOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Limit to top 5 features that have a known zone
    const visible = xai.filter((f) => f.feature in ZONE_MAP).slice(0, 5);

    for (const f of visible) {
      const zone = ZONE_MAP[f.feature];
      if (!zone) continue;

      const [nx, ny, nw, nh] = zone;
      const px = nx * canvas.width;
      const py = ny * canvas.height;
      const pw = nw * canvas.width;
      const ph = nh * canvas.height;

      ctx.fillStyle   = FILL[f.level]   ?? "rgba(148,163,184,0.10)";
      ctx.strokeStyle = STROKE[f.level] ?? "#94a3b8aa";
      ctx.lineWidth   = 1.5;

      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(px, py, pw, ph, 8);
      } else {
        ctx.rect(px, py, pw, ph);
      }
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = LABEL[f.level] ?? "#94a3b8";
      ctx.font      = 'bold 8px "SF Mono", "Fira Mono", monospace';
      ctx.fillText(f.feature.replace(/_/g, " "), px + 5, py + 13);
    }
  }, [xai, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 pointer-events-none"
      aria-hidden="true"
    />
  );
}
