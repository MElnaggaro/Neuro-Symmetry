import { useEffect, useRef } from "react";

// Normalised [x, y, w, h] zones for a frontal mirrored webcam view.
// x=0 = image-left (person's right side due to mirror).
const ZONE_MAP = {
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

const LEVEL_STYLE = {
  HIGH:   { fill: "rgba(239,68,68,0.20)",   stroke: "#ef4444cc", label: "#f87171" },
  MEDIUM: { fill: "rgba(249,115,22,0.15)",  stroke: "#f97316aa", label: "#fb923c" },
  LOW:    { fill: "rgba(16,185,129,0.10)",  stroke: "#10b98177", label: "#34d399" },
};

export default function HeatmapOverlay({ xai = [], width = 640, height = 480 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const topXai = xai.filter((f) => ZONE_MAP[f.feature]).slice(0, 5);

    for (const f of topXai) {
      const zone = ZONE_MAP[f.feature];
      if (!zone) continue;
      const st = LEVEL_STYLE[f.level] ?? LEVEL_STYLE.LOW;
      const px = zone[0] * canvas.width;
      const py = zone[1] * canvas.height;
      const pw = zone[2] * canvas.width;
      const ph = zone[3] * canvas.height;

      ctx.fillStyle   = st.fill;
      ctx.strokeStyle = st.stroke;
      ctx.lineWidth   = 1.5;

      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(px, py, pw, ph, 8);
      } else {
        ctx.rect(px, py, pw, ph);
      }
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = st.label;
      ctx.font      = "bold 8px 'SF Mono', 'Fira Mono', monospace";
      ctx.fillText(f.feature.replace(/_/g, " "), px + 5, py + 13);
    }
  }, [xai, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        pointerEvents: "none",
      }}
    />
  );
}
