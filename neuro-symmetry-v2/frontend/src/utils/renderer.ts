import { HEATMAP_CONFIG } from "@/config/palette";
import type { XAIFeature, XAILevel } from "@/types/analysis";

const { ZONE_MAP, FILL, STROKE, LABEL } = HEATMAP_CONFIG;

// ── Canvas helpers ────────────────────────────────────────────────────────────

export function clearCanvas(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  ctx.clearRect(0, 0, w, h);
}

// ── Biometric HUD overlay ─────────────────────────────────────────────────────
//
// Sci-fi biometric scan rendered directly on the FaceMeshOverlay canvas at
// ~30 FPS via the subscribe() callback. Layers (back → front):
//   1. Micro-dots    — all 468 landmarks, very transparent
//   2. Contour lines — face oval, eyes, brows, lips, nose ridge
//   3. Key-point glows — anatomical anchors with pulsing outer rings
//   4. Eye ellipses  — steady outline + animated expanding ring
//   5. Scan sweep    — animated horizontal gradient line top→bottom
//   6. Corner brackets — HUD bounding-box corners
//   7. Nose crosshair  — ± 8px orthogonal tick marks
//   8. Symmetry axis   — dashed midline through face
//   9. HUD text        — "BIOMETRIC SCAN", blink "ANALYZING", scan %

interface _LM { x: number; y: number; z?: number }

// MediaPipe 468-point landmark group indices
const _FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
const _LEFT_EYE   = [33,7,163,144,145,153,154,155,133,246,161,160,159,158,157,173];
const _RIGHT_EYE  = [263,249,390,373,374,380,381,382,362,466,388,387,386,385,384,398];
const _LEFT_BROW  = [276,283,282,295,285,300,293,334,296,336];
const _RIGHT_BROW = [46,53,52,65,55,70,63,105,66,107];
const _LIPS       = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
const _NOSE       = [168,6,197,195,5,4,1,45,220,115,48,64,98,97,2,326,327,358,440,275,278];
const _KEY_PTS    = [1,4,33,133,263,362,61,291,70,300,10,152,168,197];

// Mirror X so drawing aligns with the CSS-flipped video (scaleX(-1))
function _px(lm: _LM, w: number, h: number) {
  return { x: (1 - lm.x) * w, y: lm.y * h };
}

function _contour(
  ctx:   CanvasRenderingContext2D,
  lms:   _LM[],
  idx:   number[],
  w:     number,
  h:     number,
  close: boolean,
) {
  const valid = idx.filter((i) => i < lms.length);
  if (valid.length < 2) return;
  const p0 = _px(lms[valid[0]], w, h);
  ctx.beginPath();
  ctx.moveTo(p0.x, p0.y);
  for (let i = 1; i < valid.length; i++) {
    const p = _px(lms[valid[i]], w, h);
    ctx.lineTo(p.x, p.y);
  }
  if (close) ctx.closePath();
  ctx.stroke();
}

function _bounds(lms: _LM[], idx: number[], w: number, h: number) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const i of idx) {
    if (i >= lms.length) continue;
    const p = _px(lms[i], w, h);
    if (p.x < x0) x0 = p.x;
    if (p.x > x1) x1 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.y > y1) y1 = p.y;
  }
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, rx: (x1 - x0) / 2, ry: (y1 - y0) / 2, x0, y0, x1, y1 };
}

/**
 * Renders the full sci-fi biometric HUD onto the overlay canvas.
 * Call this every frame from the FaceTrackingProvider subscribe() callback.
 * All draws are mirrored in X to match the CSS scaleX(-1) video transform.
 */
export function drawBiometricOverlay(
  ctx:       CanvasRenderingContext2D,
  landmarks: _LM[],
  w:         number,
  h:         number,
): void {
  if (!landmarks || landmarks.length < 100) return;

  const t   = performance.now();

  // ── 1. Micro-dots — all 468 landmarks ─────────────────────────────────
  ctx.fillStyle = "rgba(34,211,238,0.13)";
  ctx.beginPath();
  for (const lm of landmarks) {
    const { x, y } = _px(lm, w, h);
    ctx.moveTo(x + 1, y);          // moveTo avoids connecting lines between arcs
    ctx.arc(x, y, 1, 0, Math.PI * 2);
  }
  ctx.fill();

  // ── 2. Structural contour lines ────────────────────────────────────────
  ctx.save();
  ctx.shadowColor = "#22d3ee";
  ctx.shadowBlur  = 5;

  ctx.strokeStyle = "rgba(34,211,238,0.28)";
  ctx.lineWidth   = 0.7;
  _contour(ctx, landmarks, _FACE_OVAL,  w, h, true);

  ctx.strokeStyle = "rgba(34,211,238,0.42)";
  ctx.lineWidth   = 0.85;
  _contour(ctx, landmarks, _LEFT_EYE,   w, h, true);
  _contour(ctx, landmarks, _RIGHT_EYE,  w, h, true);

  ctx.strokeStyle = "rgba(34,211,238,0.32)";
  ctx.lineWidth   = 0.7;
  _contour(ctx, landmarks, _LEFT_BROW,  w, h, false);
  _contour(ctx, landmarks, _RIGHT_BROW, w, h, false);
  _contour(ctx, landmarks, _LIPS,       w, h, true);
  _contour(ctx, landmarks, _NOSE,       w, h, false);

  ctx.restore();

  // ── 3. Key anatomical glow dots + outer rings ──────────────────────────
  const pulse = 0.7 + 0.3 * Math.sin(t * 0.004);

  ctx.save();
  ctx.shadowBlur  = 14;
  ctx.shadowColor = "#22d3ee";
  ctx.fillStyle   = `rgba(34,211,238,${(0.78 * pulse).toFixed(2)})`;
  ctx.beginPath();
  for (const i of _KEY_PTS) {
    if (i >= landmarks.length) continue;
    const { x, y } = _px(landmarks[i], w, h);
    ctx.moveTo(x + 3, y);
    ctx.arc(x, y, 3, 0, Math.PI * 2);
  }
  ctx.fill();

  ctx.strokeStyle = `rgba(34,211,238,${(0.35 * pulse).toFixed(2)})`;
  ctx.lineWidth   = 0.9;
  for (const i of _KEY_PTS) {
    if (i >= landmarks.length) continue;
    const { x, y } = _px(landmarks[i], w, h);
    ctx.beginPath();
    ctx.arc(x, y, 6.5, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();

  // ── 4. Eye ellipses — steady + animated expansion ring ─────────────────
  for (const grp of [_LEFT_EYE, _RIGHT_EYE]) {
    const b  = _bounds(landmarks, grp, w, h);
    if (b.rx < 3) continue;
    const rx = b.rx * 1.3, ry = b.ry * 1.55;

    ctx.save();
    ctx.shadowBlur  = 10;
    ctx.shadowColor = "#06b6d4";
    ctx.strokeStyle = "rgba(34,211,238,0.55)";
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.ellipse(b.cx, b.cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    // expanding pulse ring — loops every 2 s
    const phase  = (t * 0.0005) % 1;
    const pScale = 1 + phase * 0.55;
    const pAlpha = (1 - phase) * 0.42;
    ctx.strokeStyle = `rgba(34,211,238,${pAlpha.toFixed(2)})`;
    ctx.lineWidth   = 0.7;
    ctx.beginPath();
    ctx.ellipse(b.cx, b.cy, rx * pScale, ry * pScale, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // Face bounding box (used by scan sweep + brackets + text)
  const fb   = _bounds(landmarks, _FACE_OVAL, w, h);
  const padX = fb.rx * 0.15, padY = fb.ry * 0.08;
  const fl = fb.x0 - padX, fr = fb.x1 + padX;
  const ft = fb.y0 - padY, fb_ = fb.y1 + padY;   // fb_ = face bottom

  // ── 5. Horizontal scan sweep ───────────────────────────────────────────
  const SCAN_MS  = 2800;
  const scanFrac = (t % SCAN_MS) / SCAN_MS;
  const scanY    = ft + (fb_ - ft) * scanFrac;

  ctx.save();
  const sg = ctx.createLinearGradient(fl, 0, fr, 0);
  sg.addColorStop(0,   "rgba(34,211,238,0)");
  sg.addColorStop(0.2, "rgba(34,211,238,0.06)");
  sg.addColorStop(0.5, "rgba(34,211,238,0.22)");
  sg.addColorStop(0.8, "rgba(34,211,238,0.06)");
  sg.addColorStop(1,   "rgba(34,211,238,0)");
  ctx.fillStyle = sg;
  ctx.fillRect(fl, scanY - 3, fr - fl, 6);

  ctx.shadowBlur  = 5;
  ctx.shadowColor = "#22d3ee";
  ctx.strokeStyle = `rgba(34,211,238,${(0.42 * (1 - scanFrac * 0.4)).toFixed(2)})`;
  ctx.lineWidth   = 0.6;
  ctx.beginPath(); ctx.moveTo(fl, scanY); ctx.lineTo(fr, scanY); ctx.stroke();
  ctx.restore();

  // ── 6. Corner brackets ─────────────────────────────────────────────────
  const arm = Math.min(fr - fl, fb_ - ft) * 0.10;

  ctx.save();
  ctx.shadowBlur  = 10;
  ctx.shadowColor = "#22d3ee";
  ctx.strokeStyle = "rgba(34,211,238,0.82)";
  ctx.lineWidth   = 1.5;
  ctx.lineCap     = "square";

  for (const [bx, dx] of [[fl, 1], [fr, -1]] as [number, number][]) {
    for (const [by, dy] of [[ft, 1], [fb_, -1]] as [number, number][]) {
      ctx.beginPath();
      ctx.moveTo(bx + dx * arm, by);
      ctx.lineTo(bx, by);
      ctx.lineTo(bx, by + dy * arm);
      ctx.stroke();
    }
  }
  ctx.restore();

  // ── 7. Nose-tip crosshair ─────────────────────────────────────────────
  if (landmarks[1]) {
    const { x: nx, y: ny } = _px(landmarks[1], w, h);
    const cs = 8;
    ctx.save();
    ctx.shadowBlur  = 8;
    ctx.shadowColor = "#22d3ee";
    ctx.strokeStyle = "rgba(34,211,238,0.75)";
    ctx.lineWidth   = 1;
    ctx.lineCap     = "round";
    ctx.beginPath(); ctx.moveTo(nx - cs, ny); ctx.lineTo(nx + cs, ny); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(nx, ny - cs); ctx.lineTo(nx, ny + cs); ctx.stroke();
    ctx.fillStyle = "#22d3ee";
    ctx.beginPath(); ctx.arc(nx, ny, 2.5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  // ── 8. Vertical symmetry axis ─────────────────────────────────────────
  if (landmarks[168] && landmarks[152]) {
    const top = _px(landmarks[168], w, h);
    const bot = _px(landmarks[152], w, h);
    const mx  = (top.x + bot.x) / 2;
    ctx.save();
    ctx.setLineDash([4, 7]);
    ctx.shadowBlur  = 3;
    ctx.shadowColor = "rgba(34,211,238,0.2)";
    ctx.strokeStyle = "rgba(34,211,238,0.17)";
    ctx.lineWidth   = 0.7;
    ctx.beginPath(); ctx.moveTo(mx, ft - 8); ctx.lineTo(mx, fb_ + 8); ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // ── 9. HUD text readouts ──────────────────────────────────────────────
  const mono  = '"SF Mono","Fira Code","Courier New",monospace';
  const blink = Math.sin(t * 0.0032) > 0;

  ctx.save();
  ctx.shadowBlur  = 8;
  ctx.shadowColor = "#22d3ee";

  ctx.font      = `bold 9px ${mono}`;
  ctx.fillStyle = "rgba(34,211,238,0.82)";
  ctx.fillText("BIOMETRIC SCAN", fl + 2, ft - 5);

  if (blink) {
    ctx.fillStyle = "rgba(34,211,238,0.65)";
    ctx.font      = `bold 8px ${mono}`;
    ctx.fillText("▶ ANALYZING", fr - 70, ft - 5);
  }

  const pct = Math.floor(scanFrac * 100).toString().padStart(3, " ");
  ctx.fillStyle = "rgba(34,211,238,0.50)";
  ctx.font      = `8px ${mono}`;
  ctx.fillText(pct + "%", fr - 24, fb_ + 11);

  ctx.restore();
}

// ── Heatmap renderer ──────────────────────────────────────────────────────────

/**
 * Renders semi-transparent zone overlays for the top-5 XAI features
 * that have a registered ZONE_MAP entry.
 */
export function drawHeatmap(
  ctx:         CanvasRenderingContext2D,
  xaiFeatures: XAIFeature[],
  w:           number,
  h:           number,
): void {
  if (!xaiFeatures.length) return;

  const visible = xaiFeatures.filter((f) => f.feature in ZONE_MAP).slice(0, 5);

  ctx.font = 'bold 8px "SF Mono", "Fira Mono", monospace';

  for (const { feature, level } of visible) {
    const [nx, ny, nw, nh] = ZONE_MAP[feature as keyof typeof ZONE_MAP] || [0, 0, 0, 0];
    const x = nx * w, y = ny * h, fw = nw * w, fh = nh * h;
    const lvl = level as XAILevel;

    ctx.fillStyle   = FILL[lvl]   ?? FILL.DEFAULT;
    ctx.strokeStyle = STROKE[lvl] ?? STROKE.DEFAULT;
    ctx.lineWidth   = 1.5;

    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, y, fw, fh, 8);
    } else {
      ctx.rect(x, y, fw, fh);
    }
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = LABEL[lvl] ?? LABEL.DEFAULT;
    ctx.fillText(feature.replace(/_/g, " "), x + 5, y + 13);
  }
}