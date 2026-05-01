"use strict";

/* ── MediaPipe CDN ─────────────────────────────────────────────────────────── */
const MP_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619";

/* ── localStorage keys ─────────────────────────────────────────────────────── */
const NS_KEYS = {
  AFFECTED_SIDE: "ns_affected_side",
  PRODROMAL:     "ns_prodromal",
  GAME:          "ns_game",
  ASSESSMENTS:   "ns_assessments",
};

/* ── Web Audio beep helper ─────────────────────────────────────────────────── */
let _audioCtx = null;
function beep(freq = 880, dur = 0.08, vol = 0.25) {
  try {
    if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = _audioCtx.createOscillator();
    const gain = _audioCtx.createGain();
    osc.connect(gain); gain.connect(_audioCtx.destination);
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(vol, _audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, _audioCtx.currentTime + dur);
    osc.start(); osc.stop(_audioCtx.currentTime + dur);
  } catch (_) {}
}

/* ── Head-pose-normalised landmark transform ───────────────────────────────── *
   Shared by mirror.html and game.html.
   Translates all 468 landmarks to nose-tip origin (lm[1]), rotates to level the
   inter-eye axis, and scales by IPD so the result is pose-invariant.
   Returns null when the face is too small (ipd < 1 px in image-space).         */
function normalizeRawLandmarks(L, videoEl) {
  const sx   = Math.max(videoEl?.videoWidth  || 1, 1);
  const sy   = Math.max(videoEl?.videoHeight || 1, 1);
  const nose = L[1];
  const le   = L[33], re = L[263];
  const dxEye = re.x * sx - le.x * sx;
  const dyEye = re.y * sy - le.y * sy;
  const ipd   = Math.max(Math.hypot(dxEye, dyEye), 1e-6);
  if (ipd < 0.01) return null;
  const roll = Math.atan2(dyEye, dxEye);
  const cos  = Math.cos(roll);
  const sin  = Math.sin(roll);
  return L.map((p) => {
    const dx = p.x * sx - nose.x * sx;
    const dy = p.y * sy - nose.y * sy;
    return {
      x: ( dx * cos + dy * sin) / ipd,
      y: (-dx * sin + dy * cos) / ipd,
      z: (((p.z || 0) - (nose.z || 0)) * sx) / ipd,
    };
  });
}

/* ── Tilt-corrected landmark math ──────────────────────────────────────────── *
   Shared by sentinel.html and game.html.
   These operate on raw fractional [0,1] MediaPipe coordinates (no pixel
   conversion) — indices 33 = R_EYE_OUTER, 263 = L_EYE_OUTER are constants.   */

/** Inter-canthal distance (fractional units) between outer eye corners. */
function getICD(L) {
  const dx = L[263].x - L[33].x;
  const dy = L[263].y - L[33].y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Pre-compute cos/sin of the roll correction angle. */
function getTiltCS(L) {
  const dx = L[263].x - L[33].x;
  const dy = L[263].y - L[33].y;
  const t  = Math.atan2(dy, dx);
  return { cosT: Math.cos(-t), sinT: Math.sin(-t) };
}

/** Tilt-corrected vertical distance (signed, ICD-normalised). */
function normVert(ax, ay, bx, by, icdVal, cosT, sinT) {
  const rx = bx - ax, ry = by - ay;
  return (rx * sinT + ry * cosT) / icdVal;
}

/** Tilt-corrected horizontal distance (signed, ICD-normalised). */
function normHoriz(ax, ay, bx, by, icdVal, cosT, sinT) {
  const rx = bx - ax, ry = by - ay;
  return (rx * cosT - ry * sinT) / icdVal;
}

/* ── FaceMesh factory ──────────────────────────────────────────────────────── *
   DRY wrapper — no page has to repeat the locateFile lambda.
   opts overrides any of the default setOptions values.                          */
function createFaceMesh(opts) {
  const fm = new FaceMesh({ locateFile: (f) => `${MP_CDN}/${f}` });
  fm.setOptions({
    maxNumFaces:            1,
    refineLandmarks:        false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence:  0.5,
    ...opts,
  });
  return fm;
}

/* ── Side selector wiring ──────────────────────────────────────────────────── *
   selector  : CSS selector matching the side buttons (.side-btn or .side-pill)
   onSelect(side) : callback receiving the chosen data-side value
   Returns setActive(side) for programmatic activation (e.g. from localStorage). */
function wireSideSelector(selector, onSelect) {
  const btns = [...document.querySelectorAll(selector)];
  btns.forEach((btn) => {
    btn.addEventListener("click", () => {
      btns.forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-pressed", "false"); });
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");
      onSelect(btn.dataset.side);
    });
  });
  return function setActive(side) {
    btns.forEach((b) => {
      const a = b.dataset.side === side;
      b.classList.toggle("active", a);
      b.setAttribute("aria-pressed", a ? "true" : "false");
    });
  };
}
