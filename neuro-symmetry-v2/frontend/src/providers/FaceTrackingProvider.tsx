/**
 * FaceTrackingProvider — singleton camera + MediaPipe FaceMesh pipeline.
 *
 * ARCHITECTURE (Phase 2 — Mutable Refs Pattern)
 * ──────────────────────────────────────────────
 * Root cause of the original render thrashing: storing 468 MediaPipe landmarks
 * in React `useState` at 30 FPS → 30 full reconciliation cycles per second.
 *
 * Solution:
 *   - HIGH-FREQUENCY data (rawScreenLandmarks, mathLandmarks, features50,
 *     metrics) is stored in `useRef`. Writing to a ref NEVER triggers a render.
 *   - LOW-FREQUENCY data (status, error, trackingQuality) uses `useState`
 *     with a 200ms throttle (~5 FPS) for trackingQuality.
 *   - Consumers that need per-frame data use `subscribe()` or read from refs.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import * as FaceMeshLib from "@mediapipe/face_mesh";
import type {
  FaceMesh as FaceMeshType,
  FaceMeshResults,
  NormalizedLandmarkList,
} from "@mediapipe/face_mesh";
import { MEDIAPIPE_CONFIG, VIDEO_CONFIG } from "@/config";
import logger from "@/utils/logger";
import {
  type FeatureVector50,
  normalizeFaceGeometry,
} from "@/utils/faceMath";

// ── MediaPipe constructor resolution ──────────────────────────────────────────

const FaceMeshCtor =
  (FaceMeshLib as Record<string, unknown>).FaceMesh ??
  (window as unknown as Record<string, unknown>).FaceMesh;

const MP_CDN =
  "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619";

// ── Landmark indices for metrics ──────────────────────────────────────────────

const LM = {
  R_EYE_TOP: 159,
  R_EYE_BOT: 145,
  L_EYE_TOP: 386,
  L_EYE_BOT: 374,
  R_BROW_OUTER: 70,
  L_BROW_OUTER: 300,
  MOUTH_L: 61,
  MOUTH_R: 291,
  CHEEK_L: 234,
  CHEEK_R: 454,
} as const;

// ── Public types ──────────────────────────────────────────────────────────────

export interface FacialMetrics {
  ipd: number;
  rollRad: number;
  rEyeOpen: number;
  lEyeOpen: number;
  rBrowLift: number;
  lBrowLift: number;
  smileWidth: number;
  cheekWidth: number;
}

export type TrackingStatus = "idle" | "loading" | "ready" | "error";
export type QualityCode = "OK" | "DEGRADED" | "UNRELIABLE";

/**
 * Frame callback signature.
 * Invoked imperatively from the MediaPipe onResults handler — NOT from React state.
 */
export type FrameCallback = (
  rawLandmarks: NormalizedLandmarkList,
  normalizedLandmarks: NormalizedLandmarkList,
  metrics: FacialMetrics,
  features: FeatureVector50,
) => void;

export interface FaceTrackingContextValue {
  // ── Low-frequency React state (renders on change) ─────────────────────
  status: TrackingStatus;
  error: string | null;
  trackingQuality: QualityCode;

  // ── DOM refs ──────────────────────────────────────────────────────────
  videoRef: React.RefObject<HTMLVideoElement>;
  meshRef: React.RefObject<HTMLCanvasElement>;

  // ── High-frequency mutable refs (NEVER trigger renders) ───────────────
  rawScreenLandmarksRef: React.RefObject<NormalizedLandmarkList | null>;
  mathLandmarksRef: React.RefObject<NormalizedLandmarkList | null>;
  features50Ref: React.RefObject<FeatureVector50 | null>;
  metricsRef: React.RefObject<FacialMetrics | null>;

  // ── Controls ──────────────────────────────────────────────────────────
  start: () => void;
  stop: () => void;
  subscribe: (cb: FrameCallback) => () => void;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function absDelta(a: number, b: number): number {
  return Math.abs(a - b);
}

function computeMetrics(
  normalizedLandmarks: NormalizedLandmarkList,
  ipd: number,
  rollRad: number,
): FacialMetrics {
  const n = normalizedLandmarks;
  return {
    ipd,
    rollRad,
    rEyeOpen: absDelta(n[LM.R_EYE_TOP].y, n[LM.R_EYE_BOT].y),
    lEyeOpen: absDelta(n[LM.L_EYE_TOP].y, n[LM.L_EYE_BOT].y),
    rBrowLift: n[LM.R_EYE_TOP].y - n[LM.R_BROW_OUTER].y,
    lBrowLift: n[LM.L_EYE_TOP].y - n[LM.L_BROW_OUTER].y,
    smileWidth: absDelta(n[LM.MOUTH_L].x, n[LM.MOUTH_R].x),
    cheekWidth: absDelta(n[LM.CHEEK_L].x, n[LM.CHEEK_R].x),
  };
}

function waitForVideoReady(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Timed out waiting for camera metadata."));
    }, 8_000);

    const onReady = () => {
      cleanup();
      resolve();
    };

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("canplay", onReady);
    };

    video.addEventListener("loadedmetadata", onReady, { once: true });
    video.addEventListener("canplay", onReady, { once: true });
  });
}

// ── Throttle constant for trackingQuality updates ─────────────────────────────
const QUALITY_THROTTLE_MS = 200; // ~5 FPS max for UI quality badge

// ── Quality thresholds ────────────────────────────────────────────────────────
const IPD_DEGRADED_THRESHOLD = 30;  // pixels — face is too far from camera
const ROLL_DEGRADED_THRESHOLD = 0.5; // radians (~28°) — head tilted too much

// ── Context ───────────────────────────────────────────────────────────────────

const FaceTrackingContext = createContext<FaceTrackingContextValue | null>(null);

export function useFaceTracking(): FaceTrackingContextValue {
  const ctx = useContext(FaceTrackingContext);
  if (!ctx) throw new Error("useFaceTracking must be called inside <FaceTrackingProvider>");
  return ctx;
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function FaceTrackingProvider({ children }: { children: React.ReactNode }) {
  const videoRef = useRef<HTMLVideoElement>(null!);
  const meshRef = useRef<HTMLCanvasElement>(null!);

  // ── Low-frequency React state (OK to trigger renders) ─────────────────
  const [status, setStatus] = useState<TrackingStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [trackingQuality, setTrackingQuality] = useState<QualityCode>("UNRELIABLE");

  // ── High-frequency mutable refs (NEVER trigger renders) ───────────────
  const rawScreenLandmarksRef = useRef<NormalizedLandmarkList | null>(null);
  const mathLandmarksRef = useRef<NormalizedLandmarkList | null>(null);
  const features50Ref = useRef<FeatureVector50 | null>(null);
  const metricsRef = useRef<FacialMetrics | null>(null);

  // ── Internal refs ─────────────────────────────────────────────────────
  const subscribers = useRef<Set<FrameCallback>>(new Set());
  const faceMeshRef = useRef<FaceMeshType | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const startedRef = useRef(false);
  const processingRef = useRef(false);
  const lastQualityUpdateRef = useRef(0);

  // ── onResults — the core frame handler ────────────────────────────────
  // NOTE: Drawing is NOT done here — consumers handle it via subscribe().
  // This avoids double-drawing when multiple subscribers draw landmarks.
  const handleResults = useCallback((results: FaceMeshResults) => {
    const raw = results.multiFaceLandmarks?.[0] ?? null;

    if (!raw) {
      // No face — clear all refs
      rawScreenLandmarksRef.current = null;
      mathLandmarksRef.current = null;
      features50Ref.current = null;
      metricsRef.current = null;

      // Throttled quality update
      const now = performance.now();
      if (now - lastQualityUpdateRef.current > QUALITY_THROTTLE_MS) {
        lastQualityUpdateRef.current = now;
        setTrackingQuality("UNRELIABLE");
      }
      return;
    }

    try {
      const video = videoRef.current;

      // Phase 1 math: Translation + Roll Rotation, NO IPD scaling
      const geometry = normalizeFaceGeometry(raw, {
        width: video?.videoWidth || VIDEO_CONFIG.WIDTH,
        height: video?.videoHeight || VIDEO_CONFIG.HEIGHT,
      });

      const nextMetrics = computeMetrics(
        geometry.landmarks,
        geometry.ipd,
        geometry.rollRad,
      );

      // ── Write to mutable refs (zero renders) ─────────────────────────
      rawScreenLandmarksRef.current = raw;
      mathLandmarksRef.current = geometry.landmarks;
      features50Ref.current = geometry.features;
      metricsRef.current = nextMetrics;

      // ── Throttled quality state for UI badge (~5 FPS) ─────────────────
      // DEGRADED when face is too far (small IPD) or head tilted too much
      const now = performance.now();
      if (now - lastQualityUpdateRef.current > QUALITY_THROTTLE_MS) {
        lastQualityUpdateRef.current = now;
        const degraded =
          geometry.ipd < IPD_DEGRADED_THRESHOLD ||
          Math.abs(geometry.rollRad) > ROLL_DEGRADED_THRESHOLD;
        setTrackingQuality(degraded ? "DEGRADED" : "OK");
      }

      // ── Notify subscribers (imperative — no React state) ──────────────
      for (const cb of subscribers.current) {
        try {
          cb(raw, geometry.landmarks, nextMetrics, geometry.features);
        } catch (e) {
          logger.error("FaceTrackingProvider", "Subscriber threw", { error: e });
        }
      }
    } catch (e) {
      logger.warn("FaceTrackingProvider", "Landmark normalization skipped", {
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }, []);

  // ── stop ──────────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    faceMeshRef.current?.close();
    faceMeshRef.current = null;
    startedRef.current = false;
    processingRef.current = false;
    setStatus("idle");
  }, []);

  // ── start ─────────────────────────────────────────────────────────────
  const start = useCallback(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    setStatus("loading");
    setError(null);

    (async () => {
      try {
        const video = videoRef.current;
        if (!video) throw new Error("Video element is not mounted yet.");

        const faceMesh: FaceMeshType = new (FaceMeshCtor as new (config: Record<string, unknown>) => FaceMeshType)({
          locateFile: (file: string) => `${MP_CDN}/${file}`,
        });
        faceMesh.setOptions(MEDIAPIPE_CONFIG);
        faceMesh.onResults(handleResults);
        faceMeshRef.current = faceMesh;

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: VIDEO_CONFIG.WIDTH },
            height: { ideal: VIDEO_CONFIG.HEIGHT },
            facingMode: "user",
          },
          audio: false,
        });
        streamRef.current = stream;
        stream.getVideoTracks().forEach((track) => {
          track.addEventListener("ended", () => {
            setError("Camera disconnected. Reconnect your camera and refresh the page.");
            setStatus("error");
            stop();
            startedRef.current = false;
          });
        });
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await waitForVideoReady(video);
        await video.play();

        // ── RAF tick — send frames to FaceMesh ────────────────────────
        const tick = async () => {
          const fm = faceMeshRef.current;
          const vid = videoRef.current;
          if (!fm || !vid || !startedRef.current) return;

          if (
            !processingRef.current &&
            vid.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
          ) {
            processingRef.current = true;
            try {
              await fm.send({ image: vid });
            } finally {
              processingRef.current = false;
            }
          }
          rafRef.current = requestAnimationFrame(tick);
        };

        rafRef.current = requestAnimationFrame(tick);
        setStatus("ready");
        logger.info("FaceTrackingProvider", "Pipeline ready");
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Unknown camera error";
        setError(msg);
        setStatus("error");
        stop();
        startedRef.current = false;
        logger.error("FaceTrackingProvider", "Init failed", { error: msg });
      }
    })();
  }, [handleResults, stop]);

  // ── subscribe ─────────────────────────────────────────────────────────
  const subscribe = useCallback((cb: FrameCallback): (() => void) => {
    subscribers.current.add(cb);
    return () => { subscribers.current.delete(cb); };
  }, []);

  // ── Cleanup on unmount ────────────────────────────────────────────────
  useEffect(() => stop, [stop]);

  // ── Context value ─────────────────────────────────────────────────────
  const value: FaceTrackingContextValue = {
    status,
    error,
    trackingQuality,
    videoRef,
    meshRef,
    rawScreenLandmarksRef,
    mathLandmarksRef,
    features50Ref,
    metricsRef,
    start,
    stop,
    subscribe,
  };

  return (
    <FaceTrackingContext.Provider value={value}>
      {children}
    </FaceTrackingContext.Provider>
  );
}
