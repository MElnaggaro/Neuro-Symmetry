import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

const SEND_INTERVAL_MS = 150; // throttle to ~6-7 fps toward backend

interface UseCameraOptions {
  /** Forwarded canvas ref for the 468-point landmark overlay. */
  meshCanvasRef: RefObject<HTMLCanvasElement | null>;
  /** Called with a base-64 JPEG string whenever a frame is captured. */
  onFrame: (b64: string) => void;
  width?:  number;
  height?: number;
}

export interface UseCameraReturn {
  videoRef: RefObject<HTMLVideoElement>;
  error:    string | null;
  isReady:  boolean;
}

export function useCamera({
  meshCanvasRef,
  onFrame,
  width  = 640,
  height = 480,
}: UseCameraOptions): UseCameraReturn {
  // useRef<T>(null) → RefObject<T> (current: T | null) — compatible with JSX ref prop
  const videoRef    = useRef<HTMLVideoElement>(null);
  const captureRef  = useRef<HTMLCanvasElement | null>(null);
  const lastSendRef = useRef<number>(0);
  // Use a stable ref for onFrame so the effect never re-runs on callback identity change.
  const onFrameRef  = useRef<(b64: string) => void>(onFrame);
  onFrameRef.current = onFrame;

  const [error,   setError]   = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    let cancelled = false;

    // Hidden canvas used only for JPEG capture — not rendered to DOM.
    const cap = document.createElement("canvas");
    cap.width  = width;
    cap.height = height;
    captureRef.current = cap;

    const setup = async () => {
      try {
        const [{ FaceMesh }, { Camera }] = await Promise.all([
          import("@mediapipe/face_mesh"),
          import("@mediapipe/camera_utils"),
        ]);
        if (cancelled) return;

        const faceMesh = new FaceMesh({
          locateFile: (f) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}`,
        });

        faceMesh.setOptions({
          maxNumFaces:            1,
          refineLandmarks:        true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence:  0.5,
        });

        faceMesh.onResults(({ multiFaceLandmarks }) => {
          if (cancelled) return;

          // ── Draw cyan landmark mesh on overlay canvas ──────────────────────
          const mc  = meshCanvasRef.current;
          const ctx = mc?.getContext("2d");
          if (mc && ctx) {
            ctx.clearRect(0, 0, mc.width, mc.height);
            if (multiFaceLandmarks?.length) {
              ctx.shadowColor = "#06b6d4";
              ctx.shadowBlur  = 2.5;
              ctx.fillStyle   = "rgba(6,182,212,0.62)";
              for (const pt of multiFaceLandmarks[0]) {
                ctx.beginPath();
                ctx.arc(pt.x * mc.width, pt.y * mc.height, 1.3, 0, Math.PI * 2);
                ctx.fill();
              }
              ctx.shadowBlur = 0;
            }
          }

          // ── Throttled JPEG capture → callback ──────────────────────────────
          const now = performance.now();
          const vid = videoRef.current;
          const cap = captureRef.current;
          if (now - lastSendRef.current >= SEND_INTERVAL_MS && vid && cap) {
            lastSendRef.current = now;
            const capCtx = cap.getContext("2d");
            if (capCtx) {
              capCtx.drawImage(vid, 0, 0, width, height);
              const b64 = cap.toDataURL("image/jpeg", 0.6).split(",")[1] ?? "";
              onFrameRef.current(b64);
            }
          }
        });

        if (!videoRef.current) {
          setError("Video element not mounted.");
          return;
        }

        const camera = new Camera(videoRef.current, {
          onFrame: () => faceMesh.send({ image: videoRef.current! }),
          width,
          height,
        });

        camera.start();
        if (!cancelled) setIsReady(true);

        // Cleanup captured in outer scope
        return () => { camera.stop(); faceMesh.close(); };
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof Error
              ? e.message
              : "Camera / MediaPipe initialization failed.",
          );
        }
      }
    };

    // Store cleanup fn returned from async setup
    let innerCleanup: (() => void) | undefined;
    void setup().then((fn) => { innerCleanup = fn; });

    return () => {
      cancelled = true;
      innerCleanup?.();
    };
  // onFrame excluded: kept stable via onFrameRef.current above.
  }, [meshCanvasRef, width, height]);

  return { videoRef, error, isReady };
}
