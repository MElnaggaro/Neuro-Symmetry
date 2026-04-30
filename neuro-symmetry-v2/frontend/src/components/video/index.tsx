import { forwardRef, useEffect, useRef } from "react";
import type { XAIFeature } from "@/types/analysis";
import { clearCanvas, drawHeatmap } from "@/utils/renderer";

// ── FaceMeshOverlay ───────────────────────────────────────────────────────────

interface OverlayDimensions { width: number; height: number }

/**
 * Transparent canvas that sits above the camera feed.
 * Drawing is performed imperatively by the subscribe() callback, keeping React
 * out of the 30fps render loop.
 */
export const FaceMeshOverlay = forwardRef<HTMLCanvasElement, OverlayDimensions>(
  function FaceMeshOverlay({ width, height }, ref) {
    return (
      <canvas
        ref={ref}
        width={width}
        height={height}
        className="absolute inset-0 pointer-events-none"
        aria-hidden="true"
      />
    );
  },
);

// ── HeatmapOverlay ────────────────────────────────────────────────────────────

interface HeatmapOverlayProps extends OverlayDimensions { xai: XAIFeature[] }

export function HeatmapOverlay({ xai, width, height }: HeatmapOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    clearCanvas(ctx, canvas.width, canvas.height);
    drawHeatmap(ctx, xai, canvas.width, canvas.height);
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