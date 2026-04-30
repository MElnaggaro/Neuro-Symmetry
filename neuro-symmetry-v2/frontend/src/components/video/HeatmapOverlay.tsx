import { useEffect, useRef } from "react";
import type { XAIFeature } from "@/types/analysis";
import { drawHeatmap, clearCanvas } from "@/utils/renderer";

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
