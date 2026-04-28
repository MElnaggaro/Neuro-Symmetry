import { forwardRef } from "react";

interface FaceMeshOverlayProps {
  width:  number;
  height: number;
}

/**
 * Transparent canvas that sits on top of the camera feed.
 * Drawing is performed imperatively by the useCamera hook,
 * keeping React out of the 30fps render loop.
 */
const FaceMeshOverlay = forwardRef<HTMLCanvasElement, FaceMeshOverlayProps>(
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

export default FaceMeshOverlay;
