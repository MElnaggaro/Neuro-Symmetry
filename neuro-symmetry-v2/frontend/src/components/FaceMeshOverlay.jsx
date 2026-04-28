import { forwardRef } from "react";

// Canvas overlay drawn imperatively by parent (App.jsx) with MediaPipe landmark data.
const FaceMeshOverlay = forwardRef(function FaceMeshOverlay({ width = 640, height = 480 }, ref) {
  return (
    <canvas
      ref={ref}
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
});

export default FaceMeshOverlay;
