import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ["@mediapipe/face_mesh", "@mediapipe/camera_utils"],
  },
  server: {
    proxy: {
      "/api": {
        target:    "http://localhost:8000",
        changeOrigin: true,
        rewrite:   (path) => path.replace(/^\/api/, ""),
      },
      "/ws": {
        target:    "ws://localhost:8000",
        ws:        true,
        changeOrigin: true,
      },
    },
  },
});
