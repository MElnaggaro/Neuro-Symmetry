import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  optimizeDeps: {
    exclude: ["@mediapipe/face_mesh", "@mediapipe/camera_utils"],
  },
  server: {
    proxy: {
      "/api": {
        target:      "http://localhost:8000",
        changeOrigin: true,
        rewrite:     (p) => p.replace(/^\/api/, ""),
      },
      "/ws": {
        target:       "ws://localhost:8000",
        ws:           true,
        changeOrigin: true,
      },
    },
  },
});
