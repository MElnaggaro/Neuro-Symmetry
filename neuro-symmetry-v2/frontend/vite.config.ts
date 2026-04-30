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
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react":   ["react", "react-dom"],
          "vendor-motion":  ["framer-motion"],
          "vendor-charts":  ["recharts"],
          "vendor-lucide":  ["lucide-react"],
        },
      },
    },
  },
  server: {
    host: true,
    port: 3000,
    strictPort: true,
    proxy: {
      "/api": {
        target:      process.env.VITE_BACKEND_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite:     (p) => p.replace(/^\/api/, ""),
      },
      "/ws": {
        target:       process.env.VITE_BACKEND_URL || "http://127.0.0.1:8000",
        ws:           true,
        changeOrigin: true,
      },
    },
  },
});
