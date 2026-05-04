import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Redirect the legacy `/tracker.html` URL to the new in-SPA route. The static
// file is gone, so without this middleware bookmarks would 404. Runs in dev
// (vite serve) and preview (vite preview).
const trackerRedirect = {
  name: "tracker-html-redirect",
  configureServer(server: import("vite").ViteDevServer) {
    server.middlewares.use((req, res, next) => {
      if (req.url === "/tracker.html" || req.url?.startsWith("/tracker.html?")) {
        res.writeHead(301, { Location: "/#/tracker" });
        res.end();
        return;
      }
      next();
    });
  },
  configurePreviewServer(server: import("vite").PreviewServer) {
    server.middlewares.use((req, res, next) => {
      if (req.url === "/tracker.html" || req.url?.startsWith("/tracker.html?")) {
        res.writeHead(301, { Location: "/#/tracker" });
        res.end();
        return;
      }
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), trackerRedirect],
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
