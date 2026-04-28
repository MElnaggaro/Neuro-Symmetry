import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          base:    "#040b16",
          surface: "#070f1e",
          panel:   "#0a1628",
        },
        neu: {
          border:      "#0f2040",
          borderLight: "#1a3a5c",
        },
        risk: {
          normal:   "#10b981",
          mild:     "#f59e0b",
          high:     "#f97316",
          critical: "#ef4444",
        },
        accent: {
          blue: "#3b82f6",
          cyan: "#06b6d4",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ['"SF Mono"', '"Fira Mono"', "monospace"],
      },
      keyframes: {
        criticalPulse: {
          "0%,100%": { opacity: "1" },
          "50%":     { opacity: "0.55" },
        },
        blinkDot: {
          "0%,100%": { opacity: "1" },
          "50%":     { opacity: "0.15" },
        },
        scanline: {
          "0%":   { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
      },
      animation: {
        critical: "criticalPulse 1.2s ease-in-out infinite",
        blink:    "blinkDot 1.5s ease-in-out infinite",
        scan:     "scanline 8s linear infinite",
      },
      boxShadow: {
        "glow-cyan":   "0 0 16px rgba(6,182,212,0.25)",
        "glow-green":  "0 0 20px rgba(16,185,129,0.2)",
        "glow-amber":  "0 0 20px rgba(245,158,11,0.25)",
        "glow-orange": "0 0 24px rgba(249,115,22,0.3)",
        "glow-red":    "0 0 32px rgba(239,68,68,0.45)",
        card:          "0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03)",
      },
    },
  },
  plugins: [],
};

export default config;
