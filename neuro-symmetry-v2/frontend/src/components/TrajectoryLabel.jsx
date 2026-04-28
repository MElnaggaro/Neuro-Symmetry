const CONFIG = {
  STABLE:         { color: "#10b981", dark: "#064e3b", label: "Stable",         icon: "→",  pulse: false },
  LINEAR_DECLINE: { color: "#f59e0b", dark: "#78350f", label: "Linear Decline", icon: "↘",  pulse: false },
  SUDDEN_DROP:    { color: "#ef4444", dark: "#450a0a", label: "Sudden Drop",    icon: "↓",  pulse: true  },
  COLLAPSE:       { color: "#dc2626", dark: "#3b0000", label: "Collapse",       icon: "⬇",  pulse: true  },
  OSCILLATING:    { color: "#a78bfa", dark: "#2e1065", label: "Oscillating",    icon: "↕",  pulse: false },
};

export default function TrajectoryLabel({ trajectory = "STABLE" }) {
  const cfg = CONFIG[trajectory] ?? CONFIG.STABLE;
  return (
    <div style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 7,
      padding: "6px 14px",
      borderRadius: 24,
      background: `linear-gradient(135deg, ${cfg.dark}, rgba(0,0,0,0.3))`,
      border: `1.5px solid ${cfg.color}50`,
      boxShadow: `0 0 12px ${cfg.color}30`,
      color: cfg.color,
      fontWeight: 700,
      fontSize: 12,
      letterSpacing: 0.8,
      backdropFilter: "blur(8px)",
      WebkitBackdropFilter: "blur(8px)",
      animation: cfg.pulse ? "criticalPulse 1.4s ease-in-out infinite" : "none",
      whiteSpace: "nowrap",
    }}>
      <span style={{ fontSize: 15, lineHeight: 1, fontWeight: 900 }}>{cfg.icon}</span>
      <span style={{ textTransform: "uppercase" }}>{cfg.label}</span>
    </div>
  );
}
