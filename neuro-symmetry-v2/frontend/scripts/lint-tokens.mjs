#!/usr/bin/env node
/**
 * Verify that frontend/public/shared.css design tokens stay in sync with the
 * React SPA's src/config/palette.ts. The satellite stylesheet is hand-mirrored
 * (no build step) — this lint guards against silent drift like tracker.html
 * once forking the palette to purple.
 *
 * Checks:
 *   1. shared.css carries the SHARED_CSS_TOKEN_HASH=v2:dark+cyan marker.
 *   2. The exact hex values from palette.ts (bg.base/surface/panel + RISK_PALETTE
 *      borders) appear inside shared.css.
 */

import fs from "node:fs";
import path from "node:path";

const repoRoot   = process.cwd();
const sharedCssP = path.join(repoRoot, "public", "shared.css");
const paletteP   = path.join(repoRoot, "src", "config", "palette.ts");
const twP        = path.join(repoRoot, "tailwind.config.ts");

function read(p) {
  if (!fs.existsSync(p)) {
    console.error(`lint:tokens: missing file ${p}`);
    process.exit(2);
  }
  return fs.readFileSync(p, "utf8");
}

const shared  = read(sharedCssP);
// `palette.ts` owns the semantic palette (RISK_PALETTE, accent), while
// `tailwind.config.ts` owns the bg.base/surface/panel hex values. Concatenate
// both so the lint can verify either one carries each required token.
const sourcePalette = read(paletteP) + "\n" + read(twP);

const errors = [];

// 1) Hash marker present.
if (!/SHARED_CSS_TOKEN_HASH=v2:dark\+cyan/.test(shared)) {
  errors.push("shared.css is missing the SHARED_CSS_TOKEN_HASH=v2:dark+cyan marker");
}

// 2) Hard-coded base palette must appear in shared.css AND in palette/tailwind.
const REQUIRED_HEX = [
  "#040b16", // bg.base
  "#070f1e", // bg.surface
  "#0a1628", // bg.panel
  "#10b981", // risk normal
  "#f59e0b", // risk mild
  "#f97316", // risk high
  "#ef4444", // risk critical
  "#06b6d4", // accent.cyan
];

const sharedLow  = shared.toLowerCase();
const sourceLow  = sourcePalette.toLowerCase();
for (const hex of REQUIRED_HEX) {
  if (!sharedLow.includes(hex)) {
    errors.push(`shared.css is missing required token ${hex}`);
  }
  if (!sourceLow.includes(hex)) {
    errors.push(`palette.ts/tailwind.config.ts no longer contains expected token ${hex} — update lint-tokens.mjs after intentional palette changes`);
  }
}

if (errors.length > 0) {
  console.error("lint:tokens FAILED:");
  for (const e of errors) console.error("  - " + e);
  process.exit(1);
}

console.log("lint:tokens OK");
