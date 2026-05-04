// Motion presets for framer-motion. Wraps the raw MOTION tokens from palette.ts
// into ready-to-spread `transition` objects so call sites stay terse.

import { MOTION } from "@/config/palette";

const sec = (ms: number) => ms / 1000;

export const motion = {
  fast:    { duration: sec(MOTION.fast),  ease: [...MOTION.ease]    as [number, number, number, number] },
  base:    { duration: sec(MOTION.base),  ease: [...MOTION.ease]    as [number, number, number, number] },
  slow:    { duration: sec(MOTION.slow),  ease: [...MOTION.easeOut] as [number, number, number, number] },
  gauge:   { duration: sec(MOTION.gauge), ease: [...MOTION.easeOut] as [number, number, number, number] },
  spring:  { type: "spring" as const, stiffness: 120, damping: 22 },
} as const;

export type MotionPreset = keyof typeof motion;
