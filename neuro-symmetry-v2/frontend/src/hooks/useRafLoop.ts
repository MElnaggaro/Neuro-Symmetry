import { useEffect, useRef } from "react";
import logger from "@/utils/logger";

/**
 * useRafLoop — runs a callback on every animation frame while active.
 *
 * Uses the "latest ref" pattern: the callback is stored in a ref before
 * each render, so the RAF loop always calls the current closure without
 * needing to list volatile captures as deps (no stale-closure risk).
 *
 * The hook starts the loop when `active` becomes true and stops it when
 * it becomes false or the component unmounts.
 */
export function useRafLoop(fn: () => void, active: boolean, label?: string): void {
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!active) return;
    let running = true;
    const tick = () => {
      if (!running) return;
      fnRef.current();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    if (label) logger.info(label, "RAF loop started");
    return () => {
      running = false;
      if (label) logger.info(label, "RAF loop stopped");
    };
  }, [active, label]);
}
