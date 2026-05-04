import { useEffect } from "react";

/**
 * Bind a single keyboard combo. `combo` accepts:
 *   "mod+k"   → ⌘K on macOS, Ctrl+K elsewhere
 *   "ctrl+k"  → Ctrl+K everywhere (also catches Cmd+K so palette works on Mac)
 *   "esc", "/" — single keys
 *
 * Calls `event.preventDefault()` before invoking `handler` to swallow the OS
 * default (browser opens find bar on Ctrl+K, etc.).
 */
export function useHotkey(combo: string, handler: (e: KeyboardEvent) => void): void {
  useEffect(() => {
    const target = combo.toLowerCase();
    const onKey = (e: KeyboardEvent) => {
      const key   = e.key.toLowerCase();
      const mod   = e.metaKey || e.ctrlKey;
      const ctrl  = e.ctrlKey;
      const parts = target.split("+");
      const last  = parts[parts.length - 1];

      const wantMod  = parts.includes("mod");
      const wantCtrl = parts.includes("ctrl");

      if (key !== last) return;
      if (wantMod  && !mod)  return;
      if (wantCtrl && !ctrl && !e.metaKey) return; // accept either on portable code paths

      e.preventDefault();
      handler(e);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [combo, handler]);
}
