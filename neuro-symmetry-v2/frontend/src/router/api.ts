// Hooks + helpers for the hash router. Kept separate from the <Route>
// component so HMR fast-refresh works in router/index.tsx.

import { useSyncExternalStore } from "react";

function readHash(): string {
  if (typeof window === "undefined") return "/";
  const h = window.location.hash || "#/";
  return h.startsWith("#") ? h.slice(1) || "/" : h;
}

function subscribe(cb: () => void) {
  window.addEventListener("hashchange", cb);
  return () => window.removeEventListener("hashchange", cb);
}

export function useHashRoute(): string {
  return useSyncExternalStore(subscribe, readHash, () => "/");
}

export function navigate(path: string): void {
  if (typeof window === "undefined") return;
  if (!path.startsWith("/")) path = "/" + path;
  window.location.hash = path;
}
