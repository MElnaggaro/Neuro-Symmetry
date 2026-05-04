import { lazy, Suspense } from "react";
import App from "./App";
import { useHashRoute } from "@/router/api";

const Tracker = lazy(() => import("@/pages/Tracker"));

/**
 * Mounts <App> at "/" and <Tracker> at "/tracker". App stays mounted at all
 * times so the camera + WebSocket pipeline never tears down on navigation;
 * the tracker is rendered as a sibling that hides App via display:none.
 */
export default function Routed() {
  const route = useHashRoute();

  return (
    <>
      <div style={{ display: route === "/tracker" ? "none" : "contents" }}>
        <App />
      </div>
      {route === "/tracker" && (
        <Suspense
          fallback={
            <div className="flex items-center justify-center h-screen text-slate-500 text-label tracking-cyber">
              LOADING TRACKER…
            </div>
          }
        >
          <Tracker />
        </Suspense>
      )}
    </>
  );
}
