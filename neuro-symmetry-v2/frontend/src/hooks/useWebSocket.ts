import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisResponse, CalibrateResponse } from "@/types/analysis";

// ── Types ─────────────────────────────────────────────────────────────────────

export type WSStatus = "disconnected" | "connecting" | "connected" | "error";

export interface UseWebSocketReturn {
  status:       WSStatus;
  connect:      () => void;
  sendFrame:    (b64: string, calibrate?: boolean) => void;
  resetSession: () => void;
  lastResult:   AnalysisResponse | null;
  lastCalib:    CalibrateResponse | null;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const WS_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/stream`;

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useWebSocket(): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);

  const [status,     setStatus]     = useState<WSStatus>("disconnected");
  const [lastResult, setLastResult] = useState<AnalysisResponse | null>(null);
  const [lastCalib,  setLastCalib]  = useState<CalibrateResponse | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(WS_URL);

    ws.onopen    = () => setStatus("connected");
    ws.onclose   = () => setStatus("disconnected");
    ws.onerror   = () => setStatus("error");
    ws.onmessage = ({ data }: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(data) as Record<string, unknown>;
        if      (msg["type"] === "calibrate")     setLastCalib(msg as unknown as CalibrateResponse);
        else if (msg["status"] !== "reset")        setLastResult(msg as unknown as AnalysisResponse);
      } catch { /* discard malformed messages */ }
    };

    wsRef.current = ws;
  }, []);

  const sendFrame = useCallback((b64: string, calibrate = false) => {
    const ws = wsRef.current;
    if (ws?.readyState !== WebSocket.OPEN) return;
    // Strip the "data:image/...;base64," prefix that toDataURL() prepends
    const image = b64.includes(",") ? b64.split(",")[1] : b64;
    ws.send(JSON.stringify(calibrate ? { type: "calibrate", image } : { image }));
  }, []);

  const resetSession = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "reset" }));
  }, []);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  return { status, connect, sendFrame, resetSession, lastResult, lastCalib };
}