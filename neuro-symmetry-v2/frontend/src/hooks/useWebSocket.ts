import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisResponse, CalibrateResponse } from "@/types/analysis";

export type WSStatus = "disconnected" | "connecting" | "connected" | "error";

export interface UseWebSocketReturn {
  status:        WSStatus;
  connect:       () => void;
  sendFrame:     (b64: string, calibrate?: boolean) => void;
  resetSession:  () => void;
  lastResult:    AnalysisResponse | null;
  lastCalib:     CalibrateResponse | null;
}

const WS_URL =
  `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/stream`;

export function useWebSocket(): UseWebSocketReturn {
  const wsRef   = useRef<WebSocket | null>(null);
  const [status,     setStatus]     = useState<WSStatus>("disconnected");
  const [lastResult, setLastResult] = useState<AnalysisResponse | null>(null);
  const [lastCalib,  setLastCalib]  = useState<CalibrateResponse | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setStatus("connecting");

    const ws = new WebSocket(WS_URL);
    ws.onopen  = () => setStatus("connected");
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("error");

    ws.onmessage = ({ data }: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(data) as Record<string, unknown>;
        if (msg["type"] === "calibrate") {
          setLastCalib(msg as unknown as CalibrateResponse);
        } else if (msg["status"] !== "reset") {
          setLastResult(msg as unknown as AnalysisResponse);
        }
      } catch { /* ignore malformed WS messages */ }
    };

    wsRef.current = ws;
  }, []);

  const sendFrame = useCallback((b64: string, calibrate = false) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(calibrate ? { type: "calibrate", image: b64 } : { image: b64 }));
  }, []);

  const resetSession = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: "reset" }));
  }, []);

  // Cleanup on unmount
  useEffect(() => () => { wsRef.current?.close(); }, []);

  return { status, connect, sendFrame, resetSession, lastResult, lastCalib };
}
