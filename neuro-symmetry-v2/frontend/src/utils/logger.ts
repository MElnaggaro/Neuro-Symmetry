import { v4 as uuidv4 } from "uuid";

// ── Log level ─────────────────────────────────────────────────────────────────

const enum Level { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 }

const LEVEL_NAME: Record<Level, string> = {
  [Level.DEBUG]: "DEBUG",
  [Level.INFO]:  "INFO",
  [Level.WARN]:  "WARN",
  [Level.ERROR]: "ERROR",
};

function resolveLevel(): Level {
  const raw = (import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_LOG_LEVEL ?? "INFO";
  return ({ DEBUG: Level.DEBUG, INFO: Level.INFO, WARN: Level.WARN, ERROR: Level.ERROR } as Record<string, Level>)[
    raw.toUpperCase()
  ] ?? Level.INFO;
}

// ── Internal state ────────────────────────────────────────────────────────────

interface LogEntry {
  timestamp: string;
  level:     string;
  component: string;
  message:   string;
  metadata:  unknown;
}

const SESSION_ID = uuidv4();
const MIN_LEVEL  = resolveLevel();
const queue: LogEntry[] = [];
let   flushing = false;

function flush(entries: LogEntry[]): void {
  for (const { timestamp, level, component, message, metadata } of entries) {
    const line = `[${timestamp}] [${level}] [${component}] [${SESSION_ID}] ${message} ${JSON.stringify(metadata)}`;
    if      (level === "ERROR") console.error(line);
    else if (level === "WARN")  console.warn(line);
    else if (level === "INFO")  console.info(line);
    else                        console.debug(line);
  }
}

function processQueue(): void {
  if (!queue.length || flushing) return;
  flushing = true;
  const batch = queue.splice(0);
  const task  = () => { flush(batch); flushing = false; };
  if ("requestIdleCallback" in window) {
    requestIdleCallback(task);
  } else {
    setTimeout(task, 0);
  }
}

setInterval(processQueue, 1_000);

// ── Public API ────────────────────────────────────────────────────────────────

function log(level: Level, component: string, message: string, metadata: unknown = {}): void {
  if (level < MIN_LEVEL) return;
  queue.push({
    timestamp: new Date().toISOString(),
    level:     LEVEL_NAME[level],
    component,
    message,
    metadata,
  });
}

const logger = {
  debug: (c: string, m: string, d?: unknown) => log(Level.DEBUG, c, m, d),
  info:  (c: string, m: string, d?: unknown) => log(Level.INFO,  c, m, d),
  warn:  (c: string, m: string, d?: unknown) => log(Level.WARN,  c, m, d),
  error: (c: string, m: string, d?: unknown) => log(Level.ERROR, c, m, d),
};

export default logger;