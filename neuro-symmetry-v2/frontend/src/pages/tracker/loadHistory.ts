// TS port of public/tracker.html's loadHistory + computeKPIs helpers.
// Reads the same three localStorage keys (ns_assessments / ns_prodromal /
// ns_game) so existing data carries over from the static page.

export type TrackerEntry = {
  date:             string;          // YYYY-MM-DD
  class:            string;
  probability:      number | null;
  symmetry_score:   number | null;
  game_activations: number | null;
  source?:          string;
  raw?:             unknown;
};

const KEYS = {
  prodromal:   "ns_prodromal",
  game:        "ns_game",
  assessments: "ns_assessments",
} as const;

function alertToClass(al: string | undefined | null): string {
  if (al === "RED")    return "High Risk";
  if (al === "YELLOW") return "Mild";
  return "Normal";
}

function alertToSym(al: string | undefined | null, asym_ms: number | undefined | null): number {
  if (al === "NONE") return 0.92 + Math.random() * 0.06;
  const ms = asym_ms ?? 250;
  return Math.max(0.3, 1 - ms / 600);
}

function parseArray<T>(raw: string | null): T[] {
  if (!raw) return [];
  try {
    const v = JSON.parse(raw);
    return Array.isArray(v) ? (v as T[]) : [];
  } catch {
    return [];
  }
}

export function loadHistory(): TrackerEntry[] {
  const entries: TrackerEntry[] = [];

  // ns_assessments — main source, written by App.tsx upsertAssessment + game.html.
  type RawAssessment = {
    date?:            string;
    class?:           string;
    probability?:     number | null;
    symmetry_score?:  number | null;
    game_activations?: number | null;
    source?:          string;
  };
  const assessments = parseArray<RawAssessment>(localStorage.getItem(KEYS.assessments));
  for (const e of assessments) {
    if (!e?.date) continue;
    entries.push({
      date:             typeof e.date === "string" ? e.date.slice(0, 10) : "",
      class:            e.class            ?? "Normal",
      probability:      e.probability      ?? null,
      symmetry_score:   e.symmetry_score   ?? null,
      game_activations: e.game_activations ?? null,
      source:           e.source           ?? "live",
      raw:              e,
    });
  }

  // ns_prodromal — Sentinel records.
  type RawProdromal = {
    timestamp?:     string;
    alert_level?:   string;
    asymmetry_ms?:  number;
    confidence?:    number;
    affected_side?: string;
  };
  const prodromal = parseArray<RawProdromal>(localStorage.getItem(KEYS.prodromal));
  for (const r of prodromal) {
    const dateStr =
      (r.timestamp ?? "").slice(0, 10) || new Date().toISOString().slice(0, 10);
    entries.push({
      date:             dateStr,
      class:            alertToClass(r.alert_level),
      probability:      r.confidence ?? 0,
      symmetry_score:   alertToSym(r.alert_level, r.asymmetry_ms),
      game_activations: null,
      source:           "sentinel",
      raw:              r,
    });
  }

  // ns_game — Face-Joypad sessions.
  type RawGame = { date?: string; score?: number; activations?: number };
  const gameData = parseArray<RawGame>(localStorage.getItem(KEYS.game));
  for (const rec of gameData) {
    if (!rec?.date) continue;
    const dateStr = rec.date.slice(0, 10);
    const existing = entries.find(
      (e) => e.date === dateStr && e.source !== "sentinel",
    );
    if (existing) {
      existing.game_activations =
        (existing.game_activations ?? 0) + (rec.activations ?? 0);
    } else {
      entries.push({
        date:             dateStr,
        class:            "Normal",
        probability:      null,
        symmetry_score:   null,
        game_activations: rec.activations ?? 0,
        source:           "game",
        raw:              rec,
      });
    }
  }

  entries.sort((a, b) => a.date.localeCompare(b.date));
  return entries;
}

export type TrackerKpis = {
  todaySym:    number | null;
  yestSym:     number | null;
  sessions7:   number;
  sessions30:  number;
  best:        TrackerEntry | null;
  streak:      number;
  weeklyDelta: number | null;
};

const avg = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
const prevDay = (d: string) => {
  const dt = new Date(d);
  dt.setDate(dt.getDate() - 1);
  return dt.toISOString().slice(0, 10);
};

export function computeKpis(entries: TrackerEntry[]): TrackerKpis {
  const today      = new Date().toISOString().slice(0, 10);
  const yesterday  = new Date(Date.now() - 864e5).toISOString().slice(0, 10);
  const sevenAgo   = new Date(Date.now() - 7 * 864e5).toISOString().slice(0, 10);
  const thirtyAgo  = new Date(Date.now() - 30 * 864e5).toISOString().slice(0, 10);

  const recent7      = entries.filter((e) => e.date >= sevenAgo);
  const recent30     = entries.filter((e) => e.date >= thirtyAgo);
  const todayEntries = entries.filter((e) => e.date === today);
  const symEntries   = entries.filter((e) => e.symmetry_score != null);

  const todayValid = todayEntries.filter((e) => e.symmetry_score != null);
  const todaySym   = todayValid.length ? avg(todayValid.map((e) => e.symmetry_score as number)) : null;

  const yestEntries = entries.filter((e) => e.date === yesterday && e.symmetry_score != null);
  const yestSym     = yestEntries.length ? avg(yestEntries.map((e) => e.symmetry_score as number)) : null;

  const best = symEntries.reduce<TrackerEntry | null>(
    (b, e) => ((e.symmetry_score ?? 0) > (b?.symmetry_score ?? 0) ? e : b),
    null,
  );

  // Streak — consecutive days from today.
  let streak = 0;
  const uniqueDates = Array.from(new Set(entries.map((e) => e.date))).sort().reverse();
  let cursor = today;
  for (const d of uniqueDates) {
    if (d === cursor) {
      streak++;
      cursor = prevDay(cursor);
    } else if (d < cursor) {
      break;
    }
  }

  // Weekly improvement % (first vs last symmetry-bearing entry in the last 7d).
  let weeklyDelta: number | null = null;
  const weekVals = recent7.filter((e) => e.symmetry_score != null);
  if (weekVals.length >= 2) {
    const first = weekVals[0].symmetry_score as number;
    const last  = weekVals[weekVals.length - 1].symmetry_score as number;
    if (first > 0) weeklyDelta = ((last - first) / first) * 100;
  }

  return {
    todaySym,
    yestSym,
    sessions7:   recent7.length,
    sessions30:  recent30.length,
    best,
    streak,
    weeklyDelta,
  };
}
