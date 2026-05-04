import { jsPDF } from "jspdf";
import type { TrackerEntry, TrackerKpis } from "./loadHistory";

const today = () =>
  new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "long", year: "numeric" });

function fmtPct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

/**
 * Render the chart `<svg>` to a PNG data URL via an Image + canvas. Avoids
 * pulling in html2canvas. Returns null if the SVG can't be rasterised
 * (private mode, missing element, etc).
 */
async function chartSvgToPng(svg: SVGSVGElement | null, scale = 2): Promise<string | null> {
  if (!svg) return null;
  const xml = new XMLSerializer().serializeToString(svg);
  const w = svg.clientWidth  || svg.getBoundingClientRect().width;
  const h = svg.clientHeight || svg.getBoundingClientRect().height;
  if (!w || !h) return null;
  const data = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
  return await new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width  = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(null); return; }
      ctx.fillStyle = "#0a1628";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = () => resolve(null);
    img.src = data;
  });
}

export async function exportTrackerPdf(args: {
  entries: TrackerEntry[];
  kpis:    TrackerKpis;
  chartSvg: SVGSVGElement | null;
}): Promise<void> {
  const { entries, kpis, chartSvg } = args;
  const doc    = new jsPDF({ unit: "pt", format: "a4", orientation: "portrait" });
  const pageW  = doc.internal.pageSize.getWidth();
  const pageH  = doc.internal.pageSize.getHeight();
  const margin = 36;
  const bodyW  = pageW - margin * 2;

  let y = margin;

  const ensureSpace = (need: number) => {
    if (y + need > pageH - margin) {
      doc.addPage();
      y = margin;
    }
  };

  // Cover header
  doc.setFillColor(15, 15, 26);
  doc.rect(0, 0, pageW, pageH, "F");
  doc.setFillColor(26, 26, 46);
  doc.roundedRect(margin, y, bodyW, 60, 10, 10, "F");
  doc.setTextColor(226, 232, 240);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(17);
  doc.text("Neuro-Symmetry Progress Report", margin + 18, y + 24);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(148, 163, 184);
  doc.text(
    `Generated ${today()} | Patient de-identified | Research use only`,
    margin + 18,
    y + 44,
  );
  y += 80;

  // KPI summary
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(148, 163, 184);
  doc.text("SUMMARY", margin, y);
  y += 12;

  const cards: Array<[string, string]> = [
    ["Symmetry today", fmtPct(kpis.todaySym)],
    ["Sessions 7d",    String(kpis.sessions7)],
    ["Best score",     fmtPct(kpis.best?.symmetry_score ?? null)],
    ["Streak",         String(kpis.streak)],
  ];
  const cardW = (bodyW - 24) / 4;
  cards.forEach(([label, value], i) => {
    const x = margin + i * (cardW + 8);
    doc.setFillColor(30, 30, 53);
    doc.roundedRect(x, y, cardW, 54, 8, 8, "F");
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184);
    doc.text(label, x + 10, y + 17);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.setTextColor(226, 232, 240);
    doc.text(value, x + 10, y + 39);
  });
  y += 78;

  // Chart image
  const png = await chartSvgToPng(chartSvg, 2);
  if (png) {
    ensureSpace(190);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.setTextColor(6, 182, 212);
    doc.text("SYMMETRY TREND", margin, y);
    y += 12;
    const w = bodyW;
    const h = w * 0.34; // approximate chart aspect
    doc.addImage(png, "PNG", margin, y, w, h, undefined, "FAST");
    y += h + 18;
  }

  // Session table (first 30 rows for portability)
  ensureSpace(60);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(6, 182, 212);
  doc.text("RECENT SESSIONS", margin, y);
  y += 14;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(226, 232, 240);
  const headers = ["Date", "Class", "Symmetry", "Confidence", "Game", "Source"];
  const cols = [70, 80, 70, 80, 60, bodyW - (70 + 80 + 70 + 80 + 60)];
  let cx = margin;
  for (let i = 0; i < headers.length; i++) {
    doc.text(headers[i], cx, y);
    cx += cols[i];
  }
  y += 12;
  doc.setDrawColor(42, 42, 74);
  doc.line(margin, y - 6, margin + bodyW, y - 6);

  const rows = [...entries].reverse().slice(0, 30);
  for (const r of rows) {
    ensureSpace(14);
    cx = margin;
    const fields = [
      r.date,
      r.class,
      r.symmetry_score != null ? `${(r.symmetry_score * 100).toFixed(1)}%` : "—",
      r.probability != null ? `${Math.round(r.probability * 100)}%` : "—",
      r.game_activations != null ? String(r.game_activations) : "—",
      r.source ?? "—",
    ];
    for (let i = 0; i < fields.length; i++) {
      doc.setTextColor(226, 232, 240);
      doc.text(String(fields[i]).slice(0, 24), cx, y);
      cx += cols[i];
    }
    y += 12;
  }

  // Footer
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(100, 116, 139);
  doc.text(
    "Self-monitoring support material; not a clinical diagnosis.",
    margin,
    pageH - 18,
  );

  doc.save(`neuro-symmetry-report-${new Date().toISOString().slice(0, 10)}.pdf`);
}
