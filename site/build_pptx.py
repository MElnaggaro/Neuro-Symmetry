"""Build deck.pptx from deck.html content.

Preserves all 10 slides plus speaker notes; matches the dark theme
(#050505 background, cyan/teal accents) of the source HTML deck.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os

BG       = RGBColor(0x05, 0x05, 0x05)
FG       = RGBColor(0xF5, 0xF7, 0xFA)
MUTED    = RGBColor(0x8B, 0x94, 0xA3)
DIM      = RGBColor(0x5A, 0x63, 0x73)
HAIRLINE = RGBColor(0x22, 0x22, 0x28)
ACCENT_1 = RGBColor(0x22, 0xD3, 0xEE)  # cyan
ACCENT_2 = RGBColor(0x00, 0xE5, 0xB4)  # teal
GLASS    = RGBColor(0x10, 0x12, 0x16)

SLIDE_W  = Inches(13.333)
SLIDE_H  = Inches(7.5)


def set_slide_bg(slide, color=BG):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.line.fill.background()
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.shadow.inherit = False
    spPr = bg._element.spPr
    return bg


def add_text(slide, x, y, w, h, text, *, size=16, color=FG, bold=False,
             font="Calibri", align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             letter_spacing=None, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    if isinstance(text, str):
        runs = [(text, {})]
    else:
        runs = text
    for i, (segment, opts) in enumerate(runs):
        if i == 0:
            r = p.add_run()
        else:
            r = p.add_run()
        r.text = segment
        f = r.font
        f.name = opts.get("font", font)
        f.size = Pt(opts.get("size", size))
        f.bold = opts.get("bold", bold)
        f.italic = opts.get("italic", italic)
        f.color.rgb = opts.get("color", color)
    return tb


def add_paragraph(tf, text, *, size=14, color=MUTED, bold=False,
                  font="Calibri", align=PP_ALIGN.LEFT, space_before=0,
                  space_after=4, italic=False):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return p


def add_rect(slide, x, y, w, h, *, fill=GLASS, line=HAIRLINE, line_w=0.75,
             corner=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if corner else MSO_SHAPE.RECTANGLE,
        x, y, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(line_w)
    shape.shadow.inherit = False
    if corner is not None:
        try:
            shape.adjustments[0] = corner
        except Exception:
            pass
    return shape


def add_hline(slide, x, y, w, color=HAIRLINE, weight=0.75):
    line = slide.shapes.add_connector(1, x, y, x + w, y)
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    return line


def add_eyebrow(slide, x, y, label, timing=None):
    add_text(slide, x, y, Inches(6), Inches(0.3),
             label.upper(), size=10, color=ACCENT_1, bold=True,
             font="Consolas")
    if timing:
        add_text(slide, x + Inches(2.4), y, Inches(2), Inches(0.3),
                 timing.upper(), size=9, color=DIM, font="Consolas")


def add_notes(slide, text):
    notes_tf = slide.notes_slide.notes_text_frame
    notes_tf.text = text


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
blank = prs.slide_layouts[6]


def new_slide():
    s = prs.slides.add_slide(blank)
    set_slide_bg(s)
    return s


# Slide 1 — Cover ----------------------------------------------------------
s = new_slide()
add_text(s, Inches(0), Inches(2.4), SLIDE_W, Inches(0.4),
         "PITCH  ·  2026", size=11, color=ACCENT_1, bold=True,
         font="Consolas", align=PP_ALIGN.CENTER)
add_text(s, Inches(0), Inches(2.85), SLIDE_W, Inches(1.6),
         "Neuro-Symmetry", size=88, color=FG, bold=False,
         align=PP_ALIGN.CENTER)
add_text(s, Inches(2), Inches(4.7), Inches(9.333), Inches(1.4),
         "A research-grade facial-asymmetry classifier — engineered "
         "geometric features, calibrated MLP — for early FAST-protocol "
         "screening from a standard webcam.",
         size=18, color=MUTED, align=PP_ALIGN.CENTER)
add_rect(s, Inches(6.41), Inches(6.6), Inches(0.5), Inches(0.06),
         fill=ACCENT_1, line=ACCENT_1)
add_notes(s, "Welcome. I'll take you through an 8-minute walk-through. The "
             "core idea is simple: visible facial asymmetry is one of the "
             "earliest signs of a stroke or Bell's palsy, and we've built a "
             "research prototype that flags it from a standard webcam. I "
             "want to be precise from the start — this is a research-grade "
             "tool, not an FDA-cleared device, and the architecture is "
             "engineered features plus a calibrated MLP, not end-to-end "
             "deep learning. Both choices are deliberate, and I'll explain why.")

# Slide 2 — Problem --------------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.6), "01 · Problem", "1 min")
# big stat
add_text(s, Inches(0.8), Inches(1.3), Inches(7.5), Inches(2.6),
         [("1.9", {"size": 160, "color": ACCENT_1, "bold": False}),
          ("M",   {"size": 160, "color": ACCENT_2, "bold": False})],
         size=160, color=ACCENT_1)
add_text(s, Inches(0.8), Inches(4.4), Inches(7), Inches(0.4),
         "NEURONS LOST · PER MINUTE · UNTREATED ISCHAEMIC STROKE",
         size=11, color=MUTED, bold=True, font="Consolas")
add_text(s, Inches(0.8), Inches(4.85), Inches(7), Inches(0.4),
         "SAVER · STROKE 37(1):263–266 · 2006",
         size=10, color=DIM, font="Consolas")
# right-side body
tb = slide_tb = add_text(s, Inches(8.6), Inches(2.0), Inches(4.2), Inches(0.6),
         "Time-to-treatment is the single largest determinant of "
         "outcome. The first signs appear on the face — but the people "
         "who notice them are rarely trained to recognize and route in "
         "time.",
         size=14, color=MUTED)
add_paragraph(tb.text_frame,
              "A bystander sees a drooping mouth. A caregiver notices a "
              "lopsided smile. The “F” in the FAST protocol — face — is "
              "the earliest visible symptom, and the easiest to miss "
              "without training.",
              size=12, color=MUTED, space_before=12)
add_notes(s, "Saver's 2006 paper in the journal Stroke quantified what "
             "neurologists already knew clinically — every minute an "
             "ischaemic stroke goes untreated, roughly 1.9 million neurons "
             "die. The 'F' in the FAST protocol — Face, Arms, Speech, "
             "Time — is specifically the asymmetric facial droop that "
             "appears as the earliest visible sign. The bottleneck isn't "
             "the signal — the signal is already on the face. The "
             "bottleneck is that the people closest to the patient haven't "
             "been trained to recognize it and route to care. That's the "
             "gap our system targets.")

# Slide 3 — Solution -------------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.6), "02 · Solution", "1 min")
add_text(s, Inches(0.8), Inches(1.1), Inches(11.7), Inches(1.2),
         "A research-grade signal, from a standard webcam.",
         size=44, color=FG)
pillars = [
    ("Real-Time Landmark Capture",
     "MediaPipe FaceMesh — 468 dense facial landmarks, in-browser, on a "
     "standard webcam. Kalman + EMA smoothing, quality-gated for blur, "
     "occlusion, and pose.",
     "BROWSER · MEDIAPIPE"),
    ("Engineered 50-D Asymmetry Features",
     "40 bilateral mirror-distance pairs + 6 EAR metrics + 2 brow/mouth "
     "asymmetries + 2 texture/symmetry scalars. Nose-centered, IPD-scaled, "
     "anatomically interpretable.",
     "TABULAR · INTERPRETABLE"),
    ("Calibrated Classifier + Confirmation FSM",
     "SE + TokenAttention MLP, FocalLoss + ArcFace training, "
     "temperature-scaled probabilities, 5-frame confirmation gate. "
     "Not a black box.",
     "CALIBRATED · AUDITABLE"),
]
col_w = Inches(3.85)
gap = Inches(0.25)
start_x = Inches(0.8)
for i, (title, body, tag) in enumerate(pillars):
    x = start_x + (col_w + gap) * i
    add_rect(s, x, Inches(3.0), col_w, Inches(3.6), fill=GLASS,
             line=HAIRLINE, corner=0.06)
    add_text(s, x + Inches(0.3), Inches(3.2), col_w - Inches(0.6),
             Inches(0.6), title, size=15, color=FG, bold=True)
    add_text(s, x + Inches(0.3), Inches(3.8), col_w - Inches(0.6),
             Inches(2.0), body, size=11, color=MUTED)
    add_text(s, x + Inches(0.3), Inches(6.15), col_w - Inches(0.6),
             Inches(0.35), tag, size=9, color=ACCENT_1, bold=True,
             font="Consolas")
add_notes(s, "Three deliberate design choices. First, we use the browser as "
             "the capture surface — MediaPipe FaceMesh produces 468 dense "
             "facial landmarks per frame on commodity hardware, no "
             "specialized cameras. Second, we don't feed pixels to the "
             "network — we extract a 50-D engineered feature vector that "
             "maps directly to anatomical priors: 40 bilateral mirror-"
             "distance pairs covering eyes, brows, mouth, and nasolabial "
             "folds, plus eye-aspect-ratio metrics, brow and mouth "
             "asymmetry scalars, and texture-symmetry scores. Third, the "
             "classifier is small and calibrated — about 42 thousand "
             "parameters with a Squeeze-and-Excitation block, Token "
             "Attention, and an ArcFace margin head. Temperature scaling "
             "brings the calibration error to roughly 0.002. The whole "
             "inference path is auditable layer by layer.")

# Slide 4 — Pipeline -------------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.6), "03 · System", "2 min")
add_text(s, Inches(0.8), Inches(1.1), Inches(11.7), Inches(1.2),
         "Four stages, end to end.", size=44, color=FG)
stages = [
    ("01", "Capture & Quality Gate",
     "MediaPipe FaceMesh (468 landmarks) with Kalman + EMA smoothing. "
     "Frames rejected for blur, occlusion, low luminance, or yaw > 25°."),
    ("02", "Feature Extraction",
     "Nose-centered, IPD-scaled normalization. 50-D vector: 40 mirror-"
     "distance pairs + EAR + brow/mouth asymmetry + texture."),
    ("03", "Classification",
     "SymmetryNetV2: BatchNorm → SE → Linear(50→128) → TokenAttention → "
     "Linear(128→64→32) → ArcFace head → softmax over Normal / Mild / "
     "Severe."),
    ("04", "Temporal Engine",
     "EMA (α=0.10) + 60-frame trajectory regression + 90-frame z-score "
     "change detector + 5-frame confirmation FSM."),
]
add_hline(s, Inches(1.1), Inches(3.65), Inches(11.1), color=HAIRLINE)
col_w = Inches(2.85)
gap = Inches(0.05)
start_x = Inches(0.8)
for i, (num, title, body) in enumerate(stages):
    x = start_x + (col_w + gap) * i
    # circle
    circle = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                x + Inches(0.1), Inches(3.25),
                                Inches(0.8), Inches(0.8))
    circle.fill.solid()
    circle.fill.fore_color.rgb = BG
    circle.line.color.rgb = HAIRLINE
    circle.line.width = Pt(1)
    circle.shadow.inherit = False
    add_text(s, x + Inches(0.1), Inches(3.25), Inches(0.8), Inches(0.8),
             num, size=11, color=ACCENT_1, bold=True, font="Consolas",
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, x + Inches(0.1), Inches(4.25), col_w - Inches(0.2),
             Inches(0.5), title, size=14, color=FG, bold=True)
    add_text(s, x + Inches(0.1), Inches(4.8), col_w - Inches(0.2),
             Inches(2.4), body, size=11, color=MUTED)
add_notes(s, "Stage one is browser-side capture. MediaPipe FaceMesh "
             "produces 468 dense landmarks per frame; we apply Kalman "
             "smoothing in the FaceLandmarker, then a second EMA pass to "
             "attenuate slow drift. Quality gates reject frames that fall "
             "outside our operating envelope — luminance below 40, "
             "Laplacian blur variance below 40, frontal occlusion, or "
             "head yaw past 25 degrees. Stage two — and this is the "
             "important one — we don't feed the image to the network. We "
             "extract a 50-dimensional engineered vector. Stage three is "
             "the classifier — SymmetryNetV2. Stage four is post-hoc "
             "temporal logic: EMA, 60-frame regression, z-score change "
             "detector against a 90-frame baseline, and a five-frame "
             "FSM that prevents single-frame false alarms.")

# Slide 5 — Architecture ---------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.5), "03 · System")
add_text(s, Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.9),
         "SymmetryNetV2 — the actual architecture.",
         size=36, color=FG)
layers = [
    ("Input",    "50-D engineered feature vector  ·  single frame"),
    ("Pre-proc", "BatchNorm1d  ·  SE block (50→25→50, sigmoid gate)"),
    ("Backbone", "Linear(50→128)  ·  TokenAttention 4-head  ·  Linear(128→64→32)"),
    ("Heads",    "ArcFace margin (m=0.45, s=30)  ·  Linear(32→3) logits"),
    ("Training", "FocalLoss γ=2.0  ·  ArcFace 60/40  ·  HardNegativeMiner"),
    ("Post-hoc", "Temperature scaling (LBFGS)  ·  EMA + trajectory + z-score  ·  5-frame FSM  ·  region attribution"),
]
y = Inches(2.05)
row_h = Inches(0.55)
for name, body in layers:
    add_rect(s, Inches(0.8), y, Inches(11.7), Inches(0.5),
             fill=GLASS, line=HAIRLINE, corner=0.15)
    add_text(s, Inches(1.0), y, Inches(2.4), Inches(0.5),
             name.upper(), size=10, color=ACCENT_1, bold=True,
             font="Consolas", anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(3.5), y, Inches(8.9), Inches(0.5),
             body, size=12, color=FG, anchor=MSO_ANCHOR.MIDDLE)
    y = y + row_h
add_text(s, Inches(0.8), Inches(5.65), Inches(11.7), Inches(0.6),
         "No CNN. No end-to-end deep learning. Engineered features by "
         "design — clinically interpretable, auditable against anatomical "
         "priors. ~42K parameters · ~105 KB ONNX · CPU-only inference.",
         size=12, color=DIM, italic=True)
add_notes(s, "This is a deliberate design choice and worth defending "
             "directly. The 50-D feature vector is already extracted and "
             "geometrically normalized before the network sees it — so "
             "this is a tabular classification problem on structured "
             "anatomical geometry. SE block recalibrates per-feature "
             "importance. Four-head Token Attention learns relationships "
             "across the feature sequence. ArcFace adds an angular "
             "margin of 0.45 radians. FocalLoss γ=2.0 + HardNegativeMiner "
             "+ LBFGS temperature scaling brings ECE to ~0.002. Roughly "
             "42K parameters, 105KB ONNX, CPU-only.")

# Slide 6 — Results --------------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.6), "04 · Results", "2 min")
add_text(s, Inches(0.8), Inches(1.1), Inches(11.7), Inches(1.0),
         "Real numbers. Real caveats.", size=44, color=FG)
metrics = [
    ("0.986",  "F1 — macro",            "internal test split"),
    ("0.998",  "AUC — ROC",             "internal test split"),
    ("0.982",  "Sensitivity",            "internal test split"),
    ("0.0018", "ECE — calibration error","post temperature scaling"),
]
mx, my = Inches(0.8), Inches(2.5)
mw, mh = Inches(2.6), Inches(2.0)
for i, (val, label, sub) in enumerate(metrics):
    row, col = divmod(i, 2)
    x = mx + (mw + Inches(0.15)) * col
    y = my + (mh + Inches(0.15)) * row
    add_rect(s, x, y, mw, mh, fill=GLASS, line=HAIRLINE, corner=0.06)
    add_text(s, x + Inches(0.2), y + Inches(0.25), mw - Inches(0.4),
             Inches(0.9), val, size=36, color=ACCENT_1, bold=True)
    add_text(s, x + Inches(0.2), y + Inches(1.2), mw - Inches(0.4),
             Inches(0.4), label, size=11, color=FG, bold=True)
    add_text(s, x + Inches(0.2), y + Inches(1.55), mw - Inches(0.4),
             Inches(0.4), sub, size=9, color=DIM, font="Consolas")

# right side credibility
right_x = Inches(6.4)
right_w = Inches(6.1)
# why
add_text(s, right_x, Inches(2.5), right_w, Inches(0.4),
         "Why these numbers are plausible", size=13, color=ACCENT_2,
         bold=True)
why_items = [
    "Structured tabular problem — 50-D normalized geometry, not pixels",
    "FocalLoss + HardNegativeMiner directly target class imbalance",
    "ArcFace margin separates Normal / Mild / Severe in embedding space",
    "Temperature scaling brings ECE well below clinical thresholds",
    "Trained on 462,668 samples; ~42K-parameter model resists memorization",
]
tb = add_text(s, right_x, Inches(2.95), right_w, Inches(0.3),
              "•  " + why_items[0], size=10, color=MUTED)
for it in why_items[1:]:
    add_paragraph(tb.text_frame, "•  " + it, size=10, color=MUTED,
                  space_after=2)
# limits
add_text(s, right_x, Inches(4.5), right_w, Inches(0.4),
         "Where the numbers come from", size=13, color=ACCENT_1,
         bold=True)
limit_items = [
    "Trained on CelebA (~202k normal faces) + 300-W landmarks + YFP "
    "regional patches (~14k crops, 22 palsy patients)",
    "No actual stroke-patient data; AffectNet referenced in code but "
    "absent on disk during this run",
    "Internal held-out test split — not external clinical validation",
    "No prospective cohort, no FDA clearance, no demographic-bias audit yet",
]
tb = add_text(s, right_x, Inches(4.95), right_w, Inches(0.3),
              "•  " + limit_items[0], size=10, color=MUTED)
for it in limit_items[1:]:
    add_paragraph(tb.text_frame, "•  " + it, size=10, color=MUTED,
                  space_after=2)
add_notes(s, "F1 macro 0.986, AUC 0.998, sensitivity 0.982, ECE 0.0018 — "
             "from metrics_v2.json on a 462,668-sample internal test "
             "split. Plausible because tabular classification on 50 "
             "normalized geometric features is narrower than pixel-level "
             "scene understanding. Caveat: training corpus was CelebA + "
             "300-W + YFP regional patches (~14k crops from 22 palsy "
             "patients). No stroke-patient images. AffectNet referenced "
             "but not on disk during this run. Strong internal-test "
             "result on a narrow distribution — not yet a real-world "
             "clinical claim.")

# Slide 7 — Demo -----------------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.6), "05 · Demo", "1 min")
add_text(s, Inches(0.8), Inches(1.1), Inches(11.7), Inches(1.0),
         "The V2 system, live.", size=44, color=FG)
# video card stand-in
card = add_rect(s, Inches(0.8), Inches(2.6), Inches(11.7), Inches(4.4),
                fill=GLASS, line=HAIRLINE, corner=0.04)
# Try to embed actual video as a movie object; fall back to a tile if it fails
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "MINI DEMO.mp4")
try:
    if os.path.isfile(video_path):
        s.shapes.add_movie(video_path,
                           Inches(0.8), Inches(2.6),
                           Inches(11.7), Inches(4.4),
                           mime_type="video/mp4")
    else:
        raise FileNotFoundError(video_path)
except Exception:
    add_text(s, Inches(0.8), Inches(4.0), Inches(11.7), Inches(0.6),
             "▶  MINI DEMO.mp4", size=22, color=ACCENT_1, bold=True,
             align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.8), Inches(4.7), Inches(11.7), Inches(0.5),
             "(see site/MINI DEMO.mp4 in the repository)",
             size=12, color=MUTED, align=PP_ALIGN.CENTER)
add_notes(s, "The V2 system: React frontend, FastAPI backend, ONNX "
             "inference. docker compose launches both — backend on 8000, "
             "frontend on 3000, WebSocket at /ws/stream. Live symmetry "
             "gauge, XAI breakdown across seven facial regions, canvas "
             "heatmap on the camera feed, risk-level badge, trajectory "
             "state. Personal z-score baseline calibrates over 120 "
             "frames. Honest note: subtle asymmetries are normal — the "
             "system flags degree and pattern that exceed personal "
             "baseline drift, not every micro-asymmetry.")

# Slide 8 — Value + Future -------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.6), "06 · Value + Future", "1 min")
add_text(s, Inches(0.8), Inches(1.1), Inches(11.7), Inches(1.0),
         "Research prototype today. Validation next.",
         size=40, color=FG)
# left: immediate value
add_text(s, Inches(0.8), Inches(2.5), Inches(5.7), Inches(0.4),
         "Immediate value", size=14, color=ACCENT_2, bold=True)
val_items = [
    "Any room with a webcam becomes an early-warning layer",
    "Elder care, ambulance triage, telemedicine, remote monitoring",
    "No GPU, no specialized hardware — ONNX runtime on CPU",
    "~105 KB model · CPU inference · deploys on existing infrastructure",
    "FAST-protocol “Face” component, automated",
]
tb = add_text(s, Inches(0.8), Inches(3.0), Inches(5.7), Inches(0.3),
              "•  " + val_items[0], size=12, color=MUTED)
for it in val_items[1:]:
    add_paragraph(tb.text_frame, "•  " + it, size=12, color=MUTED,
                  space_after=4)
# right: roadmap
add_text(s, Inches(7.0), Inches(2.5), Inches(5.5), Inches(0.4),
         "Roadmap", size=14, color=ACCENT_1, bold=True)
roadmap = [
    ("Now",
     "Research-use prototype · documented as non-diagnostic · "
     "DisclaimerModal in V2 UI is the next item"),
    ("Near",
     "External-cohort validation: real stroke + Bell's palsy patients · "
     "prospective · demographic-bias audit"),
    ("Regulatory",
     "FDA SaMD pre-submission only after external validation completes · "
     "HIPAA · CE Mark on the same gate"),
    ("Long-term",
     "EHR routing integration · on-device deployment · liveness "
     "detection layer"),
]
y = Inches(3.0)
for phase, desc in roadmap:
    add_text(s, Inches(7.0), y, Inches(1.25), Inches(0.4),
             phase.upper(), size=10, color=ACCENT_1, bold=True,
             font="Consolas")
    add_text(s, Inches(8.3), y, Inches(4.2), Inches(0.9),
             desc, size=11, color=MUTED)
    y = y + Inches(0.85)
add_notes(s, "Near-term value is real — any room with a webcam becomes a "
             "detection layer. ~105 KB model, CPU, FAST 'Face' component. "
             "But careful framing: this is a research prototype today. "
             "Honest next step isn't a clinical pitch — it's an external-"
             "cohort validation study with real stroke + Bell's palsy "
             "patients, plus a demographic-bias audit. FDA SaMD only "
             "after that. CE Mark + HIPAA on the same gate. Long-term: "
             "EHR routing, on-device, liveness detection (real gap today).")

# Slide 9 — Q&A ------------------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.5), "Q & A")
qa = [
    ("Why are the results so high?",
     "Tabular geometry on 50 normalized features is a structured problem — "
     "not pixel-level scene understanding. Class imbalance is directly "
     "addressed (FocalLoss + HardNegativeMiner). ArcFace separates classes "
     "in embedding space. ECE = 0.0018 confirms the model is well-"
     "calibrated on the test distribution."),
    ("Is this overfitting?",
     "Risk is non-zero. Internal train/test split, not external. Training "
     "corpus was CelebA + 300-W + YFP regional patches — a narrow "
     "distribution. ArcFace + HardNegativeMiner reduce easy-negative "
     "shortcut learning; temperature calibration would surface confidence "
     "collapse on OOD data. A prospective clinical cohort hasn't been run."),
    ("Real-world conditions?",
     "Quality gates reject low luminance (HSV V<40), motion blur "
     "(Laplacian variance<40), occlusion (z-spread<0.01), and extreme "
     "yaw (>25°). Past those thresholds output is suppressed rather "
     "than emitted as a false alert. Yaw past 25°, common in "
     "conversation, currently produces biased features."),
    ("What are the limitations?",
     "No clinical-cohort training data. No liveness detection. "
     "Personalized-baseline calibration was removed in V2. Demographic "
     "bias not characterized. assumed_fps=30 baked into temporal math "
     "while actual WebSocket pipeline is closer to 7 FPS. Single-face "
     "only. Insensitive to pure unilateral ptosis without EAR delta."),
]
qx = Inches(0.8)
qy = Inches(1.2)
qw = Inches(5.85)
qh = Inches(2.85)
for i, (q, a) in enumerate(qa):
    row, col = divmod(i, 2)
    x = qx + (qw + Inches(0.15)) * col
    y = qy + (qh + Inches(0.15)) * row
    add_rect(s, x, y, qw, qh, fill=GLASS, line=HAIRLINE, corner=0.05)
    add_text(s, x + Inches(0.3), y + Inches(0.2), qw - Inches(0.6),
             Inches(0.5), q, size=14, color=ACCENT_1, bold=True)
    add_text(s, x + Inches(0.3), y + Inches(0.75), qw - Inches(0.6),
             qh - Inches(0.9), a, size=11, color=MUTED)
add_notes(s, "Four questions to expect from a senior reviewer. The "
             "answers above are the substance. Posture: real numbers, "
             "real caveats, no overclaim, no buzzwords without proof.")

# Slide 10 — Credibility ---------------------------------------------------
s = new_slide()
add_eyebrow(s, Inches(0.8), Inches(0.5), "Appendix · Credibility & Risk")
add_text(s, Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.9),
         "What survives scrutiny — and what doesn't.",
         size=36, color=FG)
# left
add_text(s, Inches(0.8), Inches(2.05), Inches(5.7), Inches(0.4),
         "Why the architecture is defensible", size=13,
         color=ACCENT_2, bold=True)
defensible = [
    "Tabular structured problem; pre-normalized geometric features",
    "Engineered features map directly to anatomical priors (mirror "
    "pairs, EAR, asymmetry scalars)",
    "50-D input · ~42K parameters → very low memorization risk vs. a "
    "CNN of similar accuracy",
    "Temperature scaling + ECE 0.0018 → probabilities are usable, not "
    "just argmax",
    "60/40 FocalLoss/ArcFace + HardNegativeMiner directly target "
    "shortcut learning",
    "Post-hoc temporal FSM separates single-frame noise from confirmed "
    "events",
]
tb = add_text(s, Inches(0.8), Inches(2.55), Inches(5.7), Inches(0.3),
              "•  " + defensible[0], size=11, color=MUTED)
for it in defensible[1:]:
    add_paragraph(tb.text_frame, "•  " + it, size=11, color=MUTED,
                  space_after=4)
# right
add_text(s, Inches(7.0), Inches(2.05), Inches(5.5), Inches(0.4),
         "Risks that remain", size=13, color=ACCENT_1, bold=True)
risks = [
    "Single-source training distribution (no real stroke cohort)",
    "Calibration tuned on the same distribution as test — degrades on "
    "demographic / camera shift",
    "assumed_fps=30 constant baked into onset-time math",
    "No liveness detection; no anti-spoofing layer",
    "5-frame confirmation FSM resets on 3-frame WebSocket dropout",
    "Personalized baseline removed in V2 — pre-existing asymmetry "
    "will be misread",
    "Insensitive to subtle ptosis without EAR delta and to nasolabial "
    "flattening",
]
tb = add_text(s, Inches(7.0), Inches(2.55), Inches(5.5), Inches(0.3),
              "•  " + risks[0], size=11, color=MUTED)
for it in risks[1:]:
    add_paragraph(tb.text_frame, "•  " + it, size=11, color=MUTED,
                  space_after=4)
add_text(s, Inches(0.8), Inches(6.7), Inches(11.7), Inches(0.6),
         "Built for the seconds that matter — and honest about the "
         "seconds it can't yet account for.   "
         "Source: ai/checkpoints/metrics_v2.json · "
         "Neuro-v2-Reprot.md §9 · ai/v2/model_v2.py · ai/v2/losses.py",
         size=10, color=DIM, italic=True)
add_notes(s, "The slide to leave on screen if a senior reviewer wants to "
             "dig in. Left: defensible architecture. Right: unsolved "
             "risks. Footer references actual files. Shows we know "
             "exactly where the work is real and exactly where the gaps "
             "are — that's what makes it survivable under scrutiny.")

# Save --------------------------------------------------------------------
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.pptx")
prs.save(out)
print("Wrote:", out)
