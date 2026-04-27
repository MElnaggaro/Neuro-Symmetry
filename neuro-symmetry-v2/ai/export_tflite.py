"""
Model export — Float16 ONNX + TFLite.

Steps
-----
1. Load a saved CalibratedModel (.pt checkpoint).
2. Export to standard ONNX (float32).
3. Convert ONNX weights to Float16 → model_fp16.onnx  (< 3 MB target).
4. If onnx2tf is installed: convert fp16 ONNX → TFLite flatbuffer.
   Otherwise: log a one-line install instruction and exit cleanly.

Usage
-----
    python -m ai.export_tflite --model ai/checkpoints/model_real_calibrated.pt
    python -m ai.export_tflite --model ai/checkpoints/model_calibrated.pt
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def export(model_pt: Path, out_dir: Path | None = None) -> dict[str, Path]:
    """
    Export a CalibratedModel to ONNX (float32) and Float16 ONNX.
    Attempts TFLite conversion if onnx2tf is available.

    Returns a dict of {format: output_path} for paths that were written.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper, TensorProto
    from ai.model import CalibratedModel

    out_dir = out_dir or model_pt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = model_pt.stem  # e.g. "model_calibrated" or "model_real_calibrated"
    outputs: dict[str, Path] = {}

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model = CalibratedModel.load(model_pt)
    model.eval()
    print(f"Loaded: {model_pt}")

    # ── 2. Float32 ONNX ───────────────────────────────────────────────────────
    onnx_f32 = out_dir / f"{stem}.onnx"
    model.to_onnx(onnx_f32)
    size_f32 = onnx_f32.stat().st_size
    print(f"ONNX float32: {onnx_f32}  ({size_f32/1024:.1f} KB)")
    outputs["onnx_f32"] = onnx_f32

    # ── 3. Float16 ONNX ───────────────────────────────────────────────────────
    onnx_model = onnx.load(str(onnx_f32))
    fp16_initializers = []
    for init in onnx_model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype in (np.float32, np.float64):
            arr_fp16 = arr.astype(np.float16)
            new_init = numpy_helper.from_array(arr_fp16, name=init.name)
            fp16_initializers.append(new_init)
        else:
            fp16_initializers.append(init)

    del onnx_model.graph.initializer[:]
    onnx_model.graph.initializer.extend(fp16_initializers)

    # Update input/output type annotations to float16
    for vi in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

    onnx_fp16 = out_dir / f"{stem}_fp16.onnx"
    onnx.save(onnx_model, str(onnx_fp16))
    size_fp16 = onnx_fp16.stat().st_size
    print(f"ONNX float16: {onnx_fp16}  ({size_fp16/1024:.1f} KB)")
    outputs["onnx_fp16"] = onnx_fp16

    reduction = (1 - size_fp16 / size_f32) * 100
    print(f"Size reduction: {reduction:.1f}%  (f32→f16)")

    # ── 4. TFLite (optional) ──────────────────────────────────────────────────
    tflite_path = out_dir / f"{stem}.tflite"
    try:
        import onnx2tf  # noqa: F401
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "onnx2tf",
             "-i", str(onnx_fp16),
             "-o", str(out_dir / "tflite_tmp"),
             "--output_tfv1_signaturedefs",
             "--non_verbose"],
            capture_output=True, text=True,
        )
        # onnx2tf writes a saved_model; the .tflite lives inside
        tmp_dir = out_dir / "tflite_tmp"
        tflite_candidates = list(tmp_dir.rglob("*.tflite"))
        if tflite_candidates:
            import shutil
            shutil.copy(tflite_candidates[0], tflite_path)
            size_tflite = tflite_path.stat().st_size
            print(f"TFLite:       {tflite_path}  ({size_tflite/1024:.1f} KB)")
            outputs["tflite"] = tflite_path
        else:
            print(f"onnx2tf ran but no .tflite found. stderr:\n{result.stderr[:400]}")
    except ImportError:
        print(
            "TFLite conversion skipped — onnx2tf not installed.\n"
            "  Install:  pip install onnx2tf tensorflow\n"
            f"  Then run: python -m ai.export_tflite --model {model_pt}"
        )

    return outputs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NeuroSymmetryNet to Float16 ONNX / TFLite")
    parser.add_argument("--model", required=True, help="Path to model_calibrated.pt")
    parser.add_argument("--out",   default=None,  help="Output directory (default: same as model)")
    args = parser.parse_args()

    paths = export(
        model_pt=Path(args.model),
        out_dir=Path(args.out) if args.out else None,
    )
    print("\nExported files:")
    for fmt, p in paths.items():
        print(f"  {fmt:15s} {p}  ({p.stat().st_size/1024:.1f} KB)")
