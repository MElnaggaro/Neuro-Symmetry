# Neuro-Symmetry v2.0

Multimodal Neurological Signal Detection Engine — detects early signs of stroke
and facial palsy via real-time facial asymmetry analysis.

## Quick start

```bash
# Python environment
pip install -e ".[dev]"

# Verify dataset paths
python -m datasets.celeba_loader
python -m datasets.w300_loader
python -m datasets.palsy_loader

# Backend (stub)
uvicorn backend.api.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Dataset paths

All datasets are expected at `../Datasets/` relative to this directory:

| Dataset | Path | Purpose |
|---------|------|---------|
| CelebA | `../Datasets/Celeba/` | Normal class pretraining |
| 300-W (AFW) | `../Datasets/ibug_300W_large_face_landmark_dataset/` | Landmark accuracy |
| YFP Palsy | `../Datasets/YFP_Dataset/` | Pathological fine-tuning |
| AffectNet | `../Datasets/AffectNet/` | Expression filtering (Phase 4) |

## Implementation phases

See `neuro-symmetry-v2-phases.md` in the parent directory for the full
10-phase implementation plan.

## Regulatory notice

This system is designed as an assistive early-warning tool, not a medical
diagnostic device. Clinical deployment requires FDA SaMD, HIPAA, and CE Mark
compliance review.
