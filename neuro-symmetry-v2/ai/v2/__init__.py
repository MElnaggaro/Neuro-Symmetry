"""
Neuro-Symmetry V2 — upgraded ML pipeline.

Modules
-------
feature_engineering   25-D geometric asymmetry features from LandmarkResult
model_v2              SymmetryNetV2 (SE block + token attention, ~42 K params)
losses                FocalLoss + ArcMarginHead + CombinedLoss
post_hoc              Temperature scaling, optimal thresholds, temporal smoother
train_v2              Training script (uses existing 50-D feature caches)
inference_pipeline    Production inference — Pydantic schemas, env config, FastAPI
"""
