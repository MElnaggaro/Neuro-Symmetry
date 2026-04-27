"""
FastAPI entry point — stub.

Full pipeline wired up in Phase 5. This file exists so the server can be
started and imports checked from Phase 1 onward.
"""

from fastapi import FastAPI

app = FastAPI(title="Neuro-Symmetry v2", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
