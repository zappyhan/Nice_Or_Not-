"""FastAPI service exposing the early-warning system.

Endpoints
---------
GET  /health                     liveness/readiness probe for Kubernetes
GET  /metrics-summary            metrics recorded at training time
GET  /businesses                 risk-ranked list of monitored venues
GET  /businesses/{id}/report     full report: risk, causes, evidence, action
GET  /alerts                     emerging-complaint alerts, newest first
POST /analyse                    score ad-hoc review text (no history needed)

The service loads the artifact bundle once at start-up and keeps it in memory,
so a request costs a few model evaluations rather than a retrain. In the
container image the bundle is baked in at build time (or mounted from a volume
in the Kubernetes deployment) which keeps the pods stateless and horizontally
scalable.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..config import ARTIFACT_DIR, ASPECTS
from ..models.aspect_classifier import weak_label
from ..pipeline.score import load_bundle, latest_rows, score_business, score_portfolio

LOGGER = logging.getLogger("reputation.api")

# Populated during the lifespan start-up hook below.
STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once per process; fail loudly if artifacts are missing."""
    artifact_dir = Path(os.environ.get("ARTIFACT_DIR", ARTIFACT_DIR))
    LOGGER.info("Loading model bundle from %s", artifact_dir)
    STATE["bundle"] = load_bundle(artifact_dir)
    LOGGER.info("Model bundle ready")
    yield
    STATE.clear()


app = FastAPI(
    title="Restaurant Reputation Early-Warning System",
    description=(
        "Detects emerging complaints in review streams, predicts rating decline, "
        "and recommends which operational problem to fix first."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class ReviewBatch(BaseModel):
    """Ad-hoc text submitted for aspect analysis."""

    texts: list[str] = Field(..., min_length=1, description="Raw review texts")
    stars: list[int] | None = Field(
        None, description="Optional star ratings, used to separate praise from complaint"
    )


@app.get("/health")
def health() -> dict:
    """Readiness probe: reports whether the model bundle is loaded."""
    ready = "bundle" in STATE
    return {"status": "ok" if ready else "loading", "model_loaded": ready}


@app.get("/metrics-summary")
def metrics_summary() -> dict:
    """Evaluation metrics captured when these artifacts were trained."""
    import json

    path = Path(os.environ.get("ARTIFACT_DIR", ARTIFACT_DIR)) / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return json.loads(path.read_text())


@app.get("/businesses")
def list_businesses(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    """Monitored venues ranked by predicted probability of a rating decline."""
    bundle = STATE["bundle"]
    latest = latest_rows(bundle["supervised"])          # type: ignore[index]
    risk = bundle["decline_predictor"].predict_proba(latest)  # type: ignore[index]
    ranked = (
        latest.assign(decline_risk=risk)
        .sort_values("decline_risk", ascending=False)
        .head(limit)
    )
    return [
        {
            "business_id": r.business_id,
            "period": pd.Timestamp(r.period).strftime("%Y-%m"),
            "decline_risk": round(float(r.decline_risk), 4),
            "hist_mean_stars": None if pd.isna(r.hist_mean_stars) else round(float(r.hist_mean_stars), 2),
            "hist_reviews": int(r.hist_reviews),
        }
        for r in ranked.itertuples()
    ]


@app.get("/businesses/{business_id}/report")
def business_report(business_id: str, period: str | None = None) -> dict:
    """Full early-warning report: risk, ranked causes, evidence, first action."""
    try:
        return score_business(STATE["bundle"], business_id, period)  # type: ignore[arg-type]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/alerts")
def alerts(
    severity: str | None = Query(None, pattern="^(watch|high|critical)$"),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict]:
    """Emerging-complaint alerts, most significant first."""
    frame: pd.DataFrame = STATE["bundle"]["alerts"]  # type: ignore[index]
    if frame.empty:
        return []
    if severity:
        frame = frame[frame["severity"] == severity]
    return frame.head(limit).to_dict(orient="records")


@app.get("/portfolio/report")
def portfolio_report(top: int = Query(10, ge=1, le=100)) -> list[dict]:
    """Batch view: full reports for the riskiest ``top`` venues."""
    return score_portfolio(STATE["bundle"], top=top)  # type: ignore[arg-type]


@app.post("/analyse")
def analyse(batch: ReviewBatch) -> dict:
    """Classify ad-hoc review text into aspects -- the interactive demo endpoint.

    Useful for a live demo ("type a complaint, watch it get routed") and for
    clients whose reviews do not live on Yelp: the same classifier runs on any
    text stream, which is what makes the system portable beyond one dataset.
    """
    classifier = STATE["bundle"]["aspect_classifier"]  # type: ignore[index]
    texts = pd.Series(batch.texts)
    probabilities = classifier.predict_proba(texts)
    lexicon = weak_label(texts)

    results = []
    for i, text in enumerate(batch.texts):
        stars = batch.stars[i] if batch.stars and i < len(batch.stars) else None
        ranked = probabilities.iloc[i].sort_values(ascending=False)
        results.append(
            {
                "text": text[:300],
                "stars": stars,
                "aspects": {a: round(float(probabilities.iloc[i][a]), 4) for a in ASPECTS},
                "top_aspect": str(ranked.index[0]),
                "lexicon_hits": [a for a in ASPECTS if lexicon.iloc[i][a] == 1],
                "is_complaint": None if stars is None else bool(stars <= 2),
            }
        )
    return {"n": len(results), "results": results}
