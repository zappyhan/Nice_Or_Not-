"""FastAPI service exposing the early-warning system.

Endpoints
---------
GET  /                           the owner-facing dashboard (static single page)
GET  /health                     liveness/readiness probe for Kubernetes
GET  /metrics-summary            metrics recorded at training time
GET  /businesses                 risk-ranked list of monitored venues
GET  /businesses/{id}/report     full report: risk, causes, evidence, action
GET  /businesses/{id}/history    monthly rating and complaint-rate series
GET  /businesses/{id}/aspects    all five aspects scored and ranked
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import ARTIFACT_DIR, ASPECTS
from ..models.aspect_classifier import weak_label
from ..pipeline.score import load_bundle, latest_rows, score_business, score_portfolio

LOGGER = logging.getLogger("reputation.api")

# The dashboard is plain HTML/CSS/JS shipped beside this module -- no build step,
# no CDN, so it works offline inside the container and in an exam-hall demo.
STATIC_DIR = Path(__file__).resolve().parent / "static"

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
            "name": _display_name(r.business_id),
            "period": pd.Timestamp(r.period).strftime("%Y-%m"),
            "decline_risk": round(float(r.decline_risk), 4),
            "hist_mean_stars": None if pd.isna(r.hist_mean_stars) else round(float(r.hist_mean_stars), 2),
            "hist_reviews": int(r.hist_reviews),
        }
        for r in ranked.itertuples()
    ]


def _display_name(business_id: str) -> str:
    """Human-readable venue name, falling back to the raw id.

    Yelp ships names in the business table; the dashboard shows them because
    "Sunrise Cafe" means something to an owner and a hash id does not.
    """
    businesses = STATE["bundle"].get("businesses")  # type: ignore[union-attr]
    if businesses is None or "name" not in getattr(businesses, "columns", []):
        return business_id
    match = businesses.loc[businesses["business_id"] == business_id, "name"]
    return str(match.iloc[0]) if len(match) else business_id


@app.get("/businesses/{business_id}/report")
def business_report(business_id: str, period: str | None = None) -> dict:
    """Full early-warning report: risk, ranked causes, evidence, first action."""
    try:
        return score_business(STATE["bundle"], business_id, period)  # type: ignore[arg-type]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/businesses/{business_id}/history")
def business_history(
    business_id: str, months: int = Query(24, ge=3, le=120)
) -> dict:
    """Monthly series behind the dashboard charts.

    Returns the trailing ``months`` of average rating, review volume and
    per-aspect complaint rates. Months with no reviews are present with a null
    rating rather than omitted, so a gap in the trend line reads as "nobody
    reviewed us" instead of silently interpolating over the quiet period.
    """
    panel: pd.DataFrame = STATE["bundle"]["panel"]  # type: ignore[index]
    rows = panel[panel["business_id"] == business_id].sort_values("period").tail(months)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Unknown business {business_id!r}")

    return {
        "business_id": business_id,
        "name": _display_name(business_id),
        "months": [
            {
                "period": pd.Timestamp(r.period).strftime("%Y-%m"),
                "n_reviews": int(r.n_reviews),
                "mean_stars": None if pd.isna(r.mean_stars) else round(float(r.mean_stars), 3),
                "negative_rate": None if pd.isna(r.negative_rate) else round(float(r.negative_rate), 4),
                **{
                    aspect: (
                        None
                        if pd.isna(getattr(r, f"complaint_rate_{aspect}"))
                        else round(float(getattr(r, f"complaint_rate_{aspect}")), 4)
                    )
                    for aspect in ASPECTS
                },
            }
            for r in rows.itertuples()
        ],
    }


@app.get("/businesses/{business_id}/aspects")
def business_aspects(business_id: str) -> dict:
    """All five aspects scored for the venue's latest month, ranked by priority.

    The report endpoint returns only the top three findings because that is what
    an owner should act on; the dashboard needs the full five to draw the
    "you versus the market" comparison without gaps.
    """
    supervised: pd.DataFrame = STATE["bundle"]["supervised"]  # type: ignore[index]
    rows = supervised[supervised["business_id"] == business_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Unknown business {business_id!r}")

    row = rows.sort_values("period").tail(1)
    scored = STATE["bundle"]["recommender"].explain(row)  # type: ignore[index]
    return {
        "business_id": business_id,
        "name": _display_name(business_id),
        "period": pd.Timestamp(row["period"].iloc[0]).strftime("%Y-%m"),
        "decline_risk": round(float(scored["decline_risk"].iloc[0]), 4),
        "aspects": [
            {
                "aspect": r.aspect,
                "label": r.label,
                "complaint_rate": round(float(r.complaint_rate), 4),
                "market_median": round(float(r.market_median), 4),
                "trend": round(float(r.trend), 4),
                "attributed_risk": round(float(r.attributed_risk), 4),
                "estimated_star_cost": round(float(r.estimated_star_cost), 3),
                "priority_score": round(float(r.priority_score), 4),
                "suggested_action": r.suggested_action,
            }
            for r in scored.itertuples()
        ],
    }


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


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
# Mounted last so it cannot shadow an API route. The page is a single static
# file that talks to the JSON endpoints above, which keeps the serving tier one
# container with no Node build stage.

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the owner-facing dashboard."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard assets not installed")
    return FileResponse(index)
