"""Scoring entry point: fitted artifacts -> early-warning reports.

Typical batch use (what the nightly Kubernetes CronJob runs)::

    python -m reputation.pipeline.score --top 20 --out artifacts/reports.json

The same functions back the FastAPI service, so the batch job and the online
API can never drift apart in how a report is produced.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from ..config import ARTIFACT_DIR
from ..models.recommender import build_report

LOGGER = logging.getLogger("reputation.score")


def load_bundle(artifact_dir: Path = ARTIFACT_DIR) -> dict:
    """Load the artifact bundle written by the training pipeline."""
    path = Path(artifact_dir) / "model_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m reputation.pipeline.train` first."
        )
    return joblib.load(path)


def latest_rows(supervised: pd.DataFrame) -> pd.DataFrame:
    """Most recent scored month for every business (the live view)."""
    return (
        supervised.sort_values("period")
        .groupby("business_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def score_business(bundle: dict, business_id: str, period: str | None = None) -> dict:
    """Produce one business's early-warning report, defaulting to its last month."""
    supervised = bundle["supervised"]
    rows = supervised[supervised["business_id"] == business_id]
    if rows.empty:
        raise KeyError(f"No scored months for business_id={business_id!r}")
    if period is not None:
        rows = rows[rows["period"] == pd.Timestamp(period)]
        if rows.empty:
            raise KeyError(f"No row for {business_id!r} at period {period!r}")
    row = rows.sort_values("period").tail(1)
    return build_report(
        row,
        bundle["recommender"],
        bundle["annotated_reviews"],
        alerts=bundle.get("alerts"),
        classifier=bundle.get("aspect_classifier"),
    )


def score_portfolio(bundle: dict, top: int | None = None) -> list[dict]:
    """Rank every business by predicted decline risk and report on the worst.

    ``top`` bounds the work: building a full report per venue means a
    counterfactual pass per aspect, so scoring only the riskiest N keeps the
    nightly job inside its container CPU budget.
    """
    latest = latest_rows(bundle["supervised"])
    risk = bundle["decline_predictor"].predict_proba(latest)
    latest = latest.assign(decline_risk=risk).sort_values("decline_risk", ascending=False)
    if top is not None:
        latest = latest.head(top)

    reports = []
    for _, row in latest.iterrows():
        reports.append(
            build_report(
                row.to_frame().T,
                bundle["recommender"],
                bundle["annotated_reviews"],
                alerts=bundle.get("alerts"),
                classifier=bundle.get("aspect_classifier"),
            )
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Score businesses and emit reports")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--business-id", default=None, help="Score a single venue")
    parser.add_argument("--top", type=int, default=10, help="Riskiest N venues")
    parser.add_argument("--out", default=None, help="Write JSON here instead of stdout")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    bundle = load_bundle(Path(args.artifact_dir))

    if args.business_id:
        payload: object = score_business(bundle, args.business_id)
    else:
        payload = score_portfolio(bundle, top=args.top)

    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
        LOGGER.info("Wrote %s", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
