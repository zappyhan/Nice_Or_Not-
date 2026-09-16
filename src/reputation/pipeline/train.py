"""Training entry point: raw reviews -> fitted models + evaluation report.

Run it with the bundled simulator (no download needed)::

    python -m reputation.pipeline.train --source synthetic

or against the real Yelp Open Dataset unpacked into ``./data``::

    python -m reputation.pipeline.train --source yelp --city Philadelphia

Everything the serving container needs is written to ``artifacts/``:

    model_bundle.joblib   aspect classifier, decline predictor, star model,
                          market reference, annotated reviews and panel
    metrics.json          the numbers reported in Section 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from ..config import ARTIFACT_DIR, DATA_DIR, SETTINGS
from ..data.loader import (
    generate_synthetic_reviews,
    load_yelp_businesses,
    load_yelp_reviews,
)
from ..evaluation.metrics import (
    aspect_agreement,
    classification_report_dict,
    lead_time_analysis,
    lead_time_summary,
    majority_baseline,
    naive_rating_baseline,
)
from ..features.panel import build_monthly_panel, build_supervised_frame
from ..models.aspect_classifier import AspectClassifier, annotate_reviews, weak_label
from ..models.complaint_detector import detect_emerging_complaints
from ..models.decline_predictor import DeclinePredictor, time_based_split
from ..models.recommender import ActionRecommender, StarImpactModel

LOGGER = logging.getLogger("reputation.train")


def load_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(reviews, businesses)`` for whichever data source was requested."""
    if args.source == "synthetic":
        LOGGER.info("Generating synthetic corpus (%d businesses)", args.n_businesses)
        return generate_synthetic_reviews(
            n_businesses=args.n_businesses,
            months=args.months,
            seed=SETTINGS.random_state,
        )

    LOGGER.info("Loading Yelp Open Dataset from %s", args.data_dir)
    businesses = load_yelp_businesses(Path(args.data_dir), city=args.city)
    if businesses.empty:
        raise SystemExit(f"No restaurants found for city={args.city!r}")
    reviews = load_yelp_reviews(
        Path(args.data_dir), business_ids=businesses["business_id"], limit=args.limit
    )
    LOGGER.info("Loaded %d reviews for %d businesses", len(reviews), len(businesses))
    return reviews, businesses


def run(args: argparse.Namespace) -> dict:
    """Execute the full training pipeline and return the metrics dictionary."""
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    reviews, businesses = load_source(args)

    # --- Module 1: aspect classification --------------------------------- #
    LOGGER.info("Fitting aspect classifier on %d reviews", len(reviews))
    classifier = AspectClassifier().fit(reviews["text"])
    annotated = annotate_reviews(reviews, classifier)

    # --- Panel + supervised frame ---------------------------------------- #
    panel = build_monthly_panel(annotated)
    supervised = build_supervised_frame(panel)
    LOGGER.info(
        "Panel: %d business-months, supervised rows: %d (%.1f%% declines)",
        len(panel), len(supervised), 100 * supervised["y_decline"].mean(),
    )

    # --- Module 2: emerging complaints ----------------------------------- #
    alerts = detect_emerging_complaints(panel)
    LOGGER.info("Raised %d historical complaint alerts", len(alerts))

    # --- Module 3: decline prediction ------------------------------------ #
    train, test = time_based_split(supervised)
    predictor = DeclinePredictor().fit(train)
    test_scores = predictor.predict_proba(test)

    metrics = {
        "data": {
            "source": args.source,
            "n_reviews": int(len(reviews)),
            "n_businesses": int(reviews["business_id"].nunique()),
            "n_business_months": int(len(panel)),
            "n_supervised_rows": int(len(supervised)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_period_end": str(train["period"].max().date()),
            "test_period_start": str(test["period"].min().date()),
        },
        "decline_model": classification_report_dict(test["y_decline"], test_scores),
        "baseline_recent_negativity": classification_report_dict(
            test["y_decline"], naive_rating_baseline(test)
        ),
        "baseline_majority": majority_baseline(test["y_decline"]),
        "alerts": {
            "n_alerts": int(len(alerts)),
            "by_severity": alerts["severity"].value_counts().to_dict()
            if not alerts.empty
            else {},
        },
    }

    # Feature importance is expensive, so it is opt-out rather than always on.
    if not args.skip_importance:
        importance = predictor.importance(test)
        metrics["top_features"] = importance.head(12).to_dict(orient="records")

    # Aspect-model sanity check on a held-out sample of reviews.
    sample = reviews.sample(min(2000, len(reviews)), random_state=SETTINGS.random_state)
    metrics["aspect_model"] = aspect_agreement(
        classifier.predict_mentions(sample["text"]), weak_label(sample["text"])
    ).to_dict(orient="records")

    # Ground-truth lead time is only available in synthetic runs.
    if "true_failing_aspect" in businesses.columns and not alerts.empty:
        lead = lead_time_analysis(alerts, businesses)
        metrics["early_warning"] = lead_time_summary(lead)

    # --- Module 4: recommendation ---------------------------------------- #
    star_model = StarImpactModel().fit(panel)
    recommender = ActionRecommender(predictor, star_model).fit_market_reference(supervised)
    metrics["star_impact_coefficients"] = {
        a: round(c, 4) for a, c in star_model.coefficients.items()
    }

    # --- Persist ---------------------------------------------------------- #
    bundle = {
        "aspect_classifier": classifier,
        "decline_predictor": predictor,
        "star_model": star_model,
        "recommender": recommender,
        "annotated_reviews": annotated,
        "panel": panel,
        "supervised": supervised,
        "alerts": alerts,
        "settings": SETTINGS,
    }
    joblib.dump(bundle, artifact_dir / "model_bundle.joblib", compress=3)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    LOGGER.info("Artifacts written to %s", artifact_dir)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the reputation early-warning models")
    parser.add_argument("--source", choices=["synthetic", "yelp"], default="synthetic")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--city", default=None, help="Restrict Yelp data to one city")
    parser.add_argument("--limit", type=int, default=None, help="Cap the reviews loaded")
    parser.add_argument("--n-businesses", type=int, default=120, help="Synthetic only")
    parser.add_argument("--months", type=int, default=36, help="Synthetic only")
    parser.add_argument("--skip-importance", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    metrics = run(build_arg_parser().parse_args())
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
