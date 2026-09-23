"""End-to-end tests over the synthetic corpus.

These run in well under a minute on CPU, which is the point: the same test
suite is the CI gate for the container image, so every push proves the whole
pipeline still trains, scores and serves.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reputation.config import ASPECTS, SETTINGS  # noqa: E402
from reputation.data.loader import generate_synthetic_reviews  # noqa: E402
from reputation.evaluation.metrics import (  # noqa: E402
    classification_report_dict,
    lead_time_analysis,
    lead_time_summary,
    precision_at_k,
)
from reputation.features.panel import (  # noqa: E402
    build_monthly_panel,
    build_supervised_frame,
    feature_columns,
)
from reputation.models.aspect_classifier import (  # noqa: E402
    AspectClassifier,
    annotate_reviews,
    weak_label,
)
from reputation.models.complaint_detector import detect_emerging_complaints  # noqa: E402
from reputation.models.decline_predictor import (  # noqa: E402
    DeclinePredictor,
    time_based_split,
)
from reputation.models.recommender import (  # noqa: E402
    ActionRecommender,
    StarImpactModel,
    build_report,
)

# Big enough for the proposal's decline label (P2) to be learnable, small enough
# for CI. The label compares a 3-month future window against a 3-month trailing
# window, so it carries more short-run noise than a longer baseline would: below
# roughly 60 businesses the test slice is too small to measure signal at all.
SMALL = dict(n_businesses=70, months=34, seed=7)


@pytest.fixture(scope="module")
def corpus():
    return generate_synthetic_reviews(**SMALL)


@pytest.fixture(scope="module")
def fitted(corpus):
    """Train every module once and share the result across tests."""
    reviews, businesses = corpus
    classifier = AspectClassifier().fit(reviews["text"])
    annotated = annotate_reviews(reviews, classifier)
    panel = build_monthly_panel(annotated)
    supervised = build_supervised_frame(panel)
    train, test = time_based_split(supervised)
    predictor = DeclinePredictor().fit(train)
    star_model = StarImpactModel().fit(panel)
    recommender = ActionRecommender(predictor, star_model).fit_market_reference(supervised)
    return {
        "reviews": reviews,
        "businesses": businesses,
        "classifier": classifier,
        "annotated": annotated,
        "panel": panel,
        "supervised": supervised,
        "train": train,
        "test": test,
        "predictor": predictor,
        "recommender": recommender,
    }


# --------------------------------------------------------------------------- #
# Data + features
# --------------------------------------------------------------------------- #


def test_synthetic_corpus_schema(corpus):
    reviews, businesses = corpus
    assert set(reviews.columns) == {"review_id", "business_id", "date", "stars", "text"}
    assert reviews["stars"].between(1, 5).all()
    assert reviews["review_id"].is_unique
    # Some venues must be degrading, otherwise the evaluation has no positives.
    assert businesses["true_failing_aspect"].notna().any()


def test_weak_labels_are_multi_label(corpus):
    reviews, _ = corpus
    labels = weak_label(reviews["text"].head(500))
    assert list(labels.columns) == list(ASPECTS)
    assert labels.to_numpy().sum(axis=1).max() >= 2


def test_panel_has_dense_calendar(fitted):
    panel = fitted["panel"]
    per_business = panel.groupby("business_id")["period"].nunique()
    # Every business spans the same dense month index.
    assert per_business.nunique() == 1


def test_supervised_frame_has_no_label_leakage(fitted):
    """Future columns must never reach the model's feature list."""
    features = feature_columns(fitted["supervised"])
    for leaked in ("future_mean_stars", "future_reviews", "star_change", "y_decline"):
        assert leaked not in features
    assert len(features) > 15


def test_time_split_is_chronological(fitted):
    train, test = fitted["train"], fitted["test"]
    assert train["period"].max() < test["period"].min()


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def test_aspect_classifier_detects_obvious_complaints(fitted):
    classifier = fitted["classifier"]
    probabilities = classifier.predict_proba(
        pd.Series(
            [
                "the toilet was filthy and there were flies everywhere",
                "we waited 45 minutes for cold food",
            ]
        )
    )
    assert probabilities.iloc[0]["cleanliness"] > 0.5
    assert probabilities.iloc[1][["wait_time", "food_quality"]].max() > 0.5


def test_complaint_detector_fires_after_a_real_event(fitted):
    """Alerts must concentrate on the venues that actually degraded."""
    alerts = detect_emerging_complaints(fitted["panel"])
    assert not alerts.empty

    truth = fitted["businesses"].dropna(subset=["true_failing_aspect"])
    lead = lead_time_analysis(alerts, truth)
    summary = lead_time_summary(lead)
    # The detector should catch the clear majority of injected events.
    assert summary["detection_rate"] >= 0.7


def test_decline_predictor_beats_chance(fitted):
    test = fitted["test"]
    scores = fitted["predictor"].predict_proba(test)
    assert len(scores) == len(test)
    assert ((scores >= 0) & (scores <= 1)).all()

    report = classification_report_dict(test["y_decline"], scores)
    assert report["roc_auc"] > 0.55, f"model no better than chance: {report}"
    # Precision at the top of the ranking must beat the base rate, since that
    # is the operating point a manager actually uses.
    assert report[f"precision_at_{SETTINGS.top_k_precision}"] >= report["positive_rate"]


def test_precision_at_k_matches_manual_calculation():
    y_true = [0, 1, 1, 0, 1]
    scores = [0.1, 0.9, 0.8, 0.7, 0.2]
    assert precision_at_k(y_true, scores, k=2) == 1.0
    # Top-4 by score are 0.9(1), 0.8(1), 0.7(0), 0.2(1) -> 3 of 4 correct.
    assert precision_at_k(y_true, scores, k=4) == pytest.approx(0.75, abs=1e-6)


def test_counterfactual_attribution_responds_to_the_feature(fitted):
    """Patching an aspect's features must actually move the prediction."""
    supervised, predictor = fitted["supervised"], fitted["predictor"]
    worst = supervised.sort_values("hist_complaint_rate_service").tail(20)

    drops = [
        predictor.counterfactual_drop(
            worst.iloc[[i]],
            {
                "hist_complaint_rate_service": 0.0,
                "recent_complaint_rate_service": 0.0,
                "delta_complaint_rate_service": 0.0,
            },
        )
        for i in range(len(worst))
    ]
    # Isotonic calibration makes the output a step function, so an individual
    # row can legitimately sit inside a flat segment and not move at all. What
    # must not happen is the whole feature group being ignored.
    assert any(abs(d) > 1e-6 for d in drops)


# --------------------------------------------------------------------------- #
# Reporting contract (this is what the API serves)
# --------------------------------------------------------------------------- #


def test_report_contract(fitted):
    supervised = fitted["supervised"]
    row = supervised.sort_values("period").tail(1)
    report = build_report(
        row,
        fitted["recommender"],
        fitted["annotated"],
        classifier=fitted["classifier"],
    )

    assert set(report) >= {
        "business_id", "period", "decline_risk", "risk_band", "summary", "findings",
    }
    assert 0.0 <= report["decline_risk"] <= 1.0
    assert report["risk_band"] in {"low", "medium", "high"}
    assert 1 <= len(report["findings"]) <= 3
    # Findings arrive ranked by priority, so the first one is the recommendation.
    scores = [f["priority_score"] for f in report["findings"]]
    assert scores == sorted(scores, reverse=True)
    for finding in report["findings"]:
        assert finding["aspect"] in ASPECTS
        assert finding["suggested_action"]
        assert len(finding["evidence"]) == len(set(finding["evidence"]))


def test_settings_are_immutable_but_overridable():
    """Experiments override settings via replace(), never by mutating globals."""
    with pytest.raises(Exception):
        SETTINGS.horizon_months = 99  # type: ignore[misc]
    assert replace(SETTINGS, horizon_months=6).horizon_months == 6
