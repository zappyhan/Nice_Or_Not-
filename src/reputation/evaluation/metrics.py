"""Evaluation metrics for Section 4 of the report.

Three families, because the system makes three different kinds of claim:

1. ``classification_report_dict`` -- ranking quality of the decline predictor
   (ROC-AUC, PR-AUC, precision@k, Brier score). PR-AUC and precision@k matter
   most: declines are the minority class and an operator can only act on a
   handful of alerts per week, so performance *at the top of the ranking* is
   the thing being sold.
2. ``lead_time_analysis`` -- how many months before the visible rating drop the
   early-warning alert fired. This is the claim that separates the system from
   a plain sentiment dashboard, so it needs its own metric.
3. ``aspect_agreement`` -- agreement between the weakly supervised aspect model
   and the seed lexicon on held-out reviews, plus support counts, as a sanity
   check on the noisy-label assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import ASPECTS, SETTINGS


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of true declines among the k highest-risk businesses.

    This mirrors the real workflow: a district manager reviews a fixed-size
    worklist each week, not every venue above some probability cut-off.
    """
    k = min(k, len(scores))
    if k == 0:
        return float("nan")
    top = np.argsort(scores)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[top]))


def classification_report_dict(
    y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5, k: int = SETTINGS.top_k_precision
) -> dict[str, float]:
    """Headline metrics for the decline predictor on a held-out slice."""
    y_true = np.asarray(y_true)
    predicted = (scores >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, scores)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        f"precision_at_{k}": precision_at_k(y_true, scores, k),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
    }


def majority_baseline(y_true: np.ndarray) -> dict[str, float]:
    """Trivial baseline ("nothing ever declines") for the comparison table."""
    y_true = np.asarray(y_true)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == 0)),
        "pr_auc": float(np.mean(y_true)),   # a random ranker scores the base rate
    }


def naive_rating_baseline(test: pd.DataFrame) -> np.ndarray:
    """Baseline scorer: rank by *current* negative rate only.

    This is what a conventional star-rating dashboard gives a manager, so it is
    the honest comparison point for "does aspect-level modelling add value?".
    """
    return test["recent_negative_rate"].fillna(0.0).to_numpy()


def lead_time_analysis(
    alerts: pd.DataFrame, businesses: pd.DataFrame
) -> pd.DataFrame:
    """Compare alert timing against known degradation events (synthetic runs).

    ``businesses`` must carry ``true_failing_aspect`` and ``true_event_month``,
    which only the synthetic generator provides -- on real Yelp data the same
    table is produced by labelling a sample of venues by hand.

    Returned columns: business_id, true aspect/month, first matching alert
    month, ``lead_months`` (positive = fired before the event became visible),
    and ``detected``.
    """
    if "true_failing_aspect" not in businesses.columns:
        raise ValueError("lead_time_analysis needs ground-truth event columns")

    truth = businesses.dropna(subset=["true_failing_aspect"]).copy()
    rows = []
    for _, venue in truth.iterrows():
        event = pd.Timestamp(venue["true_event_month"])
        matched = alerts[
            (alerts["business_id"] == venue["business_id"])
            & (alerts["aspect"] == venue["true_failing_aspect"])
        ]
        first = pd.to_datetime(matched["period"]).min() if not matched.empty else pd.NaT
        # The rating itself typically only moves once the horizon window has
        # filled, so "visible" is the event month plus the label horizon.
        visible = event + pd.DateOffset(months=SETTINGS.horizon_months)
        lead = np.nan if pd.isna(first) else (visible.to_period("M") - first.to_period("M")).n
        rows.append(
            {
                "business_id": venue["business_id"],
                "aspect": venue["true_failing_aspect"],
                "event_month": event,
                "first_alert": first,
                "lead_months": lead,
                "detected": bool(pd.notna(first)),
            }
        )
    return pd.DataFrame(rows)


def lead_time_summary(lead_frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate the lead-time table into the numbers quoted in the abstract."""
    detected = lead_frame[lead_frame["detected"]]
    return {
        "n_events": int(len(lead_frame)),
        "detection_rate": float(lead_frame["detected"].mean()) if len(lead_frame) else float("nan"),
        "median_lead_months": float(detected["lead_months"].median()) if len(detected) else float("nan"),
        "share_detected_before_visible": (
            float((detected["lead_months"] > 0).mean()) if len(detected) else float("nan")
        ),
    }


def aspect_agreement(
    classifier_mentions: pd.DataFrame, lexicon_mentions: pd.DataFrame
) -> pd.DataFrame:
    """Per-aspect agreement between the trained model and the seed lexicon.

    Low agreement is not automatically bad -- the point of training a model on
    weak labels is to *generalise beyond* the lexicon -- but a collapse to
    near-zero support signals that the weak labels were too sparse to learn
    from, which is the failure mode to watch for.
    """
    rows = []
    for aspect in ASPECTS:
        model = classifier_mentions[aspect].to_numpy()
        lexicon = lexicon_mentions[aspect].to_numpy()
        rows.append(
            {
                "aspect": aspect,
                "lexicon_support": int(lexicon.sum()),
                "model_support": int(model.sum()),
                "agreement": float(np.mean(model == lexicon)),
                "f1_vs_lexicon": float(f1_score(lexicon, model, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)
