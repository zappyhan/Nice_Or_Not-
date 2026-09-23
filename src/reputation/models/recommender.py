"""Module 4 -- Cause summarisation and action prioritisation.

This is the module that turns analysis into a decision. For one business at one
month it answers three questions:

    What is going wrong?      -> ranked aspects + representative review quotes
    How much does it cost?    -> attributed share of predicted decline risk and
                                 an estimated star impact
    What do I fix first?      -> a priority score that trades impact against
                                 how hard the lever is to move

Attribution uses a *counterfactual intervention on the model*: for each aspect
we ask the fitted decline predictor what the venue's risk would be if its
complaint rate for that aspect sat at the market median instead of its actual
value. The drop in predicted probability is that aspect's attributed risk.
This is an associational, model-based attribution -- not a causal effect -- and
the report states that limitation explicitly. What it buys is a ranking that
respects interactions the model has learned, rather than simply sorting by raw
complaint counts (which always promotes whichever aspect people mention most).

The star-impact estimate comes from a separate, deliberately simple ridge
regression of monthly mean stars on complaint rates. Its coefficients are
readable by a non-technical client ("each 10pp of cleanliness complaints costs
about 0.2 stars"), which is what makes the recommendation persuasive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ..config import ASPECTS, SETTINGS, Settings
from .decline_predictor import DeclinePredictor

ASPECT_LABELS = {
    "service": "front-of-house service",
    "food_quality": "food quality and consistency",
    "cleanliness": "cleanliness and hygiene",
    "price_value": "price / value perception",
    "ambience": "ambience, noise and comfort",
    "wait_time": "waiting and queue time",
}

PLAYBOOK = {
    "service": "brief and re-train floor staff, review shift rosters at peak hours",
    "food_quality": "audit recipes and supplier consistency, re-check portioning",
    "cleanliness": "reinstate hourly cleaning checklists and restroom audits",
    "price_value": "revisit portion-to-price ratio or introduce a value set menu",
    "ambience": "address noise and seating comfort; review lighting and layout",
    "wait_time": "re-sequence kitchen tickets and add staff to the peak window",
}


class StarImpactModel:
    """Ridge regression: monthly mean stars ~ per-aspect complaint rates.

    Fitted across the whole market (all businesses, all months) so that the
    coefficients describe the typical cost of a complaint type rather than one
    venue's idiosyncrasies. Ridge rather than OLS because the aspect rates are
    correlated -- unpenalised coefficients flip sign and become unusable for
    client-facing explanation.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.columns = [f"complaint_rate_{a}" for a in ASPECTS]

    def fit(self, panel: pd.DataFrame) -> "StarImpactModel":
        usable = panel.dropna(subset=self.columns + ["mean_stars"])
        # Weight by review volume: a month with 40 reviews is stronger evidence
        # about the rating than a month with 2.
        self.model.fit(
            usable[self.columns],
            usable["mean_stars"],
            sample_weight=usable["n_reviews"],
        )
        return self

    @property
    def coefficients(self) -> dict[str, float]:
        """Star change per unit (0 -> 1) of complaint rate, by aspect."""
        return {a: float(c) for a, c in zip(ASPECTS, self.model.coef_)}

    def star_cost(self, aspect: str, excess_rate: float) -> float:
        """Estimated stars lost to the *excess* complaint rate for one aspect."""
        return float(-self.coefficients[aspect] * max(excess_rate, 0.0))


@dataclass
class ActionRecommender:
    """Ranks the operational problems a venue should address first."""

    predictor: DeclinePredictor
    star_model: StarImpactModel
    settings: Settings = SETTINGS
    market_median: dict[str, float] = field(default_factory=dict)

    def fit_market_reference(self, supervised: pd.DataFrame) -> "ActionRecommender":
        """Store the market-median complaint rate used as the counterfactual."""
        self.market_median = {
            aspect: float(
                supervised[f"hist_complaint_rate_{aspect}"].median(skipna=True)
            )
            for aspect in ASPECTS
        }
        return self

    def explain(self, row: pd.DataFrame) -> pd.DataFrame:
        """Score every aspect for a single business-month row of the panel.

        ``row`` must be a one-row DataFrame taken from the supervised frame.
        Returns one line per aspect with the attributed risk, estimated star
        cost, effort-adjusted priority and a suggested first action.
        """
        if len(row) != 1:
            raise ValueError("explain() expects exactly one business-month row")

        risk = float(self.predictor.predict_proba(row)[0])
        records = []
        for aspect in ASPECTS:
            column = f"hist_complaint_rate_{aspect}"
            actual = float(row[column].iloc[0]) if pd.notna(row[column].iloc[0]) else 0.0
            benchmark = self.market_median.get(aspect, 0.0)
            excess = max(actual - benchmark, 0.0)

            # Counterfactual: what if this aspect were merely average *and*
            # no longer deteriorating? All three features describing the aspect
            # move together (see DeclinePredictor.counterfactual_drop).
            attributed = self.predictor.counterfactual_drop(
                row,
                {
                    f"hist_complaint_rate_{aspect}": benchmark,
                    f"recent_complaint_rate_{aspect}": benchmark,
                    f"delta_complaint_rate_{aspect}": 0.0,
                    f"peer_hist_complaint_rate_{aspect}": 0.0,
                },
            )
            star_cost = self.star_model.star_cost(aspect, excess)
            effort = self.settings.effort_weights.get(aspect, 1.0)

            records.append(
                {
                    "aspect": aspect,
                    "label": ASPECT_LABELS[aspect],
                    "complaint_rate": actual,
                    "market_median": benchmark,
                    "excess_rate": excess,
                    "trend": float(row[f"delta_complaint_rate_{aspect}"].iloc[0])
                    if pd.notna(row[f"delta_complaint_rate_{aspect}"].iloc[0])
                    else 0.0,
                    "attributed_risk": max(attributed, 0.0),
                    "estimated_star_cost": star_cost,
                    "effort_weight": effort,
                    # Priority mixes both value signals, then discounts by effort.
                    # The 0.5 weight on star cost puts the two terms on a
                    # comparable scale (risk is a probability in [0,1], star
                    # cost is in stars, typically 0-1).
                    "priority_score": (
                        max(attributed, 0.0) + 0.5 * star_cost
                    ) / effort,
                    "suggested_action": PLAYBOOK[aspect],
                }
            )

        frame = pd.DataFrame(records).sort_values("priority_score", ascending=False)
        frame["decline_risk"] = risk
        return frame.reset_index(drop=True)


def representative_quotes(
    annotated_reviews: pd.DataFrame,
    business_id: str,
    aspect: str,
    as_of: pd.Timestamp,
    months: int = 3,
    limit: int = 3,
    classifier=None,
) -> list[str]:
    """Pull the most useful verbatim complaints to show alongside a finding.

    Managers trust a number far more when three real sentences sit next to it,
    so the report/API always ships evidence with every recommendation. We take
    the lowest-rated recent reviews that the aspect model flagged, trimmed to a
    readable length.
    """
    window_start = pd.Timestamp(as_of) - pd.DateOffset(months=months)
    subset = annotated_reviews[
        (annotated_reviews["business_id"] == business_id)
        & (annotated_reviews["date"] > window_start)
        & (annotated_reviews["date"] <= pd.Timestamp(as_of) + pd.offsets.MonthEnd(0))
        & (annotated_reviews[f"complaint_{aspect}"] == 1)
    ]
    if subset.empty:
        return []

    if classifier is not None:
        # Rank by how strongly the model believes the review is about *this*
        # aspect. Without this, a review that merely brushes past the aspect
        # ("quick meal") can outrank a review squarely about it, because the
        # binary complaint flag treats both the same.
        subset = subset.assign(
            _relevance=classifier.predict_proba(subset["text"])[aspect].to_numpy()
        ).sort_values(["_relevance", "stars"], ascending=[False, True])
    else:
        subset = subset.sort_values(["stars", "date"], ascending=[True, False])

    # De-duplicate: chains and template-like reviews repeat the same sentence,
    # and three identical quotes in a report look like a bug to the client.
    quotes: list[str] = []
    for text in subset["text"]:
        quote = " ".join(str(text).split())[:220]
        if quote not in quotes:
            quotes.append(quote)
        if len(quotes) == limit:
            break
    return quotes


def build_report(
    row: pd.DataFrame,
    recommender: ActionRecommender,
    annotated_reviews: pd.DataFrame,
    alerts: pd.DataFrame | None = None,
    top_n: int = 3,
    classifier=None,
) -> dict:
    """Assemble the client-facing early-warning report for one business-month.

    The dictionary returned here is exactly what the FastAPI endpoint serves and
    what the demo dashboard renders, so the contract lives in one place.
    """
    business_id = str(row["business_id"].iloc[0])
    period = pd.Timestamp(row["period"].iloc[0])
    scored = recommender.explain(row)
    risk = float(scored["decline_risk"].iloc[0])

    findings = []
    for _, item in scored.head(top_n).iterrows():
        findings.append(
            {
                "aspect": item["aspect"],
                "label": item["label"],
                "complaint_rate": round(item["complaint_rate"], 4),
                "market_median": round(item["market_median"], 4),
                "trend": round(item["trend"], 4),
                "attributed_risk": round(item["attributed_risk"], 4),
                "estimated_star_cost": round(item["estimated_star_cost"], 3),
                "priority_score": round(item["priority_score"], 4),
                "suggested_action": item["suggested_action"],
                "evidence": representative_quotes(
                    annotated_reviews,
                    business_id,
                    item["aspect"],
                    period,
                    classifier=classifier,
                ),
            }
        )

    if alerts is not None and not alerts.empty:
        active = alerts[
            (alerts["business_id"] == business_id)
            & (alerts["period"] == period.strftime("%Y-%m"))
        ].to_dict(orient="records")
    else:
        active = []

    top = findings[0] if findings else None
    summary = (
        f"{business_id}: {risk:.0%} probability of a rating decline over the next "
        f"{SETTINGS.horizon_months} months. "
        + (
            f"Largest attributable driver is {top['label']} "
            f"({top['complaint_rate']:.0%} of recent reviews complain, market median "
            f"{top['market_median']:.0%}). First action: {top['suggested_action']}."
            if top
            else "No aspect stands out above the market benchmark."
        )
    )

    return {
        "business_id": business_id,
        "period": period.strftime("%Y-%m"),
        "decline_risk": round(risk, 4),
        "risk_band": "high" if risk >= 0.6 else "medium" if risk >= 0.35 else "low",
        "summary": summary,
        "findings": findings,
        "active_alerts": active,
    }
