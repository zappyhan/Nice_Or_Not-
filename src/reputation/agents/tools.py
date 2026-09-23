"""M5 -- the tool surface the agent is allowed to call.

The agent never touches a DataFrame. Everything it knows about a restaurant it
learns by calling one of the four tools below, each of which returns plain
JSON-serialisable data with an explicit provenance field. Three reasons that
boundary matters:

1. **Verifiability.** Every number in the final plan can be traced back to the
   tool call that produced it, which is what makes the claim verifier possible.
2. **Swappability.** The same toolbox backs the rule-based planner and the LLM
   planner, so the two are comparable in the evaluation (M5's "groundedness and
   usefulness versus baseline" row).
3. **Cost.** Tools return compact summaries rather than raw reviews, so a plan
   costs a few thousand tokens instead of a few hundred thousand.

Tools mirror the agent workflow in the proposal: retrieve aspect trends, call
the forecaster for risk and per-aspect contributions, pull representative
negative reviews, and compare the venue with its peers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import ASPECTS, SETTINGS
from ..models.recommender import ASPECT_LABELS


class ToolError(RuntimeError):
    """Raised when a tool is called with arguments that do not resolve.

    Surfaced to the agent as a tool result rather than an exception, so the
    model can correct itself (e.g. retry with a valid business id) instead of
    the whole run dying.
    """


@dataclass
class AnalyticsToolbox:
    """Read-only access to the fitted M2/M3/M4 models for one artifact bundle."""

    bundle: dict

    # ----------------------------------------------------------------- utils
    def _row(self, business_id: str) -> pd.DataFrame:
        """The latest scored business-month for a venue, as a one-row frame."""
        supervised: pd.DataFrame = self.bundle["supervised"]
        rows = supervised[supervised["business_id"] == business_id]
        if rows.empty:
            raise ToolError(
                f"No scored months for business_id={business_id!r}. "
                "Call list_businesses to see the ids that exist."
            )
        return rows.sort_values("period").tail(1)

    # ----------------------------------------------------------------- tools
    def list_businesses(self, limit: int = 10) -> dict:
        """Highest-risk monitored venues. Use this when no business id is known."""
        supervised: pd.DataFrame = self.bundle["supervised"]
        latest = (
            supervised.sort_values("period")
            .groupby("business_id", as_index=False)
            .tail(1)
        )
        risk = self.bundle["decline_predictor"].predict_proba(latest)
        ranked = latest.assign(risk=risk).sort_values("risk", ascending=False).head(limit)
        return {
            "businesses": [
                {"business_id": r.business_id, "decline_risk": round(float(r.risk), 4)}
                for r in ranked.itertuples()
            ]
        }

    def get_aspect_trends(self, business_id: str, months: int = 6) -> dict:
        """Step 1 of the workflow: how each complaint type has moved recently.

        Returns the per-aspect complaint rate over the trailing window, the
        market median for comparison, and the direction of travel.
        """
        panel: pd.DataFrame = self.bundle["panel"]
        history = panel[panel["business_id"] == business_id].sort_values("period")
        if history.empty:
            raise ToolError(f"Unknown business_id={business_id!r}")

        window = history.tail(months)
        earlier = history.tail(months * 2).head(months)
        row = self._row(business_id)

        trends = []
        for aspect in ASPECTS:
            recent = float(window[f"n_complaint_{aspect}"].sum())
            recent_reviews = max(float(window["n_reviews"].sum()), 1.0)
            prior = float(earlier[f"n_complaint_{aspect}"].sum())
            prior_reviews = max(float(earlier["n_reviews"].sum()), 1.0)
            rate, prior_rate = recent / recent_reviews, prior / prior_reviews
            change = rate - prior_rate
            trends.append(
                {
                    "aspect": aspect,
                    "label": ASPECT_LABELS[aspect],
                    "complaint_rate": round(rate, 4),
                    "complaint_count": int(recent),
                    "previous_rate": round(prior_rate, 4),
                    "change": round(change, 4),
                    "direction": "rising" if change > 0.01 else "falling" if change < -0.01 else "flat",
                }
            )

        return {
            "business_id": business_id,
            "period": pd.Timestamp(row["period"].iloc[0]).strftime("%Y-%m"),
            "window_months": months,
            "reviews_in_window": int(window["n_reviews"].sum()),
            "mean_stars_in_window": round(
                float(window["sum_stars"].sum() / max(window["n_reviews"].sum(), 1)), 3
            ),
            "trends": sorted(trends, key=lambda t: t["complaint_rate"], reverse=True),
            "source": "M3 trend detector over the M1 monthly panel",
        }

    def get_decline_risk(self, business_id: str) -> dict:
        """Step 2: the forecaster's probability plus per-aspect contributions.

        Contributions come from the counterfactual intervention described in
        `DeclinePredictor.counterfactual_drop`: how far the predicted risk falls
        if this aspect were at the market median and no longer deteriorating.
        """
        row = self._row(business_id)
        scored = self.bundle["recommender"].explain(row)
        return {
            "business_id": business_id,
            "period": pd.Timestamp(row["period"].iloc[0]).strftime("%Y-%m"),
            "decline_risk": round(float(scored["decline_risk"].iloc[0]), 4),
            "horizon_months": SETTINGS.horizon_months,
            "threshold_stars": SETTINGS.decline_threshold,
            "contributions": [
                {
                    "aspect": r.aspect,
                    "attributed_risk": round(float(r.attributed_risk), 4),
                    "estimated_star_cost": round(float(r.estimated_star_cost), 3),
                    "priority_score": round(float(r.priority_score), 4),
                    "suggested_action": r.suggested_action,
                }
                for r in scored.itertuples()
            ],
            "source": "M4 decline forecaster with counterfactual attribution",
        }

    def get_evidence_reviews(
        self, business_id: str, aspect: str, limit: int = 4, months: int = 6
    ) -> dict:
        """Step 3: the actual reviews behind a complaint, for citation.

        Every returned review carries a ``review_id``. The claim verifier
        requires the planner to cite those ids, which is what stops the agent
        from asserting causes the reviews do not support.
        """
        if aspect not in ASPECTS:
            raise ToolError(f"Unknown aspect {aspect!r}; valid aspects are {list(ASPECTS)}")

        reviews: pd.DataFrame = self.bundle["annotated_reviews"]
        row = self._row(business_id)
        as_of = pd.Timestamp(row["period"].iloc[0])
        window_start = as_of - pd.DateOffset(months=months)

        subset = reviews[
            (reviews["business_id"] == business_id)
            & (reviews["date"] > window_start)
            & (reviews["date"] <= as_of + pd.offsets.MonthEnd(0))
            & (reviews[f"complaint_{aspect}"] == 1)
        ]
        if subset.empty:
            return {
                "business_id": business_id,
                "aspect": aspect,
                "reviews": [],
                "note": "No negative reviews mention this aspect in the window.",
                "source": "M2 aspect classifier over raw review text",
            }

        # Rank by how strongly the classifier believes the review is about this
        # aspect, so a passing mention never outranks a review squarely about it.
        relevance = self.bundle["aspect_classifier"].predict_proba(subset["text"])[aspect]
        subset = subset.assign(_relevance=relevance.to_numpy()).sort_values(
            ["_relevance", "stars"], ascending=[False, True]
        )

        return {
            "business_id": business_id,
            "aspect": aspect,
            "reviews": [
                {
                    "review_id": r.review_id,
                    "date": pd.Timestamp(r.date).strftime("%Y-%m-%d"),
                    "stars": int(r.stars),
                    "text": " ".join(str(r.text).split())[:400],
                }
                for r in subset.head(limit).itertuples()
            ],
            "source": "M2 aspect classifier over raw review text",
        }

    def compare_with_peers(self, business_id: str) -> dict:
        """Step 4: where this venue sits against the rest of the market."""
        supervised: pd.DataFrame = self.bundle["supervised"]
        row = self._row(business_id)
        period = row["period"].iloc[0]
        market = supervised[supervised["period"] == period]

        comparisons = []
        for aspect in ASPECTS:
            column = f"hist_complaint_rate_{aspect}"
            value = float(row[column].iloc[0]) if pd.notna(row[column].iloc[0]) else 0.0
            peers = market[column].dropna()
            percentile = float((peers < value).mean()) if len(peers) else float("nan")
            comparisons.append(
                {
                    "aspect": aspect,
                    "your_rate": round(value, 4),
                    "market_median": round(float(peers.median()), 4) if len(peers) else None,
                    "percentile": round(percentile, 3),
                    "worse_than_market": bool(len(peers) and value > peers.median()),
                }
            )

        stars = row["hist_mean_stars"].iloc[0]
        market_stars = market["hist_mean_stars"].dropna()
        return {
            "business_id": business_id,
            "period": pd.Timestamp(period).strftime("%Y-%m"),
            "peer_count": int(len(market)),
            "your_mean_stars": None if pd.isna(stars) else round(float(stars), 2),
            "market_mean_stars": round(float(market_stars.median()), 2) if len(market_stars) else None,
            "aspects": comparisons,
            "source": "M1 panel, peer comparison within the same month",
        }


# The JSON schemas the LLM planner advertises. Kept beside the implementations
# so a signature change cannot silently drift from what the model is told.
TOOL_SCHEMAS = [
    {
        "name": "get_aspect_trends",
        "description": (
            "Complaint rate per operational aspect over the trailing window, with "
            "the direction of travel. Call this first to see what is moving."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "business_id": {"type": "string", "description": "The venue to analyse"},
                "months": {"type": "integer", "description": "Trailing window, default 6"},
            },
            "required": ["business_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_decline_risk",
        "description": (
            "Predicted probability that the venue's rating falls over the forecast "
            "horizon, plus how much of that risk is attributable to each aspect."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"business_id": {"type": "string"}},
            "required": ["business_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_evidence_reviews",
        "description": (
            "Actual customer reviews complaining about one aspect, each with a "
            "review_id. You MUST cite these ids for any claim about that aspect."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "business_id": {"type": "string"},
                "aspect": {"type": "string", "enum": list(ASPECTS)},
                "limit": {"type": "integer", "description": "Max reviews, default 4"},
            },
            "required": ["business_id", "aspect"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_with_peers",
        "description": (
            "How this venue's complaint rates compare with other restaurants in "
            "the same market and month."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"business_id": {"type": "string"}},
            "required": ["business_id"],
            "additionalProperties": False,
        },
    },
]
