"""M5 tests: the tool surface, the planners, and above all the claim verifier.

The verifier is the component the proposal's trust argument rests on, so most
of this file is adversarial: a planner that fabricates review ids, numbers and
quotes must have every one of those claims rejected. A verifier that only ever
sees well-behaved input has not been tested.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reputation.agents import (  # noqa: E402
    AnalyticsToolbox,
    ClaimVerifier,
    ReviewRadarAgent,
    RuleBasedPlanner,
    ToolError,
    compare_planners,
)
from reputation.agents.planner import ClaudePlanner  # noqa: E402
from reputation.agents.verifier import collect_facts, groundedness  # noqa: E402
from reputation.config import ASPECTS  # noqa: E402
from reputation.data.loader import generate_synthetic_reviews  # noqa: E402
from reputation.features.panel import build_monthly_panel, build_supervised_frame  # noqa: E402
from reputation.models.aspect_classifier import AspectClassifier, annotate_reviews  # noqa: E402
from reputation.models.decline_predictor import DeclinePredictor, time_based_split  # noqa: E402
from reputation.models.recommender import ActionRecommender, StarImpactModel  # noqa: E402


@pytest.fixture(scope="module")
def bundle():
    """A small fitted bundle, the same shape the training pipeline writes."""
    reviews, businesses = generate_synthetic_reviews(n_businesses=30, months=30, seed=5)
    classifier = AspectClassifier().fit(reviews["text"])
    annotated = annotate_reviews(reviews, classifier)
    panel = build_monthly_panel(annotated)
    supervised = build_supervised_frame(panel)
    train, _ = time_based_split(supervised)
    predictor = DeclinePredictor().fit(train)
    star_model = StarImpactModel().fit(panel)
    recommender = ActionRecommender(predictor, star_model).fit_market_reference(supervised)
    return {
        "aspect_classifier": classifier,
        "decline_predictor": predictor,
        "star_model": star_model,
        "recommender": recommender,
        "businesses": businesses,
        "annotated_reviews": annotated,
        "panel": panel,
        "supervised": supervised,
        "alerts": None,
    }


@pytest.fixture(scope="module")
def business_id(bundle):
    return bundle["supervised"]["business_id"].iloc[0]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def test_tools_return_serialisable_facts_with_provenance(bundle, business_id):
    toolbox = AnalyticsToolbox(bundle)

    trends = toolbox.get_aspect_trends(business_id)
    assert len(trends["trends"]) == len(ASPECTS)
    assert trends["source"]

    risk = toolbox.get_decline_risk(business_id)
    assert 0.0 <= risk["decline_risk"] <= 1.0
    assert len(risk["contributions"]) == len(ASPECTS)

    peers = toolbox.compare_with_peers(business_id)
    assert peers["peer_count"] >= 1
    assert all("percentile" in a for a in peers["aspects"])


def test_evidence_tool_returns_citable_reviews(bundle, business_id):
    toolbox = AnalyticsToolbox(bundle)
    risk = toolbox.get_decline_risk(business_id)
    top = max(risk["contributions"], key=lambda c: c["priority_score"])["aspect"]

    evidence = toolbox.get_evidence_reviews(business_id, top)
    for review in evidence["reviews"]:
        # Ids are what the verifier checks citations against.
        assert review["review_id"]
        assert 1 <= review["stars"] <= 5
        assert review["text"]


def test_tools_reject_unknown_arguments(bundle, business_id):
    toolbox = AnalyticsToolbox(bundle)
    with pytest.raises(ToolError):
        toolbox.get_decline_risk("not-a-real-business")
    with pytest.raises(ToolError):
        toolbox.get_evidence_reviews(business_id, "atmosphere_vibes")


# --------------------------------------------------------------------------- #
# The verifier -- adversarial
# --------------------------------------------------------------------------- #


@dataclass
class HallucinatingPlanner:
    """A planner that invents everything the verifier is supposed to catch."""

    name: str = "hallucinating"

    def plan(self, business_id, toolbox, context):
        real_aspect = next(iter(context["evidence"]))
        real_ids = [r["review_id"] for r in context["evidence"][real_aspect]]
        return {
            "summary": "Fabricated plan used to test the verifier.",
            "items": [
                {   # fabricated citation
                    "rank": 1,
                    "aspect": real_aspect,
                    "claim": "Customers are unhappy.",
                    "action": "Do something",
                    "evidence_review_ids": ["totally_made_up_id"],
                    "quotes": [],
                },
                {   # real citation, invented quote
                    "rank": 2,
                    "aspect": real_aspect,
                    "claim": "Customers are unhappy.",
                    "action": "Do something",
                    "evidence_review_ids": real_ids[:1],
                    "quotes": ["the chef personally insulted my grandmother"],
                },
                {   # real citation, fabricated statistic
                    "rank": 3,
                    "aspect": real_aspect,
                    "claim": "A full 97.3% of reviews complain about this.",
                    "action": "Do something",
                    "evidence_review_ids": real_ids[:1],
                    "quotes": [],
                },
                {   # no citation at all
                    "rank": 4,
                    "aspect": real_aspect,
                    "claim": "Trust me on this one.",
                    "action": "Do something",
                    "evidence_review_ids": [],
                    "quotes": [],
                },
            ],
        }


def test_verifier_rejects_every_kind_of_fabrication(bundle, business_id):
    agent = ReviewRadarAgent(bundle, planner=HallucinatingPlanner())
    result = agent.run(business_id)

    assert result["n_drafted"] == 4
    assert result["n_accepted"] == 0, "fabricated claims reached the owner"
    assert result["plan"] == []
    assert result["groundedness"] == 0.0

    failures = " ".join(" ".join(item["checks"][i]["detail"] for i in range(3))
                        for item in result["rejected"])
    assert "not returned by the evidence tool" in failures   # fake id
    assert "not found verbatim" in failures                  # fake quote
    assert "do not match any tool output" in failures        # fake number
    assert "must cite evidence" in failures                  # no citation


def test_verifier_accepts_a_claim_backed_by_the_tools():
    evidence = {
        "cleanliness": [
            {"review_id": "r1", "text": "the toilet was filthy and the floor was sticky"},
            {"review_id": "r2", "text": "found a hair in my soup"},
        ]
    }
    verifier = ClaimVerifier(evidence=evidence, facts=[0.32, 12.0, 0.04])
    verified = verifier.verify(
        [
            {
                "aspect": "cleanliness",
                "claim": "32% of recent reviews mention hygiene, against a market median of 4%.",
                "evidence_review_ids": ["r1", "r2"],
                "quotes": ["the toilet was filthy"],
            }
        ]
    )
    assert verified[0].accepted, verified[0].failures
    assert groundedness(verified) == 1.0


def test_verifier_tolerates_rounding_but_not_invention():
    evidence = {"service": [{"review_id": "r1", "text": "the waiter was rude"}]}
    verifier = ClaimVerifier(evidence=evidence, facts=[0.174])

    rounded = verifier.verify(
        [{"aspect": "service", "claim": "17% of reviews complain.",
          "evidence_review_ids": ["r1"], "quotes": []}]
    )
    assert rounded[0].accepted, "a correctly rounded number must survive"

    invented = verifier.verify(
        [{"aspect": "service", "claim": "62% of reviews complain.",
          "evidence_review_ids": ["r1"], "quotes": []}]
    )
    assert not invented[0].accepted, "an invented number must be rejected"


def test_collect_facts_walks_nested_tool_output():
    facts = collect_facts([{"a": 1, "b": {"c": [2, 3]}, "d": "text", "e": True}])
    assert sorted(facts) == [1.0, 2.0, 3.0]   # booleans are not numbers here


# --------------------------------------------------------------------------- #
# Planners
# --------------------------------------------------------------------------- #


def test_rule_based_planner_is_grounded_by_construction(bundle, business_id):
    agent = ReviewRadarAgent(bundle, planner=RuleBasedPlanner())
    result = agent.run(business_id)

    assert result["planner"] == "rule_based"
    assert result["groundedness"] == 1.0
    assert result["n_accepted"] == result["n_drafted"] >= 1
    assert result["summary"]
    for i, item in enumerate(result["plan"], start=1):
        assert item["rank"] == i          # ranks stay contiguous
        assert item["aspect"] in ASPECTS
        assert item["evidence_review_ids"]
        assert item["action"]


class FakeAnthropicClient:
    """Stands in for the Anthropic SDK: one tool round, then a structured plan.

    Lets the tool-calling loop, the evidence recording and the plan parsing be
    tested without a network call or an API key.
    """

    def __init__(self, aspect: str, review_ids: list[str], quote: str):
        self.aspect, self.review_ids, self.quote = aspect, review_ids, quote
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _FakeResponse(
                stop_reason="tool_use",
                content=[
                    _FakeToolUse(
                        id="call_1",
                        name="get_evidence_reviews",
                        input={
                            "business_id": self._business_id(kwargs),
                            "aspect": self.aspect,
                        },
                    )
                ],
            )
        plan = {
            "summary": "Fix the top issue first.",
            "items": [
                {
                    "rank": 1,
                    "aspect": self.aspect,
                    "claim": "Recent reviews complain about this repeatedly.",
                    "action": "Address it this week",
                    "evidence_review_ids": self.review_ids[:1],
                    "quotes": [self.quote],
                }
            ],
        }
        import json as _json

        return _FakeResponse(stop_reason="end_turn", content=[_FakeText(_json.dumps(plan))])

    @staticmethod
    def _business_id(kwargs) -> str:
        import re

        match = re.search(r"restaurant (\S+)", kwargs["messages"][0]["content"])
        return match.group(1) if match else ""


@dataclass
class _FakeResponse:
    stop_reason: str
    content: list
    stop_details: object = None


@dataclass
class _FakeText:
    text: str
    type: str = "text"


@dataclass
class _FakeToolUse:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


def test_claude_planner_loop_executes_tools_and_returns_a_verified_plan(bundle, business_id):
    toolbox = AnalyticsToolbox(bundle)
    risk = toolbox.get_decline_risk(business_id)
    aspect = max(risk["contributions"], key=lambda c: c["priority_score"])["aspect"]
    evidence = toolbox.get_evidence_reviews(business_id, aspect)
    if not evidence["reviews"]:
        pytest.skip("no evidence reviews for this venue/aspect in the fixture")

    review = evidence["reviews"][0]
    quote = " ".join(review["text"].split()[:5])

    planner = ClaudePlanner(
        client=FakeAnthropicClient(aspect, [review["review_id"]], quote)
    )
    result = ReviewRadarAgent(bundle, planner=planner).run(business_id)

    assert result["planner"] == "claude"
    assert planner.client.calls == 2, "the loop must run the tool round then draft"
    assert result["n_accepted"] == 1
    assert result["plan"][0]["quotes"] == [quote]
    assert result["groundedness"] == 1.0


def test_compare_planners_produces_the_m5_evaluation_row(bundle, business_id):
    rows = compare_planners(bundle, business_id, [RuleBasedPlanner(), HallucinatingPlanner()])
    assert {r["planner"] for r in rows} == {"rule_based", "hallucinating"}
    by_name = {r["planner"]: r for r in rows}
    assert by_name["rule_based"]["groundedness"] == 1.0
    assert by_name["hallucinating"]["groundedness"] == 0.0
    assert all("latency_seconds" in r for r in rows)
