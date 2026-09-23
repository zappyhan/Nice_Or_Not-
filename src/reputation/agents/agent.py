"""M5 -- the agent that runs the proposal's five-step workflow.

    1. retrieve the venue's aspect trends            (M3)
    2. call the forecaster for risk and contributions (M4)
    3. pull the most representative negative reviews  (M2)
    4. compare the venue with similar restaurants     (M1 panel)
    5. write a ranked plan -- then verify every claim against 1-4

Step 5's verification is what separates this from a chatbot summarising a
dashboard: the plan that reaches the owner contains only claims that survived
a mechanical check against the tool outputs. Rejected items are not silently
dropped either -- they are returned alongside the reason, so the failure mode
is visible in the evaluation instead of hidden.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .planner import Planner, RuleBasedPlanner, default_planner
from .tools import AnalyticsToolbox
from .verifier import ClaimVerifier, VerifiedItem, collect_facts, groundedness

LOGGER = logging.getLogger("reputation.agents")


@dataclass
class ReviewRadarAgent:
    """Orchestrates tools, planner and verifier for one restaurant."""

    bundle: dict
    planner: Planner | None = None
    evidence_limit: int = 4
    candidate_aspects: int = 3

    def __post_init__(self) -> None:
        self.toolbox = AnalyticsToolbox(self.bundle)
        if self.planner is None:
            self.planner = default_planner()

    def gather(self, business_id: str) -> dict:
        """Steps 1-4: everything the planner is allowed to reason from.

        Evidence is pre-fetched for the highest-priority aspects so that even a
        planner which calls no tools of its own has citable reviews to work
        with -- and so the verifier has a corpus to check quotes against.
        """
        trends = self.toolbox.get_aspect_trends(business_id)
        risk = self.toolbox.get_decline_risk(business_id)
        peers = self.toolbox.compare_with_peers(business_id)

        top = sorted(
            risk["contributions"], key=lambda c: c["priority_score"], reverse=True
        )[: self.candidate_aspects]

        evidence: dict[str, list[dict]] = {}
        evidence_outputs = []
        for contribution in top:
            output = self.toolbox.get_evidence_reviews(
                business_id, contribution["aspect"], limit=self.evidence_limit
            )
            evidence[output["aspect"]] = output["reviews"]
            evidence_outputs.append(output)

        return {
            "business_id": business_id,
            "trends": trends,
            "risk": risk,
            "peers": peers,
            "evidence": evidence,
            # Every tool result the planner may quote from, for the verifier.
            "tool_outputs": [trends, risk, peers, *evidence_outputs],
        }

    def run(self, business_id: str) -> dict:
        """Produce a verified action plan for one venue."""
        started = time.perf_counter()
        context = self.gather(business_id)

        drafted = self.planner.plan(business_id, self.toolbox, context)
        draft_items = drafted.get("items", [])

        verifier = ClaimVerifier(
            evidence=context["evidence"], facts=collect_facts(context["tool_outputs"])
        )
        verified: list[VerifiedItem] = verifier.verify(draft_items)
        accepted = [v for v in verified if v.accepted]
        rejected = [v for v in verified if not v.accepted]

        for item in rejected:
            LOGGER.warning(
                "Rejected plan item (%s): %s",
                item.item.get("aspect", "?"),
                "; ".join(item.failures),
            )

        return {
            "business_id": business_id,
            "period": context["risk"]["period"],
            "planner": self.planner.name,
            "decline_risk": context["risk"]["decline_risk"],
            "summary": drafted.get("summary", ""),
            # Only verified items are shown to the owner, re-ranked so the
            # numbering stays contiguous after any rejection.
            "plan": [
                {**item.as_dict(), "rank": i}
                for i, item in enumerate(accepted, start=1)
            ],
            "rejected": [item.as_dict() for item in rejected],
            "groundedness": round(groundedness(verified), 4) if verified else None,
            "n_drafted": len(draft_items),
            "n_accepted": len(accepted),
            "tools_called": len(context["tool_outputs"]),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


def compare_planners(bundle: dict, business_id: str, planners: list[Planner]) -> list[dict]:
    """Run the same venue through several planners for the M5 comparison table.

    This is the experiment the proposal's M5 evaluation row describes: the LLM
    planner against the rule-based baseline on groundedness, latency and the
    number of usable items produced.
    """
    results = []
    for planner in planners:
        agent = ReviewRadarAgent(bundle, planner=planner)
        try:
            plan = agent.run(business_id)
        except Exception as exc:                        # one planner failing is data
            LOGGER.error("Planner %s failed: %s", getattr(planner, "name", planner), exc)
            results.append({"planner": getattr(planner, "name", "?"), "error": str(exc)})
            continue
        results.append(
            {
                "planner": plan["planner"],
                "groundedness": plan["groundedness"],
                "n_drafted": plan["n_drafted"],
                "n_accepted": plan["n_accepted"],
                "latency_seconds": plan["latency_seconds"],
                "summary": plan["summary"],
            }
        )
    return results


def build_agent(bundle: dict, planner_name: str = "auto") -> ReviewRadarAgent:
    """Construct an agent by planner name (``auto``, ``claude`` or ``rule_based``)."""
    if planner_name == "rule_based":
        return ReviewRadarAgent(bundle, planner=RuleBasedPlanner())
    if planner_name == "claude":
        from .planner import ClaudePlanner

        return ReviewRadarAgent(bundle, planner=ClaudePlanner())
    return ReviewRadarAgent(bundle, planner=default_planner())
