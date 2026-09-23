"""M5 -- the planners that turn tool output into a ranked action plan.

Two implementations behind one protocol:

``RuleBasedPlanner``   deterministic, no API key, no network. It is both the
                       offline fallback and the **baseline** the LLM planner is
                       measured against in the evaluation -- "does the LLM beat
                       a sensible heuristic?" is the question the proposal's
                       M5 row actually asks.
``ClaudePlanner``      Claude with tool calling. The model decides which tools
                       to call and in what order, then drafts a structured plan
                       whose every claim cites review ids.

Both emit the *same* structure, so the verifier, the API and the dashboard do
not care which one produced a plan, and a run can be A/B'd by swapping one
argument.

The LLM planner is optional: `anthropic` is imported lazily, so the package,
the tests and the container all work without it installed or configured.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

from ..config import ASPECTS
from ..models.recommender import ASPECT_LABELS, PLAYBOOK
from .tools import TOOL_SCHEMAS, AnalyticsToolbox

LOGGER = logging.getLogger("reputation.agents.planner")

# Opus 5 is the default: the plan is short but the reasoning behind the ranking
# is the whole product, and a wrong ranking costs an owner real money.
DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You are ReviewRadar's operations analyst. You advise independent \
restaurant owners on which operational problem to fix first, based only on what the \
analytics tools tell you.

Rules you must follow:
- Call the tools to gather facts. Never assert anything the tools did not return.
- Every plan item must cite the review_ids returned by get_evidence_reviews for that \
aspect. An item with no citation is worthless and will be rejected.
- Any text you put in "quotes" must be copied verbatim from a cited review.
- Every number you state must come from a tool result. Do not round aggressively, \
estimate, or infer numbers the tools did not give you.
- Rank by what will actually help the owner: how much of the predicted risk the aspect \
explains, how fast it is getting worse, and how hard it is to fix. An aspect that is \
high but stable and already at the market median is usually not the first thing to fix.
- Write for a busy owner, not an analyst. Plain sentences, no jargon, no hedging.

A claim verifier checks your output mechanically against the tool results. Claims that \
cannot be traced to a tool result are discarded, so precision matters more than \
fluency."""

# Structured output schema: the plan must be machine-checkable, so the model
# returns JSON rather than prose.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Two sentences for the owner: the situation and the single first action.",
        },
        "items": {
            "type": "array",
            "description": "Ranked problems, most urgent first, at most three.",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "aspect": {"type": "string", "enum": list(ASPECTS)},
                    "claim": {
                        "type": "string",
                        "description": "What is wrong and how you know, including the numbers from the tools.",
                    },
                    "action": {"type": "string", "description": "The concrete first step to take."},
                    "evidence_review_ids": {"type": "array", "items": {"type": "string"}},
                    "quotes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Verbatim snippets from the cited reviews.",
                    },
                },
                "required": ["rank", "aspect", "claim", "action", "evidence_review_ids", "quotes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "items"],
    "additionalProperties": False,
}


class Planner(Protocol):
    """What the agent needs from a planner."""

    name: str

    def plan(self, business_id: str, toolbox: AnalyticsToolbox, context: dict) -> dict:
        """Return ``{"summary": str, "items": [...]}`` from gathered context."""


# --------------------------------------------------------------------------- #
# Deterministic planner (always available; also the LLM's baseline)
# --------------------------------------------------------------------------- #


@dataclass
class RuleBasedPlanner:
    """Ranks by the recommender's priority score and writes templated claims.

    No LLM, so it cannot hallucinate -- it scores 100% groundedness by
    construction. That is exactly what makes it the right baseline: the
    interesting question for M5 is whether the LLM planner can match that
    groundedness while producing something an owner finds more useful.
    """

    name: str = "rule_based"
    top_n: int = 3

    def plan(self, business_id: str, toolbox: AnalyticsToolbox, context: dict) -> dict:
        risk = context["risk"]
        trends = {t["aspect"]: t for t in context["trends"]["trends"]}
        peers = {a["aspect"]: a for a in context["peers"]["aspects"]}

        ranked = sorted(
            risk["contributions"], key=lambda c: c["priority_score"], reverse=True
        )[: self.top_n]

        items = []
        for rank, contribution in enumerate(ranked, start=1):
            aspect = contribution["aspect"]
            trend = trends.get(aspect, {})
            peer = peers.get(aspect, {})
            reviews = context["evidence"].get(aspect, [])

            # Percentages, not raw rates: the owner reads this, not an analyst.
            # The verifier parses these back out and checks them against the
            # tool output, so the wording has to stay numerically faithful.
            median = peer.get("market_median")
            claim = (
                f"{trend.get('complaint_count', 0)} of the last "
                f"{context['trends']['reviews_in_window']} reviews complain about "
                f"{ASPECT_LABELS[aspect]} "
                f"({trend.get('complaint_rate', 0.0) * 100:.1f}% of reviews"
                + (f", against a market median of {median * 100:.1f}%" if median is not None else "")
                + f"). The trend is {trend.get('direction', 'flat')}."
            )
            items.append(
                {
                    "rank": rank,
                    "aspect": aspect,
                    "claim": claim,
                    "action": contribution["suggested_action"] or PLAYBOOK[aspect],
                    "evidence_review_ids": [r["review_id"] for r in reviews[:3]],
                    "quotes": [],   # templated claims quote nothing, so nothing to verify
                }
            )

        lead = items[0] if items else None
        summary = (
            f"There is a {risk['decline_risk'] * 100:.0f}% chance this restaurant's "
            f"rating falls over the next {risk['horizon_months']} months. "
            + (
                f"Start with {ASPECT_LABELS[lead['aspect']]}: {lead['action']}."
                if lead
                else "No aspect stands out above the market benchmark."
            )
        )
        return {"summary": summary, "items": items}


# --------------------------------------------------------------------------- #
# Claude planner
# --------------------------------------------------------------------------- #


@dataclass
class ClaudePlanner:
    """Claude with tool calling, drafting a structured, citable plan.

    The loop is written out rather than delegated to the SDK's tool runner
    because the agent must record *every* tool result it saw: those results are
    the ground truth the verifier checks the plan against. A helper that hides
    the intermediate results would leave nothing to verify against.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 8000
    max_iterations: int = 8
    name: str = "claude"
    client: object | None = None

    def _get_client(self):
        if self.client is not None:
            return self.client
        try:
            import anthropic
        except ImportError as exc:                       # pragma: no cover - env dependent
            raise RuntimeError(
                "The Claude planner needs the Anthropic SDK: pip install anthropic"
            ) from exc
        # A bare client resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or a
        # stored CLI profile, in that order.
        self.client = anthropic.Anthropic()
        return self.client

    def plan(self, business_id: str, toolbox: AnalyticsToolbox, context: dict) -> dict:
        client = self._get_client()

        # The pre-fetched context is handed over as the opening turn so the
        # model starts from facts rather than spending turns rediscovering them;
        # it can still call any tool for anything it wants to check.
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Write the action plan for restaurant {business_id}.\n\n"
                    f"Here is what the analytics modules already report:\n"
                    f"{json.dumps(context['risk'], indent=2)}\n\n"
                    f"{json.dumps(context['trends'], indent=2)}\n\n"
                    f"{json.dumps(context['peers'], indent=2)}\n\n"
                    "Call get_evidence_reviews for any aspect you intend to name, then "
                    "return the ranked plan. Cite the review_ids you were given."
                ),
            }
        ]

        for _ in range(self.max_iterations):
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                tools=TOOL_SCHEMAS,
                messages=messages,
                output_config={"format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
            )

            if response.stop_reason == "refusal":       # pragma: no cover - policy path
                raise RuntimeError(
                    "Claude declined to produce a plan: "
                    f"{getattr(response.stop_details, 'explanation', 'no explanation')}"
                )

            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return self._parse_plan(response)

            # Execute every requested tool and return all results in ONE user
            # message -- splitting them teaches the model to stop calling tools
            # in parallel.
            results = []
            for block in tool_uses:
                try:
                    output = toolbox_dispatch(toolbox, block.name, block.input)
                    # Record what the model saw: this is the verifier's evidence.
                    if block.name == "get_evidence_reviews":
                        context["evidence"][output["aspect"]] = output["reviews"]
                    context["tool_outputs"].append(output)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(output),
                        }
                    )
                except Exception as exc:                # tool errors are data, not crashes
                    LOGGER.warning("Tool %s failed: %s", block.name, exc)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": results})

        raise RuntimeError(
            f"Claude planner did not finish within {self.max_iterations} tool rounds"
        )

    @staticmethod
    def _parse_plan(response) -> dict:
        """Pull the structured plan out of the final message."""
        for block in response.content:
            if block.type == "text":
                try:
                    payload = json.loads(block.text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and "items" in payload:
                    return payload
        raise RuntimeError("Claude returned no parsable plan")


def toolbox_dispatch(toolbox: AnalyticsToolbox, name: str, arguments: dict) -> dict:
    """Map a tool name from the model onto the toolbox method that implements it."""
    handlers = {
        "list_businesses": toolbox.list_businesses,
        "get_aspect_trends": toolbox.get_aspect_trends,
        "get_decline_risk": toolbox.get_decline_risk,
        "get_evidence_reviews": toolbox.get_evidence_reviews,
        "compare_with_peers": toolbox.compare_with_peers,
    }
    if name not in handlers:
        raise KeyError(f"Unknown tool {name!r}")
    return handlers[name](**arguments)


def default_planner() -> Planner:
    """Pick the LLM planner when credentials exist, the rule-based one otherwise.

    Deliberately silent about it in the return value -- the agent records which
    planner ran in the plan itself, so a report can never mistake a fallback
    plan for an LLM one.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return ClaudePlanner()
    LOGGER.info("No Anthropic credentials found; using the rule-based planner")
    return RuleBasedPlanner()
