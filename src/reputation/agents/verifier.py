"""M5 -- the claim verifier.

The proposal's trust requirement: *"A verifier step will reject any agent claim
that is not backed by a retrieved review."* This module is that step, and it is
deliberately mechanical rather than another LLM call -- an LLM grading its own
output is not independent evidence, and a check that can itself hallucinate is
not a check.

Every plan item is verified on three independent axes:

``citation``   every cited review_id must appear in what the evidence tool
               actually returned for that aspect
``quotation``  any quoted text must appear verbatim in a cited review
``statistic``  every number the claim asserts must match the tool output that
               produced it, within a stated tolerance

An item failing ``citation`` or ``quotation`` is **rejected** -- it never
reaches the owner. A statistic mismatch is rejected too, because a plausible
plan with a wrong number is more dangerous than no plan. Items that pass are
marked verified, and the plan carries a groundedness score (the share of items
that survived), which is the M5 evaluation metric in the proposal's Table 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbers the model might quote back: "52%", "0.52", "3 reviews", "0.95 stars".
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")

# Percentages are quoted rounded, so allow a point of slack; absolute counts
# and star values must be much closer.
PERCENT_TOLERANCE = 0.015
VALUE_TOLERANCE = 0.05


@dataclass
class CheckResult:
    """One verification check applied to one claim."""

    axis: str            # citation | quotation | statistic
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return {"axis": self.axis, "passed": self.passed, "detail": self.detail}


@dataclass
class VerifiedItem:
    """A plan item plus the verdict of every check run against it."""

    item: dict
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[str]:
        return [c.detail for c in self.checks if not c.passed]

    def as_dict(self) -> dict:
        return {
            **self.item,
            "verified": self.accepted,
            "checks": [c.as_dict() for c in self.checks],
        }


def _extract_numbers(text: str) -> list[tuple[float, bool]]:
    """Return ``(value, is_percentage)`` for every number in a claim."""
    found = []
    for token in _NUMBER_RE.findall(text or ""):
        if token.endswith("%"):
            found.append((float(token[:-1]) / 100.0, True))
        else:
            found.append((float(token), False))
    return found


def _matches_any(value: float, is_percent: bool, facts: list[float]) -> bool:
    """Is this number close enough to some fact the tools returned?"""
    tolerance = PERCENT_TOLERANCE if is_percent else VALUE_TOLERANCE
    for fact in facts:
        if abs(value - fact) <= tolerance:
            return True
        # A rate may be quoted either as 0.52 or as 52 ("52 percent of reviews").
        if is_percent is False and abs(value / 100.0 - fact) <= PERCENT_TOLERANCE:
            return True
    return False


@dataclass
class ClaimVerifier:
    """Checks a drafted plan against the tool outputs that informed it.

    Parameters
    ----------
    evidence
        ``{aspect: [review, ...]}`` exactly as returned by
        ``AnalyticsToolbox.get_evidence_reviews``.
    facts
        Every numeric value the tools reported, flattened. A claim may only
        assert numbers that appear here.
    """

    evidence: dict[str, list[dict]]
    facts: list[float]

    def _check_citations(self, item: dict) -> CheckResult:
        aspect = item.get("aspect", "")
        cited = list(item.get("evidence_review_ids", []) or [])
        available = {r["review_id"] for r in self.evidence.get(aspect, [])}

        if not cited:
            return CheckResult(
                "citation", False, f"No reviews cited for '{aspect}'; every claim must cite evidence."
            )
        unknown = [rid for rid in cited if rid not in available]
        if unknown:
            return CheckResult(
                "citation",
                False,
                f"Cited review ids not returned by the evidence tool for '{aspect}': {unknown}",
            )
        return CheckResult("citation", True, f"{len(cited)} cited review(s) all exist")

    def _check_quotation(self, item: dict) -> CheckResult:
        quotes = list(item.get("quotes", []) or [])
        if not quotes:
            return CheckResult("quotation", True, "No direct quotes to verify")

        aspect = item.get("aspect", "")
        cited = set(item.get("evidence_review_ids", []) or [])
        corpus = " ".join(
            " ".join(str(r["text"]).split()).lower()
            for r in self.evidence.get(aspect, [])
            if r["review_id"] in cited
        )
        for quote in quotes:
            normalised = " ".join(str(quote).split()).lower().strip(' ".,')
            if normalised and normalised not in corpus:
                return CheckResult(
                    "quotation", False, f"Quote not found verbatim in the cited reviews: {quote!r}"
                )
        return CheckResult("quotation", True, f"{len(quotes)} quote(s) found verbatim")

    def _check_statistics(self, item: dict) -> CheckResult:
        claim = str(item.get("claim", ""))
        numbers = _extract_numbers(claim)
        if not numbers:
            return CheckResult("statistic", True, "Claim asserts no numbers")

        unsupported = [
            f"{value:.4g}{'%' if pct else ''}"
            for value, pct in numbers
            if not _matches_any(value, pct, self.facts)
        ]
        if unsupported:
            return CheckResult(
                "statistic",
                False,
                f"Numbers in the claim do not match any tool output: {unsupported}",
            )
        return CheckResult("statistic", True, f"{len(numbers)} number(s) match tool output")

    def verify(self, items: list[dict]) -> list[VerifiedItem]:
        """Run every check against every plan item."""
        verified = []
        for item in items:
            verified.append(
                VerifiedItem(
                    item=item,
                    checks=[
                        self._check_citations(item),
                        self._check_quotation(item),
                        self._check_statistics(item),
                    ],
                )
            )
        return verified


def groundedness(verified: list[VerifiedItem]) -> float:
    """Share of drafted items that survived verification (M5's headline metric)."""
    if not verified:
        return float("nan")
    return sum(1 for v in verified if v.accepted) / len(verified)


def collect_facts(tool_outputs: list[dict]) -> list[float]:
    """Flatten every number any tool returned into a list the verifier can check.

    Walks the nested JSON rather than requiring each tool to declare its own
    numeric fields, so a new tool is covered automatically.
    """
    facts: list[float] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            facts.append(float(node))

    for output in tool_outputs:
        walk(output)
    return facts
