"""Run the M5 agent from the command line.

    python -m reputation.agents.cli --business-id syn_b0007
    python -m reputation.agents.cli --compare          # LLM vs rule-based baseline

With no Anthropic credentials configured this runs the deterministic planner
and says so, so a demo never fails for want of an API key.
"""

from __future__ import annotations

import argparse
import json
import logging

from ..config import ARTIFACT_DIR
from ..pipeline.score import load_bundle, latest_rows
from .agent import build_agent, compare_planners
from .planner import ClaudePlanner, RuleBasedPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ReviewRadar agent (M5)")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--business-id", default=None, help="Defaults to the riskiest venue")
    parser.add_argument("--planner", choices=["auto", "claude", "rule_based"], default="auto")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both planners on the same venue and print the comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    bundle = load_bundle(args.artifact_dir)

    business_id = args.business_id
    if business_id is None:
        latest = latest_rows(bundle["supervised"])
        risk = bundle["decline_predictor"].predict_proba(latest)
        business_id = str(latest.assign(r=risk).sort_values("r", ascending=False).iloc[0]["business_id"])

    if args.compare:
        payload = compare_planners(
            bundle, business_id, [RuleBasedPlanner(), ClaudePlanner()]
        )
    else:
        payload = build_agent(bundle, args.planner).run(business_id)

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
