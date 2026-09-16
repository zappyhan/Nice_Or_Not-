"""Central configuration for the restaurant reputation early-warning system.

Every tunable constant lives here so that experiments in the report (Section 4)
can be reproduced by changing one file rather than hunting through the code.
Values are deliberately conservative defaults chosen for the Yelp Open Dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"            # raw Yelp JSON lives here (git-ignored)
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"   # fitted models + reports land here

# Yelp Open Dataset file names (as shipped by Yelp, unchanged).
YELP_BUSINESS_FILE = "yelp_academic_dataset_business.json"
YELP_REVIEW_FILE = "yelp_academic_dataset_review.json"

# --------------------------------------------------------------------------- #
# Aspect taxonomy
# --------------------------------------------------------------------------- #
# These are the operational levers a restaurant manager can actually pull.
# Keeping the taxonomy small keeps the multi-label problem learnable and keeps
# the final recommendation actionable ("fix X first") rather than vague.

ASPECTS: tuple[str, ...] = (
    "service",
    "food_quality",
    "cleanliness",
    "price_value",
    "wait_time",
)

# A review with <= this many stars is treated as expressing dissatisfaction.
# Used to turn "mentions cleanliness" into "complains about cleanliness".
NEGATIVE_STAR_THRESHOLD = 2

# --------------------------------------------------------------------------- #
# Model + windowing hyper-parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    """Hyper-parameters shared by the training and scoring pipelines."""

    # --- panel construction -------------------------------------------------
    freq: str = "MS"                 # monthly buckets (month start)
    history_months: int = 6          # trailing window used to build features
    horizon_months: int = 3          # look-ahead window used to build the label
    min_reviews_history: int = 8     # ignore businesses with too little signal
    min_reviews_horizon: int = 3     # need enough future reviews for a fair label

    # --- decline label ------------------------------------------------------
    decline_threshold: float = 0.20  # a drop of >= 0.2 stars counts as a decline

    # --- aspect classifier --------------------------------------------------
    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 3
    aspect_threshold: float = 0.50   # P(aspect) above which we count a mention

    # --- emerging-complaint detector ---------------------------------------
    ewma_alpha: float = 0.35         # smoothing for the expected complaint rate
    burn_in_months: int = 4          # months needed before we trust the baseline
    alert_z: float = 2.0             # robust z-score above which we raise an alert
    min_mentions_for_alert: int = 3  # suppress alerts built on 1-2 reviews

    # --- evaluation ---------------------------------------------------------
    test_size_months: int = 6        # final N months held out (time-based split)
    top_k_precision: int = 20        # precision@k reported in Section 4
    random_state: int = 42

    # --- prioritisation -----------------------------------------------------
    # Effort multipliers: how hard each lever is to move in practice. Used to
    # convert "biggest impact" into "best first action". Sourced from the
    # operations-management literature and adjustable per client.
    effort_weights: dict[str, float] = field(
        default_factory=lambda: {
            "service": 1.0,        # retraining / rostering: fast
            "food_quality": 1.4,   # recipe, supplier or chef changes: slower
            "cleanliness": 0.9,    # checklists and audits: fastest
            "price_value": 1.6,    # menu re-pricing: strategic, slow
            "wait_time": 1.1,      # process / staffing changes: moderate
        }
    )


SETTINGS = Settings()
