"""Loading layer: real Yelp Open Dataset JSON, or a synthetic stand-in.

The Yelp Open Dataset ships as newline-delimited JSON totalling ~9 GB, which is
too large to keep in a git repository or to pull inside a unit test. This module
therefore exposes two interchangeable sources that return the *same* schema:

    load_yelp_reviews(...)        -> real data, streamed line by line
    generate_synthetic_reviews()  -> small simulated corpus for demos/CI

Because both return an identical DataFrame contract, every downstream module
(features, models, API) is written once and runs unchanged in both modes.

Returned review frame columns
-----------------------------
review_id, business_id, date (datetime64), stars (int 1-5), text (str)

Returned business frame columns
-------------------------------
business_id, name, city, state, categories (str), stars (float), review_count
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ..config import DATA_DIR, YELP_BUSINESS_FILE, YELP_REVIEW_FILE

REVIEW_COLUMNS = ["review_id", "business_id", "date", "stars", "text"]
BUSINESS_COLUMNS = [
    "business_id",
    "name",
    "city",
    "state",
    "categories",
    "stars",
    "review_count",
]


# --------------------------------------------------------------------------- #
# Real Yelp Open Dataset
# --------------------------------------------------------------------------- #


def _iter_json_lines(path: Path) -> Iterator[dict]:
    """Yield one dict per line of a newline-delimited JSON file.

    Streaming keeps peak memory flat (~a few MB) even for the 5 GB review file,
    which matters when the container is memory-capped by Kubernetes.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_yelp_businesses(
    data_dir: Path = DATA_DIR,
    category_filter: str = "Restaurants",
    city: str | None = None,
) -> pd.DataFrame:
    """Load the business table, keeping only restaurants (optionally one city).

    Filtering to a single city is the usual way to keep the project tractable:
    it bounds the review volume and removes cross-market confounding when we
    compare a venue against its peers.
    """
    path = Path(data_dir) / YELP_BUSINESS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Yelp Open Dataset from "
            "https://www.yelp.com/dataset and unpack it into ./data, "
            "or use generate_synthetic_reviews() for a demo run."
        )

    rows = []
    for record in _iter_json_lines(path):
        categories = record.get("categories") or ""
        if category_filter and category_filter.lower() not in categories.lower():
            continue
        if city and (record.get("city") or "").lower() != city.lower():
            continue
        rows.append(
            {
                "business_id": record["business_id"],
                "name": record.get("name", ""),
                "city": record.get("city", ""),
                "state": record.get("state", ""),
                "categories": categories,
                "stars": float(record.get("stars", np.nan)),
                "review_count": int(record.get("review_count", 0)),
            }
        )

    return pd.DataFrame(rows, columns=BUSINESS_COLUMNS)


def load_yelp_reviews(
    data_dir: Path = DATA_DIR,
    business_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load reviews, optionally restricted to a set of businesses.

    Parameters
    ----------
    business_ids
        Restrict to these businesses (typically the restaurant IDs returned by
        :func:`load_yelp_businesses`). ``None`` keeps everything.
    limit
        Stop after this many *kept* reviews. Useful for smoke tests.
    """
    path = Path(data_dir) / YELP_REVIEW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See load_yelp_businesses() for download instructions."
        )

    wanted = set(business_ids) if business_ids is not None else None
    rows = []
    for record in _iter_json_lines(path):
        if wanted is not None and record["business_id"] not in wanted:
            continue
        rows.append(
            {
                "review_id": record["review_id"],
                "business_id": record["business_id"],
                "date": record["date"],
                "stars": int(record["stars"]),
                "text": record.get("text", ""),
            }
        )
        if limit is not None and len(rows) >= limit:
            break

    frame = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic stand-in (demo, CI, and controlled experiments)
# --------------------------------------------------------------------------- #

# Phrase banks per aspect. Each entry is (positive_phrase, negative_phrase) so
# a single generator can emit either polarity for the same underlying aspect.
_PHRASES: dict[str, list[tuple[str, str]]] = {
    "service": [
        ("the staff were friendly and attentive", "the waiter was rude and ignored us"),
        ("service was quick and polite", "terrible service, nobody came to our table"),
        ("our server checked on us often", "the manager was unhelpful when we complained"),
    ],
    "food_quality": [
        ("the chicken rice was delicious and fresh", "the food was bland and clearly reheated"),
        ("portions were generous and tasty", "my steak arrived cold and overcooked"),
        ("best noodles I have had in ages", "the fish tasted stale, we sent it back"),
    ],
    "cleanliness": [
        ("the dining area was spotless", "the toilet was filthy and the floor was sticky"),
        ("tables were wiped down between guests", "there were flies around the counter"),
        ("kitchen looked very hygienic", "found a hair in my soup, disgusting"),
    ],
    "price_value": [
        ("great value for the price", "way overpriced for such small portions"),
        ("cheap and cheerful lunch deal", "prices went up again and quality did not"),
        ("worth every dollar", "expensive for what you get, not worth it"),
    ],
    "wait_time": [
        ("we were seated immediately", "we waited 45 minutes for the food to arrive"),
        ("no queue at all on a weekday", "the queue was insane and moved slowly"),
        ("order came out fast", "long wait even with a reservation"),
    ],
}

_FILLERS = [
    "Came here with friends on a Saturday.",
    "Second time visiting this branch.",
    "Dropped by after work for a quick meal.",
    "We ordered a few dishes to share.",
    "Would consider coming back.",
]


def generate_synthetic_reviews(
    n_businesses: int = 60,
    months: int = 30,
    start: str = "2022-01-01",
    base_reviews_per_month: int = 14,
    degrading_fraction: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a restaurant review corpus with known ground-truth decline events.

    A fraction of the businesses are given a hidden *degradation event*: from a
    randomly chosen month onwards, one aspect starts failing and the probability
    of a negative review climbs. This gives the experiments in Section 4 a
    ground truth for "did the early-warning system fire before the rating fell",
    which the real Yelp data cannot provide directly.

    Returns
    -------
    (reviews, businesses)
        Frames matching the schema documented at module level, plus two extra
        columns on ``businesses``: ``true_failing_aspect`` and
        ``true_event_month`` (NaT for healthy venues) for evaluation only --
        no model is allowed to read them.
    """
    rng = np.random.default_rng(seed)
    aspects = list(_PHRASES)
    month_index = pd.date_range(start=start, periods=months, freq="MS")

    business_rows, review_rows = [], []
    review_counter = 0

    for b in range(n_businesses):
        business_id = f"syn_b{b:04d}"
        # Baseline health of the venue: its long-run probability of a bad review.
        base_neg_rate = float(rng.uniform(0.08, 0.30))
        popularity = float(rng.uniform(0.6, 1.8))

        is_degrading = rng.random() < degrading_fraction
        failing_aspect = str(rng.choice(aspects)) if is_degrading else None
        # Event must leave room for both the feature window and the label window.
        event_idx = int(rng.integers(8, months - 4)) if is_degrading else -1
        event_month = month_index[event_idx] if is_degrading else pd.NaT

        for m_idx, month in enumerate(month_index):
            n_reviews = max(
                1, int(rng.poisson(base_reviews_per_month * popularity))
            )
            # Degradation ramps in over ~4 months rather than switching instantly,
            # mirroring how a staffing or supplier problem actually shows up.
            ramp = 0.0
            if is_degrading and m_idx >= event_idx:
                ramp = min(1.0, (m_idx - event_idx + 1) / 4.0)
            neg_rate = min(0.92, base_neg_rate + 0.45 * ramp)

            for _ in range(n_reviews):
                negative = rng.random() < neg_rate
                # Which aspects this reviewer talks about (1-2 of them).
                if negative and is_degrading and m_idx >= event_idx and rng.random() < 0.6 + 0.3 * ramp:
                    mentioned = [failing_aspect]
                else:
                    mentioned = list(
                        rng.choice(aspects, size=int(rng.integers(1, 3)), replace=False)
                    )

                parts = [str(rng.choice(_FILLERS))]
                for aspect in mentioned:
                    pos, neg = _PHRASES[aspect][int(rng.integers(0, len(_PHRASES[aspect])))]
                    parts.append(neg if negative else pos)
                text = " ".join(parts)

                # Stars follow polarity with realistic spread.
                stars = int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20])) if negative \
                    else int(rng.choice([3, 4, 5], p=[0.15, 0.40, 0.45]))

                day = int(rng.integers(1, 28))
                review_rows.append(
                    {
                        "review_id": f"syn_r{review_counter:07d}",
                        "business_id": business_id,
                        "date": month + pd.Timedelta(days=day - 1),
                        "stars": stars,
                        "text": text,
                    }
                )
                review_counter += 1

        business_rows.append(
            {
                "business_id": business_id,
                "name": f"Synthetic Eatery {b:03d}",
                "city": "Demo City",
                "state": "DC",
                "categories": "Restaurants, Asian Fusion",
                "stars": np.nan,          # filled below from the generated reviews
                "review_count": 0,
                "true_failing_aspect": failing_aspect,
                "true_event_month": event_month,
            }
        )

    reviews = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    businesses = pd.DataFrame(business_rows)

    # Backfill the aggregate columns so the synthetic frame matches Yelp's.
    agg = reviews.groupby("business_id")["stars"].agg(["mean", "count"])
    businesses = businesses.set_index("business_id")
    businesses["stars"] = agg["mean"].round(1)
    businesses["review_count"] = agg["count"]
    businesses = businesses.reset_index()

    return reviews.sort_values("date").reset_index(drop=True), businesses
