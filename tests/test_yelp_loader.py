"""Tests for the real Yelp Open Dataset code path.

The dataset itself is ~9 GB and cannot live in a repository or a CI job, so
these tests write small fixtures **in Yelp's exact on-disk format** -- the real
key names, the real value types (``stars`` is a float even in the review file),
the real timestamp format, the extra fields the pipeline does not use, and the
nulls Yelp actually ships -- then run the loader over them.

This is what stands between "works on our simulator" and "works on the real
download": every field-level assumption the loader makes is asserted here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reputation.config import YELP_BUSINESS_FILE, YELP_REVIEW_FILE  # noqa: E402
from reputation.data.loader import (  # noqa: E402
    load_yelp_businesses,
    load_yelp_reviews,
)

# Verbatim-shaped records from the Yelp Open Dataset documentation: same keys,
# same types, same extras. Only the values are invented.
BUSINESSES = [
    {
        "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
        "name": "Turning Point of North Wales",
        "address": "1460 Bethlehem Pike",
        "city": "North Wales",
        "state": "PA",
        "postal_code": "19454",
        "latitude": 40.2199,
        "longitude": -75.2377,
        "stars": 3.0,
        "review_count": 169,
        "is_open": 1,
        "attributes": {"RestaurantsDelivery": "False", "BusinessParking": "{'garage': False}"},
        "categories": "Restaurants, Breakfast & Brunch, Food, Juice Bars & Smoothies",
        "hours": {"Monday": "7:30-15:0", "Tuesday": "7:30-15:0"},
    },
    {
        "business_id": "MTSW4McQd7CbVtyjqoe9mw",
        "name": "St Honore Pastries",
        "address": "935 Race St",
        "city": "Philadelphia",
        "state": "PA",
        "postal_code": "19107",
        "latitude": 39.9555,
        "longitude": -75.1555,
        "stars": 4.0,
        "review_count": 80,
        "is_open": 1,
        "attributes": None,          # Yelp ships nulls here
        "categories": "Restaurants, Food, Bubble Tea, Coffee & Tea, Bakeries",
        "hours": None,
    },
    {
        "business_id": "Pns2l4eNsfO8kk83dixA6A",
        "name": "Abby Rappoport, LAC, CMQ",
        "address": "1616 Chapala St, Ste 2",
        "city": "Santa Barbara",
        "state": "CA",
        "postal_code": "93101",
        "latitude": 34.4266787,
        "longitude": -119.7111968,
        "stars": 5.0,
        "review_count": 7,
        "is_open": 0,
        "attributes": {"ByAppointmentOnly": "True"},
        "categories": "Doctors, Traditional Chinese Medicine, Health & Medical",
        "hours": None,
    },
    {
        "business_id": "no-categories-at-all",
        "name": "Mystery Venue",
        "city": "Philadelphia",
        "state": "PA",
        "stars": 2.5,
        "review_count": 3,
        "categories": None,          # a real and surprisingly common case
        "hours": None,
    },
]

REVIEWS = [
    {
        "review_id": "KU_O5udG6zpxOg-VcAEodg",
        "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
        "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
        "stars": 3.0,                # float, not int, in the real file
        "useful": 0,
        "funny": 0,
        "cool": 0,
        "text": "If you decide to eat here, just be aware it is going to take about two hours.",
        "date": "2018-07-07 22:09:11",   # space-separated, not ISO 'T'
    },
    {
        "review_id": "BiTunyQ73aT9WBnpR9DZGw",
        "user_id": "OyoGAe7OKpv6SyGZT5g77Q",
        "business_id": "MTSW4McQd7CbVtyjqoe9mw",
        "stars": 5.0,
        "useful": 1,
        "funny": 0,
        "cool": 1,
        "text": "The egg tarts are fresh and the staff were friendly. Great value for the price.",
        "date": "2021-01-03 18:21:04",
    },
    {
        "review_id": "saUsX_uimxRlCVr67Z4Jig",
        "user_id": "8g_iMtfSiwikVnbP2etR0A",
        "business_id": "Pns2l4eNsfO8kk83dixA6A",   # not a restaurant
        "stars": 1.0,
        "useful": 0,
        "funny": 0,
        "cool": 0,
        "text": "Waited forever and the room was not clean.",
        "date": "2017-12-15 23:14:02",
    },
]


@pytest.fixture(scope="module")
def yelp_dir(tmp_path_factory) -> Path:
    """Write newline-delimited JSON exactly as Yelp ships it."""
    directory = tmp_path_factory.mktemp("yelp")
    for filename, records in (
        (YELP_BUSINESS_FILE, BUSINESSES),
        (YELP_REVIEW_FILE, REVIEWS),
    ):
        with (directory / filename).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        # Yelp's files end with a trailing newline; a blank line must not crash
        # the streaming parser.
        with (directory / filename).open("a", encoding="utf-8") as handle:
            handle.write("\n")
    return directory


def test_business_loader_keeps_only_restaurants(yelp_dir):
    frame = load_yelp_businesses(yelp_dir)
    ids = set(frame["business_id"])
    assert "XQfwVwDr-v0ZS3_CbbE5Xw" in ids
    assert "MTSW4McQd7CbVtyjqoe9mw" in ids
    assert "Pns2l4eNsfO8kk83dixA6A" not in ids   # a doctor, not a restaurant
    assert "no-categories-at-all" not in ids     # null categories must not crash


def test_business_loader_filters_by_city_case_insensitively(yelp_dir):
    frame = load_yelp_businesses(yelp_dir, city="philadelphia")
    assert list(frame["business_id"]) == ["MTSW4McQd7CbVtyjqoe9mw"]
    assert frame["stars"].dtype.kind == "f"
    assert frame["review_count"].dtype.kind == "i"


def test_review_loader_parses_yelp_types_and_timestamps(yelp_dir):
    restaurants = load_yelp_businesses(yelp_dir)
    reviews = load_yelp_reviews(yelp_dir, business_ids=restaurants["business_id"])

    assert len(reviews) == 2                      # the doctor's review is excluded
    assert reviews["stars"].dtype.kind == "i"     # 3.0 (float) -> 3 (int)
    assert set(reviews["stars"]) == {3, 5}
    assert str(reviews["date"].dtype).startswith("datetime64")
    assert reviews["date"].min().year == 2018
    assert reviews["text"].str.len().min() > 0


def test_review_loader_respects_the_limit(yelp_dir):
    assert len(load_yelp_reviews(yelp_dir, limit=1)) == 1


def test_missing_dataset_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="yelp.com/dataset"):
        load_yelp_businesses(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_yelp_reviews(tmp_path)


# --------------------------------------------------------------------------- #
# Dataset preflight
# --------------------------------------------------------------------------- #


def test_city_summary_ranks_by_review_volume(yelp_dir):
    """`reputation.data.inspect` must rank markets without touching reviews."""
    from reputation.data.inspect import summarise_cities

    restaurants = load_yelp_businesses(yelp_dir)
    cities = summarise_cities(restaurants, top=5)

    assert list(cities.columns) == [
        "city", "state", "restaurants", "reviews", "median_reviews", "mean_stars",
    ]
    # North Wales (169 reviews) outranks Philadelphia (80) on volume.
    assert cities.iloc[0]["city"] == "North Wales"
    assert cities["reviews"].is_monotonic_decreasing


def test_date_range_sampling_reads_only_the_sample(yelp_dir):
    from reputation.data.inspect import review_date_range

    first, last, n = review_date_range(yelp_dir, sample=2)
    assert n == 2
    assert first <= last
