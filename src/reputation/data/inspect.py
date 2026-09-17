"""Dataset preflight: look at the Yelp download before training on it.

    python -m reputation.data.inspect --data-dir data

The Yelp Open Dataset is ~9 GB unpacked and training on all of it at once is
rarely what you want: the market-relative features compare a venue against its
peers, and a restaurant in Philadelphia is not a peer of one in Santa Barbara.
The usual move is to pick one city. This command tells you which cities are
worth picking, and roughly what each will cost to run, *before* you start a job
that takes an hour.

It reads only ``business.json`` (~120 MB) by default, because Yelp already
stores each venue's review count there -- no need to stream the 5 GB review
file just to rank cities. Pass ``--sample-reviews`` to also report the date
range, which does require touching the review file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR, YELP_BUSINESS_FILE, YELP_REVIEW_FILE
from .loader import _iter_json_lines, load_yelp_businesses


def summarise_cities(businesses: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Rank cities by how much restaurant review volume they carry."""
    grouped = (
        businesses.groupby(["city", "state"])
        .agg(
            restaurants=("business_id", "count"),
            reviews=("review_count", "sum"),
            median_reviews=("review_count", "median"),
            mean_stars=("stars", "mean"),
        )
        .reset_index()
        .sort_values("reviews", ascending=False)
    )
    return grouped.head(top)


def review_date_range(data_dir: Path, sample: int) -> tuple[str, str, int]:
    """Scan the first ``sample`` reviews for their date range.

    Yelp's review file is not sorted by date, so a sample gives a good estimate
    of the span without reading all 7 million records.
    """
    dates = []
    for i, record in enumerate(_iter_json_lines(Path(data_dir) / YELP_REVIEW_FILE)):
        if i >= sample:
            break
        dates.append(record["date"])
    if not dates:
        return ("", "", 0)
    series = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    return (str(series.min().date()), str(series.max().date()), len(series))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Yelp Open Dataset download")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--top", type=int, default=15, help="Cities to list")
    parser.add_argument(
        "--sample-reviews",
        type=int,
        default=0,
        help="Also sample this many reviews to report the date range (0 = skip)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Dataset directory: {data_dir.resolve()}\n")

    missing = []
    for filename in (YELP_BUSINESS_FILE, YELP_REVIEW_FILE):
        path = data_dir / filename
        if path.exists():
            print(f"  [ok]      {filename}  ({path.stat().st_size / 1e9:.2f} GB)")
        else:
            missing.append(filename)
            print(f"  [MISSING] {filename}")
    if missing:
        raise SystemExit(
            "\nDownload the dataset from https://www.yelp.com/dataset and unpack "
            f"it into {data_dir.resolve()}."
        )

    print("\nScanning businesses ...")
    all_businesses = load_yelp_businesses(data_dir, category_filter="")
    restaurants = load_yelp_businesses(data_dir)
    print(f"  {len(all_businesses):,} businesses, of which {len(restaurants):,} are restaurants")
    print(f"  {int(restaurants['review_count'].sum()):,} restaurant reviews in total")

    if args.sample_reviews:
        print(f"\nSampling {args.sample_reviews:,} reviews for the date range ...")
        first, last, n = review_date_range(data_dir, args.sample_reviews)
        print(f"  {n:,} sampled reviews span {first} to {last}")

    print(f"\nTop {args.top} cities by restaurant review volume:\n")
    cities = summarise_cities(restaurants, args.top)
    header = f"{'City':<22}{'State':<7}{'Restaurants':>12}{'Reviews':>12}{'Median':>9}{'Stars':>7}"
    print(header)
    print("-" * len(header))
    for row in cities.itertuples():
        print(
            f"{str(row.city)[:21]:<22}{str(row.state):<7}{row.restaurants:>12,}"
            f"{int(row.reviews):>12,}{row.median_reviews:>9.0f}{row.mean_stars:>7.2f}"
        )

    best = cities.iloc[0]
    print(
        "\nA single city is the usual unit of analysis: the peer-relative features\n"
        "compare each venue against the others in the same market.\n"
    )
    print("Suggested first run:\n")
    print(f'  python -m reputation.pipeline.train --source yelp --city "{best.city}"\n')
    print(
        f"  ~{int(best.reviews):,} reviews, ~{best.restaurants:,} restaurants.\n"
        "  Expect roughly 5-15 minutes on a laptop, and a few GB of RAM while the\n"
        "  review text is loaded. Add --limit to cap the reviews for a quick trial."
    )


if __name__ == "__main__":
    main()
