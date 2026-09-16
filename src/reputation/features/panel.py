"""Panel construction: turn annotated reviews into a business-by-month table.

Everything downstream of the text models is a *panel* problem: for each
business and each month we need (a) a description of the recent past and
(b) what happened next. Two functions do that work:

    build_monthly_panel()     reviews  -> business x month aggregates
    build_supervised_frame()  panel    -> (X, y) with strictly past-only features

The hard requirement here is **no leakage**: a row stamped at month *t* may only
use reviews dated <= *t*, while its label is computed from months *t+1 .. t+H*.
Both windows are enforced in one place so the guarantee is easy to audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ASPECTS, SETTINGS, Settings

# pandas uses different aliases for date ranges ("MS" = month start) and for
# Period conversion ("M"); this map keeps a single frequency setting in config.
_PERIOD_ALIAS = {"MS": "M", "W": "W", "QS": "Q"}


# Columns that are counts and therefore legitimately zero in a quiet month.
_COUNT_COLUMNS = ["n_reviews", "sum_stars", "n_negative"] + [
    f"n_complaint_{a}" for a in ASPECTS
] + [f"n_mention_{a}" for a in ASPECTS]


def build_monthly_panel(annotated: pd.DataFrame, freq: str = SETTINGS.freq) -> pd.DataFrame:
    """Aggregate annotated reviews into one row per (business_id, period).

    Months with no reviews are inserted with zero counts so that a sudden drop
    in review volume -- itself an early warning sign -- is visible to the model
    instead of silently disappearing from the index.
    """
    frame = annotated.copy()
    alias = _PERIOD_ALIAS.get(freq, freq)
    frame["period"] = frame["date"].dt.to_period(alias).dt.to_timestamp()

    agg = {
        "n_reviews": ("stars", "size"),
        "sum_stars": ("stars", "sum"),
        "n_negative": ("is_negative", "sum"),
        "mean_text_len": ("text", lambda s: float(np.mean([len(t) for t in s]))),
    }
    for aspect in ASPECTS:
        agg[f"n_complaint_{aspect}"] = (f"complaint_{aspect}", "sum")
        agg[f"n_mention_{aspect}"] = (f"mention_{aspect}", "sum")

    panel = frame.groupby(["business_id", "period"]).agg(**agg).reset_index()

    # Dense calendar: every business gets every month between the global first
    # and last observed month.
    all_periods = pd.date_range(panel["period"].min(), panel["period"].max(), freq=freq)
    index = pd.MultiIndex.from_product(
        [sorted(panel["business_id"].unique()), all_periods],
        names=["business_id", "period"],
    )
    panel = (
        panel.set_index(["business_id", "period"])
        .reindex(index)
        .reset_index()
    )
    panel[_COUNT_COLUMNS] = panel[_COUNT_COLUMNS].fillna(0)
    panel["mean_text_len"] = panel["mean_text_len"].fillna(0.0)

    # Derived rates. Guarded division keeps quiet months as NaN rather than inf.
    denom = panel["n_reviews"].replace(0, np.nan)
    panel["mean_stars"] = panel["sum_stars"] / denom
    panel["negative_rate"] = panel["n_negative"] / denom
    for aspect in ASPECTS:
        panel[f"complaint_rate_{aspect}"] = panel[f"n_complaint_{aspect}"] / denom
        panel[f"mention_rate_{aspect}"] = panel[f"n_mention_{aspect}"] / denom

    return panel.sort_values(["business_id", "period"]).reset_index(drop=True)


def _window_sum(group: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Trailing sum over ``window`` months, inclusive of the current month."""
    return group[column].rolling(window=window, min_periods=1).sum()


def build_supervised_frame(
    panel: pd.DataFrame, settings: Settings = SETTINGS
) -> pd.DataFrame:
    """Build the modelling table: trailing-window features plus a future label.

    Feature blocks
    --------------
    ``hist_*``    aggregates over the trailing ``history_months`` window
    ``recent_*``  aggregates over the trailing 3 months
    ``delta_*``   recent minus the preceding block (the momentum signals that
                  do most of the work -- a *rising* complaint rate matters far
                  more than a high but stable one)
    ``peer_*``    the same quantity relative to the market median that month,
                  which controls for seasonality and city-wide shocks

    Label
    -----
    ``y_decline`` = 1 when mean stars over the next ``horizon_months`` fall at
    least ``decline_threshold`` below the trailing mean.
    """
    h, horizon = settings.history_months, settings.horizon_months
    rows = []

    for business_id, group in panel.groupby("business_id", sort=True):
        group = group.sort_values("period").reset_index(drop=True)

        # --- trailing windows (past-only, inclusive of t) -------------------
        hist_n = _window_sum(group, "n_reviews", h)
        hist_stars = _window_sum(group, "sum_stars", h)
        hist_neg = _window_sum(group, "n_negative", h)
        recent_n = _window_sum(group, "n_reviews", 3)
        recent_stars = _window_sum(group, "sum_stars", 3)
        recent_neg = _window_sum(group, "n_negative", 3)
        # "Prior" block = trailing window minus the recent block.
        prior_n = (hist_n - recent_n).clip(lower=0)
        prior_stars = hist_stars - recent_stars
        prior_neg = hist_neg - recent_neg

        feat = pd.DataFrame({"business_id": business_id, "period": group["period"]})
        feat["hist_reviews"] = hist_n
        feat["hist_mean_stars"] = hist_stars / hist_n.replace(0, np.nan)
        feat["hist_negative_rate"] = hist_neg / hist_n.replace(0, np.nan)
        feat["recent_mean_stars"] = recent_stars / recent_n.replace(0, np.nan)
        feat["recent_negative_rate"] = recent_neg / recent_n.replace(0, np.nan)
        feat["delta_mean_stars"] = feat["recent_mean_stars"] - (
            prior_stars / prior_n.replace(0, np.nan)
        )
        feat["delta_negative_rate"] = feat["recent_negative_rate"] - (
            prior_neg / prior_n.replace(0, np.nan)
        )
        feat["volume_ratio"] = recent_n / prior_n.replace(0, np.nan)
        feat["star_volatility"] = (
            group["mean_stars"].rolling(window=h, min_periods=2).std()
        )
        feat["mean_text_len"] = group["mean_text_len"].rolling(h, min_periods=1).mean()
        feat["months_active"] = np.arange(len(group)) + 1

        for aspect in ASPECTS:
            hist_c = _window_sum(group, f"n_complaint_{aspect}", h)
            recent_c = _window_sum(group, f"n_complaint_{aspect}", 3)
            prior_c = hist_c - recent_c
            feat[f"hist_complaint_rate_{aspect}"] = hist_c / hist_n.replace(0, np.nan)
            feat[f"recent_complaint_rate_{aspect}"] = recent_c / recent_n.replace(0, np.nan)
            feat[f"delta_complaint_rate_{aspect}"] = feat[
                f"recent_complaint_rate_{aspect}"
            ] - (prior_c / prior_n.replace(0, np.nan))

        # --- forward label (future-only, strictly after t) ------------------
        future_stars = (
            group["sum_stars"][::-1]
            .rolling(window=horizon, min_periods=1)
            .sum()[::-1]
            .shift(-1)
        )
        future_n = (
            group["n_reviews"][::-1]
            .rolling(window=horizon, min_periods=1)
            .sum()[::-1]
            .shift(-1)
        )
        feat["future_reviews"] = future_n
        feat["future_mean_stars"] = future_stars / future_n.replace(0, np.nan)
        rows.append(feat)

    supervised = pd.concat(rows, ignore_index=True)

    # --- peer-relative features: subtract the market median for that month ---
    peer_columns = ["hist_negative_rate"] + [
        f"hist_complaint_rate_{a}" for a in ASPECTS
    ]
    market = supervised.groupby("period")[peer_columns].transform("median")
    for column in peer_columns:
        supervised[f"peer_{column}"] = supervised[column] - market[column]

    # --- label ---------------------------------------------------------------
    supervised["star_change"] = (
        supervised["future_mean_stars"] - supervised["hist_mean_stars"]
    )
    supervised["y_decline"] = (
        supervised["star_change"] <= -settings.decline_threshold
    ).astype(int)

    # Keep only rows where both windows carry enough evidence to be meaningful.
    usable = (
        # A full trailing window must exist, otherwise the "momentum" features
        # are computed against a partially observed history and are misleading.
        (supervised["months_active"] >= settings.history_months)
        & (supervised["hist_reviews"] >= settings.min_reviews_history)
        & (supervised["future_reviews"] >= settings.min_reviews_horizon)
        & supervised["star_change"].notna()
    )
    return supervised.loc[usable].reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns: everything except identifiers, labels and leakage."""
    excluded = {
        "business_id",
        "period",
        "future_reviews",
        "future_mean_stars",
        "star_change",
        "y_decline",
    }
    return [
        c
        for c in frame.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]
