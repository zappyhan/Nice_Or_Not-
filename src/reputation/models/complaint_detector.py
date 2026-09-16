"""Module 2 -- Emerging complaint detection.

A restaurant's star rating is a lagging indicator: by the time the average
visibly drops, months of bad experiences have already been published. The
complaint rate for a *specific aspect*, however, moves first. This module
watches each (business, aspect) complaint-rate series and fires an alert when
the latest months break away from that venue's own recent baseline.

Method
------
For each series we maintain an EWMA baseline and a robust dispersion estimate
(median absolute deviation, scaled to be comparable with a standard deviation).
The alert statistic is a robust z-score

    z_t = (rate_t - ewma_{t-1}) / (1.4826 * MAD + eps)

Why robust statistics: complaint rates are bounded, skewed and spiky, so a
plain mean/standard-deviation control chart raises alerts on ordinary noise.
MAD ignores the very outliers we are trying to detect, which keeps the baseline
stable while the venue degrades.

Two guards suppress the classic false positives:
  * ``burn_in_months`` -- no alerts until a baseline actually exists;
  * ``min_mentions_for_alert`` -- a 100% complaint rate over two reviews is not
    evidence, so alerts require an absolute volume of complaints as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ASPECTS, SETTINGS, Settings

_MAD_TO_SIGMA = 1.4826  # makes MAD comparable to a standard deviation


@dataclass(frozen=True)
class ComplaintAlert:
    """One (business, aspect, month) early-warning signal."""

    business_id: str
    aspect: str
    period: pd.Timestamp
    rate: float
    baseline: float
    z_score: float
    n_complaints: int
    n_reviews: int

    @property
    def severity(self) -> str:
        """Coarse triage band shown in the dashboard and the API response."""
        if self.z_score >= 4.0:
            return "critical"
        if self.z_score >= 3.0:
            return "high"
        return "watch"

    def as_dict(self) -> dict:
        return {
            "business_id": self.business_id,
            "aspect": self.aspect,
            "period": self.period.strftime("%Y-%m"),
            "rate": round(float(self.rate), 4),
            "baseline": round(float(self.baseline), 4),
            "z_score": round(float(self.z_score), 2),
            "n_complaints": int(self.n_complaints),
            "n_reviews": int(self.n_reviews),
            "severity": self.severity,
        }


def _robust_scale(values: np.ndarray) -> float:
    """Scaled MAD with a floor, so a flat history cannot produce infinite z."""
    if values.size == 0:
        return 0.05
    mad = float(np.median(np.abs(values - np.median(values))))
    return max(_MAD_TO_SIGMA * mad, 0.03)


def score_series(rates: pd.Series, alpha: float = SETTINGS.ewma_alpha) -> pd.DataFrame:
    """Return the EWMA baseline and robust z-score for one rate series.

    The baseline is shifted by one month so that the current observation is
    never part of its own expectation -- the same causality rule enforced in
    the panel features.
    """
    clean = rates.astype(float).ffill().fillna(0.0)
    baseline = clean.ewm(alpha=alpha, adjust=False).mean().shift(1)
    scale = [
        _robust_scale(clean.iloc[:i].to_numpy()) if i > 0 else np.nan
        for i in range(len(clean))
    ]
    z = (clean - baseline) / pd.Series(scale, index=clean.index)
    return pd.DataFrame({"rate": clean, "baseline": baseline, "z_score": z})


def detect_emerging_complaints(
    panel: pd.DataFrame,
    settings: Settings = SETTINGS,
    aspects: tuple[str, ...] = ASPECTS,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Scan every (business, aspect) series and return the alerts as a frame.

    Parameters
    ----------
    panel
        Output of :func:`reputation.features.panel.build_monthly_panel`.
    as_of
        Only report alerts for this month (what the nightly job does). ``None``
        returns the full alert history, which is what Section 4's lead-time
        experiment needs.
    """
    alerts: list[ComplaintAlert] = []

    for business_id, group in panel.groupby("business_id", sort=True):
        group = group.sort_values("period").reset_index(drop=True)
        for aspect in aspects:
            scored = score_series(group[f"complaint_rate_{aspect}"], settings.ewma_alpha)
            for i, row in scored.iterrows():
                if i < settings.burn_in_months or not np.isfinite(row["z_score"]):
                    continue
                n_complaints = int(group.loc[i, f"n_complaint_{aspect}"])
                if (
                    row["z_score"] < settings.alert_z
                    or n_complaints < settings.min_mentions_for_alert
                ):
                    continue
                alerts.append(
                    ComplaintAlert(
                        business_id=str(business_id),
                        aspect=aspect,
                        period=group.loc[i, "period"],
                        rate=float(row["rate"]),
                        baseline=float(row["baseline"]),
                        z_score=float(row["z_score"]),
                        n_complaints=n_complaints,
                        n_reviews=int(group.loc[i, "n_reviews"]),
                    )
                )

    frame = pd.DataFrame([a.as_dict() for a in alerts])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "business_id", "aspect", "period", "rate", "baseline",
                "z_score", "n_complaints", "n_reviews", "severity",
            ]
        )
    if as_of is not None:
        frame = frame[frame["period"] == pd.Timestamp(as_of).strftime("%Y-%m")]
    return frame.sort_values("z_score", ascending=False).reset_index(drop=True)
