"""Module 3 -- Will this restaurant's rating decline?

Binary classification on the business-month panel: given everything observable
up to month *t*, predict whether the mean star rating over months
*t+1 .. t+H* falls at least ``decline_threshold`` below the trailing mean.

Design choices worth defending in the report
--------------------------------------------
* **Gradient-boosted trees** (`HistGradientBoostingClassifier`) rather than a
  linear model: the interaction between "complaints rising" and "volume
  falling" is genuinely non-linear, and trees handle the NaNs produced by quiet
  months natively, so no imputation step can leak information.
* **Time-based split**, never a random one. A random split would place month
  *t+1* of a business in training and month *t* in test, which leaks the
  outcome and inflates AUC by a large margin. The last ``test_size_months``
  of the calendar are held out instead.
* **Probability calibration** on a validation slice, because the downstream
  recommender treats the output as an expected-loss weight; a miscalibrated
  score would rank the wrong action first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ..config import SETTINGS, Settings
from ..features.panel import feature_columns


def time_based_split(
    frame: pd.DataFrame, settings: Settings = SETTINGS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel by calendar time: earlier months train, later months test.

    Returns ``(train, test)``. The split point is chosen so the test block holds
    the final ``test_size_months`` distinct periods present in the data.
    """
    periods = np.sort(frame["period"].unique())
    if len(periods) <= settings.test_size_months + 1:
        raise ValueError(
            f"Need more than {settings.test_size_months + 1} periods to split; "
            f"got {len(periods)}. Lower Settings.test_size_months for small runs."
        )
    cutoff = periods[-settings.test_size_months]
    return frame[frame["period"] < cutoff].copy(), frame[frame["period"] >= cutoff].copy()


@dataclass
class DeclinePredictor:
    """Calibrated gradient-boosted classifier over the panel features."""

    settings: Settings = SETTINGS
    model: CalibratedClassifierCV | None = None
    features: list[str] = field(default_factory=list)

    def _base_estimator(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_depth=4,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=self.settings.random_state,
        )

    def fit(self, train: pd.DataFrame) -> "DeclinePredictor":
        """Fit on a training slice produced by :func:`time_based_split`."""
        self.features = feature_columns(train)
        X, y = train[self.features], train["y_decline"]
        if y.nunique() < 2:
            raise ValueError("Training slice contains a single class; widen the window.")
        # cv="prefit" is avoided on purpose: letting CalibratedClassifierCV do
        # its own internal CV keeps the calibration set disjoint from the fit.
        self.model = CalibratedClassifierCV(
            self._base_estimator(), method="isotonic", cv=3
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """P(decline) for each row, aligned to the training feature order."""
        if self.model is None:
            raise RuntimeError("DeclinePredictor.fit() must be called first")
        missing = [c for c in self.features if c not in frame.columns]
        for column in missing:                 # tolerate schema drift at serve time
            frame = frame.assign(**{column: np.nan})
        return self.model.predict_proba(frame[self.features])[:, 1]

    def importance(
        self, test: pd.DataFrame, n_repeats: int = 5
    ) -> pd.DataFrame:
        """Permutation importance on held-out data (drop in average precision).

        Permutation importance is used instead of tree split-gain because the
        panel features are strongly correlated; split-gain would arbitrarily
        credit one of a correlated pair and mislead the write-up.
        """
        if self.model is None:
            raise RuntimeError("DeclinePredictor.fit() must be called first")
        result = permutation_importance(
            self.model,
            test[self.features],
            test["y_decline"],
            scoring="average_precision",
            n_repeats=n_repeats,
            random_state=self.settings.random_state,
        )
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def counterfactual_drop(
        self, row: pd.DataFrame, assignments: dict[str, float]
    ) -> float:
        """How much P(decline) falls if the given features are moved to new values.

        This is the bridge from *prediction* to *recommendation*. To ask "how
        much of this venue's risk comes from its cleanliness problem?" we move
        every cleanliness feature to the state the venue would be in if the
        problem were fixed -- complaint level at the market median *and* no
        upward trend -- and measure the fall in predicted probability.

        Intervening on the whole feature group matters: the panel carries a
        level, a recent level and a momentum term per aspect, and patching only
        one of the three leaves the model looking at a contradictory row (a
        venue with average complaints that is nonetheless deteriorating fast),
        which produces meaningless attributions.

        This is an intervention on the *model*, not a causal effect, and the
        report states that limitation. Under a stable environment it still
        ranks actions far better than raw complaint counts, which always
        promote whichever aspect customers mention most often.
        """
        baseline = float(self.predict_proba(row)[0])
        patched = row.copy()
        for column, value in assignments.items():
            if column in patched.columns:
                patched.loc[:, column] = value
        return baseline - float(self.predict_proba(patched)[0])
