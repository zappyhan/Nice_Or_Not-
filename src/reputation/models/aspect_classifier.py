"""Module 1 -- Aspect classification: what is each review actually about?

Yelp reviews carry a star rating but no aspect labels, so supervised training
data does not exist off the shelf. We therefore use **weak supervision**:

1. A hand-written seed lexicon produces noisy multi-label targets
   (``service``, ``food_quality``, ``cleanliness``, ``price_value``,
   ``wait_time``) for every review.
2. A TF-IDF + one-vs-rest logistic regression model is trained on those noisy
   labels. The classifier generalises past the lexicon -- it learns that
   "waited forever for the bill" is a wait-time complaint even though none of
   those words are seed terms -- and outputs calibrated probabilities instead of
   a brittle yes/no keyword hit.

Polarity is taken from the star rating rather than a separate sentiment model:
a review with ``stars <= NEGATIVE_STAR_THRESHOLD`` that mentions an aspect is
counted as a complaint about that aspect. This is the standard weak-labelling
trick for Yelp and it keeps the pipeline cheap enough to retrain nightly in a
container. The trade-off (a 1-star review may be angry about only one of the
two aspects it mentions) is quantified in the evaluation section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from ..config import ASPECTS, NEGATIVE_STAR_THRESHOLD, SETTINGS

# --------------------------------------------------------------------------- #
# Seed lexicon used only to bootstrap the weak labels
# --------------------------------------------------------------------------- #

ASPECT_LEXICON: dict[str, tuple[str, ...]] = {
    "service": (
        "service", "staff", "waiter", "waitress", "server", "manager", "rude",
        "friendly", "attentive", "polite", "hostess", "cashier", "greeted",
    ),
    "food_quality": (
        "food", "taste", "tasty", "flavour", "flavor", "bland", "fresh", "stale",
        "undercooked", "overcooked", "portion", "dish", "delicious", "soggy",
        "cold food", "reheated",
    ),
    "cleanliness": (
        "clean", "dirty", "filthy", "hygiene", "hygienic", "toilet", "restroom",
        "sticky", "smell", "roach", "cockroach", "fly", "flies", "hair in",
        "spotless", "sanitary",
    ),
    "price_value": (
        "price", "priced", "overpriced", "expensive", "cheap", "value", "worth",
        "cost", "bill", "pricey", "affordable", "rip off",
    ),
    "ambience": (
        "ambience", "ambiance", "atmosphere", "decor", "music", "noisy", "noise",
        "loud", "lighting", "cosy", "cozy", "cramped", "seating", "vibe",
        "crowded", "comfortable", "interior",
    ),
    "wait_time": (
        "wait", "waited", "waiting", "queue", "slow", "long line", "quick",
        "fast", "minutes", "delay", "seated", "reservation",
    ),
}

_TOKEN_RE = re.compile(r"[a-z']+")


def normalise(text: str) -> str:
    """Lowercase and strip punctuation -- shared by the lexicon and the model."""
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def weak_label(texts: pd.Series) -> pd.DataFrame:
    """Return a 0/1 mention matrix (n_reviews x n_aspects) from the seed lexicon.

    Multi-label by construction: one review can mention several aspects, which
    is exactly what happens in practice ("slow service and cold food").
    """
    normalised = texts.map(normalise)
    out = {}
    for aspect, terms in ASPECT_LEXICON.items():
        pattern = "|".join(re.escape(t) for t in terms)
        out[aspect] = normalised.str.contains(pattern, regex=True).astype(int)
    return pd.DataFrame(out, index=texts.index)[list(ASPECTS)]


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #


@dataclass
class AspectClassifier:
    """TF-IDF + one-vs-rest logistic regression over the aspect taxonomy.

    Chosen over a transformer baseline (e.g. fine-tuned DistilBERT) as the
    default because it trains in seconds on CPU inside a modest container,
    which keeps the whole pipeline reproducible on a free-tier cloud VM. The
    interface below is model-agnostic, so swapping in a transformer for the
    accuracy comparison in Section 4 only touches this class.
    """

    threshold: float = SETTINGS.aspect_threshold
    pipeline: Pipeline | None = None

    def build(self) -> Pipeline:
        """Construct the (untrained) sklearn pipeline."""
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalise,
                        ngram_range=SETTINGS.ngram_range,
                        max_features=SETTINGS.max_features,
                        min_df=SETTINGS.min_df,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            max_iter=1000,
                            C=4.0,
                            class_weight="balanced",
                            random_state=SETTINGS.random_state,
                        ),
                        n_jobs=None,
                    ),
                ),
            ]
        )

    def fit(self, texts: pd.Series, labels: pd.DataFrame | None = None) -> "AspectClassifier":
        """Fit on weak labels (generated here when ``labels`` is omitted)."""
        y = weak_label(texts) if labels is None else labels[list(ASPECTS)]
        self.pipeline = self.build()
        self.pipeline.fit(texts, y.to_numpy())
        return self

    def predict_proba(self, texts: pd.Series) -> pd.DataFrame:
        """P(review mentions aspect) for every aspect, as a tidy DataFrame."""
        if self.pipeline is None:
            raise RuntimeError("AspectClassifier.fit() must be called first")
        probs = self.pipeline.predict_proba(texts)
        return pd.DataFrame(probs, columns=list(ASPECTS), index=texts.index)

    def predict_mentions(self, texts: pd.Series) -> pd.DataFrame:
        """Binarised mentions at the configured probability threshold."""
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def top_terms(self, aspect: str, n: int = 12) -> list[tuple[str, float]]:
        """Highest-weight n-grams for one aspect -- used for model transparency.

        Being able to show a client *why* a review was tagged "cleanliness"
        is what makes the downstream recommendation credible, so this is part
        of the product, not just a debugging aid.
        """
        if self.pipeline is None:
            raise RuntimeError("AspectClassifier.fit() must be called first")
        idx = list(ASPECTS).index(aspect)
        vectoriser: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        coefs = self.pipeline.named_steps["clf"].estimators_[idx].coef_[0]
        vocabulary = np.array(vectoriser.get_feature_names_out())
        order = np.argsort(coefs)[::-1][:n]
        return [(str(vocabulary[i]), float(coefs[i])) for i in order]


def annotate_reviews(
    reviews: pd.DataFrame, classifier: AspectClassifier
) -> pd.DataFrame:
    """Attach aspect mention/complaint flags to a review frame.

    Adds, for each aspect ``a``:
      * ``mention_a``   -- 1 if the review discusses that aspect
      * ``complaint_a`` -- 1 if it discusses it *and* the review is negative

    The complaint flags are the raw material for both the emerging-complaint
    detector (Module 2) and the decline predictor (Module 3).
    """
    mentions = classifier.predict_mentions(reviews["text"])
    annotated = reviews.copy()
    is_negative = (annotated["stars"] <= NEGATIVE_STAR_THRESHOLD).astype(int)
    annotated["is_negative"] = is_negative
    for aspect in ASPECTS:
        annotated[f"mention_{aspect}"] = mentions[aspect]
        annotated[f"complaint_{aspect}"] = mentions[aspect] * is_negative
    return annotated
