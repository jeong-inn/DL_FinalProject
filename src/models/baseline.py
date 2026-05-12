from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class TfidfBaseline:
    """TF-IDF (1-2 gram) + Logistic Regression."""

    def __init__(self, max_features: int = 10_000, C: float = 1.0):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=max_features,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
            )),
        ])
        self.le = LabelEncoder()

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfBaseline":
        y = self.le.fit_transform(labels)
        self.pipeline.fit(texts, y)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.le.inverse_transform(self.pipeline.predict(texts))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        proba = self.pipeline.predict_proba(texts)
        sli_idx = list(self.le.classes_).index("SLI")
        return proba[:, sli_idx]
