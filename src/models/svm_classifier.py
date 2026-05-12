from __future__ import annotations

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False


class LinguisticSVM:
    """
    SLI classifier using hand-crafted linguistic features (MLU, TTR, MATTR, …).
    Applies SMOTE to handle the SLI/TD class imbalance when imbalanced-learn is installed.
    """

    _PARAM_GRID = [
        {"clf__C": [0.1, 1, 10, 50], "clf__kernel": ["linear"], "clf__class_weight": ["balanced"]},
        {"clf__C": [0.1, 1, 10, 50], "clf__kernel": ["rbf"], "clf__gamma": ["scale", 0.01, 0.001], "clf__class_weight": ["balanced"]},
    ]

    def __init__(self, use_smote: bool = True, cv: int = 5):
        self.use_smote = use_smote and _SMOTE_AVAILABLE
        self.cv = cv
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.clf: SVC | None = None

    def fit(self, X: np.ndarray, labels: list[str]) -> "LinguisticSVM":
        y = self.le.fit_transform(labels)
        X_scaled = self.scaler.fit_transform(X)

        if self.use_smote:
            smote = SMOTE(random_state=42)
            X_scaled, y = smote.fit_resample(X_scaled, y)

        grid = GridSearchCV(
            Pipeline([("clf", SVC(probability=True))]),
            self._PARAM_GRID,
            cv=min(self.cv, _min_class_count(y)),
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(X_scaled, y)
        self.clf = grid.best_estimator_
        print(f"Best SVM params: {grid.best_params_}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.le.inverse_transform(self.clf.predict(X_scaled))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        proba = self.clf.predict_proba(X_scaled)
        sli_idx = list(self.le.classes_).index("SLI")
        return proba[:, sli_idx]


def _min_class_count(y: np.ndarray) -> int:
    unique, counts = np.unique(y, return_counts=True)
    return int(counts.min())
