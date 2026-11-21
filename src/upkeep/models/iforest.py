# Isolation Forest model.
# Looks at all 3 parts' residuals together for one reading and gives one
# score for how unusual that reading looks overall.
# Same shape as the other models, but uses predict_score() instead of
# predict() since there is no single expected temperature here.

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from upkeep.ids import dataset_number
from upkeep.parts import PARTS
from upkeep.paths import models_dir


class IsolationForestAnomalyModel:
    def __init__(self):
        self.model = IsolationForest(n_estimators=200, random_state=42)
        self.features = None
        # Used to turn the raw score into a plain 0-1 number.
        self.score_lo = None
        self.score_hi = None

    # Training the Isolation Forest on a table of residuals.
    def fit(self, X, y=None):
        self.features = list(X.columns)
        self.model.fit(X[self.features])

        # Working out the 0-1 scale from the training data itself.
        raw = -self.model.score_samples(X[self.features])
        self.score_lo = float(np.percentile(raw, 1))
        self.score_hi = float(np.percentile(raw, 99))
        return self

    # Scoring how unusual each row looks, from 0 (normal) to 1 (very unusual).
    def predict_score(self, X) -> np.ndarray:
        raw = -self.model.score_samples(X[self.features])
        spread = self.score_hi - self.score_lo
        if spread <= 0:
            return np.clip(raw - self.score_lo, 0.0, 1.0)
        scaled = (raw - self.score_lo) / spread
        return np.clip(scaled, 0.0, 1.0)

    # Saving the trained model
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "features": self.features,
            "score_lo": self.score_lo,
            "score_hi": self.score_hi,
        }, path)

    @classmethod
    # Restoring the saved model
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.features = data["features"]
        instance.score_lo = data["score_lo"]
        instance.score_hi = data["score_hi"]
        return instance


# Plugs this model into live.py's AnomalyScorer slot. Loads one model
# per dataset, same idea as ModelStore.
class IForestScorer:
    def __init__(self, models_root=None, parts=PARTS):
        self.models_root = Path(models_root) if models_root is not None else models_dir()
        self.parts = tuple(parts)
        self._cache = {}

    # Working out the file path for one dataset's model.
    def _path(self, dataset_id):
        return self.models_root / f"iforest_{dataset_number(dataset_id)}.joblib"

    # Loading a dataset's model once, then reusing it.
    def _model(self, dataset_id):
        n = dataset_number(dataset_id)
        if n not in self._cache:
            path = self._path(n)
            self._cache[n] = IsolationForestAnomalyModel.load(path) if path.is_file() else None
        return self._cache[n]

    # Matches live.py's score(dataset_id, residuals, reading) -> float or None.
    def score(self, dataset_id, residuals, reading):
        model = self._model(dataset_id)
        if model is None:
            return None

        # Never guessing a missing part's residual.
        row = {}
        for part in self.parts:
            value = residuals.get(part)
            if value is None:
                return None
            row[part] = value

        X = pd.DataFrame([row], columns=self.parts)
        try:
            return float(model.predict_score(X)[0])
        except Exception:
            return None
