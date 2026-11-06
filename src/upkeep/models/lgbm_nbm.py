# LightGBM model.
# Predicts the gearbox temperature the same way as linear_nbm.py but using gradient-boosted trees instead of a straight line.

from pathlib import Path
import joblib
from lightgbm import LGBMRegressor


class LightGBMNormalModel:
    def __init__(self):
        self.model = LGBMRegressor(n_estimators=200, learning_rate=0.05, verbosity=-1)
        self.features = None

    # Training the model
    def fit(self, X, y):
        self.features = list(X.columns)
        self.model.fit(X, y)

    # Predicting expected temperature
    def predict(self, X):
        return self.model.predict(X[self.features])

    # Saving the trained model
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    # Restoring the saved model
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.features = data["features"]
        return instance