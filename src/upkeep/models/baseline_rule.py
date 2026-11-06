# Fixed-limit baseline model.
# The model that learns a fixed temperature threshold from training data,
# Predicts that same threshold for every reading and supports saving and loading the model.

from pathlib import Path
import joblib
import numpy as np


class FixedLimitBaseline:
    def __init__(self, margin_std=0.0):
        # margin_std = How many standard deviations of extra tolerance to allow.
        self.margin_std = margin_std
        self.fixed_value = None

    # Training function to calculate the fixed temperature using healthy training temperature.
    def fit(self, X, y):
        # mean = Average healthy temperature.
        mean = y.mean()

        # std = How much temperatures normally vary.
        std = y.std()

        # Fixed value
        self.fixed_value = mean + self.margin_std * std

    # Predicting new turbine readining based on learned temperature.
    def predict(self, X):
        # Same predicted value for every row
        return np.full(len(X), self.fixed_value)

    # Saving the trained model.
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Saves:
        # fixed_value = Learned temperature limit.
        # margin_std = Standard deviation multiplier.
        joblib.dump({"fixed_value": self.fixed_value, "margin_std": self.margin_std}, path)

    @classmethod
    # To restore the saved temperature limit.
    def load(cls, path):
        data = joblib.load(path)
        model = cls(margin_std=data["margin_std"])
        model.fixed_value = data["fixed_value"]
        return model
