# Linear regression model.
# Predicts a part's expected temperature
# Model that learns how operating conditions (wind speed, power output, outside temperature, etc.) 
# Affect normal turbine component temperatures

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class LinearNormalModel:
    def __init__(self):
        self.model = LinearRegression()
        self.features = None

    # Training the linear regression model
    # X - Input features 
    # Y - actual temperature
    def fit(self, X, y):
        self.features = list(X.columns)
        self.model.fit(X, y)

    # Predicting expected temperature 
    def predict(self, X):
        return self.model.predict(X[self.features])

    # Saving the results
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Saves:
        # model = Learned trainined model.
        # features = Feature names
        joblib.dump({"model": self.model, "features": self.features}, path)

    @classmethod
    # To restore the saved model.
    def load(cls, path):
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.features = data["features"]
        return instance
    