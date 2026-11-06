import numpy as np
import pandas as pd
import pickle


def make_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Create temporary features using Satujan's feature definitions."""

    features = pd.DataFrame(index=readings.index)

    features["wind_speed_avg"] = readings["wind_speed_avg"]
    features["grid_power_avg"] = readings["grid_power_avg"]
    features["ambient_temp"] = readings["ambient_temp"]

    avg_wind = readings["wind_speed_avg"].replace(0, np.nan)

    features["gust_factor"] = readings["wind_speed_max"] / avg_wind
    features["turbulence"] = readings["wind_speed_std"] / avg_wind
    features["wind_jump"] = readings["wind_speed_avg"].diff()

    # Temporary variable used to calculate gust counts
    gust_events = (features["gust_factor"] >= 1.5).astype(float)
    gust_events[features["gust_factor"].isna()] = np.nan

    timestamps = pd.to_datetime(readings["timestamp"])

    features["gusts_last_24h"] = 0
    features["gusts_last_7d"] = 0

    for i in range(len(readings)):
        previous_24h = (
            (timestamps < timestamps.iloc[i]) &
            (timestamps >= timestamps.iloc[i] - pd.Timedelta(hours=24))
        )

        previous_7d = (
            (timestamps < timestamps.iloc[i]) &
            (timestamps >= timestamps.iloc[i] - pd.Timedelta(days=7))
        )

        features.iloc[i, features.columns.get_loc("gusts_last_24h")] = (
            gust_events[previous_24h].sum()
        )

        features.iloc[i, features.columns.get_loc("gusts_last_7d")] = (
            gust_events[previous_7d].sum()
        )

    # Temperature from exactly one hour earlier
    gearbox_temp = readings["gearbox_bearing_temp"]
    temp_1h_ago = gearbox_temp.shift(6)

    valid_time = (
        timestamps - timestamps.shift(6) == pd.Timedelta(hours=1)
    )

    features["temp_1h_ago"] = temp_1h_ago.where(valid_time)
    features["hour_of_day"] = timestamps.dt.hour

    return features


class NormalModel:
    """Temporary model that returns a fixed temperature."""

    def __init__(self, temperature=60.0):
        self.temperature = temperature

    def fit(self, X, y):
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.temperature)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            return pickle.load(file)


def explain(recent: pd.DataFrame, residuals: dict, gusts: dict) -> dict:
    """Return a temporary explanation for a turbine warning."""

    reasons = []

    for part, residual in residuals.items():
        if pd.notna(residual) and residual > 5:
            reasons.append({
                "part": part,
                "residual_c": float(residual),
                "hours": 72,
                "text": (
                    f"{part.replace('_', ' ').title()} "
                    f"{residual:.1f}°C hotter than expected for 3 days"
                )
            })

    reasons = reasons[:3]

    gust_count = gusts.get("gusts_last_7d", 0)
    gust_context = f"{gust_count} high-gust periods this week"

    if reasons:
        sentence = (
            f"{reasons[0]['part'].replace('_', ' ').title()} has run "
            f"{reasons[0]['residual_c']:.1f}°C hotter than expected "
            "for 3 days, after a gusty week."
        )
    else:
        sentence = "No unusual temperature readings detected."

    return {
        "reasons": reasons,
        "gust_context": gust_context,
        "sentence": sentence
    }


def score_reading(reading) -> dict:
    """Return a temporary turbine health record."""

    residuals = {
        "gearbox_bearing": 6.1,
        "generator_bearing_1": 1.2,
        "generator_bearing_2": 0.8
    }

    explanation = explain(
        pd.DataFrame([reading]),
        residuals,
        {"gusts_last_7d": 14}
    )

    return {
        "row_id": reading["row_id"],
        "turbine_id": reading["turbine_id"],
        "timestamp": str(reading["timestamp"]),
        "health": "yellow",
        "health_score": 0.64,
        "residuals": residuals,
        "anomaly_score": 0.58,
        "gusts_last_7d": 14,
        "reasons": explanation["reasons"],
        "sentence": explanation["sentence"],
        "fault_event": None,
        "days_to_fault": None,
        "latency_ms": 0
    }
