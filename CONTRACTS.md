# Upkeep Code Contracts

This document explains how the different parts of Upkeep communicate with each other. Everyone should follow these formats so the components can work together.

## 1. Reading Format

**Owner:** Everyone

Each reading represents one 10-minute sensor reading from a turbine.

The original dataset columns are converted into standardized names using Satujan's `remap.py`.

```python
{
    "row_id": 1001,
    "turbine_id": "A-T3",
    "dataset_id": "A-68",
    "timestamp": "2022-03-14T06:20:00",
    "wind_speed_avg": 11.2,
    "wind_speed_max": 17.9,
    "wind_speed_std": 2.4,
    "grid_power_avg": 1830.0,
    "rotor_rpm_avg": 15.1,
    "ambient_temp": 9.5,
    "gearbox_bearing_temp": 64.2,
    "generator_bearing_1_temp": 58.7,
    "generator_bearing_2_temp": 59.1,
    "status_id": 1
}
```

All sensor names should follow `SENSORS.md` and `remap.py`.

## 2. Features Format

**Owner:** Satujan

Function:

```python
def make_features(readings: pd.DataFrame) -> pd.DataFrame:
```

Takes readings for one turbine, sorted by timestamp, and returns one row per `row_id`.

The features include:

- `wind_speed_avg`
- `grid_power_avg`
- `ambient_temp`
- `gust_factor` (maximum / average wind speed)
- `turbulence` (wind speed standard deviation / average wind speed)
- `wind_jump` (change from the previous reading)
- `gusts_last_24h`
- `gusts_last_7d`
- `temp_1h_ago`
- `hour_of_day`

`gust_events` is a temporary variable used to calculate gust counts. It is 1 when `gust_factor >= 1.5`, 0 otherwise, and missing when gust factor is missing.

Gust counts exclude the current reading. Empty windows return 0.

`temp_1h_ago` uses `gearbox_bearing_temp.shift(6)` and is missing unless six readings span exactly one hour.

The complete feature list is maintained in `FEATURES.md`.

Output is saved as:

`data/processed/features_<dataset>.parquet`

## 3. Model Format

**Owner:** Varun

Every normal-behaviour model must follow the same class structure.

```python
class NormalModel:

    def fit(self, X, y):
        pass

    def predict(self, X) -> np.ndarray:
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
```

The model predicts the expected temperature of a turbine part in Celsius.

The residual is calculated as:

```python
residual = actual_temperature - expected_temperature
```

Predictions are saved as CSV files:

`predictions/<model>_<part>_<dataset>.csv`

Each CSV contains:

- `row_id`
- `actual`
- `expected`
- `residual`

The Isolation Forest follows the same general structure but uses `predict_score()` instead of `predict()`.

The model should distinguish between gearbox bearing, generator bearing 1, and generator bearing 2.

## 4. Explain Format

**Owner:** Karim

Function:

```python
def explain(recent: pd.DataFrame, residuals: dict, gusts: dict) -> dict:
```

Returns the reasons behind a turbine warning.

Expected output:

```python
{
    "reasons": [
        {
            "part": "gearbox_bearing",
            "residual_c": 6.1,
            "hours": 72,
            "text": "Gearbox bearing 6°C hotter than expected for 3 days"
        }
    ],
    "gust_context": "14 high-gust periods this week",
    "sentence": "Gearbox bearing has run 6°C hotter than expected for 3 days, after a gusty week."
}
```

The reasons list contains up to three reasons.

Each reason should explain which part has a problem, how far its temperature is from expected, and how long the problem has lasted.

## 5. Health Status Format

**Owner:** Nmesoma

Function:

```python
def score_reading(reading) -> dict:
```

This function connects the features, models, anomaly detection and explanations to produce the final health status of a turbine.

Expected output:

```python
{
    "row_id": 1001,
    "turbine_id": "A-T3",
    "timestamp": "2022-03-14T06:20:00",
    "health": "yellow",
    "health_score": 0.64,
    "residuals": {
        "gearbox_bearing": 6.1,
        "generator_bearing_1": 1.2,
        "generator_bearing_2": 0.8
    },
    "anomaly_score": 0.58,
    "gusts_last_7d": 14,
    "reasons": [],
    "sentence": "Gearbox bearing has run hotter than expected.",
    "fault_event": "A-68",
    "days_to_fault": 5.3,
    "latency_ms": 120
}
```

Health levels:

- green: Normal operation
- yellow: Possible problem detected
- red: Serious problem detected

Each result is written as one line in:

`data/live/health.jsonl`

The `fault_event` and `days_to_fault` fields are only used for dashboard display and evaluation. Models must never use them as inputs.

The exact health-score calculation and classification thresholds will be agreed upon before final integration.

## General Rules

1. Everyone must follow these formats when connecting their code.
2. Changes to a contract must be discussed and submitted through a pull request.
3. Each developer is responsible for testing their own component.
4. The main branch should always remain working.
