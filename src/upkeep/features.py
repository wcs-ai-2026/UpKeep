import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per reading for one turbine sorted by timestamp."""
    features = readings[["row_id"]].copy()

    features["wind_speed_avg"] = readings["wind_speed_avg"]
    features["grid_power_avg"] = readings["grid_power_avg"]
    features["ambient_temp"] = readings["ambient_temp"]

    average_wind = readings["wind_speed_avg"].replace(0, np.nan)
    features["gust_factor"] = readings["wind_speed_max"] / average_wind
    features["turbulence"] = readings["wind_speed_std"] / average_wind
    features["wind_jump"] = readings["wind_speed_avg"].diff()

    timestamps = pd.to_datetime(readings["timestamp"])
    gust_events = features["gust_factor"].ge(1.5).astype(float)
    gust_events = gust_events.where(features["gust_factor"].notna())
    # Combine readings at the same timestamp so the current timestamp is excluded.
    gusts_by_time = gust_events.groupby(timestamps, sort=False).sum()
    for window, column in (("24h", "gusts_last_24h"), ("7D", "gusts_last_7d")):
        counts = gusts_by_time.rolling(window, closed="left").sum()
        features[column] = timestamps.map(counts).fillna(0).astype("int64")

    one_hour_apart = timestamps - timestamps.shift(6) == pd.Timedelta(hours=1)
    features["temp_1h_ago"] = readings["gearbox_bearing_temp"].shift(6).where(one_hour_apart)
    features["hour_of_day"] = timestamps.dt.hour

    for sensor in ("generator_bearing_1_temp", "generator_bearing_2_temp"):
        lag = f"{sensor}_1h_ago"
        if sensor in readings:
            features[lag] = readings[sensor].shift(6).where(one_hour_apart)
        else:
            features[lag] = np.nan
    return features


def main():
    parser = argparse.ArgumentParser(description="Build feature Parquet files from cleaned datasets.")
    parser.add_argument("datasets", nargs="+", type=int, help="Cleaned dataset IDs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        source = root / "Data" / "cln" / f"{dataset}_cln.csv"
        readings = pd.read_csv(source, sep=";", parse_dates=["timestamp"])
        features = make_features(readings.sort_values("timestamp", kind="stable"))
        target = output / f"features_{dataset}.parquet"
        features.to_parquet(target, index=False)
        print(f"Saved {target} ({len(features):,} rows)")


if __name__ == "__main__":
    main()
