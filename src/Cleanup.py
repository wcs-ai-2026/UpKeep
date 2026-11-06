#Functions to cleanup and add features to datasets

import pandas as pd

from remap import sensor_names


GFT = 1.5 #gusty when 1.5x the average


def load_data(path):
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns=sensor_names)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def clean_data(df):
    df = df[df["status_id"] == 0].copy()
    required_columns = [
        "timestamp", "wind_speed_avg", "wind_speed_max", "wind_speed_std",
        "grid_power_avg", "ambient_temp", "gearbox_bearing_temp",
    ]
    df = df.dropna(subset=required_columns)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def make_features(df):
    df = df.copy()
    average_wind = df["wind_speed_avg"].replace(0, float("nan"))
    df["gust_factor"] = df["wind_speed_max"] / average_wind
    df["turbulence"] = df["wind_speed_std"] / average_wind
    df["wind_jump"] = df["wind_speed_avg"].diff()

    gust_events = (df["gust_factor"] >= GFT).astype(float)
    gust_events = gust_events.where(df["gust_factor"].notna())
    gust_events.index = pd.DatetimeIndex(df["timestamp"])
    df["gusts_last_24h"] = gust_events.rolling("24h", closed="left", min_periods=1).sum().fillna(0).to_numpy()
    df["gusts_last_7d"] = gust_events.rolling("7D", closed="left", min_periods=1).sum().fillna(0).to_numpy()

    df["temp_1h_ago"] = df["gearbox_bearing_temp"].shift(6)
    one_hour_apart = df["timestamp"] - df["timestamp"].shift(6) == pd.Timedelta(hours=1)
    df["temp_1h_ago"] = df["temp_1h_ago"].where(one_hour_apart)
    df["hour_of_day"] = df["timestamp"].dt.hour
    return df


#Dataset 0: Generator bearing fault Dataset 10: Gearbox Failure, Dataset 25: Healthy
for dataset_name in ["0", "10", "25"]:
    df = load_data("Data/datasets/" + dataset_name + ".csv")
    df = clean_data(df)
    df = make_features(df)
    df.to_csv("Data/cln/" + dataset_name + "_cln.csv", sep=";", index=False)
    print(dataset_name + "_cln.csv saved with", len(df), "rows")


