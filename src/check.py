#checks for all wind columns in each dataset

import pandas as pd


df = pd.read_csv("Data/datasets/0.csv", sep=";")
wind_columns = df.filter(like="wind_speed").columns.tolist()
print("Available wind speed columns:", wind_columns)

frames = []
datasets = [
    "0", "10", "13", "14", "17", "22", "24", "25", "26", "3", "38",
    "40", "42", "45", "51", "68", "69", "71", "72", "73", "84", "92",
]

for dataset in datasets:
    df = pd.read_csv("Data/datasets/" + dataset + ".csv", sep=";", usecols=wind_columns)
    print("Dataset:", dataset)
    print(df.isna().sum())
    frames.append(df)

all_wind_data = pd.concat(frames, ignore_index=True)
print("Total rows:", len(all_wind_data))
print("Total NaNs:")
print(all_wind_data.isna().sum())
