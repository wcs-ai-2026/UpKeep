import pandas as pd
import sys
sys.path.insert(0, "src/upkeep/models")
from lgbm_nbm import LightGBMNormalModel

# Storing all the required features.
FEATURES = ["wind_speed_avg", "grid_power_avg", "ambient_temp", "gust_factor", "turbulence", "wind_jump", "gusts_last_24h", "gusts_last_7d", "temp_1h_ago", "hour_of_day"]
TARGET = "gearbox_bearing_temp"
DATASETS = ["0", "10", "25"]

# Looping through the dataset.
for dataset in DATASETS:
    df = pd.read_csv(f"Data/cln/{dataset}_cln.csv", sep=";")

    # Dropping rows with no real target value - never fit or predict on a made-up number.
    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped = before - len(df)

    train = df[df["split"] == "train"]

    # Calling Lightgbm model
    model = LightGBMNormalModel()
    model.fit(train[FEATURES], train[TARGET])
    model.save(f"models_saved/lgbm_{TARGET}_{dataset}.joblib")

    # Predicting the features.
    pred = model.predict(df[FEATURES])
    out = pd.DataFrame({
        "reading_id": df["row_id"],
        "split": df["split"],
        "actual": df[TARGET],
        "expected": pred,
        "residual": df[TARGET] - pred,
    })

    # Storing result into a CSV file.
    out.to_csv(f"predictions/lgbm_{TARGET}_{dataset}.csv", index=False)

    print(f"Dataset {dataset}: trained on {len(train)} rows, dropped {dropped} missing-target row(s), wrote {len(out)} predictions")
