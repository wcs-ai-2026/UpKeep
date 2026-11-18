import pandas as pd
import sys
sys.path.insert(0, "src/upkeep/models")
from lgbm_nbm import LightGBMNormalModel

# Storing the shared features. Each target gets its own lag feature below,
# instead of temp_1h_ago, since that one is actually the gearbox's temperature.
BASE_FEATURES = ["wind_speed_avg", "grid_power_avg", "ambient_temp", "gust_factor", "turbulence", "wind_jump", "gusts_last_24h", "gusts_last_7d", "hour_of_day"]
TARGETS = ["generator_bearing_1_temp", "generator_bearing_2_temp"]
DATASETS = ["0", "10", "25"]

# Looping through each target and dataset.
for target in TARGETS:
    lag_feature = f"{target}_1h_ago"
    FEATURES = BASE_FEATURES + [lag_feature]

    for dataset in DATASETS:
        readings = pd.read_csv(f"Data/cln/{dataset}_cln.csv", sep=";")
        feats = pd.read_parquet(f"data/processed/features_{dataset}.parquet")
        df = readings[["row_id", "split", target]].merge(feats, on="row_id")

        # Dropping rows with no real target value - never fit or predict on a made-up number.
        before = len(df)
        df = df.dropna(subset=[target])
        dropped = before - len(df)

        train = df[df["split"] == "train"]

        # Calling Lightgbm model
        model = LightGBMNormalModel()
        model.fit(train[FEATURES], train[target])
        model.save(f"models_saved/lgbm_{target}_{dataset}.joblib")

        # Predicting the features.
        pred = model.predict(df[FEATURES])
        out = pd.DataFrame({
            "reading_id": df["row_id"],
            "split": df["split"],
            "actual": df[target],
            "expected": pred,
            "residual": df[target] - pred,
        })

        # Storing result into a CSV file.
        out.to_csv(f"predictions/lgbm_{target}_{dataset}.csv", index=False)

        print(f"{target} | Dataset {dataset}: trained on {len(train)} rows, dropped {dropped} missing-target row(s), wrote {len(out)} predictions")
