import pandas as pd
import sys
sys.path.insert(0, "src/upkeep/models")
from lgbm_nbm import LightGBMNormalModel

# Storing all the required features.
FEATURES = ["wind_speed_avg", "grid_power_avg", "ambient_temp", "gust_factor", "turbulence", "wind_jump", "gusts_last_24h", "gusts_last_7d", "temp_1h_ago", "hour_of_day"]
TARGETS = ["generator_bearing_1_temp", "generator_bearing_2_temp"]
DATASETS = ["0", "10", "25"]

# Looping through each target and dataset.
for target in TARGETS:
    for dataset in DATASETS:
        df = pd.read_csv(f"Data/cln/{dataset}_cln.csv", sep=";")
        train = df[df["split"] == "train"]

        # Calling Lightgbm model
        model = LightGBMNormalModel()
        model.fit(train[FEATURES], train[target])
        model.save(f"models_saved/lgbm_{target}_{dataset}.joblib")

        # Predicting the features.
        pred = model.predict(df[FEATURES])
        out = pd.DataFrame({
            "reading_id": df["row_id"],
            "actual": df[target],
            "expected": pred,
            "residual": df[target] - pred,
        })

        # Storing result into a CSV file.
        out.to_csv(f"predictions/lgbm_{target}_{dataset}.csv", index=False)

        # Testing all the predictions.
        test = df[df["split"] == "prediction"]
        test_mae = (test[target] - model.predict(test[FEATURES])).abs().mean() if len(test) else float("nan")
        print(f"{target} | Dataset {dataset}: held-out MAE = {test_mae:.3f} degC")
