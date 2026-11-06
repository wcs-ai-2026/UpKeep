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
    train = df[df["split"] == "train"]

    # Calling Lightgbm model
    model = LightGBMNormalModel()
    model.fit(train[FEATURES], train[TARGET])
    model.save(f"models_saved/lgbm_gearbox_bearing_temp_{dataset}.joblib")

    # Predicting the features.
    pred = model.predict(df[FEATURES])
    out = pd.DataFrame({
        "reading_id": df["row_id"],
        "actual": df[TARGET],
        "expected": pred,
        "residual": df[TARGET] - pred,
    })

    # Storing result into a CSV file.
    out.to_csv(f"predictions/lgbm_gearbox_bearing_temp_{dataset}.csv", index=False)

    # Testing all the predictions.
    test = df[df["split"] == "prediction"]
    test_mae = (test[TARGET] - model.predict(test[FEATURES])).abs().mean()
    print(f"Dataset {dataset}: held-out MAE = {test_mae:.3f} degC")