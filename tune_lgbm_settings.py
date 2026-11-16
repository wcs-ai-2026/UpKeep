import pandas as pd
from lightgbm import LGBMRegressor

# Storing all the required features.
FEATURES = ["wind_speed_avg", "grid_power_avg", "ambient_temp", "gust_factor", "turbulence", "wind_jump", "gusts_last_24h", "gusts_last_7d", "temp_1h_ago", "hour_of_day"]
TARGETS = ["gearbox_bearing_temp", "generator_bearing_1_temp", "generator_bearing_2_temp", "gearbox_oil_temp"]
DATASETS = ["0", "10", "25"]

# The settings we're comparing.
CANDIDATES = [
    {"n_estimators": 100, "learning_rate": 0.1},
    {"n_estimators": 200, "learning_rate": 0.05},
    {"n_estimators": 300, "learning_rate": 0.05},
    {"n_estimators": 300, "learning_rate": 0.03},
    {"n_estimators": 500, "learning_rate": 0.03},
]

# Looping through every target, dataset and candidate setting.
rows = []
for target in TARGETS:
    for dataset in DATASETS:
        df = pd.read_csv(f"Data/cln/{dataset}_cln.csv", sep=";")
        df = df.dropna(subset=[target])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        train_all = df[df["split"] == "train"]

        # Validation = last 2 months of train, per the plan. Never touching the locked test rows.
        cutoff = train_all["timestamp"].max() - pd.DateOffset(months=2)
        actual_train = train_all[train_all["timestamp"] < cutoff]
        validation = train_all[train_all["timestamp"] >= cutoff]

        for params in CANDIDATES:
            model = LGBMRegressor(**params, verbosity=-1, random_state=42)
            model.fit(actual_train[FEATURES], actual_train[target])
            pred = model.predict(validation[FEATURES])
            val_mae = (validation[target] - pred).abs().mean()
            rows.append({
                "target": target, "dataset": dataset,
                "n_estimators": params["n_estimators"], "learning_rate": params["learning_rate"],
                "validation_mae": round(val_mae, 4),
            })

# Saving every result so the decision is reproducible.
results = pd.DataFrame(rows)
results.to_csv("tuning_results.csv", index=False)
print(results.to_string(index=False))
