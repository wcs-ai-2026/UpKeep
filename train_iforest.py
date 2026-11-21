import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from upkeep.health.noise import calibration_split, load_cleaned
from upkeep.models.iforest import IsolationForestAnomalyModel
from upkeep.parts import PART_SENSORS, PARTS
from upkeep.stream.model_store import ModelStore

DATASETS = ["0", "10", "25"]


# Adding the engineered features (lag, gust_factor, etc) onto the raw
# readings. The models need these columns and they only live in the
# features parquet, same as the other training scripts.
# Only keeping the columns we need from readings first, so merging does
# not clash with the raw sensor columns of the same name in the parquet.
def with_features(df, dataset):
    targets = [PART_SENSORS[part] for part in PARTS]
    readings = df[["row_id", "timestamp", "split"] + targets]
    feats = pd.read_parquet(f"data/processed/features_{dataset}.parquet")
    return readings.merge(feats, on="row_id")


# Training a fresh model on the train split, then predicting on the
# calibration window only, so these residuals are out of sample.
def out_of_sample_residuals(df, part, deployed_model):
    target = PART_SENSORS[part]
    features = list(deployed_model.features)
    fit, calibration, _ = calibration_split(df, target)

    model = type(deployed_model)()
    model.fit(fit[features], fit[target])

    usable = calibration.dropna(subset=features)
    pred = model.predict(usable[features])
    residual = usable[target].to_numpy(dtype="float64") - np.asarray(pred, dtype="float64")
    return pd.DataFrame({"row_id": usable["row_id"].to_numpy(), part: residual})


# Using the already deployed model to get residuals for every reading.
def deployed_residuals(df, part, deployed_model):
    target = PART_SENSORS[part]
    features = list(deployed_model.features)
    usable = df.dropna(subset=features + [target])
    pred = deployed_model.predict(usable[features])
    residual = usable[target].to_numpy(dtype="float64") - np.asarray(pred, dtype="float64")
    return pd.DataFrame({"row_id": usable["row_id"].to_numpy(), part: residual})


for dataset in DATASETS:
    df = with_features(load_cleaned(dataset), dataset)
    store = ModelStore("lgbm", parts=PARTS)
    deployed = {part: store.get(dataset, part) for part in PARTS}

    # Building the out of sample residual table, one column per part.
    oos = out_of_sample_residuals(df, PARTS[0], deployed[PARTS[0]])
    for part in PARTS[1:]:
        oos = oos.merge(out_of_sample_residuals(df, part, deployed[part]), on="row_id")

    # Training the Isolation Forest model
    model = IsolationForestAnomalyModel()
    model.fit(oos[list(PARTS)])
    model.save(f"models_saved/iforest_{dataset}.joblib")

    # Scoring every reading using the deployed models' residuals.
    scored = deployed_residuals(df, PARTS[0], deployed[PARTS[0]])
    for part in PARTS[1:]:
        scored = scored.merge(deployed_residuals(df, part, deployed[part]), on="row_id")

    anomaly_score = model.predict_score(scored[list(PARTS)])

    # Storing result into a CSV file.
    out = pd.DataFrame({
        "reading_id": scored["row_id"],
        "anomaly_score": anomaly_score,
    }).merge(df[["row_id", "split"]].rename(columns={"row_id": "reading_id"}), on="reading_id")
    out = out[["reading_id", "split", "anomaly_score"]]
    out.to_csv(f"predictions/iforest_{dataset}.csv", index=False)

    print(f"Dataset {dataset}: trained on {len(oos)} out-of-sample residual rows, scored {len(out)} readings")
