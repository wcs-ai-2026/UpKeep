import pandas as pd

# Storing all the required targets and datasets.
TARGETS = ["generator_bearing_1_temp", "generator_bearing_2_temp"]
DATASETS = ["0", "10", "25"]

# Loading the fault info for every dataset.
events = pd.read_csv("Data/event_info.csv", sep=";").set_index("event_id")

# Looping through each target and dataset.
rows = []
for target in TARGETS:
    for dataset in DATASETS:
        df = pd.read_csv(f"Data/cln/{dataset}_cln.csv", sep=";")
        pred = pd.read_csv(f"predictions/lgbm_{target}_{dataset}.csv")

        # Keeping only the held-out test rows.
        pred = pred.rename(columns={"reading_id": "row_id"})
        pred = pred.merge(df[["row_id", "split"]], on="row_id", how="left")
        test = pred[pred["split"] == "prediction"].copy()

        # Working out which rows are healthy vs during a fault.
        event = events.loc[int(dataset)]
        if str(event["event_label"]).strip().lower() == "anomaly":
            fault_start_row_id = int(event["event_start_id"])
            test["period"] = test["row_id"].apply(lambda r: "fault" if r >= fault_start_row_id else "healthy")
        else:
            test["period"] = "healthy"

        # Computing MAE separately for each period.
        for period in ("healthy", "fault"):
            sub = test[test["period"] == period]
            mae = sub["residual"].abs().mean() if len(sub) else float("nan")
            rows.append({"target": target, "dataset": dataset, "period": period, "n_rows": len(sub), "mae": mae})

# Storing result into a CSV file.
out = pd.DataFrame(rows)
out.to_csv("generator_healthy_fault_mae.csv", index=False)
print(out.to_string(index=False))
