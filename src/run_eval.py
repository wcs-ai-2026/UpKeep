from pathlib import Path
import pandas as pd


scores = []



project_root = Path(__file__).resolve().parents[1]


for dataset in ["0", "10", "25"]: 


    data = pd.read_csv(project_root / f"Data/cln/{dataset}_cln.csv", sep = ';')
    predictions = pd.read_csv(project_root / f"predictions/lgbm_gearbox_bearing_temp_{dataset}.csv")

    results = predictions.merge(data[["row_id", "split"]], left_on = "reading_id", right_on = "row_id", validate = "one_to_one")
    test = results[results["split"] == "prediction"].copy()
    if test.empty:
        raise ValueError("No held-out prediction rows found.")

    test["absolute_error"] = (test["actual"] - test["expected"]).abs()

    mae = test["absolute_error"].mean()

    scores.append({"dataset": dataset, 
                   "model": "lgbm", 
                   "part": "gearbox_bearing_temp", 
                   "readings": len(test), 
                   "mae_c": mae
                   })

 
    print("Readings eval", len(test))
    print("Mean Absolute error", round(mae, 3), "°C")


scores_table = pd.DataFrame(scores)
scores_table.to_csv(
    project_root / "eval_results.csv",
    index=False
)
