import argparse
from pathlib import Path

import pandas as pd

from upkeep.fault_eval import evaluate_faults
from upkeep.paths import PROJECT_ROOT, data_dir


def evaluate_mae(data_root: Path) -> pd.DataFrame:
    scores = []
    for dataset in ("0", "10", "25"):
        data = pd.read_csv(
            data_root / "cln" / f"{dataset}_cln.csv", sep=";", usecols=["row_id", "split"]
        ).rename(columns={"split": "reading_split"})
        predictions = pd.read_csv(PROJECT_ROOT / "predictions" / f"lgbm_gearbox_bearing_temp_{dataset}.csv")
        results = predictions.merge(data, left_on="reading_id", right_on="row_id", validate="one_to_one")
        test = results.loc[results["reading_split"].eq("prediction")]
        if test.empty:
            raise ValueError(f"Dataset {dataset} has no held-out prediction rows")
        mae = (test["actual"] - test["expected"]).abs().mean()
        scores.append({
            "dataset": dataset,
            "model": "lgbm",
            "part": "gearbox_bearing_temp",
            "readings": len(test),
            "mae_c": mae,
        })
        print(f"Dataset {dataset}: {len(test)} held-out readings, gearbox bearing MAE {mae:.3f} C")
    return pd.DataFrame(scores)


def main():
    data_root = data_dir()
    parser = argparse.ArgumentParser(description="Evaluate prediction error and Farm A's advance warnings.")
    parser.add_argument("--health-dir", type=Path, default=data_root / "live" / "eval",
                        help="directory containing per-dataset replay health JSONL files")
    parser.add_argument("--window-days", type=int, default=7,
                        help="number of days before each fault to check (default: 7)")
    parser.add_argument("--fault-output", type=Path, default=PROJECT_ROOT / "fault_eval_results.csv",
                        help="path for the per-fault report (default: fault_eval_results.csv)")
    args = parser.parse_args()
    if args.window_days <= 0:
        parser.error("--window-days must be positive")

    faults = evaluate_faults(data_root, args.health_dir, pd.Timedelta(days=args.window_days))
    evaluate_mae(data_root).to_csv(PROJECT_ROOT / "eval_results.csv", index=False)
    faults.to_csv(args.fault_output, index=False)
    print(f"Saved {len(faults)} fault results to {args.fault_output}")
    print(faults["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
