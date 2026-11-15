"""Convert the three SCRUM-22 cleaned datasets to Parquet."""
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "Data" / "seed"
    output.mkdir(parents=True, exist_ok=True)
    for dataset_id in (0, 10, 25):
        source = root / "Data" / "cln" / f"{dataset_id}_cln.csv"
        frame = pd.read_csv(source, sep=";", parse_dates=["timestamp"])
        target = output / f"{dataset_id}.parquet"
        frame.to_parquet(target, engine="pyarrow", index=False)
        print(f"Saved {target}: {len(frame):,} rows, {len(frame.columns)} columns")


if __name__ == "__main__":
    main()
