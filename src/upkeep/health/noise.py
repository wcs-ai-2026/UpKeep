"""Calibrate each part's normal noise level (T-16, proposed; needs team agreement).

L-03 divides every residual by its part's normal noise level, so the noise
level sets how many "noise units" a residual is worth. If it is too small,
ordinary readings look abnormal and cause false alarms.

Method (agreed by the Lead for SCRUM-45; T-16 still needs team agreement):

1. Read the dataset's cleaned readings (``Data/cln/<n>_cln.csv``, the data the
   deployed models were trained on) and drop rows whose target is missing.
2. Keep only the training split. The locked test rows (``split ==
   "prediction"``) are discarded before anything else and never used.
3. Split the training year at ``last training timestamp - 2 months``, the same
   validation window as the plan and Varun's tuning (DECISIONS.md, SCRUM-36).
   Rows before the cutoff are the *fit* rows; rows from the cutoff on are the
   *calibration* rows.
4. Read the deployed model's feature list through :class:`ModelStore` (which
   also refuses forbidden inputs), then fit a **fresh** instance of the same
   model class (Varun's ``upkeep/models/``) on the fit rows only.
5. Predict the calibration rows and take the residuals. Only rows where every
   feature is present are used, because those are the only rows that get a
   residual in live scoring (L-07).
6. Noise level = ``1.4826 * MAD`` of those residuals (a robust standard
   deviation that a few odd readings cannot inflate).

The deployed models themselves are never refitted or overwritten: residuals of a
model on its own training rows are too optimistic, which is why a fresh,
out-of-sample fit is used here.

Output: ``models_saved/<model>_noise_<n>.json`` (:func:`upkeep.health.score.noise_path`),
``{"<part>": <float or null>, ..., "_meta": {...}}``. The loader
(:func:`upkeep.health.score.load_noise_levels`) reads only the part keys; a
``null`` or missing part is treated as unusable (L-07).

CLI::

    PYTHONPATH=src python -m upkeep.health.noise --datasets 0 10 25
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from upkeep.health.score import noise_path
from upkeep.ids import dataset_number, format_dataset_id
from upkeep.parts import PART_SENSORS, PARTS
from upkeep.paths import data_dir
from upkeep.stream.model_store import DEFAULT_MODEL, ModelStore

log = logging.getLogger(__name__)

MAD_TO_STD = 1.4826
CALIBRATION_MONTHS = 2
MIN_CALIBRATION_ROWS = 100
TRAIN_SPLIT = "train"
TEST_SPLIT = "prediction"
METHOD = "out-of-sample 1.4826*MAD on the last 2 months of the training split"


def robust_spread(residuals) -> Optional[float]:
    """``1.4826 * median(|r - median(r)|)`` over finite values; ``None`` if unusable."""
    values = np.asarray(pd.Series(residuals, dtype="float64"), dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    spread = MAD_TO_STD * float(np.median(np.abs(values - np.median(values))))
    return spread if math.isfinite(spread) and spread > 0 else None


def cln_path(dataset_id, root: Optional[Path] = None) -> Path:
    return Path(root or data_dir()) / "cln" / f"{dataset_number(dataset_id)}_cln.csv"


def load_cleaned(dataset_id, root: Optional[Path] = None) -> pd.DataFrame:
    """Read one dataset's cleaned readings (the models' training data)."""
    path = cln_path(dataset_id, root)
    if not path.is_file():
        raise FileNotFoundError(f"No cleaned data for dataset {format_dataset_id(dataset_id)}: {path}")
    df = pd.read_csv(path, sep=";")
    for column in ("timestamp", "split"):
        if column not in df.columns:
            raise ValueError(f"{path} is missing the {column!r} column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def calibration_split(df: pd.DataFrame, target: str,
                      months: int = CALIBRATION_MONTHS) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Return ``(fit_rows, calibration_rows, cutoff)`` from the training split only.

    Rows with a missing target are dropped first. Test rows are discarded and
    must all come after the training split; otherwise the split is ambiguous
    and an error is raised.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found")
    if df["timestamp"].isna().any():
        raise ValueError("readings with no timestamp cannot be split in time")
    train = df[df["split"] == TRAIN_SPLIT]
    test = df[df["split"] == TEST_SPLIT]
    if train.empty:
        raise ValueError("no training rows")
    if not test.empty and train["timestamp"].max() >= test["timestamp"].min():
        raise ValueError("training rows overlap the locked test period; refusing to calibrate")
    train = train.dropna(subset=[target]).sort_values("timestamp", kind="stable")
    cutoff = train["timestamp"].max() - pd.DateOffset(months=months)
    fit = train[train["timestamp"] < cutoff]
    calibration = train[train["timestamp"] >= cutoff]
    if fit.empty or calibration.empty:
        raise ValueError("training split too short for a fit/calibration split")
    return fit, calibration, cutoff


def calibrate_part(df: pd.DataFrame, part: str, deployed_model,
                   months: int = CALIBRATION_MONTHS) -> dict:
    """Fit a fresh copy of ``deployed_model``'s class out of sample and measure noise."""
    target = PART_SENSORS[part]
    features = list(getattr(deployed_model, "features", None) or [])
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"features {missing} not in the data")
    fit, calibration, cutoff = calibration_split(df, target, months)

    model = type(deployed_model)()  # same class (Varun's), fresh and untrained
    model.fit(fit[features], fit[target])

    usable = calibration.dropna(subset=features) if features else calibration
    residuals = usable[target].to_numpy(dtype="float64") - np.asarray(model.predict(usable[features]), dtype="float64")
    noise = robust_spread(residuals) if len(usable) >= MIN_CALIBRATION_ROWS else None
    if noise is None:
        log.warning("Part %s: no usable noise level (%d calibration rows)", part, len(usable))
    return {
        "noise": noise,
        "n_fit": int(len(fit)),
        "n_calibration": int(len(usable)),
        "n_calibration_dropped_missing_features": int(len(calibration) - len(usable)),
        "fit_start": str(fit["timestamp"].min()),
        "cutoff": str(cutoff),
        "calibration_end": str(calibration["timestamp"].max()),
        "calibration_mae": float(np.mean(np.abs(residuals))) if len(residuals) else None,
        "features": features,
    }


def compute_noise_levels(dataset_id, model_name: str = DEFAULT_MODEL, models_root: Optional[Path] = None,
                         data_root: Optional[Path] = None, parts: Tuple[str, ...] = PARTS,
                         months: int = CALIBRATION_MONTHS) -> Tuple[Dict[str, Optional[float]], dict]:
    """Return ``({part: noise or None}, metadata)`` for one dataset."""
    store = ModelStore(model_name, models_root=models_root, parts=parts)
    df = load_cleaned(dataset_id, data_root)
    levels: Dict[str, Optional[float]] = {}
    details: Dict[str, dict] = {}
    for part in parts:
        deployed = store.get(dataset_id, part)
        if deployed is None:
            levels[part] = None
            details[part] = {"noise": None, "skipped": store.status(dataset_id)[part]}
            continue
        info = calibrate_part(df, part, deployed, months)
        info["model_file"] = store.model_path(dataset_id, part).name
        levels[part] = info["noise"]
        details[part] = info
    meta = {
        "dataset_id": format_dataset_id(dataset_id),
        "model": model_name,
        "method": METHOD,
        "status": "T-16 proposed; not validated",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "parts": details,
    }
    return levels, meta


def write_noise_file(dataset_id, levels: Dict[str, Optional[float]], meta: Optional[dict] = None,
                     model_name: str = DEFAULT_MODEL, models_root: Optional[Path] = None) -> Path:
    """Write ``{part: noise, "_meta": ...}`` where :func:`load_noise_levels` reads it."""
    path = noise_path(dataset_id, model_name, models_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {part: (None if value is None else round(float(value), 6)) for part, value in levels.items()}
    if meta is not None:
        payload["_meta"] = meta
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate per-part noise levels (T-16) out of sample.",
                                     allow_abbrev=False)
    parser.add_argument("--datasets", nargs="+", required=True, help="e.g. 0 10 25 or A-0 A-10")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model type whose residuals are calibrated")
    parser.add_argument("--models-root", type=Path, help="default: models_saved/")
    parser.add_argument("--data-dir", type=Path, help="folder holding cln/<n>_cln.csv (default: Data/)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ids = [format_dataset_id(d) for d in args.datasets]
    if len(set(ids)) != len(ids):
        raise SystemExit("Each dataset may be given only once")
    summary = {}
    for dataset_id in ids:
        levels, meta = compute_noise_levels(dataset_id, args.model, args.models_root, args.data_dir)
        path = write_noise_file(dataset_id, levels, meta, args.model, args.models_root)
        summary[dataset_id] = {"file": str(path), "noise": levels}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
