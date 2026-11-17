"""Replay seed datasets through the live scorer.

Loading (T-02): each ``<n>.parquet`` gets ``dataset_id = "A-<n>"`` and
``turbine_id = "A-T<asset>"``; the original integer asset is kept as
``asset_id``. The seed's precomputed features are used as they are. Rows with
a missing (NaT) timestamp are rejected with an error.

Modes (L-12):

- **farm**: the dashboard replay. Readings from all datasets are interleaved
  by timestamp into one ``health.jsonl`` (``--output``). Refuses two datasets
  for the same physical turbine.
- **evaluation**: each dataset is replayed on its own, with its own scorer,
  into ``<output_dir>/health_<dataset_id>.jsonl`` (``--output-dir``).

Output files are **overwritten by default**, so rerunning a replay cannot
duplicate readings. ``--append`` appends on purpose; the caller is then
responsible for not replaying the same readings twice.

By default only the ``prediction`` (test) split is replayed. The fault lookup
always sees the whole dataset (T-11).

Usage::

    PYTHONPATH=src python -m upkeep.stream.replay --datasets 0 10 25 \
        --mode farm --untuned-placeholder-config

``--untuned-placeholder-config`` is required because no tuned thresholds exist
yet (T-08 pending). Output from it is NOT validated. Without
``--stub-explain`` alerts carry no reasons (the stub's text is not meaningful;
see :func:`upkeep.stream.live.stub_explain`).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from upkeep.health.levels import HealthConfig, UNTUNED_PLACEHOLDER_CONFIG
from upkeep.ids import dataset_number, format_dataset_id, format_turbine_id
from upkeep.parts import PARTS
from upkeep.paths import data_dir
from upkeep.stream.faults import load_events
from upkeep.stream.live import (
    HealthWriter, LiveScorer, default_output_path, set_default_scorer, stub_explain,
)

log = logging.getLogger(__name__)

SPLITS = ("prediction", "train", "all")
MODES = ("farm", "evaluation")
REQUIRED_COLUMNS = ("row_id", "timestamp", "turbine_id", "status_id")
ALERTS = ("yellow", "red")


def default_seed_dir() -> Path:
    return data_dir() / "seed"


def default_eval_dir() -> Path:
    return data_dir() / "live" / "eval"


def load_dataset(dataset_id, seed_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load one seed dataset and add the T-02 ids. Validates order and identity."""
    number = dataset_number(dataset_id)
    path = Path(seed_dir or default_seed_dir()) / f"{number}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No seed file for dataset {format_dataset_id(number)}: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if "dataset_id" in df.columns:
        raise ValueError(f"{path} already has a dataset_id column")
    assets = pd.unique(df["turbine_id"])
    if len(assets) != 1:
        raise ValueError(f"{path} must contain exactly one turbine, found {list(assets)}")
    if not df["row_id"].is_unique:
        raise ValueError(f"{path} has duplicate row_id values")

    df = df.copy()
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path} has unparseable timestamps: {exc}") from None
    bad = df["timestamp"].isna()
    if bad.any():
        rows = df.loc[bad, "row_id"].head(5).tolist()
        raise ValueError(f"{path} has {int(bad.sum())} reading(s) with no timestamp (NaT), e.g. row_id {rows}")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if not df["timestamp"].is_unique:
        raise ValueError(f"{path} has duplicate timestamps")
    df["asset_id"] = int(assets[0])
    df["turbine_id"] = format_turbine_id(int(assets[0]))
    df["dataset_id"] = format_dataset_id(number)
    return df


def check_farm(frames: Dict[str, pd.DataFrame]) -> None:
    """L-12 (3): farm mode refuses more than one dataset per physical turbine."""
    seen: Dict[str, str] = {}
    for dataset_id, df in frames.items():
        turbine = df["turbine_id"].iloc[0]
        if turbine in seen:
            raise ValueError(
                f"Farm mode cannot load {seen[turbine]} and {dataset_id}: both are turbine {turbine} (L-12)"
            )
        seen[turbine] = dataset_id


def select_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "all":
        return df
    if "split" not in df.columns:
        raise ValueError("dataset has no split column")
    return df[df["split"] == split]


def interleave(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge datasets in timestamp order; ties keep dataset order (stable)."""
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values("timestamp", kind="stable").reset_index(drop=True)


class _Stats:
    def __init__(self):
        self.read = 0
        self.written = 0
        self.skipped_status = 0
        self.alerts_without_reasons = 0
        self.health: Dict[str, int] = {}

    def add(self, record: Optional[dict]) -> None:
        self.read += 1
        if record is None:
            self.skipped_status += 1
            return
        self.written += 1
        key = record["health"] or "unknown"
        self.health[key] = self.health.get(key, 0) + 1
        if key in ALERTS and not record["reasons"]:
            self.alerts_without_reasons += 1

    def summary(self) -> dict:
        return {
            "readings": self.read,
            "lines_written": self.written,
            "skipped_status_not_0": self.skipped_status,
            "health_counts": dict(sorted(self.health.items())),
            "alerts_without_reasons": self.alerts_without_reasons,
        }


def _describe(fn) -> Optional[str]:
    """Name of the explainer for the summary, or ``None`` when there is none."""
    if fn is None:
        return None
    name = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    return f"{getattr(fn, '__module__', '?')}.{name}"


def _run(scorer: LiveScorer, rows: pd.DataFrame, speed: float, limit: Optional[int],
         stats: Dict[str, _Stats]) -> None:
    for i, reading in enumerate(rows.to_dict("records")):
        if limit is not None and i >= limit:
            break
        record = scorer.score_reading(reading)
        stats.setdefault(reading["dataset_id"], _Stats()).add(record)
        if speed > 0:
            time.sleep(speed)


def replay(
    dataset_ids: Iterable,
    *,
    config: HealthConfig,
    mode: str = "farm",
    split: str = "prediction",
    seed_dir: Optional[Path] = None,
    output: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    append: bool = False,
    speed: float = 0.0,
    limit: Optional[int] = None,
    model_name: Union[str, Mapping[str, str]] = "lgbm",
    models_root: Optional[Path] = None,
    model_store=None,
    noise_levels: Optional[Dict] = None,
    explain_fn=None,
    anomaly_scorer=None,
) -> dict:
    """Replay datasets and return a summary.

    Output is overwritten unless ``append=True``. ``limit`` caps readings per
    scorer run (farm: total; evaluation: per dataset). ``noise_levels``
    optionally maps ``dataset_id`` to a noise dict, overriding the noise files.
    ``model_name`` is one model type for every part or a ``{part: type}`` map.
    ``wall_time_s`` covers loading, scoring and writing for the whole call.
    """
    started = time.perf_counter()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if mode == "farm" and output_dir is not None:
        raise ValueError("output_dir is only used in evaluation mode; use output for farm mode")
    if mode == "evaluation" and output is not None:
        raise ValueError("output is only used in farm mode; use output_dir for evaluation mode")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if speed < 0:
        raise ValueError("speed cannot be negative")
    ids = [format_dataset_id(d) for d in dataset_ids]
    if not ids:
        raise ValueError("No datasets given")
    if len(set(ids)) != len(ids):
        raise ValueError("Each dataset may be replayed only once")
    noise_levels = {format_dataset_id(k): v for k, v in (noise_levels or {}).items()}
    unknown_noise = set(noise_levels) - set(ids)
    if unknown_noise:
        raise ValueError(f"noise_levels given for datasets not being replayed: {sorted(unknown_noise)}")

    frames = {d: load_dataset(d, seed_dir) for d in ids}
    if mode == "farm":
        check_farm(frames)
    events = load_events()
    stats: Dict[str, _Stats] = {}
    outputs: Dict[str, str] = {}
    issues: Dict[str, int] = {}

    def make_scorer(writer):
        return LiveScorer(config, model_store=model_store, model_name=model_name, models_root=models_root,
                          events=events, explain_fn=explain_fn, anomaly_scorer=anomaly_scorer, writer=writer)

    def collect(scorer):
        for key, n in scorer.explanation_issues.items():
            issues[key] = issues.get(key, 0) + n

    if mode == "farm":
        path = Path(output) if output is not None else default_output_path()
        with HealthWriter(path, append=append) as writer:
            scorer = make_scorer(writer)
            for d, df in frames.items():
                scorer.register_dataset(d, df, noise_levels.get(d))
            set_default_scorer(scorer)
            try:
                _run(scorer, interleave([select_split(df, split) for df in frames.values()]), speed, limit, stats)
            finally:
                scorer.writer = None  # file closes with this block; later calls just return records
        collect(scorer)
        outputs = {d: str(path) for d in ids}
    else:
        set_default_scorer(None)  # evaluation scorers are per dataset; no farm-wide default
        directory = Path(output_dir) if output_dir is not None else default_eval_dir()
        for d, df in frames.items():
            path = directory / f"health_{d}.jsonl"
            with HealthWriter(path, append=append) as writer:
                scorer = make_scorer(writer)
                scorer.register_dataset(d, df, noise_levels.get(d))
                _run(scorer, select_split(df, split), speed, limit, stats)
            collect(scorer)
            outputs[d] = str(path)

    return {
        "mode": mode,
        "split": split,
        "validated": False,
        "appended": append,
        "explainer": _describe(explain_fn),
        "explanation_issues": dict(sorted(issues.items())),
        "total_readings": sum(s.read for s in stats.values()),
        "wall_time_s": round(time.perf_counter() - started, 3),
        "datasets": {d: {**stats.get(d, _Stats()).summary(), "output": outputs[d]} for d in ids},
    }


def _part_model(value: str):
    part, sep, name = value.partition("=")
    if not sep or part not in PARTS or not name:
        raise argparse.ArgumentTypeError(f"expected <part>=<model> with part in {list(PARTS)}, got {value!r}")
    return part, name


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if not number >= 0:
        raise argparse.ArgumentTypeError("must be zero or more")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay UpKeep seed datasets through live scoring.",
                                     allow_abbrev=False)
    parser.add_argument("--datasets", nargs="+", required=True, help="e.g. 0 10 25 or A-0 A-10")
    parser.add_argument("--mode", choices=MODES, default="farm")
    parser.add_argument("--split", choices=SPLITS, default="prediction")
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--output", type=Path, help="farm mode only (default data/live/health.jsonl)")
    parser.add_argument("--output-dir", type=Path, help="evaluation mode only (default data/live/eval/)")
    parser.add_argument("--append", action="store_true",
                        help="append instead of overwriting (may duplicate readings if rerun)")
    parser.add_argument("--speed", type=_non_negative_float, default=0.0, help="seconds between readings")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--model", default="lgbm", help="model type for every part")
    parser.add_argument("--part-model", type=_part_model, action="append", default=[],
                        metavar="PART=MODEL", help="override the model type for one part (repeatable)")
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--stub-explain", action="store_true",
                        help="attach the interim stub explanations (not meaningful; wiring/demo only)")
    parser.add_argument("--untuned-placeholder-config", action="store_true",
                        help="required: run with UNTUNED placeholder thresholds (T-08 pending)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # unknown arguments are an error
    if not args.untuned_placeholder_config:
        parser.error("No tuned thresholds exist yet (T-08). Pass --untuned-placeholder-config to run anyway.")
    if args.mode == "farm" and args.output_dir is not None:
        parser.error("--output-dir is only used with --mode evaluation")
    if args.mode == "evaluation" and args.output is not None:
        parser.error("--output is only used with --mode farm")
    parts = dict(args.part_model)
    if len(parts) != len(args.part_model):
        parser.error("--part-model given more than once for the same part")
    model_name = {**{p: args.model for p in PARTS}, **parts} if parts else args.model

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    log.warning("Using UNTUNED placeholder thresholds; results are not validated.")
    if args.stub_explain:
        log.warning("Using the stub explainer: reason durations and selection are placeholders.")
    summary = replay(
        args.datasets, config=UNTUNED_PLACEHOLDER_CONFIG, mode=args.mode, split=args.split,
        seed_dir=args.seed_dir, output=args.output, output_dir=args.output_dir, append=args.append,
        speed=args.speed, limit=args.limit, model_name=model_name, models_root=args.models_root,
        explain_fn=stub_explain() if args.stub_explain else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
