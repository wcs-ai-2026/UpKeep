"""Integration tests: replay -> live scoring -> health.jsonl.

No trained models exist, so these tests use either the real (empty) model
store or a test double that returns synthetic residuals. Neither trains a
model. Thresholds are test-only values, not tuned ones.
"""
import json
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from upkeep.health.levels import GREEN, RED, YELLOW, HealthConfig
from upkeep.parts import PARTS
from upkeep.paths import data_dir
from upkeep.stream import live
from upkeep.stream.faults import load_events
from upkeep.stream.live import HealthWriter, LiveScorer, dumps_line, to_json_safe
from upkeep.stream.replay import load_dataset, main, replay

SEED = data_dir() / "seed"
pytestmark = pytest.mark.skipif(not (SEED / "0.parquet").exists(), reason="seed data not present")

CFG = HealthConfig(yellow_z=3.0, red_z=5.0, persistence_hours=1.0, max_gap_minutes=30.0)
NOISE = {p: 1.0 for p in PARTS}
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
FIELDS = {
    "row_id", "dataset_id", "turbine_id", "timestamp", "health", "health_score", "residuals",
    "anomaly_score", "gusts_last_7d", "reasons", "sentence", "fault_event", "days_to_fault", "latency_ms",
}
PRED_ROWS = {"A-0": 2365, "A-10": 1268, "A-25": 1942}
TURBINES = {"A-0": "A-T0", "A-10": "A-T10", "A-25": "A-T11"}


def _reject_constant(name):
    raise ValueError(f"non-JSON constant {name} in output")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line, parse_constant=_reject_constant) for line in f]


class FakeStore:
    """Test double for ModelStore: synthetic residuals, no models."""

    def __init__(self, fn):
        self.fn = fn

    def residuals(self, dataset_id, reading):
        return self.fn(dataset_id, reading)

    def status(self, dataset_id):
        return {p: "loaded" for p in PARTS}


class SpyExplain:
    def __init__(self):
        self.calls = []

    def __call__(self, recent, residuals, gusts):
        self.calls.append((recent, residuals, gusts))
        return {"reasons": [{"part": "gearbox_bearing", "residual_c": 1.0, "hours": 1.0, "text": "t"}],
                "gust_context": "", "sentence": "test sentence"}


@pytest.fixture(scope="module")
def farm_no_models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("farm")
    out = tmp / "health.jsonl"
    summary = replay([0, 10, 25], config=CFG, mode="farm", output=out,
                     models_root=tmp / "no_models")
    return summary, read_jsonl(out)


# ---- farm replay with no trained models ---------------------------------------

def test_farm_replay_writes_one_valid_line_per_test_reading(farm_no_models):
    summary, lines = farm_no_models
    assert len(lines) == sum(PRED_ROWS.values())
    assert summary["validated"] is False
    for d, n in PRED_ROWS.items():
        assert summary["datasets"][d]["lines_written"] == n
        assert summary["datasets"][d]["health_counts"] == {"unknown": n}
    for rec in lines:
        assert set(rec) == FIELDS
        assert TS_RE.match(rec["timestamp"])
        assert rec["turbine_id"] == TURBINES[rec["dataset_id"]]


def test_missing_models_give_null_residuals_and_unknown_health(farm_no_models):
    _, lines = farm_no_models
    for rec in lines:
        assert rec["residuals"] == {p: None for p in PARTS}
        assert rec["health"] is None and rec["health_score"] is None  # never green (L-13)
        assert rec["reasons"] == [] and rec["sentence"] == ""
        assert rec["anomaly_score"] is None


def test_farm_lines_are_in_time_order_and_ids_unique(farm_no_models):
    _, lines = farm_no_models
    stamps = [rec["timestamp"] for rec in lines]
    assert stamps == sorted(stamps)
    keys = [(rec["dataset_id"], rec["row_id"]) for rec in lines]
    assert len(keys) == len(set(keys))  # L-12 identity


# ---- fault-event handling (T-11) -----------------------------------------------

def test_fault_fields_in_replay_output(farm_no_models):
    _, lines = farm_no_models
    by_ds = {d: [r for r in lines if r["dataset_id"] == d] for d in PRED_ROWS}
    for d, start in (("A-0", 52436), ("A-10", 52611)):
        recs = by_ds[d]
        assert all(r["fault_event"] == d for r in recs)
        at_start = [r for r in recs if r["row_id"] == start]
        assert len(at_start) == 1 and at_start[0]["days_to_fault"] == 0
        assert all(r["days_to_fault"] > 0 for r in recs if r["row_id"] < start)
        assert all(r["days_to_fault"] < 0 for r in recs if r["row_id"] > start)
    assert all(r["fault_event"] is None and r["days_to_fault"] is None for r in by_ds["A-25"])


# ---- L-12 modes -------------------------------------------------------------

@pytest.fixture
def same_turbine_seeds(tmp_path):
    seeds = tmp_path / "seed"
    seeds.mkdir()
    shutil.copy(SEED / "0.parquet", seeds / "0.parquet")
    # Dataset 24 is another dataset of turbine 0 (event 24 is "normal").
    pd.read_parquet(SEED / "0.parquet").tail(300).to_parquet(seeds / "24.parquet", index=False)
    return seeds


def test_farm_mode_refuses_two_datasets_of_one_turbine(same_turbine_seeds, tmp_path):
    out = tmp_path / "farm.jsonl"
    with pytest.raises(ValueError, match="L-12"):
        replay([0, 24], config=CFG, mode="farm", seed_dir=same_turbine_seeds, output=out,
               models_root=tmp_path / "none")
    assert not out.exists()


def test_evaluation_mode_keeps_datasets_separate(same_turbine_seeds, tmp_path):
    out_dir = tmp_path / "eval"
    summary = replay([0, 24], config=CFG, mode="evaluation", split="all", seed_dir=same_turbine_seeds,
                     output_dir=out_dir, limit=50, models_root=tmp_path / "none")
    for d in ("A-0", "A-24"):
        recs = read_jsonl(out_dir / f"health_{d}.jsonl")
        assert len(recs) == 50
        assert {r["dataset_id"] for r in recs} == {d}
        assert summary["datasets"][d]["lines_written"] == 50
    # Same physical readings, separate identities and fault info.
    assert read_jsonl(out_dir / "health_A-24.jsonl")[0]["fault_event"] is None


def test_replay_rejects_duplicate_datasets(tmp_path):
    with pytest.raises(ValueError):
        replay([0, "A-0"], config=CFG, output=tmp_path / "x.jsonl")


# ---- persistence end-to-end with synthetic residuals -------------------------------

def ramp_store(onset, red_from):
    """Gearbox residual 0 before ``onset``, 4 (yellow level) until ``red_from``, then 6."""
    def fn(dataset_id, reading):
        ts = pd.Timestamp(reading["timestamp"])
        if dataset_id != "A-10":
            return {p: 0.0 for p in PARTS}
        g = 0.0 if ts < onset else (4.0 if ts < red_from else 6.0)
        return {"gearbox_bearing": g, "generator_bearing_1": 0.0, "generator_bearing_2": None}
    return FakeStore(fn)


def test_persistence_and_explain_through_replay(tmp_path):
    fault_start = pd.Timestamp("2023-10-12 08:40:00")
    onset, red_from = fault_start - pd.Timedelta(days=1), fault_start
    spy = SpyExplain()
    out = tmp_path / "h.jsonl"
    replay([10, 25], config=CFG, mode="farm", output=out,
           model_store=ramp_store(onset, red_from), noise_levels={10: NOISE, 25: NOISE}, explain_fn=spy)
    recs = read_jsonl(out)
    a10 = [r for r in recs if r["dataset_id"] == "A-10"]
    a25 = [r for r in recs if r["dataset_id"] == "A-25"]

    def ts(r):
        return pd.Timestamp(r["timestamp"])

    first_yellow = next(r for r in a10 if r["health"] in (YELLOW, RED))
    first_red = next(r for r in a10 if r["health"] == RED)
    assert ts(first_yellow) >= onset + pd.Timedelta(hours=1)
    assert ts(first_red) >= red_from + pd.Timedelta(hours=1)
    assert all(r["health"] == GREEN for r in a10 if ts(r) < onset + pd.Timedelta(hours=1))
    assert all(r["health"] == GREEN for r in a25)  # independent state (L-12)

    alerts = [r for r in recs if r["health"] in (YELLOW, RED)]
    assert len(spy.calls) == len(alerts) > 0  # explain only for yellow/red (L-05)
    assert all(r["sentence"] == "test sentence" and r["reasons"] for r in alerts)
    assert all(r["reasons"] == [] and r["sentence"] == "" for r in recs if r["health"] == GREEN)
    assert all(r["residuals"]["generator_bearing_2"] is None for r in a10)

    recent, residuals, gusts = spy.calls[-1]
    assert list(recent.columns) == ["timestamp"] + [f"residual_{p}" for p in PARTS]
    assert recent["timestamp"].max() - recent["timestamp"].min() <= pd.Timedelta(days=7)
    assert set(gusts) == {"gusts_last_24h", "gusts_last_7d"}


def test_isolated_spike_does_not_alert(tmp_path):
    spike_row = 52400

    def fn(dataset_id, reading):
        return {"gearbox_bearing": 50.0 if reading["row_id"] == spike_row else 0.0,
                "generator_bearing_1": 0.0, "generator_bearing_2": 0.0}

    out = tmp_path / "h.jsonl"
    replay([10], config=CFG, output=out, model_store=FakeStore(fn), noise_levels={10: NOISE},
           explain_fn=SpyExplain())
    recs = read_jsonl(out)
    assert any(r["row_id"] == spike_row and r["residuals"]["gearbox_bearing"] == 50.0 for r in recs)
    assert {r["health"] for r in recs} == {GREEN}


# ---- live scorer directly ------------------------------------------------------

@pytest.fixture
def scorer25(tmp_path):
    df = load_dataset(25)
    writer = HealthWriter(tmp_path / "live.jsonl")
    s = LiveScorer(CFG, model_store=FakeStore(lambda d, r: {p: float("nan") for p in PARTS}),
                   events=load_events(), explain_fn=SpyExplain(), writer=writer)
    s.register_dataset(25, df, NOISE)
    yield s, df, writer
    writer.close()


def test_status_not_zero_is_skipped_and_writes_nothing(scorer25):
    s, df, writer = scorer25
    reading = df.iloc[0].to_dict()
    reading["status_id"] = 3
    assert s.score_reading(reading) is None
    assert writer.lines == 0
    reading["status_id"] = 0
    assert s.score_reading(reading) is not None
    assert writer.lines == 1


def test_nan_inputs_become_null(scorer25):
    s, df, writer = scorer25
    reading = df.iloc[0].to_dict()
    reading["gusts_last_7d"] = np.nan
    rec = s.score_reading(reading)
    assert rec["residuals"] == {p: None for p in PARTS}
    assert rec["gusts_last_7d"] is None
    writer.close()
    line = open(writer.path, encoding="utf-8").read()
    assert "NaN" not in line and "Infinity" not in line
    assert json.loads(line)["health"] is None


def test_unregistered_dataset_and_turbine_mismatch(scorer25):
    s, df, _ = scorer25
    reading = df.iloc[0].to_dict()
    with pytest.raises(KeyError):
        s.score_reading({**reading, "dataset_id": "A-0"})
    with pytest.raises(ValueError):
        s.score_reading({**reading, "turbine_id": "A-T0"})


def test_json_safety_helpers():
    rec = {"a": float("nan"), "b": np.float32(1.5), "c": np.int64(3), "d": float("inf"),
           "e": pd.Timestamp("2023-01-02 03:04:05"), "f": [np.nan, 1], "g": np.bool_(True)}
    assert to_json_safe(rec) == {"a": None, "b": 1.5, "c": 3, "d": None,
                                 "e": "2023-01-02T03:04:05", "f": [None, 1], "g": True}
    assert "NaN" not in dumps_line(rec)


def test_module_level_score_reading_uses_replay_scorer(tmp_path):
    replay([25], config=CFG, output=tmp_path / "h.jsonl", limit=1, models_root=tmp_path / "none")
    reading = load_dataset(25).iloc[-1].to_dict()
    rec = live.score_reading(reading)
    assert rec["dataset_id"] == "A-25" and rec["health"] is None
    live.set_default_scorer(None)
    with pytest.raises(RuntimeError):
        live.score_reading(reading)


def test_cli_requires_explicit_untuned_flag():
    with pytest.raises(SystemExit):
        main(["--datasets", "25"])


# ---- stub stays unchanged (L-10) --------------------------------------------------

def test_stub_score_reading_unchanged():
    from upkeep.stubs import score_reading as stub_score
    rec = stub_score({"row_id": 1001, "turbine_id": "A-T3", "timestamp": "2022-03-14T06:20:00"})
    assert rec["health"] == "yellow" and rec["health_score"] == 0.64
    assert rec["residuals"] == {"gearbox_bearing": 6.1, "generator_bearing_1": 1.2, "generator_bearing_2": 0.8}
    assert rec["anomaly_score"] == 0.58 and rec["latency_ms"] == 0
    assert "dataset_id" not in rec
