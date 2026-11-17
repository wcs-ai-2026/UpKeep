"""SCRUM-44: score_reading() writing one health line per reading to health.jsonl.

Direct tests of the live scorer and the JSONL writer. Nothing here uses the
replay (SCRUM-43): seed readings are loaded with pandas and given their T-02
ids with the ``upkeep.ids`` helpers. No models are trained. Where coloured
output is needed, a test-only residual source stands in for the model store.
Thresholds are test-only values, not tuned ones.
"""
import json

import numpy as np
import pandas as pd
import pytest

from upkeep.health.levels import GREEN, RED, YELLOW, HealthConfig
from upkeep.ids import format_dataset_id, format_turbine_id
from upkeep.parts import PARTS
from upkeep.paths import data_dir
from upkeep.stream import live
from upkeep.stream.faults import load_events
from upkeep.stream.live import HealthWriter, LiveScorer, default_output_path

SEED = data_dir() / "seed"
needs_seed = pytest.mark.skipif(not (SEED / "25.parquet").exists(), reason="seed data not present")

CFG = HealthConfig(yellow_z=3.0, red_z=5.0, persistence_hours=1.0, max_gap_minutes=30.0)
NOISE = {p: 1.0 for p in PARTS}
CONTRACT_FIELDS = {
    "row_id", "turbine_id", "timestamp", "health", "health_score", "residuals", "anomaly_score",
    "gusts_last_7d", "reasons", "sentence", "fault_event", "days_to_fault", "latency_ms",
}
OUTPUT_FIELDS = CONTRACT_FIELDS | {"dataset_id"}  # dataset_id: L-12


def _reject_constant(name):
    raise ValueError(f"non-JSON constant {name} in output")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line, parse_constant=_reject_constant) for line in f]


def load_seed(number):
    """Seed readings with T-02 ids, in time order (no replay involved)."""
    df = pd.read_parquet(SEED / f"{number}.parquet")
    df["turbine_id"] = df["turbine_id"].map(format_turbine_id)
    df["dataset_id"] = format_dataset_id(number)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


class FakeStore:
    """Test-only stand-in for ModelStore: residuals from a function, no models."""

    def __init__(self, fn):
        self.fn = fn

    def residuals(self, dataset_id, reading):
        return self.fn(dataset_id, reading)

    def status(self, dataset_id):
        return {p: "loaded" for p in PARTS}


def constant(value):
    return FakeStore(lambda d, r: {p: value for p in PARTS})


class SpyExplain:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def __call__(self, recent, residuals, gusts):
        self.calls += 1
        return self.result


@pytest.fixture(scope="module")
def seed25():
    return load_seed(25)


@pytest.fixture(scope="module")
def events():
    return load_events()


def make_scorer(df, events, tmp_path, store=None, noise=None, **kwargs):
    writer = HealthWriter(tmp_path / "health.jsonl")
    scorer = LiveScorer(CFG, model_store=store, models_root=tmp_path / "no_models",
                        events=events, writer=writer, **kwargs)
    scorer.register_dataset(df["dataset_id"].iloc[0], df, noise)
    return scorer, writer


# ---- SCRUM-44: one health line per reading ------------------------------------

@needs_seed
def test_one_line_per_reading_in_input_order(seed25, events, tmp_path):
    readings = seed25[seed25["split"] == "prediction"].head(200)
    scorer, writer = make_scorer(seed25, events, tmp_path)
    returned = [scorer.score_reading(r) for r in readings.to_dict("records")]
    writer.close()

    lines = read_jsonl(writer.path)
    assert len(lines) == len(readings) == 200
    assert [line["row_id"] for line in lines] == readings["row_id"].tolist()
    assert lines == [json.loads(live.dumps_line(r)) for r in returned]  # file == returned records
    assert len({(line["dataset_id"], line["row_id"]) for line in lines}) == 200  # L-12 identity


@needs_seed
def test_every_line_has_the_contract_fields(seed25, events, tmp_path):
    scorer, writer = make_scorer(seed25, events, tmp_path)
    for r in seed25.head(20).to_dict("records"):
        scorer.score_reading(r)
    writer.close()
    for line in read_jsonl(writer.path):
        assert set(line) == OUTPUT_FIELDS
        assert "latency_ms" in line  # field required by the contract; value filled by SCRUM-47
        assert line["dataset_id"] == "A-25" and line["turbine_id"] == "A-T11"
        assert len(line["timestamp"]) == 19 and line["timestamp"][10] == "T"  # L-04
        assert set(line["residuals"]) == set(PARTS)


def test_default_output_is_data_live_health_jsonl():
    path = default_output_path()
    assert path == data_dir() / "live" / "health.jsonl"
    assert path.parent.parent.name.lower() == "data"


@needs_seed
def test_module_level_score_reading_writes_through_default_scorer(seed25, events, tmp_path):
    scorer, writer = make_scorer(seed25, events, tmp_path)
    live.set_default_scorer(scorer)
    try:
        record = live.score_reading(seed25.iloc[0].to_dict())
        writer.close()
        assert read_jsonl(writer.path) == [json.loads(live.dumps_line(record))]
    finally:
        live.set_default_scorer(None)
    with pytest.raises(RuntimeError):
        live.score_reading(seed25.iloc[1].to_dict())


# ---- missing data and skipped readings ----------------------------------------

@needs_seed
def test_missing_models_give_null_residuals_and_unknown_health(seed25, events, tmp_path):
    scorer, writer = make_scorer(seed25, events, tmp_path)  # real ModelStore, empty folder
    records = [scorer.score_reading(r) for r in seed25.head(30).to_dict("records")]
    for rec in records:
        assert rec["residuals"] == {p: None for p in PARTS}
        assert rec["health"] is None and rec["health_score"] is None  # L-13: never green
        assert rec["reasons"] == [] and rec["sentence"] == ""
        assert rec["anomaly_score"] is None


@needs_seed
def test_status_not_zero_is_skipped_and_writes_nothing(seed25, events, tmp_path):
    scorer, writer = make_scorer(seed25, events, tmp_path)
    reading = seed25.iloc[0].to_dict()
    assert scorer.score_reading({**reading, "status_id": 3}) is None  # L-08
    assert writer.lines == 0
    assert scorer.score_reading(reading) is not None
    assert writer.lines == 1


@needs_seed
def test_nan_values_are_written_as_null(seed25, events, tmp_path):
    scorer, writer = make_scorer(seed25, events, tmp_path, store=constant(float("nan")), noise=NOISE)
    rec = scorer.score_reading({**seed25.iloc[0].to_dict(), "gusts_last_7d": np.nan})
    writer.close()
    text = writer.path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert rec["residuals"] == {p: None for p in PARTS}
    assert rec["gusts_last_7d"] is None and rec["health"] is None


@needs_seed
@pytest.mark.parametrize("change,error", [
    ({"timestamp": None}, ValueError),
    ({"timestamp": pd.NaT}, ValueError),
    ({"timestamp": np.nan}, ValueError),
    ({"status_id": None}, ValueError),
    ({"dataset_id": "A-0"}, KeyError),      # not registered
    ({"turbine_id": "A-T0"}, ValueError),   # wrong turbine for A-25
])
def test_invalid_readings_are_rejected_and_not_written(seed25, events, tmp_path, change, error):
    scorer, writer = make_scorer(seed25, events, tmp_path)
    with pytest.raises(error):
        scorer.score_reading({**seed25.iloc[0].to_dict(), **change})
    assert writer.lines == 0


# ---- JSONL writer ---------------------------------------------------------------

def test_writer_overwrites_by_default(tmp_path):
    path = tmp_path / "health.jsonl"
    path.write_text('{"old": 1}\n', encoding="utf-8")
    with HealthWriter(path) as w:
        w.write({"new": 1})
    assert read_jsonl(path) == [{"new": 1}]


def test_writer_appends_only_when_asked(tmp_path):
    path = tmp_path / "health.jsonl"
    with HealthWriter(path) as w:
        w.write({"a": 1})
    with HealthWriter(path, append=True) as w:
        w.write({"b": 2})
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_writer_flushes_each_line_and_creates_folder(tmp_path):
    path = tmp_path / "live" / "health.jsonl"
    w = HealthWriter(path)
    try:
        w.write({"row_id": 1, "x": float("nan"), "t": pd.Timestamp("2023-01-02 03:04:05")})
        assert read_jsonl(path) == [{"row_id": 1, "x": None, "t": "2023-01-02T03:04:05"}]  # before close
    finally:
        w.close()


# ---- fault fields through the scorer (T-11) ----------------------------------------

@needs_seed
def test_fault_fields_before_at_and_after_fault_start(events, tmp_path):
    df0 = load_seed(0)
    start = df0.index[df0["row_id"] == 52436][0]
    scorer, writer = make_scorer(df0, events, tmp_path)
    before, at, after = (scorer.score_reading(df0.iloc[i].to_dict()) for i in (start - 1, start, start + 1))
    assert {r["fault_event"] for r in (before, at, after)} == {"A-0"}
    assert before["days_to_fault"] > 0
    assert at["days_to_fault"] == 0
    assert after["days_to_fault"] < 0


@needs_seed
def test_no_recorded_fault_gives_nulls(seed25, events, tmp_path):
    scorer, _ = make_scorer(seed25, events, tmp_path)
    rec = scorer.score_reading(seed25.iloc[0].to_dict())
    assert rec["fault_event"] is None and rec["days_to_fault"] is None


# ---- colours, persistence and explanations through the scorer -------------------------

@needs_seed
def test_isolated_spike_stays_green(seed25, events, tmp_path):
    spike_row = int(seed25["row_id"].iloc[10])
    store = FakeStore(lambda d, r: {p: (50.0 if r["row_id"] == spike_row else 0.0) for p in PARTS})
    scorer, _ = make_scorer(seed25, events, tmp_path, store=store, noise=NOISE)
    records = [scorer.score_reading(r) for r in seed25.head(20).to_dict("records")]
    assert {r["health"] for r in records} == {GREEN}


@needs_seed
def test_sustained_problem_turns_yellow_only_after_the_window(seed25, events, tmp_path):
    scorer, _ = make_scorer(seed25, events, tmp_path, store=constant(4.0), noise=NOISE)
    rows = seed25.head(30).to_dict("records")
    records = [scorer.score_reading(r) for r in rows]
    start = pd.Timestamp(rows[0]["timestamp"])
    for row, rec in zip(rows, records):
        elapsed = pd.Timestamp(row["timestamp"]) - start
        if elapsed < pd.Timedelta(hours=1):
            assert rec["health"] == GREEN
    assert records[-1]["health"] == YELLOW


@needs_seed
def test_explainer_called_only_for_alerts_and_checked(seed25, events, tmp_path):
    spy = SpyExplain({
        "reasons": [{"part": "gearbox_bearing", "residual_c": 6.0, "hours": 72, "text": "t"}],
        "sentence": "No unusual temperature readings detected.",
    })
    scorer, _ = make_scorer(seed25, events, tmp_path, store=constant(6.0), noise=NOISE, explain_fn=spy)
    records = [scorer.score_reading(r) for r in seed25.head(30).to_dict("records")]
    alerts = [r for r in records if r["health"] in (YELLOW, RED)]
    assert alerts and spy.calls == len(alerts)  # L-05: only yellow/red
    for rec in alerts:
        assert rec["sentence"] == ""  # contradictory sentence removed (L-17)
        assert rec["reasons"][0]["hours"] == 72.0 and isinstance(rec["reasons"][0]["hours"], float)
    for rec in records:
        if rec["health"] == GREEN:
            assert rec["reasons"] == [] and rec["sentence"] == ""


@needs_seed
def test_no_explainer_gives_empty_reasons_on_alerts(seed25, events, tmp_path):
    scorer, _ = make_scorer(seed25, events, tmp_path, store=constant(6.0), noise=NOISE)
    records = [scorer.score_reading(r) for r in seed25.head(30).to_dict("records")]
    alerts = [r for r in records if r["health"] in (YELLOW, RED)]
    assert alerts and all(r["reasons"] == [] and r["sentence"] == "" for r in alerts)
