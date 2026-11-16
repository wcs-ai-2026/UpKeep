"""Tests for the review fixes: explanation validation, rerun/append, timestamps,
CLI argument checks, wall-clock summary, per-part models and the anomaly-scorer
extension point. No models are trained; synthetic residuals come from a test
double. Thresholds are test-only values."""
import json

import numpy as np
import pandas as pd
import pytest

from upkeep.health.levels import RED, YELLOW, HealthConfig
from upkeep.health.score import load_noise_levels, noise_path
from upkeep.parts import PARTS
from upkeep.paths import data_dir
from upkeep.stream.live import (
    HealthWriter, LiveScorer, is_contradictory, stub_explain, validate_explanation,
)
from upkeep.stream.model_store import ModelStore
from upkeep.stream.replay import load_dataset, main, replay

SEED = data_dir() / "seed"
needs_seed = pytest.mark.skipif(not (SEED / "0.parquet").exists(), reason="seed data not present")
CFG = HealthConfig(yellow_z=3.0, red_z=5.0, persistence_hours=1.0, max_gap_minutes=30.0)
NOISE = {p: 1.0 for p in PARTS}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class FakeStore:
    def __init__(self, fn):
        self.fn = fn

    def residuals(self, dataset_id, reading):
        return self.fn(dataset_id, reading)

    def status(self, dataset_id):
        return {p: "loaded" for p in PARTS}


def hot(value):
    return FakeStore(lambda d, r: {"gearbox_bearing": value, "generator_bearing_1": 0.0,
                                   "generator_bearing_2": 0.0})


# ---- explanation validation (C2) ------------------------------------------------

def reason(part="gearbox_bearing", residual=6.0, hours=72, text="t"):
    return {"part": part, "residual_c": residual, "hours": hours, "text": text}


def test_hours_become_float():
    reasons, _, issues = validate_explanation({"reasons": [reason(hours=72)], "sentence": "s"})
    assert reasons[0]["hours"] == 72.0 and isinstance(reasons[0]["hours"], float)
    assert issues == []


def test_reasons_sorted_by_residual_and_cut_to_three():
    raw = [reason("a", 2.0), reason("b", None), reason("c", 9.0), reason("d", 5.0), reason("e", 7.0)]
    reasons, _, issues = validate_explanation({"reasons": raw, "sentence": "s"})
    assert [r["part"] for r in reasons] == ["c", "e", "d"]
    assert any("cut 2" in i for i in issues)


def test_reasons_without_residual_go_last_in_original_order():
    raw = [reason("x", None), reason("y", None), reason("z", 1.0)]
    reasons, _, _ = validate_explanation({"reasons": raw})
    assert [r["part"] for r in reasons] == ["z", "x", "y"]


@pytest.mark.parametrize("bad", [
    "not a dict", {"part": "", "hours": 1, "text": "t"}, {"part": "a", "hours": "x", "text": "t"},
    {"part": "a", "hours": -1, "text": "t"}, {"part": "a", "hours": 1}, {"part": "a", "hours": float("nan"), "text": "t"},
    {"part": "a", "hours": 1, "text": "t", "residual_c": "hot"},
])
def test_invalid_reasons_are_dropped(bad):
    reasons, _, issues = validate_explanation({"reasons": [bad, reason()]})
    assert [r["part"] for r in reasons] == ["gearbox_bearing"]
    assert "dropped an invalid reason" in issues


def test_contradictory_sentence_removed():
    _, sentence, issues = validate_explanation(
        {"reasons": [], "sentence": "No unusual temperature readings detected."})
    assert sentence == "" and any("contradicts" in i for i in issues)
    assert is_contradictory("  NO UNUSUAL readings ")
    assert not is_contradictory("Gearbox bearing has run 6°C hotter than expected.")


@pytest.mark.parametrize("result", [None, [], "text", {"reasons": "x", "sentence": 5}])
def test_malformed_explainer_output_is_safe(result):
    reasons, sentence, issues = validate_explanation(result)
    assert reasons == [] and sentence == "" and issues


@needs_seed
def test_stub_explain_through_pipeline_never_contradicts_alert(tmp_path):
    # Residual 4 degC: yellow/red with noise 1, but below the stub's 5 degC cut-off,
    # so the stub returns "No unusual temperature readings detected."
    out = tmp_path / "h.jsonl"
    summary = replay([25], config=CFG, output=out, model_store=hot(4.0), noise_levels={25: NOISE},
                     explain_fn=stub_explain())
    alerts = [r for r in read_jsonl(out) if r["health"] in (YELLOW, RED)]
    assert alerts
    assert all(r["sentence"] == "" and r["reasons"] == [] for r in alerts)
    assert summary["datasets"]["A-25"]["alerts_without_reasons"] == len(alerts)
    assert summary["explanation_issues"]["removed a sentence that contradicts the alert"] == len(alerts)
    assert summary["explainer"] == "upkeep.stubs.explain"


@needs_seed
def test_stub_reason_hours_are_float_in_output(tmp_path):
    out = tmp_path / "h.jsonl"
    replay([25], config=CFG, output=out, model_store=hot(9.0), noise_levels={25: NOISE},
           explain_fn=stub_explain(), limit=20)
    alerts = [r for r in read_jsonl(out) if r["health"] in (YELLOW, RED)]
    assert alerts and all(isinstance(x["hours"], float) for r in alerts for x in r["reasons"])
    assert '"hours": 72.0' in out.read_text()


@needs_seed
def test_no_explainer_gives_empty_reasons(tmp_path):
    out = tmp_path / "h.jsonl"
    summary = replay([25], config=CFG, output=out, model_store=hot(9.0), noise_levels={25: NOISE}, limit=20)
    alerts = [r for r in read_jsonl(out) if r["health"] in (YELLOW, RED)]
    assert alerts and all(r["reasons"] == [] and r["sentence"] == "" for r in alerts)
    assert summary["explainer"] is None


# ---- rerun / append (C5) ----------------------------------------------------------

@needs_seed
def test_rerun_overwrites_by_default(tmp_path):
    out = tmp_path / "h.jsonl"
    for _ in range(2):
        replay([25], config=CFG, output=out, limit=5, models_root=tmp_path / "none")
    recs = read_jsonl(out)
    assert len(recs) == 5 and len({(r["dataset_id"], r["row_id"]) for r in recs}) == 5


@needs_seed
def test_explicit_append_appends(tmp_path):
    out = tmp_path / "h.jsonl"
    replay([25], config=CFG, output=out, limit=5, models_root=tmp_path / "none")
    summary = replay([25], config=CFG, output=out, limit=5, append=True, models_root=tmp_path / "none")
    assert len(read_jsonl(out)) == 10 and summary["appended"] is True


@needs_seed
def test_cli_overwrites_and_appends(tmp_path, capsys):
    out = tmp_path / "cli.jsonl"
    base = ["--datasets", "25", "--limit", "3", "--untuned-placeholder-config", "--output", str(out),
            "--models-root", str(tmp_path / "none")]
    main(base)
    main(base)
    assert len(read_jsonl(out)) == 3
    main(base + ["--append"])
    assert len(read_jsonl(out)) == 6


@needs_seed
def test_evaluation_mode_overwrites_each_file(tmp_path):
    for _ in range(2):
        replay([25], config=CFG, mode="evaluation", output_dir=tmp_path, limit=4, models_root=tmp_path / "none")
    assert len(read_jsonl(tmp_path / "health_A-25.jsonl")) == 4


def test_health_writer_default_overwrites(tmp_path):
    path = tmp_path / "w.jsonl"
    path.write_text('{"old": 1}\n')
    with HealthWriter(path) as w:
        w.write({"new": 1})
    assert read_jsonl(path) == [{"new": 1}]
    with HealthWriter(path, append=True) as w:
        w.write({"more": 1})
    assert read_jsonl(path) == [{"new": 1}, {"more": 1}]


# ---- timestamps --------------------------------------------------------------------

@needs_seed
def test_nat_timestamp_rejected_on_load(tmp_path):
    df = pd.read_parquet(SEED / "25.parquet").head(20)
    df.loc[5, "timestamp"] = pd.NaT
    df.to_parquet(tmp_path / "25.parquet", index=False)
    with pytest.raises(ValueError, match="NaT"):
        load_dataset(25, tmp_path)


@needs_seed
def test_unparseable_timestamp_rejected_on_load(tmp_path):
    df = pd.read_parquet(SEED / "25.parquet").head(20)
    df["timestamp"] = df["timestamp"].astype(str)
    df.loc[3, "timestamp"] = "not a time"
    df.to_parquet(tmp_path / "25.parquet", index=False)
    with pytest.raises(ValueError, match="timestamp"):
        load_dataset(25, tmp_path)


@needs_seed
@pytest.mark.parametrize("value", [None, pd.NaT, np.nan])
def test_live_rejects_missing_timestamp(value, tmp_path):
    df = load_dataset(25)
    s = LiveScorer(CFG, model_store=hot(0.0))
    s.register_dataset(25, df, NOISE)
    reading = {**df.iloc[0].to_dict(), "timestamp": value}
    with pytest.raises(ValueError, match="timestamp"):
        s.score_reading(reading)


# ---- argument checks -------------------------------------------------------------------

@pytest.mark.parametrize("extra", [
    ["--mode", "farm", "--output-dir", "x"],
    ["--mode", "evaluation", "--output", "x.jsonl"],
    ["--limit", "0"],
    ["--speed", "-1"],
    ["--unknown-flag"],
    ["--out", "x.jsonl"],                 # abbreviations are not accepted
    ["--part-model", "rotor=lgbm"],
    ["--part-model", "gearbox_bearing"],
    ["--part-model", "gearbox_bearing=lgbm", "--part-model", "gearbox_bearing=linear"],
])
def test_cli_rejects_unsupported_or_ignored_arguments(extra):
    with pytest.raises(SystemExit) as exc:
        main(["--datasets", "25", "--untuned-placeholder-config", *extra])
    assert exc.value.code == 2


@pytest.mark.parametrize("kwargs", [
    {"mode": "farm", "output_dir": "x"},
    {"mode": "evaluation", "output": "x.jsonl"},
    {"limit": 0}, {"speed": -0.5}, {"mode": "live"}, {"split": "validation"},
    {"noise_levels": {10: NOISE}},
])
def test_replay_function_rejects_ignored_arguments(kwargs):
    with pytest.raises(ValueError):
        replay([25], config=CFG, **kwargs)


def test_replay_requires_datasets():
    with pytest.raises(ValueError):
        replay([], config=CFG)


# ---- wall-clock summary ---------------------------------------------------------------

@needs_seed
@pytest.mark.parametrize("mode", ["farm", "evaluation"])
def test_summary_reports_wall_time(tmp_path, mode):
    target = {"output": tmp_path / "h.jsonl"} if mode == "farm" else {"output_dir": tmp_path}
    summary = replay([10, 25], config=CFG, mode=mode, limit=10, models_root=tmp_path / "none", **target)
    assert isinstance(summary["wall_time_s"], float) and summary["wall_time_s"] > 0
    assert summary["total_readings"] == (10 if mode == "farm" else 20)
    assert summary["validated"] is False


# ---- per-part model types --------------------------------------------------------------

def test_model_store_per_part_types(tmp_path):
    store = ModelStore({"gearbox_bearing": "lgbm", "generator_bearing_1": "linear"}, models_root=tmp_path)
    assert store.part_models == {"gearbox_bearing": "lgbm", "generator_bearing_1": "linear",
                                 "generator_bearing_2": "lgbm"}
    assert store.candidate_paths(3, "generator_bearing_1")[0].name == "linear_generator_bearing_1_3.joblib"
    assert store.candidate_paths(3, "gearbox_bearing")[0].name == "lgbm_gearbox_bearing_3.joblib"
    assert store.model_name == store.part_models
    assert ModelStore("linear", models_root=tmp_path).model_name == "linear"


@pytest.mark.parametrize("bad", [{"rotor": "lgbm"}, {"gearbox_bearing": "xgboost"}])
def test_model_store_rejects_bad_mapping(bad, tmp_path):
    with pytest.raises(ValueError):
        ModelStore(bad, models_root=tmp_path)


def test_noise_read_from_each_parts_own_model_file(tmp_path):
    noise_path(7, "lgbm", tmp_path).write_text(json.dumps({"gearbox_bearing": 1.1, "generator_bearing_1": 9.9}))
    noise_path(7, "linear", tmp_path).write_text(json.dumps({"generator_bearing_1": 2.2}))
    levels = load_noise_levels(7, {"gearbox_bearing": "lgbm", "generator_bearing_1": "linear"}, tmp_path)
    assert levels == {"gearbox_bearing": 1.1, "generator_bearing_1": 2.2, "generator_bearing_2": None}


# ---- anomaly scorer extension point -------------------------------------------------------

class ConstantAnomaly:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def score(self, dataset_id, residuals, reading):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


@needs_seed
def test_anomaly_score_is_reported_but_never_changes_colour(tmp_path):
    outs = {}
    for value in (0.0, 1.0):
        out = tmp_path / f"h{value}.jsonl"
        replay([25], config=CFG, output=out, model_store=hot(4.0), noise_levels={25: NOISE},
               anomaly_scorer=ConstantAnomaly(value), limit=30)
        outs[value] = read_jsonl(out)
    assert all(r["anomaly_score"] == 1.0 for r in outs[1.0])
    assert [r["health"] for r in outs[0.0]] == [r["health"] for r in outs[1.0]]
    assert [r["health_score"] for r in outs[0.0]] == [r["health_score"] for r in outs[1.0]]


@needs_seed
@pytest.mark.parametrize("value,expected", [(RuntimeError("boom"), None), (float("nan"), None), (1.7, 1.7)])
def test_anomaly_scorer_failures_and_out_of_range(value, expected, tmp_path):
    out = tmp_path / "h.jsonl"
    replay([25], config=CFG, output=out, model_store=hot(0.0), noise_levels={25: NOISE},
           anomaly_scorer=ConstantAnomaly(value), limit=3)
    assert [r["anomaly_score"] for r in read_jsonl(out)] == [expected] * 3


@needs_seed
def test_no_anomaly_scorer_gives_null(tmp_path):
    out = tmp_path / "h.jsonl"
    replay([25], config=CFG, output=out, model_store=hot(0.0), noise_levels={25: NOISE}, limit=3)
    assert all(r["anomaly_score"] is None for r in read_jsonl(out))
