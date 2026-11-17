"""SCRUM-45 checks with the real trained models.

Skipped automatically unless ``models_saved/`` holds a loadable LightGBM model
for every part of datasets 0, 10 and 25 (the folder is gitignored; see the
PR description for how the files were produced). Noise levels come from the
T-16 files when present, otherwise they are calibrated here without writing.

Thresholds are the UNTUNED placeholder (T-08 pending): these tests check wiring
and the persistence behaviour, not detection quality.
"""
import math

import numpy as np
import pandas as pd
import pytest

from upkeep.health.levels import GREEN, RED, UNTUNED_PLACEHOLDER_CONFIG as CFG, YELLOW
from upkeep.health.noise import compute_noise_levels
from upkeep.health.score import load_noise_levels
from upkeep.parts import PART_SENSORS, PARTS
from upkeep.stream.faults import load_events
from upkeep.stream.live import LiveScorer
from upkeep.stream.model_store import ModelStore
from upkeep.stream.replay import load_dataset, replay

DATASETS = (0, 10, 25)


def _all_models_loaded() -> bool:
    try:
        store = ModelStore("lgbm")
        return all(s == "loaded" for n in DATASETS for s in store.status(n).values())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _all_models_loaded(), reason="real models not present in models_saved/")


@pytest.fixture(scope="module")
def store():
    return ModelStore("lgbm")


@pytest.fixture(scope="module")
def noise25():
    levels = load_noise_levels(25)
    if any(v is None for v in levels.values()):
        levels, _ = compute_noise_levels(25)  # same method, nothing written
    assert all(v is not None and v > 0 for v in levels.values())
    return levels


@pytest.fixture(scope="module")
def test25():
    df = load_dataset(25)
    return df[df["split"] == "prediction"].reset_index(drop=True)


def _complete(df, store, n):
    features = set()
    for part in PARTS:
        features |= set(store.get(n, part).features)
    return df.dropna(subset=sorted(features) + list(PART_SENSORS.values()))


# ---- real residuals -----------------------------------------------------------

@pytest.mark.parametrize("n", DATASETS)
def test_real_models_give_finite_residuals_on_seed_readings(store, n):
    df = load_dataset(n)
    rows = _complete(df[df["split"] == "prediction"], store, n).head(300)
    assert len(rows) == 300
    for reading in rows.to_dict("records"):
        res = store.residuals(n, reading)
        assert set(res) == set(PARTS)
        for part, value in res.items():
            assert isinstance(value, float) and math.isfinite(value), (part, value)
            assert abs(value) < 50  # a temperature error in degC, not garbage


def test_models_use_no_forbidden_inputs(store):
    for n in DATASETS:
        for part in PARTS:
            features = set(store.get(n, part).features)
            assert not features & {"status_id", "split", "fault_event", "days_to_fault"}
            assert PART_SENSORS[part] not in features


# ---- colours instead of null -----------------------------------------------------

def test_real_model_replay_produces_colours(tmp_path, noise25):
    out = tmp_path / "health.jsonl"
    summary = replay([25], config=CFG, output=out, limit=720, noise_levels={25: noise25})
    counts = summary["datasets"]["A-25"]["health_counts"]
    scored = sum(v for k, v in counts.items() if k != "unknown")
    assert summary["datasets"]["A-25"]["lines_written"] == 720
    assert set(counts) <= {GREEN, YELLOW, RED, "unknown"}
    assert counts.get(GREEN, 0) > 0
    assert scored >= 0.7 * 720  # unknown only where a feature such as temp_1h_ago is missing (L-07)


# ---- persistence with real models ---------------------------------------------------

def _run(df, noise, transform=None):
    scorer = LiveScorer(CFG, model_store=ModelStore("lgbm"), events=load_events())
    scorer.register_dataset(25, load_dataset(25), noise)
    records = []
    for i, reading in enumerate(df.to_dict("records")):
        if transform is not None:
            reading = transform(i, reading)
        records.append(scorer.score_reading(reading))
    return records


def _clean_stretch(df, records, hours):
    """Index i where readings i-18 .. i+hours*6 are 10 min apart, all green, gearbox scored."""
    need = int(hours * 6)
    ts = pd.to_datetime(df["timestamp"])
    step_ok = (ts.diff() == pd.Timedelta("10min")).to_numpy()
    ok = np.array([r["health"] == GREEN and r["residuals"]["gearbox_bearing"] is not None for r in records])
    for i in range(18, len(df) - need):
        if step_ok[i - 17:i + need + 1].all() and ok[i - 18:i + need + 1].all():
            return i
    pytest.skip("no clean all-green stretch long enough in this sample")


@pytest.fixture(scope="module")
def baseline(test25, noise25):
    df = test25.head(1000)
    return df, _run(df, noise25)


def test_single_anomalous_reading_does_not_change_colour(baseline, noise25):
    df, base = baseline
    k = _clean_stretch(df, base, hours=1)
    spike = 20 * noise25["gearbox_bearing"]

    def transform(i, reading):
        if i == k:
            reading = dict(reading)
            reading["gearbox_bearing_temp"] += spike
        return reading

    spiked = _run(df, noise25, transform)
    assert spiked[k]["residuals"]["gearbox_bearing"] >= base[k]["residuals"]["gearbox_bearing"] + spike - 1e-6
    assert spiked[k]["health"] == base[k]["health"] == GREEN  # one bad reading: colour unchanged
    assert [r["health"] for r in spiked] == [r["health"] for r in base]


def test_colour_changes_only_after_the_problem_lasts(baseline, noise25):
    df, base = baseline
    hours = CFG.persistence_hours
    k = _clean_stretch(df, base, hours=hours + 1)
    shift = (CFG.red_z + 2) * noise25["gearbox_bearing"]  # clearly above red, sustained
    ts = pd.to_datetime(df["timestamp"])
    onset = ts[k]

    def transform(i, reading):
        if i >= k:
            reading = dict(reading)
            reading["gearbox_bearing_temp"] += shift
        return reading

    shifted = _run(df, noise25, transform)
    before_window = [i for i in range(k, len(df)) if ts[i] - onset < pd.Timedelta(hours=hours)]
    assert all(shifted[i]["health"] == GREEN for i in before_window)  # not before the window is covered
    red_at = next(i for i in range(k, len(df)) if shifted[i]["health"] == RED)
    assert ts[red_at] - onset >= pd.Timedelta(hours=hours)
    assert ts[red_at] - onset <= pd.Timedelta(hours=hours + 1)
