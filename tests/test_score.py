import json

import pandas as pd

from upkeep.health.levels import HealthConfig, YELLOW
from upkeep.health.score import DatasetHealth, load_noise_levels, noise_path
from upkeep.parts import PARTS

CFG = HealthConfig(yellow_z=3.0, red_z=5.0, persistence_hours=1.0, max_gap_minutes=30.0)


def test_missing_noise_file_gives_all_none(tmp_path):
    assert load_noise_levels("A-0", models_root=tmp_path) == {p: None for p in PARTS}


def test_noise_file_is_read_and_bad_values_dropped(tmp_path):
    path = noise_path("A-10", "lgbm", tmp_path)
    assert path.name == "lgbm_noise_10.json"
    path.write_text(json.dumps({"gearbox_bearing": 0.8, "generator_bearing_1": -1, "generator_bearing_2": "x"}))
    assert load_noise_levels(10, models_root=tmp_path) == {
        "gearbox_bearing": 0.8, "generator_bearing_1": None, "generator_bearing_2": None,
    }


def test_corrupt_noise_file_gives_all_none(tmp_path):
    noise_path(0, "lgbm", tmp_path).write_text("{not json")
    assert load_noise_levels(0, models_root=tmp_path) == {p: None for p in PARTS}


def test_dataset_health_without_noise_is_unknown():
    h = DatasetHealth("A-0", CFG, {p: None for p in PARTS})
    a = h.assess(pd.Timestamp("2023-01-01"), {"gearbox_bearing": 10.0})
    assert a.health is None and a.health_score is None


def test_dataset_health_normalises_with_its_own_noise():
    h = DatasetHealth(0, CFG, {"gearbox_bearing": 2.0})
    t0 = pd.Timestamp("2023-01-01")
    for i in range(7):
        a = h.assess(t0 + pd.Timedelta(minutes=10 * i), {"gearbox_bearing": 8.0})  # z = 4
    assert a.health == YELLOW and a.sustained_z == 4.0
    assert h.dataset_id == "A-0"
