"""Model store tests.

Fixture models are fitted on tiny synthetic data inside ``tmp_path`` only, to
exercise loading. No project models are trained or saved.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from upkeep.models.baseline_rule import FixedLimitBaseline
from upkeep.models.linear_nbm import LinearNormalModel
from upkeep.stream.model_store import PARTS, ModelStore

FEATURES = ["wind_speed_avg", "temp_1h_ago"]


def save_linear(path, slope=1.0, features=FEATURES):
    X = pd.DataFrame({f: np.arange(10, dtype=float) + i for i, f in enumerate(features)})
    y = slope * X[features[0]] + 20.0
    model = LinearNormalModel()
    model.fit(X, y)
    model.save(path)


def reading(**overrides):
    r = {
        "wind_speed_avg": 5.0, "temp_1h_ago": 6.0,
        "gearbox_bearing_temp": 30.0,
        "generator_bearing_1_temp": 40.0,
        "generator_bearing_2_temp": 41.0,
        "status_id": 0,
    }
    r.update(overrides)
    return r


def test_no_models_gives_all_null_residuals(tmp_path):
    store = ModelStore("lgbm", models_root=tmp_path / "does_not_exist")
    assert store.residuals("A-0", reading()) == {p: None for p in PARTS}
    assert store.status("A-0") == {p: "missing" for p in PARTS}


def test_missing_model_logs_once_and_is_cached(tmp_path, caplog):
    store = ModelStore("lgbm", models_root=tmp_path)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            store.residuals(0, reading())
    assert len([r for r in caplog.records if "No lgbm model" in r.message]) == len(PARTS)


def test_loads_canonical_name_and_computes_residual(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_25.joblib")
    store = ModelStore("linear", models_root=tmp_path)
    res = store.residuals("A-25", reading())
    assert res["gearbox_bearing"] == pytest.approx(30.0 - 25.0)
    assert res["generator_bearing_1"] is None and res["generator_bearing_2"] is None


def test_models_are_per_dataset(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_25.joblib")
    store = ModelStore("linear", models_root=tmp_path)
    assert store.residuals("A-0", reading())["gearbox_bearing"] is None


def test_legacy_sensor_column_name_is_accepted(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_temp_10.joblib")
    store = ModelStore("linear", models_root=tmp_path)
    assert store.model_path(10, "gearbox_bearing").name == "linear_gearbox_bearing_temp_10.joblib"
    assert store.residuals(10, reading())["gearbox_bearing"] == pytest.approx(5.0)


def test_canonical_name_wins_over_legacy(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_10.joblib", slope=1.0)
    save_linear(tmp_path / "linear_gearbox_bearing_temp_10.joblib", slope=2.0)
    store = ModelStore("linear", models_root=tmp_path)
    assert store.model_path(10, "gearbox_bearing").name == "linear_gearbox_bearing_10.joblib"


@pytest.mark.parametrize("override", [
    {"temp_1h_ago": None}, {"temp_1h_ago": float("nan")}, {"wind_speed_avg": np.nan},
])
def test_missing_feature_gives_null_residual(tmp_path, override):
    save_linear(tmp_path / "linear_gearbox_bearing_0.joblib")
    store = ModelStore("linear", models_root=tmp_path)
    assert store.residuals(0, reading(**override))["gearbox_bearing"] is None


def test_feature_absent_from_reading_gives_null(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_0.joblib")
    r = reading()
    del r["temp_1h_ago"]
    assert ModelStore("linear", models_root=tmp_path).residuals(0, r)["gearbox_bearing"] is None


def test_missing_actual_gives_null_residual(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_0.joblib")
    store = ModelStore("linear", models_root=tmp_path)
    assert store.residuals(0, reading(gearbox_bearing_temp=np.nan))["gearbox_bearing"] is None


def test_corrupt_file_gives_null_and_error_status(tmp_path):
    (tmp_path / "linear_gearbox_bearing_0.joblib").write_bytes(b"not a joblib file")
    store = ModelStore("linear", models_root=tmp_path)
    assert store.residuals(0, reading())["gearbox_bearing"] is None
    assert store.status(0)["gearbox_bearing"].startswith("error")


def test_model_using_forbidden_input_is_refused(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_0.joblib", features=["wind_speed_avg", "status_id"])
    store = ModelStore("linear", models_root=tmp_path)
    assert store.get(0, "gearbox_bearing") is None
    assert "forbidden" in store.status(0)["gearbox_bearing"]


def test_model_using_its_own_target_is_refused(tmp_path):
    save_linear(tmp_path / "linear_gearbox_bearing_0.joblib",
                features=["wind_speed_avg", "gearbox_bearing_temp"])
    assert ModelStore("linear", models_root=tmp_path).get(0, "gearbox_bearing") is None


def test_untrained_baseline_gives_null_not_none_prediction_crash(tmp_path):
    FixedLimitBaseline().save(tmp_path / "baseline_gearbox_bearing_0.joblib")  # fixed_value None
    store = ModelStore("baseline", models_root=tmp_path)
    assert store.residuals(0, reading())["gearbox_bearing"] is None


def test_baseline_without_feature_list_works(tmp_path):
    m = FixedLimitBaseline()
    m.fit(None, pd.Series([20.0, 22.0]))
    m.save(tmp_path / "baseline_gearbox_bearing_0.joblib")
    store = ModelStore("baseline", models_root=tmp_path)
    assert store.residuals(0, reading())["gearbox_bearing"] == pytest.approx(9.0)


def test_unknown_model_or_part_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        ModelStore("xgboost", models_root=tmp_path)
    with pytest.raises(ValueError):
        ModelStore("lgbm", models_root=tmp_path).get(0, "rotor")


def test_seed_row_with_no_models_gives_nulls(tmp_path):
    from upkeep.paths import data_dir
    seed = data_dir() / "seed" / "25.parquet"
    if not seed.exists():
        pytest.skip("seed data not present")
    row = pd.read_parquet(seed).iloc[100]
    store = ModelStore("lgbm", models_root=tmp_path)
    assert store.residuals("A-25", row) == {p: None for p in PARTS}
