"""T-16 noise calibration (upkeep.health.noise) on synthetic data.

No real data or trained models are used. A spy model records exactly which rows
it is fitted and evaluated on, so the split rules can be checked directly.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from upkeep.health import noise
from upkeep.health.score import load_noise_levels, noise_path
from upkeep.models.linear_nbm import LinearNormalModel
from upkeep.parts import PART_SENSORS, PARTS

TARGET = PART_SENSORS["gearbox_bearing"]
FEATURES = ["x1", "x2"]


def synthetic(seed=0, train_days=180, test_days=7, sigma=1.5):
    """10-minute readings: a training half-year, then a locked test week."""
    rng = np.random.default_rng(seed)
    n_train, n_test = train_days * 144, test_days * 144
    ts = pd.date_range("2023-01-01", periods=n_train + n_test, freq="10min")
    df = pd.DataFrame({"timestamp": ts, "row_id": np.arange(len(ts)),
                       "split": ["train"] * n_train + ["prediction"] * n_test,
                       "status_id": 0,
                       "x1": rng.normal(size=len(ts)), "x2": rng.normal(size=len(ts))})
    for i, sensor in enumerate(PART_SENSORS.values()):
        df[sensor] = 40 + (2 + i) * df["x1"] - df["x2"] + rng.normal(scale=sigma, size=len(ts))
    return df


class Spy:
    """Minimal model with the owner's fit/predict interface that records its rows."""

    fitted_on = []
    predicted_on = []

    def __init__(self):
        self.features = FEATURES
        self.mean = None

    def fit(self, X, y):
        Spy.fitted_on.append((X.index.copy(), y.copy()))
        self.mean = float(y.mean())

    def predict(self, X):
        Spy.predicted_on.append(X.index.copy())
        return np.full(len(X), self.mean)


@pytest.fixture(autouse=True)
def _reset_spy():
    Spy.fitted_on, Spy.predicted_on = [], []


# ---- robust spread -----------------------------------------------------------

def test_robust_spread_is_scaled_mad():
    # median 3, |deviations| = 2, 1, 0, 1, 97 -> MAD 1
    assert noise.robust_spread([1, 2, 3, 4, 100]) == pytest.approx(1.4826)


def test_robust_spread_ignores_non_finite_and_rejects_unusable():
    assert noise.robust_spread([1, 2, 3, 4, 100, np.nan, np.inf]) == pytest.approx(1.4826)
    assert noise.robust_spread([]) is None
    assert noise.robust_spread([np.nan, np.nan]) is None
    assert noise.robust_spread([5.0] * 10) is None  # zero spread would divide by zero


def test_robust_spread_matches_normal_sigma():
    sample = np.random.default_rng(1).normal(scale=2.0, size=50_000)
    assert noise.robust_spread(sample) == pytest.approx(2.0, rel=0.03)


# ---- split rules ------------------------------------------------------------

def test_split_uses_last_two_months_of_training_only():
    df = synthetic()
    fit, cal, cutoff = noise.calibration_split(df, TARGET)
    train = df[df.split == "train"]
    assert cutoff == train.timestamp.max() - pd.DateOffset(months=2)
    assert (fit.timestamp < cutoff).all() and (cal.timestamp >= cutoff).all()
    assert len(fit) + len(cal) == len(train)
    assert set(fit.split) == {"train"} and set(cal.split) == {"train"}
    test_ids = set(df.loc[df.split == "prediction", "row_id"])
    assert not test_ids & (set(fit.row_id) | set(cal.row_id))


def test_calibration_rows_come_after_fit_rows():
    fit, cal, _ = noise.calibration_split(synthetic(), TARGET)
    assert fit.timestamp.max() < cal.timestamp.min()


def test_missing_targets_are_dropped_before_fit_and_calibration():
    df = synthetic()
    _, cal0, _ = noise.calibration_split(df, TARGET)
    holes = [100, 200, int(cal0.index[5]), int(cal0.index[50])]
    df.loc[holes, TARGET] = np.nan
    fit, cal, _ = noise.calibration_split(df, TARGET)
    assert fit[TARGET].notna().all() and cal[TARGET].notna().all()
    assert not set(holes) & (set(fit.index) | set(cal.index))

    noise.calibrate_part(df, "gearbox_bearing", Spy())
    (fit_idx, y), = Spy.fitted_on
    assert y.notna().all() and not set(holes) & set(fit_idx)
    assert not set(holes) & set(Spy.predicted_on[0])


def test_model_sees_only_fit_rows_and_is_scored_only_on_calibration_rows():
    df = synthetic()
    fit, cal, _ = noise.calibration_split(df, TARGET)
    info = noise.calibrate_part(df, "gearbox_bearing", Spy())
    (fit_idx, _), = Spy.fitted_on
    (pred_idx,) = Spy.predicted_on
    assert set(fit_idx) == set(fit.index)
    assert set(pred_idx) == set(cal.index)
    test_idx = set(df.index[df.split == "prediction"])
    assert not test_idx & (set(fit_idx) | set(pred_idx))  # locked test rows never touched
    assert info["n_fit"] == len(fit) and info["n_calibration"] == len(cal)


def test_calibration_rows_with_missing_features_are_left_out():
    df = synthetic()
    _, cal, _ = noise.calibration_split(df, TARGET)
    gap = list(cal.index[:30])
    df.loc[gap, "x2"] = np.nan  # no residual for these in live scoring either (L-07)
    info = noise.calibrate_part(df, "gearbox_bearing", Spy())
    assert not set(gap) & set(Spy.predicted_on[0])
    assert info["n_calibration_dropped_missing_features"] == 30


def test_refuses_when_training_overlaps_test_period():
    df = synthetic()
    df.loc[df.index[-1], "split"] = "train"  # a training row after the test rows begin
    with pytest.raises(ValueError, match="locked test"):
        noise.calibration_split(df, TARGET)


def test_too_few_calibration_rows_gives_no_noise_level():
    df = synthetic()
    _, cal, _ = noise.calibration_split(df, TARGET)
    df.loc[cal.index[: len(cal) - 50], "x1"] = np.nan
    assert noise.calibrate_part(df, "gearbox_bearing", Spy())["noise"] is None


# ---- end to end: saved model -> noise file -> existing loader -----------------------

@pytest.fixture
def project(tmp_path):
    data_root, models_root = tmp_path / "Data", tmp_path / "models_saved"
    (data_root / "cln").mkdir(parents=True)
    df = synthetic(sigma=1.5)
    df.to_csv(data_root / "cln" / "7_cln.csv", sep=";", index=False)
    train = df[df.split == "train"]
    for part in ("gearbox_bearing", "generator_bearing_1"):  # generator_bearing_2 has no model
        model = LinearNormalModel()
        model.fit(train[FEATURES], train[PART_SENSORS[part]])
        model.save(models_root / f"linear_{PART_SENSORS[part]}_7.joblib")  # Varun's file naming
    return data_root, models_root


def test_noise_file_loads_through_existing_loader(project):
    data_root, models_root = project
    levels, meta = noise.compute_noise_levels(7, "linear", models_root, data_root)
    path = noise.write_noise_file(7, levels, meta, "linear", models_root)
    assert path == noise_path(7, "linear", models_root) == models_root / "linear_noise_7.json"

    loaded = load_noise_levels("A-7", "linear", models_root)
    assert set(loaded) == set(PARTS)
    for part in ("gearbox_bearing", "generator_bearing_1"):
        assert loaded[part] == pytest.approx(levels[part], abs=1e-6)
        assert loaded[part] == pytest.approx(1.5, rel=0.1)  # recovers the injected noise
    assert loaded["generator_bearing_2"] is None  # no model -> unusable, never a guess

    raw = json.loads(path.read_text())
    assert raw["generator_bearing_2"] is None and raw["_meta"]["status"].startswith("T-16")
    assert raw["_meta"]["parts"]["gearbox_bearing"]["model_file"] == "linear_gearbox_bearing_temp_7.joblib"


def test_cli_writes_noise_files(project, capsys):
    data_root, models_root = project
    assert noise.main(["--datasets", "7", "--model", "linear",
                       "--models-root", str(models_root), "--data-dir", str(data_root)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert math.isfinite(out["A-7"]["noise"]["gearbox_bearing"])
    assert (models_root / "linear_noise_7.json").is_file()


def test_cli_rejects_duplicate_datasets(project):
    data_root, models_root = project
    with pytest.raises(SystemExit):
        noise.main(["--datasets", "7", "A-7", "--model", "linear",
                    "--models-root", str(models_root), "--data-dir", str(data_root)])
