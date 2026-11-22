# Tests for the Isolation Forest model.
# All models here are fitted on small fake residual data inside tmp_path.
# No real project models are trained or touched.
import numpy as np
import pandas as pd
import pytest

from upkeep.models.iforest import IForestScorer, IsolationForestAnomalyModel
from upkeep.parts import PARTS


# Making fake residuals that look like normal, healthy readings.
def normal_residuals(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({part: rng.normal(0.0, 1.0, n) for part in PARTS})


def test_predict_score_is_between_0_and_1():
    model = IsolationForestAnomalyModel()
    model.fit(normal_residuals())
    scores = model.predict_score(normal_residuals(n=20, seed=1))
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_obviously_unusual_row_scores_higher_than_typical_rows():
    model = IsolationForestAnomalyModel()
    model.fit(normal_residuals())

    typical = pd.DataFrame({part: [0.0] for part in PARTS})
    unusual = pd.DataFrame({part: [50.0] for part in PARTS})

    typical_score = model.predict_score(typical)[0]
    unusual_score = model.predict_score(unusual)[0]
    assert unusual_score > typical_score


def test_save_load_roundtrip(tmp_path):
    model = IsolationForestAnomalyModel()
    model.fit(normal_residuals())
    path = tmp_path / "iforest_0.joblib"
    model.save(path)

    loaded = IsolationForestAnomalyModel.load(path)
    sample = normal_residuals(n=5, seed=2)
    assert np.allclose(model.predict_score(sample), loaded.predict_score(sample))
    assert loaded.features == list(PARTS)


# Tests below are for IForestScorer, the live.py plug-in piece.

def test_scorer_returns_none_when_no_model_file(tmp_path):
    scorer = IForestScorer(models_root=tmp_path)
    residuals = {part: 0.0 for part in PARTS}
    assert scorer.score("A-0", residuals, {}) is None


def test_scorer_returns_none_when_a_residual_is_missing(tmp_path):
    model = IsolationForestAnomalyModel()
    model.fit(normal_residuals())
    model.save(tmp_path / "iforest_0.joblib")

    scorer = IForestScorer(models_root=tmp_path)
    residuals = {part: 0.0 for part in PARTS}
    residuals[PARTS[0]] = None
    assert scorer.score(0, residuals, {}) is None


def test_scorer_scores_a_complete_reading(tmp_path):
    model = IsolationForestAnomalyModel()
    model.fit(normal_residuals())
    model.save(tmp_path / "iforest_0.joblib")

    scorer = IForestScorer(models_root=tmp_path)
    residuals = {part: 0.0 for part in PARTS}
    score = scorer.score(0, residuals, {})
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_scorer_caches_per_dataset(tmp_path):
    model = IsolationForestAnomalyModel()
    model.fit(normal_residuals())
    model.save(tmp_path / "iforest_0.joblib")

    scorer = IForestScorer(models_root=tmp_path)
    residuals = {part: 0.0 for part in PARTS}
    scorer.score(0, residuals, {})
    assert 0 in scorer._cache
    assert scorer._cache[0] is not None
