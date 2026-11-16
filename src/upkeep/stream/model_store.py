"""Load saved normal-behaviour models and turn readings into residuals.

- Models are found at ``models_saved/<model>_<part>_<dataset>.joblib``
  (T-07, pending team approval), where ``<dataset>`` is the integer dataset
  number, e.g. ``models_saved/lgbm_gearbox_bearing_25.joblib``.
- The model type can differ per part: pass ``model_name`` as one name for all
  parts (``"lgbm"``) or as a mapping (``{"gearbox_bearing": "lgbm",
  "generator_bearing_1": "linear", ...}``). Parts left out of a mapping use
  ``DEFAULT_MODEL``.
- Under T-05, existing files named after the sensor column
  (``lgbm_gearbox_bearing_temp_25.joblib``) are still accepted. If both
  exist, the canonical part name wins.
- Models are loaded with the owner's own ``load()`` classmethod
  (``upkeep/models/``, Varun). Nothing is trained here.
- Each model is fed the feature list it saved (``model.features``), read from
  the reading as-is. The seed files already contain the precomputed features,
  so no feature engineering is repeated here.

Missing data never produces a made-up number (L-07). A part's residual is
``None`` when:

- no model file exists, or the file fails to load;
- a feature the model needs is missing or NaN in the reading;
- the part's actual sensor value is missing or NaN;
- the prediction is not a finite number, or prediction raises.

Per L-12, models are looked up per dataset and never shared across datasets.
"""
from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import pandas as pd

from upkeep.ids import dataset_number
from upkeep.parts import PART_SENSORS, PARTS
from upkeep.paths import models_dir

log = logging.getLogger(__name__)

# Model name used in the file name -> (module, class) in upkeep/models/.
MODEL_CLASSES: Dict[str, Tuple[str, str]] = {
    "lgbm": ("upkeep.models.lgbm_nbm", "LightGBMNormalModel"),
    "linear": ("upkeep.models.linear_nbm", "LinearNormalModel"),
    "baseline": ("upkeep.models.baseline_rule", "FixedLimitBaseline"),
}

DEFAULT_MODEL = "lgbm"


def resolve_part_models(model_name: Union[str, Mapping[str, str]],
                        parts: Tuple[str, ...] = PARTS) -> Dict[str, str]:
    """Turn ``"lgbm"`` or ``{part: name}`` into a full ``{part: name}`` mapping."""
    if isinstance(model_name, str):
        mapping = {part: model_name for part in parts}
    else:
        extra = set(model_name) - set(PART_SENSORS)
        if extra:
            raise ValueError(f"Unknown parts in model mapping: {sorted(extra)}")
        mapping = {part: model_name.get(part, DEFAULT_MODEL) for part in parts}
    bad = {name for name in mapping.values() if name not in MODEL_CLASSES}
    if bad:
        raise ValueError(f"Unknown model(s) {sorted(bad)}; expected one of {sorted(MODEL_CLASSES)}")
    return mapping


# General Rule 7: fault dates and status codes are never model inputs.
FORBIDDEN_FEATURES = frozenset({"status_id", "split", "fault_event", "days_to_fault"})

LOADED = "loaded"
MISSING = "missing"
ERROR = "error"


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ModelStore:
    """Lazily loads and caches one model per (dataset, part).

    Lookups that fail are cached too, so a missing model costs one disk check
    per (dataset, part), not one per reading.
    """

    def __init__(self, model_name: Union[str, Mapping[str, str]] = DEFAULT_MODEL,
                 models_root: Optional[Path] = None, parts: Tuple[str, ...] = PARTS):
        unknown = set(parts) - set(PART_SENSORS)
        if unknown:
            raise ValueError(f"Unknown parts: {sorted(unknown)}")
        self.parts = tuple(parts)
        self.part_models: Dict[str, str] = resolve_part_models(model_name, self.parts)
        self.models_root = Path(models_root) if models_root is not None else models_dir()
        self._cache: Dict[Tuple[int, str], object] = {}
        self._status: Dict[Tuple[int, str], str] = {}
        self._warned: set = set()

    @property
    def model_name(self) -> Union[str, Dict[str, str]]:
        """One name if every part uses the same model type, else the mapping."""
        names = set(self.part_models.values())
        return names.pop() if len(names) == 1 else dict(self.part_models)

    # ---- locating and loading -------------------------------------------

    def candidate_paths(self, dataset_id, part: str):
        n = dataset_number(dataset_id)
        model = self.part_models[part]
        names = [f"{model}_{part}_{n}.joblib"]
        legacy = f"{model}_{PART_SENSORS[part]}_{n}.joblib"  # T-05
        if legacy not in names:
            names.append(legacy)
        return [self.models_root / name for name in names]

    def model_path(self, dataset_id, part: str) -> Optional[Path]:
        for path in self.candidate_paths(dataset_id, part):
            if path.is_file():
                return path
        return None

    def _model_class(self, part: str):
        module_name, class_name = MODEL_CLASSES[self.part_models[part]]
        return getattr(importlib.import_module(module_name), class_name)

    def get(self, dataset_id, part: str):
        """Return the loaded model, or ``None`` if unavailable."""
        if part not in PART_SENSORS:
            raise ValueError(f"Unknown part {part!r}")
        key = (dataset_number(dataset_id), part)
        if key in self._status:
            return self._cache.get(key)

        path = self.model_path(*key)
        if path is None:
            self._status[key] = MISSING
            log.warning("No %s model for dataset %s, part %s; residuals will be null",
                        self.part_models[part], key[0], part)
            return None
        try:
            model = self._model_class(part).load(path)
            features = getattr(model, "features", None) or []
            bad = (set(features) & FORBIDDEN_FEATURES) | ({PART_SENSORS[part]} & set(features))
            if bad:
                raise ValueError(f"model uses forbidden input(s) {sorted(bad)}")
        except Exception as exc:  # corrupt file, wrong class, missing library ...
            self._status[key] = f"{ERROR}: {exc}"
            log.warning("Could not load %s: %s; residuals will be null", path, exc)
            return None
        self._cache[key] = model
        self._status[key] = LOADED
        return model

    def status(self, dataset_id) -> Dict[str, str]:
        """``{part: "loaded" | "missing" | "error: ..."}`` for one dataset."""
        n = dataset_number(dataset_id)
        for part in self.parts:
            self.get(n, part)
        return {part: self._status[(n, part)] for part in self.parts}

    # ---- prediction ----------------------------------------------------

    def expected(self, dataset_id, part: str, reading: Mapping) -> Optional[float]:
        """Expected temperature for one reading, or ``None`` (never a guess)."""
        model = self.get(dataset_id, part)
        if model is None:
            return None
        features = list(getattr(model, "features", None) or [])
        row = {}
        for name in features:
            value = reading.get(name) if hasattr(reading, "get") else None
            if value is None or pd.isna(value):
                return None  # L-07: a feature the model needs is missing
            row[name] = value
        X = pd.DataFrame([row], columns=features) if features else pd.DataFrame(index=[0])
        try:
            prediction = model.predict(X)
            value = _finite(prediction[0])
        except Exception as exc:
            key = (dataset_number(dataset_id), part)
            if key not in self._warned:
                self._warned.add(key)
                log.warning("Prediction failed for dataset %s, part %s: %s", key[0], part, exc)
            return None
        return value

    def residuals(self, dataset_id, reading: Mapping) -> Dict[str, Optional[float]]:
        """``{part: actual - expected}`` with ``None`` wherever either is missing."""
        out: Dict[str, Optional[float]] = {}
        for part in self.parts:
            actual = _finite(reading.get(PART_SENSORS[part]) if hasattr(reading, "get") else None)
            if actual is None:
                out[part] = None
                continue
            expected = self.expected(dataset_id, part, reading)
            out[part] = None if expected is None else actual - expected
        return out
