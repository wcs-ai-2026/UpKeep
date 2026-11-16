"""Per-dataset health scoring: residuals -> normalised -> persistence -> colour.

Glues :mod:`upkeep.health.levels` to one dataset's normal noise levels (L-03)
and persistence state (L-12). One :class:`DatasetHealth` per ``dataset_id``.

Noise levels
------------
L-03 normalises each part's residual by that part's normal noise level, which
must come from residuals on healthy *training* rows of the same dataset. No
such numbers exist yet (no trained models). Proposed storage (T-16, pending):
``models_saved/<model>_noise_<dataset>.json`` holding ``{"<part>": <float>}``.
A missing file or entry gives ``None`` for that part, so the part is left out
of the score (L-07) and, if no part is usable, ``health`` is ``None`` (L-13).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

from upkeep.health.levels import Assessment, HealthConfig, PersistenceTracker, normalise_residuals
from upkeep.ids import dataset_number, format_dataset_id
from upkeep.parts import PARTS
from upkeep.paths import models_dir

log = logging.getLogger(__name__)


def noise_path(dataset_id, model_name: str = "lgbm", models_root: Optional[Path] = None) -> Path:
    root = Path(models_root) if models_root is not None else models_dir()
    return root / f"{model_name}_noise_{dataset_number(dataset_id)}.json"


def _read_noise_file(path: Path) -> Optional[dict]:
    if not path.is_file():
        log.warning("No noise levels at %s; parts cannot be normalised", path)
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("expected a JSON object")
    except Exception as exc:
        log.warning("Could not read %s: %s; parts cannot be normalised", path, exc)
        return None
    return raw


def load_noise_levels(dataset_id, model_name: Union[str, Mapping[str, str]] = "lgbm",
                      models_root: Optional[Path] = None,
                      parts: Tuple[str, ...] = PARTS) -> Dict[str, Optional[float]]:
    """Read one dataset's noise levels. Unusable or missing values become ``None``.

    Noise depends on the model that produced the residuals, so with a per-part
    model mapping each part's value is read from its own model's noise file.
    """
    if isinstance(model_name, str):
        part_models = {part: model_name for part in parts}
    else:
        part_models = {part: model_name.get(part, "lgbm") for part in parts}
    files: Dict[str, Optional[dict]] = {}
    levels: Dict[str, Optional[float]] = {}
    for part in parts:
        name = part_models[part]
        if name not in files:
            files[name] = _read_noise_file(noise_path(dataset_id, name, models_root))
        raw = files[name]
        levels[part] = None
        if raw is None:
            continue
        try:
            value = float(raw.get(part))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            levels[part] = value
    return levels


class DatasetHealth:
    """Noise levels plus persistence state for exactly one dataset."""

    def __init__(self, dataset_id, config: HealthConfig, noise_levels: Mapping[str, Optional[float]]):
        self.dataset_id = format_dataset_id(dataset_id)
        self.config = config
        self.noise_levels = dict(noise_levels)
        self.tracker = PersistenceTracker(config)

    def assess(self, timestamp, residuals: Mapping[str, Optional[float]]) -> Assessment:
        return self.tracker.update(timestamp, normalise_residuals(residuals, self.noise_levels))
