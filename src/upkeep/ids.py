"""Identifier helpers for turbines and datasets.

Formats follow T-02 (Lead approved 2026-11-15, team to confirm in PR):

- ``turbine_id = "A-T<asset>"``  e.g. asset 11 -> ``"A-T11"``
- ``dataset_id = "A-<n>"``       e.g. ``25.parquet`` -> ``"A-25"``

Under L-12 a scored reading is identified by ``(dataset_id, row_id)``.
The data files keep their integer ids; these helpers only convert.
"""
from __future__ import annotations

import re

FARM_PREFIX = "A"

_DATASET_RE = re.compile(r"^(?:A-)?(\d+)$")
_TURBINE_RE = re.compile(r"^(?:A-T)?(\d+)$")


def _as_number(value, pattern: re.Pattern, kind: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {kind}: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"Invalid {kind}: {value!r}")
        return value
    # numpy integers and similar
    if hasattr(value, "__index__") and not isinstance(value, str):
        return _as_number(int(value), pattern, kind)
    match = pattern.match(str(value).strip())
    if not match:
        raise ValueError(f"Invalid {kind}: {value!r}")
    return int(match.group(1))


def dataset_number(dataset_id) -> int:
    """Return the integer dataset number from ``"A-25"``, ``"25"`` or ``25``."""
    return _as_number(dataset_id, _DATASET_RE, "dataset_id")


def turbine_number(turbine_id) -> int:
    """Return the integer asset number from ``"A-T11"``, ``"11"`` or ``11``."""
    return _as_number(turbine_id, _TURBINE_RE, "turbine_id")


def format_dataset_id(dataset_id) -> str:
    """``25`` / ``"25"`` / ``"A-25"`` -> ``"A-25"``."""
    return f"{FARM_PREFIX}-{dataset_number(dataset_id)}"


def format_turbine_id(turbine_id) -> str:
    """``11`` / ``"11"`` / ``"A-T11"`` -> ``"A-T11"``."""
    return f"{FARM_PREFIX}-T{turbine_number(turbine_id)}"
