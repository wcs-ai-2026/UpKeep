"""Monitored parts and their sensor columns.

T-04 (pending team approval): the three proposed parts. T-05 (pending): the
part name is used everywhere (residual keys, model and noise file names).
Column names come from ``remap.py`` / ``SENSORS.md``.
"""
from typing import Dict, Tuple

PART_SENSORS: Dict[str, str] = {
    "gearbox_bearing": "gearbox_bearing_temp",
    "generator_bearing_1": "generator_bearing_1_temp",  # sensor_14, non-drive end
    "generator_bearing_2": "generator_bearing_2_temp",  # sensor_13, drive end
}
PARTS: Tuple[str, ...] = tuple(PART_SENSORS)
