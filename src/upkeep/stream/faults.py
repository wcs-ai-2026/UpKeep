"""Fault-event lookup for the health output (T-11; Lead approved 2026-11-15).

Rules, as written in ``CONTRACTS.md`` (Health section):

- ``fault_event = "A-<n>"`` for the dataset being replayed, set only when its
  event in ``event_info.csv`` is labelled ``anomaly``. Otherwise ``None``.
- The fault-start reading is found with ``event_start_id`` (a **row id**),
  never with the event timestamps, which do not match the data.
- If that ``row_id`` was removed during cleaning, the first later reading in the
  dataset is used. If the dataset has no reading at or after
  ``event_start_id``, the fault start is unknown: ``days_to_fault`` is ``None``
  while ``fault_event`` is still set.
- ``days_to_fault = (fault-start timestamp - reading timestamp)`` in days:
  positive before, ``0`` at the fault-start reading, negative after.

Per L-12 everything here is per dataset. Pass the dataset's *full* readings
(training and test rows) so the fault start can be found even when only part
of the dataset is replayed.

These values are for display and evaluation only. They must never be model
inputs (General Rule 7).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from upkeep.ids import dataset_number, format_dataset_id
from upkeep.paths import event_info_path

ANOMALY_LABEL = "anomaly"
_SECONDS_PER_DAY = 86_400.0


def load_events(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Read ``event_info.csv`` (``;``-separated), indexed by integer ``event_id``."""
    path = Path(path) if path is not None else event_info_path()
    events = pd.read_csv(path, sep=";")
    required = {"event_id", "event_label", "event_start_id"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    events["event_id"] = events["event_id"].astype(int)
    if events["event_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate event_id values")
    return events.set_index("event_id")


@dataclass(frozen=True)
class FaultInfo:
    """Fault facts for one dataset."""

    dataset_id: str
    fault_event: Optional[str]
    fault_start_row_id: Optional[int]
    fault_start: Optional[pd.Timestamp]

    def days_to_fault(self, timestamp) -> Optional[float]:
        return days_to_fault(timestamp, self.fault_start)


def fault_info(dataset_id, readings: pd.DataFrame, events: pd.DataFrame) -> FaultInfo:
    """Work out ``fault_event`` and the fault-start reading for one dataset.

    ``readings`` must be this dataset's readings with ``row_id`` and
    ``timestamp`` columns. Raises ``KeyError`` if the dataset has no row in
    ``events``: that is a configuration error, not a "no fault" case.
    """
    number = dataset_number(dataset_id)
    if number not in events.index:
        raise KeyError(f"Dataset {format_dataset_id(number)} has no row in event_info.csv")
    event = events.loc[number]
    label = str(event["event_label"]).strip().lower()
    if label != ANOMALY_LABEL:
        return FaultInfo(format_dataset_id(number), None, None, None)

    missing = {"row_id", "timestamp"} - set(readings.columns)
    if missing:
        raise ValueError(f"readings are missing columns: {sorted(missing)}")

    start_id = int(event["event_start_id"])
    at_or_after = readings.loc[readings["row_id"] >= start_id, ["row_id", "timestamp"]]
    if at_or_after.empty:
        return FaultInfo(format_dataset_id(number), format_dataset_id(number), None, None)

    first = at_or_after.loc[at_or_after["row_id"].idxmin()]
    return FaultInfo(
        dataset_id=format_dataset_id(number),
        fault_event=format_dataset_id(number),
        fault_start_row_id=int(first["row_id"]),
        fault_start=pd.Timestamp(first["timestamp"]),
    )


def days_to_fault(timestamp, fault_start: Optional[pd.Timestamp]) -> Optional[float]:
    """Days from ``timestamp`` to ``fault_start``; ``None`` if the start is unknown."""
    if fault_start is None or pd.isna(fault_start):
        return None
    delta = pd.Timestamp(fault_start) - pd.Timestamp(timestamp)
    return delta.total_seconds() / _SECONDS_PER_DAY
