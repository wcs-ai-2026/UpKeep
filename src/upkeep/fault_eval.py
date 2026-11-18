

import json
from pathlib import Path

import pandas as pd

from upkeep.health.levels import GREEN, RED, YELLOW
from upkeep.ids import dataset_number, format_turbine_id
from upkeep.stream.faults import ANOMALY_LABEL, fault_info


def score_window(health_path: Path, event, eligible: pd.Series) -> dict:
    
    result = {
        "eligible_readings": len(eligible),
        "scored_readings": 0,
        "warning_detected": None,
        "first_warning_timestamp": None,
        "first_warning_health": None,
        "lead_time_hours": None,
    }
    if eligible.empty:
        return {**result, "status": "no_held_out_readings"}
    if not health_path.is_file():
        return {**result, "status": "missing_predictions"}

    scored = {}
    with health_path.open(encoding="utf-8") as health_file:
        for line in health_file:
            if not line.strip():
                continue
            prediction = json.loads(line)
            row_id = prediction["row_id"]
            if (
                row_id in eligible.index
                and prediction["dataset_id"] == event.dataset_id
                and prediction["turbine_id"] == event.turbine_id
                and pd.Timestamp(prediction["timestamp"]) == eligible.loc[row_id]
            ):
                health = prediction["health"]
                if health in (GREEN, YELLOW, RED):
                    if row_id in scored and scored[row_id] != health:
                        raise ValueError(f"{health_path} has conflicting health predictions for row {row_id}")
                    scored[row_id] = health

    result["scored_readings"] = len(scored)
    status = "complete" if len(scored) == len(eligible) else "partial_predictions"
    if not scored:
        status = "no_scorable_predictions"

    warnings = [row_id for row_id, health in scored.items() if health in (YELLOW, RED)]
    if warnings:
        first_row = eligible.loc[warnings].idxmin()
        result.update(
            warning_detected=True,
            first_warning_timestamp=eligible.loc[first_row],
            first_warning_health=scored[first_row],
        )
    elif status == "complete":
        result["warning_detected"] = False
    return {**result, "status": status}


def evaluate_faults(data_root: Path, health_dir: Path, window: pd.Timedelta) -> pd.DataFrame:
    """Return one report row per fault, including events with unavailable predictions."""
    if pd.isna(window) or window <= pd.Timedelta(0):
        raise ValueError("The warning window must be positive")

    events = pd.read_csv(data_root / "fault_events.csv", parse_dates=["fault_timestamp"])
    events["event_id"] = events["dataset_id"].map(dataset_number)
    lookup = events.assign(event_label=ANOMALY_LABEL).set_index("event_id")
    if not lookup.index.is_unique:
        raise ValueError("The fault event table contains duplicate datasets")

    results = []
    for event in events.itertuples(index=False):
        readings = pd.read_csv(
            data_root / "cln" / f"{event.event_id}_cln.csv",
            sep=";",
            usecols=["row_id", "turbine_id", "timestamp", "split", "status_id"],
            parse_dates=["timestamp"],
        )
        turbines = {format_turbine_id(turbine) for turbine in readings["turbine_id"].unique()}
        if turbines != {event.turbine_id}:
            raise ValueError(f"{event.dataset_id} readings do not match fault turbine {event.turbine_id}")
        if not readings["row_id"].is_unique:
            raise ValueError(f"{event.dataset_id} contains duplicate reading row IDs")

        fault_start = fault_info(event.dataset_id, readings, lookup).fault_start
        held_out = readings.loc[readings["split"].eq("prediction") & readings["status_id"].eq(0)]
        eligible = held_out.set_index("row_id")["timestamp"]
        if fault_start is None:
            eligible = eligible.iloc[:0]
        else:
            eligible = eligible[(eligible >= fault_start - window) & (eligible < fault_start)]
        result = score_window(health_dir / f"health_{event.dataset_id}.jsonl", event, eligible)
        if fault_start is None and not held_out.empty:
            result["status"] = "unknown_fault_start"
        if result["warning_detected"]:
            lead = fault_start - result["first_warning_timestamp"]
            result["lead_time_hours"] = round(lead / pd.Timedelta(hours=1), 3)

        results.append({
            "dataset_id": event.dataset_id,
            "turbine_id": event.turbine_id,
            "fault_type": event.fault_type,
            "fault_timestamp": event.fault_timestamp,
            "replay_fault_timestamp": fault_start,
            "window_days": window / pd.Timedelta(days=1),
            **result,
        })
    return pd.DataFrame(results)
