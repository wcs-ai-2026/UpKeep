"""Live scoring: one reading in, one health record out (CONTRACTS.md, Health).

``LiveScorer.score_reading(reading)`` joins the model store, per-dataset health
state, explanations and fault lookup, and optionally writes the record to
``data/live/health.jsonl``.

Rules applied here:

- **L-12 / T-02.** Every reading must carry ``dataset_id`` (``"A-<n>"``) and
  ``turbine_id`` (``"A-T<asset>"``), supplied by the caller. All
  state (persistence, recent history, fault info, models, noise levels) is kept
  per ``dataset_id``. Output includes ``dataset_id``.
- **L-08.** Readings with ``status_id != 0`` are not scored, return ``None`` and
  write nothing. Their state is left untouched.
- **L-07.** Missing models, features or sensor values give ``null`` residuals.
- **L-13 (pending).** If no part is usable, ``health`` and ``health_score`` are
  ``null`` with ``reasons: []`` and ``sentence: ""``. Never green.
- **L-03.** ``anomaly_score`` comes from an optional :class:`AnomalyScorer`
  (extension point for the Isolation Forest, T-09). It is reported but never
  passed to the health logic, so it cannot affect the colour. ``null`` when no
  scorer is configured.
- **L-05.** The explainer is only called for yellow or red. Green and unknown
  readings get ``reasons: []`` and ``sentence: ""``. Explainer output is checked
  at this boundary (:func:`validate_explanation`).
- **T-11.** ``fault_event`` / ``days_to_fault`` from :mod:`upkeep.stream.faults`.
  Display and evaluation only, never model inputs.
- **L-06.** ``latency_ms`` is part of every health record, as the contract
  requires, but it is written as ``null`` until latency logging is added
  (SCRUM-47). L-04 allows ``null`` for numbers that are not available; a
  made-up value such as ``0`` is never written.
- **L-04.** JSONL output is UTF-8, one object per line, ``null`` instead of
  NaN/inf, timestamps ``YYYY-MM-DDTHH:MM:SS``. :class:`HealthWriter` overwrites
  its file by default so a rerun cannot duplicate readings; ``append=True``
  appends on purpose.

Nothing here is validated: thresholds are untuned (T-08 pending) and no trained
models exist yet.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Mapping, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

from upkeep.health.levels import RED, YELLOW, HealthConfig
from upkeep.health.score import DatasetHealth, load_noise_levels
from upkeep.ids import format_dataset_id, format_turbine_id
from upkeep.parts import PARTS
from upkeep.paths import data_dir
from upkeep.stream.faults import FaultInfo, fault_info, load_events
from upkeep.stream.model_store import ModelStore

log = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
RECENT_DAYS = 7  # T-10 (pending): the explainer gets the last 7 days of scored readings
MAX_REASONS = 3
# Sentences that deny a problem. They must never accompany a yellow/red alert.
CONTRADICTORY_PHRASES = ("no unusual",)


def default_output_path() -> Path:
    return data_dir() / "live" / "health.jsonl"


# ---- JSON safety (L-04) ---------------------------------------------------

def format_timestamp(value) -> str:
    return pd.Timestamp(value).strftime(TIMESTAMP_FORMAT)


def to_json_safe(obj):
    """Convert to plain JSON types. NaN/inf -> None; timestamps -> L-04 format."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, _dt.datetime, np.datetime64)):
        return None if pd.isna(obj) else format_timestamp(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, Mapping):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_json_safe(v) for v in obj]
    raise TypeError(f"Cannot serialise {type(obj).__name__} to JSON")


def dumps_line(record: Mapping) -> str:
    """One JSONL line. ``allow_nan=False`` guarantees no NaN token slips through."""
    return json.dumps(to_json_safe(record), allow_nan=False, ensure_ascii=False)


class HealthWriter:
    """Writes records to a JSONL file, flushing each line for the dashboard.

    Overwrites the file by default, so writing the same readings again cannot
    duplicate records. Pass ``append=True`` only when appending is intended;
    nothing then stops the same ``(dataset_id, row_id)`` appearing twice.
    """

    def __init__(self, path: Optional[Path] = None, append: bool = False):
        self.path = Path(path) if path is not None else default_output_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.append = append
        self._file = open(self.path, "a" if append else "w", encoding="utf-8")
        self.lines = 0

    def write(self, record: Mapping) -> None:
        self._file.write(dumps_line(record) + "\n")
        self._file.flush()
        self.lines += 1

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---- helpers ---------------------------------------------------------------

def _number(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count(value) -> Optional[int]:
    number = _number(value)
    return None if number is None else int(round(number))


def _round(value, digits) -> Optional[float]:
    number = _number(value)
    return None if number is None else round(number, digits)


def stub_explain() -> Callable:
    """The interim ``upkeep.stubs.explain`` (unchanged; Karim's real one pending).

    Limitations: it only lists parts more than 5 degC above expected (raw degC,
    not normalised), always reports 72 hours ("3 days") whatever the real
    duration, and does not sort reasons. Use for wiring and demos only.
    """
    from upkeep.stubs import explain
    return explain


# ---- explanation validation (C2) ---------------------------------------------

def _clean_reason(reason) -> Optional[dict]:
    """Return a contract-shaped reason, or ``None`` if it cannot be made valid."""
    if not isinstance(reason, Mapping):
        return None
    part = reason.get("part")
    text = reason.get("text")
    hours = _number(reason.get("hours"))
    if not isinstance(part, str) or not part or not isinstance(text, str) or hours is None or hours < 0:
        return None
    raw_residual = reason.get("residual_c")
    residual = _number(raw_residual)
    if raw_residual is not None and residual is None:
        return None  # present but not a finite number
    return {"part": part, "residual_c": residual, "hours": float(hours), "text": text}


def is_contradictory(sentence: str) -> bool:
    lowered = sentence.strip().lower()
    return any(phrase in lowered for phrase in CONTRADICTORY_PHRASES)


def validate_explanation(result) -> Tuple[List[dict], str, List[str]]:
    """Check explainer output for an alert. Returns ``(reasons, sentence, issues)``.

    - ``result`` must be a mapping; otherwise nothing is attached.
    - Each reason needs a non-empty ``part``, a ``text`` and a finite,
      non-negative ``hours`` (converted to float). ``residual_c`` must be a
      finite number or absent. Invalid reasons are dropped.
    - Reasons are sorted by ``residual_c`` (largest first; reasons without one
      go last, keeping their order), then cut to three.
    - A sentence that denies a problem ("No unusual ...") is removed: an alert
      never carries it.
    """
    issues: List[str] = []
    if not isinstance(result, Mapping):
        return [], "", ["explainer did not return a mapping"]

    raw = result.get("reasons", [])
    if not isinstance(raw, (list, tuple)):
        issues.append("reasons is not a list")
        raw = []
    reasons = []
    for item in raw:
        cleaned = _clean_reason(item)
        if cleaned is None:
            issues.append("dropped an invalid reason")
        else:
            reasons.append(cleaned)
    reasons.sort(key=lambda r: -r["residual_c"] if r["residual_c"] is not None else math.inf)
    if len(reasons) > MAX_REASONS:
        issues.append(f"cut {len(reasons) - MAX_REASONS} reason(s) beyond {MAX_REASONS}")
        reasons = reasons[:MAX_REASONS]

    sentence = result.get("sentence", "")
    if not isinstance(sentence, str):
        issues.append("sentence is not a string")
        sentence = ""
    if is_contradictory(sentence):
        issues.append("removed a sentence that contradicts the alert")
        sentence = ""
    return reasons, sentence, issues


# ---- anomaly scorer extension point (T-09) --------------------------------------

class AnomalyScorer(Protocol):
    """Extension point for Varun's Isolation Forest (T-09, pending).

    Implementations return one value for the reading, or ``None`` if they cannot
    score it. T-09 proposes values in ``[0, 1]`` with higher = more unusual;
    values outside that range are reported as-is with a one-off warning. The
    result is only written to ``anomaly_score``; it never reaches the health
    logic (L-03).
    """

    def score(self, dataset_id: str, residuals: Mapping[str, Optional[float]],
              reading: Mapping) -> Optional[float]:
        ...


# ---- per-dataset state -------------------------------------------------------

@dataclass
class DatasetContext:
    dataset_id: str
    turbine_id: str
    health: DatasetHealth
    fault: FaultInfo
    recent: Deque[Tuple[pd.Timestamp, Dict[str, Optional[float]]]] = field(default_factory=deque)

    def remember(self, ts: pd.Timestamp, residuals: Mapping[str, Optional[float]], days: float) -> None:
        self.recent.append((ts, dict(residuals)))
        cutoff = ts - pd.Timedelta(days=days)
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()

    def recent_frame(self) -> pd.DataFrame:
        """T-10 proposal: ``timestamp`` plus one ``residual_<part>`` column per part."""
        rows = [
            {"timestamp": ts, **{f"residual_{p}": (np.nan if r.get(p) is None else r[p]) for p in PARTS}}
            for ts, r in self.recent
        ]
        return pd.DataFrame(rows, columns=["timestamp"] + [f"residual_{p}" for p in PARTS])


class LiveScorer:
    """Scores readings for one or more registered datasets, each with its own state.

    ``explain_fn`` is optional. Without one, alerts carry ``reasons: []`` and
    ``sentence: ""``: no explanation is better than an invented one. Pass
    :func:`stub_explain` to use the interim stub (see its limitations).
    """

    def __init__(
        self,
        config: HealthConfig,
        *,
        model_store=None,
        model_name: Union[str, Mapping[str, str]] = "lgbm",
        models_root: Optional[Path] = None,
        events: Optional[pd.DataFrame] = None,
        explain_fn: Optional[Callable] = None,
        anomaly_scorer: Optional[AnomalyScorer] = None,
        writer: Optional[HealthWriter] = None,
        recent_days: float = RECENT_DAYS,
    ):
        self.config = config
        self.models_root = models_root
        self.store = model_store if model_store is not None else ModelStore(model_name, models_root)
        self.model_name = getattr(self.store, "part_models", model_name)
        self.events = events if events is not None else load_events()
        self.explain_fn = explain_fn
        self.anomaly_scorer = anomaly_scorer
        self.writer = writer
        self.recent_days = recent_days
        self._contexts: Dict[str, DatasetContext] = {}
        self._warned: set = set()
        self.explanation_issues: Dict[str, int] = {}

    @property
    def datasets(self) -> Tuple[str, ...]:
        return tuple(self._contexts)

    def _warn_once(self, key: str, message: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message, *args)

    def context(self, dataset_id) -> DatasetContext:
        key = format_dataset_id(dataset_id)
        try:
            return self._contexts[key]
        except KeyError:
            raise KeyError(f"Dataset {key} is not registered; call register_dataset() first") from None

    def register_dataset(self, dataset_id, readings: pd.DataFrame,
                         noise_levels: Optional[Mapping[str, Optional[float]]] = None) -> DatasetContext:
        """Set up state for one dataset.

        ``readings`` must be the dataset's FULL readings (all splits) so the
        fault-start reading can be found (T-11).
        """
        key = format_dataset_id(dataset_id)
        if key in self._contexts:
            raise ValueError(f"Dataset {key} is already registered")
        turbines = pd.unique(readings["turbine_id"])
        if len(turbines) != 1:
            raise ValueError(f"Dataset {key} must contain exactly one turbine, found {list(turbines)}")
        if noise_levels is None:
            noise_levels = load_noise_levels(key, self.model_name, self.models_root)
        if hasattr(self.store, "status"):
            status = self.store.status(key)  # pre-load models so the first reading doesn't pay the load cost
            if not any(s == "loaded" for s in status.values()):
                log.warning("Dataset %s has no usable models: health will be null (L-13)", key)
        context = DatasetContext(
            dataset_id=key,
            turbine_id=format_turbine_id(turbines[0]),
            health=DatasetHealth(key, self.config, noise_levels),
            fault=fault_info(key, readings, self.events),
        )
        self._contexts[key] = context
        return context

    def _explain(self, context: DatasetContext, residuals, gusts) -> Tuple[List[dict], str]:
        if self.explain_fn is None:
            return [], ""
        try:
            result = self.explain_fn(context.recent_frame(), dict(residuals), gusts)
        except Exception as exc:
            self._warn_once("explain-raised", "explain() failed (%s); reasons left empty", exc)
            return [], ""
        reasons, sentence, issues = validate_explanation(result)
        for issue in issues:
            self.explanation_issues[issue] = self.explanation_issues.get(issue, 0) + 1
            self._warn_once(f"explain:{issue}", "Explanation check: %s", issue)
        return reasons, sentence

    def _anomaly(self, dataset_id: str, residuals, reading) -> Optional[float]:
        if self.anomaly_scorer is None:
            return None
        try:
            value = _number(self.anomaly_scorer.score(dataset_id, dict(residuals), reading))
        except Exception as exc:
            self._warn_once("anomaly-raised", "Anomaly scorer failed (%s); anomaly_score is null", exc)
            return None
        if value is not None and not 0.0 <= value <= 1.0:
            self._warn_once("anomaly-range", "Anomaly score %s outside [0, 1] (T-09 proposal)", value)
        return value

    def score_reading(self, reading: Mapping) -> Optional[dict]:
        """Score one reading. Returns the health record, or ``None`` if skipped (L-08)."""
        status = _number(reading.get("status_id"))
        if status is None:
            raise ValueError("reading has no usable status_id")
        if status != 0:
            return None  # L-08: not scored, no line written

        context = self.context(reading.get("dataset_id"))
        turbine = format_turbine_id(reading.get("turbine_id"))
        if turbine != context.turbine_id:
            raise ValueError(f"Reading turbine {turbine} does not match {context.dataset_id} ({context.turbine_id})")
        raw_ts = reading.get("timestamp")
        ts = pd.Timestamp(raw_ts) if raw_ts is not None else pd.NaT
        if pd.isna(ts):
            raise ValueError(f"Reading {context.dataset_id} row {reading.get('row_id')} has no valid timestamp")

        residuals = self.store.residuals(context.dataset_id, reading)
        assessment = context.health.assess(ts, residuals)
        context.remember(ts, residuals, self.recent_days)

        gusts = {
            "gusts_last_24h": _count(reading.get("gusts_last_24h")),
            "gusts_last_7d": _count(reading.get("gusts_last_7d")),
        }
        if assessment.health in (YELLOW, RED):
            reasons, sentence = self._explain(context, residuals, gusts)
        else:
            reasons, sentence = [], ""

        days = context.fault.days_to_fault(ts)
        record = {
            "row_id": int(reading.get("row_id")),
            "dataset_id": context.dataset_id,
            "turbine_id": context.turbine_id,
            "timestamp": format_timestamp(ts),
            "health": assessment.health,
            "health_score": _round(assessment.health_score, 3),
            "residuals": {part: _round(residuals.get(part), 3) for part in PARTS},
            "anomaly_score": _round(self._anomaly(context.dataset_id, residuals, reading), 4),
            "gusts_last_7d": gusts["gusts_last_7d"],
            "reasons": reasons,
            "sentence": sentence,
            "fault_event": context.fault.fault_event,
            "days_to_fault": _round(days, 4),
        }
        record["latency_ms"] = None  # contract field; measured from SCRUM-47 (latency logging)
        if self.writer is not None:
            self.writer.write(record)
        return record


# ---- module-level entry point (contract signature) --------------------------

_default_scorer: Optional[LiveScorer] = None


def set_default_scorer(scorer: Optional[LiveScorer]) -> None:
    global _default_scorer
    _default_scorer = scorer


def get_default_scorer() -> Optional[LiveScorer]:
    return _default_scorer


def score_reading(reading: Mapping) -> Optional[dict]:
    """Contract entry point: ``score_reading(reading) -> dict``.

    Uses the scorer set with :func:`set_default_scorer` (the caller sets
    one). ``upkeep.stubs.score_reading`` is separate and unchanged (L-10).
    """
    if _default_scorer is None:
        raise RuntimeError(
            "No live scorer configured; call set_default_scorer() with a LiveScorer "
            "and register each dataset before scoring"
        )
    return _default_scorer.score_reading(reading)
