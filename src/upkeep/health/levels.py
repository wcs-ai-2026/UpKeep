"""Health score and green/yellow/red levels.

Implements the Lead decisions in ``DECISIONS.md``:

- **L-03, worst part wins.** Each part's residual is divided by that part's
  own normal noise level (``z = residual / noise``). ``anomaly_score`` is *not*
  an input here, so it cannot influence the colour.
- **L-02 / L-14, per-part persistence.** Each part must sustain its severity on
  its own. A part's *sustained* severity is its lowest ``z`` over the last
  ``persistence_hours``. After persistence is applied, the part with the highest
  sustained severity sets the turbine's colour and score (L-03). Two parts that
  take turns being high therefore never add up to an alert.
- **L-01.** ``health_score`` runs from 0 to 1, where 1 means fully healthy.
- **L-07 / L-13.** A part with a missing residual or an unusable noise level is
  left out of that reading. When no part is usable, ``health`` and
  ``health_score`` are ``None``: an unscorable turbine is never called green.
- **L-12.** Persistence state belongs to one dataset. Keep one
  :class:`PersistenceTracker` per ``dataset_id``.

Two kinds of number live in :class:`HealthConfig` and must not be confused:

- **Detection thresholds** (``yellow_z``, ``red_z``, ``persistence_hours``,
  ``max_gap_minutes``) decide the colour. They are NOT tuned: T-08 (validation
  window) is not approved, so they have no defaults.
- **Display anchors** (``score_at_yellow = 0.7``, ``score_at_red = 0.4``) only
  shape the ``health_score`` fill bar so its bands line up with the colours.
  They never decide a colour. Provisional (L-15).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Mapping, Optional, Tuple

import pandas as pd

GREEN = "green"
YELLOW = "yellow"
RED = "red"

# Provisional display anchors (L-15). Not detection thresholds.
PROVISIONAL_SCORE_AT_YELLOW = 0.7
PROVISIONAL_SCORE_AT_RED = 0.4


@dataclass(frozen=True)
class HealthConfig:
    """Detection thresholds (untuned, required) and display anchors (provisional).

    Detection thresholds:
      ``yellow_z`` / ``red_z``: sustained normalised residual (in units of the
      part's normal noise level) at which the turbine turns yellow / red.
      ``persistence_hours``: how long a part must stay at or above a threshold.
      ``max_gap_minutes``: a gap between a part's usable readings longer than
      this restarts that part's persistence window.

    Display anchors (do not affect colour):
      ``score_at_yellow`` / ``score_at_red``: ``health_score`` at the yellow /
      red thresholds. ``zero_score_z``: ``z`` where the score reaches 0
      (defaults to ``2 * red_z``).
    """

    yellow_z: float
    red_z: float
    persistence_hours: float
    max_gap_minutes: float
    score_at_yellow: float = PROVISIONAL_SCORE_AT_YELLOW
    score_at_red: float = PROVISIONAL_SCORE_AT_RED
    zero_score_z: Optional[float] = None

    def __post_init__(self):
        if not (0 < self.yellow_z < self.red_z):
            raise ValueError("Need 0 < yellow_z < red_z")
        if self.persistence_hours <= 0:
            raise ValueError("persistence_hours must be positive")
        if self.max_gap_minutes <= 0:
            raise ValueError("max_gap_minutes must be positive")
        if self.max_gap_minutes >= self.persistence_hours * 60:
            # Otherwise a single reading after a permitted gap could fill the
            # whole persistence window on its own.
            raise ValueError("max_gap_minutes must be strictly smaller than the persistence window")
        if not (0 < self.score_at_red < self.score_at_yellow < 1):
            raise ValueError("Need 0 < score_at_red < score_at_yellow < 1")
        if self.zero_score_z is not None and self.zero_score_z <= self.red_z:
            raise ValueError("zero_score_z must be greater than red_z")

    @property
    def zero_z(self) -> float:
        return self.zero_score_z if self.zero_score_z is not None else 2 * self.red_z


# Development placeholder ONLY. Not tuned, not validated (T-08 pending).
# The 3-hour persistence matches the rule used in the L-03 comparison; the
# z thresholds are arbitrary round numbers.
UNTUNED_PLACEHOLDER_CONFIG = HealthConfig(
    yellow_z=3.0,
    red_z=5.0,
    persistence_hours=3.0,
    max_gap_minutes=60.0,
)


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalise_residuals(
    residuals: Mapping[str, Optional[float]],
    noise_levels: Mapping[str, Optional[float]],
) -> dict:
    """Return ``{part: residual / noise}``, or ``None`` for unusable parts (L-07)."""
    normalised = {}
    for part, residual in residuals.items():
        r = _finite(residual)
        s = _finite(noise_levels.get(part))
        normalised[part] = r / s if (r is not None and s is not None and s > 0) else None
    return normalised


def worst_part(values: Mapping[str, Optional[float]]) -> Tuple[Optional[str], Optional[float]]:
    """Return ``(part, value)`` for the highest value; ``(None, None)`` if none usable.

    ``None`` entries are skipped. Ties go to the first part in iteration order.
    """
    best_part, best = None, None
    for part, value in values.items():
        value = _finite(value)
        if value is None:
            continue
        if best is None or value > best:
            best_part, best = part, value
    return best_part, best


def health_level(sustained_z: float, config: HealthConfig) -> str:
    """Colour for a sustained severity (detection thresholds only)."""
    if sustained_z >= config.red_z:
        return RED
    if sustained_z >= config.yellow_z:
        return YELLOW
    return GREEN


def health_score(sustained_z: float, config: HealthConfig) -> float:
    """Map sustained severity to ``[0, 1]``, 1 = fully healthy (L-01).

    Piecewise linear and non-increasing: ``1.0`` at ``z <= 0``,
    ``score_at_yellow`` at ``yellow_z``, ``score_at_red`` at ``red_z``, ``0.0``
    at ``zero_z``. The anchors are display values only (L-15).
    """
    z = sustained_z
    y, r, zero = config.yellow_z, config.red_z, config.zero_z
    sy, sr = config.score_at_yellow, config.score_at_red
    if z <= 0:
        return 1.0
    if z <= y:
        return 1.0 - (1.0 - sy) * z / y
    if z <= r:
        return sy - (sy - sr) * (z - y) / (r - y)
    if z < zero:
        return sr - sr * (z - r) / (zero - r)
    return 0.0


@dataclass(frozen=True)
class Assessment:
    """Result of one :meth:`PersistenceTracker.update` call.

    ``health`` / ``health_score`` / ``sustained_z`` / ``worst_part`` are
    ``None`` when no part was usable at this reading (``scored`` is False).
    ``worst_part`` is the part with the highest *sustained* severity, i.e. the
    part that set the colour. ``normalised`` holds this reading's raw ``z``
    values and ``sustained`` each usable part's sustained ``z``.
    """

    health: Optional[str]
    health_score: Optional[float]
    sustained_z: Optional[float]
    worst_part: Optional[str]
    normalised: dict = field(default_factory=dict)
    sustained: dict = field(default_factory=dict)
    scored: bool = True


class _PartWindow:
    """Persistence window for one part of one dataset."""

    __slots__ = ("values", "run_start")

    def __init__(self):
        self.values: Deque[Tuple[pd.Timestamp, float]] = deque()
        self.run_start: Optional[pd.Timestamp] = None

    def update(self, ts: pd.Timestamp, z: float, window: pd.Timedelta, max_gap: pd.Timedelta) -> float:
        if self.values and ts - self.values[-1][0] > max_gap:
            self.values.clear()
        if not self.values:
            self.run_start = ts
        self.values.append((ts, z))
        cutoff = ts - window
        # Keep the most recent reading at or before the cutoff: it is the
        # part's last known state at the start of the window. Dropping it would
        # let a permitted gap that straddles the cutoff shorten persistence.
        while len(self.values) > 1 and self.values[1][0] <= cutoff:
            self.values.popleft()
        sustained = min(v for _, v in self.values)
        if self.run_start > cutoff:
            # History does not yet cover the whole window: the unseen time
            # counts as normal, so the colour cannot change yet.
            sustained = min(sustained, 0.0)
        return sustained


class PersistenceTracker:
    """Per-dataset, per-part persistence state (L-02, L-12, L-14).

    Feed readings of ONE dataset in strictly increasing time order. Each part
    keeps its own window. A part's sustained severity is the minimum of its
    ``z`` over the last ``persistence_hours``, including the most recent reading
    at or before the window start (the part's state when the window began), so
    a gap can never shorten the required duration; if its continuous history does
    not yet cover that window (start of replay, after the part was unusable, or
    after a gap longer than ``max_gap_minutes``), the uncovered time counts as
    ``z = 0``. A part that is unusable at this reading is left out of this
    reading's score (L-07) and its window is not extended.
    """

    def __init__(self, config: HealthConfig):
        self.config = config
        self._parts: Dict[str, _PartWindow] = {}
        self._last_seen: Optional[pd.Timestamp] = None
        self._last: Optional[Assessment] = None
        self._window = pd.Timedelta(hours=config.persistence_hours)
        self._max_gap = pd.Timedelta(minutes=config.max_gap_minutes)

    @property
    def last(self) -> Optional[Assessment]:
        return self._last

    def reset(self) -> None:
        self._parts.clear()
        self._last_seen = None
        self._last = None

    def update(self, timestamp, normalised: Mapping[str, Optional[float]]) -> Assessment:
        ts = pd.Timestamp(timestamp)
        if self._last_seen is not None and ts <= self._last_seen:
            raise ValueError(
                f"Readings must arrive in strictly increasing time order: {ts} after {self._last_seen}"
            )
        self._last_seen = ts
        normalised = {part: _finite(z) for part, z in normalised.items()}

        sustained = {}
        for part, z in normalised.items():
            if z is None:
                continue
            window = self._parts.get(part)
            if window is None:
                window = self._parts[part] = _PartWindow()
            sustained[part] = window.update(ts, z, self._window, self._max_gap)

        part, turbine_z = worst_part(sustained)
        if turbine_z is None:
            assessment = Assessment(None, None, None, None, normalised, {}, scored=False)
        else:
            assessment = Assessment(
                health=health_level(turbine_z, self.config),
                health_score=health_score(turbine_z, self.config),
                sustained_z=turbine_z,
                worst_part=part,
                normalised=normalised,
                sustained=sustained,
                scored=True,
            )
        self._last = assessment
        return assessment
