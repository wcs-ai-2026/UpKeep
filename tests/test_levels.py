import math

import pandas as pd
import pytest

from upkeep.health.levels import (
    GREEN, RED, YELLOW, HealthConfig, PersistenceTracker, health_level,
    health_score, normalise_residuals, worst_part,
)

# Test-only thresholds. Not tuned values.
CFG = HealthConfig(yellow_z=3.0, red_z=5.0, persistence_hours=1.0, max_gap_minutes=30.0)
T0 = pd.Timestamp("2023-01-01 00:00:00")


def at(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def feed(tracker, zs, start=0, step=10, part="gearbox_bearing"):
    out = []
    for i, z in enumerate(zs):
        out.append(tracker.update(at(start + i * step), {part: z}))
    return out


# ---- normalisation and worst part (L-03, L-07) ----------------------------

def test_normalise_divides_by_each_parts_own_noise():
    z = normalise_residuals({"a": 6.0, "b": 6.0}, {"a": 2.0, "b": 3.0})
    assert z == {"a": 3.0, "b": 2.0}


@pytest.mark.parametrize("residual,noise", [
    (None, 1.0), (float("nan"), 1.0), (1.0, None), (1.0, 0.0), (1.0, -1.0), (1.0, float("inf")),
])
def test_normalise_unusable_part_is_none(residual, noise):
    assert normalise_residuals({"a": residual}, {"a": noise}) == {"a": None}


def test_normalise_part_without_noise_entry_is_none():
    assert normalise_residuals({"a": 1.0}, {}) == {"a": None}


def test_worst_part_picks_highest_normalised_not_highest_raw():
    raw = {"gearbox_bearing": 6.0, "generator_bearing_1": 4.0}
    noise = {"gearbox_bearing": 3.0, "generator_bearing_1": 1.0}
    assert worst_part(normalise_residuals(raw, noise)) == ("generator_bearing_1", 4.0)


def test_worst_part_skips_none_and_handles_all_missing():
    assert worst_part({"a": None, "b": -1.0}) == ("b", -1.0)
    assert worst_part({"a": None, "b": None}) == (None, None)
    assert worst_part({}) == (None, None)


# ---- score and colour (L-01) ----------------------------------------------

def test_score_anchors_and_range():
    assert health_score(-2.0, CFG) == 1.0
    assert health_score(0.0, CFG) == 1.0
    assert health_score(3.0, CFG) == pytest.approx(0.7)
    assert health_score(5.0, CFG) == pytest.approx(0.4)
    assert health_score(10.0, CFG) == 0.0
    assert health_score(99.0, CFG) == 0.0


def test_score_is_non_increasing_and_agrees_with_colour():
    zs = [i / 10 for i in range(-20, 150)]
    scores = [health_score(z, CFG) for z in zs]
    assert all(a >= b for a, b in zip(scores, scores[1:]))
    for z, s in zip(zs, scores):
        level = health_level(z, CFG)
        if level == GREEN:
            assert s > CFG.score_at_yellow
        elif level == YELLOW:
            assert CFG.score_at_red < s <= CFG.score_at_yellow
        else:
            assert s <= CFG.score_at_red


def test_config_has_no_default_thresholds_and_validates():
    with pytest.raises(TypeError):
        HealthConfig()  # thresholds must be passed explicitly (T-08 pending)
    with pytest.raises(ValueError):
        HealthConfig(yellow_z=5, red_z=3, persistence_hours=1, max_gap_minutes=30)
    with pytest.raises(ValueError):
        HealthConfig(yellow_z=3, red_z=5, persistence_hours=0, max_gap_minutes=30)


# ---- persistence (L-02) ---------------------------------------------------

def test_single_spike_never_changes_colour_or_score():
    t = PersistenceTracker(CFG)
    results = feed(t, [0.0] * 10 + [50.0] + [0.0] * 10)
    assert all(r.health == GREEN for r in results)
    assert all(r.health_score == 1.0 for r in results)
    assert results[10].normalised["gearbox_bearing"] == 50.0  # raw z still visible


def test_turns_yellow_only_after_full_persistence_window():
    t = PersistenceTracker(CFG)
    results = feed(t, [0.0] * 6 + [4.0] * 8)  # z=4 from minute 60 onwards
    # yellow run starts at minute 60; window is 60 min, so yellow at minute 120 (index 12)
    assert [r.health for r in results[:12]] == [GREEN] * 12
    assert results[12].health == YELLOW
    assert results[12].sustained_z == 4.0


def test_start_of_replay_counts_as_unconfirmed():
    t = PersistenceTracker(CFG)
    results = feed(t, [6.0] * 8)
    assert [r.health for r in results[:6]] == [GREEN] * 6
    assert results[6].health == RED  # first reading whose history covers 60 minutes


def test_red_requires_red_level_for_the_whole_window():
    t = PersistenceTracker(CFG)
    results = feed(t, [4.0] * 7 + [6.0] * 6 + [6.0])
    assert results[6].health == YELLOW
    assert results[12].health == YELLOW  # red run only 50 minutes so far
    assert results[13].health == RED


def test_dip_below_threshold_restarts_the_run():
    t = PersistenceTracker(CFG)
    results = feed(t, [4.0] * 7 + [1.0] + [4.0] * 7)
    assert results[6].health == YELLOW
    assert results[7].health == GREEN
    assert results[13].health == GREEN
    assert results[14].health == YELLOW


def test_gap_longer_than_max_gap_resets_window():
    t = PersistenceTracker(CFG)
    feed(t, [4.0] * 7)
    assert t.last.health == YELLOW
    after_gap = t.update(at(60 + 45), {"p": 4.0})  # 45 min gap > 30 min
    assert after_gap.health == GREEN


def test_gap_within_max_gap_keeps_run():
    t = PersistenceTracker(CFG)
    results = [t.update(at(m), {"p": 4.0}) for m in (0, 20, 40, 60)]  # 20-min steps
    assert results[-1].health == YELLOW


def test_unscorable_reading_is_unknown_never_green():
    t = PersistenceTracker(CFG)
    first = t.update(at(0), {"p": None})
    assert first.health is None and first.health_score is None and not first.scored
    feed(t, [4.0] * 7, start=10)
    assert t.last.health == YELLOW
    r = t.update(at(80), {"p": None, "q": None})
    assert not r.scored
    assert r.health is None and r.health_score is None  # L-13: unknown, not carried over
    assert r.worst_part is None and r.sustained_z is None


# ---- per-part persistence (L-14) ---------------------------------------------

def test_parts_taking_turns_never_sustain_an_alert():
    # Each part is high for 40 min, then the other: the turbine-wide worst z is
    # always 4, but no single part stays high for the 60-minute window.
    t = PersistenceTracker(CFG)
    results = []
    for i in range(24):
        a_high = (i // 4) % 2 == 0
        results.append(t.update(at(i * 10), {"a": 4.0 if a_high else 0.0, "b": 0.0 if a_high else 4.0}))
    assert all(r.health == GREEN for r in results)


def test_worst_sustained_part_sets_colour():
    t = PersistenceTracker(CFG)
    results = []
    for i in range(8):
        # a: sustained z=4 the whole time. b: a single spike of 50 at the end.
        results.append(t.update(at(i * 10), {"a": 4.0, "b": 50.0 if i == 7 else 0.0}))
    last = results[-1]
    assert last.health == YELLOW
    assert last.worst_part == "a"
    assert last.sustained == {"a": 4.0, "b": 0.0}


def test_each_part_needs_its_own_full_window():
    t = PersistenceTracker(CFG)
    results = [t.update(at(i * 10), {"a": 6.0, "b": 4.0 if i >= 3 else 0.0}) for i in range(10)]
    assert results[6].health == RED and results[6].worst_part == "a"
    assert results[6].sustained["b"] == 0.0  # b only high for 40 minutes


def test_unusable_part_is_left_out_of_that_reading():
    t = PersistenceTracker(CFG)
    feed(t, [6.0] * 7, part="a")  # minutes 0..60: a is red
    r = t.update(at(70), {"a": None, "b": 0.0})
    assert r.health == GREEN and r.worst_part == "b"  # only b usable now
    r = t.update(at(80), {"a": 6.0, "b": 0.0})
    assert r.health == RED  # a's run continues: its gap (20 min) is within max_gap


def test_part_gap_resets_only_that_part():
    t = PersistenceTracker(CFG)
    for i in range(7):
        t.update(at(i * 10), {"a": 6.0, "b": 4.0})
    for m in (70, 80, 90, 100):  # a unusable for 40 minutes (> 30 min max gap)
        t.update(at(m), {"a": None, "b": 4.0})
    r = t.update(at(110), {"a": 6.0, "b": 4.0})
    assert r.sustained["a"] == 0.0  # a restarted
    assert r.health == YELLOW and r.worst_part == "b"


def test_anomaly_score_is_not_an_input():
    import inspect
    params = inspect.signature(PersistenceTracker.update).parameters
    assert "anomaly_score" not in params


def test_timestamps_must_increase():
    t = PersistenceTracker(CFG)
    t.update(at(10), {"p": 0.0})
    with pytest.raises(ValueError):
        t.update(at(10), {"p": 0.0})
    with pytest.raises(ValueError):
        t.update(at(0), {"p": 0.0})


def test_trackers_are_independent_per_dataset():
    a, b = PersistenceTracker(CFG), PersistenceTracker(CFG)
    feed(a, [6.0] * 7)
    r = b.update(at(60), {"p": 0.0})
    assert a.last.health == RED
    assert r.health == GREEN


def test_outputs_are_finite_numbers():
    t = PersistenceTracker(CFG)
    for r in feed(t, [0.5, 2.0, 7.0, 3.5]):
        assert math.isfinite(r.health_score) and 0.0 <= r.health_score <= 1.0


# ---- B1: permitted gap straddling the window cutoff ---------------------------

def test_permitted_gap_cannot_shorten_persistence():
    cfg = HealthConfig(yellow_z=3.0, red_z=5.0, persistence_hours=3.0, max_gap_minutes=60.0)
    t = PersistenceTracker(cfg)
    for m in range(0, 61, 10):
        t.update(at(m), {"a": 0.0})          # last normal reading at minute 60
    results = {m: t.update(at(m), {"a": 4.0}).health for m in range(115, 400, 10)}  # 55-min gap
    first_yellow = min(m for m, h in results.items() if h == YELLOW)
    # Must wait until the minute-60 reading (the part's state at the window start)
    # has left the window: 60 + 180 < m, and the next reading is at 115 -> 115 + 180.
    assert first_yellow == 295
    assert first_yellow - 115 >= 180


def test_reading_exactly_at_cutoff_counts_as_window_start():
    t = PersistenceTracker(CFG)
    results = feed(t, [0.0] + [4.0] * 7)  # minute 0 normal, high from minute 10
    assert results[6].health == GREEN     # minute 60: window [0, 60] still holds the normal reading
    assert results[7].health == YELLOW    # minute 70: high for the full 60 minutes


# ---- B2: max gap must be smaller than the persistence window -----------------

@pytest.mark.parametrize("hours,gap", [(1.0, 60.0), (1.0, 120.0), (0.5, 45.0)])
def test_max_gap_not_smaller_than_window_is_rejected(hours, gap):
    with pytest.raises(ValueError, match="max_gap_minutes"):
        HealthConfig(yellow_z=3, red_z=5, persistence_hours=hours, max_gap_minutes=gap)


def test_max_gap_just_below_window_is_allowed():
    HealthConfig(yellow_z=3, red_z=5, persistence_hours=1.0, max_gap_minutes=59.9)


# ---- missing component during an existing alert (C1 pending: CURRENT behaviour) ----

def test_missing_component_during_alert_current_behaviour():
    """Documents L-07 as implemented today. If the team approves C1 option (a)
    (hold a part's sustained value through a short gap), this test must change."""
    t = PersistenceTracker(CFG)
    seq = []
    for i in range(12):
        g = None if i == 9 else 6.0
        seq.append(t.update(at(i * 10), {"gearbox_bearing": g, "generator_bearing_1": 0.0}))
    assert [r.health for r in seq[6:]] == [RED, RED, RED, GREEN, RED, RED]
    assert seq[9].worst_part == "generator_bearing_1"
    assert seq[10].sustained["gearbox_bearing"] == 6.0  # run resumed: gap within max_gap

    t = PersistenceTracker(CFG)
    only = [t.update(at(i * 10), {"a": None if i == 9 else 6.0}).health for i in range(12)]
    assert only[6:] == [RED, RED, RED, None, RED, RED]  # unknown, never green (L-13)
