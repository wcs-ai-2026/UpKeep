import pandas as pd
import pytest

from upkeep.ids import dataset_number, format_dataset_id, format_turbine_id, turbine_number
from upkeep.paths import data_dir
from upkeep.stream.faults import days_to_fault, fault_info, load_events

EVENTS = pd.DataFrame({
    "event_id": [7, 8, 9],
    "event_label": ["anomaly", "normal", "anomaly"],
    # Event timestamps deliberately wrong: the lookup must ignore them.
    "event_start": ["1999-01-01 00:00:00"] * 3,
    "event_start_id": [100, 100, 500],
}).set_index("event_id")


def readings(row_ids, start="2023-01-01 00:00:00"):
    ts = pd.date_range(start, periods=len(row_ids), freq="10min")
    return pd.DataFrame({"row_id": row_ids, "timestamp": ts})


# ---- ids (T-02) ------------------------------------------------------------

def test_id_formats():
    assert format_dataset_id(25) == format_dataset_id("25") == format_dataset_id("A-25") == "A-25"
    assert format_turbine_id(11) == format_turbine_id("A-T11") == "A-T11"
    assert dataset_number("A-0") == 0 and turbine_number("A-T10") == 10
    for bad in ("A-T11", "B-3", "", -1, True):
        with pytest.raises(ValueError):
            dataset_number(bad)


# ---- fault lookup (T-11) ---------------------------------------------------

def test_anomaly_uses_event_start_row_id_not_timestamp():
    df = readings([98, 99, 100, 101])
    info = fault_info(7, df, EVENTS)
    assert info.fault_event == "A-7"
    assert info.fault_start_row_id == 100
    assert info.fault_start == pd.Timestamp("2023-01-01 00:20:00")


def test_days_to_fault_sign_and_zero():
    df = readings([98, 99, 100, 101])
    info = fault_info(7, df, EVENTS)
    assert info.days_to_fault(df.timestamp[0]) == pytest.approx(20 / 1440)
    assert info.days_to_fault(df.timestamp[2]) == 0
    assert info.days_to_fault(df.timestamp[3]) == pytest.approx(-10 / 1440)


def test_removed_start_row_uses_first_later_reading():
    df = readings([98, 99, 103, 105])  # rows 100-102 removed in cleaning
    info = fault_info("A-7", df, EVENTS)
    assert info.fault_start_row_id == 103
    assert info.fault_start == df.timestamp[2]


def test_first_later_reading_found_by_row_id_even_if_unsorted():
    df = readings([105, 98, 103, 99])
    assert fault_info(7, df, EVENTS).fault_start_row_id == 103


def test_no_reading_at_or_after_start_keeps_fault_event_but_null_days():
    df = readings([97, 98, 99])
    info = fault_info(7, df, EVENTS)
    assert info.fault_event == "A-7"
    assert info.fault_start is None
    assert info.days_to_fault(df.timestamp[0]) is None


def test_normal_event_gives_nulls():
    info = fault_info(8, readings([99, 100, 101]), EVENTS)
    assert info.fault_event is None and info.fault_start is None
    assert info.days_to_fault("2023-01-01") is None


def test_unknown_dataset_is_an_error():
    with pytest.raises(KeyError):
        fault_info(42, readings([1]), EVENTS)


def test_days_to_fault_with_unknown_start():
    assert days_to_fault("2023-01-01", None) is None


# ---- real seed files ---------------------------------------------------------

SEED = data_dir() / "seed"


@pytest.mark.skipif(not (SEED / "0.parquet").exists(), reason="seed data not present")
@pytest.mark.parametrize("dataset,event,start_row,start_ts", [
    (0, "A-0", 52436, "2023-08-05 06:10:00"),
    (10, "A-10", 52611, "2023-10-12 08:40:00"),
    (25, None, None, None),
])
def test_seed_datasets(dataset, event, start_row, start_ts):
    df = pd.read_parquet(SEED / f"{dataset}.parquet", columns=["row_id", "timestamp"])
    info = fault_info(dataset, df, load_events())
    assert info.dataset_id == f"A-{dataset}"
    assert info.fault_event == event
    assert info.fault_start_row_id == start_row
    if start_ts is None:
        assert info.fault_start is None
    else:
        assert info.fault_start == pd.Timestamp(start_ts)
        at_start = df.loc[df.row_id == start_row, "timestamp"].iloc[0]
        assert info.days_to_fault(at_start) == 0
