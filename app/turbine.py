
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

st.set_page_config(
    page_title="Upkeep | Turbine Details",
    page_icon="🌬️",
    layout="wide"
)

# Files supplied by the data and ML teams
DATASETS = {
    "0": (
        "Data/seed/0.parquet",
        "predictions/lgbm_gearbox_bearing_temp_0.csv"
    ),
    "10": (
        "Data/seed/10.parquet",
        "predictions/lgbm_gearbox_bearing_temp_10.csv"
    ),
    "25": (
        "Data/seed/25.parquet",
        "predictions/lgbm_gearbox_bearing_temp_25.csv"
    )
}


@st.cache_data
def load_data(dataset):
    seed_file, prediction_file = DATASETS[dataset]

    readings = pd.read_parquet(
        PROJECT_ROOT / seed_file
    )

    predictions = pd.read_csv(
        PROJECT_ROOT / prediction_file
    )

    # Prediction CSV uses reading_id instead of row_id.
    # Assumes both identifiers refer to the same records.
    predictions = predictions.rename(
        columns={"reading_id": "row_id"}
    )

    required_readings = {
        "row_id", "turbine_id", "timestamp"
    }
    required_predictions = {
        "row_id", "actual", "expected", "residual"
    }

    missing_readings = required_readings - set(readings.columns)
    missing_predictions = required_predictions - set(predictions.columns)

    if missing_readings or missing_predictions:
        raise ValueError(
            f"Missing reading columns: {missing_readings}. "
            f"Missing prediction columns: {missing_predictions}."
        )

    if readings["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id values in seed data.")

    if predictions["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id values in predictions.")

    # Join observations with model predictions
    df = readings.merge(
        predictions,
        on="row_id",
        how="inner",
        validate="one_to_one"
    )

    if df.empty:
        raise ValueError(
            "No matching row IDs found between seed data "
            "and prediction data. Check the ID mapping."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    return df


# Header
st.title("Upkeep | Turbine Details")
st.write("Wind Turbine Temperature Monitoring")
st.caption("Actual vs Expected Gearbox Bearing Temperature")

st.divider()

# Dataset selection
dataset = st.selectbox(
    "Select Dataset",
    list(DATASETS.keys())
)

try:
    df = load_data(dataset)
except Exception as error:
    st.error(f"Unable to load turbine data: {error}")
    st.stop()

# Turbine selection
turbines = sorted(df["turbine_id"].dropna().unique())

if len(turbines) == 0:
    st.warning("No turbines found in the selected dataset.")
    st.stop()

selected_turbine = st.selectbox(
    "Select Turbine",
    turbines
)

turbine_data = df[
    df["turbine_id"] == selected_turbine
].copy()

st.divider()

# Temperature summary
st.subheader(f"Turbine {selected_turbine}")

latest = turbine_data.iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Actual Temperature",
    f"{latest['actual']:.1f} °C"
)

col2.metric(
    "Expected Temperature",
    f"{latest['expected']:.1f} °C"
)

col3.metric(
    "Temperature Deviation",
    f"{latest['residual']:+.1f} °C"
)

st.divider()

# Actual vs expected temperature chart
st.subheader("Actual vs Expected Temperature")

chart_data = turbine_data.set_index("timestamp")[
    ["actual", "expected"]
]

st.line_chart(
    chart_data,
    x_label="Time",
    y_label="Temperature (°C)",
    width="stretch"
)

st.caption(
    "The chart compares measured gearbox bearing "
    "temperatures with LightGBM model predictions."
)

st.divider()

# Temperature data table
st.subheader("Temperature Readings")

display_data = turbine_data[
    ["timestamp", "actual", "expected", "residual"]
].rename(
    columns={
        "timestamp": "Timestamp",
        "actual": "Actual (°C)",
        "expected": "Expected (°C)",
        "residual": "Deviation (°C)"
    }
)

st.dataframe(
    display_data,
    hide_index=True,
    width="stretch"
)

st.caption(
    "Upkeep | Western Cyber Society | Turbine Dashboard"
)

