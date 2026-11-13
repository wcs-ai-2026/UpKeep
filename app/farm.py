import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from upkeep.stubs import score_reading
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Upkeep | Wind Farm A",
    page_icon="🌬️",
    layout="wide"
)

# Simulated turbine data
# This will be replaced with the team's data later
turbines = []

for number in range(1, 6):
    reading = {
        "row_id": f"demo-{number}",
        "turbine_id": f"A-T{number}",
        "timestamp": "2026-11-13T12:00:00"
    }

  


    result = score_reading(reading)

   

    turbines.append({
        "id": result["turbine_id"],
        "health": result["health"],
        "score": result["health_score"]
    })


# Health status colours
status_colors = {
    "green": "#22C55E",
    "yellow": "#FACC15",
    "red": "#EF4444"
}

status_icons = {
    "green": "🟢",
    "yellow": "🟡",
    "red": "🔴"
}

# Header
st.title("Upkeep | Wind Farm A")
st.write("Wind Turbine Health Monitoring")
st.caption("Prototype | Simulated turbine data")

st.divider()

# Wind farm summary
st.subheader("Wind Farm Overview")

df = pd.DataFrame(turbines)

healthy = (df["health"] == "green").sum()
warning = (df["health"] == "yellow").sum()
critical = (df["health"] == "red").sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Turbines", len(turbines))
col2.metric("Healthy", int(healthy))
col3.metric("Warning", int(warning))
col4.metric("Critical", int(critical))

st.divider()

# Turbine status section
st.subheader("Turbine Status")

columns = st.columns(5, gap="medium")

for column, turbine in zip(columns, turbines):
    with column:
        turbine_id = turbine["id"]
        health = turbine["health"]
        score = turbine["score"]

        color = status_colors[health]
        icon = status_icons[health]

        st.subheader(turbine_id)

        st.write(
            f"{icon} {health.upper()}"
        )

        st.metric(
            "Health Score",
            f"{score:.0%}"
        )

        # Custom coloured progress bar
        # Uses HTML and CSS for individual turbine colours
        st.markdown(
            f"""
            <div style="
                width: 100%;
                height: 12px;
                background-color: #30343F;
                border-radius: 8px;
                overflow: hidden;
            ">
                <div style="
                    width: {score * 100}%;
                    height: 100%;
                    background-color: {color};
                    border-radius: 8px;
                "></div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

# Additional turbine data table
st.subheader("Turbine Health Data")

st.dataframe(
    df.rename(
        columns={
            "id": "Turbine ID",
            "health": "Health Status",
            "score": "Health Score"
        }
    ),
    hide_index=True,
   width="stretch"
)

st.caption(
    "Upkeep | Western Cyber Society | Demo Dashboard"
)
