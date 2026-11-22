import sys
sys.path.insert(0, "src")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from upkeep.parts import PARTS
from upkeep.stream.model_store import ModelStore

DATASETS = ["0", "10", "25"]


# Working out how much each feature helped the model, using gain.
# Gain means how much a feature actually lowered the error, not just
# how many times it was used. Turning it into a percent of the total
# so every model's numbers add up to 100.
def importance_for(model):
    gain = model.model.booster_.feature_importance(importance_type="gain")
    total = gain.sum()
    pct = (gain / total) * 100 if total > 0 else gain
    return pd.DataFrame({"feature": model.features, "importance_pct": pct})


store = ModelStore("lgbm", parts=PARTS)
rows = []
for part in PARTS:
    for dataset in DATASETS:
        model = store.get(dataset, part)
        table = importance_for(model)
        table["part"] = part
        table["dataset"] = dataset
        rows.append(table)

# Storing result into a CSV file, one row per part per dataset per feature.
all_rows = pd.concat(rows, ignore_index=True)
all_rows = all_rows[["part", "dataset", "feature", "importance_pct"]]
all_rows.to_csv("feature_importance.csv", index=False)

# Averaging the 3 datasets together, so one dataset being unusual
# doesn't skew which feature looks most important.
averaged = (
    all_rows.groupby(["part", "feature"], as_index=False)["importance_pct"]
    .mean()
    .sort_values("importance_pct", ascending=False)
)

# Building one chart with a section per part.
fig = make_subplots(rows=len(PARTS), cols=1, subplot_titles=[p.replace("_", " ") for p in PARTS])
for i, part in enumerate(PARTS, start=1):
    part_rows = averaged[averaged["part"] == part].sort_values("importance_pct")
    fig.add_trace(
        go.Bar(x=part_rows["importance_pct"], y=part_rows["feature"], orientation="h"),
        row=i, col=1,
    )

fig.update_layout(
    height=350 * len(PARTS),
    title="Which inputs matter most, per part",
    showlegend=False,
)
fig.update_xaxes(title_text="importance (%)")
fig.write_html("feature_importance.html")

print("Wrote feature_importance.csv and feature_importance.html")
print(averaged.to_string(index=False))
