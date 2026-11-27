
import pandas as pd
import math


def build_reasons(recent, residuals, threshold=5.0):
    reasons = []

    for part, residual in residuals.items():

        residual = float(residual)

        if not math.isfinite(residual):
            continue

        if abs(residual) < threshold:
            continue

        hours = 0.0
        column = f"{part}_residual"

        if column in recent.columns and "timestamp" in recent.columns:
            history = recent.copy()
            history["timestamp"] = pd.to_datetime(
                history["timestamp"]
            )
            history = history.sort_values("timestamp")

            latest = history["timestamp"].iloc[-1] if not history.empty else None
            previous = None
            earliest = None

            for _, row in history.iloc[::-1].iterrows():
                value = row[column]
                timestamp = row["timestamp"]

                if pd.isna(value) or pd.isna(timestamp):
                    break

                if abs(value) < threshold or value * residual <= 0:
                    break

                if previous is not None:
                    if previous - timestamp != pd.Timedelta(minutes=10):
                        break

                earliest = timestamp
                previous = timestamp

            if earliest is not None:
                hours = (latest - earliest).total_seconds() / 3600

        direction = "hotter" if residual > 0 else "colder"
        name = part.replace("_", " ").capitalize()

        text = (
            f"{name} {abs(residual):.1f}°C "
            f"{direction} than expected"
        )

        if hours > 0:
            text += f" for at least {hours:.1f} hours"

        reasons.append({
            "part": part,
            "residual_c": round(residual, 2),
            "hours": round(hours, 2),
            "text": text
        })

    reasons.sort(
        key=lambda reason: abs(reason["residual_c"]),
        reverse=True
    )

    return reasons[:3]
