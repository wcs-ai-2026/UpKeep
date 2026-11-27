
import pandas as pd

from .reasons import build_reasons


def format_duration(hours):
    if hours <= 0:
        return ""

    if hours >= 24:
        value = hours / 24
        unit = "day" if value == 1 else "days"
    else:
        value = hours
        unit = "hour" if value == 1 else "hours"

    return f" for at least {value:.2g} {unit}"


def explain(recent: pd.DataFrame,
            residuals: dict,
            gusts: dict) -> dict:

    reasons = build_reasons(recent, residuals)

    gust_count = gusts.get("gusts_last_7d")

    if pd.isna(gust_count):
        gust_context = "Weekly gust data unavailable"
    else:
        gust_count = int(gust_count)
        gust_context = (
            f"{gust_count} high-gust periods this week"
        )

    if reasons:
        descriptions = []

        for reason in reasons:
            part = reason["part"].replace("_", " ")
            residual = reason["residual_c"]
            hours = reason["hours"]

            direction = "hotter" if residual > 0 else "colder"

            description = (
                f"{part} has run {abs(residual):.1f}°C "
                f"{direction} than expected"
            )

            description += format_duration(hours)
            descriptions.append(description)

        sentence = "; ".join(descriptions)
        sentence = sentence[0].upper() + sentence[1:] + "."

    else:
        sentence = "No unusual temperature readings detected."

    sentence += f" {gust_context}."

    return {
        "reasons": reasons,
        "gust_context": gust_context,
        "sentence": sentence
    }
