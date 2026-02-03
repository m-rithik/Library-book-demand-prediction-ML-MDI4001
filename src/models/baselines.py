from __future__ import annotations

import pandas as pd


def seasonal_naive_forecast(monthly: pd.DataFrame, fallback_window: int = 3) -> pd.DataFrame:
    forecasts = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month")
        last_month = group["month"].iloc[-1]
        next_month = last_month + pd.offsets.MonthBegin(1)
        target_month = next_month - pd.DateOffset(years=1)
        match = group[group["month"] == target_month]
        if not match.empty:
            pred = match["count"].iloc[0]
        else:
            pred = group["count"].iloc[-1]
        if pred == 0 and len(group) >= fallback_window:
            recent_avg = group["count"].tail(fallback_window).mean()
            if recent_avg > 0:
                pred = recent_avg
        forecasts.append({"category": category, "month": next_month, "predicted": float(pred)})
    return pd.DataFrame(forecasts)
