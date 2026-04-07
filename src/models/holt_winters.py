from __future__ import annotations

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def holt_winters_forecast(monthly: pd.DataFrame, seasonal_periods: int = 12) -> pd.DataFrame:
    forecasts = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month")
        series = group["count"]
        last_month = group["month"].iloc[-1]
        next_month = last_month + pd.offsets.MonthBegin(1)
        pred = float(series.tail(3).mean()) if len(series) >= 3 else float(series.iloc[-1])

        if len(series) >= seasonal_periods * 2:
            # Full Holt-Winters with trend + seasonality
            try:
                model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                pred = float(fit.forecast(1).iloc[0])
            except Exception:
                pass

        elif len(series) >= seasonal_periods:
            # Holt's linear trend — no seasonal component (not enough data for a full cycle)
            try:
                model = ExponentialSmoothing(
                    series, trend="add", seasonal=None,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                pred = float(fit.forecast(1).iloc[0])
            except Exception:
                pass

        elif len(series) >= 4:
            # Simple exponential smoothing — short series, just smooth the level
            try:
                model = ExponentialSmoothing(
                    series, trend=None, seasonal=None,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                pred = float(fit.forecast(1).iloc[0])
            except Exception:
                pass

        forecasts.append({"category": category, "month": next_month, "predicted": pred})
    return pd.DataFrame(forecasts)
