from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _make_period_series(group: pd.DataFrame) -> pd.Series:
    """Return a PeriodIndex-based series to avoid statsmodels FutureWarning."""
    s = group.set_index("month")["count"].copy()
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def sarima_forecast(monthly: pd.DataFrame, seasonal_periods: int = 12) -> pd.DataFrame:
    forecasts = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month")
        series = group["count"]
        last_month = group["month"].iloc[-1]
        next_month = last_month + pd.offsets.MonthBegin(1)
        pred = float(series.tail(3).mean()) if len(series) >= 3 else float(series.iloc[-1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress FutureWarning + ConvergenceWarning

            if len(series) >= seasonal_periods * 2:
                # Full seasonal ARIMA
                try:
                    ps = _make_period_series(group)
                    model = SARIMAX(
                        ps,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, seasonal_periods),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = model.fit(disp=False)
                    pred = float(fit.forecast(1).iloc[0])
                except Exception:
                    pass

            elif len(series) >= 6:
                # ARIMA without seasonal component — not enough data for full seasonal model
                try:
                    ps = _make_period_series(group)
                    model = SARIMAX(
                        ps,
                        order=(1, 1, 1),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = model.fit(disp=False)
                    pred = float(fit.forecast(1).iloc[0])
                except Exception:
                    pass

        forecasts.append({"category": category, "month": next_month, "predicted": pred})
    return pd.DataFrame(forecasts)
