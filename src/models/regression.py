from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["month_num"] = data["month"].dt.month
    data["year"] = data["month"].dt.year
    return data


def regression_forecast(monthly: pd.DataFrame) -> pd.DataFrame:
    forecasts = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month")
        last_month = group["month"].iloc[-1]
        next_month = last_month + pd.offsets.MonthBegin(1)

        features = _calendar_features(group)
        X = features[["month_num", "year"]]
        y = group["count"]

        pred = float(y.iloc[-1])
        if len(group) >= 6:
            model = LinearRegression()
            model.fit(X, y)
            future_X = pd.DataFrame({
                "month_num": [next_month.month],
                "year": [next_month.year],
            })
            pred = float(model.predict(future_X)[0])

        forecasts.append({"category": category, "month": next_month, "predicted": pred})

    return pd.DataFrame(forecasts)
