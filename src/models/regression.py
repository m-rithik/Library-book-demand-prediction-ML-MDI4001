from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


def regression_forecast(monthly: pd.DataFrame) -> pd.DataFrame:
    forecasts = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month").copy()
        last_month = group["month"].iloc[-1]
        next_month = last_month + pd.offsets.MonthBegin(1)

        group["month_num"] = group["month"].dt.month
        group["year"] = group["month"].dt.year
        group["lag1"] = group["count"].shift(1)
        group["avg3"] = group["count"].shift(1).rolling(3).mean()

        pred = float(group["count"].iloc[-1])

        if len(group) >= 6:
            rich = group.dropna(subset=["lag1", "avg3"])
            if len(rich) >= 4:
                X = rich[["month_num", "year", "lag1", "avg3"]]
                y = rich["count"]
                model = LinearRegression()
                model.fit(X, y)
                last_row = group.iloc[-1]
                future_X = pd.DataFrame({
                    "month_num": [next_month.month],
                    "year": [next_month.year],
                    "lag1": [last_row["count"]],
                    "avg3": [group["count"].tail(3).mean()],
                })
                pred = float(model.predict(future_X)[0])
            else:
                X = group[["month_num", "year"]]
                y = group["count"]
                model = LinearRegression()
                model.fit(X, y)
                future_X = pd.DataFrame({
                    "month_num": [next_month.month],
                    "year": [next_month.year],
                })
                pred = float(model.predict(future_X)[0])

        forecasts.append({"category": category, "month": next_month, "predicted": pred})

    return pd.DataFrame(forecasts)
