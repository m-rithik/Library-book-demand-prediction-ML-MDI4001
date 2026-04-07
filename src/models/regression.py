from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge


def regression_forecast(monthly: pd.DataFrame) -> pd.DataFrame:
    forecasts = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month").copy()
        last_month = group["month"].iloc[-1]
        next_month = last_month + pd.offsets.MonthBegin(1)

        group["month_num"] = group["month"].dt.month
        group["year"]      = group["month"].dt.year
        group["lag1"]      = group["count"].shift(1)
        group["avg3"]      = group["count"].shift(1).rolling(3).mean()
        group["lag12"]     = group["count"].shift(12)   # same month last year — captures seasonality

        # Default: recent 3-month average
        pred = float(group["count"].tail(3).mean()) if len(group) >= 3 else float(group["count"].iloc[-1])

        last_row = group.iloc[-1]

        if len(group) >= 6:
            # Try full feature set (lag1 + avg3 + lag12 when available)
            feature_cols = ["month_num", "year", "lag1", "avg3"]
            has_lag12 = not pd.isna(last_row["lag12"])
            if has_lag12:
                feature_cols.append("lag12")

            rich = group.dropna(subset=["lag1", "avg3"])
            if has_lag12:
                rich = rich.dropna(subset=["lag12"])

            if len(rich) >= 4:
                X = rich[feature_cols]
                y = rich["count"]
                model = Ridge(alpha=1.0)
                model.fit(X, y)
                future_X = pd.DataFrame({
                    "month_num": [next_month.month],
                    "year":      [next_month.year],
                    "lag1":      [last_row["count"]],
                    "avg3":      [group["count"].tail(3).mean()],
                })
                if has_lag12:
                    future_X["lag12"] = [last_row["lag12"]]
                pred = float(model.predict(future_X)[0])
            else:
                # Short series — calendar features only
                X = group[["month_num", "year"]]
                y = group["count"]
                model = Ridge(alpha=1.0)
                model.fit(X, y)
                future_X = pd.DataFrame({
                    "month_num": [next_month.month],
                    "year":      [next_month.year],
                })
                pred = float(model.predict(future_X)[0])

        forecasts.append({"category": category, "month": next_month, "predicted": pred})

    return pd.DataFrame(forecasts)
