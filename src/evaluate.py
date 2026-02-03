from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .models.baselines import seasonal_naive_forecast
from .models.holt_winters import holt_winters_forecast
from .models.regression import regression_forecast
from .models.sarima import sarima_forecast


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    error = df["predicted"] - df["actual"]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denom = np.maximum(df["actual"].values, 1)
    mape = float(np.mean(np.abs(error) / denom))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate_models(
    monthly: pd.DataFrame,
    models: Iterable[str],
    backtest_months: int = 3,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    unique_months = sorted(monthly["month"].unique())
    if len(unique_months) < 2:
        return results

    eval_months = unique_months[-backtest_months:]
    for model_name in models:
        all_rows = []
        for eval_month in eval_months:
            train_df = monthly[monthly["month"] < eval_month]
            actual_df = monthly[monthly["month"] == eval_month][["category", "count"]].rename(
                columns={"count": "actual"}
            )
            if train_df.empty or actual_df.empty:
                continue

            if model_name == "naive":
                forecast_df = seasonal_naive_forecast(train_df)
            elif model_name == "holt":
                forecast_df = holt_winters_forecast(train_df)
            elif model_name == "sarima":
                forecast_df = sarima_forecast(train_df)
            elif model_name == "regression":
                forecast_df = regression_forecast(train_df)
            else:
                continue

            merged = actual_df.merge(forecast_df[["category", "predicted"]], on="category", how="inner")
            if merged.empty:
                continue
            all_rows.append(merged)

        if all_rows:
            concat = pd.concat(all_rows, ignore_index=True)
            metrics = _compute_metrics(concat)
            metrics["samples"] = int(len(concat))
            results[model_name] = metrics
    return results
