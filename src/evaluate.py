from __future__ import annotations

import warnings
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .models.baselines import seasonal_naive_forecast
from .models.gradient_boost import gradient_boost_forecast
from .models.holt_winters import holt_winters_forecast
from .models.naive_bayes import naive_bayes_forecast
from .models.random_forest import random_forest_forecast
from .models.regression import regression_forecast
from .models.sarima import sarima_forecast


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    error = df["predicted"] - df["actual"]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denom = np.maximum(df["actual"].values, 1)
    mape = float(np.mean(np.abs(error) / denom))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _forecast_for_model(model_name: str, train_df: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        if model_name == "naive_bayes":
            return naive_bayes_forecast(train_df)
        elif model_name == "regression":
            return regression_forecast(train_df)
        elif model_name == "holt":
            return holt_winters_forecast(train_df)
        elif model_name == "sarima":
            return sarima_forecast(train_df)
        elif model_name == "naive":
            return seasonal_naive_forecast(train_df)
        elif model_name == "random_forest":
            return random_forest_forecast(train_df)
        elif model_name == "gradient_boost":
            return gradient_boost_forecast(train_df)
    return pd.DataFrame()


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
        if model_name == "ensemble":
            continue
        all_rows = []
        for eval_month in eval_months:
            train_df = monthly[monthly["month"] < eval_month]
            actual_df = monthly[monthly["month"] == eval_month][["category", "count"]].rename(
                columns={"count": "actual"}
            )
            if train_df.empty or actual_df.empty:
                continue

            try:
                forecast_df = _forecast_for_model(model_name, train_df)
            except Exception:
                continue

            if forecast_df.empty:
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
