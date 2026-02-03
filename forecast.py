from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATA_URL
from src.evaluate import evaluate_models
from src.io import load_data
from src.models.baselines import seasonal_naive_forecast
from src.models.holt_winters import holt_winters_forecast
from src.models.regression import regression_forecast
from src.models.sarima import sarima_forecast
from src.preprocess import build_monthly_series, complete_monthly_index


def _run_models(monthly: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    forecasts = []
    for model_name in models:
        if model_name == "naive":
            forecast_df = seasonal_naive_forecast(monthly)
        elif model_name == "holt":
            forecast_df = holt_winters_forecast(monthly)
        elif model_name == "sarima":
            forecast_df = sarima_forecast(monthly)
        elif model_name == "regression":
            forecast_df = regression_forecast(monthly)
        else:
            continue
        forecast_df["model"] = model_name
        forecasts.append(forecast_df)
    if not forecasts:
        return pd.DataFrame()
    combined = pd.concat(forecasts, ignore_index=True)
    combined["predicted"] = combined["predicted"].clip(lower=0)
    combined = combined.sort_values(["model", "predicted"], ascending=[True, False])
    combined["rank"] = combined.groupby("model")["predicted"].rank(method="first", ascending=False).astype(int)
    return combined


def _filter_top_items(monthly: pd.DataFrame, max_items_value: int) -> pd.DataFrame:
    if not max_items_value or max_items_value <= 0:
        return monthly
    totals = monthly.groupby("category")["count"].sum().sort_values(ascending=False)
    top_items = totals.head(int(max_items_value)).index
    return monthly[monthly["category"].isin(top_items)].copy()


def _filter_sparse_items(
    monthly: pd.DataFrame,
    min_history: int,
    recent_window_value: int,
    min_recent_nonzero_value: int,
) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    history = monthly.groupby("category")["month"].nunique()
    keep = history[history >= int(min_history)].index
    filtered = monthly[monthly["category"].isin(keep)].copy()
    if min_recent_nonzero_value > 0 and not filtered.empty:
        filtered = filtered.sort_values("month")
        recent = filtered.groupby("category").tail(int(recent_window_value))
        nonzero = recent[recent["count"] > 0].groupby("category").size()
        keep_recent = nonzero[nonzero >= int(min_recent_nonzero_value)].index
        filtered = filtered[filtered["category"].isin(keep_recent)].copy()
    return filtered


def _recent_average(monthly: pd.DataFrame, window: int) -> pd.Series:
    monthly = monthly.sort_values("month")
    recent = monthly.groupby("category").tail(int(window))
    return recent.groupby("category")["count"].mean()


def _apply_guardrails_to_forecasts(
    forecasts: pd.DataFrame,
    recent_avg: pd.Series,
    multiplier: float,
) -> pd.DataFrame:
    if forecasts.empty or recent_avg.empty:
        return forecasts
    merged = forecasts.merge(recent_avg.rename("recent_avg"), on="category", how="left")
    mask = merged["recent_avg"] > 0
    merged.loc[mask, "predicted"] = np.minimum(
        merged.loc[mask, "predicted"],
        merged.loc[mask, "recent_avg"] * float(multiplier),
    )
    return merged.drop(columns=["recent_avg"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Library book demand forecasting (books only)")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--year-col", type=str, default=None)
    parser.add_argument("--month-col", type=str, default=None)
    parser.add_argument("--category-col", type=str, default=None)
    parser.add_argument("--count-col", type=str, default=None)
    parser.add_argument("--book-col", type=str, default=None)
    parser.add_argument("--model", type=str, default="all", choices=["naive", "holt", "sarima", "regression", "all"])
    parser.add_argument("--backtest-months", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--min-history-months", type=int, default=12)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--min-recent-nonzero", type=int, default=2)
    parser.add_argument("--guardrail-multiplier", type=float, default=3.0)
    parser.add_argument("--output-forecast", type=str, default="outputs/forecast_next_month.csv")
    parser.add_argument("--output-metrics", type=str, default="outputs/metrics_report.json")
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None

    data = load_data(
        data_path=args.data_path,
        data_url=args.data_url,
        max_rows=max_rows,
        cache_path=args.cache_path,
    )
    monthly = build_monthly_series(
        data,
        date_col=args.date_col,
        year_col=args.year_col,
        month_col=args.month_col,
        category_col=args.category_col,
        count_col=args.count_col,
        book_col=args.book_col,
        only_books=True,
    )
    monthly = complete_monthly_index(monthly)
    monthly = _filter_top_items(monthly, args.max_items)
    monthly = _filter_sparse_items(
        monthly,
        min_history=int(args.min_history_months),
        recent_window_value=int(args.recent_window),
        min_recent_nonzero_value=int(args.min_recent_nonzero),
    )

    model_list = [args.model] if args.model != "all" else ["naive", "regression", "holt", "sarima"]

    forecasts = _run_models(monthly, model_list)
    if forecasts.empty:
        raise RuntimeError("No forecasts were generated. Check data columns and history length.")
    recent_avg = _recent_average(monthly, int(args.recent_window))
    forecasts = _apply_guardrails_to_forecasts(forecasts, recent_avg, float(args.guardrail_multiplier))

    os.makedirs(os.path.dirname(args.output_forecast), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)

    forecasts.to_csv(args.output_forecast, index=False)

    metrics = evaluate_models(monthly, model_list, backtest_months=args.backtest_months)
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(len(monthly)),
        "categories": int(monthly["category"].nunique()),
        "months": int(monthly["month"].nunique()),
        "models": metrics,
    }
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    top = forecasts.sort_values(["model", "rank"]).groupby("model").head(5)
    print("Top books/categories by model:\n", top[["model", "category", "predicted", "rank"]].to_string(index=False))
    print(f"\nSaved forecast to {args.output_forecast}")
    print(f"Saved metrics to {args.output_metrics}")


if __name__ == "__main__":
    main()
