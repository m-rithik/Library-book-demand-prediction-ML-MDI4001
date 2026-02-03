from __future__ import annotations

import json
import warnings
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import DEFAULT_DATA_URL
from src.evaluate import evaluate_models
from src.io import download_csv
from src.models.baselines import seasonal_naive_forecast
from src.models.holt_winters import holt_winters_forecast
from src.models.regression import regression_forecast
from src.models.sarima import sarima_forecast
from src.preprocess import build_monthly_series, complete_monthly_index


warnings.filterwarnings("ignore", category=ConvergenceWarning)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


st.set_page_config(page_title="Library Book Demand Prediction", layout="wide")

st.title("Library Book Demand Prediction")
st.caption("Books only: predict which book titles will be in demand next month")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose data source", ["URL", "Upload CSV"], horizontal=True)
    data_url = None
    uploaded_file = None
    if source == "URL":
        data_url = st.text_input("CSV URL", value=DEFAULT_DATA_URL)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.header("Book Filter")
    book_col = st.text_input("Book filter column (optional)", value="")
    st.caption("Defaults to material type columns like materialtype.")

    st.header("Prediction Level")
    level = st.radio("Predict by", ["Title (Book names)", "Category"], index=0, horizontal=False)

    st.header("Columns")
    date_mode = st.radio("Date source", ["Single date column", "Year + Month columns"], horizontal=False)
    group_col_label = "Title column" if level.startswith("Title") else "Category column"
    default_group_col = "title" if level.startswith("Title") else ""
    category_col = st.text_input(f"{group_col_label} (optional)", value=default_group_col)
    count_col = st.text_input("Count column (optional)", value="")
    date_col = ""
    year_col = ""
    month_col = ""
    st.caption("Leave blank to auto-detect when possible.")

    st.header("Models")
    models = st.multiselect(
        "Models",
        ["holt", "regression", "sarima", "naive"],
        default=["holt", "regression", "sarima"],
    )
    st.caption("Tip: 'naive' is a baseline; Holt/Regression are more stable for titles.")
    backtest_months = st.number_input("Backtest months", min_value=0, max_value=12, value=3, step=1)

    st.header("Performance")
    max_rows = st.number_input(
        "Max rows to load (0 = no limit)",
        min_value=0,
        max_value=500000,
        value=200000,
        step=10000,
    )
    max_items = st.number_input(
        "Max titles to predict (top by history)",
        min_value=0,
        max_value=200,
        value=30,
        step=5,
    )
    st.header("Data Quality")
    exclude_latest = st.checkbox("Exclude latest month (if partial data)", value=True)
    min_history_months = st.number_input(
        "Min history months per title",
        min_value=3,
        max_value=36,
        value=6,
        step=1,
    )
    recent_window = st.number_input(
        "Recent window (months)",
        min_value=3,
        max_value=24,
        value=6,
        step=1,
    )
    min_recent_nonzero = st.number_input(
        "Min nonzero months in recent window",
        min_value=0,
        max_value=24,
        value=1,
        step=1,
    )
    apply_guardrails = st.checkbox("Apply prediction guardrails", value=True)
    guardrail_multiplier = st.number_input(
        "Max multiple of recent average",
        min_value=1.5,
        max_value=10.0,
        value=3.0,
        step=0.5,
    )

if date_mode == "Single date column":
    date_col = st.text_input("Date column (optional)", value="")
else:
    year_col = st.text_input("Year column (optional)", value="")
    month_col = st.text_input("Month column (optional)", value="")


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


def _apply_fallback_if_all_zero(
    forecasts: pd.DataFrame,
    recent_avg: pd.Series,
    overall_avg: pd.Series,
) -> tuple[pd.DataFrame, List[str]]:
    if forecasts.empty:
        return forecasts, []
    affected_models: List[str] = []

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        if group["predicted"].sum() > 0:
            return group
        affected_models.append(group["model"].iloc[0])
        g = group.merge(recent_avg.rename("recent_avg"), on="category", how="left")
        g = g.merge(overall_avg.rename("overall_avg"), on="category", how="left")
        g["predicted"] = g["predicted"].where(g["predicted"] > 0, g["recent_avg"])
        g["predicted"] = g["predicted"].where(g["predicted"] > 0, g["overall_avg"])
        g["predicted"] = g["predicted"].fillna(0)
        return g.drop(columns=["recent_avg", "overall_avg"])

    fixed = forecasts.groupby("model", group_keys=False).apply(_apply)
    return fixed, affected_models


def _fmt(value: float) -> str:
    return f"{value:,.0f}"


if st.button("Run Forecast", type="primary"):
    with st.spinner("Loading data..."):
        if source == "URL":
            if not data_url:
                st.error("Please enter a CSV URL.")
                st.stop()
            limit = int(max_rows) if max_rows and max_rows > 0 else None
            data = download_csv(data_url, max_rows=limit)
        else:
            if uploaded_file is None:
                st.error("Please upload a CSV file.")
                st.stop()
            nrows = int(max_rows) if max_rows and max_rows > 0 else None
            data = pd.read_csv(uploaded_file, nrows=nrows)

    if data.empty:
        st.error("No data loaded. Check the URL or uploaded file.")
        st.stop()

    if level.startswith("Title") and not category_col:
        category_col = "title"

    monthly = build_monthly_series(
        data,
        date_col=date_col or None,
        year_col=year_col or None,
        month_col=month_col or None,
        category_col=category_col or None,
        count_col=count_col or None,
        book_col=book_col or None,
        only_books=True,
    )
    monthly = complete_monthly_index(monthly)
    total_titles = int(monthly["category"].nunique())
    total_months = int(monthly["month"].nunique())
    if exclude_latest:
        latest = monthly["month"].max()
        monthly = monthly[monthly["month"] < latest]
    months_after_exclude = int(monthly["month"].nunique())
    base_after_exclude = monthly.copy()

    effective_max_items = int(max_items)
    effective_min_history = int(min_history_months)
    effective_recent_window = min(int(recent_window), max(months_after_exclude, 1))
    effective_min_recent_nonzero = int(min_recent_nonzero)
    auto_relaxed = False

    monthly = _filter_top_items(base_after_exclude, effective_max_items)
    monthly = _filter_sparse_items(
        monthly,
        min_history=effective_min_history,
        recent_window_value=effective_recent_window,
        min_recent_nonzero_value=effective_min_recent_nonzero,
    )

    if monthly.empty:
        auto_relaxed = True
        effective_max_items = 0
        effective_min_history = min(3, max(months_after_exclude, 1))
        effective_recent_window = min(6, max(months_after_exclude, 1))
        effective_min_recent_nonzero = 1
        monthly = _filter_top_items(base_after_exclude, effective_max_items)
        monthly = _filter_sparse_items(
            monthly,
            min_history=effective_min_history,
            recent_window_value=effective_recent_window,
            min_recent_nonzero_value=effective_min_recent_nonzero,
        )

    if monthly.empty:
        st.error("No data left after filtering.")
        st.caption(
            f"Available months: {total_months} (after excluding latest: {months_after_exclude}). "
            f"Try: min history ≤ {months_after_exclude}, recent window ≤ {months_after_exclude}, "
            "min nonzero = 0 or 1, increase max rows, or disable exclude latest month."
        )
        st.stop()

    if not models:
        st.error("Select at least one model.")
        st.stop()

    forecasts = _run_models(monthly, models)
    if forecasts.empty:
        st.error("No forecasts were generated. Check data columns and history length.")
        st.stop()
    recent_avg = _recent_average(monthly, int(effective_recent_window))
    overall_avg = monthly.groupby("category")["count"].mean()
    if apply_guardrails:
        forecasts = _apply_guardrails_to_forecasts(forecasts, recent_avg, float(guardrail_multiplier))
    forecasts, zero_fixed_models = _apply_fallback_if_all_zero(forecasts, recent_avg, overall_avg)

    primary_model = st.selectbox("Primary model for charts", models)
    predicted_df = forecasts[forecasts["model"] == primary_model].copy()

    last_month = monthly["month"].max()
    last_month_label = last_month.strftime("%Y-%m")
    next_month_label = (last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    last_month_counts = monthly[monthly["month"] == last_month].set_index("category")["count"]
    last_month_total = float(last_month_counts.sum())

    predicted_df["last_month"] = predicted_df["category"].map(last_month_counts).fillna(0)
    predicted_df["delta"] = predicted_df["predicted"] - predicted_df["last_month"]

    predicted_total = float(predicted_df["predicted"].sum())
    delta_total = predicted_total - last_month_total
    pct_change = (delta_total / last_month_total * 100) if last_month_total > 0 else None

    st.subheader("Key Insights")
    if effective_max_items and int(effective_max_items) > 0:
        st.caption(f"Insights are based on the top {int(effective_max_items)} titles by historical demand.")
    if auto_relaxed:
        st.caption(
            f"Filters were auto-relaxed to min history {effective_min_history}, "
            f"recent window {effective_recent_window}, min nonzero {effective_min_recent_nonzero}."
        )
    if apply_guardrails:
        st.caption(
            f"Guardrails active: predictions capped at {guardrail_multiplier}x the recent average "
            f"({int(effective_recent_window)}-month window)."
        )
    if zero_fixed_models:
        st.caption(
            "Fallback applied for models with all-zero predictions: "
            + ", ".join(sorted(set(zero_fixed_models)))
            + "."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Last month total ({last_month_label})", _fmt(last_month_total))
    c2.metric(f"Predicted total ({next_month_label})", _fmt(predicted_total), _fmt(delta_total))
    c3.metric("Change vs last month", f"{pct_change:.1f}%" if pct_change is not None else "N/A")
    top10_share = (
        predicted_df.sort_values("predicted", ascending=False).head(10)["predicted"].sum() / predicted_total
        if predicted_total > 0
        else 0
    )
    c4.metric("Top 10 share", f"{top10_share * 100:.1f}%")

    st.subheader("Predicted Top Book Titles (Next Month)")
    top = predicted_df.sort_values("predicted", ascending=False).head(15).copy()
    st.dataframe(top[["category", "predicted", "last_month", "delta", "rank"]], width="stretch")
    st.bar_chart(top.set_index("category")["predicted"])

    st.subheader("Biggest Movers (vs Last Month)")
    movers_up = predicted_df.sort_values("delta", ascending=False).head(10)
    movers_down = predicted_df.sort_values("delta", ascending=True).head(10)
    m1, m2 = st.columns(2)
    m1.markdown("**Top Increases**")
    m1.dataframe(movers_up[["category", "predicted", "last_month", "delta"]], width="stretch")
    m2.markdown("**Top Decreases**")
    m2.dataframe(movers_down[["category", "predicted", "last_month", "delta"]], width="stretch")

    st.subheader("Historical Trend + Forecast (Selected Title)")
    item_choice = st.selectbox("Select title", top["category"].tolist())
    history = monthly[monthly["category"] == item_choice].sort_values("month")
    forecast_value = top[top["category"] == item_choice]["predicted"].iloc[0]
    next_month = history["month"].iloc[-1] + pd.offsets.MonthBegin(1)
    extended = pd.concat(
        [
            history[["month", "count"]].rename(columns={"count": "value"}),
            pd.DataFrame({"month": [next_month], "value": [forecast_value]}),
        ],
        ignore_index=True,
    )
    st.line_chart(extended.set_index("month")["value"])

    st.subheader("Seasonality (Average by Month)")
    totals = monthly.groupby("month")["count"].sum().reset_index()
    totals["month_num"] = totals["month"].dt.month
    seasonal = totals.groupby("month_num")["count"].mean().reindex(range(1, 13), fill_value=0)
    seasonal_df = pd.DataFrame({"month": MONTH_NAMES, "avg_checkouts": seasonal.values})
    st.bar_chart(seasonal_df.set_index("month")["avg_checkouts"])

    st.subheader("Total Books Borrowed Over Time")
    total_series = monthly.groupby("month")["count"].sum().sort_index()
    st.line_chart(total_series)

    st.subheader("All Forecasts")
    st.dataframe(forecasts, width="stretch")

    metrics = evaluate_models(monthly, models, backtest_months=int(backtest_months))
    st.subheader("Backtest Metrics")
    st.json(metrics)

    csv_data = forecasts.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast CSV", csv_data, "forecast_next_month.csv", "text/csv")
    metrics_json = json.dumps(metrics, indent=2).encode("utf-8")
    st.download_button("Download Metrics JSON", metrics_json, "metrics_report.json", "application/json")
