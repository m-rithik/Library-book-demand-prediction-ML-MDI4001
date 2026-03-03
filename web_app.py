from __future__ import annotations

import warnings
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import DEFAULT_DATA_URL
from src.evaluate import evaluate_models
from src.io import download_csv
from src.models.baselines import seasonal_naive_forecast
from src.models.holt_winters import holt_winters_forecast
from src.models.naive_bayes import naive_bayes_forecast
from src.models.regression import regression_forecast
from src.models.sarima import sarima_forecast
from src.preprocess import build_monthly_series, complete_monthly_index


warnings.filterwarnings("ignore", category=ConvergenceWarning)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MODEL_LABELS = {
    "holt":        "Holt-Winters",
    "regression":  "Regression",
    "sarima":      "SARIMA",
    "naive":       "Seasonal Naive",
    "naive_bayes": "Naive Bayes",
    "ensemble":    "Ensemble (avg)",
}

MODEL_DESCRIPTIONS = {
    "holt":        "Captures trend + seasonality using exponential smoothing. Great for stable patterns.",
    "regression":  "Fits a line through historical demand using calendar features and recent lags.",
    "sarima":      "Statistical model for time-series with seasonal cycles. Strong for regular patterns.",
    "naive":       "Predicts next month = same month last year. Simple, but a solid baseline.",
    "naive_bayes": "Classifies demand into bins using Gaussian Naive Bayes. Works well with noisy data.",
    "ensemble":    "Average of all selected models. Reduces the risk of any single model being wrong.",
}

C_BLUE  = "#2563eb"
C_GREEN = "#16a34a"
C_RED   = "#dc2626"
C_GRAY  = "#6b7280"


# ── Page header ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Library Book Demand Prediction", layout="wide")

st.title("📚 Library Book Demand Prediction")
st.markdown(
    "Predict which book titles will be **most borrowed next month** — "
    "so librarians can plan stock orders, manage hold queues, and allocate budgets before demand arrives."
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Data Source")
    source = st.radio("Choose data source", ["URL", "Upload CSV"], horizontal=True)
    data_url = None
    uploaded_file = None
    if source == "URL":
        data_url = st.text_input(
            "CSV URL",
            value=DEFAULT_DATA_URL,
            help="Paste a direct link to a CSV file. Supports Seattle Public Library's Socrata API automatically.",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Upload your own checkout data. Must contain a date column, a title/category column, and a checkout count column.",
        )

    st.subheader("What to Predict")
    level = st.radio(
        "Predict by",
        ["Title (Book names)", "Category"],
        index=0,
        help="'Title' predicts individual book names. 'Category' predicts broader groups like Fiction, Non-Fiction, Children's, etc.",
    )

    st.subheader("Column Names")
    st.caption("Leave blank — the app will auto-detect columns from your CSV headers.")
    date_mode = st.radio(
        "How is the date stored?",
        ["Year + Month columns", "Single date column"],
        help="Seattle Public Library data uses separate Year and Month columns.",
    )
    group_col_label = "Title column" if level.startswith("Title") else "Category column"
    default_group_col = "title" if level.startswith("Title") else ""
    category_col = st.text_input(
        f"{group_col_label} (optional)",
        value=default_group_col,
        help="The column that contains book titles or category names. Default 'title' works for SPL data.",
    )
    count_col = st.text_input(
        "Checkout count column (optional)",
        value="",
        help="The column with the number of checkouts per row. Auto-detects 'checkouts', 'count', etc.",
    )
    book_col = st.text_input(
        "Book filter column (optional)",
        value="",
        help="Column used to filter rows to books only (e.g. materialtype). Auto-detected when blank.",
    )
    date_col = year_col = month_col = ""
    if date_mode == "Year + Month columns":
        year_col  = st.text_input("Year column (optional)",  value="",
                                  help="e.g. 'checkoutyear' in SPL data")
        month_col = st.text_input("Month column (optional)", value="",
                                  help="e.g. 'checkoutmonth' in SPL data")
    else:
        date_col = st.text_input("Date column (optional)", value="",
                                 help="A single column with a full date string like '2024-03-01'")

    st.subheader("Forecast Models")
    models = st.multiselect(
        "Models to run",
        ["holt", "regression", "sarima", "naive", "naive_bayes"],
        default=["holt", "regression", "sarima"],
        format_func=lambda m: MODEL_LABELS.get(m, m),
        help="Select one or more forecasting models. An Ensemble (average) is computed automatically when more than one is selected.",
    )
    backtest_months = st.number_input(
        "Backtest months",
        min_value=0, max_value=12, value=3, step=1,
        help="How many past months to use for testing model accuracy. Set to 0 to skip the accuracy check.",
    )

    st.subheader("Performance")
    max_rows = st.number_input(
        "Max rows to load (0 = all)",
        min_value=0, max_value=500000, value=200000, step=10000,
        help="Limit the number of rows fetched. Reduce this if loading is slow.",
    )
    max_items = st.number_input(
        "Max titles to forecast",
        min_value=0, max_value=200, value=30, step=5,
        help="Only forecast the top N titles by total historical checkouts. Increase for broader coverage.",
    )

    st.subheader("Data Quality Filters")
    exclude_latest = st.checkbox(
        "Exclude latest month",
        value=True,
        help="Recommended: the most recent month in the dataset is often incomplete (partial data). Excluding it avoids underestimates.",
    )
    min_history_months = st.number_input(
        "Min months of history required",
        min_value=2, max_value=36, value=3, step=1,
        help="Titles with fewer than this many months of data are excluded — too little history to forecast reliably.",
    )
    recent_window = st.number_input(
        "Recent activity window (months)",
        min_value=3, max_value=24, value=6, step=1,
        help="How many recent months to check for active checkouts. Titles inactive in this window are filtered out.",
    )
    min_recent_nonzero = st.number_input(
        "Min active months in window",
        min_value=0, max_value=24, value=1, step=1,
        help="Title must have checkouts in at least this many months within the recent window.",
    )
    apply_guardrails = st.checkbox(
        "Cap extreme predictions",
        value=True,
        help="Prevents unrealistically high forecasts by capping predictions at a multiple of the recent average.",
    )
    guardrail_multiplier = st.number_input(
        "Cap at (× recent average)",
        min_value=1.5, max_value=10.0, value=3.0, step=0.5,
        help="e.g. 3.0 means a title predicted at more than 3× its recent average gets capped.",
    )


# ── Core pipeline helpers ─────────────────────────────────────────────────────

def _run_models(monthly: pd.DataFrame, model_list: List[str]) -> pd.DataFrame:
    forecasts = []
    for name in model_list:
        if   name == "naive":       df = seasonal_naive_forecast(monthly)
        elif name == "holt":        df = holt_winters_forecast(monthly)
        elif name == "sarima":      df = sarima_forecast(monthly)
        elif name == "regression":  df = regression_forecast(monthly)
        elif name == "naive_bayes": df = naive_bayes_forecast(monthly)
        else: continue
        df["model"] = name
        forecasts.append(df)
    if not forecasts:
        return pd.DataFrame()
    combined = pd.concat(forecasts, ignore_index=True)
    combined["predicted"] = combined["predicted"].clip(lower=0)
    if len(forecasts) > 1:
        ens = combined.groupby("category")["predicted"].mean().reset_index()
        ens["model"] = "ensemble"
        combined = pd.concat([combined, ens], ignore_index=True)
    combined = combined.sort_values(["model", "predicted"], ascending=[True, False])
    combined["rank"] = (
        combined.groupby("model")["predicted"]
        .rank(method="first", ascending=False).astype(int)
    )
    return combined


def _filter_top_items(monthly: pd.DataFrame, n: int) -> pd.DataFrame:
    if not n or n <= 0:
        return monthly
    top = monthly.groupby("category")["count"].sum().nlargest(n).index
    return monthly[monthly["category"].isin(top)].copy()


def _filter_sparse_items(
    monthly: pd.DataFrame, min_history: int, window: int, min_nonzero: int,
) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    keep = monthly.groupby("category")["month"].nunique()
    keep = keep[keep >= min_history].index
    filtered = monthly[monthly["category"].isin(keep)].copy()
    if min_nonzero > 0 and not filtered.empty:
        filtered = filtered.sort_values("month")
        recent = filtered.groupby("category").tail(window)
        nz = recent[recent["count"] > 0].groupby("category").size()
        keep2 = nz[nz >= min_nonzero].index
        filtered = filtered[filtered["category"].isin(keep2)].copy()
    return filtered


def _recent_average(monthly: pd.DataFrame, window: int) -> pd.Series:
    return (
        monthly.sort_values("month")
        .groupby("category").tail(window)
        .groupby("category")["count"].mean()
    )


def _apply_guardrails(forecasts: pd.DataFrame, recent_avg: pd.Series, mult: float) -> pd.DataFrame:
    if forecasts.empty or recent_avg.empty:
        return forecasts
    m = forecasts.merge(recent_avg.rename("ra"), on="category", how="left")
    mask = m["ra"] > 0
    m.loc[mask, "predicted"] = np.minimum(m.loc[mask, "predicted"], m.loc[mask, "ra"] * mult)
    return m.drop(columns=["ra"])


def _fallback_zeros(
    forecasts: pd.DataFrame, recent_avg: pd.Series, overall_avg: pd.Series,
) -> tuple[pd.DataFrame, List[str]]:
    if forecasts.empty:
        return forecasts, []
    affected: List[str] = []

    def _fix(g: pd.DataFrame) -> pd.DataFrame:
        if g["predicted"].sum() > 0:
            return g
        affected.append(g["model"].iloc[0])
        g = g.merge(recent_avg.rename("ra"), on="category", how="left")
        g = g.merge(overall_avg.rename("oa"), on="category", how="left")
        g["predicted"] = g["predicted"].where(g["predicted"] > 0, g["ra"])
        g["predicted"] = g["predicted"].where(g["predicted"] > 0, g["oa"])
        g["predicted"] = g["predicted"].fillna(0)
        return g.drop(columns=["ra", "oa"])

    return forecasts.groupby("model", group_keys=False).apply(_fix), affected


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


def _trunc(s: str, n: int = 48) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _chart_top_demand(df: pd.DataFrame, item_label: str) -> alt.Chart:
    d = df.copy()
    d["label"] = d["category"].apply(lambda s: _trunc(str(s), 50))
    d["delta_sign"] = d["delta"].apply(lambda v: f"+{_fmt(v)}" if v >= 0 else _fmt(v))

    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("predicted:Q", title="Predicted Checkouts", axis=alt.Axis(format=",.0f")),
            y=alt.Y("label:N", sort=alt.SortField("predicted", order="descending"),
                    title=None, axis=alt.Axis(labelLimit=380, labelFontSize=12)),
            color=alt.Color("predicted:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=[
                alt.Tooltip("category:N", title=item_label),
                alt.Tooltip("predicted:Q", title="Predicted Checkouts", format=",.0f"),
                alt.Tooltip("last_month:Q", title="Last Month Actual", format=",.0f"),
                alt.Tooltip("delta_sign:N", title="Change vs Last Month"),
                alt.Tooltip("rank:Q", title="Rank"),
            ],
        )
    )
    text = bars.mark_text(align="left", dx=5, fontSize=11, color="#374151").encode(
        text=alt.Text("predicted:Q", format=",.0f")
    )
    return (bars + text).properties(height=max(280, len(d) * 30))


def _chart_movers(df: pd.DataFrame, item_label: str) -> alt.Chart:
    d = df.copy()
    d["label"] = d["category"].apply(lambda s: _trunc(str(s), 45))
    d["direction"] = d["delta"].apply(lambda v: "Rising" if v >= 0 else "Falling")

    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("delta:Q", title="Change in Checkouts vs Last Month",
                    axis=alt.Axis(format="+,.0f")),
            y=alt.Y("label:N", sort=alt.SortField("delta", order="descending"),
                    title=None, axis=alt.Axis(labelLimit=320, labelFontSize=11)),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Rising", "Falling"], range=[C_GREEN, C_RED]),
                legend=alt.Legend(title="Trend", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("category:N", title=item_label),
                alt.Tooltip("predicted:Q", title="Predicted", format=",.0f"),
                alt.Tooltip("last_month:Q", title="Last Month", format=",.0f"),
                alt.Tooltip("delta:Q", title="Change", format="+,.0f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="#374151", strokeWidth=1.5)
        .encode(x="x:Q")
    )
    return (bars + rule).properties(height=max(200, len(d) * 28))


def _chart_history_forecast(
    history: pd.DataFrame, forecast_val: float,
    next_month_dt: pd.Timestamp, item_name: str,
) -> alt.Chart:
    hist = history[["month", "count"]].copy()
    last_date = hist["month"].iloc[-1]
    last_val  = float(hist["count"].iloc[-1])

    base = alt.Chart(hist).encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("count:Q", title="Checkouts", axis=alt.Axis(format=",.0f")),
    )
    hist_line = base.mark_line(color=C_BLUE, strokeWidth=2.5)
    hist_dots = base.mark_circle(color=C_BLUE, size=35, opacity=0.6).encode(
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("count:Q", title="Checkouts", format=",.0f"),
        ]
    )

    seg = pd.DataFrame({"month": [last_date, next_month_dt], "val": [last_val, forecast_val]})
    bridge = (
        alt.Chart(seg)
        .mark_line(strokeDash=[6, 4], color=C_RED, strokeWidth=2.2)
        .encode(x="month:T", y="val:Q")
    )

    fc_df = pd.DataFrame({
        "month": [next_month_dt], "val": [forecast_val],
        "lbl":   [f"Forecast: {_fmt(forecast_val)}"],
    })
    fc_dot = (
        alt.Chart(fc_df).mark_circle(size=160, color=C_RED)
        .encode(
            x="month:T", y="val:Q",
            tooltip=[
                alt.Tooltip("month:T", title="Forecast Month", format="%b %Y"),
                alt.Tooltip("val:Q", title="Predicted Checkouts", format=",.0f"),
            ],
        )
    )
    fc_label = (
        alt.Chart(fc_df)
        .mark_text(align="left", dx=10, dy=-10, fontSize=12, color=C_RED, fontWeight="bold")
        .encode(x="month:T", y="val:Q", text="lbl:N")
    )

    return (hist_line + hist_dots + bridge + fc_dot + fc_label).properties(
        height=300,
        title=alt.Title(
            text=_trunc(item_name, 60),
            subtitle="Blue line = past checkouts   |   Red dashed + dot = next month forecast",
            fontSize=14,
        ),
    )


def _chart_seasonality(seasonal_df: pd.DataFrame) -> alt.Chart:
    d = seasonal_df.copy()
    peak_thresh = d["Avg Checkouts"].quantile(0.67)
    d["Season"] = d["Avg Checkouts"].apply(lambda v: "Peak" if v >= peak_thresh else "Normal")

    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Month:O", sort=MONTH_NAMES, title=None,
                    axis=alt.Axis(labelFontSize=12)),
            y=alt.Y("Avg Checkouts:Q", title="Avg Checkouts",
                    axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "Season:N",
                scale=alt.Scale(domain=["Peak", "Normal"], range=[C_RED, C_BLUE]),
                legend=alt.Legend(title="Season"),
            ),
            tooltip=[
                alt.Tooltip("Month:O"),
                alt.Tooltip("Avg Checkouts:Q", format=",.0f"),
                alt.Tooltip("Season:N"),
            ],
        )
    )
    text = bars.mark_text(dy=-7, fontSize=10, color="#374151").encode(
        text=alt.Text("Avg Checkouts:Q", format=",.0f")
    )
    return (bars + text).properties(
        height=250,
        title=alt.Title(
            text="Seasonal Demand by Month",
            subtitle="Red = peak months (top third of average demand). Forecasts automatically adjust for these patterns.",
            fontSize=14,
        ),
    )


def _chart_overall_trend(trend_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(trend_df)
        .mark_area(opacity=0.25, color=C_BLUE, line={"color": C_BLUE, "strokeWidth": 2})
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("count:Q", title="Total Checkouts", axis=alt.Axis(format=",.0f")),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("count:Q", title="Total Checkouts", format=",.0f"),
            ],
        )
        .properties(
            height=250,
            title=alt.Title(
                text="Library Borrowing Trend Over Time",
                subtitle="Total checkouts across all tracked titles per month",
                fontSize=14,
            ),
        )
    )


def _chart_model_accuracy(metrics: dict) -> alt.Chart:
    rows = [
        {"Model": MODEL_LABELS.get(m, m),
         "MAPE (%)": round(v["mape"] * 100, 1),
         "MAE": round(v["mae"], 1)}
        for m, v in metrics.items()
    ]
    df = pd.DataFrame(rows).sort_values("MAPE (%)")
    best = df["MAPE (%)"].min()
    df["tag"] = df["MAPE (%)"].apply(lambda v: "Best" if v == best else "Other")

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("MAPE (%):Q",
                    title="MAPE — Mean Absolute % Error (lower = more accurate)",
                    axis=alt.Axis(format=".1f")),
            y=alt.Y("Model:N", sort=alt.SortField("MAPE (%)", order="ascending"),
                    title=None, axis=alt.Axis(labelFontSize=12)),
            color=alt.Color(
                "tag:N",
                scale=alt.Scale(domain=["Best", "Other"], range=[C_GREEN, C_BLUE]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("MAPE (%):Q", format=".1f",
                            title="MAPE — avg % off per prediction"),
                alt.Tooltip("MAE:Q", format=".1f",
                            title="MAE — avg checkouts off per prediction"),
            ],
        )
    )
    text = bars.mark_text(align="left", dx=4, fontSize=11, color="#374151").encode(
        text=alt.Text("MAPE (%):Q", format=".1f")
    )
    return (bars + text).properties(
        height=max(140, len(df) * 38),
        title=alt.Title(
            text="Which model was most accurate? (Backtest)",
            subtitle="Tested against actual past months. Green = best performer.",
            fontSize=14,
        ),
    )


# ── Pre-run intro ─────────────────────────────────────────────────────────────

st.info(
    "**How to use this app:**\n\n"
    "1. The sidebar is pre-configured for the Seattle Public Library dataset — just press **Run Forecast**.\n"
    "2. Results will show the top predicted titles, what's rising/falling, and actionable planning steps.\n"
    "3. Adjust **Max titles to forecast** (sidebar) to see more or fewer titles.",
    icon="ℹ️",
)

if st.button("Run Forecast", type="primary"):

    # ── Load ──────────────────────────────────────────────────────────────────
    with st.spinner("Fetching data…"):
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
        st.error("No data loaded. Check the URL or file.")
        st.stop()

    if level.startswith("Title") and not category_col:
        category_col = "title"

    # ── Preprocess ────────────────────────────────────────────────────────────
    with st.spinner("Processing checkout records…"):
        monthly = build_monthly_series(
            data,
            date_col=date_col or None, year_col=year_col or None,
            month_col=month_col or None, category_col=category_col or None,
            count_col=count_col or None, book_col=book_col or None,
            only_books=True,
        )
        monthly = complete_monthly_index(monthly)

    total_titles = int(monthly["category"].nunique())
    total_months = int(monthly["month"].nunique())
    date_min     = monthly["month"].min().strftime("%b %Y")
    date_max     = monthly["month"].max().strftime("%b %Y")

    if exclude_latest:
        monthly = monthly[monthly["month"] < monthly["month"].max()]
    months_after = int(monthly["month"].nunique())
    base = monthly.copy()

    eff_max  = int(max_items)
    eff_hist = int(min_history_months)
    eff_win  = min(int(recent_window), max(months_after, 1))
    eff_nz   = int(min_recent_nonzero)
    relaxed  = False

    monthly = _filter_top_items(base, eff_max)
    monthly = _filter_sparse_items(monthly, eff_hist, eff_win, eff_nz)

    # Auto-relax if too few titles survive (< 5 titles is not useful for a top-10 list)
    if monthly["category"].nunique() < 5:
        relaxed = True
        eff_max  = 0                              # no cap — use all available titles
        eff_hist = min(2, max(months_after, 1))   # only 2 months minimum
        eff_win  = min(3, max(months_after, 1))   # shorter recent window
        eff_nz   = 0                              # don't require recent activity
        monthly  = _filter_top_items(base, eff_max)
        monthly  = _filter_sparse_items(monthly, eff_hist, eff_win, eff_nz)

    if monthly.empty:
        st.error("No titles remained after filtering.")
        st.caption(
            f"Available months: {total_months} (after excluding latest: {months_after}). "
            "Try lowering 'Min months of history', 'Min active months', or disabling 'Exclude latest month'."
        )
        st.stop()

    if not models:
        st.error("Select at least one model in the sidebar.")
        st.stop()

    # ── Run models ────────────────────────────────────────────────────────────
    with st.spinner("Running forecasting models…"):
        forecasts = _run_models(monthly, models)

    if forecasts.empty:
        st.error("No forecasts were generated. Check data columns and history length.")
        st.stop()

    recent_avg  = _recent_average(monthly, eff_win)
    overall_avg = monthly.groupby("category")["count"].mean()
    if apply_guardrails:
        forecasts = _apply_guardrails(forecasts, recent_avg, float(guardrail_multiplier))
    forecasts, zero_models = _fallback_zeros(forecasts, recent_avg, overall_avg)

    last_month  = monthly["month"].max()
    last_lbl    = last_month.strftime("%b %Y")
    next_lbl    = (last_month + pd.offsets.MonthBegin(1)).strftime("%b %Y")
    last_counts = monthly[monthly["month"] == last_month].set_index("category")["count"]
    last_total  = float(last_counts.sum())
    item_label  = "Title" if level.startswith("Title") else "Category"
    hist_avg    = monthly.groupby("category")["count"].mean()
    n_modelled  = int(monthly["category"].nunique())

    # ── Data summary banner ───────────────────────────────────────────────────
    st.divider()
    st.success(
        f"**Data loaded successfully.** "
        f"Found **{total_titles:,} unique {item_label.lower()}s** across **{total_months} months** "
        f"({date_min} – {date_max}). "
        f"Forecasting the top **{n_modelled}** most active {item_label.lower()}s for **{next_lbl}**."
    )
    if relaxed:
        st.warning(
            f"**Filters were automatically relaxed** — the loaded data only covers "
            f"**{months_after} month(s)**, so the original 'min history' setting was too strict. "
            f"Now using: min history = {eff_hist} month(s), no recent-activity requirement. "
            f"**To see more titles:** increase 'Max rows to load' in the sidebar, or lower "
            "'Min months of history required' manually.",
            icon="⚠️",
        )
    if zero_models:
        st.caption(
            "Note: Some models produced all-zero predictions and fell back to the recent average — "
            + ", ".join(MODEL_LABELS.get(m, m) for m in zero_models) + "."
        )

    # ── Model picker ──────────────────────────────────────────────────────────
    avail  = forecasts["model"].unique().tolist()
    disp   = [MODEL_LABELS.get(m, m) for m in avail]
    def_m  = "ensemble" if "ensemble" in avail else avail[0]
    def_i  = avail.index(def_m) if def_m in avail else 0

    pick_col, desc_col = st.columns([2, 3])
    with pick_col:
        chosen_label = st.selectbox(
            "Which model's predictions to display?",
            disp, index=def_i,
            help="Ensemble averages all models together — usually the safest choice.",
        )
    primary = avail[disp.index(chosen_label)]
    with desc_col:
        st.caption(f"**{MODEL_LABELS.get(primary, primary)}** — {MODEL_DESCRIPTIONS.get(primary, '')}")

    pred_df = forecasts[forecasts["model"] == primary].copy()
    pred_df["last_month"] = pred_df["category"].map(last_counts).fillna(0)
    pred_df["delta"]      = pred_df["predicted"] - pred_df["last_month"]
    pred_df["hist_avg"]   = pred_df["category"].map(hist_avg).fillna(0)
    pred_df["vs_avg_pct"] = (
        (pred_df["predicted"] - pred_df["hist_avg"]) / pred_df["hist_avg"].clip(lower=1) * 100
    ).round(1)

    pred_total  = float(pred_df["predicted"].sum())
    delta_total = pred_total - last_total
    pct_change  = (delta_total / last_total * 100) if last_total > 0 else None
    top10_share = (
        pred_df.nlargest(10, "predicted")["predicted"].sum() / pred_total
        if pred_total > 0 else 0
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # KEY METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📊 Forecast at a Glance")
    st.caption(
        f"Comparing **{last_lbl}** (last known month) to the **{next_lbl}** forecast "
        f"across the top {n_modelled} tracked {item_label.lower()}s."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"Total Checkouts — {last_lbl}",
        _fmt(last_total),
        help="Actual total checkouts recorded in the most recent complete month.",
    )
    k2.metric(
        f"Predicted Total — {next_lbl}",
        _fmt(pred_total),
        f"{'+' if delta_total >= 0 else ''}{_fmt(delta_total)}",
        help="Sum of all individual title forecasts for next month. The delta shows the change vs last month.",
    )
    k3.metric(
        "Overall Change",
        f"{'+' if (pct_change or 0) >= 0 else ''}{pct_change:.1f}%" if pct_change is not None else "N/A",
        help="Percentage change in total predicted checkouts compared to last month.",
    )
    k4.metric(
        "Top-10 Demand Share",
        f"{top10_share * 100:.1f}%",
        help="The top 10 predicted titles account for this share of all predicted checkouts. High % = demand is concentrated in a few titles.",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — TOP 10 PREDICTED DEMAND
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(f"🏆 Top 10 Most In-Demand {item_label}s — {next_lbl}")
    st.markdown(
        f"The **10 {item_label.lower()}s most likely to be borrowed next month**, ranked by predicted checkouts. "
        "Use this list to **prioritise purchase orders, licence renewals, and hold queue slots** before demand arrives."
    )

    top10 = pred_df.nlargest(10, "predicted").copy().reset_index(drop=True)
    top   = pred_df.nlargest(15, "predicted").copy()  # used by deep-dive picker below

    n_top10 = len(top10)
    if n_top10 < 10:
        st.warning(
            f"Only **{n_top10} title(s)** had enough data to forecast — showing all of them below. "
            "To get more titles: **increase 'Max rows to load'** (sidebar) or **lower 'Min months of history required'**.",
            icon="⚠️",
        )

    MEDALS = {0: "🥇", 1: "🥈", 2: "🥉"}

    list_col, chart_col = st.columns([1, 1], gap="large")

    # ── Numbered list ─────────────────────────────────────────────────────────
    with list_col:
        st.markdown("##### Ranked List")
        for i, row in top10.iterrows():
            medal  = MEDALS.get(i, f"**{i + 1}.**")
            delta  = row["delta"]
            arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
            d_clr  = "green" if delta > 0 else ("red" if delta < 0 else "gray")
            d_str  = f"+{_fmt(delta)}" if delta > 0 else _fmt(delta)
            pct    = row["vs_avg_pct"]
            p_str  = f"{'+' if pct >= 0 else ''}{pct:.0f}% vs avg"

            st.markdown(
                f"{medal} &nbsp; **{_trunc(str(row['category']), 44)}**  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
                f"Predicted: **{_fmt(row['predicted'])}** checkouts &nbsp;|&nbsp; "
                f":{d_clr}[{arrow} {d_str}] &nbsp;|&nbsp; {p_str}"
            )
            if i < 9:
                st.markdown("<hr style='margin:4px 0; border-color:#e5e7eb'>", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    with chart_col:
        st.markdown("##### Demand Chart (Top 10)")
        st.altair_chart(_chart_top_demand(top10, item_label), use_container_width=True)

    # ── Full leaderboard table (collapsed by default) ─────────────────────────
    with st.expander("📋 Full prediction table (top 15) + column guide", expanded=False):
        st.markdown(
            "| Column | What it means |\n"
            "|---|---|\n"
            "| **Rank** | Position by predicted demand (1 = highest) |\n"
            f"| **{item_label}** | Book title or category name |\n"
            "| **Predicted Checkouts** | Estimated borrows next month (progress bar = share of the #1 title) |\n"
            "| **Last Month** | Actual checkouts in the most recent complete month |\n"
            "| **Change vs Last Month** | Predicted increase (+) or decrease (−) |\n"
            "| **vs Historical Avg** | +50% means 50% above this title's long-run monthly average |\n"
        )
        table_df = top[["rank", "category", "predicted", "last_month", "delta", "vs_avg_pct"]].copy()
        table_df = table_df.rename(columns={
            "rank": "Rank", "category": item_label, "predicted": "Predicted",
            "last_month": "Last Month", "delta": "Change", "vs_avg_pct": "vs Avg (%)",
        })
        st.dataframe(
            table_df,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                item_label: st.column_config.TextColumn(item_label, width="large"),
                "Predicted": st.column_config.ProgressColumn(
                    "Predicted Checkouts",
                    min_value=0,
                    max_value=float(top["predicted"].max()) * 1.1,
                    format="%d",
                ),
                "Last Month": st.column_config.NumberColumn("Last Month (actual)", format="%d"),
                "Change": st.column_config.NumberColumn("Change vs Last Month", format="%+d"),
                "vs Avg (%)": st.column_config.NumberColumn(
                    "vs Historical Avg", format="%+.1f%%",
                    help="Positive = above this title's long-run average.",
                ),
            },
            hide_index=True,
        )
        st.caption("Click any column header to sort. Hover a row for tooltips.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — DEMAND MOVERS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📈 Demand Movers — What's Rising & Falling")
    st.markdown(
        "These titles show the **biggest shifts in predicted demand** compared to last month. "
        "Green bars extend to the right (more checkouts expected); red bars to the left (fewer expected). "
        "The centre line represents **no change**."
    )

    rising  = pred_df[pred_df["delta"] > 0].nlargest(12, "delta")
    falling = pred_df[pred_df["delta"] < 0].nsmallest(12, "delta")
    movers  = pd.concat([rising, falling]).copy()

    if not movers.empty:
        st.altair_chart(_chart_movers(movers, item_label), use_container_width=True)

        mv1, mv2 = st.columns(2)
        with mv1:
            st.markdown(f"##### Rising {item_label}s — act before demand peaks")
            if rising.empty:
                st.caption("No titles with predicted increases.")
            else:
                for _, r in rising.head(8).iterrows():
                    pct = r["vs_avg_pct"]
                    badge = f"  *(+{pct:.0f}% vs avg)*" if pct > 0 else ""
                    st.markdown(
                        f"**#{int(r['rank'])}** {_trunc(str(r['category']), 50)} "
                        f"— **+{_fmt(r['delta'])}** checkouts{badge}"
                    )
        with mv2:
            st.markdown(f"##### Falling {item_label}s — safe to reduce orders")
            if falling.empty:
                st.caption("No titles with predicted decreases.")
            else:
                for _, r in falling.head(8).iterrows():
                    st.markdown(
                        f"**#{int(r['rank'])}** {_trunc(str(r['category']), 50)} "
                        f"— **{_fmt(r['delta'])}** checkouts "
                        f"*(predicted: {_fmt(r['predicted'])})*"
                    )
    else:
        st.info("Not enough variation in the data to identify movers.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — TITLE DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(f"🔍 {item_label} Deep Dive — Historical Trend & Forecast")
    st.caption(
        f"Select any {item_label.lower()} from the top predictions to see its full checkout history "
        "and exactly where the forecast sits relative to past performance."
    )

    item_options = top["category"].tolist()
    item_choice  = st.selectbox(
        f"Select a {item_label.lower()} to inspect",
        item_options,
        help="Only the top predicted titles are listed here. Change 'Max titles to forecast' in the sidebar to see more.",
    )

    history = monthly[monthly["category"] == item_choice].sort_values("month")
    fc_val  = float(top[top["category"] == item_choice]["predicted"].iloc[0])
    next_dt = history["month"].iloc[-1] + pd.offsets.MonthBegin(1)

    st.altair_chart(
        _chart_history_forecast(history, fc_val, next_dt, item_choice),
        use_container_width=True,
    )

    # Stats below the chart
    row = pred_df[pred_df["category"] == item_choice].iloc[0]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        f"Predicted — {next_lbl}", _fmt(fc_val),
        help="Model's estimated checkouts for this title next month.",
    )
    s2.metric(
        f"Actual — {last_lbl}", _fmt(float(row["last_month"])),
        help="How many times this title was actually borrowed last month.",
    )
    s3.metric(
        "Month-on-Month Change",
        f"{'+' if row['delta'] >= 0 else ''}{_fmt(row['delta'])}",
        help="Difference between the forecast and last month's actual. Positive = demand expected to grow.",
    )
    s4.metric(
        "vs Long-Run Average",
        f"{'+' if row['vs_avg_pct'] >= 0 else ''}{row['vs_avg_pct']:.1f}%",
        help="Compares the forecast to this title's historical monthly average. +50% means 50% above its normal demand level.",
    )

    # How many months of data
    n_months = len(history)
    avg_val  = float(history["count"].mean())
    max_val  = float(history["count"].max())
    st.caption(
        f"Based on {n_months} months of history for this title. "
        f"Historical average: {_fmt(avg_val)} checkouts/month. "
        f"All-time peak: {_fmt(max_val)} checkouts/month."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — SEASONAL PATTERNS & OVERALL TREND
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📅 Seasonal Patterns & Long-Term Trend")
    st.caption(
        "Understanding seasonal peaks helps models make better forecasts — "
        "and helps you plan staffing, programming, and stock levels around the year."
    )

    col_s, col_t = st.columns(2)
    with col_s:
        totals   = monthly.groupby("month")["count"].sum().reset_index()
        seas_grp = totals.groupby(totals["month"].dt.month)["count"].mean()
        seas_val = seas_grp.reindex(range(1, 13), fill_value=0)
        seas_df  = pd.DataFrame({"Month": MONTH_NAMES, "Avg Checkouts": seas_val.values})

        peak_thresh2 = seas_val.quantile(0.67)
        peak_months  = [MONTH_NAMES[i - 1] for i in range(1, 13) if seas_val[i] >= peak_thresh2]
        st.altair_chart(_chart_seasonality(seas_df), use_container_width=True)
        if peak_months:
            st.caption(f"Peak season months: **{', '.join(peak_months)}** — plan extra stock for these periods.")

    with col_t:
        trend_df = monthly.groupby("month")["count"].sum().reset_index()
        trend_df.columns = ["month", "count"]
        st.altair_chart(_chart_overall_trend(trend_df), use_container_width=True)

        first_val = float(trend_df["count"].iloc[0])
        last_val2 = float(trend_df["count"].iloc[-1])
        overall_trend = "growing" if last_val2 > first_val else "declining"
        pct_trend = abs((last_val2 - first_val) / max(first_val, 1) * 100)
        st.caption(
            f"Overall borrowing is **{overall_trend}** by ~{pct_trend:.0f}% "
            f"from {date_min} to {last_month.strftime('%b %Y')}."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — LIBRARY PLANNING ACTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(f"✅ Library Planning Actions for {next_lbl}")
    st.markdown(
        "These are the **most important actions** to take based on the forecast. "
        "Each section is ranked by impact."
    )

    top5_demand  = pred_df.nlargest(5, "predicted")
    top5_rising  = pred_df[pred_df["delta"] > 0].nlargest(5, "delta")
    top5_falling = pred_df[pred_df["delta"] < 0].nsmallest(5, "delta")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("#### 📦 Order or License More Copies")
        st.caption(
            "These titles have the **highest predicted demand**. "
            "Ensure enough physical copies and/or digital licences are available before the month starts."
        )
        for i, (_, r) in enumerate(top5_demand.iterrows(), 1):
            ratio = r["predicted"] / max(r["last_month"], 1)
            urgency = "🔴 High" if ratio > 1.5 else ("🟡 Moderate" if ratio > 1.1 else "🟢 Stable")
            st.markdown(
                f"**{i}. {_trunc(str(r['category']), 42)}**  \n"
                f"Predicted: **{_fmt(r['predicted'])}** checkouts — {urgency}"
            )

    with a2:
        st.markdown("#### ⚠️ Watch Hold Queues")
        st.caption(
            "These titles show the **biggest predicted increases**. "
            "Proactively manage hold lists and consider rush orders or inter-branch transfers."
        )
        if top5_rising.empty:
            st.caption("No significant increases predicted this month.")
        else:
            for i, (_, r) in enumerate(top5_rising.iterrows(), 1):
                st.markdown(
                    f"**{i}. {_trunc(str(r['category']), 42)}**  \n"
                    f"Expected **+{_fmt(r['delta'])}** more than last month "
                    f"({_fmt(r['last_month'])} → {_fmt(r['predicted'])})"
                )

    with a3:
        st.markdown("#### 💰 Reallocate Budget")
        st.caption(
            "These titles show the **biggest predicted drops**. "
            "It is safe to reduce purchase orders and redirect that budget to higher-demand titles."
        )
        if top5_falling.empty:
            st.caption("No significant decreases predicted this month.")
        else:
            for i, (_, r) in enumerate(top5_falling.iterrows(), 1):
                st.markdown(
                    f"**{i}. {_trunc(str(r['category']), 42)}**  \n"
                    f"Expected **{_fmt(r['delta'])}** fewer than last month "
                    f"({_fmt(r['last_month'])} → {_fmt(r['predicted'])})"
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — MODEL ACCURACY
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🎯 Model Accuracy — How Reliable are These Forecasts?")
    st.markdown(
        f"Each model was tested in a **backtest**: it was given data up to {backtest_months} month(s) ago, "
        "asked to predict the following month(s), and its predictions were compared to what actually happened. "
        "This tells you how much to trust the forecasts."
    )

    if int(backtest_months) > 0:
        with st.spinner("Running accuracy backtest…"):
            metrics = evaluate_models(monthly, models, backtest_months=int(backtest_months))

        if metrics:
            st.altair_chart(_chart_model_accuracy(metrics), use_container_width=True)

            with st.expander("📋 Full accuracy table + what these numbers mean", expanded=False):
                rows = [
                    {
                        "Model":   MODEL_LABELS.get(m, m),
                        "MAE":     f"{v['mae']:.1f}",
                        "RMSE":    f"{v['rmse']:.1f}",
                        "MAPE":    f"{v['mape']*100:.1f}%",
                        "Samples": int(v.get("samples", 0)),
                    }
                    for m, v in metrics.items()
                ]
                st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
                st.markdown(
                    "**How to read these numbers:**\n\n"
                    "- **MAE (Mean Absolute Error)** — On average, the model was off by this many checkouts per title. "
                    "Lower is better. e.g. MAE = 5 means predictions were ~5 checkouts away from actuals.\n"
                    "- **RMSE (Root Mean Square Error)** — Similar to MAE but penalises large errors more. "
                    "Useful for spotting models that occasionally make big mistakes.\n"
                    "- **MAPE (Mean Absolute % Error)** — Average % error per prediction. "
                    "e.g. 20% MAPE means predictions were ~20% off on average. Lower = more accurate.\n"
                    "- **Samples** — How many title-month combinations were evaluated."
                )

            # Plain English verdict
            best_model = min(metrics, key=lambda m: metrics[m]["mape"])
            best_mape  = metrics[best_model]["mape"] * 100
            if best_mape < 20:
                verdict = f"**{MODEL_LABELS.get(best_model, best_model)}** had the best accuracy at {best_mape:.1f}% average error — these forecasts are reasonably reliable."
            elif best_mape < 50:
                verdict = f"**{MODEL_LABELS.get(best_model, best_model)}** was the most accurate at {best_mape:.1f}% average error — use forecasts as a directional guide rather than exact figures."
            else:
                verdict = f"All models had high error rates ({best_mape:.1f}%+ MAPE). Treat forecasts as rough estimates. Consider adding more historical data."
            st.info(f"**Verdict:** {verdict}", icon="🎯")

        else:
            st.info(
                "Not enough data to run a backtest. "
                "Try reducing 'Backtest months' in the sidebar or loading more historical data."
            )
    else:
        st.info("Backtest is disabled (set to 0 months in the sidebar).")

    # ═══════════════════════════════════════════════════════════════════════════
    # DOWNLOADS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("⬇️ Download Results")
    st.caption("Export the full forecast data for use in spreadsheets, reports, or acquisition systems.")

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download Forecast CSV",
        forecasts.to_csv(index=False).encode("utf-8"),
        "forecast_next_month.csv",
        "text/csv",
        help="All model predictions for every title, in a flat CSV file.",
    )

    with st.expander("🔍 View raw forecast data (all models)"):
        st.caption(
            "This table shows predictions from every model for every title. "
            "Use the main results above for the recommended view."
        )
        st.dataframe(forecasts, use_container_width=True)
