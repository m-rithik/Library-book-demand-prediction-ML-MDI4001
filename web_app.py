from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import DEFAULT_DATA_URL
from src.evaluate import evaluate_models
from src.io import download_csv
from src.models.baselines import seasonal_naive_forecast
from src.models.gradient_boost import gradient_boost_forecast
from src.models.holt_winters import holt_winters_forecast
from src.models.naive_bayes import naive_bayes_forecast
from src.models.random_forest import random_forest_forecast
from src.models.regression import regression_forecast
from src.models.sarima import sarima_forecast
from src.preprocess import build_monthly_series, complete_monthly_index


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MODEL_LABELS = {
    "holt":           "Holt-Winters",
    "regression":     "Ridge Regression",
    "sarima":         "SARIMA",
    "naive":          "Seasonal Naive",
    "naive_bayes":    "Naive Bayes",
    "random_forest":  "Random Forest",
    "gradient_boost": "Gradient Boosting",
    "ensemble":       "Ensemble (avg)",
    "ensemble_w":     "Ensemble (weighted)",
}

MODEL_DESCRIPTIONS = {
    "holt":           "Captures trend + seasonality via exponential smoothing. Great for stable seasonal patterns.",
    "regression":     "Ridge-regularised linear model with lag, rolling average, and same-month-last-year features.",
    "sarima":         "Seasonal ARIMA for time-series with repeating annual cycles. Strong with 2+ years of data.",
    "naive":          "Predicts next month = same month last year. Simple but a solid baseline.",
    "naive_bayes":    "Global Gaussian NB classifier that bins demand across all titles. Robust to noise.",
    "random_forest":  "300-tree Random Forest trained across all titles with cyclic month + lag features.",
    "gradient_boost": "HistGradient Boosting — fast, handles missing values natively, often the most accurate.",
    "ensemble":       "Simple average of all selected models. Reduces the risk of any single model being wrong.",
    "ensemble_w":     "Weighted average where each model's weight = 1/MAPE from the backtest. Best for accuracy.",
}

MODEL_COLORS = {
    "holt":           "#60A5FA",
    "regression":     "#A78BFA",
    "sarima":         "#F59E0B",
    "naive":          "#71717A",
    "naive_bayes":    "#EC4899",
    "random_forest":  "#34D399",
    "gradient_boost": "#10B981",
    "ensemble":       "#F4F4F5",
    "ensemble_w":     "#FFD700",
}

# ── Design palette ─────────────────────────────────────────────────────────────
C_EMERALD = "#34D399"
C_EMERALD_MID = "#10B981"
C_BLUE    = "#60A5FA"
C_RED     = "#F87171"
C_AMBER   = "#FBBF24"
C_PURPLE  = "#A78BFA"
C_GRAY    = "#71717A"
C_GREEN   = C_EMERALD

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Library Book Demand Prediction",
    layout="wide",
    page_icon="📚",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],.main{
    background-color:#0A0A0A !important;
    font-family:'Inter',sans-serif !important;
}
section.main>div{padding-top:0.5rem !important;}

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0D0D0D 0%,#0F0F12 100%) !important;
    border-right:1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] *{color:#D4D4D8 !important;font-family:'Inter',sans-serif !important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{
    color:#34D399 !important;font-weight:600 !important;letter-spacing:-0.02em !important;
}
[data-testid="stSidebar"] .stCaption p{color:#52525B !important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] select{
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.09) !important;
    color:#F4F4F5 !important;border-radius:8px !important;
}

/* Typography */
h1,h2,h3,h4,h5,h6{font-family:'Inter',sans-serif !important;color:#F4F4F5 !important;}
h1{font-weight:700 !important;letter-spacing:-0.05em !important;}
h2{font-weight:600 !important;letter-spacing:-0.03em !important;}
h3{font-weight:600 !important;letter-spacing:-0.025em !important;}
p,li,span{font-family:'Inter',sans-serif !important;color:#A1A1AA !important;}
strong,b{color:#F4F4F5 !important;font-weight:600 !important;}

/* Metrics */
[data-testid="metric-container"]{
    background:rgba(255,255,255,0.04) !important;
    backdrop-filter:blur(12px) !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-radius:16px !important;
    padding:1.25rem 1.25rem 1rem !important;
    transition:all 0.3s cubic-bezier(0.16,1,0.3,1) !important;
}
[data-testid="metric-container"]:hover{
    background:rgba(255,255,255,0.07) !important;
    border-color:rgba(52,211,153,0.28) !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 32px rgba(52,211,153,0.08) !important;
}
[data-testid="stMetricLabel"] p{
    color:#52525B !important;font-size:11px !important;
    font-weight:700 !important;letter-spacing:0.1em !important;text-transform:uppercase !important;
}
[data-testid="stMetricValue"]{
    color:#F4F4F5 !important;font-size:1.9rem !important;
    font-weight:700 !important;letter-spacing:-0.03em !important;
}
[data-testid="stMetricDelta"]{font-size:0.82rem !important;font-weight:500 !important;}

/* Buttons */
.stButton>button{
    font-family:'Inter',sans-serif !important;font-weight:600 !important;
    border-radius:9999px !important;
    transition:all 0.3s cubic-bezier(0.16,1,0.3,1) !important;
    letter-spacing:-0.01em !important;
}
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#34D399 0%,#10B981 100%) !important;
    color:#000 !important;border:none !important;
    padding:0.65rem 2.2rem !important;font-size:0.95rem !important;
}
.stButton>button[kind="primary"]:hover{
    transform:scale(1.05) !important;
    box-shadow:0 0 28px rgba(52,211,153,0.42) !important;
}
.stButton>button:not([kind="primary"]){
    background:rgba(255,255,255,0.05) !important;
    color:#E4E4E7 !important;border:1px solid rgba(255,255,255,0.1) !important;
}
.stButton>button:not([kind="primary"]):hover{
    background:rgba(255,255,255,0.09) !important;
    border-color:rgba(52,211,153,0.35) !important;
}
[data-testid="stDownloadButton"]>button{
    background:rgba(52,211,153,0.09) !important;color:#34D399 !important;
    border:1px solid rgba(52,211,153,0.28) !important;
    border-radius:9999px !important;font-weight:600 !important;
    transition:all 0.3s ease !important;
}
[data-testid="stDownloadButton"]>button:hover{
    background:rgba(52,211,153,0.18) !important;transform:scale(1.03) !important;
}

/* Alerts */
div[data-testid="stAlert"]{border-radius:12px !important;backdrop-filter:blur(12px) !important;}
.stAlert{border-radius:12px !important;}

/* Expanders */
[data-testid="stExpander"]{
    background:rgba(255,255,255,0.025) !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:12px !important;
}
[data-testid="stExpander"] summary p{color:#C4C4C8 !important;font-weight:500 !important;}

/* Inputs */
[data-testid="stSelectbox"]>div>div,
[data-testid="stMultiSelect"]>div>div{
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.09) !important;
    border-radius:10px !important;color:#F4F4F5 !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input{
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.09) !important;
    color:#F4F4F5 !important;border-radius:8px !important;
}
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p{color:#C4C4C8 !important;}
[data-testid="stFileUploader"]{
    background:rgba(255,255,255,0.025) !important;
    border:1px dashed rgba(255,255,255,0.12) !important;
    border-radius:12px !important;
}

/* Misc */
hr{border-color:rgba(255,255,255,0.07) !important;margin:1.5rem 0 !important;}
.stSpinner>div{border-top-color:#34D399 !important;}
.stCaption p{color:#52525B !important;}
[data-testid="stDataFrame"]{border-radius:12px !important;overflow:hidden !important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0A0A0A;}
::-webkit-scrollbar-thumb{background:#27272A;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#3F3F46;}

@keyframes fadeInUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}
.fade-in{animation:fadeInUp 0.8s cubic-bezier(0.16,1,0.3,1) forwards;}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fade-in" style="
    background:linear-gradient(135deg,#0D1410 0%,#0A0A0A 50%,#0D1018 100%);
    border:1px solid rgba(255,255,255,0.08);border-radius:24px;
    padding:2.8rem 2.5rem 2.4rem;margin-bottom:1.5rem;
    position:relative;overflow:hidden;">
    <div style="position:absolute;top:-80px;right:-80px;width:340px;height:340px;
        background:radial-gradient(circle,rgba(52,211,153,0.13) 0%,transparent 68%);pointer-events:none;"></div>
    <div style="position:absolute;bottom:-50px;left:25%;width:220px;height:220px;
        background:radial-gradient(circle,rgba(96,165,250,0.07) 0%,transparent 68%);pointer-events:none;"></div>
    <div style="position:absolute;top:50%;right:3rem;transform:translateY(-50%);
        font-size:9vw;font-weight:700;color:rgba(255,255,255,0.02);
        letter-spacing:-0.05em;pointer-events:none;user-select:none;font-family:Inter,sans-serif;">
        PREDICT
    </div>
    <div style="display:inline-flex;align-items:center;gap:0.5rem;
        background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.22);
        border-radius:9999px;padding:0.28rem 1rem;margin-bottom:1.2rem;">
        <span style="width:6px;height:6px;border-radius:50%;background:#34D399;display:inline-block;
            box-shadow:0 0 6px #34D399;"></span>
        <span style="color:#34D399;font-size:10px;font-weight:700;letter-spacing:0.18em;
            text-transform:uppercase;font-family:Inter,sans-serif;">
            Seattle Public Library · ML Forecasting
        </span>
    </div>
    <h1 style="font-family:Inter,sans-serif;font-weight:700;
        font-size:clamp(1.9rem,3.8vw,3rem);letter-spacing:-0.05em;line-height:1.06;
        color:#F4F4F5;margin:0 0 1rem 0;">
        Library Book<br><span style="color:#34D399;">Demand Prediction</span>
    </h1>
    <p style="font-family:Inter,sans-serif;font-weight:300;font-size:1rem;
        color:#71717A;max-width:580px;line-height:1.65;margin:0 0 1.5rem 0;">
        Predict which book titles will be
        <strong style="color:#A1A1AA;font-weight:500;">most borrowed next month</strong> —
        so librarians can plan stock orders, manage hold queues,
        and allocate budgets before demand arrives.
    </p>
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:0.75rem 1.25rem;text-align:center;">
            <p style="color:#34D399;font-size:1.4rem;font-weight:700;margin:0;letter-spacing:-0.03em;">7</p>
            <p style="color:#52525B;font-size:10px;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;margin:0;">Models</p>
        </div>
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:0.75rem 1.25rem;text-align:center;">
            <p style="color:#60A5FA;font-size:1.4rem;font-weight:700;margin:0;letter-spacing:-0.03em;">SPL</p>
            <p style="color:#52525B;font-size:10px;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;margin:0;">Dataset</p>
        </div>
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:0.75rem 1.25rem;text-align:center;">
            <p style="color:#A78BFA;font-size:1.4rem;font-weight:700;margin:0;letter-spacing:-0.03em;">1-mo</p>
            <p style="color:#52525B;font-size:10px;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;margin:0;">Horizon</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Data Source")
    source = st.radio("Choose data source", ["URL", "Upload CSV"], horizontal=True)
    data_url = None
    uploaded_file = None
    if source == "URL":
        data_url = st.text_input("CSV URL", value=DEFAULT_DATA_URL,
            help="Paste a direct link to a CSV file.")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            st.session_state["_uploaded_csv"] = uploaded_file

    st.subheader("What to Predict")
    level = st.radio("Predict by", ["Title (Book names)", "Category"], index=0)

    st.subheader("Column Names")
    st.caption("Leave blank — the app will auto-detect columns from your CSV headers.")
    date_mode = st.radio("How is the date stored?",
                         ["Year + Month columns", "Single date column"])
    group_col_label   = "Title column" if level.startswith("Title") else "Category column"
    default_group_col = "title" if level.startswith("Title") else ""
    category_col = st.text_input(f"{group_col_label} (optional)", value=default_group_col)
    count_col    = st.text_input("Checkout count column (optional)", value="")
    book_col     = st.text_input("Book filter column (optional)", value="")
    date_col = year_col = month_col = ""
    if date_mode == "Year + Month columns":
        year_col  = st.text_input("Year column (optional)",  value="")
        month_col = st.text_input("Month column (optional)", value="")
    else:
        date_col = st.text_input("Date column (optional)", value="")

    st.subheader("Forecast Models")
    models = st.multiselect(
        "Models to run",
        ["holt", "regression", "sarima", "naive", "naive_bayes", "random_forest", "gradient_boost"],
        default=["regression", "random_forest", "gradient_boost"],
        format_func=lambda m: MODEL_LABELS.get(m, m),
        help="Select models. Ensemble (avg) and Weighted Ensemble are auto-computed.",
    )
    backtest_months = st.number_input(
        "Backtest months", min_value=0, max_value=12, value=3, step=1,
        help="Months used to test model accuracy. Also used to weight the Weighted Ensemble.",
    )

    st.subheader("Performance")
    max_rows = st.number_input("Max rows to load (0 = all)",
        min_value=0, max_value=500000, value=200000, step=10000)
    max_items = st.number_input("Max titles to forecast",
        min_value=0, max_value=200, value=30, step=5)

    st.subheader("Data Quality Filters")
    exclude_latest = st.checkbox("Exclude latest month", value=True)
    min_history_months = st.number_input("Min months of history required",
        min_value=2, max_value=36, value=3, step=1)
    recent_window = st.number_input("Recent activity window (months)",
        min_value=3, max_value=24, value=6, step=1)
    min_recent_nonzero = st.number_input("Min active months in window",
        min_value=0, max_value=24, value=1, step=1)
    apply_guardrails = st.checkbox("Cap extreme predictions", value=True)
    guardrail_multiplier = st.number_input("Cap at (× recent average)",
        min_value=1.5, max_value=10.0, value=3.0, step=0.5)


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _run_models(
    monthly: pd.DataFrame,
    model_list: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    forecasts = []
    for name in model_list:
        if   name == "naive":          df = seasonal_naive_forecast(monthly)
        elif name == "holt":           df = holt_winters_forecast(monthly)
        elif name == "sarima":         df = sarima_forecast(monthly)
        elif name == "regression":     df = regression_forecast(monthly)
        elif name == "naive_bayes":    df = naive_bayes_forecast(monthly)
        elif name == "random_forest":  df = random_forest_forecast(monthly)
        elif name == "gradient_boost": df = gradient_boost_forecast(monthly)
        else: continue
        df["model"] = name
        forecasts.append(df)
    if not forecasts:
        return pd.DataFrame()

    combined = pd.concat(forecasts, ignore_index=True)
    combined["predicted"] = combined["predicted"].clip(lower=0)

    if len(forecasts) > 1:
        # Simple average ensemble
        ens = combined.groupby("category")["predicted"].mean().reset_index()
        ens["model"] = "ensemble"
        combined = pd.concat([combined, ens], ignore_index=True)

        # Weighted ensemble (uses backtest MAPE weights when available)
        if weights:
            valid = combined[combined["model"].isin(weights)].copy()
            valid["w"] = valid["model"].map(weights)
            valid["wpred"] = valid["predicted"] * valid["w"]
            ens_w = valid.groupby("category")["wpred"].sum().reset_index()
            ens_w = ens_w.rename(columns={"wpred": "predicted"})
            ens_w["model"] = "ensemble_w"
            combined = pd.concat([combined, ens_w], ignore_index=True)

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


def _filter_sparse_items(monthly, min_history, window, min_nonzero):
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


def _apply_guardrails(forecasts, recent_avg, mult):
    if forecasts.empty or recent_avg.empty:
        return forecasts
    m = forecasts.merge(recent_avg.rename("ra"), on="category", how="left")
    mask = m["ra"] > 0
    m.loc[mask, "predicted"] = np.minimum(
        m.loc[mask, "predicted"], m.loc[mask, "ra"] * mult
    )
    return m.drop(columns=["ra"])


def _fallback_zeros(forecasts, recent_avg, overall_avg):
    if forecasts.empty:
        return forecasts, []
    affected: List[str] = []

    def _fix(g):
        if g["predicted"].sum() > 0:
            return g
        affected.append(g["model"].iloc[0])
        g = g.merge(recent_avg.rename("ra"), on="category", how="left")
        g = g.merge(overall_avg.rename("oa"), on="category", how="left")
        g["predicted"] = g["predicted"].where(g["predicted"] > 0, g["ra"])
        g["predicted"] = g["predicted"].where(g["predicted"] > 0, g["oa"])
        g["predicted"] = g["predicted"].fillna(0)
        return g.drop(columns=["ra", "oa"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return forecasts.groupby("model", group_keys=False).apply(_fix), affected


def _fmt(v: float) -> str:
    return f"{v:,.0f}"

def _trunc(s: str, n: int = 48) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub_html = (
        f'<p style="font-family:Inter,sans-serif;font-weight:300;font-size:0.875rem;'
        f'color:#52525B;margin:0.3rem 0 0 0;line-height:1.5;">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:0.8rem;margin:1.25rem 0 0.8rem 0;">
        <div style="width:4px;min-height:2.4rem;
            background:linear-gradient(180deg,#34D399,#10B981);
            border-radius:2px;flex-shrink:0;margin-top:5px;"></div>
        <div>
            <h2 style="font-family:Inter,sans-serif;font-weight:600;font-size:1.35rem;
                letter-spacing:-0.03em;color:#F4F4F5;margin:0;">{icon} {title}</h2>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _dark_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart
        .configure_view(fill="transparent", stroke="transparent")
        .configure_axis(
            labelColor="#71717A", gridColor="#1C1C20", titleColor="#A1A1AA",
            domainColor="#3F3F46", tickColor="#3F3F46",
            labelFont="Inter", titleFont="Inter", labelFontSize=11,
        )
        .configure_title(
            color="#E4E4E7", font="Inter", fontWeight=600,
            subtitleColor="#52525B", subtitleFont="Inter",
        )
        .configure_legend(
            labelColor="#A1A1AA", titleColor="#D4D4D8",
            labelFont="Inter", titleFont="Inter",
            fillColor="rgba(0,0,0,0)", strokeColor="rgba(255,255,255,0.08)",
        )
        .configure_text(color="#C4C4C8", font="Inter")
    )


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _chart_top_demand(df: pd.DataFrame, item_label: str) -> alt.Chart:
    d = df.copy()
    d["label"] = d["category"].apply(lambda s: _trunc(str(s), 50))
    d["delta_sign"] = d["delta"].apply(lambda v: f"+{_fmt(v)}" if v >= 0 else _fmt(v))
    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            x=alt.X("predicted:Q", title="Predicted Checkouts", axis=alt.Axis(format=",.0f")),
            y=alt.Y("label:N", sort=alt.SortField("predicted", order="descending"),
                    title=None, axis=alt.Axis(labelLimit=380, labelFontSize=12)),
            color=alt.Color("predicted:Q",
                scale=alt.Scale(range=["#064E3B", "#059669", "#34D399"]), legend=None),
            tooltip=[
                alt.Tooltip("category:N", title=item_label),
                alt.Tooltip("predicted:Q", title="Predicted Checkouts", format=",.0f"),
                alt.Tooltip("last_month:Q", title="Last Month Actual", format=",.0f"),
                alt.Tooltip("delta_sign:N", title="Change vs Last Month"),
                alt.Tooltip("rank:Q", title="Rank"),
            ],
        )
    )
    text = bars.mark_text(align="left", dx=5, fontSize=11, color="#A1A1AA").encode(
        text=alt.Text("predicted:Q", format=",.0f")
    )
    return _dark_chart((bars + text).properties(height=max(280, len(d) * 30)))


def _chart_movers(df: pd.DataFrame, item_label: str) -> alt.Chart:
    d = df.copy()
    d["label"] = d["category"].apply(lambda s: _trunc(str(s), 45))
    d["direction"] = d["delta"].apply(lambda v: "Rising" if v >= 0 else "Falling")
    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("delta:Q", title="Change in Checkouts vs Last Month",
                    axis=alt.Axis(format="+,.0f")),
            y=alt.Y("label:N", sort=alt.SortField("delta", order="descending"),
                    title=None, axis=alt.Axis(labelLimit=320, labelFontSize=11)),
            color=alt.Color("direction:N",
                scale=alt.Scale(domain=["Rising","Falling"], range=[C_GREEN, C_RED]),
                legend=alt.Legend(title="Trend", orient="top")),
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
        .mark_rule(color="#3F3F46", strokeWidth=1.5)
        .encode(x="x:Q")
    )
    return _dark_chart((bars + rule).properties(height=max(200, len(d) * 28)))


def _chart_history_forecast(
    history: pd.DataFrame, forecast_val: float,
    next_month_dt: pd.Timestamp, item_name: str,
    all_preds: Optional[pd.Series] = None,
) -> alt.Chart:
    hist = history[["month", "count"]].copy()
    last_date = hist["month"].iloc[-1]
    last_val  = float(hist["count"].iloc[-1])

    base = alt.Chart(hist).encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("count:Q", title="Checkouts", axis=alt.Axis(format=",.0f")),
    )
    hist_area = base.mark_area(color=C_BLUE, opacity=0.08)
    hist_line = base.mark_line(color=C_BLUE, strokeWidth=2.5)
    hist_dots = base.mark_circle(color=C_BLUE, size=40, opacity=0.7).encode(
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("count:Q", title="Checkouts", format=",.0f"),
        ]
    )

    layers = [hist_area, hist_line, hist_dots]

    # Inter-model confidence band when multiple model predictions are provided
    if all_preds is not None and len(all_preds) >= 2:
        lo, hi = float(all_preds.min()), float(all_preds.max())
        band_df = pd.DataFrame({
            "month": [next_month_dt, next_month_dt],
            "y_lo": [lo, lo], "y_hi": [hi, hi],
        })
        band = (
            alt.Chart(band_df)
            .mark_bar(color=C_EMERALD, opacity=0.12, width=12)
            .encode(x="month:T", y="y_lo:Q", y2="y_hi:Q")
        )
        layers.append(band)

    seg = pd.DataFrame({"month": [last_date, next_month_dt], "val": [last_val, forecast_val]})
    bridge = (
        alt.Chart(seg)
        .mark_line(strokeDash=[6, 4], color=C_EMERALD, strokeWidth=2.4)
        .encode(x="month:T", y="val:Q")
    )
    fc_df = pd.DataFrame({
        "month": [next_month_dt], "val": [forecast_val],
        "lbl":   [f"Forecast: {_fmt(forecast_val)}"],
    })
    fc_dot = (
        alt.Chart(fc_df).mark_circle(size=180, color=C_EMERALD)
        .encode(x="month:T", y="val:Q",
                tooltip=[
                    alt.Tooltip("month:T", title="Forecast Month", format="%b %Y"),
                    alt.Tooltip("val:Q", title="Predicted", format=",.0f"),
                ])
    )
    fc_label = (
        alt.Chart(fc_df)
        .mark_text(align="left", dx=10, dy=-10, fontSize=12, color=C_EMERALD, fontWeight="bold")
        .encode(x="month:T", y="val:Q", text="lbl:N")
    )
    layers += [bridge, fc_dot, fc_label]

    return _dark_chart(
        alt.layer(*layers).properties(
            height=300,
            title=alt.Title(
                text=_trunc(item_name, 60),
                subtitle="Blue = history   |   Green dashed + dot = forecast   |   Green band = inter-model range",
                fontSize=14,
            ),
        )
    )


def _chart_seasonality(seasonal_df: pd.DataFrame) -> alt.Chart:
    d = seasonal_df.copy()
    peak_thresh = d["Avg Checkouts"].quantile(0.67)
    d["Season"] = d["Avg Checkouts"].apply(lambda v: "Peak" if v >= peak_thresh else "Normal")
    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Month:O", sort=MONTH_NAMES, title=None,
                    axis=alt.Axis(labelFontSize=12)),
            y=alt.Y("Avg Checkouts:Q", title="Avg Checkouts",
                    axis=alt.Axis(format=",.0f")),
            color=alt.Color("Season:N",
                scale=alt.Scale(domain=["Peak","Normal"], range=[C_EMERALD,"#3F3F46"]),
                legend=alt.Legend(title="Season")),
            tooltip=[alt.Tooltip("Month:O"),
                     alt.Tooltip("Avg Checkouts:Q", format=",.0f"),
                     alt.Tooltip("Season:N")],
        )
    )
    text = bars.mark_text(dy=-7, fontSize=10, color="#A1A1AA").encode(
        text=alt.Text("Avg Checkouts:Q", format=",.0f")
    )
    return _dark_chart(
        (bars + text).properties(
            height=250,
            title=alt.Title(text="Seasonal Demand by Month",
                            subtitle="Green = peak months (top third of avg demand).", fontSize=14),
        )
    )


def _chart_overall_trend(trend_df: pd.DataFrame) -> alt.Chart:
    return _dark_chart(
        alt.Chart(trend_df)
        .mark_area(opacity=0.18, color=C_BLUE,
                   line={"color": C_BLUE, "strokeWidth": 2.2})
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("count:Q", title="Total Checkouts", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"),
                     alt.Tooltip("count:Q", title="Total Checkouts", format=",.0f")],
        )
        .properties(
            height=250,
            title=alt.Title(text="Library Borrowing Trend Over Time",
                            subtitle="Total checkouts across all tracked titles per month",
                            fontSize=14),
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
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            x=alt.X("MAPE (%):Q",
                    title="MAPE — Mean Absolute % Error (lower = more accurate)",
                    axis=alt.Axis(format=".1f")),
            y=alt.Y("Model:N", sort=alt.SortField("MAPE (%)", order="ascending"),
                    title=None, axis=alt.Axis(labelFontSize=12)),
            color=alt.Color("tag:N",
                scale=alt.Scale(domain=["Best","Other"], range=[C_EMERALD,"#3F3F46"]),
                legend=None),
            tooltip=[alt.Tooltip("Model:N"),
                     alt.Tooltip("MAPE (%):Q", format=".1f"),
                     alt.Tooltip("MAE:Q", format=".1f")],
        )
    )
    text = bars.mark_text(align="left", dx=4, fontSize=11, color="#A1A1AA").encode(
        text=alt.Text("MAPE (%):Q", format=".1f")
    )
    return _dark_chart(
        (bars + text).properties(
            height=max(140, len(df) * 38),
            title=alt.Title(text="Model Accuracy — Backtest Results",
                            subtitle="Tested against actual past months. Green = best performer.",
                            fontSize=14),
        )
    )


def _chart_heatmap(monthly: pd.DataFrame, top_cats: list, max_titles: int = 20) -> alt.Chart:
    d = monthly[monthly["category"].isin(top_cats[:max_titles])].copy()
    d["month_str"] = d["month"].dt.strftime("%b %Y")
    d["label"] = d["category"].apply(lambda s: _trunc(str(s), 32))
    month_order = sorted(d["month_str"].unique(), key=lambda s: pd.to_datetime(s))
    return _dark_chart(
        alt.Chart(d)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("month_str:O", sort=month_order, title="Month",
                    axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
            y=alt.Y("label:N", title=None, axis=alt.Axis(labelFontSize=10)),
            color=alt.Color("count:Q",
                scale=alt.Scale(range=["#0A1A14","#064E3B","#059669","#34D399","#A7F3D0"]),
                legend=alt.Legend(title="Checkouts")),
            tooltip=[alt.Tooltip("category:N", title="Title"),
                     alt.Tooltip("month_str:O", title="Month"),
                     alt.Tooltip("count:Q", title="Checkouts", format=",.0f")],
        )
        .properties(
            height=max(220, min(max_titles, len(top_cats)) * 22),
            title=alt.Title(
                text="Checkout Heatmap — Top Titles by Month",
                subtitle="Brighter green = more checkouts. Reveals seasonal patterns per title.",
                fontSize=14),
        )
    )


def _chart_sparklines(monthly: pd.DataFrame, top_cats: list, n: int = 12) -> alt.Chart:
    """Small-multiple sparklines — one per title, last n months."""
    cats = top_cats[:12]
    d = (
        monthly[monthly["category"].isin(cats)]
        .sort_values("month")
        .groupby("category")
        .tail(n)
        .copy()
    )
    d["label"] = d["category"].apply(lambda s: _trunc(str(s), 34))

    base = alt.Chart(d).encode(
        x=alt.X("month:T", title=None, axis=None),
        y=alt.Y("count:Q", title=None, axis=None, scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("month:T", format="%b %Y"),
            alt.Tooltip("count:Q", title="Checkouts", format=",.0f"),
        ],
    )
    area = base.mark_area(color=C_EMERALD, opacity=0.12, strokeWidth=0)
    line = base.mark_line(color=C_EMERALD, strokeWidth=1.8)

    return _dark_chart(
        alt.layer(area, line)
        .properties(width=160, height=60)
        .facet(
            facet=alt.Facet("label:N",
                            header=alt.Header(
                                labelColor="#C4C4C8", labelFont="Inter",
                                labelFontSize=10, labelFontWeight=500,
                                labelPadding=6,
                            )),
            columns=3,
        )
        .configure_facet(spacing=10)
        .configure_view(
            fill="rgba(255,255,255,0.025)",
            stroke="rgba(255,255,255,0.06)",
            strokeWidth=1,
            cornerRadius=8,
        )
        .resolve_scale(y="independent")
    )


def _chart_model_compare(
    forecasts: pd.DataFrame, top_cats: list, item_label: str,
) -> alt.Chart:
    cats5 = top_cats[:6]
    d = forecasts[
        (forecasts["category"].isin(cats5)) &
        (~forecasts["model"].isin(["ensemble", "ensemble_w"]))
    ].copy()
    d["model_label"] = d["model"].map(MODEL_LABELS).fillna(d["model"])
    d["cat_label"] = d["category"].apply(lambda s: _trunc(str(s), 30))
    d["model_color"] = d["model"].map(MODEL_COLORS).fillna("#71717A")

    color_domain = d["model_label"].unique().tolist()
    color_range  = [
        MODEL_COLORS.get(m, "#71717A")
        for m in d["model"].unique().tolist()
    ]

    return _dark_chart(
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusTopLeft=3)
        .encode(
            x=alt.X("model_label:N", title=None,
                    axis=alt.Axis(labelAngle=-30, labelFontSize=10)),
            y=alt.Y("predicted:Q", title="Predicted Checkouts",
                    axis=alt.Axis(format=",.0f")),
            color=alt.Color("model_label:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Model")),
            column=alt.Column("cat_label:N",
                header=alt.Header(
                    labelColor="#C4C4C8", labelFont="Inter",
                    labelFontSize=10, labelFontWeight=500,
                    titleColor="#71717A",
                )),
            tooltip=[
                alt.Tooltip("cat_label:N", title=item_label),
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("predicted:Q", title="Predicted", format=",.0f"),
            ],
        )
        .properties(
            width=110, height=200,
            title=alt.Title(
                text="Model Comparison — Top Titles",
                subtitle="Each bar is one model's prediction for a title. Compare how models agree or disagree.",
                fontSize=14,
            ),
        )
        .configure_facet(spacing=8)
    )


def _chart_scatter_backtest(bt_rows: pd.DataFrame) -> alt.Chart:
    d = bt_rows.copy()
    d["model_label"] = d["model"].map(MODEL_LABELS).fillna(d["model"])
    d["error_pct"] = ((d["predicted"] - d["actual"]).abs() / d["actual"].clip(lower=1) * 100).round(1)

    color_domain = d["model_label"].unique().tolist()
    color_range  = [MODEL_COLORS.get(m, "#71717A") for m in d["model"].unique().tolist()]

    max_val = float(max(d["predicted"].max(), d["actual"].max())) * 1.05
    diag = pd.DataFrame({"x": [0, max_val], "y": [0, max_val]})

    scatter = (
        alt.Chart(d)
        .mark_circle(size=50, opacity=0.65)
        .encode(
            x=alt.X("actual:Q", title="Actual Checkouts", axis=alt.Axis(format=",.0f")),
            y=alt.Y("predicted:Q", title="Predicted Checkouts", axis=alt.Axis(format=",.0f")),
            color=alt.Color("model_label:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Model")),
            tooltip=[
                alt.Tooltip("category:N", title="Title"),
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("actual:Q", title="Actual", format=",.0f"),
                alt.Tooltip("predicted:Q", title="Predicted", format=",.0f"),
                alt.Tooltip("error_pct:Q", title="Error %", format=".1f"),
            ],
        )
    )
    perfect = (
        alt.Chart(diag)
        .mark_line(strokeDash=[4, 4], color="#52525B", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )
    return _dark_chart(
        (perfect + scatter).properties(
            height=320,
            title=alt.Title(
                text="Predicted vs Actual — Backtest Scatter",
                subtitle="Points close to the dashed diagonal = accurate predictions. Spread = variance.",
                fontSize=14,
            ),
        )
    )


def _chart_forecast_dist(pred_df: pd.DataFrame, next_lbl: str) -> alt.Chart:
    d = pred_df[["category", "predicted", "last_month"]].melt(
        id_vars="category",
        value_vars=["predicted", "last_month"],
        var_name="period", value_name="checkouts",
    )
    d["period"] = d["period"].map({"predicted": f"Forecast ({next_lbl})", "last_month": "Last Month"})

    return _dark_chart(
        alt.Chart(d)
        .transform_density("checkouts", as_=["checkouts", "density"], groupby=["period"])
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X("checkouts:Q", title="Checkouts per Title", axis=alt.Axis(format=",.0f")),
            y=alt.Y("density:Q", title="Density", axis=None),
            color=alt.Color("period:N",
                scale=alt.Scale(
                    domain=[f"Forecast ({next_lbl})", "Last Month"],
                    range=[C_EMERALD, C_BLUE],
                ),
                legend=alt.Legend(title="Period")),
            tooltip=[
                alt.Tooltip("period:N", title="Period"),
                alt.Tooltip("checkouts:Q", title="Checkouts", format=",.0f"),
            ],
        )
        .properties(
            height=200,
            title=alt.Title(
                text="Demand Distribution — Forecast vs Last Month",
                subtitle="How spread out predicted demand is. Right shift = overall demand rising.",
                fontSize=14,
            ),
        )
    )


def _chart_coverage(monthly: pd.DataFrame, top_cats: list) -> alt.Chart:
    cats = top_cats[:8]
    other_total = (
        monthly[~monthly["category"].isin(cats)]
        .groupby("month")["count"].sum().reset_index()
    )
    other_total["category"] = "All Others"

    top_data = monthly[monthly["category"].isin(cats)].copy()
    top_data["category"] = top_data["category"].apply(lambda s: _trunc(str(s), 28))
    combined = pd.concat([top_data[["month","category","count"]], other_total], ignore_index=True)

    order = ["All Others"] + [_trunc(str(c), 28) for c in cats]
    clr_range = ["#27272A"] + [
        "#34D399","#10B981","#059669","#064E3B",
        "#60A5FA","#3B82F6","#1D4ED8","#1E3A5F",
    ]

    return _dark_chart(
        alt.Chart(combined)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("count:Q", stack="normalize", title="Share of Total",
                    axis=alt.Axis(format=".0%")),
            color=alt.Color("category:N",
                scale=alt.Scale(domain=order, range=clr_range[:len(order)]),
                legend=alt.Legend(title="Title", labelLimit=200)),
            order=alt.Order("color_category_sort_index:Q"),
            tooltip=[
                alt.Tooltip("month:T", title="Month", format="%b %Y"),
                alt.Tooltip("category:N", title="Title"),
                alt.Tooltip("count:Q", title="Checkouts", format=",.0f"),
            ],
        )
        .properties(
            height=240,
            title=alt.Title(
                text="Demand Concentration — Share of Total Checkouts",
                subtitle="How dominant the top titles are vs the long tail. Wider coloured band = more concentration.",
                fontSize=14,
            ),
        )
    )


# ── Pre-run intro ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:rgba(96,165,250,0.06);border:1px solid rgba(96,165,250,0.15);
    border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1rem;">
    <p style="color:#93C5FD;font-size:0.85rem;font-weight:600;letter-spacing:0.08em;
        text-transform:uppercase;margin:0 0 0.5rem 0;">How to use</p>
    <ol style="color:#A1A1AA;font-size:0.9rem;margin:0;padding-left:1.2rem;line-height:1.8;">
        <li>Pre-configured for Seattle Public Library — press
            <strong style="color:#F4F4F5;">Run Forecast</strong> to start.</li>
        <li><strong style="color:#F4F4F5;">Random Forest</strong> +
            <strong style="color:#F4F4F5;">Gradient Boosting</strong> are now default models
            (most accurate for this dataset).</li>
        <li>The <strong style="color:#34D399;">Weighted Ensemble</strong> picks the best combination
            based on backtest MAPE — try that for the most reliable forecast.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

if st.button("Run Forecast", type="primary"):

    # ── Progress track ────────────────────────────────────────────────────────
    _has_backtest = int(backtest_months) > 0
    _steps = (
        ["Load Data", "Preprocess", "Backtest", "Run Models", "Done"]
        if _has_backtest else
        ["Load Data", "Preprocess", "Run Models", "Done"]
    )

    def _render_progress(current: int, status: str = "") -> str:
        n = len(_steps)
        nodes = ""
        for i, lbl in enumerate(_steps):
            if i < current:
                dot = ("background:#34D399;border:2px solid #34D399;"
                       "box-shadow:0 0 10px rgba(52,211,153,0.55);")
                col = "#34D399"; ico = "✓"
            elif i == current:
                dot = ("background:transparent;border:2px solid #34D399;"
                       "box-shadow:0 0 14px rgba(52,211,153,0.45);"
                       "animation:pdot 1.3s ease-in-out infinite;")
                col = "#F4F4F5"; ico = "●"
            else:
                dot = "background:#1C1C20;border:2px solid #3F3F46;"
                col = "#52525B"; ico = str(i + 1)

            line = ""
            if i < n - 1:
                if i < current:
                    bg = "linear-gradient(90deg,#34D399,#10B981)"
                elif i == current:
                    bg = "linear-gradient(90deg,#34D399 0%,#3F3F46 100%)"
                else:
                    bg = "#27272A"
                line = (f'<div style="flex:1;height:2px;margin:0 4px;align-self:flex-start;'
                        f'margin-top:17px;background:{bg};border-radius:1px;"></div>')

            nodes += (
                f'<div style="display:flex;flex-direction:column;align-items:center;flex:0 0 auto;">'
                f'<div style="width:34px;height:34px;border-radius:50%;display:flex;'
                f'align-items:center;justify-content:center;font-size:13px;font-weight:700;'
                f'color:#fff;{dot}">{ico}</div>'
                f'<span style="margin-top:8px;font-size:11px;font-weight:600;'
                f'color:{col};white-space:nowrap;font-family:Inter,sans-serif;">{lbl}</span>'
                f'</div>{line}'
            )

        sub = (
            f'<p style="color:#71717A;font-size:0.82rem;margin:0.85rem 0 0;'
            f'font-family:Inter,sans-serif;font-style:italic;">{status}</p>'
            if status else ""
        )
        return (
            "<style>@keyframes pdot{"
            "0%,100%{box-shadow:0 0 6px rgba(52,211,153,0.25);}"
            "50%{box-shadow:0 0 22px rgba(52,211,153,0.75);}}"
            "</style>"
            '<div style="background:rgba(255,255,255,0.03);'
            'border:1px solid rgba(255,255,255,0.08);border-radius:16px;'
            'padding:1.4rem 1.8rem 1.2rem;margin:0.75rem 0 1.2rem;">'
            '<p style="color:#52525B;font-size:10px;font-weight:700;letter-spacing:0.14em;'
            'text-transform:uppercase;margin:0 0 1.1rem;font-family:Inter,sans-serif;">'
            'Pipeline Progress</p>'
            f'<div style="display:flex;align-items:flex-start;width:100%;">{nodes}</div>'
            f'{sub}</div>'
        )

    _prog = st.empty()
    _prog.markdown(_render_progress(0, "Fetching data…"), unsafe_allow_html=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    if True:
        if source == "URL":
            if not data_url:
                st.error("Please enter a CSV URL.")
                st.stop()
            limit = int(max_rows) if max_rows and max_rows > 0 else None
            try:
                data = download_csv(data_url, max_rows=limit)
            except Exception as _dl_err:
                _prog.empty()
                st.error(
                    "**Download failed** — the remote server timed out or refused the connection.\n\n"
                    f"Error: `{_dl_err}`\n\n"
                    "**Quick fix:** Switch to **Upload CSV** mode in the sidebar and upload "
                    "the local file `tmmm-ytt6.csv` that is already in your project folder.",
                    icon="🌐",
                )
                st.stop()
        else:
            csv_source = uploaded_file if uploaded_file is not None \
                else st.session_state.get("_uploaded_csv")
            if csv_source is None:
                st.error("Please upload a CSV file.")
                st.stop()
            nrows = int(max_rows) if max_rows and max_rows > 0 else None
            csv_source.seek(0)
            data = pd.read_csv(csv_source, nrows=nrows)
            data.columns = [c.strip().strip('"').strip("'").lower() for c in data.columns]
    _prog.markdown(_render_progress(1, "Processing checkout records…"), unsafe_allow_html=True)

    if data.empty:
        st.error("No data loaded. Check the URL or file.")
        st.stop()

    if source != "URL":
        n_loaded = len(data)
        st.info(
            f"**Loaded {n_loaded:,} rows** from your CSV.  \n"
            f"Detected columns: {', '.join(f'`{c}`' for c in data.columns)}",
            icon="📄",
        )
        if n_loaded < 5000:
            st.warning(
                f"**Your CSV only has {n_loaded:,} rows.** "
                "Use **URL** mode instead for the full dataset.",
                icon="⚠️",
            )

    if level.startswith("Title") and not category_col:
        category_col = "title"

    # ── Preprocess ────────────────────────────────────────────────────────────
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

    if monthly["category"].nunique() < 5:
        relaxed = True
        eff_max = 0; eff_hist = min(2, max(months_after, 1))
        eff_win = min(3, max(months_after, 1)); eff_nz = 0
        monthly = _filter_top_items(base, eff_max)
        monthly = _filter_sparse_items(monthly, eff_hist, eff_win, eff_nz)

    if monthly.empty:
        st.error("No titles remained after filtering.")
        st.stop()

    if not models:
        st.error("Select at least one model in the sidebar.")
        st.stop()

    # ── Backtest first (for weighted ensemble weights) ────────────────────────
    metrics: Dict = {}
    ensemble_weights: Optional[Dict[str, float]] = None

    if _has_backtest:
        _prog.markdown(
            _render_progress(2,
                f"Backtesting {len(models)} model(s) over {backtest_months} month(s)…"),
            unsafe_allow_html=True,
        )
        metrics = evaluate_models(monthly, models, backtest_months=int(backtest_months))
        if metrics:
            raw_w = {m: 1.0 / max(v["mape"], 0.001) for m, v in metrics.items()}
            total_w = sum(raw_w.values())
            ensemble_weights = {m: w / total_w for m, w in raw_w.items()}

    # ── Run models ────────────────────────────────────────────────────────────
    _model_step = 3 if _has_backtest else 2
    _prog.markdown(
        _render_progress(_model_step,
            "Running: " + ", ".join(MODEL_LABELS.get(m, m) for m in models) + "…"),
        unsafe_allow_html=True,
    )
    forecasts = _run_models(monthly, models, weights=ensemble_weights)

    # ── Complete ──────────────────────────────────────────────────────────────
    _prog.markdown(_render_progress(len(_steps) - 1), unsafe_allow_html=True)

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

    # ── Save all results to session_state so widget interactions don't re-run ──
    st.session_state["_results"] = dict(
        forecasts=forecasts, monthly=monthly, metrics=metrics,
        zero_models=zero_models, last_counts=last_counts,
        last_total=last_total, hist_avg=hist_avg,
        item_label=item_label, n_modelled=n_modelled,
        total_titles=total_titles, total_months=total_months,
        date_min=date_min, date_max=date_max,
        last_lbl=last_lbl, next_lbl=next_lbl,
        relaxed=relaxed, months_after=months_after,
        eff_hist=eff_hist, eff_win=eff_win,
    )

# ── Render results (runs on every rerun if session state populated) ────────────
if "_results" in st.session_state:
    _r          = st.session_state["_results"]
    forecasts   = _r["forecasts"]
    monthly     = _r["monthly"]
    metrics     = _r["metrics"]
    zero_models = _r["zero_models"]
    last_counts = _r["last_counts"]
    last_total  = _r["last_total"]
    hist_avg    = _r["hist_avg"]
    item_label  = _r["item_label"]
    n_modelled  = _r["n_modelled"]
    total_titles= _r["total_titles"]
    total_months= _r["total_months"]
    date_min    = _r["date_min"]
    date_max    = _r["date_max"]
    last_lbl    = _r["last_lbl"]
    next_lbl    = _r["next_lbl"]
    relaxed     = _r["relaxed"]
    months_after= _r["months_after"]
    eff_hist    = _r["eff_hist"]
    eff_win     = _r["eff_win"]
    st.markdown(f"""
    <div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.18);
        border-radius:14px;padding:1rem 1.4rem;margin:1rem 0;">
        <p style="color:#34D399;font-size:0.8rem;font-weight:700;letter-spacing:0.1em;
            text-transform:uppercase;margin:0 0 0.3rem 0;">Data loaded successfully</p>
        <p style="color:#A1A1AA;font-size:0.9rem;margin:0;">
            Found <strong style="color:#F4F4F5;">{total_titles:,} unique {item_label.lower()}s</strong>
            across <strong style="color:#F4F4F5;">{total_months} months</strong>
            ({date_min} – {date_max}).
            Forecasting the top <strong style="color:#34D399;">{n_modelled}</strong>
            most active {item_label.lower()}s for
            <strong style="color:#34D399;">{next_lbl}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if relaxed:
        st.warning(
            f"**Filters automatically relaxed** — data only covers {months_after} month(s). "
            f"Using: min history = {eff_hist} month(s). "
            "Increase 'Max rows to load' or lower 'Min months of history' for more titles.",
            icon="⚠️",
        )
    if zero_models:
        st.caption(
            "Note: Some models produced all-zero predictions and fell back to the recent average — "
            + ", ".join(MODEL_LABELS.get(m, m) for m in zero_models) + "."
        )

    # ── Model picker ──────────────────────────────────────────────────────────
    avail = forecasts["model"].unique().tolist()
    disp  = [MODEL_LABELS.get(m, m) for m in avail]

    # Default to weighted ensemble if available, otherwise simple ensemble
    if "ensemble_w" in avail:
        def_m = "ensemble_w"
    elif "ensemble" in avail:
        def_m = "ensemble"
    else:
        def_m = avail[0]
    def_i = avail.index(def_m)

    pick_col, desc_col = st.columns([2, 3])
    with pick_col:
        chosen_label = st.selectbox(
            "Which model's predictions to display?",
            disp, index=def_i,
            help="Weighted Ensemble is the most accurate choice when backtest data is available.",
        )
    primary = avail[disp.index(chosen_label)]
    with desc_col:
        st.markdown(
            f'<p style="color:#71717A;font-size:0.875rem;padding-top:0.3rem;">'
            f'<strong style="color:{MODEL_COLORS.get(primary, C_EMERALD)};">'
            f'{MODEL_LABELS.get(primary, primary)}</strong>'
            f' — {MODEL_DESCRIPTIONS.get(primary, "")}</p>',
            unsafe_allow_html=True,
        )

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

    top10 = pred_df.nlargest(10, "predicted").copy().reset_index(drop=True)
    top   = pred_df.nlargest(15, "predicted").copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # KEY METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    _section_header(
        "📊", "Forecast at a Glance",
        f"Comparing {last_lbl} (last known month) to the {next_lbl} forecast "
        f"across the top {n_modelled} tracked {item_label.lower()}s.",
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(f"Checkouts — {last_lbl}", _fmt(last_total),
              help="Actual total checkouts in the most recent complete month.")
    k2.metric(f"Forecast — {next_lbl}", _fmt(pred_total),
              f"{'+' if delta_total >= 0 else ''}{_fmt(delta_total)}",
              help="Sum of all title forecasts for next month.")
    k3.metric("Overall Change",
              f"{'+' if (pct_change or 0) >= 0 else ''}{pct_change:.1f}%"
              if pct_change is not None else "N/A",
              help="% change in total predicted checkouts vs last month.")
    k4.metric("Top-10 Share", f"{top10_share * 100:.1f}%",
              help="Share of total predicted demand held by the top 10 titles.")
    k5.metric("Titles Forecasted", f"{n_modelled}",
              help="Number of titles with enough history to forecast.")
    k6.metric("Models Run", str(len(models)),
              help="Number of forecasting models used.")

    # Forecast distribution chart (below metrics)
    st.altair_chart(_chart_forecast_dist(pred_df, next_lbl), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — TOP 10
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header(
        "🏆", f"Top 10 Most In-Demand {item_label}s — {next_lbl}",
        f"The 10 {item_label.lower()}s most likely to be borrowed next month, ranked by predicted checkouts. "
        "Use this to prioritise purchase orders, licence renewals, and hold queue slots.",
    )

    n_top10 = len(top10)
    if n_top10 < 10:
        st.warning(
            f"Only **{n_top10} title(s)** had enough data. "
            "Increase 'Max rows to load' or lower 'Min months of history'.", icon="⚠️",
        )

    MEDALS = {0: "🥇", 1: "🥈", 2: "🥉"}
    list_col, chart_col = st.columns([1, 1], gap="large")

    with list_col:
        st.markdown(
            '<p style="color:#52525B;font-size:11px;font-weight:700;'
            'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.75rem;">Ranked List</p>',
            unsafe_allow_html=True,
        )
        for i, row in top10.iterrows():
            medal = MEDALS.get(i, f"**{i+1}.**")
            delta = row["delta"]
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
            d_clr = "#34D399" if delta > 0 else ("#F87171" if delta < 0 else "#71717A")
            d_str = f"+{_fmt(delta)}" if delta > 0 else _fmt(delta)
            pct   = row["vs_avg_pct"]
            p_str = f"{'+' if pct >= 0 else ''}{pct:.0f}% vs avg"
            st.markdown(
                f"{medal} &nbsp; **{_trunc(str(row['category']), 44)}**  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Predicted: **{_fmt(row['predicted'])}**"
                f" &nbsp;|&nbsp; "
                f'<span style="color:{d_clr};font-weight:600;">{arrow} {d_str}</span>'
                f" &nbsp;|&nbsp; {p_str}",
                unsafe_allow_html=True,
            )
            if i < n_top10 - 1:
                st.markdown(
                    "<hr style='margin:4px 0;border-color:rgba(255,255,255,0.06)'>",
                    unsafe_allow_html=True,
                )

    with chart_col:
        st.markdown(
            '<p style="color:#52525B;font-size:11px;font-weight:700;'
            'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.75rem;">Demand Chart</p>',
            unsafe_allow_html=True,
        )
        st.altair_chart(_chart_top_demand(top10, item_label), use_container_width=True)

    with st.expander("📋 Full prediction table (top 15)", expanded=False):
        table_df = top[["rank","category","predicted","last_month","delta","vs_avg_pct"]].copy()
        table_df = table_df.rename(columns={
            "rank":"Rank","category":item_label,"predicted":"Predicted",
            "last_month":"Last Month","delta":"Change","vs_avg_pct":"vs Avg (%)",
        })
        st.dataframe(
            table_df, width="stretch",
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                item_label: st.column_config.TextColumn(item_label, width="large"),
                "Predicted": st.column_config.ProgressColumn(
                    "Predicted Checkouts", min_value=0,
                    max_value=float(top["predicted"].max()) * 1.1, format="%d",
                ),
                "Last Month": st.column_config.NumberColumn("Last Month", format="%d"),
                "Change": st.column_config.NumberColumn("Change vs Last Month", format="%+d"),
                "vs Avg (%)": st.column_config.NumberColumn("vs Historical Avg", format="%+.1f%%"),
            },
            hide_index=True,
        )

    # ── Sparklines ────────────────────────────────────────────────────────────
    _section_header(
        "✨", f"Trend Sparklines — Top {item_label}s",
        "Each panel shows the last 12 months of checkout history for a top title.",
    )
    spark_cats = top10["category"].tolist()
    st.altair_chart(_chart_sparklines(monthly, spark_cats), use_container_width=True)

    # ── Demand Concentration ──────────────────────────────────────────────────
    _section_header(
        "📐", "Demand Concentration Over Time",
        "How much of total borrowing is driven by the top titles vs the long tail.",
    )
    coverage_cats = pred_df.nlargest(8, "predicted")["category"].tolist()
    st.altair_chart(_chart_coverage(monthly, coverage_cats), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — DEMAND MOVERS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header(
        "📈", "Demand Movers — What's Rising & Falling",
        "Titles with the biggest predicted shifts. Green = more expected; Red = fewer expected.",
    )

    rising  = pred_df[pred_df["delta"] > 0].nlargest(12, "delta")
    falling = pred_df[pred_df["delta"] < 0].nsmallest(12, "delta")
    movers  = pd.concat([rising, falling]).copy()

    if not movers.empty:
        st.altair_chart(_chart_movers(movers, item_label), use_container_width=True)
        mv1, mv2 = st.columns(2)
        with mv1:
            st.markdown(
                '<p style="color:#34D399;font-size:11px;font-weight:700;'
                'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">'
                'Rising — act before demand peaks</p>',
                unsafe_allow_html=True,
            )
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
            st.markdown(
                '<p style="color:#F87171;font-size:11px;font-weight:700;'
                'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">'
                'Falling — safe to reduce orders</p>',
                unsafe_allow_html=True,
            )
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
        st.info("Not enough variation to identify movers.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — TITLE DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header(
        "🔍", f"{item_label} Deep Dive — Historical Trend & Forecast",
        f"Select any {item_label.lower()} to see its full checkout history and forecast. "
        "The green band shows the range across all models.",
    )

    item_options = top["category"].tolist()
    item_choice  = st.selectbox(f"Select a {item_label.lower()} to inspect", item_options)

    history = monthly[monthly["category"] == item_choice].sort_values("month")
    fc_val  = float(top[top["category"] == item_choice]["predicted"].iloc[0])
    next_dt = history["month"].iloc[-1] + pd.offsets.MonthBegin(1)

    # Inter-model range for confidence band
    all_model_preds = forecasts[
        (forecasts["category"] == item_choice) &
        (~forecasts["model"].isin(["ensemble", "ensemble_w"]))
    ]["predicted"]

    st.altair_chart(
        _chart_history_forecast(history, fc_val, next_dt, item_choice, all_model_preds),
        use_container_width=True,
    )

    row = pred_df[pred_df["category"] == item_choice].iloc[0]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(f"Predicted — {next_lbl}", _fmt(fc_val))
    s2.metric(f"Actual — {last_lbl}", _fmt(float(row["last_month"])))
    s3.metric("Month-on-Month Change",
              f"{'+' if row['delta'] >= 0 else ''}{_fmt(row['delta'])}")
    s4.metric("vs Long-Run Average",
              f"{'+' if row['vs_avg_pct'] >= 0 else ''}{row['vs_avg_pct']:.1f}%")

    n_months = len(history)
    avg_val  = float(history["count"].mean())
    max_val  = float(history["count"].max())
    st.caption(
        f"Based on {n_months} months of history. "
        f"Historical avg: {_fmt(avg_val)}/month. All-time peak: {_fmt(max_val)}/month."
    )

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.divider()
    _section_header(
        "⚖️", "Model Comparison — Do They Agree?",
        "Grouped bars show each model's prediction for the top titles. "
        "Strong agreement = higher confidence in the forecast.",
    )
    compare_cats = pred_df.nlargest(6, "predicted")["category"].tolist()
    st.altair_chart(_chart_model_compare(forecasts, compare_cats, item_label), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header(
        "🗺️", "Demand Heatmap — Top Titles Over Time",
        "Brighter cells = more checkouts. Reveals which titles dominate each month.",
    )
    heatmap_cats = pred_df.nlargest(20, "predicted")["category"].tolist()
    st.altair_chart(_chart_heatmap(monthly, heatmap_cats), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — SEASONAL PATTERNS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header(
        "📅", "Seasonal Patterns & Long-Term Trend",
        "Seasonal peaks drive model predictions — understanding them helps plan staffing and stock year-round.",
    )

    col_s, col_t = st.columns(2)
    with col_s:
        totals   = monthly.groupby("month")["count"].sum().reset_index()
        seas_grp = totals.groupby(totals["month"].dt.month)["count"].mean()
        seas_val = seas_grp.reindex(range(1, 13), fill_value=0)
        seas_df  = pd.DataFrame({"Month": MONTH_NAMES, "Avg Checkouts": seas_val.values})
        peak_thresh2 = seas_val.quantile(0.67)
        peak_months  = [MONTH_NAMES[i-1] for i in range(1, 13) if seas_val[i] >= peak_thresh2]
        st.altair_chart(_chart_seasonality(seas_df), use_container_width=True)
        if peak_months:
            st.caption(f"Peak season: **{', '.join(peak_months)}** — plan extra stock.")
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
            f"from {date_min} to {last_lbl}."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — LIBRARY PLANNING ACTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header(
        "✅", f"Library Planning Actions for {next_lbl}",
        "Most important actions based on the forecast, ranked by impact.",
    )

    top5_demand  = pred_df.nlargest(5, "predicted")
    top5_rising  = pred_df[pred_df["delta"] > 0].nlargest(5, "delta")
    top5_falling = pred_df[pred_df["delta"] < 0].nsmallest(5, "delta")

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        <div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.15);
            border-radius:14px;padding:1rem 1.2rem 0.25rem;margin-bottom:0.75rem;">
            <p style="color:#34D399;font-size:11px;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;margin:0 0 0.4rem 0;">📦 Order or License More Copies</p>
            <p style="color:#71717A;font-size:0.8rem;margin:0 0 0.75rem 0;">
                Highest predicted demand — ensure copies available before the month starts.
            </p>
        </div>""", unsafe_allow_html=True)
        for i, (_, r) in enumerate(top5_demand.iterrows(), 1):
            ratio = r["predicted"] / max(r["last_month"], 1)
            urgency = "🔴 High" if ratio > 1.5 else ("🟡 Moderate" if ratio > 1.1 else "🟢 Stable")
            st.markdown(
                f"**{i}. {_trunc(str(r['category']), 42)}**  \n"
                f"Predicted: **{_fmt(r['predicted'])}** — {urgency}"
            )
    with a2:
        st.markdown("""
        <div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);
            border-radius:14px;padding:1rem 1.2rem 0.25rem;margin-bottom:0.75rem;">
            <p style="color:#FCD34D;font-size:11px;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;margin:0 0 0.4rem 0;">⚠️ Watch Hold Queues</p>
            <p style="color:#71717A;font-size:0.8rem;margin:0 0 0.75rem 0;">
                Biggest predicted increases — proactively manage hold lists.
            </p>
        </div>""", unsafe_allow_html=True)
        if top5_rising.empty:
            st.caption("No significant increases predicted.")
        else:
            for i, (_, r) in enumerate(top5_rising.iterrows(), 1):
                st.markdown(
                    f"**{i}. {_trunc(str(r['category']), 42)}**  \n"
                    f"Expected **+{_fmt(r['delta'])}** more "
                    f"({_fmt(r['last_month'])} → {_fmt(r['predicted'])})"
                )
    with a3:
        st.markdown("""
        <div style="background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.15);
            border-radius:14px;padding:1rem 1.2rem 0.25rem;margin-bottom:0.75rem;">
            <p style="color:#F87171;font-size:11px;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;margin:0 0 0.4rem 0;">💰 Reallocate Budget</p>
            <p style="color:#71717A;font-size:0.8rem;margin:0 0 0.75rem 0;">
                Biggest predicted drops — safe to reduce orders and redirect budget.
            </p>
        </div>""", unsafe_allow_html=True)
        if top5_falling.empty:
            st.caption("No significant decreases predicted.")
        else:
            for i, (_, r) in enumerate(top5_falling.iterrows(), 1):
                st.markdown(
                    f"**{i}. {_trunc(str(r['category']), 42)}**  \n"
                    f"Expected **{_fmt(r['delta'])}** fewer "
                    f"({_fmt(r['last_month'])} → {_fmt(r['predicted'])})"
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — MODEL ACCURACY
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header(
        "🎯", "Model Accuracy — How Reliable Are These Forecasts?",
        f"Each model was backtested over the last {backtest_months} month(s). "
        "Weighted Ensemble auto-favours the best-performing models.",
    )

    if int(backtest_months) > 0:
        if metrics:
            acc1, acc2 = st.columns([3, 2])
            with acc1:
                st.altair_chart(_chart_model_accuracy(metrics), use_container_width=True)
            with acc2:
                best_model = min(metrics, key=lambda m: metrics[m]["mape"])
                best_mape  = metrics[best_model]["mape"] * 100
                st.markdown("""
                <p style="color:#52525B;font-size:11px;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;margin-bottom:0.75rem;">Accuracy Summary</p>
                """, unsafe_allow_html=True)
                for m, v in sorted(metrics.items(), key=lambda x: x[1]["mape"]):
                    mape = v["mape"] * 100
                    bar_w = max(4, int((1 - min(mape / 100, 1)) * 100))
                    clr = C_EMERALD if m == best_model else "#3F3F46"
                    w_str = (
                        f'<span style="color:#FBBF24;font-size:10px;"> '
                        f'w={ensemble_weights[m]:.2f}</span>'
                        if ensemble_weights and m in ensemble_weights else ""
                    )
                    st.markdown(f"""
                    <div style="margin-bottom:0.6rem;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                            <span style="color:#C4C4C8;font-size:0.8rem;font-weight:500;">
                                {MODEL_LABELS.get(m,m)}{w_str}
                            </span>
                            <span style="color:#71717A;font-size:0.8rem;">{mape:.1f}%</span>
                        </div>
                        <div style="background:#1C1C20;border-radius:4px;height:6px;">
                            <div style="background:{clr};border-radius:4px;
                                height:6px;width:{bar_w}%;transition:width 0.5s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Scatter plot
            st.markdown("---")
            # Rebuild backtest rows for scatter
            bt_frames = []
            for m_name in models:
                from src.evaluate import _forecast_for_model
                for eval_m in sorted(monthly["month"].unique())[-int(backtest_months):]:
                    train = monthly[monthly["month"] < eval_m]
                    actual = monthly[monthly["month"] == eval_m][["category","count"]].rename(columns={"count":"actual"})
                    if train.empty or actual.empty:
                        continue
                    try:
                        fc = _forecast_for_model(m_name, train)
                        if fc.empty:
                            continue
                        merged = actual.merge(fc[["category","predicted"]], on="category", how="inner")
                        merged["model"] = m_name
                        bt_frames.append(merged)
                    except Exception:
                        continue

            if bt_frames:
                bt_all = pd.concat(bt_frames, ignore_index=True)
                st.altair_chart(_chart_scatter_backtest(bt_all), use_container_width=True)

            with st.expander("📋 Full accuracy table", expanded=False):
                rows = [
                    {"Model": MODEL_LABELS.get(m, m),
                     "MAE": f"{v['mae']:.1f}", "RMSE": f"{v['rmse']:.1f}",
                     "MAPE": f"{v['mape']*100:.1f}%", "Samples": int(v.get("samples",0))}
                    for m, v in metrics.items()
                ]
                st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

            best_mape = metrics[best_model]["mape"] * 100
            if best_mape < 20:
                verdict = f"**{MODEL_LABELS.get(best_model)}** achieved {best_mape:.1f}% MAPE — forecasts are reliably accurate."
            elif best_mape < 50:
                verdict = f"**{MODEL_LABELS.get(best_model)}** was most accurate at {best_mape:.1f}% MAPE — use as directional guidance."
            else:
                verdict = f"All models showed >50% error. Consider loading more historical data."
            st.info(f"**Verdict:** {verdict}", icon="🎯")

            if ensemble_weights:
                top_w = max(ensemble_weights, key=ensemble_weights.get)
                st.success(
                    f"**Weighted Ensemble** assigns highest weight to "
                    f"**{MODEL_LABELS.get(top_w, top_w)}** ({ensemble_weights[top_w]:.1%}). "
                    "Select 'Ensemble (weighted)' above for the best predictions.",
                    icon="⚡",
                )
        else:
            st.info("Not enough data to backtest. Reduce 'Backtest months' or load more data.")
    else:
        st.info("Backtest disabled (set to 0 months).")

    # ═══════════════════════════════════════════════════════════════════════════
    # DOWNLOADS
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    _section_header("⬇️", "Download Results",
                    "Export forecast data for spreadsheets, reports, or acquisition systems.")

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download Forecast CSV",
        forecasts.to_csv(index=False).encode("utf-8"),
        "forecast_next_month.csv", "text/csv",
    )

    with st.expander("🔍 View raw forecast data (all models)"):
        st.dataframe(forecasts, width="stretch")
