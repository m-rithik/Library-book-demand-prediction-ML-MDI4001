# Library Book Demand Prediction — Project Guide

A complete walkthrough of how this project works, from raw data to forecasts displayed in the web app.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Project Structure](#2-project-structure)
3. [The Data](#3-the-data)
4. [Data Pipeline](#4-data-pipeline)
   - [Step 1: Loading Data](#step-1-loading-data)
   - [Step 2: Preprocessing](#step-2-preprocessing)
   - [Step 3: Filtering](#step-3-filtering)
5. [Forecasting Models](#5-forecasting-models)
   - [Seasonal Naive](#seasonal-naive)
   - [Linear Regression](#linear-regression)
   - [Holt-Winters](#holt-winters)
   - [SARIMA](#sarima)
   - [Naive Bayes](#naive-bayes)
   - [Ensemble](#ensemble)
6. [Model Evaluation (Backtesting)](#6-model-evaluation-backtesting)
7. [The Web App](#7-the-web-app)
   - [Sidebar Controls](#sidebar-controls)
   - [App Sections](#app-sections)
8. [Command-Line Interface](#8-command-line-interface)
9. [How to Run](#9-how-to-run)
10. [Key Design Decisions](#10-key-design-decisions)

---

## 1. What This Project Does

Libraries need to plan ahead — ordering books, managing hold queues, and allocating budgets. This project answers: **"How many checkouts will each book category get next month?"**

Given historical monthly checkout data from the Seattle Public Library (SPL), it:

- Trains multiple forecasting models per category
- Predicts next month's checkout demand for each category
- Evaluates model accuracy via backtesting
- Presents results in an interactive web app with actionable recommendations

---

## 2. Project Structure

```
Library-book-demand-prediction-ML-MDI4001-1/
│
├── web_app.py              # Streamlit web application (main UI)
├── forecast.py             # Command-line entry point
│
├── src/
│   ├── config.py           # Column name candidates, default URL, constants
│   ├── io.py               # Data loading + Socrata API pagination
│   ├── preprocess.py       # Book filtering, monthly aggregation, gap filling
│   ├── evaluate.py         # Backtesting and metric computation
│   │
│   └── models/
│       ├── baselines.py    # Seasonal Naive model
│       ├── regression.py   # Linear Regression with lag features
│       ├── holt_winters.py # Holt-Winters Exponential Smoothing
│       ├── sarima.py       # SARIMA (statsmodels)
│       └── naive_bayes.py  # Gaussian Naive Bayes (classification → regression)
│
└── outputs/                # Generated forecast CSVs and metric reports
```

---

## 3. The Data

**Source:** Seattle Public Library — Checkouts by Title dataset
**API endpoint:** `https://data.seattle.gov/resource/tmmm-ytt6.csv` (Socrata Open Data API)

The dataset contains one row per title per checkout month, with these key columns:

| Column          | Description                                      |
|-----------------|--------------------------------------------------|
| `checkoutyear`  | Year of checkout (e.g., 2023)                    |
| `checkoutmonth` | Month number of checkout (1–12)                  |
| `materialtype`  | Type of item (BOOK, EBOOK, AUDIOBOOK, DVD, etc.) |
| `checkouts`     | Number of checkouts for that title that month    |
| `title`         | Book/item title                                  |

The dataset is large (millions of rows). The app fetches it in **50,000-row batches** via the Socrata `$limit`/`$offset` pagination parameters. You can cap the download with a **Max Rows** setting to speed things up.

---

## 4. Data Pipeline

### Step 1: Loading Data

**File:** `src/io.py` — `load_data()`

Three loading strategies, tried in order:

1. **Local CSV file** — if `--data-path` is given, reads directly from disk. Fastest option.
2. **Cached download** — if a cache file exists at `--cache-path`, reads from cache instead of re-downloading.
3. **Live download** — fetches from the Socrata API using `_download_socrata()`, which paginates through the dataset in 50,000-row chunks until it either hits `max_rows` or exhausts the data. After downloading, saves to the cache file if one is configured.

```
load_data()
  ├── data_path given?  → pd.read_csv(data_path)
  ├── cache exists?     → pd.read_csv(cache_path)
  └── else              → _download_socrata() → save to cache → return
```

### Step 2: Preprocessing

**File:** `src/preprocess.py` — `build_monthly_series()` + `complete_monthly_index()`

**2a. Filter to books only**

The dataset contains many material types (DVDs, eBooks, magazines). The pipeline filters to rows where `materialtype == "BOOK"`. If none match exactly, it falls back to any value containing "book" but not "ebook" or "audiobook".

**2b. Column auto-detection**

The code tries to detect the right columns by checking against a list of known candidate names (case-insensitive). For example, it looks for the month column in: `checkoutmonth`, `CheckoutMonth`, `month`, `Month`, etc. If you supply a custom dataset, you can override these in the sidebar.

**2c. Monthly aggregation**

Rows are grouped by `(month, category)` and summed:

```
Raw rows (one per title per month)
  → group by (month, materialtype)
  → sum checkouts
  → result: one row per category per month
```

The "category" dimension defaults to `materialtype`, but you can switch it to `collection`, `checkouttype`, or others in the sidebar.

**2d. Gap filling**

`complete_monthly_index()` ensures every category has a continuous monthly series with no gaps. Missing months are filled with `0` checkouts. This is important for models that expect evenly spaced time steps.

```
Category A: Jan, Mar, Apr  →  Jan, Feb(0), Mar, Apr
```

### Step 3: Filtering

**File:** `forecast.py` — `_filter_top_items()` + `_filter_sparse_items()`

After building the monthly series, two optional filters prune the data before forecasting:

**Top-N filter** (`--max-items`): Keeps only the N categories with the highest total historical checkouts. Useful for focusing on high-demand items.

**Sparsity filter** (`--min-history-months`, `--recent-window`, `--min-recent-nonzero`): Removes categories that don't have enough history to produce a reliable forecast. A category is kept only if:
- It has at least `min_history_months` months of data
- In the last `recent_window` months, at least `min_recent_nonzero` have non-zero checkouts

Categories that barely circulate (e.g., a single checkout years ago) are excluded because forecasting them adds noise without value.

---

## 5. Forecasting Models

All models share the same interface: they receive a DataFrame of historical monthly counts and return a DataFrame of next-month predictions, one row per category.

Each model is fit **independently per category** — there is no cross-category information sharing (except Naive Bayes, which trains a single global model).

---

### Seasonal Naive

**File:** `src/models/baselines.py`

**Idea:** "Next month will look like the same month last year."

This is the simplest possible forecast and acts as a baseline. For January 2025, it returns the actual January 2024 count. If no year-ago data exists, it falls back to last month's count. If that is zero, it uses the 3-month average.

**When it works well:** Stable, strongly seasonal series (e.g., summer reading programs spike every July).

**Limitation:** Ignores trend. If checkouts have been growing steadily, this will always underestimate.

---

### Linear Regression

**File:** `src/models/regression.py`

**Idea:** Fit a line through historical data using features of the date and recent demand, then extrapolate to next month.

Features used:

| Feature      | Description                              |
|--------------|------------------------------------------|
| `month_num`  | Calendar month (1–12), captures seasonality |
| `year`       | Calendar year, captures long-term trend  |
| `lag1`       | Last month's actual checkout count       |
| `avg3`       | Rolling 3-month average of checkouts     |

The model requires at least 6 months of history. With enough data, it uses all four features (`month_num`, `year`, `lag1`, `avg3`). With less data, it falls back to just `month_num` and `year`.

**When it works well:** Series with a clear upward or downward trend and mild seasonality.

**Limitation:** Linear relationships only; can produce negative predictions (clipped to 0 downstream).

---

### Holt-Winters

**File:** `src/models/holt_winters.py`

**Idea:** Exponential smoothing that separately models three components: level (current average), trend (direction of change), and seasonality (repeating monthly pattern).

Uses **additive** trend and **additive** seasonality with a 12-month seasonal period. Requires at least **24 months** of history (two full seasonal cycles) to fit the seasonal component.

The model parameters are automatically optimized by `statsmodels` to minimize squared error on the training data.

**When it works well:** Series with both a clear trend and strong seasonal patterns. This is typically the best model for library data.

**Limitation:** Needs 2+ years of data per category. Falls back to last known value if insufficient data.

---

### SARIMA

**File:** `src/models/sarima.py`

**Full name:** Seasonal AutoRegressive Integrated Moving Average

**Idea:** A statistical model that captures autocorrelation (a month's checkouts depend on previous months) plus seasonal autocorrelation (this January depends on last January). Uses differencing to handle trend.

The model order used is `SARIMA(1,1,1)(1,1,0)[12]`:
- `(1,1,1)`: one autoregressive term, one difference, one moving-average term
- `(1,1,0)[12]`: one seasonal autoregressive term, one seasonal difference, 12-month period

**When it works well:** Series where recent history strongly predicts the next value and seasonal patterns are important.

**Limitation:** Computationally heavier than other models; can have convergence issues on short or irregular series (warnings suppressed automatically).

---

### Naive Bayes

**File:** `src/models/naive_bayes.py`

**Idea:** Rather than directly predicting a count, this model *classifies* demand into buckets (Low / Medium / High / Very High), then maps each bucket back to a typical count using the median of that bucket.

**How it works step by step:**

1. **Feature engineering** (same as regression): `month_num`, `year`, `lag1` (last month's count), `avg3` (3-month rolling average), `avg6` (6-month rolling average).

2. **Binning targets**: Historical checkout counts are split into 4 classes using quantiles of the non-zero values:
   - Class 0: Zero checkouts
   - Class 1: Low (up to 33rd percentile of non-zero counts)
   - Class 2: Medium (33rd–66th percentile)
   - Class 3: High (above 66th percentile)

3. **Global training**: A single `GaussianNB` classifier is trained on all categories combined. This means the model learns demand patterns across the whole library system, not just one category.

4. **Prediction**: For each category, the model predicts a demand class, then returns the **median checkout count** of all historical rows that fell into that class.

**When it works well:** When categorical demand patterns are similar across book types. Works even with limited per-category history because it pools all data.

**Limitation:** The bucketed approach loses precision; the median mapping can be imprecise for categories with unusual distributions.

---

### Ensemble

**Not a separate model file** — computed in the web app.

The ensemble takes the **average prediction** across all selected models for each category. By averaging, extreme predictions from any single model are dampened. In practice, the ensemble often outperforms any individual model on unseen data.

---

## 6. Model Evaluation (Backtesting)

**File:** `src/evaluate.py` — `evaluate_models()`

Instead of evaluating on a held-out test set, the project uses **rolling backtesting**: it simulates what the model would have predicted in the past, then compares to what actually happened.

**Process:**

For each of the last `backtest_months` (default: 3) months in the dataset:

1. **Split**: All data *before* that month = training set.
2. **Predict**: Run the model on the training set to get next-month predictions.
3. **Compare**: Merge predictions with the actual counts for that month.
4. **Aggregate**: Collect all errors across all backtest months.

**Metrics computed:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\|predicted − actual\|) | Average absolute error in checkouts |
| **RMSE** | √mean((predicted − actual)²) | Penalises large errors more heavily |
| **MAPE** | mean(\|error\| / max(actual, 1)) | Percentage error; scale-independent |

Lower is better for all three. MAPE is the primary metric shown in the web app because it's comparable across categories with different checkout volumes.

---

## 7. The Web App

**File:** `web_app.py`
**Framework:** Streamlit

Run it with:
```bash
streamlit run web_app.py
```

The app is divided into a **sidebar** (all settings) and a **main area** (all results).

---

### Sidebar Controls

| Section | Controls |
|---------|----------|
| **Data Source** | Data URL or local CSV path; max rows to download; cache file path |
| **What to Predict** | Category column (materialtype, collection, etc.); whether to exclude the most recent month |
| **Column Names** | Override auto-detected date/year/month/count/book columns |
| **Forecast Models** | Checkboxes for each model; ensemble is auto-added |
| **Performance** | Number of backtest months |
| **Data Quality Filters** | Min history months, max items, recent window, min recent non-zero, guardrail multiplier |

**Guardrail multiplier**: Caps any prediction at `recent_avg × multiplier`. Prevents runaway forecasts for categories with a sudden spike in training data. Default = 3×.

**Auto-relax**: If filters are so strict that fewer than 5 categories survive, the app automatically relaxes them to show at least some results and warns you.

---

### App Sections

After clicking **Run Forecast**, the app displays results in this order:

#### Key Metrics (top row)
Four summary numbers in coloured boxes:
- **Categories forecast** — how many distinct categories have predictions
- **Predicted total checkouts** — sum of all next-month predictions
- **Best model** — model with lowest MAPE in backtesting
- **Data span** — date range of the training data

#### Top 10 Most In-Demand
Two-column layout:
- **Left**: Numbered list with 🥇🥈🥉 medals for top 3, predicted checkouts, change arrow (▲/▼) vs previous month, and % vs historical average
- **Right**: Horizontal bar chart with a colour gradient (teal) ordered by predicted demand

Below the two columns: an expandable table showing the full top 15.

#### Demand Movers
Categories predicted to change the most compared to their recent average. Shows:
- A **diverging bar chart** (green = rising, red = falling) for top movers in each direction
- Two text lists: biggest risers and biggest fallers with % change

#### Title Deep Dive
A dropdown to select any category and see:
- A **line + forecast chart**: solid line for history, dashed line extending to the prediction point
- Four stat boxes: predicted checkouts, vs previous month, vs historical average, vs same month last year
- A caption with the historical average and all-time peak

#### Seasonal Patterns & Long-Term Trend
Two charts side by side:
- **Seasonality bars**: Average checkouts by calendar month across all categories; months above overall average are highlighted in red ("Peak"), others in blue ("Normal")
- **Trend area chart**: Total checkouts across all categories over time, showing the long-term trajectory

#### Library Planning Actions
Three recommendation columns based on forecast levels vs historical average:
- **Order More 📦**: Categories with predicted demand > 120% of their average
- **Watch Hold Queues ⚠️**: Categories with predicted demand 100–120% of their average
- **Reallocate Budget 💰**: Categories with predicted demand < 80% of their average

Each item has an urgency badge: 🔴 High / 🟡 Medium / 🟢 Low based on how far the prediction deviates from the average.

#### Model Accuracy
- **Horizontal bar chart**: MAPE for each model; best model shown in green, others in grey
- **Detail table**: MAE, RMSE, MAPE, and number of backtest samples per model
- **Plain-English verdict**: e.g., "Holt-Winters has the lowest error (MAPE 12.3%). Predictions are most reliable for categories with 24+ months of history."

#### Downloads
- **Download CSV button**: Exports the full forecast table
- **Raw data expander**: Shows the raw monthly aggregate table used for training

---

## 8. Command-Line Interface

**File:** `forecast.py`

You can run forecasts without the web UI:

```bash
python forecast.py \
  --data-url https://data.seattle.gov/resource/tmmm-ytt6.csv \
  --max-rows 200000 \
  --model all \
  --min-history-months 12 \
  --output-forecast outputs/forecast_next_month.csv \
  --output-metrics outputs/metrics_report.json
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | None | Path to a local CSV file |
| `--data-url` | SPL Socrata URL | URL to download from |
| `--max-rows` | 0 (all) | Limit rows downloaded |
| `--model` | `all` | `naive_bayes`, `regression`, or `all` |
| `--backtest-months` | 3 | How many months to use in backtesting |
| `--min-history-months` | 12 | Minimum months of data per category |
| `--max-items` | 0 (all) | Keep only top-N categories by total demand |
| `--guardrail-multiplier` | 3.0 | Cap forecasts at N × recent average |
| `--output-forecast` | `outputs/forecast_next_month.csv` | Where to save predictions |
| `--output-metrics` | `outputs/metrics_report.json` | Where to save backtest metrics |

The CLI prints a summary of the top 5 predicted categories per model to the terminal and saves full results to the output files.

---

## 9. How to Run

**Prerequisites:**
```bash
pip install streamlit pandas numpy scikit-learn statsmodels altair requests
```

**Option A — Web App (recommended):**
```bash
streamlit run web_app.py
```
Then open `http://localhost:8501` in your browser.

- Adjust settings in the sidebar
- Click **Run Forecast**
- Explore the results sections

**Option B — Command Line:**
```bash
python forecast.py --max-rows 100000
```
Results saved to `outputs/`.

**Option C — Use a local CSV (fastest):**

Download the SPL dataset once, then point the app to it:
```bash
# In sidebar: set "Local CSV path" to your file, or via CLI:
python forecast.py --data-path /path/to/checkouts.csv
```

---

## 10. Key Design Decisions

**Why monthly aggregation?**
Individual title-level daily data is too sparse and noisy. Aggregating by category and month gives enough signal to forecast reliably, while remaining actionable for library planning (purchase orders, staffing, budget cycles are all monthly).

**Why multiple models?**
No single model dominates all categories. Some categories have strong seasonality (Holt-Winters wins), others have clear trends (regression wins), and short-history categories need the pooled Naive Bayes approach. The ensemble hedges across all of them.

**Why backtest instead of train/test split?**
A single train/test split on time-series data wastes data and gives only one evaluation point. Rolling backtesting gives multiple evaluation points and better estimates true out-of-sample performance.

**Why clip predictions to 0?**
Linear regression and SARIMA can produce negative predictions for low-demand categories. Since checkouts cannot be negative, all predictions are clipped at 0 before display.

**Why the guardrail multiplier?**
Forecasting models can overfit to a single unusual month and predict extreme values. The guardrail caps any prediction at `3 × recent_average` by default, preventing wildly unrealistic procurement recommendations.

**Why auto-relax filters?**
The Socrata API returns rows without an ORDER BY guarantee — a partial download may cover only a narrow date range, leaving most categories with fewer than the minimum required months of history. The auto-relax ensures the app always shows results rather than a blank screen, while warning the user about the data limitation.
