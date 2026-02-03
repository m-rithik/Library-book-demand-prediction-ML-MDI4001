Library Book Demand Prediction
Forecasting Which Categories Will Be Borrowed Next Month
Name: M Rithik
Regd Num: 23MID0413

Domain Selection
Chosen Domain: Education / Library & Information Services (public library operations and service planning).

Problem Identification
Public and research libraries also face decisions regarding what formats to buy and license (print books, e-books, and audiobooks), and quantities for each category. Demand also varies from one month to another depending on seasons and community activities. In most cases, if a library cannot forecast future demand, it will experience a backlog in demand for certain formats and overstocking for other formats not in high demand.

Introduction
This specific project involves the prediction of those categories of the library’s collection that will be in greatest demand in the coming month. Being able to accurately predict the coming month’s categories will help (i) collection development (selecting the correct categories to acquire/license), (ii) effectively manage the hold queues, (iii) programming, staff, and (iv) format budgets. A significant reason why the problem of predicting the demand of the coming month’s categories matters has to do with the fact that circulation represents a real-world demand pattern.

This can be achieved by the availability of circulation statistics that can be grabbed freely by libraries. This data not only offers the checkouts at a timestamped format. It also offers the monthly aggregation of checkouts that can be sorted into categories. The categories can be utilized based on machine learning and forecasting statistics based on procedures involving next month borrow volume statistics.

Study Section
Dataset Used (Standard Public Dataset):
Open data offerings at the Seattle Public Library (SPL) circulation or “checkout” data sets. Two popular versions are:
• Title Checkouts (monthly aggregate)
• Checkouts by Title (Physical Items / transaction-level physical checkouts)
These data sets can be aggregated to the month level and categorized using the collection codes, item type codes, audience indicators, or material formats.

Novel Observations / Data Trends to Examine:
• Seasonality: recurring monthly patterns (e.g., summer reading, holiday borrowing spikes)
• Trend: long-term changes by format (print vs digital), and by audience (adult vs children)
• Category heterogeneity: some categories are stable (reference/non-fiction), while others are spiky (popular fiction, children)
• Event sensitivity: short-term surges linked to holidays, major releases, or local programs

Comparative Analysis of Existing Models:
Model / Approach
Strengths
Limitations
Notes in Library Demand Context
Naive seasonal baselines
(last month / last year same month / moving average)
Fast, strong sanity-check baseline; often competitive for seasonal data
Cannot adapt well to regime changes; weak on sudden shifts
Must be included to prove ML adds value beyond simple seasonality
Regression with calendar features
(month dummies, trend)
Interpretable; easy to include holidays, term-time indicators
Limited non-linear modeling; may miss complex patterns
Good for explaining drivers, but may underfit spiky categories
ARIMA / SARIMA
Strong for linear temporal dependence and stable seasonality
Requires stationarity handling; less robust to abrupt structural breaks
Common benchmark for circulation forecasting; captures lag and seasonal effects
Exponential smoothing
(Holt–Winters)
Good for trend + seasonality; simple to tune
May lag behind during sudden shifts
Strong baseline for monthly series; compare against SARIMA
Tree ML with lag features
(Random Forest / XGBoost)
Handles non-linearities; works well with engineered lags & calendar features
Feature engineering required; risk of leakage without time-series CV
Often strong at category-level when series are noisy
Deep learning
(LSTM/GRU)
Learns complex non-linear patterns; can ingest multiple signals
Needs more data; lower interpretability; harder to validate for policy decisions
Consider only if sufficient history and clear uplift over simpler baselines

Summary of strengths/limitations: Classical time-series (SARIMA, Holt-Winters) are strong for stable seasonality and provide reliable baselines. Modern ML (XGBoost/LSTM) can improve accuracy for heterogeneous category behavior, but must be carefully validated using time-series cross-validation to avoid data leakage.

Objectives & Justification
Objective 1 — Next-Month Category Forecasting:
Build a forecasting model predicting the number of borrows in the coming month in each category (such as adult fiction, children’s, non-fiction, audiobooks) and a ranked list of categories in high demand in the coming month.
Justification: Decisions are made monthly regarding acquisitions and licenses. Category-level forecasting reduces holding times for high-demand categories and reduces purchasing for low-demand categories. Although SARIMA/HW are very effective models for seasonal trends, considerations arise regarding category-level variability and adapting differently across categories and incorporating event features.

Objective 2 — Actionable Insights & Model Improvement Over Baselines:
Identify those drivers improving the forecast reliability: seasonality strength, holiday/event effects, and format shift. Quantify the improvement over the standard baseline methods using time-series validation and error-by-category reporting: Naive Seasonal, SARIMA, and Holt-Winters.
Justification: Past research has shown that simple baselines can be surprisingly competitive in circulation forecasting, so the project needs to demonstrate measurable uplift and explain where/why improvements occur. Linking errors to categories enables actionable planning (e.g., which categories require additional copies or license expansions).

Dataset Link and Base Paper Link
Dataset:
• Seattle Open Data – Checkouts by Title (monthly aggregated): https://data.seattle.gov/Community-and-Culture/Checkouts-by-Title/tmmm-ytt6
• Data.gov catalog entry – Checkouts by Title (Physical Items): https://catalog.data.gov/dataset/checkouts-by-title-physical-items-08293
Base Paper:
• Library borrowing forecasting using ARIMA modelling (Library Philosophy and Practice): https://digitalcommons.unl.edu/libphilprac/1395/

References
Seattle Open Data. (n.d.). Checkouts by Title. Seattle Public Library / City of Seattle Open Data.
Library Philosophy and Practice. (n.d.). ARIMA modelling for predicting book borrowing (DigitalCommons@University of Nebraska–Lincoln).

Books-Only Filter
The code filters the dataset to ONLY rows where material type is Book (case-insensitive). If the dataset uses a different column for material type, provide --book-col.

Model Safeguards
To reduce obvious errors, the app applies:
- Minimum history months per title
- Minimum nonzero months in a recent window
- Optional guardrails that cap predictions to a multiple of recent average
These filters help avoid extreme spikes caused by sparse data or partial months.

Visual Outputs
The Streamlit app shows:
- Key insights (last month total, predicted total, change, top 10 share)
- Predicted top book titles for next month
- Bar chart of top predictions
- Biggest movers (increases/decreases vs last month)
- Historical trend + forecast line chart for a selected title
- Seasonality chart (average by month)
- Total books borrowed over time

Quick Start
1) Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2) Run CLI forecast (SPL dataset example, books only, by title)
python forecast.py --data-url "https://data.seattle.gov/resource/tmmm-ytt6.csv" \
  --year-col checkoutyear --month-col checkoutmonth \
  --book-col materialtype --category-col title --count-col checkouts \
  --model all --max-rows 200000 --max-items 50 --cache-path data/spl_cache.csv

3) Run the web interface
streamlit run web_app.py
