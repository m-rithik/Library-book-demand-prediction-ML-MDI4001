from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLS = [
    "month_sin", "month_cos", "year_norm",
    "lag1", "lag2", "lag3", "avg3", "avg6", "lag12",
]


def _build_features(group: pd.DataFrame, lag12_fill: float = 0.0) -> pd.DataFrame:
    data = group.sort_values("month").copy()
    m = data["month"].dt.month
    data["month_sin"]  = np.sin(2 * np.pi * m / 12)
    data["month_cos"]  = np.cos(2 * np.pi * m / 12)
    data["year_norm"]  = data["month"].dt.year - 2010
    data["lag1"]  = data["count"].shift(1)
    data["lag2"]  = data["count"].shift(2)
    data["lag3"]  = data["count"].shift(3)
    data["avg3"]  = data["count"].shift(1).rolling(3).mean()
    data["avg6"]  = data["count"].shift(1).rolling(6).mean()
    data["lag12"] = data["count"].shift(12).fillna(lag12_fill)
    return data


def random_forest_forecast(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()

    # Build cross-category training set
    frames = []
    for _, grp in monthly.groupby("category"):
        frames.append(_build_features(grp))
    full = pd.concat(frames, ignore_index=True)

    # Core features that must not be NaN
    core = ["month_sin", "month_cos", "year_norm", "lag1", "lag2", "lag3", "avg3", "avg6"]
    full = full.dropna(subset=core + ["count"])
    lag12_mean = float(full["lag12"].mean()) if full["lag12"].notna().any() else 0.0
    full["lag12"] = full["lag12"].fillna(lag12_mean)

    if len(full) < 10:
        # Not enough cross-category data — fall back to recent average
        forecasts = []
        for cat, grp in monthly.groupby("category"):
            grp = grp.sort_values("month")
            lm = grp["month"].iloc[-1]
            nm = lm + pd.offsets.MonthBegin(1)
            pred = float(grp["count"].tail(3).mean()) if len(grp) >= 3 else float(grp["count"].iloc[-1])
            forecasts.append({"category": cat, "month": nm, "predicted": pred})
        return pd.DataFrame(forecasts)

    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    model.fit(full[FEATURE_COLS], full["count"])

    forecasts = []
    for cat, grp in monthly.groupby("category"):
        grp = grp.sort_values("month")
        lm = grp["month"].iloc[-1]
        nm = lm + pd.offsets.MonthBegin(1)

        feat = _build_features(grp, lag12_fill=lag12_mean).iloc[-1]
        if feat[core].isna().any():
            pred = float(grp["count"].tail(3).mean()) if len(grp) >= 3 else float(grp["count"].iloc[-1])
        else:
            pred = float(model.predict(feat[FEATURE_COLS].to_frame().T)[0])

        forecasts.append({"category": cat, "month": nm, "predicted": pred})

    return pd.DataFrame(forecasts)
