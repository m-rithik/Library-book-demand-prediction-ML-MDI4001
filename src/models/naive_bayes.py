from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


FEATURE_COLS = ["month_num", "year", "lag1", "avg3", "avg6", "lag12"]


def _build_features(group: pd.DataFrame) -> pd.DataFrame:
    data = group.sort_values("month").copy()
    data["month_num"] = data["month"].dt.month
    data["year"]      = data["month"].dt.year
    data["lag1"]      = data["count"].shift(1)
    data["avg3"]      = data["count"].shift(1).rolling(3).mean()
    data["avg6"]      = data["count"].shift(1).rolling(6).mean()
    data["lag12"]     = data["count"].shift(12)
    return data


def _make_bins(y: np.ndarray) -> np.ndarray:
    nonzero = y[y > 0]
    if len(nonzero) < 20:
        return np.array([-0.1, 0.0, 5.0, 20.0, np.inf])
    q33, q66 = np.quantile(nonzero, [0.33, 0.66])
    q33 = max(q33, 1.0)
    q66 = max(q66, q33 + 1.0)
    return np.array([-0.1, 0.0, q33, q66, np.inf])


def _bin_targets(y: np.ndarray, bins: np.ndarray) -> np.ndarray:
    labels = pd.cut(y, bins=bins, labels=False, include_lowest=True)
    return np.asarray(labels, dtype=int)


def _class_medians(y: np.ndarray, classes: np.ndarray, n_classes: int) -> Dict[int, float]:
    medians: Dict[int, float] = {}
    overall = float(np.median(y)) if len(y) else 0.0
    for c in range(n_classes):
        values = y[classes == c]
        if len(values) == 0:
            medians[c] = overall
        else:
            medians[c] = float(np.median(values))
    return medians


def _train_global_model(train_df: pd.DataFrame) -> Tuple[GaussianNB, np.ndarray, Dict[int, float]]:
    y = train_df["count"].to_numpy()
    bins = _make_bins(y)
    classes = _bin_targets(y, bins)
    X = train_df[FEATURE_COLS].to_numpy()
    model = GaussianNB()
    model.fit(X, classes)
    medians = _class_medians(y, classes, len(bins) - 1)
    return model, bins, medians


def naive_bayes_forecast(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()

    frames = []
    for _, group in monthly.groupby("category"):
        data = _build_features(group)
        frames.append(data)
    full = pd.concat(frames, ignore_index=True)

    # Drop rows that lack the core lag/rolling features (lag1, avg3, avg6).
    # For lag12 (same-month-last-year) use mean imputation so short series
    # still contribute training samples rather than being silently dropped.
    core_cols = ["month_num", "year", "lag1", "avg3", "avg6"]
    full = full.dropna(subset=core_cols + ["count"])
    lag12_mean = full["lag12"].mean() if full["lag12"].notna().any() else 0.0
    full["lag12"] = full["lag12"].fillna(lag12_mean)

    if len(full) < 20:
        # Fallback to recent average if there is not enough training data.
        forecasts = []
        for category, group in monthly.groupby("category"):
            group = group.sort_values("month")
            last_month = group["month"].iloc[-1]
            next_month = last_month + pd.offsets.MonthBegin(1)
            pred = float(group["count"].tail(3).mean()) if len(group) >= 3 else float(group["count"].iloc[-1])
            forecasts.append({"category": category, "month": next_month, "predicted": pred})
        return pd.DataFrame(forecasts)

    model, bins, medians = _train_global_model(full)

    forecasts = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month")
        last_month = group["month"].iloc[-1]
        next_month = last_month + pd.offsets.MonthBegin(1)

        features = _build_features(group).iloc[-1].copy()
        # Impute lag12 with the global mean used during training
        if pd.isna(features["lag12"]):
            features["lag12"] = lag12_mean
        core_cols_pred = ["month_num", "year", "lag1", "avg3", "avg6"]
        if features[core_cols_pred].isna().any():
            recent_avg = group["count"].tail(3).mean()
            pred = float(recent_avg) if recent_avg > 0 else float(group["count"].iloc[-1])
        else:
            X_pred = features[FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
            pred_class = int(model.predict(X_pred)[0])
            pred = medians.get(pred_class, float(group["count"].tail(3).mean()))

        forecasts.append({"category": category, "month": next_month, "predicted": float(pred)})

    return pd.DataFrame(forecasts)
