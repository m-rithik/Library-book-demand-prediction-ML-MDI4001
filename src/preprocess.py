from __future__ import annotations

from typing import Optional, Sequence

import difflib
import pandas as pd

from .config import (
    DEFAULT_BOOK_COL_CANDIDATES,
    DEFAULT_CATEGORY_COL_CANDIDATES,
    DEFAULT_COUNT_COL_CANDIDATES,
    DEFAULT_DATE_COL_CANDIDATES,
    DEFAULT_MONTH_COL_CANDIDATES,
    DEFAULT_YEAR_COL_CANDIDATES,
)


def _format_available_columns(df: pd.DataFrame, limit: int = 25) -> str:
    cols = list(df.columns)
    shown = ", ".join(cols[:limit])
    if len(cols) > limit:
        shown += f", ... (+{len(cols) - limit} more)"
    return shown


def _pick_column(
    df: pd.DataFrame,
    provided: Optional[str],
    candidates: Sequence[str],
    label: str,
) -> Optional[str]:
    if provided:
        if provided in df.columns:
            return provided
        for col in df.columns:
            if col.lower() == provided.lower():
                return col
        matches = difflib.get_close_matches(provided, df.columns, n=5, cutoff=0.6)
        hint = ""
        if matches:
            hint += " Did you mean: " + ", ".join(matches) + "?"
        available = _format_available_columns(df)
        raise ValueError(f"{label} column not found: {provided}.{hint} Available columns: {available}.")

    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        key = col.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _filter_only_books(df: pd.DataFrame, book_col: Optional[str]) -> pd.DataFrame:
    book_col = _pick_column(df, book_col, DEFAULT_BOOK_COL_CANDIDATES, "Book filter")
    if not book_col:
        available = _format_available_columns(df)
        raise ValueError(
            "No material type column found to filter books. Provide --book-col. "
            f"Available columns: {available}."
        )

    values = df[book_col].astype(str).str.strip()
    lower = values.str.lower()

    # Primary filter: exact BOOK
    mask = lower == "book"

    # Fallback: values containing book but not e-book/audiobook
    if mask.sum() == 0:
        mask = lower.str.contains("book") & ~lower.str.contains("ebook") & ~lower.str.contains("audiobook")

    if mask.sum() == 0:
        unique = ", ".join(sorted(values.dropna().unique())[:15])
        raise ValueError(
            "Book filter removed all rows. "
            f"Check values in {book_col}. Sample values: {unique}"
        )

    return df[mask].copy()


def build_monthly_series(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    year_col: Optional[str] = None,
    month_col: Optional[str] = None,
    category_col: Optional[str] = None,
    count_col: Optional[str] = None,
    book_col: Optional[str] = None,
    only_books: bool = True,
) -> pd.DataFrame:
    data = df.copy()

    if only_books:
        data = _filter_only_books(data, book_col)

    category_col = _pick_column(data, category_col, DEFAULT_CATEGORY_COL_CANDIDATES, "Category")
    if not category_col:
        available = _format_available_columns(data)
        raise ValueError(
            "No category column found. Specify --category-col. "
            f"Available columns: {available}."
        )

    count_col = _pick_column(data, count_col, DEFAULT_COUNT_COL_CANDIDATES, "Count")

    date_col = _pick_column(data, date_col, DEFAULT_DATE_COL_CANDIDATES, "Date")
    if date_col:
        data["month"] = pd.to_datetime(data[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    else:
        year_col = _pick_column(data, year_col, DEFAULT_YEAR_COL_CANDIDATES, "Year")
        month_col = _pick_column(data, month_col, DEFAULT_MONTH_COL_CANDIDATES, "Month")
        if not year_col or not month_col:
            available = _format_available_columns(data)
            raise ValueError(
                "No date column found. Provide --date-col or year/month columns. "
                f"Available columns: {available}."
            )
        data["month"] = pd.to_datetime(
            {
                "year": pd.to_numeric(data[year_col], errors="coerce"),
                "month": pd.to_numeric(data[month_col], errors="coerce"),
                "day": 1,
            },
            errors="coerce",
        )

    data = data.dropna(subset=["month", category_col])

    if count_col and count_col in data.columns:
        data["count"] = pd.to_numeric(data[count_col], errors="coerce").fillna(0)
    else:
        data["count"] = 1

    monthly = (
        data.groupby(["month", category_col], as_index=False)["count"]
        .sum()
        .rename(columns={category_col: "category"})
    )
    return monthly.sort_values(["category", "month"]).reset_index(drop=True)


def complete_monthly_index(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    frames = []
    for category, group in monthly.groupby("category"):
        group = group.sort_values("month")
        full_index = pd.date_range(group["month"].min(), group["month"].max(), freq="MS")
        filled = (
            group.set_index("month")
            .reindex(full_index, fill_value=0)
            .reset_index()
            .rename(columns={"index": "month"})
        )
        filled["category"] = category
        frames.append(filled)
    return pd.concat(frames, ignore_index=True).sort_values(["category", "month"]).reset_index(drop=True)
