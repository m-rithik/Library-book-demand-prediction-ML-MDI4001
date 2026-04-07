from __future__ import annotations

import io
import os
import time
from typing import Callable, Dict, Optional

import pandas as pd
import requests


def _append_query(url: str, params: Dict[str, str]) -> str:
    sep = "&" if "?" in url else "?"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{url}{sep}{query}"


def _get_with_retry(
    url: str,
    timeout: int = 120,
    retries: int = 3,
    backoff: float = 2.0,
) -> requests.Response:
    """GET with retry + exponential backoff."""
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last_exc


def _download_socrata(
    url: str,
    limit: int = 100000,
    max_rows: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> pd.DataFrame:
    frames = []
    offset = 0
    target = max_rows or 200000  # estimate for progress
    while True:
        if max_rows is not None:
            remaining = max_rows - offset
            if remaining <= 0:
                break
            batch_limit = min(limit, remaining)
        else:
            batch_limit = limit

        chunk_url = _append_query(url, {"$limit": str(batch_limit), "$offset": str(offset)})
        resp = _get_with_retry(chunk_url)
        text = resp.text.strip()
        if not text:
            break
        chunk = pd.read_csv(io.StringIO(text))
        if chunk.empty:
            break
        frames.append(chunk)
        offset += len(chunk)
        if on_progress:
            on_progress(min(offset / target, 0.99), f"Fetched {offset:,} rows…")
        if len(chunk) < batch_limit:
            break

    if on_progress:
        on_progress(1.0, f"Done — {offset:,} rows loaded.")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_csv(
    url: str,
    max_rows: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> pd.DataFrame:
    if "data.seattle.gov/resource" in url and "$limit" not in url:
        return _download_socrata(url, max_rows=max_rows, on_progress=on_progress)
    if on_progress:
        on_progress(0.3, "Downloading CSV…")
    resp = _get_with_retry(url)
    if on_progress:
        on_progress(1.0, "Download complete.")
    return pd.read_csv(io.StringIO(resp.text), nrows=max_rows)


def _safe_mkdir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_data(
    data_path: Optional[str] = None,
    data_url: Optional[str] = None,
    max_rows: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
        return pd.read_csv(data_path, nrows=max_rows)
    if data_url:
        if cache_path and os.path.exists(cache_path):
            return pd.read_csv(cache_path, nrows=max_rows)
        data = download_csv(data_url, max_rows=max_rows)
        if cache_path:
            _safe_mkdir_for_file(cache_path)
            data.to_csv(cache_path, index=False)
        return data
    raise ValueError("Provide either data_path or data_url.")
