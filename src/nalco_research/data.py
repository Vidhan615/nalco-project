"""Local CSV ingestion with strict validation and source-date alignment."""
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def number(value):
    """Parse numeric values and optional K/M/B suffixes without truncating digits."""
    text = str(value).strip().replace(",", "")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([KMB]?)", text, re.I)
    if not match:
        return np.nan
    return float(match[1]) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[match[2].upper()]


def load_csv(path, *, close_only=False):
    """Accept canonical, Investing.com and prefixed external-series CSV schemas.

    Invalid OHLC is rejected, never silently filled or removed. External research
    may explicitly request close-only mode, retaining an audit of the raw OHLC.
    """
    path = Path(path)
    df = pd.read_csv(path)
    renames = {}
    for column in df:
        clean = column.strip().strip('"').lower()
        if clean == "date":
            renames[column] = "date"
        elif clean in ("price", "close") or clean.endswith("_close"):
            renames[column] = "close"
        elif clean in ("open", "high", "low"):
            renames[column] = clean
        elif any(clean.endswith("_" + field) for field in ("open", "high", "low")):
            renames[column] = clean.rsplit("_", 1)[1]
        elif clean in ("volume", "vol."):
            renames[column] = "volume"
    df = df.rename(columns=renames)
    if df.columns.duplicated().any():
        raise ValueError("Ambiguous columns after normalization")
    required = ["date", "close"] if close_only else ["date", *OHLC]
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="raise")
    if df.date.isna().any() or df.date.duplicated().any():
        raise ValueError("Dates must be nonmissing and unique")
    if df.date.dt.tz is not None or not df.date.eq(df.date.dt.normalize()).all():
        raise ValueError("Expected daily, timezone-naive date labels")
    for column in [*OHLC, "volume"]:
        if column in df:
            df[column] = df[column].map(number)
    fields = ["close"] if close_only else OHLC
    values = df[fields].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite, positive and complete; repair source data")
    if not close_only:
        bad = ((df.low > df.high) | (df.open < df.low) | (df.open > df.high)
               | (df.close < df.low) | (df.close > df.high))
        if bad.any():
            raise ValueError(f"OHLC range violations on {int(bad.sum())} rows")
    if "volume" in df and (df.volume.dropna() < 0).any():
        raise ValueError("Volume cannot be negative")
    return df.sort_values("date").reset_index(drop=True)


def source_fingerprint(path):
    path = Path(path)
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def external_features(base, sources, max_age_days=7):
    """Strictly earlier source dates; block stale inputs instead of hiding gaps.

    Native-market returns are computed before alignment, so local holidays do
    not turn carried prices into manufactured zero-return observations.
    """
    result = base.copy().sort_values("date")
    names = []
    for name, frame in sources.items():
        ext = frame[["date", "close"]].sort_values("date").copy()
        features = []
        for window in (1, 5, 20):
            col = f"{name}_r{window}"
            ext[col] = np.log(ext.close).diff(window)
            features.append(col)
        source_date = f"{name}_source_date"
        ext = ext.rename(columns={"date": source_date}).drop(columns="close")
        result = pd.merge_asof(result, ext, left_on="date", right_on=source_date,
                               direction="backward", allow_exact_matches=False)
        age = (result.date - result[source_date]).dt.days
        result[f"{name}_age_days"] = age
        result.loc[age.gt(max_age_days), features] = np.nan
        names.extend(features)
    return result, names
