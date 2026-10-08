"""Causal indicators. Every value uses its own row and earlier rows only."""
import numpy as np
import pandas as pd


def atr(df, period=14):
    previous = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - previous).abs(),
                    (df.low - previous).abs()], axis=1).max(axis=1)
    # Explicit SMA ATR convention; different Wilder initializations may differ.
    return tr.rolling(period, min_periods=period).mean()


def supertrend(df, period=10, multiplier=3.0):
    """Final-band recurrence, SMA ATR and close crossings.

    Initialize bearish at the first finite ATR. Preserve prior final bands until
    their reset condition is met; do not replace them unconditionally.
    """
    mid = (df.high + df.low) / 2
    upper = (mid + multiplier * atr(df, period)).to_numpy()
    lower = (mid - multiplier * atr(df, period)).to_numpy()
    close = df.close.to_numpy()
    fu, fl = upper.copy(), lower.copy()
    line = np.full(len(df), np.nan)
    direction = np.full(len(df), np.nan)
    for i in range(period - 1, len(df)):
        if i == period - 1:
            line[i], direction[i] = fu[i], -1
            continue
        fu[i] = upper[i] if upper[i] < fu[i-1] or close[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lower[i] if lower[i] > fl[i-1] or close[i-1] < fl[i-1] else fl[i-1]
        if direction[i-1] < 0:
            direction[i] = 1 if close[i] > fu[i] else -1
        else:
            direction[i] = -1 if close[i] < fl[i] else 1
        line[i] = fl[i] if direction[i] > 0 else fu[i]
    return pd.DataFrame({"supertrend": line, "st_direction": direction,
                         "final_upper": fu, "final_lower": fl}, index=df.index)


def build_features(df):
    df = df.copy()
    delta = df.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi"] = 50
    macd = df.close.ewm(span=12, adjust=False).mean() - df.close.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    mean = df.close.rolling(20).mean()
    std = df.close.rolling(20).std()
    df["bb_position"] = (df.close - (mean - 2 * std)) / (4 * std.replace(0, np.nan))
    df.loc[std == 0, "bb_position"] = 0.5
    df["atr"] = atr(df)
    df["st_direction"] = supertrend(df).st_direction
    for horizon in (1, 5, 20):
        df[f"r{horizon}"] = np.log(df.close).diff(horizon)
    df["vol20"] = df.r1.rolling(20).std()
    df["range"] = (df.high - df.low) / df.close
    # Retain every input session. Warm-up is handled at the experiment boundary.
    return df
