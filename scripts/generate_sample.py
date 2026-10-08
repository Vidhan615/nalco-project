"""Deterministic fictional prices. No real financial prices are embedded."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(615)
    dates = pd.bdate_range("2022-01-03", "2026-04-08")
    n = len(dates)
    change = rng.normal(.00025, .018, n) + .002 * np.sin(np.arange(n) / 40)
    close = 100 * np.exp(np.cumsum(change))
    open_ = np.r_[100, close[:-1]] * np.exp(rng.normal(0, .004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(.002, .018, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(.002, .018, n))
    frame = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                          "close": close, "volume": rng.integers(100_000, 5_000_000, n)})
    target = Path(__file__).resolve().parents[1] / "data/sample/synthetic_nalco.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, float_format="%.6f")
    print(f"Wrote {n} fictional sessions to {target.name}")


if __name__ == "__main__":
    main()
