# Data

## Public, runnable example

`sample/synthetic_nalco.csv` contains **fictional** daily OHLCV generated with
NumPy seed 615. It is not NALCO history and must not support market-performance
claims. Regenerate it with `python scripts/generate_sample.py`.

## Local historical inputs

Raw historical files are excluded from Git. Obtain data through a source you
are entitled to use and put it under `data/raw/`, or pass another local path.
The original raw files are preserved locally; they are not redistributed here.

Canonical equity CSV schema:

```csv
date,open,high,low,close,volume
2025-01-01,100,103,99,102,1000000
```

The example row is illustrative. Dates are unique, timezone-naive daily labels;
prices must be finite, positive and within the supplied low/high range. Volume
is optional for the current models. Investing.com `Date,Price,Open,High,Low,Vol.`
and standard `Date,Open,High,Low,Close,Volume` formats are accepted. US-style
month/day/year is used for slash dates. Numeric commas and K/M/B suffixes are
supported. Chronological sorting is explicit; missing prices or duplicates fail.

External files may use `Date,LME_Close` and corresponding prefixed OHLC columns,
or the canonical schema. External research requests close-only loading explicitly.
That permits malformed unused OHLC columns without concealing their documented
quality issues. Closing prices must still be complete and positive.

| Suggested local filename | Series | Information still needed |
|---|---|---|
| `nalco_daily.csv` | NALCO Indian equity | Verified exchange/ticker, vendor, adjustments, timezone |
| `hindalco.csv` | Hindalco Indian equity | Recover January 2026 missing observations before use |
| `vedanta.csv` | Vedanta Indian equity | Verified ticker and corporate-action treatment |
| `lme.csv` | Aluminium prices | Cash/three-month/other instrument, units, source and close timestamp |
| `dxy.csv` | Dollar index | Exact instrument, vendor and daily observation definition |
| `hongqiao.csv` | China Hongqiao | Exact listing/ticker, currency, timezone and adjustment treatment |
| `chalco.csv` | Aluminium Corp of China | Exact listing/ticker, currency, timezone and adjustment treatment |

The supplied Indian history begins in January 2020 and ends on 8 April 2026.
Source hashes and actual experiment dates are in each historical run manifest.
Matching hashes reproduce the exact local snapshot; a hash does not certify the
data provider. A later independent price snapshot may legitimately differ.

For external features, all source dates must precede their decision dates and
must be at most seven calendar days old. Multi-asset trading requires identical
complete calendars for all assets after initial warm-up. Repair missing source
observations rather than dropping sessions or forward-filling tradable OHLC.

## Rights

The repository's MIT license applies to its source code and synthetic example.
It does not grant rights to any third-party market data. Historical raw data,
CV contact details and unrelated internship documents are not included.
