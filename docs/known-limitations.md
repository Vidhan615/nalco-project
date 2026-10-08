# Known limitations

## Source data cannot be repaired by guessing

The supplied files end on **8 April 2026**. They are not a live market feed. Raw
data is kept local because source licensing and redistribution rights have not
been established. Exact source instruments, publication times and corporate-
action adjustment conventions are incomplete. Derived historical results remain
conditional on those supplied prices.

The original audit found 1,088 of 1,583 LME rows with inconsistent OHLC ranges.
The corrected external research uses closes only and says so explicitly; this
does not independently certify those closes. DXY and Hongqiao also have some
OHLC anomalies. Hindalco has seven missing OHLC cells across 28-30 January 2026.
The strict equity loader rejects that input. No prices are fabricated, no
valuation calendar is reduced to an intersection, and no corrected Hindalco
backtest is published until the source prices are recovered and checked.

Exchange calendars, corporate actions, dividends, short availability, margin,
liquidity, realistic closing auctions and broker-specific charges remain future
data/modeling work. Weekend rows are preserved pending exchange-calendar
verification because special sessions can occur.

## What the corrected implementation changes

| Original defect | Corrected behavior |
|---|---|
| Supertrend discarded prior final bands | Retained-band recurrence and crossing tests |
| Close-based features filled at that same close | Next-session open fills |
| Full-history replay labelled as generalization | Strict chronological train/evaluation boundary |
| Monthly return used the next month's first close | Prior month ends at its own final observed close |
| "Intraday" shorts crossed dates | Prior-session decision, next open entry, same-session close exit |
| RL states omitted portfolio/target information | Portfolio, unrealized return and monthly state included |
| Price movement rewarded repeatedly | One stepwise log-equity reward, including terminal costs |
| No cost assumptions | Explicit adverse slippage and per-fill fees |
| Unseeded training and missing configurations | Seeds, Q-tables, source hashes and run manifests saved |
| Separate agents' Q-values ranked across stocks | Single portfolio-aware policy with a common reward |
| Macro inputs absent from RL experiments | Optional lagged external return signs and separate continuous baselines |
| Hardcoded machine paths | Explicit portable CSV and output arguments |
| Manual confidence scores described as probabilities | No calibrated probability or success-rate claim |

The original scripts and reports remain preserved in the source workspace. The
repository publishes the corrected shared package and selected original audit
tables. Original statistics are under `results/original-audit/`; corrected runs
are under `results/historical/`. Never combine them as one performance record.

## Remaining research limitations

Tabular discretization can lose useful information and create sparse/unseen
states. The implementation is intentionally small and interpretable, not evidence
of a profitable trading policy. State buckets and reward parameters are research
choices. More seeds, walk-forward evaluations and transaction-cost sensitivity
are useful checks but do not remove source-data uncertainty or test-window reuse.

The project does not contain a broker connection, continuous live service,
production trading infrastructure, independently measured 60% monitoring saving
or a guaranteed Sharpe ratio. Report these capabilities only after implementing
and measuring them. A monthly return target is never an achieved-return promise.
