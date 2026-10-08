# Methodology

## Decisions and execution

Daily features are calculated after a completed session close. The agent selects
a target exposure (cash or one asset). Rebalancing occurs at the following
session's open, with adverse slippage and a fee on each fill. A maintained long
position carries overnight risk. Equity is marked at every input session close.
The published experiments begin execution on 1 January 2025; the previous close
is included as the initial capital snapshot.

At the predeclared final evaluation session, remaining long positions are sold
at the close with costs. This terminal liquidation is included in both portfolio
equity and the terminal learning reward. This assumes a scheduled close order;
it is not an order decided using that day's completed indicators.

Intraday mode chooses cash, long or short using the previous session's features,
enters at the next open and closes at that same session's close. Daily bars
support this restricted open-to-close experiment only. They do not support
intra-session signals, stop orders or guaranteed stop-loss fills. Shorts use at
most 95% of available equity as one-times notional with a fee reserve; short
sale proceeds cannot finance another position. This is a simplified research
cash ledger, not a reconstruction of a broker's margin/borrow rules. Insolvency
raises an error because daily bars cannot reconstruct margin liquidation.

## Indicators

RSI uses exponentially smoothed gains/losses with alpha 1/14 and adjust=False;
zero-loss rising series return 100 and flat series return 50. MACD uses 12/26
EMAs and a 9-period signal. Bollinger position uses a 20-session mean and sample
standard deviation, with +/-2 standard deviation bands. ATR is explicitly a
simple moving average of true range, not Wilder's alternative smoothing.

Supertrend retains previous final bands until their reset conditions are met,
and switches direction on close crossings. Its recurrence follows the
[TradingView formula description](https://www.tradingview.com/support/solutions/43000634738-supertrend/).
The SMA ATR choice and initial bearish state are explicit conventions; exact
numeric identity with platforms using other ATR initializations is not claimed.
Warm-up rows are trimmed only at the beginning. Missing values after warm-up
are rejected; valuation sessions are never silently dropped.

## Q-learning

This is tabular Q-learning, not a neural-network/DQN model. Each asset contributes
RSI, MACD histogram sign, Bollinger position, Supertrend direction, five-session
log return and twenty-session volatility buckets. Portfolio observations include
the selected asset/flat status, unrealized-return bucket, month-to-date return,
calendar time remaining and monthly-target status. Price, cash and equity remain
continuous internally; discretized observations are an approximation and do
not guarantee a fully Markov state.

The reward is log(E[t+1] / E[t]), once per step, after all trading costs. Entry-to-
exit profit is not rewarded a second time. Training and evaluation share the same
execution, feasible-action and accounting code. Alpha=0.10, gamma=0.95;
epsilon=max(0.05, 0.8*0.98**episode). All ties prefer cash. This can create
cash-heavy policies for unseen states, which is reported rather than interpreted
as successful market timing.

Seeds 7, 42 and 99 are fixed. The initial published corrected runs use 200 training
episodes per seed with no performance-based seed selection. Models, configurations,
source hashes, trades and equity curves are saved. Multi-asset mode trains one
portfolio agent; it never compares uncalibrated Q-values of independent agents.
Every asset must have exactly the same complete valuation calendar.

## Monthly targets and returns

The first day's return in a new month belongs to that new month. Its capital
base is the previous month's final observed close, before the new day's gap.
After a target is observed at a close, the next action is cash (exit at the next
open) until the month changes. A gap can move the eventual exit below the target.
The monthly target is a trading rule, not a guaranteed result.

Monthly tables use each month's final observed equity and the previous month's
final equity. An initial partial month and a truncated final month are labelled.
The completion convention conservatively requires the horizon to reach calendar
month-end; an exchange calendar is not inferred from the file. Thus a month
ending on a weekend can be flagged partial when its last trading session is the
data cutoff. Target crossing during a month and month-end target attainment
are distinct metrics; only the latter is in the monthly return table.

## External inputs and prediction

Optional LME, DXY, Hongqiao and Chalco closes are aligned using source date
strictly earlier than the NALCO decision date. Exact release timestamps are
unknown, so potentially usable same-day information is deliberately excluded.
Source-native one/five/twenty-observation returns are computed before alignment.
Source dates and ages are saved. An age above seven calendar days blocks a run;
it is not silently imputed or dropped. Only closes are used from the external
series; malformed LME OHLC remains a documented source-data problem.

The baseline target is the next session's open-to-close log return. Fixed-penalty
ridge regression (penalty 10) uses only training-fit standardization. Models
compare NALCO-only returns/volatility/range with LME/DXY and all four external
series, alongside zero-return and historical-training-mean baselines. All models
use identical eligible observations. Training labels whose endpoints reach the
test period are excluded. Direction accuracy measures sign agreement (strictly
positive vs non-positive); magnitude, costs and turnover still matter.

External inputs can optionally enter the Q-state as coarse return signs. This
increases state sparsity and does not imply improved performance. The continuous
ridge models provide the clearer incremental-value comparison.

Correlations are descriptive natural-log return correlations. Shared price
dates are intersected before computing each pair's changes, so return endpoints
match. Weekly/monthly values use last shared observations and exclude a final
incomplete period. Different markets' closes are not simultaneous; correlation
does not establish a forecasting lead or causal relationship.

## Metrics and benchmarks

Returns include illustrative fees of 10 basis points and slippage of 5 basis
points per fill. These are reproducible assumptions, not a verified all-in Indian
charge schedule. CAGR uses elapsed calendar days/365.25; Sharpe uses daily simple
returns, 252 annualization and a zero risk-free rate. A constant cash curve has
undefined Sharpe, represented as null. Drawdown includes the initial capital base.
Benchmarks use the same start/end dates, capital, 95% initial allocation, integer
shares, cash residual, costs and terminal liquidation. Dividends and cash interest
are excluded. Intraday policies have lower overnight exposure than buy-and-hold,
so the comparison alone cannot establish superior risk-adjusted performance.

Chronological separation is enforced, but the 2025-April 2026 period was already
examined in the original project. It is a reused historical evaluation window,
not a fresh final deployment test. No profitability, causal or statistical-
significance claim is made from a single run or a favourable subset of seeds.
