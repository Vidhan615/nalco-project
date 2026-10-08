> **Legacy audit, 5 October 2026.** This describes the original implementation, not the corrected package in this repository. Original source and raw files remain preserved locally. Selected cited tables are published under `results/original-audit/`; other legacy artifact names are references only. See [the correction record](../known-limitations.md) and [current methodology](../methodology.md) for the replacement implementation.

# NALCO project analysis

Audit date: 5 October 2026. Historical data ends on 8 April 2026.

**The files support a relationship between aluminium and aluminium-company stocks, especially over longer periods. They do not yet establish a reliable forecasting or trading advantage.** The strongest immediate result is sector co-movement. The existing reinforcement-learning backtests do not incorporate your LME, dollar-index or Chinese-company inputs, and several implementation errors undermine their interpretation.

All original data, code and results were preserved. New analysis, reproducible audit scripts and supporting tables are in this `project_audit` folder. Statistical findings are conditional on the supplied closing prices: their external provenance and adjustment conventions could not be verified from the folder, and the LME OHLC columns have substantial internal inconsistencies.

## 1. What the project currently does

| Component | Actual role | Assessment |
|---|---|---|
| `nalco_analysis.py` | Latest-snapshot technical indicators, macro trend rules, BUY/HOLD/SIP/EXIT/SHORT labels | A hand-written signal engine; no trained forecast or calibrated confidence probability |
| `rl_backtest.py` | Tabular Q-learning; chronological 80/20 split | Training and testing are separated, but same-close execution and simulator design need correction |
| `rl_backtest_full.py` | Train on first 40%; replay the entire history | Combines training-period and later-period performance; not wholly out-of-sample |
| `rl_backtest_5pct_monthly.py` | Train before 2025; evaluate from 2025; stop opening positions after a monthly target | Better chronological separation, but monthly attribution is wrong |
| `multi_stock_engine.py`, `run_multi_backtest.py` | Independently trained NALCO, Hindalco and Vedanta agents | Still stock-only technical states; no cross-market predictors |
| `shuffle_engine.py`, `run_shuffle_bot.py` | Choose one stock using separate agents' BUY Q-values | Q-values are not calibrated expected stock returns; dates are reduced to their intersection |
| `rl_intraday_short/` | Five-action agent adding SHORT and COVER | Daily shorts are covered at a later day's close; the saved trades are overnight trades |
| CSV, JSON, HTML and PDF outputs | Saved equity curves, transactions and reports | Saved total returns/drawdowns reconcile, but some labels and conclusions overstate the evidence |

The agents use RSI buckets, MACD crossover, MACD histogram sign, Bollinger position, Supertrend direction and five-session return buckets. Computing ATR or volume features does not mean the agent observes them: they are omitted from its state. Five-session return is momentum, not volatility. This is **tabular Q-learning**, with no neural network, despite the main report's “Deep Q-Learning” description.

## 2. Data inventory and reliability

| Series | Rows | First date | Last date | Main finding |
|---|---:|---|---|---|
| NALCO daily | 1,556 | 2020-01-01 | 2026-04-08 | No duplicate dates, missing OHLC values or OHLC range violations |
| Hindalco daily | 1,556 | 2020-01-01 | 2026-04-08 | Seven missing OHLC cells across three January 2026 rows |
| Vedanta daily | 1,557 | 2020-01-01 | 2026-04-08 | No duplicate dates, missing OHLC values or OHLC range violations |
| LME aluminium | 1,583 | 2020-01-02 | 2026-04-08 | 1,088 rows violate one or more OHLC consistency rules |
| Dollar index | 1,629 | 2020-01-01 | 2026-04-08 | Opening price outside the high/low range on 15 rows |
| China Hongqiao | 1,540 | 2020-01-02 | 2026-04-08 | One OHLC range violation |
| Aluminium Corp of China / Chalco | 1,540 | 2020-01-02 | 2026-04-08 | No duplicate dates, missing OHLC values or range violations |
| NALCO weekly | 327 | 2020-01-05 | 2026-04-05 | Sunday period labels; latest weekly bar is partial |
| NALCO monthly | 76 | 2020-01-01 | 2026-04-01 | Month-start labels; April 2026 is partial |

The LME issue is more serious than an occasional opening-price discrepancy: **420 rows have Low greater than High; 1,059 have Close outside the supplied High/Low range; 728 have Open outside that range.** These counts overlap. A mixed contract or inconsistent column definition is possible, but the folder does not establish the cause. Verify the export and instrument before using LME High/Low for indicators. The close-to-close changes agree with the supplied percentage-change column within 0.1 percentage point; this is an internal consistency check, not independent validation of the closes.

Hindalco has missing prices on 28–30 January 2026. Rolling indicators propagate those missing values beyond those three dates. Its saved test curve contains **292 observations**, versus **314** for NALCO and Vedanta. The shuffle bot inherits the shorter date intersection. Removing missing dates can skip real price risk and trading opportunities; recover the original prices rather than blindly filling OHLC data.

Indian daily files include weekend dates. Do not automatically remove them: special trading sessions can occur on weekends. Validate against exchange calendars. Conversely, weekly Sunday labels identify periods and are not evidence of Sunday trading.

There are no adjusted-close fields, dividend ledgers, split histories, exact instrument tickers, contract tenor definitions, observation timestamps or data-source manifests. Clarify whether LME means cash settlement, three-month futures or another series, and which Chalco listing the file represents. Keep raw and adjusted stock prices distinct; benchmark total returns require dividends as well as price changes.

The seven files inside `csv files.zip` match the extracted CSV files byte-for-byte. The archive is a duplicate, not a second independent data source. The data is historical through April 2026; latest-snapshot signals cannot describe October 2026 conditions.

## 3. Testing your correlation theory

I used natural-log returns rather than relying on raw price levels. For each pair, I intersected price dates first and then calculated returns between the same calendar endpoints. A “daily” observation can span extra sessions around differing holidays; it is not a synchronized intraday measurement. Weekly and monthly results use the last shared observation in each period, excluding the incomplete final period.

| Relationship with NALCO | Daily return correlation | Weekly return correlation | Monthly return correlation |
|---|---:|---:|---:|
| Hindalco | **0.666** | 0.643 | 0.679 |
| Vedanta | **0.610** | 0.593 | 0.608 |
| LME aluminium | **0.247** | **0.423** | **0.604** |
| China Hongqiao | 0.299 | 0.355 | 0.306 |
| Chalco | 0.356 | 0.477 | 0.507 |
| Dollar index | -0.049 | -0.131 | -0.292 |

Daily NALCO–LME correlation uses 1,508 observations. Its 95% percentile moving-block bootstrap interval is approximately **0.196 to 0.297**, using 2,000 resamples with 20-observation blocks. Calculating each exchange's native daily returns and then joining common dates gives **0.222** instead of 0.247, so the calendar convention changes the number but not the broad interpretation.

The monthly NALCO–LME result uses **74 complete return observations**, a much smaller sample. It is evidence of stronger monthly co-movement, not a test of monthly forecasting skill. There is no multiple-comparison-adjusted significance claim in this audit.

The aluminium relationship also appears across the other Indian stocks:

| LME relationship | Daily correlation | Monthly correlation |
|---|---:|---:|
| NALCO | 0.247 | 0.604 |
| Hindalco | 0.275 | 0.651 |
| Vedanta | 0.196 | 0.464 |

Raw price-level correlations look much stronger: NALCO–LME is 0.670, NALCO–Hongqiao 0.861 and NALCO–Chalco 0.888. Shared long-term trends can make those figures misleading for trading. The return correlations are more relevant to changes in investor wealth.

The relationship is not constant. NALCO–LME daily correlation is about 0.154 in 2024, 0.191 in 2025 and 0.441 in the partial 2026 sample. NALCO's correlation with Hongqiao rises from 0.251 before 2025 to 0.527 from 2025 onward; Chalco rises from 0.322 to 0.515. These changes justify rolling estimates rather than a permanent fixed coefficient. They do not identify a causal driver: broad equity-market returns are absent, so commodity exposure and shared market risk have not been separated.

**Your theory should be stated as: aluminium-company stocks share sector and commodity exposure, with a relationship that varies by horizon and market conditions.** The supplied data does not support assuming a near one-to-one daily relationship or a dependable next-day lead.

## 4. Does earlier external information improve prediction?

I added a separate research experiment without changing the original models. It compares fixed-penalty linear ridge models using:

1. NALCO's own one-, five- and twenty-session returns, twenty-session volatility and daily price range.
2. Those features plus LME and DXY returns over the same lookbacks.
3. Those features plus Hongqiao and Chalco returns.

Models use penalty 10, training-set-only standardization and no parameter tuning. For the fixed holdout, training ends before 2025; labels whose future endpoint reaches the test period are excluded from training. The historical 2025–April 2026 window has already been used repeatedly in this project, so it is a chronological holdout for this experiment rather than a fresh, untouched final deployment test.

External data is joined using a strict rule: **external source date must be earlier than the NALCO decision date**. This is conservative and can discard useful same-day information from markets that have already closed. Exact release timestamps would allow a more efficient availability rule. A latest close is carried forward across holidays; associated returns can therefore be zero while a market is closed. Maximum source age is recorded in the machine-readable results.

This timing matters: LME states that its official prices are published between 12:20 and 13:25 London time. Those prices arrive after India's regular close, so blindly joining same-calendar-day values would leak later information into an Indian close-time decision. The exact timestamp of the supplied `LME_Close` remains unspecified. [LME Official Prices](https://www.lme.com/Market-data/Reports-and-data/LME-Official-Prices)

For next-close forecasts, 313 test observations produced:

| Model | Forecast RMSE, percentage points of log return | Direction accuracy |
|---|---:|---:|
| Historical training mean | **2.636** | **53.67%** |
| NALCO features only | 2.649 | 51.44% |
| NALCO + LME + DXY | 2.658 | 51.76% |
| NALCO + all four external series | 2.647 | 53.04% |

The training-mean forecast is positive here, so its directional accuracy is the same as always predicting “up.” The full external model narrowly reduces RMSE relative to the NALCO-only model, but fails to beat this simple baseline. The block-bootstrap interval for its improvement relative to NALCO-only includes zero.

I also tested the next session's **open-to-close** return, a target compatible with deciding after today's close and entering at tomorrow's open. The full external model reaches 54.95% directional accuracy versus 51.76% for always predicting up, but RMSE is 2.323 versus 2.309 for a zero-return forecast. Its loss-difference interval relative to NALCO-only includes zero. Accuracy alone does not establish profitability; return magnitude, costs and position sizing matter.

For five-session close-to-close returns, the full external model gives RMSE **5.660**, compared with **5.701** for the training mean: a modest improvement of about 1.44% in squared-error terms. Its improvement versus NALCO-only is not conclusive under the block bootstrap. For twenty-session returns, it performs worse: RMSE **11.409** versus **10.787** for the training mean. Longer-horizon labels overlap, so their observation counts do not represent independent trades; the bootstrap intervals are exploratory and sensitive to block length.

The output includes annual forward evaluations for 2023, 2024, 2025 and partial 2026, individual predictions, lagged correlations and coefficients. These are diagnostic baselines, not an exhaustive search and not a direct comparison against the existing RL policy. Failure of these linear models does not prove that no useful nonlinear relationship exists. It does establish that the theory alone is insufficient evidence for an exploitable forecast.

## 5. Material code and backtest findings

### A. Supertrend is broken and constant

In `multi_stock_engine.py`, the lower-band expression simplifies to the current lower band in both branches; the upper-band expression likewise simplifies to the current upper band. Previous final bands are not retained as intended. The same implementation is copied into the other indicator scripts.

Recomputing the supplied NALCO data produces **1,537 bearish states and zero bullish states** after warm-up. This is verified from the feature builder, not inferred from one chart. Consequently the Supertrend state bit is constant and the `supertrend_bullish` rule in `nalco_analysis.py` blocks BUY throughout the supplied NALCO history. Replace the implementation, verify its band recurrence and crossings against a trusted reference, then regenerate every affected result.

### B. Full-period results include training data

`rl_backtest_full.py` trains on the first 40% and then evaluates `df_full`, which includes those same rows. The actual training range is **28 January 2020–15 July 2022**, not through December 2022 as reported. The entire-history result can be a descriptive replay, but cannot be called wholly out-of-sample. Evaluate only dates after training when estimating generalization.

### C. Actions fill at the same close used to calculate the state

The environment reads indicators based on today's completed close and executes BUY/SELL at that close. A completed closing-price indicator cannot ordinarily be observed and then filled retrospectively at the same official closing print. Use a documented decision cutoff with a subsequent tradable fill, usually the next open for daily data. Include gap risk and execution costs.

### D. “Intraday” shorts cross dates

`environment.py` waits until the date changes before covering. The cover helper uses that later row's `Price`, i.e. its close, despite the comment mentioning the open. **All 10 saved short trades have different entry and exit dates.** Several span weekends. For example, the 1 January 2026 entry at 314.60 is covered on 2 January at 330.30, realizing -2,700.40 in the saved log.

The existing constraint checker can also miss a position closed at the next row before the snapshot is recorded. Check entry/exit timestamps and end-of-session positions. Genuine intraday strategy evaluation needs intraday bars or a deliberately restricted prior-day-signal/next-day-open-to-close model.

### E. Monthly returns use the next month's first close

`rl_backtest_5pct_monthly.py` marks the old month using the current row after its date has entered the new month. The multi-stock, shuffle and short environments repeat this pattern. It misattributes the first new session's price movement and resets the monthly target against the wrong value.

For the original 5% strategy:

| Month | Reported return | Correct calendar return |
|---|---:|---:|
| April 2025 | -7.60% | **-9.51%** |
| May 2025 | 9.83% | **12.14%** |
| July 2025 | 2.17% | **4.64%** |
| August 2025 | 5.01% | **2.54%** |

Thus calendar months returning at least 5% fall from **8/16 to 7/16**. Excluding partial April 2026 gives **7/15 completed months**. This correction reattributes the saved equity curve; it does not rerun the policy with corrected month-start logic. A rerun could change trades and total returns. Also distinguish “target crossed at some point during the month” from “calendar month ended above target”; they are different metrics.

### F. The RL state does not describe the portfolio

The state omits flat/long/short status, entry price, unrealized profit, monthly target progress and time remaining in the month. The same market tuple therefore corresponds to different valid actions and rewards. External action masks prevent some invalid orders but do not restore the missing information needed by the Q-table.

MACD crossover and histogram sign are identical on every usable NALCO row, so two nominal inputs duplicate one signal. Together with constant Supertrend, only **69 distinct market tuples** appear in the supplied usable NALCO data. Audit feature usefulness before increasing model complexity.

### G. Rewards and execution assumptions need redesign

HOLD earns a scaled price-change reward; SELL earns the entire entry-to-exit return again. The reward is not a consistent stepwise portfolio-return objective, can reward the same price movement twice and is independent of actual capital exposure. Forced final liquidation adds to reported profit but does not consistently feed that final trade outcome into learning.

Training and inference also sanitize repeated BUY actions differently in the long-only engines: training can leave an invalid BUY with zero reward, while inference changes it to HOLD and awards holding reward. Apply identical feasible-action handling in both paths.

Stop-loss and take-profit values are printed in trade logs but are not enforced by the environment. Trades generally allocate essentially all available cash. No brokerage, slippage, exchange charges, taxes on transactions, short borrow/margin friction or dividends are modeled. Reported Sharpe uses zero risk-free return. Cost assumptions should be documented and stressed, rather than silently treating gross returns as net performance.

### H. Results are not reproducible training evidence

The training scripts set neither Python nor NumPy random seeds, do not preserve library versions and do not save all agents/configurations. Similar NALCO 5%-target runs produce **58.20%** and **12.64%** total returns. Their disparity is a warning about training variability and result provenance; a single attractive run is insufficient. Evaluate multiple seeds and report the distribution, with validation separated from the final test.

### I. Shuffle Q-values do not rank expected stock returns

The selector compares raw Q-values from separately trained agents. Those values represent discounted, shaped rewards under different stock histories and state coverage, not calibrated prospective investment returns. Use a common prediction target and calibration, or a portfolio-aware policy trained across assets. Preserve all portfolio valuation dates even when one asset's data is unavailable.

### J. Snapshot explanations overstate what is measured

Confidence values such as 65, 70 or 85 are hand-assigned scores, not measured success probabilities. An equity's rise is not a direct measurement of Chinese physical demand. The divergence routine takes separate price and RSI minima rather than RSI at matched price pivots; its bearish branch compares lows while describing highs. “No trades” can follow Q-table defaults or a learned preference for HOLD, so it is not sufficient evidence that the agent recognized a bad market.

## 6. What saved performance actually shows

Saved JSON total returns and maximum drawdowns match independently recomputed equity curves for all ten metrics files. This confirms arithmetic consistency, not valid execution or prediction skill.

The following are gross, price-only results. Benchmarks use whole shares, residual cash and exactly matching saved start/end dates; Hindalco is valued only on its saved dates, so missing intermediate dates limit the drawdown comparison.

| Saved strategy | Strategy return | Matched stock buy-and-hold | Strategy max drawdown | Buy-and-hold max drawdown |
|---|---:|---:|---:|---:|
| Original NALCO 5% monthly, Jan 2025–Apr 2026 | 58.20% | **86.87%** | 18.29% | 33.86% |
| Multi-stock NALCO 5% | 12.64% | **86.87%** | 28.58% | 33.86% |
| Multi-stock Hindalco 5% | 10.94% | **60.50%** | 17.91% | 20.41% |
| Multi-stock Hindalco 10% | 41.54% | **60.50%** | 15.02% | 20.41% |
| Multi-stock Vedanta 5% | 38.65% | **62.26%** | 15.68% | 21.53% |
| Initial chronological NALCO RL, reinvest, Jan 2025–Apr 2026 | 82.54% | **100.44%** | 22.16% | 31.87% |
| Full-history NALCO replay, reinvest, Jan 2020–Apr 2026 | 221.09% | **790.79%** | 44.75% | 48.38% |

The initial chronological RL comparison begins on 9 January 2025; the monthly strategies begin on 1 January. Their benchmark numbers consequently differ. This is why matching windows matters. The full-history replay includes its training period and is not comparable to a clean holdout.

The shuffle strategies return 37.96% and 36.67%, with drawdowns of 18.29% and 25.81%. A proper portfolio benchmark would compare them with an explicitly specified equal-weight or sector portfolio on a complete calendar. Their stock-selection advantage is not established by the existing report.

Lower drawdowns are useful descriptive results, but superior risk-adjusted performance needs a fair exposure-matched comparison after execution corrections. A high win rate with a few large losing trades can still underperform. Targeting 5% or 10% every month corresponds to about **79.6% or 213.8% compounded annually**, respectively; putting a target into an environment does not make that return achievable.

## 7. A better model-development sequence

1. **Repair the data and mechanics first.** Verify LME contract/column provenance; recover Hindalco prices; record stock adjustment conventions and timestamps; fix Supertrend, monthly attribution and short handling; introduce subsequent-session fills and documented costs. Regenerate results under saved configurations and seeds.
2. **Define the target precisely.** Start with next-session open-to-close return and five-session future return. Treat each horizon separately. Price-level forecasts alone can look good simply by copying yesterday's price.
3. **Build one availability-aware dataset.** Keep source timestamps, market calendars, stale-data ages and explicit labels. Use only information available at the decision cutoff. Resample weekly/monthly indicators from daily history and exclude incomplete bars where the strategy assumes completed periods.
4. **Measure incremental value.** Compare simple forecasts, NALCO-only models, LME/DXY additions, Chinese-stock additions and broader Indian-market controls. Test each feature group's additional value. Use chronological forward validation; purge training labels whose future endpoints overlap the validation/test boundary. [Time-series validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
5. **Add economically relevant inputs when verified data is available.** Alumina prices, USD/INR, energy/coal costs, inventory measures, broad Indian equity and metals indices, production/earnings releases and event calendars are candidate research inputs, not assumed predictors. NALCO itself attributes FY2024–25 performance partly to favourable **alumina and aluminium** prices, which supports testing alumina separately. [NALCO FY2024–25 results discussion](https://nalcoindia.com/pre-rel/post-historic-financial-results-nalco-cmd-focuses-on-completion-of-strategic-project-expansions/)
6. **Allow company-specific exposures.** Hindalco includes Novelis and copper; Vedanta spans additional commodities. A common “aluminium stock” coefficient may not suit every firm. This is an inference from their reported business mixes. [Hindalco annual report](https://www.hindalco.com/Upload/PDF/hindalco-annualreport-2024-25.pdf), [Vedanta annual report](https://www.vedantalimited.com/vedantaFY25/)
7. **Return to RL only after the baseline is credible.** Add portfolio status, cash/exposure, unrealized return, costs and target progress to observations. Use a consistent stepwise net-return reward and identical action handling in training/testing. Assess returns, drawdowns, turnover and stability across seeds against matched benchmarks.

The next useful deliverable is a corrected data pipeline and baseline experiment, followed by corrected backtests. Optimizing the current bots before fixing their inputs and simulator would optimize results that remain difficult to trust.

## 8. Audit outputs and verification

The principal tables are [correlations.csv](../../results/original-audit/correlations.csv), [prediction_metrics.csv](../../results/original-audit/prediction_metrics.csv), [saved_backtest_comparison.csv](../../results/original-audit/saved_backtest_comparison.csv), [monthly_attribution_errors.csv](../../results/original-audit/monthly_attribution_errors.csv) and [short_trade_date_audit.csv](../../results/original-audit/short_trade_date_audit.csv). Full source hashes, parameters, date ranges and machine-readable findings are in `audit_results.json` and `additional_checks.json`.

The audit parsed every original Python file successfully, recomputed the technical feature behavior, checked feature prefix invariance for RSI/MACD, verified every external input precedes its decision date, verified every next-session target follows its decision date, reconciled ten saved metrics files, audited all short entry/exit dates, compared ZIP contents against extracted sources and extracted the text of all five PDF reports. PDF text was checked for substantive claims and numbers; this audit does not certify chart layout or rendering fidelity. Full RL retraining, independent market-data validation, formal causal attribution and a live-trading simulation were not performed.
