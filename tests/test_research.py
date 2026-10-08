import numpy as np
import pandas as pd
import pytest

from nalco_research.backtest import Config, PortfolioEnv, QAgent, evaluate, metrics, monthly_returns, train
from nalco_research.data import external_features, load_csv, number
from nalco_research.indicators import build_features, supertrend
from nalco_research.research import chronological_split, prediction_dataset, ridge_predict


def frame(dates, opens, closes):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "open": opens, "close": closes})
    df["high"] = df[["open", "close"]].max(axis=1) + 1
    df["low"] = df[["open", "close"]].min(axis=1) - 1
    for name, value in {"rsi": 50, "macd_hist": .1, "bb_position": .5,
                        "st_direction": 1, "r5": .01, "vol20": .02}.items():
        df[name] = value
    return df


def test_volume_plain_and_suffix():
    assert number("12345") == 12345
    assert number("21.40M") == 21_400_000
    assert number("1,234.5") == 1234.5
    assert np.isnan(number("-"))


def test_loader_rejects_missing_prices_and_duplicates(tmp_path):
    path = tmp_path / "bad.csv"
    frame(["2025-01-01", "2025-01-02"], [10, np.nan], [10, 11]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="complete"):
        load_csv(path)
    frame(["2025-01-01", "2025-01-01"], [10, 11], [10, 11]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_csv(path)


def test_loader_rejects_ohlc_but_close_only_is_explicit(tmp_path):
    df = frame(["2025-01-01", "2025-01-02"], [10, 11], [10, 11])
    df.loc[0, "high"] = 5
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="OHLC"):
        load_csv(path)
    assert len(load_csv(path, close_only=True)) == 2


def test_supertrend_known_crossings_and_retained_bands():
    closes = np.array([10., 10., 14., 15., 9., 8.])
    df = pd.DataFrame({"close": closes, "high": closes + 1, "low": closes - 1})
    st = supertrend(df, period=2, multiplier=1)
    assert st.st_direction.iloc[1:].tolist() == [-1, 1, 1, -1, -1]
    assert st.final_upper.iloc[2] == 12  # previous band retained despite higher basic upper
    assert st.final_lower.iloc[4] == 11.5
    assert st.supertrend.iloc[2] == 10.5
    assert st.supertrend.iloc[4] == 13.5


def test_all_indicators_are_prefix_invariant():
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .02, 120)))
    df = frame(pd.bdate_range("2024-01-01", periods=120), close, close)
    a = build_features(df.iloc[:80])
    b = build_features(df).iloc[:80]
    pd.testing.assert_frame_equal(a, b)


def test_external_inputs_strictly_precede_decisions_and_staleness():
    base = pd.DataFrame({"date": pd.to_datetime(["2025-02-03", "2025-02-20"])})
    ext = pd.DataFrame({"date": pd.bdate_range("2025-01-01", "2025-02-03"),
                        "close": np.arange(24, dtype=float) + 100})
    aligned, columns = external_features(base, {"LME": ext}, max_age_days=7)
    assert aligned.LME_source_date.iloc[0] == pd.Timestamp("2025-01-31")
    assert (aligned.LME_source_date < aligned.date).all()
    assert aligned.loc[1, columns].isna().all()


def test_next_open_fill_and_terminal_accounting():
    df = frame(["2025-01-01", "2025-01-02", "2025-01-03"], [10, 20, 1000], [10, 30, 40])
    config = Config(initial_capital=1000, fee_bps=0, slippage_bps=0, allocation=1)
    env = PortfolioEnv({"NALCO": df}, config)
    h, trades = evaluate(env, fixed_action=1)
    assert h.equity.tolist() == [1000, 1500, 2000]
    assert trades.iloc[0].fill == 20
    assert trades.iloc[0].decision_date < trades.iloc[0].execution_date
    assert h.shares.iloc[-1] == 0
    assert h.cash.iloc[-1] == 2000
    assert metrics(h, config)["total_return_pct"] == 100


def test_reward_counts_equity_once_including_terminal_costs():
    df = frame(["2025-01-01", "2025-01-02"], [10, 10], [10, 10])
    config = Config(initial_capital=1000, fee_bps=100, slippage_bps=100, allocation=1)
    env = PortfolioEnv({"NALCO": df}, config)
    _, reward, done = env.step(1)
    assert done
    quantity = int(1000 // (10 * 1.01 * 1.01))
    expected = 1000 - quantity * 10.1 * 1.01 + quantity * 9.9 * .99
    assert env.cash == pytest.approx(expected)
    assert reward == pytest.approx(np.log(expected / 1000))
    assert env.total_fees > 0


@pytest.mark.parametrize("action,expected", [(1, 1100), (2, 900)])
def test_intraday_long_short_close_same_session(action, expected):
    df = frame(["2025-01-01", "2025-01-02"], [10, 20], [10, 22])
    config = Config(initial_capital=1000, fee_bps=0, slippage_bps=0, allocation=1, intraday=True)
    env = PortfolioEnv({"NALCO": df}, config)
    h, trades = evaluate(env, fixed_action=action)
    assert h.cash.iloc[-1] == expected
    assert h.shares.eq(0).all()
    assert trades.execution_date.nunique() == 1
    assert trades.phase.tolist() == ["OPEN", "CLOSE"]


def test_calendar_month_returns_do_not_use_next_month_price():
    history = pd.DataFrame({"date": pd.to_datetime(["2024-12-31", "2025-01-31", "2025-02-03", "2025-02-28"]),
                            "equity": [1000, 1100, 2200, 1210]})
    monthly = monthly_returns(history, .05)
    assert monthly.return_pct.tolist() == pytest.approx([10, 10])
    assert monthly.partial.tolist() == [False, False]


def test_month_target_resets_before_next_month_execution():
    df = frame(["2025-01-30", "2025-01-31", "2025-02-03"], [10, 10, 20], [10, 20, 30])
    config = Config(initial_capital=1000, fee_bps=0, slippage_bps=0, allocation=1, monthly_target=.05)
    env = PortfolioEnv({"NALCO": df}, config)
    env.step(1)
    assert env.allowed_actions() == [0, 1]  # reset for February, no old-month lock
    assert env.month_start == 2000
    env.step(1)
    monthly = monthly_returns(pd.DataFrame(env.history), .05)
    assert monthly.return_pct.tolist() == pytest.approx([100, 50])
    assert monthly.partial.tolist() == [True, True]


def test_target_blocks_new_risk_and_exits_next_open():
    df = frame(["2025-01-01", "2025-01-02", "2025-01-03"], [10, 10, 21], [10, 20, 5])
    env = PortfolioEnv({"NALCO": df}, Config(initial_capital=1000, fee_bps=0,
                         slippage_bps=0, allocation=1, monthly_target=.05))
    env.step(1)
    assert env.allowed_actions() == [0]
    with pytest.raises(ValueError, match="feasible"):
        env.step(1)
    env.step(0)
    assert env.cash == 2100


def test_multi_asset_rebalance_conserves_cash_and_uses_common_policy():
    a = frame(["2025-01-01", "2025-01-02", "2025-01-03"], [10, 10, 15], [10, 12, 15])
    b = frame(a.date, [20, 20, 30], [20, 20, 33])
    env = PortfolioEnv({"A": a, "B": b}, Config(initial_capital=1000, fee_bps=0, slippage_bps=0, allocation=1))
    env.step(1)
    env.step(2)
    assert env.cash == 1650  # 100 A shares sold at 15, then 50 B shares sold at 33
    assert len(env.trades) == 4
    assert env.n_actions == 3


def test_missing_asset_calendar_is_rejected():
    a = frame(["2025-01-01", "2025-01-02", "2025-01-03"], [10]*3, [10]*3)
    with pytest.raises(ValueError, match="calendars"):
        PortfolioEnv({"A": a, "B": a.iloc[:2]})


def test_training_and_test_label_endpoints_do_not_overlap():
    df = frame(pd.bdate_range("2024-12-01", "2025-01-10"), [10]*30, [10]*30)
    data = prediction_dataset(df).iloc[:-1]
    train_data, test_data = chronological_split(data, "2025-01-01")
    assert train_data.target_date.max() < pd.Timestamp("2025-01-01")
    assert test_data.target_date.min() >= pd.Timestamp("2025-01-01")
    assert set(train_data.target_date).isdisjoint(test_data.target_date)


def test_ridge_standardization_does_not_fit_test_data():
    train_data = pd.DataFrame({"x": [1, 2, 3, 4], "target": [.1, .2, .3, .4]})
    prediction = ridge_predict(train_data, pd.DataFrame({"x": [5]}), ["x"])
    extended = ridge_predict(train_data, pd.DataFrame({"x": [5, 1e9]}), ["x"])
    assert prediction[0] == extended[0]


def test_training_is_reproducible_from_seed():
    df = frame(pd.bdate_range("2024-01-01", periods=12), list(range(10, 22)), list(range(11, 23)))
    config = Config(initial_capital=1000, fee_bps=5, slippage_bps=5)
    a = train(PortfolioEnv({"A": df}, config), seed=7, episodes=8)
    b = train(PortfolioEnv({"A": df}, config), seed=7, episodes=8)
    assert set(a.q) == set(b.q)
    for state in a.q:
        np.testing.assert_array_equal(a.q[state], b.q[state])
    ah, at = evaluate(PortfolioEnv({"A": df}, config), a)
    bh, bt = evaluate(PortfolioEnv({"A": df}, config), b)
    pd.testing.assert_frame_equal(ah, bh)
    pd.testing.assert_frame_equal(at, bt)


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_saved_policy_restores_the_same_evaluation(tmp_path, suffix):
    df = frame(pd.bdate_range("2024-01-01", periods=12), list(range(10, 22)), list(range(11, 23)))
    env = PortfolioEnv({"A": df})
    agent = train(env, seed=7, episodes=6)
    path = tmp_path / ("policy" + suffix)
    agent.save(path)
    loaded = QAgent.load(path)
    original, _ = evaluate(env, agent)
    restored, _ = evaluate(env, loaded)
    pd.testing.assert_frame_equal(original, restored)
