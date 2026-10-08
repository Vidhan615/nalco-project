"""Shared portfolio simulator: close decision, next open execution, daily marking."""
from dataclasses import asdict, dataclass
import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    initial_capital: float = 500_000
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    allocation: float = 0.95
    monthly_target: float | None = None
    intraday: bool = False

    def __post_init__(self):
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("Capital must be positive and finite")
        if not (0 <= self.fee_bps < 1000 and 0 <= self.slippage_bps < 1000):
            raise ValueError("Costs must be between 0 and 1000 bps")
        if not (0 < self.allocation <= 1):
            raise ValueError("Allocation must be in (0, 1]")
        if self.monthly_target is not None and not (0 < self.monthly_target < 1):
            raise ValueError("Monthly target must be a fraction in (0, 1)")


MARKET_COLUMNS = ["rsi", "macd_hist", "bb_position", "st_direction", "r5", "vol20"]


class PortfolioEnv:
    """One common Q-policy for cash or one selected asset; no raw-Q ranking.

    Long-only actions: 0 cash, 1..N long asset. Intraday actions: 0 cash,
    1..N long, N+1..2N short, all flattened at the same session's close.
    A short uses at most `allocation` of equity as one-times notional and
    reserves sale proceeds; there is no leverage or broker margin emulation.
    """

    def __init__(self, frames, config=Config(), external_columns=()):
        if not frames:
            raise ValueError("At least one asset is required")
        self.names = list(frames)
        self.frames = [frames[n].reset_index(drop=True) for n in self.names]
        self.config = config
        self.external_columns = tuple(external_columns)
        reference = self.frames[0].date
        if len(reference) < 2:
            raise ValueError("Need a decision row and at least one execution session")
        if not reference.is_monotonic_increasing or reference.duplicated().any():
            raise ValueError("Sessions must be unique and chronologically sorted")
        for frame in self.frames:
            if not frame.date.equals(reference):
                raise ValueError("Asset calendars must match exactly; repair missing data, do not intersect away risk")
            columns = MARKET_COLUMNS + list(self.external_columns)
            if not np.isfinite(frame[columns].to_numpy(float)).all():
                raise ValueError("Incomplete features; warm up or repair missing/stale source data")
            prices = frame[["open", "close"]].to_numpy(float)
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError("Execution and valuation prices must be positive and finite")
        self.dates = reference.to_numpy()
        self.opens = np.column_stack([f.open for f in self.frames])
        self.closes = np.column_stack([f.close for f in self.frames])
        self.market_states = []
        for i in range(len(reference)):
            state = []
            for frame in self.frames:
                row = frame.iloc[i]
                state.extend([int(np.digitize(row.rsi, [30, 50, 70])), int(row.macd_hist > 0),
                              int(np.digitize(row.bb_position, [0, .5, 1])), int(row.st_direction > 0),
                              int(np.digitize(row.r5, [-.03, 0, .03])), int(row.vol20 > .025)])
            # Coarse signs keep a tabular state tractable; baselines use continuous features.
            for column in self.external_columns:
                state.append(int(self.frames[0].iloc[i][column] > 0))
            self.market_states.append(tuple(state))
        self.n_actions = 1 + len(self.names) * (2 if config.intraday else 1)
        self.reset()

    def reset(self):
        self.idx = 0
        self.cash = float(self.config.initial_capital)
        self.shares = 0
        self.asset = -1
        self.entry = 0.0
        self.equity = self.cash
        self.month = pd.Timestamp(self.dates[1]).to_period("M")
        self.month_start = self.equity
        self.target_crossed = False
        self.total_fees = 0.0
        self.trades = []
        self.history = [{"date": pd.Timestamp(self.dates[0]), "equity": self.equity,
                         "cash": self.cash, "asset": "CASH", "shares": 0}]
        return self.state()

    def _prepare_month(self):
        month = pd.Timestamp(self.dates[self.idx + 1]).to_period("M")
        if month != self.month:
            # Previous observed close is the capital base; no next-month price leaks.
            self.month, self.month_start = month, self.equity
            self.target_crossed = False

    def state(self):
        self._prepare_month()
        unrealized = 0.0 if not self.shares else self.closes[self.idx, self.asset] / self.entry - 1
        progress = self.equity / self.month_start - 1
        next_day = pd.Timestamp(self.dates[self.idx + 1])
        remaining = next_day.days_in_month - next_day.day
        return (*self.market_states[self.idx], self.asset + 1,
                int(np.digitize(unrealized, [-.05, 0, .05])),
                int(np.digitize(progress, [-.05, 0, .025, .05, .10])),
                int(remaining // 7), int(self.target_crossed))

    def allowed_actions(self):
        return [0] if self.target_crossed else list(range(self.n_actions))

    def _trade(self, quantity, asset, mid, date, phase, reason):
        """Positive quantity buys; negative quantity sells. Cash includes fees."""
        price = mid * (1 + np.sign(quantity) * self.config.slippage_bps / 10_000)
        fee = abs(quantity) * price * self.config.fee_bps / 10_000
        self.cash -= quantity * price + fee
        self.total_fees += fee
        self.trades.append({"decision_date": pd.Timestamp(self.dates[self.idx]),
                            "execution_date": pd.Timestamp(date), "phase": phase,
                            "asset": self.names[asset], "side": "BUY" if quantity > 0 else "SELL",
                            "quantity": abs(quantity), "mid": mid, "fill": price,
                            "fee": fee, "reason": reason})
        return price

    def step(self, action):
        self._prepare_month()
        if action not in self.allowed_actions():
            raise ValueError("Action is not feasible in the observed portfolio state")
        old_equity = self.equity
        j = self.idx + 1
        date = self.dates[j]
        n = len(self.names)
        target = (action - 1) % n if action else -1
        side = -1 if self.config.intraday and action > n else 1
        if self.shares and self.asset != target:
            self._trade(-self.shares, self.asset, self.opens[j, self.asset], date, "OPEN", "rebalance")
            self.shares, self.asset, self.entry = 0, -1, 0.0
        if target >= 0 and self.shares == 0:
            mid = self.opens[j, target]
            # Reserve enough for both sides' fees and slippage for a one-times short.
            unit_budget = mid * (1 + (self.config.fee_bps + self.config.slippage_bps) / 10_000)
            if side < 0:
                unit_budget = mid * (1 + 2 * (self.config.fee_bps + self.config.slippage_bps) / 10_000)
            quantity = int(self.cash * self.config.allocation // unit_budget)
            if quantity:
                self.entry = self._trade(side * quantity, target, mid, date, "OPEN", "signal")
                self.asset, self.shares = target, side * quantity
        if self.shares and (self.config.intraday or j == len(self.dates) - 1):
            self._trade(-self.shares, self.asset, self.closes[j, self.asset], date, "CLOSE",
                        "session_end" if self.config.intraday else "scheduled_end")
            self.shares, self.asset, self.entry = 0, -1, 0.0
        self.equity = self.cash + (self.shares * self.closes[j, self.asset] if self.shares else 0)
        if self.equity <= 0:
            raise ValueError("Portfolio insolvent: daily bars cannot simulate the required margin liquidation")
        if self.config.monthly_target is not None:
            self.target_crossed |= self.equity >= self.month_start * (1 + self.config.monthly_target)
        self.history.append({"date": pd.Timestamp(date), "equity": self.equity,
                             "cash": self.cash, "asset": self.names[self.asset] if self.asset >= 0 else "CASH",
                             "shares": self.shares})
        self.idx = j
        done = j == len(self.dates) - 1
        # The complete equity move is rewarded once, including all terminal costs.
        reward = float(np.log(self.equity / old_equity))
        return None if done else self.state(), reward, done


class QAgent:
    def __init__(self, n_actions, seed=42, alpha=.10, gamma=.95):
        self.n_actions, self.alpha, self.gamma = n_actions, alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.q = {}

    def values(self, state):
        return self.q.setdefault(state, np.zeros(self.n_actions))

    def act(self, state, allowed, epsilon=0):
        if self.rng.random() < epsilon:
            return int(self.rng.choice(allowed))
        # Deterministic tie break prefers cash; explicitly report cash-heavy policies.
        return int(allowed[int(np.argmax(self.values(state)[allowed]))])

    def update(self, state, action, reward, next_state, allowed, done):
        target = reward if done else reward + self.gamma * self.values(next_state)[allowed].max()
        self.values(state)[action] += self.alpha * (target - self.values(state)[action])

    def save(self, path):
        payload = {"alpha": self.alpha, "gamma": self.gamma, "n_actions": self.n_actions,
                   "states": [{"state": list(k), "values": v.tolist()} for k, v in sorted(self.q.items())]}
        path = Path(path)
        content = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        path.write_bytes(gzip.compress(content, mtime=0) if path.suffix == ".gz" else content)

    @classmethod
    def load(cls, path):
        """Restore a saved policy; seed only affects any later exploratory actions."""
        path = Path(path)
        content = path.read_bytes()
        payload = json.loads(gzip.decompress(content) if path.suffix == ".gz" else content)
        agent = cls(payload["n_actions"], alpha=payload["alpha"], gamma=payload["gamma"])
        agent.q = {tuple(row["state"]): np.array(row["values"], dtype=float)
                   for row in payload["states"]}
        return agent


def train(env, seed=42, episodes=200):
    if episodes < 1:
        raise ValueError("Episodes must be positive")
    agent = QAgent(env.n_actions, seed=seed)
    for episode in range(episodes):
        state = env.reset()
        epsilon = max(.05, .8 * .98 ** episode)
        while True:
            action = agent.act(state, env.allowed_actions(), epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state,
                         [] if done else env.allowed_actions(), done)
            if done:
                break
            state = next_state
    return agent


def evaluate(env, agent=None, fixed_action=None):
    state = env.reset()
    while True:
        action = fixed_action if fixed_action is not None else agent.act(state, env.allowed_actions())
        if action not in env.allowed_actions():
            action = 0
        state, _, done = env.step(action)
        if done:
            break
    return pd.DataFrame(env.history), pd.DataFrame(env.trades)


def metrics(history, config=Config()):
    equity = history.equity
    returns = equity.pct_change().dropna()
    sd = returns.std(ddof=1)
    elapsed = (history.date.iloc[-1] - history.date.iloc[0]).days / 365.25
    return {"start_decision_date": str(history.date.iloc[0].date()),
            "first_execution_date": str(history.date.iloc[1].date()),
            "end_date": str(history.date.iloc[-1].date()), "sessions": len(returns),
            "total_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100),
            "cagr_pct": float(((equity.iloc[-1] / equity.iloc[0]) ** (1 / elapsed) - 1) * 100),
            "max_drawdown_pct": float((1 - equity / equity.cummax()).max() * 100),
            "sharpe_zero_rf_252": float(returns.mean() / sd * np.sqrt(252)) if sd > 0 else None,
            "config": asdict(config)}


def monthly_returns(history, target=None):
    """Actual last observed close in each month; partial endpoints are explicit.

    Completion means the data horizon reaches calendar month-end, not an
    inferred exchange calendar. The final month is conservatively partial if
    the file stops earlier (even if its last date could be a final session).
    """
    equity = history.set_index("date").equity
    baseline = equity.iloc[0]
    rows = []
    # Initial snapshot is a capital base, not an execution observation.
    for period, group in equity.iloc[1:].groupby(equity.index[1:].to_period("M")):
        end = float(group.iloc[-1])
        ret = end / baseline - 1
        first_partial = period == equity.index[1].to_period("M") and equity.index[0] >= period.start_time
        last_partial = period == equity.index[-1].to_period("M") and equity.index[-1].normalize() < period.end_time.normalize()
        rows.append({"month": str(period), "start_equity": baseline, "end_equity": end,
                     "last_observed_session": str(group.index[-1].date()), "return_pct": ret * 100,
                     "partial": bool(first_partial or last_partial),
                     "month_end_target_met": None if target is None else bool(ret >= target)})
        baseline = end
    return pd.DataFrame(rows)
