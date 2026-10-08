"""Portable CLI; all output locations and research assumptions are explicit."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .backtest import Config, PortfolioEnv, evaluate, metrics, monthly_returns, train
from .data import external_features, load_csv, source_fingerprint
from .indicators import build_features
from .research import baselines, correlations


def named_paths(values):
    paths = {}
    for item in values:
        name, sep, path = item.partition("=")
        if not sep or not name or name in paths:
            raise ValueError("Expected unique NAME=path inputs")
        paths[name] = Path(path)
    return paths


def prepare(paths, external_paths):
    frames = {name: build_features(load_csv(path)) for name, path in paths.items()}
    ext = {name: load_csv(path, close_only=True) for name, path in external_paths.items()}
    columns = []
    for name, df in frames.items():
        if ext:
            df, columns = external_features(df, ext)
        required = ["rsi", "macd_hist", "bb_position", "st_direction", "r5", "r20", "vol20", "range", *columns]
        ready = np.isfinite(df[required].to_numpy(float)).all(axis=1)
        if not ready.any():
            raise ValueError(f"No usable features for {name}")
        start = int(np.flatnonzero(ready)[0])
        df = df.iloc[start:].reset_index(drop=True)
        if not np.isfinite(df[required].to_numpy(float)).all():
            raise ValueError(f"Missing/stale features inside {name}; fix sources rather than dropping valuation dates")
        frames[name] = df
    # Multi-asset warmup ends at the latest valid starting date.
    start = max(df.date.min() for df in frames.values())
    frames = {n: f[f.date >= start].reset_index(drop=True) for n, f in frames.items()}
    return frames, ext, columns


def split_frames(frames, test_start, test_end):
    start = pd.Timestamp(test_start)
    training, testing = {}, {}
    for name, df in frames.items():
        before = df[df.date < start]
        if len(before) < 40:
            raise ValueError("Need at least 40 pre-test sessions")
        training[name] = before.reset_index(drop=True)
        test = df[df.date >= before.date.iloc[-1]]
        if test_end:
            test = test[test.date <= pd.Timestamp(test_end)]
        if len(test) < 21:
            raise ValueError("Need at least 20 test executions")
        testing[name] = test.reset_index(drop=True)
    return training, testing


def plots(curves, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for name, history in curves.items():
        normalized = history.equity / history.equity.iloc[0]
        axes[0].plot(history.date, normalized, label=name, linewidth=1.5)
        axes[1].plot(history.date, (history.equity / history.equity.cummax() - 1)*100, linewidth=1)
    axes[0].set(title=title, ylabel="Equity / initial capital")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set(ylabel="Drawdown (%)", xlabel="Session date")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "equity_drawdown.png", dpi=160)
    plt.close(fig)


def run(args):
    paths = {"NALCO": Path(args.data), **named_paths(args.asset)}
    external_paths = named_paths(args.external)
    frames, ext, columns = prepare(paths, external_paths)
    training, testing = split_frames(frames, args.test_start, args.test_end)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                    intraday=args.intraday, monthly_target=args.monthly_target)
    curves, rows = {}, []
    for seed in args.seeds:
        agent = train(PortfolioEnv(training, config, columns), seed=seed, episodes=args.episodes)
        agent.save(output / f"agent_seed_{seed}.json.gz")
        env = PortfolioEnv(testing, config, columns)
        history, trades = evaluate(env, agent=agent)
        history.to_csv(output / f"equity_seed_{seed}.csv", index=False)
        trades.reindex(columns=["decision_date", "execution_date", "phase", "asset", "side", "quantity", "mid", "fill", "fee", "reason"]).to_csv(output / f"trades_seed_{seed}.csv", index=False)
        monthly_returns(history, config.monthly_target).to_csv(output / f"monthly_seed_{seed}.csv", index=False)
        row = metrics(history, config)
        row.update(model="Q-learning", seed=seed, total_fees=env.total_fees,
                   fills=len(trades), invested_session_fraction=float(history.shares.iloc[1:].ne(0).mean()) if not args.intraday else float(len(trades)/2/(len(history)-1)))
        row.pop("config")
        rows.append(row)
        curves[f"Q-learning seed {seed}"] = history
    # Benchmark matches dates, capital, allocation, fees and end liquidation.
    benchmark_config = replace(config, intraday=False, monthly_target=None)
    for i, name in enumerate(testing, start=1):
        history, trades = evaluate(PortfolioEnv(testing, benchmark_config, columns), fixed_action=i)
        history.to_csv(output / f"benchmark_{name}.csv", index=False)
        row = metrics(history, benchmark_config)
        row.update(model=f"{name} buy-and-hold", seed=None, total_fees=float(trades.fee.sum()), fills=len(trades),
                   invested_session_fraction=float(history.shares.iloc[1:].ne(0).mean()))
        row.pop("config")
        rows.append(row)
        curves[f"{name} buy-and-hold (95% initial allocation)"] = history
    table = pd.DataFrame(rows)
    table.to_csv(output / "backtest_metrics.csv", index=False)
    base, predictions = baselines(frames["NALCO"], args.test_start, args.test_end, columns)
    base.to_csv(output / "prediction_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    if ext:
        correlations(frames["NALCO"], ext).to_csv(output / "correlations.csv", index=False)
    # Save dates and source ages, but avoid publishing the original raw prices.
    date_columns = ["date"] + [c for c in frames["NALCO"] if c.endswith(("_source_date", "_age_days"))]
    frames["NALCO"][date_columns].to_csv(output / "availability_audit.csv", index=False)
    manifest = {"dataset_kind": args.dataset_kind, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__, "episodes": args.episodes,
                "seeds": args.seeds, "config": asdict(config), "external_features": columns,
                "source_fingerprints": {n: source_fingerprint(p) for n, p in {**paths, **external_paths}.items()},
                "train_start": str(training["NALCO"].date.min().date()),
                "train_last_execution": str(training["NALCO"].date.max().date()),
                "test_first_execution": str(testing["NALCO"].date.iloc[1].date()),
                "test_last_execution": str(testing["NALCO"].date.iloc[-1].date()),
                "notes": ["Reused historical evaluation window; not a fresh final deployment test",
                          "Costs are illustrative, not a complete Indian brokerage/tax schedule",
                          "Raw source provenance and corporate-action adjustments unverified",
                          "No broker, live feed, guaranteed returns or validated forecasting edge"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    label = "SYNTHETIC DEMO" if args.dataset_kind == "synthetic" else "Historical research - corrected simulator"
    plots(curves, output, label)
    print(table[["model", "seed", "total_return_pct", "max_drawdown_pct", "sharpe_zero_rf_252"]].to_string(index=False))
    print(f"Outputs: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "demo"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--data", default="data/sample/synthetic_nalco.csv")
        cmd.add_argument("--asset", action="append", default=[], metavar="NAME=CSV")
        cmd.add_argument("--external", action="append", default=[], metavar="NAME=CSV")
        cmd.add_argument("--test-start", default="2025-01-01")
        cmd.add_argument("--test-end", default=None)
        cmd.add_argument("--output", default="results/local" if name == "run" else "results/demo")
        cmd.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 99])
        cmd.add_argument("--episodes", type=int, default=200)
        cmd.add_argument("--fee-bps", type=float, default=10)
        cmd.add_argument("--slippage-bps", type=float, default=5)
        cmd.add_argument("--monthly-target", type=float, default=None, help="Fraction, e.g. 0.05")
        cmd.add_argument("--intraday", action="store_true")
        cmd.add_argument("--dataset-kind", choices=["historical_unverified", "synthetic"],
                         default="historical_unverified" if name == "run" else "synthetic")
    args = parser.parse_args(argv)
    if args.command == "demo" and args.data != "data/sample/synthetic_nalco.csv":
        parser.error("demo uses the bundled synthetic data; use run for another input")
    if args.command == "run" and args.data == "data/sample/synthetic_nalco.csv":
        args.dataset_kind = "synthetic"
    try:
        run(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
