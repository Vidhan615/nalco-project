"""Reproduce the published experiments from local source files.

Raw inputs remain local. Every run includes its own model/configuration manifest.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from nalco_research.backtest import Config, PortfolioEnv, evaluate, metrics, train
from nalco_research.cli import main as run_cli, prepare
from nalco_research.research import baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="results/historical")
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    source = Path(args.data_dir)
    target = Path(args.output)
    common = ["run", "--data", str(source / "nalco_daily.csv"), "--episodes", str(args.episodes),
              "--seeds", "7", "42", "99"]
    for name, options in [("long_only", []), ("monthly_5pct", ["--monthly-target", "0.05"]),
                          ("intraday", ["--intraday"]), ("external", [
                              "--external", f"LME={source / 'lme.csv'}", "--external", f"DXY={source / 'dxy.csv'}",
                              "--external", f"HONGQIAO={source / 'hongqiao.csv'}",
                              "--external", f"CHALCO={source / 'chalco.csv'}"])]:
        print(f"Running {name}", flush=True)
        run_cli(common + ["--output", str(target / name)] + options)
    diagnostics(source, target, args.episodes)


def diagnostics(source, target, episodes=200):
    """Fixed-policy cost sensitivity and annual expanding-training baselines."""
    target.mkdir(parents=True, exist_ok=True)
    paths = {"NALCO": source / "nalco_daily.csv"}
    external = {n: source / f"{file}.csv" for n, file in [
        ("LME", "lme"), ("DXY", "dxy"), ("HONGQIAO", "hongqiao"), ("CHALCO", "chalco")]}
    frames, _, columns = prepare(paths, external)
    tables = []
    for year in (2023, 2024, 2025, 2026):
        end = min(pd.Timestamp(f"{year}-12-31"), frames["NALCO"].date.max())
        table, _ = baselines(frames["NALCO"], f"{year}-01-01", str(end.date()), columns)
        table["evaluation_year"] = year
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(target / "annual_forward_baselines.csv", index=False)
    frames, _, _ = prepare(paths, {})
    from nalco_research.cli import split_frames
    training, testing = split_frames(frames, "2025-01-01", None)
    config = Config()
    agent = train(PortfolioEnv(training, config), seed=42, episodes=episodes)
    rows = []
    for fee, slip in [(0, 0), (10, 5), (20, 10), (40, 20)]:
        cost_config = replace(config, fee_bps=fee, slippage_bps=slip)
        history, trades = evaluate(PortfolioEnv(testing, cost_config), agent)
        item = metrics(history, cost_config)
        item.pop("config")
        item.update(fee_bps=fee, slippage_bps=slip, seed=42, training_fee_bps=10,
                    training_slippage_bps=5, episodes=episodes, fills=len(trades))
        rows.append(item)
    pd.DataFrame(rows).to_csv(target / "fixed_policy_cost_sensitivity.csv", index=False)
    print("Saved annual forward baselines and fixed-policy cost sensitivity", flush=True)


if __name__ == "__main__":
    main()
