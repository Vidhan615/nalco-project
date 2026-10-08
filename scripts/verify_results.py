"""Independent ledger checks for generated research artifacts."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/historical")
    count = 0
    for folder in sorted(p.parent for p in root.rglob("manifest.json")):
        manifest = json.loads((folder / "manifest.json").read_text())
        capital = manifest["config"]["initial_capital"]
        assert pd.Timestamp(manifest["train_last_execution"]) < pd.Timestamp(manifest["test_first_execution"])
        table = pd.read_csv(folder / "backtest_metrics.csv")
        for seed in manifest["seeds"]:
            history = pd.read_csv(folder / f"equity_seed_{seed}.csv", parse_dates=["date"])
            trades = pd.read_csv(folder / f"trades_seed_{seed}.csv", parse_dates=["decision_date", "execution_date"])
            monthly = pd.read_csv(folder / f"monthly_seed_{seed}.csv")
            assert history.date.is_monotonic_increasing and not history.date.duplicated().any()
            assert history.shares.iloc[-1] == 0
            assert np.isclose(history.cash.iloc[-1], history.equity.iloc[-1])
            assert (trades.decision_date < trades.execution_date).all()
            assert (trades.fee >= 0).all()
            signed = np.where(trades.side.eq("BUY"), -1, 1)
            # Terminal cash must reconcile from initial cash and all signed fills.
            ledger = capital + (signed * trades.quantity * trades["fill"] - trades.fee).sum()
            assert np.isclose(ledger, history.equity.iloc[-1])
            row = table[table.seed.eq(seed)].iloc[0]
            assert np.isclose(row.total_return_pct, (ledger/capital-1)*100)
            assert np.isclose(np.prod(1+monthly.return_pct/100), ledger/capital)
            if manifest["config"]["intraday"]:
                assert history.shares.eq(0).all()
                assert trades.groupby(["execution_date", "asset"]).side.count().eq(2).all()
                positions = trades.assign(signed_quantity=np.where(trades.side.eq("BUY"), 1, -1) * trades.quantity)
                assert positions.groupby(["execution_date", "asset"]).signed_quantity.sum().eq(0).all()
            count += 1
        availability = pd.read_csv(folder / "availability_audit.csv", parse_dates=["date"])
        for column in availability:
            if column.endswith("_source_date"):
                source_dates = pd.to_datetime(availability[column])
                assert (source_dates < availability.date).all()
        print(f"Verified {folder.name}: {len(manifest['seeds'])} seeds")
    if not count:
        raise ValueError("No generated manifests found")
    print(f"Verified {count} saved simulations and cash ledgers")


if __name__ == "__main__":
    main()
