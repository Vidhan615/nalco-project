"""Chronological return-prediction baselines and descriptive correlations."""
import numpy as np
import pandas as pd

OWN_FEATURES = ["r1", "r5", "r20", "vol20", "range"]


def prediction_dataset(features):
    df = features.copy()
    # Labels belong to the next session; feature rows are observed at today's close.
    df["target"] = np.log(df.close.shift(-1) / df.open.shift(-1))
    df["target_date"] = df.date.shift(-1)
    return df


def chronological_split(df, test_start, test_end=None):
    start = pd.Timestamp(test_start)
    train = df[(df.date < start) & (df.target_date < start)].copy()
    test = df[df.target_date.ge(start)].copy()
    if test_end:
        test = test[test.target_date.le(pd.Timestamp(test_end))]
    test = test[test.date.ge(train.date.max())].copy()
    return train, test


def ridge_predict(train, test, columns, penalty=10):
    x, z = train[columns].to_numpy(float), test[columns].to_numpy(float)
    mu, sigma = x.mean(axis=0), x.std(axis=0)
    sigma[sigma == 0] = 1
    x, z = (x - mu) / sigma, (z - mu) / sigma
    y = train.target.to_numpy(float)
    beta = np.linalg.solve(x.T @ x + penalty * np.eye(len(columns)), x.T @ (y - y.mean()))
    return y.mean() + z @ beta


def baselines(features, test_start, test_end=None, external_columns=()):
    groups = {"NALCO_only": OWN_FEATURES}
    if external_columns:
        ld = [c for c in external_columns if c.startswith(("LME_", "DXY_"))]
        if ld:
            groups["NALCO_LME_DXY"] = OWN_FEATURES + ld
        groups["NALCO_all_external"] = OWN_FEATURES + list(external_columns)
    df = prediction_dataset(features)
    # Compare every model on the same finite observations.
    required = list(dict.fromkeys(OWN_FEATURES + list(external_columns) + ["target"]))
    # Only the final unavailable label may be removed; interior gaps fail explicitly.
    df = df.iloc[:-1].copy()
    if not np.isfinite(df[required].to_numpy(float)).all():
        raise ValueError("Prediction data has missing inputs or labels")
    train, test = chronological_split(df, test_start, test_end)
    if len(train) < 40 or len(test) < 20:
        raise ValueError("Need at least 40 train and 20 test observations")
    y = test.target.to_numpy()
    predictions = {"zero_return": np.zeros(len(test)),
                   "train_mean": np.full(len(test), train.target.mean())}
    for name, columns in groups.items():
        predictions[name] = ridge_predict(train, test, columns)
    rows, detailed = [], []
    for name, pred in predictions.items():
        rows.append({"model": name, "target": "next_session_open_to_close_log_return",
                     "train_n": len(train), "test_n": len(test),
                     "last_training_label_date": str(train.target_date.max().date()),
                     "first_test_execution_date": str(test.target_date.min().date()),
                     "last_test_execution_date": str(test.target_date.max().date()),
                     "rmse_pct": np.sqrt(np.mean((y-pred)**2))*100,
                     "direction_accuracy": np.mean((pred > 0) == (y > 0)),
                     "always_up_accuracy": np.mean(y > 0), "ridge_penalty": 10})
        detailed.extend({"model": name, "decision_date": d, "target_date": t,
                         "actual": actual, "prediction": p}
                        for d, t, actual, p in zip(test.date, test.target_date, y, pred))
    return pd.DataFrame(rows), pd.DataFrame(detailed)


def correlations(stock, sources):
    rows = []
    for name, ext in sources.items():
        pair = pd.concat([stock.set_index("date").close.rename("NALCO"),
                          ext.set_index("date").close.rename(name)], axis=1).dropna()
        for rule, frequency in [(None, "daily_shared_endpoints"), ("W-FRI", "weekly"), ("ME", "monthly")]:
            prices = pair if rule is None else pair.resample(rule).last()
            if rule is not None:
                prices = prices[prices.index <= pair.index.max()]
            ret = np.log(prices).diff().dropna()
            rows.append({"driver": name, "frequency": frequency, "n": len(ret),
                         "start": str(ret.index.min().date()), "end": str(ret.index.max().date()),
                         "return_correlation": ret.NALCO.corr(ret[name])})
    return pd.DataFrame(rows)
