"""Training/validation-only diagnostic using next-session open as the theoretical entry."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_baseline_models_v1 import EVENT_KEY, event_weights, metrics

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "data/model_ready_v1/20260909T064151673764Z"
PRICES = ROOT / "data/clean/v3/20260909T012417705069Z/prices.csv"
OUT_ROOT = ROOT / "data/analysis/next_open_target_diagnostics_v1"
HORIZONS = [0, 1, 2, 3, 5, 10]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bools(values):
    values = pd.Series(values).astype("string").str.lower()
    if set(values.dropna()) - {"true", "false"} or values.isna().any():
        raise ValueError("invalid boolean")
    return values.eq("true")


def raw_rank_metrics(frame, signal, target):
    usable = frame.loc[frame[signal].notna() & frame[target].notna()].copy()
    event_ics = []
    for _, group in usable.groupby(EVENT_KEY):
        if len(group) >= 3 and group[signal].nunique() > 1 and group[target].nunique() > 1:
            value = spearmanr(group[signal], group[target]).statistic
            if np.isfinite(value):
                event_ics.append(value)
    return {
        "rows": int(len(usable)),
        "events": int(usable[EVENT_KEY].drop_duplicates().shape[0]),
        "pooled_spearman": float(spearmanr(usable[signal], usable[target]).statistic),
        "mean_event_spearman": float(np.mean(event_ics)) if event_ics else np.nan,
        "event_ic_groups": int(len(event_ics)),
    }


def ridge(alpha):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha)),
    ])


def main():
    paths = {
        "features": MODEL / "model_features.csv",
        "metadata": MODEL / "metadata.csv",
        "eligibility": MODEL / "eligibility.csv",
        "prices": PRICES,
        "code": Path(__file__).resolve(),
        "baseline_code": ROOT / "run_baseline_models_v1.py",
    }
    before = {name: sha256(path) for name, path in paths.items()}
    features = pd.read_csv(paths["features"])
    metadata = pd.read_csv(paths["metadata"], usecols=[
        "sample_id", *EVENT_KEY, "receiver", "announcement_day", "entry_session"])
    eligibility = pd.read_csv(paths["eligibility"], low_memory=False)
    data = features.merge(metadata, on="sample_id", validate="one_to_one").merge(
        eligibility[["sample_id", "split", "split_retained", "entry_trade_eligible",
                     "feature_core_available"]], on="sample_id", validate="one_to_one")
    for column in ["split_retained", "entry_trade_eligible", "feature_core_available"]:
        data[column] = bools(data[column]).to_numpy()
    data["announcement_day"] = pd.to_datetime(data.announcement_day).dt.normalize()
    data["entry_session"] = pd.to_datetime(data.entry_session).dt.normalize()
    data = data.loc[data.split.isin(["training", "validation"]) & data.split_retained &
                    data.entry_trade_eligible & data.feature_core_available].copy()

    prices = pd.read_csv(paths["prices"], usecols=[
        "Instrument", "Date", "OPEN_PRC", "TRDPRC_1", "price_adjustments"])
    prices["Date"] = pd.to_datetime(prices.Date).dt.normalize()
    if prices.duplicated(["Instrument", "Date"]).any():
        raise ValueError("duplicate clean price keys")
    adjustment_values = sorted(prices.price_adjustments.dropna().astype(str).unique())
    indexed = prices.set_index(["Instrument", "Date"])
    spy = prices.loc[prices.Instrument.eq("SPY.P")].sort_values("Date")
    sessions = pd.DatetimeIndex(spy.loc[spy.TRDPRC_1.notna(), "Date"].unique()).sort_values()
    entry_i = sessions.get_indexer(data.entry_session)
    if (entry_i < 0).any():
        raise ValueError("entry session is absent from SPY close calendar")

    receiver_keys = pd.MultiIndex.from_arrays([data.receiver, data.entry_session])
    spy_entry_keys = pd.MultiIndex.from_arrays([np.repeat("SPY.P", len(data)), data.entry_session])
    data["receiver_entry_open"] = indexed.OPEN_PRC.reindex(receiver_keys).to_numpy()
    data["benchmark_entry_open"] = indexed.OPEN_PRC.reindex(spy_entry_keys).to_numpy()
    data["entry_open_available"] = (data.receiver_entry_open.gt(0) & data.benchmark_entry_open.gt(0))

    target_audit = []
    for horizon in HORIZONS:
        exit_i = entry_i + horizon
        exit_valid = exit_i < len(sessions)
        exits = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
        exits.loc[exit_valid] = sessions[exit_i[exit_valid]].to_numpy()
        receiver_exit_keys = pd.MultiIndex.from_arrays([data.receiver, exits])
        spy_exit_keys = pd.MultiIndex.from_arrays([np.repeat("SPY.P", len(data)), exits])
        receiver_exit_close = indexed.TRDPRC_1.reindex(receiver_exit_keys).to_numpy()
        benchmark_exit_close = indexed.TRDPRC_1.reindex(spy_exit_keys).to_numpy()
        boundary_cross = ((data.split.eq("training") & exits.ge(pd.Timestamp("2021-01-01"))) |
                          (data.split.eq("validation") & exits.ge(pd.Timestamp("2023-01-01"))))
        price_complete = (data.entry_open_available & np.isfinite(receiver_exit_close) &
                          np.isfinite(benchmark_exit_close) & (receiver_exit_close > 0) &
                          (benchmark_exit_close > 0))
        eligible = exit_valid & ~boundary_cross.to_numpy() & price_complete.to_numpy()
        target = np.full(len(data), np.nan)
        target[eligible] = (receiver_exit_close[eligible] / data.receiver_entry_open.to_numpy()[eligible] - 1 -
                            (benchmark_exit_close[eligible] / data.benchmark_entry_open.to_numpy()[eligible] - 1))
        data[f"exit_session_{horizon}"] = exits
        data[f"target_open_to_close_{horizon}"] = target
        data[f"boundary_purged_{horizon}"] = boundary_cross
        data[f"target_available_{horizon}"] = eligible
        target_audit.append({
            "horizon": horizon,
            "rows": len(data),
            "entry_open_missing": int((~data.entry_open_available).sum()),
            "exit_close_missing_or_nonpositive": int((~(np.isfinite(receiver_exit_close) &
                                                        np.isfinite(benchmark_exit_close) &
                                                        (receiver_exit_close > 0) &
                                                        (benchmark_exit_close > 0))).sum()),
            "boundary_purged": int(boundary_cross.sum()),
            "target_available": int(eligible.sum()),
        })

    all_features = [column for column in features if column != "sample_id"]
    receiver_features = [column for column in all_features
                         if column.startswith("receiver_") or column.startswith("market_")]
    feature_sets = {"simple_network": ["network_signal"],
                    "receiver_ridge": receiver_features,
                    "network_ridge": all_features}
    fixed_models = {
        "simple_network": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")),
                                             ("regressor", LinearRegression())]),
        "receiver_ridge": lambda: ridge(0.01),
        "network_ridge": lambda: ridge(100.0),
    }
    metric_rows, prediction_rows, coefficient_rows = [], [], []
    for horizon in HORIZONS:
        target = f"target_open_to_close_{horizon}"
        train = data.loc[data.split.eq("training") & data[target].notna()].copy().rename(columns={target: "target"})
        valid = data.loc[data.split.eq("validation") & data[target].notna()].copy().rename(columns={target: "target"})
        for split_name, frame in [("training", train), ("validation", valid)]:
            raw = raw_rank_metrics(frame, "network_signal", "target")
            metric_rows.append({"split": split_name, "horizon": horizon,
                                "model": "raw_network_signal", **raw})
        for model_name, factory in fixed_models.items():
            model = factory()
            columns = feature_sets[model_name]
            model.fit(train[columns], train.target, **{"regressor__sample_weight": event_weights(train)})
            for split_name, frame in [("training", train), ("validation", valid)]:
                prediction = model.predict(frame[columns])
                row = metrics(frame, prediction)
                metric_rows.append({"split": split_name, "horizon": horizon,
                                    "model": model_name, **row})
                if split_name == "validation":
                    prediction_rows.extend({"sample_id": sid, "horizon": horizon,
                                            "model": model_name, "target": actual,
                                            "prediction": predicted}
                                           for sid, actual, predicted in zip(
                                               frame.sample_id, frame.target, prediction))
            if model_name == "simple_network":
                coefficient_rows.append({"horizon": horizon,
                                         "intercept": float(model.named_steps["regressor"].intercept_),
                                         "network_signal_coefficient": float(
                                             model.named_steps["regressor"].coef_[0])})

    metrics_df = pd.DataFrame(metric_rows)
    audit_df = pd.DataFrame(target_audit)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    target_columns = (["sample_id", *EVENT_KEY, "receiver", "announcement_day", "entry_session",
                       "split", "receiver_entry_open", "benchmark_entry_open", "entry_open_available"] +
                      [column for column in data if column.startswith(("exit_session_", "target_open_", "boundary_", "target_available_"))])
    data[target_columns].to_csv(out / "open_targets_and_availability.csv", index=False)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    audit_df.to_csv(out / "target_audit.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "validation_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out / "simple_network_coefficients.csv", index=False)
    feature_missing = data[all_features].isna().sum()
    imputation_log = {
        "policy": "Training-set median inside each model pipeline; no source rows or stored features changed.",
        "development_feature_missing_counts_nonzero": {
            key: int(value) for key, value in feature_missing.items() if value > 0},
        "fixed_hyperparameters_inherited_from_close_target_baseline": {
            "receiver_ridge_alpha": 0.01, "network_ridge_alpha": 100.0},
    }
    (out / "imputation_log.json").write_text(json.dumps(imputation_log, indent=2), encoding="utf-8")
    after = {name: sha256(path) for name, path in paths.items()}
    checks = {
        "inputs_unchanged": before == after,
        "no_test_rows": not data.split.eq("test").any(),
        "entry_strictly_after_announcement_day": bool(data.entry_session.gt(data.announcement_day).all()),
        "targets_finite_when_available": bool(np.isfinite(data[[f"target_open_to_close_{h}" for h in HORIZONS]].stack()).all()),
        "clean_price_keys_unique": not prices.duplicated(["Instrument", "Date"]).any(),
    }
    output_files = ["open_targets_and_availability.csv", "metrics.csv", "target_audit.csv",
                    "validation_predictions.csv", "simple_network_coefficients.csv", "imputation_log.json"]
    summary = {
        "run_id": run_id,
        "scope": "exploratory next-session-open price-return targets; training/validation only",
        "target_definition": "receiver open-to-exit-close price return minus SPY open-to-exit-close price return",
        "limitations": ["Open price assumes theoretical execution at the official open and excludes slippage.",
                        "Price return is not a dividend total return."],
        "rows_retained": len(data),
        "target_audit": target_audit,
        "price_adjustment_values": adjustment_values,
        "imputation": imputation_log,
        "checks": checks,
        "input_hashes": before,
        "output_hashes": {name: sha256(out / name) for name in output_files},
        "test_policy": "No test rows, test targets, predictions, or metrics were computed.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "checks": checks, "target_audit": target_audit,
                      "validation_metrics": metrics_df.loc[metrics_df.split.eq("validation")].to_dict(orient="records")},
                     indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
