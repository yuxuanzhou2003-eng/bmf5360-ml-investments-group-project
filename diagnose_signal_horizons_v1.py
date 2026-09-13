"""Training/validation-only horizon and confidence diagnostics for network_signal."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "data/model_ready_v1/20260909T064151673764Z"
RETURNS = ROOT / "data/clean/v2/returns.csv"
OUT_ROOT = ROOT / "data/analysis/validation_signal_diagnostics_v1"
EVENT_KEY = ["source", "announcement", "period_end"]


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


def signal_metrics(frame):
    if len(frame) < 30 or frame.target.nunique() < 2 or frame.network_signal.nunique() < 2:
        return {"rows": len(frame), "events": frame[EVENT_KEY].drop_duplicates().shape[0],
                "pooled_spearman": np.nan, "mean_event_spearman": np.nan,
                "top_minus_bottom_target": np.nan}
    pooled = spearmanr(frame.target, frame.network_signal).statistic
    event_ic = []
    for _, group in frame.groupby(EVENT_KEY):
        if len(group) >= 3 and group.target.nunique() > 1 and group.network_signal.nunique() > 1:
            value = spearmanr(group.target, group.network_signal).statistic
            if np.isfinite(value):
                event_ic.append(value)
    low, high = frame.network_signal.quantile([0.1, 0.9])
    spread = frame.loc[frame.network_signal >= high, "target"].mean() - frame.loc[
        frame.network_signal <= low, "target"].mean()
    return {"rows": len(frame), "events": frame[EVENT_KEY].drop_duplicates().shape[0],
            "pooled_spearman": pooled, "mean_event_spearman": np.mean(event_ic) if event_ic else np.nan,
            "event_ic_groups": len(event_ic), "top_minus_bottom_target": spread}


def main():
    paths = {"features": MODEL / "model_features.csv", "metadata": MODEL / "metadata.csv",
             "eligibility": MODEL / "eligibility.csv", "returns": RETURNS,
             "code": Path(__file__).resolve()}
    before = {name: sha256(path) for name, path in paths.items()}
    features = pd.read_csv(paths["features"], usecols=[
        "sample_id", "network_signal", "standardized_surprise", "residual_correlation"])
    metadata = pd.read_csv(paths["metadata"], usecols=[
        "sample_id", *EVENT_KEY, "receiver", "announcement_day", "entry_session"])
    eligibility = pd.read_csv(paths["eligibility"], low_memory=False)
    data = features.merge(metadata, on="sample_id", validate="one_to_one").merge(
        eligibility[["sample_id", "split", "split_retained", "entry_trade_eligible", "feature_core_available"]],
        on="sample_id", validate="one_to_one")
    for column in ["split_retained", "entry_trade_eligible", "feature_core_available"]:
        data[column] = bools(data[column]).to_numpy()
    data["announcement_day"] = pd.to_datetime(data.announcement_day).dt.normalize()
    data["entry_session"] = pd.to_datetime(data.entry_session).dt.normalize()
    data = data.loc[data.split.isin(["training", "validation"]) & data.split_retained &
                    data.entry_trade_eligible & data.feature_core_available].copy()

    returns = pd.read_csv(paths["returns"], usecols=["Instrument", "Date", "return_decimal"])
    returns["Date"] = pd.to_datetime(returns.Date).dt.normalize()
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    wide = wide.loc[wide["SPY.P"].notna()]
    sessions, instruments = wide.index, wide.columns
    row_i = sessions.get_indexer(data.entry_session)
    receiver_i = instruments.get_indexer(data.receiver)
    spy_i = int(instruments.get_loc("SPY.P"))
    if (row_i < 0).any() or (receiver_i < 0).any():
        raise RuntimeError("label lookup key missing")
    horizons = [1, 2, 3, 5, 10]
    horizon_purge_counts = {}
    for horizon in horizons:
        forward = (1 + wide).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-horizon) - 1
        data[f"target_{horizon}"] = forward.to_numpy()[row_i, receiver_i] - forward.to_numpy()[row_i, spy_i]
        exit_i = row_i + horizon
        exits = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
        valid_exit = exit_i < len(sessions)
        exits.loc[valid_exit] = sessions[exit_i[valid_exit]].to_numpy()
        boundary_cross = ((data.split.eq("training") & exits.ge(pd.Timestamp("2021-01-01"))) |
                          (data.split.eq("validation") & exits.ge(pd.Timestamp("2023-01-01"))))
        data[f"exit_session_{horizon}"] = exits
        data[f"horizon_boundary_purged_{horizon}"] = boundary_cross
        data.loc[boundary_cross, f"target_{horizon}"] = np.nan
        horizon_purge_counts[str(horizon)] = int(boundary_cross.sum())

    training = data.loc[data.split.eq("training")]
    surprise_cut = {q: training.standardized_surprise.abs().quantile(q) for q in [0.5, 0.75, 0.9]}
    corr_75 = training.residual_correlation.abs().quantile(0.75)
    filters = {
        "all": lambda x: pd.Series(True, index=x.index),
        "positive_correlation": lambda x: x.residual_correlation > 0,
        "negative_correlation": lambda x: x.residual_correlation < 0,
        "high_abs_corr_train_q75": lambda x: x.residual_correlation.abs() >= corr_75,
        "high_surprise_train_q50": lambda x: x.standardized_surprise.abs() >= surprise_cut[0.5],
        "high_surprise_train_q75": lambda x: x.standardized_surprise.abs() >= surprise_cut[0.75],
        "high_surprise_train_q90": lambda x: x.standardized_surprise.abs() >= surprise_cut[0.9],
        "high_surprise_q75_positive_corr": lambda x: (
            x.standardized_surprise.abs() >= surprise_cut[0.75]) & (x.residual_correlation > 0),
    }
    rows = []
    for split_name, split_frame in [("training", training),
                                    ("validation", data.loc[data.split.eq("validation")])]:
        for horizon in horizons:
            usable = split_frame.loc[split_frame[f"target_{horizon}"].notna()].copy()
            usable["target"] = usable[f"target_{horizon}"]
            for filter_name, rule in filters.items():
                subset = usable.loc[rule(usable)]
                result = signal_metrics(subset)
                rows.append({"split": split_name, "horizon": horizon, "filter": filter_name, **result})
            if split_name == "training":
                for year, group in usable.groupby(usable.announcement_day.dt.year):
                    rows.append({"split": f"training_year_{year}", "horizon": horizon, "filter": "all",
                                 **signal_metrics(group)})
    results = pd.DataFrame(rows)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    results.to_csv(out / "diagnostics.csv", index=False)
    after = {name: sha256(path) for name, path in paths.items()}
    checks = {"inputs_unchanged": before == after, "no_test_rows": not data.split.eq("test").any(),
              "horizons_exact": set(results.horizon) == set(horizons),
              "all_filter_present": set(filters).issubset(set(results.loc[results.split.eq("validation"), "filter"]))}
    summary = {"run_id": run_id, "scope": "exploratory training/validation diagnostics; multiple specifications",
               "rows": len(data), "training_rows": int(data.split.eq("training").sum()),
               "validation_rows": int(data.split.eq("validation").sum()),
               "training_fitted_thresholds": {"surprise_abs": surprise_cut, "abs_correlation_q75": corr_75},
               "horizon_boundary_purged_rows": horizon_purge_counts,
               "checks": checks, "input_hashes": before, "output_hash": sha256(out / "diagnostics.csv"),
               "test_policy": "No test rows or test targets were computed."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "checks": checks,
                      "validation": results.loc[results.split.eq("validation")].to_dict(orient="records")},
                     indent=2, ensure_ascii=False, default=str))
    if not all(checks.values()):
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
