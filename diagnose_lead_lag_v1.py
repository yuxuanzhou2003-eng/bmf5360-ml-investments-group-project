"""Training/validation-only directional lead-lag diagnostic on frozen v3 edges."""
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
OUT_ROOT = ROOT / "data/analysis/lead_lag_diagnostics_v1"
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


def residual_history(wide, day, candidates, lookback=126, min_obs=100):
    history = wide.loc[wide.index < day, ["SPY.P", *sorted(candidates)]].tail(lookback)
    market = history["SPY.P"]
    block = history.drop(columns="SPY.P")
    valid = block.notna().mul(market.notna(), axis=0)
    count = valid.sum()
    x = block.where(valid)
    sum_x = x.sum()
    sum_m = valid.mul(market, axis=0).sum()
    denominator = valid.mul(market.pow(2), axis=0).sum() - sum_m.pow(2) / count
    numerator = x.mul(market, axis=0).sum() - sum_x * sum_m / count
    beta = (numerator / denominator).where(
        (count >= min_obs) & np.isfinite(denominator) & denominator.gt(0)).dropna()
    return block[beta.index] - pd.DataFrame(
        np.outer(market.to_numpy(), beta.to_numpy()), index=history.index, columns=beta.index)


def lag_corr(left, right, min_obs=100):
    x, y = left.iloc[:-1].to_numpy(float), right.iloc[1:].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < min_obs:
        return np.nan, int(valid.sum())
    return float(np.corrcoef(x[valid], y[valid])[0, 1]), int(valid.sum())


def score(frame, column, target):
    usable = frame.loc[frame[column].notna() & frame[target].notna()].copy()
    event_ics = []
    for _, group in usable.groupby(EVENT_KEY):
        if len(group) >= 3 and group[column].nunique() > 1 and group[target].nunique() > 1:
            value = spearmanr(group[column], group[target]).statistic
            if np.isfinite(value):
                event_ics.append(value)
    return {"rows": len(usable), "events": usable[EVENT_KEY].drop_duplicates().shape[0],
            "pooled_spearman": spearmanr(usable[column], usable[target]).statistic,
            "mean_event_spearman": np.mean(event_ics), "event_ic_groups": len(event_ics)}


def main():
    paths = {"features": MODEL / "model_features.csv", "metadata": MODEL / "metadata.csv",
             "targets": MODEL / "targets.csv", "eligibility": MODEL / "eligibility.csv",
             "returns": RETURNS, "code": Path(__file__).resolve()}
    before = {name: sha256(path) for name, path in paths.items()}
    features = pd.read_csv(paths["features"], usecols=[
        "sample_id", "standardized_surprise", "residual_correlation", "network_signal"])
    metadata = pd.read_csv(paths["metadata"], usecols=[
        "sample_id", *EVENT_KEY, "receiver", "announcement_day", "entry_session"])
    targets = pd.read_csv(paths["targets"], usecols=["sample_id", "forward_benchmark_excess"])
    eligibility = pd.read_csv(paths["eligibility"], low_memory=False)
    data = features.merge(metadata, on="sample_id", validate="one_to_one").merge(
        targets, on="sample_id", validate="one_to_one").merge(
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

    lead_rows = []
    groups = list(data.groupby("announcement_day", sort=True))
    for number, (day, group) in enumerate(groups, 1):
        if number % 250 == 0:
            print(f"{number}/{len(groups)} announcement days", flush=True)
        candidates = set(group.source) | set(group.receiver)
        residuals = residual_history(wide, day, candidates)
        history_last_date = residuals.index.max() if len(residuals) else pd.NaT
        for row in group.itertuples():
            if row.source not in residuals or row.receiver not in residuals:
                forward, reverse, n_forward, n_reverse = np.nan, np.nan, 0, 0
            else:
                forward, n_forward = lag_corr(residuals[row.source], residuals[row.receiver])
                reverse, n_reverse = lag_corr(residuals[row.receiver], residuals[row.source])
            lead_rows.append({"sample_id": row.sample_id, "lead_corr_source_to_receiver": forward,
                              "lead_corr_receiver_to_source": reverse, "lead_asymmetry": forward - reverse,
                              "lead_obs": n_forward, "reverse_obs": n_reverse,
                              "lead_history_last_date": history_last_date})
    lead = pd.DataFrame(lead_rows)
    data = data.merge(lead, on="sample_id", validate="one_to_one")
    data["lead_signal"] = data.standardized_surprise * data.lead_corr_source_to_receiver
    data["asymmetry_signal"] = data.standardized_surprise * data.lead_asymmetry

    sessions, instruments = wide.index, wide.columns
    row_i = sessions.get_indexer(data.entry_session)
    receiver_i = instruments.get_indexer(data.receiver)
    spy_i = int(instruments.get_loc("SPY.P"))
    if (row_i < 0).any() or (receiver_i < 0).any():
        raise ValueError("entry session or receiver missing from return matrix")
    horizon_purge_counts = {}
    for horizon in [5, 10]:
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
    result_rows = []
    for split in ["training", "validation"]:
        frame = data.loc[data.split.eq(split)]
        for horizon in [5, 10]:
            for column in ["network_signal", "lead_signal", "asymmetry_signal"]:
                result_rows.append({"split": split, "horizon": horizon, "signal": column,
                                    **score(frame, column, f"target_{horizon}")})
    results = pd.DataFrame(result_rows)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    lead.to_csv(out / "lead_lag_features.csv", index=False)
    results.to_csv(out / "metrics.csv", index=False)
    after = {name: sha256(path) for name, path in paths.items()}
    temporal_check = data["lead_history_last_date"].lt(data["announcement_day"]).all()
    checks = {"inputs_unchanged": before == after, "no_test_rows": not data.split.eq("test").any(),
              "all_development_rows_retained": len(lead) == len(data),
              "lead_observations_never_use_event_day": bool(temporal_check)}
    summary = {"run_id": run_id, "scope": "exploratory training/validation lead-lag diagnostic",
               "rows": len(data), "lead_feature_missing": int(lead.lead_corr_source_to_receiver.isna().sum()),
               "horizon_boundary_purged_rows": horizon_purge_counts,
               "checks": checks, "input_hashes": before,
               "output_hashes": {"features": sha256(out / "lead_lag_features.csv"),
                                 "metrics": sha256(out / "metrics.csv")},
               "test_policy": "No test rows or test targets were computed."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "lead_feature_missing": summary["lead_feature_missing"],
                      "checks": checks, "metrics": results.to_dict(orient="records")}, indent=2))
    if not all(checks.values()):
        raise RuntimeError(checks)


if __name__ == "__main__":
    main()
