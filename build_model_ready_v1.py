"""Build a leakage-controlled model-ready layer from panel v3.

This stage engineers only backward-looking features and explicit eligibility
flags. It does not fit preprocessing, a model, or a portfolio, and it never
physically drops panel samples.
"""
from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PANEL_RUN_ID = "20260909T035245922341Z"
PANEL = ROOT / "data" / "panel_v3" / PANEL_RUN_ID
CLEAN = ROOT / "data" / "clean" / "v3" / "20260909T012417705069Z"
RETURNS_PATH = ROOT / "data" / "clean" / "v2" / "returns.csv"
CONFIG_PATH = ROOT / "model_ready_v1_config.json"
OUT_ROOT = ROOT / "data" / "model_ready_v1"
EVENT_KEY = ["source", "announcement", "period_end"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_date(values, name: str) -> pd.Series:
    result = pd.to_datetime(values, format="mixed", errors="raise").dt.normalize()
    if result.isna().any():
        raise ValueError(f"{name} contains missing dates")
    return result


def parse_bool(values, name: str) -> pd.Series:
    normalized = pd.Series(values).astype("string").str.strip().str.lower()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name} contains invalid booleans: {invalid[:10]}")
    return normalized.eq("true")


def matrix_lookup(matrix: pd.DataFrame, instruments: pd.Series, dates: pd.Series, name: str) -> np.ndarray:
    """Exact instrument/date lookup without temporal filling."""
    row_i = matrix.index.get_indexer(pd.DatetimeIndex(dates))
    col_i = matrix.columns.get_indexer(pd.Index(instruments))
    missing_instruments = sorted(set(instruments[col_i < 0]))
    if missing_instruments:
        raise RuntimeError(f"{name}: instruments absent from matrix: {missing_instruments[:10]}")
    output = np.full(len(instruments), np.nan)
    valid = (row_i >= 0) & (col_i >= 0)
    output[valid] = matrix.to_numpy()[row_i[valid], col_i[valid]]
    return output


def rolling_beta_idio(wide: pd.DataFrame, benchmark: pd.Series, window: int, min_obs: int):
    """Pairwise rolling OLS beta and annualized residual volatility."""
    valid = wide.notna().mul(benchmark.notna(), axis=0)
    count = valid.rolling(window, min_periods=1).sum()
    x = wide.where(valid)
    market = valid.mul(benchmark, axis=0)
    sum_x = x.rolling(window, min_periods=1).sum()
    sum_m = market.rolling(window, min_periods=1).sum()
    sum_x2 = x.pow(2).rolling(window, min_periods=1).sum()
    sum_m2 = market.pow(2).rolling(window, min_periods=1).sum()
    sum_xm = x.mul(benchmark, axis=0).rolling(window, min_periods=1).sum()
    covariance_numerator = sum_xm - sum_x * sum_m / count
    market_ss = sum_m2 - sum_m.pow(2) / count
    asset_ss = sum_x2 - sum_x.pow(2) / count
    valid_fit = (count >= min_obs) & np.isfinite(market_ss) & market_ss.gt(0)
    beta = (covariance_numerator / market_ss).where(valid_fit)
    residual_ss = (asset_ss - covariance_numerator.pow(2) / market_ss).clip(lower=0)
    residual_variance = residual_ss / (count - 2).where(count.gt(2))
    idio_vol = (np.sqrt(residual_variance) * np.sqrt(252.0)).where(valid_fit & count.gt(2))
    return beta, idio_vol, count.where(beta.notna())


def split_for_day(day: pd.Timestamp) -> str:
    if pd.Timestamp("2015-01-01") <= day <= pd.Timestamp("2020-12-31"):
        return "training"
    if pd.Timestamp("2021-01-01") <= day <= pd.Timestamp("2022-12-31"):
        return "validation"
    if pd.Timestamp("2023-01-01") <= day <= pd.Timestamp("2026-06-30"):
        return "test"
    return "out_of_window"


def semicolon_reasons(index: pd.Index, rules: list[tuple[str, pd.Series]]) -> pd.Series:
    output = pd.Series("", index=index, dtype="string")
    for reason, failed in rules:
        failed = failed.fillna(True).astype(bool)
        output.loc[failed] = output.loc[failed].map(lambda value: reason if not value else f"{value};{reason}")
    return output


def main():
    paths = {
        "features": PANEL / "features.csv",
        "labels": PANEL / "labels.csv",
        "execution": PANEL / "execution_inputs.csv",
        "diagnostics": PANEL / "diagnostics_ex_post.csv",
        "returns": RETURNS_PATH,
        "prices": CLEAN / "prices.csv",
        "config": CONFIG_PATH,
        "builder_code": Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    base = pd.read_csv(paths["features"], low_memory=False)
    labels = pd.read_csv(paths["labels"])
    execution = pd.read_csv(paths["execution"])
    diagnostics = pd.read_csv(paths["diagnostics"])
    for frame, name in [(base, "features"), (labels, "labels"),
                        (execution, "execution"), (diagnostics, "diagnostics")]:
        if frame.sample_id.duplicated().any() or frame.sample_id.isna().any():
            raise RuntimeError(f"{name} sample_id is not nonmissing and unique")
        if set(frame.sample_id) != set(base.sample_id):
            raise RuntimeError(f"{name} sample_id set differs from features")
    labels = base[["sample_id"]].merge(labels, on="sample_id", validate="one_to_one")
    execution = base[["sample_id"]].merge(execution, on="sample_id", validate="one_to_one")
    diagnostics = base[["sample_id"]].merge(diagnostics, on="sample_id", validate="one_to_one")
    base["announcement_day"] = parse_date(base.announcement_day, "announcement_day")
    base["period_end"] = parse_date(base.period_end, "period_end")
    base["snapshot"] = parse_date(base.snapshot, "snapshot")
    base["liquidity_reference_session"] = parse_date(
        base.liquidity_reference_session, "liquidity_reference_session")
    labels["entry_session"] = parse_date(labels.entry_session, "entry_session")
    labels["exit_session"] = parse_date(labels.exit_session, "exit_session")
    execution["entry_session"] = parse_date(execution.entry_session, "execution entry_session")
    if not labels.entry_session.equals(execution.entry_session):
        raise RuntimeError("Label and execution entry sessions differ")
    if not (base.liquidity_reference_session < labels.entry_session).all():
        raise RuntimeError("Feature reference session is not strictly before entry")

    returns = pd.read_csv(paths["returns"], usecols=[
        "Instrument", "Date", "return_decimal", "in_sp500_that_day"])
    returns["Date"] = parse_date(returns.Date, "returns Date")
    returns["in_sp500_that_day"] = parse_bool(
        returns.in_sp500_that_day, "returns in_sp500_that_day").to_numpy()
    if returns.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("Duplicate returns keys")
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    if "SPY.P" not in wide:
        raise RuntimeError("SPY.P missing")
    wide = wide.loc[wide["SPY.P"].notna()]
    sessions = wide.index
    if not base.liquidity_reference_session.isin(sessions).all():
        raise RuntimeError("Reference sessions are outside the SPY calendar")
    membership = returns.loc[~returns.Instrument.eq("SPY.P")].pivot(
        index="Date", columns="Instrument", values="in_sp500_that_day")
    membership = membership.reindex(index=sessions, columns=wide.columns).astype("boolean").fillna(False).astype(bool)

    engineered = pd.DataFrame({"sample_id": base.sample_id})
    engineered["standardized_surprise"] = pd.to_numeric(base.standardized_surprise, errors="coerce")
    engineered["surprise_abs"] = engineered.standardized_surprise.abs()
    engineered["surprise_sign"] = np.sign(engineered.standardized_surprise)
    engineered["residual_correlation"] = pd.to_numeric(base.residual_correlation, errors="coerce")
    engineered["abs_residual_correlation"] = engineered.residual_correlation.abs()
    engineered["network_signal"] = engineered.standardized_surprise * engineered.residual_correlation
    engineered["network_signal_abs"] = engineered.network_signal.abs()
    group = base.groupby(EVENT_KEY, sort=False, dropna=False)
    engineered["event_neighbor_count"] = group.sample_id.transform("size").astype(float)
    engineered["neighbor_abs_corr_rank"] = group.residual_correlation.transform(
        lambda values: values.abs().rank(method="first", ascending=False)).astype(float)
    engineered["snapshot_age_days"] = pd.to_numeric(base.snapshot_age_days, errors="coerce")
    analysts = pd.to_numeric(base.analysts, errors="coerce")
    engineered["analysts_log1p"] = np.log1p(analysts.where(analysts.ge(0)))

    print(f"samples {len(base)}; computing return features", flush=True)
    reference = base.liquidity_reference_session
    source, receiver = base.source, base.receiver
    if (wide.le(-1) & wide.notna()).any(axis=None):
        raise RuntimeError("Total return <= -100% prevents log compounding")
    log_returns = np.log1p(wide)
    return_matrices = {"ret_1": wide}
    for window in [5, 20, 60]:
        return_matrices[f"mom_{window}"] = np.expm1(
            log_returns.rolling(window, min_periods=window).sum())
    for window, min_obs in [(20, 15), (60, 40)]:
        return_matrices[f"vol_{window}_ann"] = (
            wide.rolling(window, min_periods=min_obs).std(ddof=1) * np.sqrt(252.0))
    beta, idio, beta_count = rolling_beta_idio(wide, wide["SPY.P"], 126, 100)
    return_matrices["beta_126"] = beta
    return_matrices["idio_vol_126_ann"] = idio
    return_matrices["beta_obs_126"] = beta_count

    for prefix, instruments in [("source", source), ("receiver", receiver)]:
        for name, matrix in return_matrices.items():
            engineered[f"{prefix}_{name}"] = matrix_lookup(
                matrix, instruments, reference, f"{prefix}_{name}")
    receiver_mom_rank = return_matrices["mom_20"].where(membership).rank(axis=1, pct=True)
    receiver_vol_rank = return_matrices["vol_20_ann"].where(membership).rank(axis=1, pct=True)
    engineered["receiver_mom_20_pct_rank"] = matrix_lookup(
        receiver_mom_rank, receiver, reference, "receiver_mom_20_pct_rank")
    engineered["receiver_vol_20_pct_rank"] = matrix_lookup(
        receiver_vol_rank, receiver, reference, "receiver_vol_20_pct_rank")
    ref_i = sessions.get_indexer(reference)
    if (ref_i < 0).any():
        raise RuntimeError("Market feature reference lookup failed")
    market_feature_map = {
        "market_mom_20": return_matrices["mom_20"]["SPY.P"],
        "market_mom_60": return_matrices["mom_60"]["SPY.P"],
        "market_vol_20_ann": return_matrices["vol_20_ann"]["SPY.P"],
        "market_vol_60_ann": return_matrices["vol_60_ann"]["SPY.P"],
    }
    for name, series in market_feature_map.items():
        engineered[name] = series.to_numpy()[ref_i]
    del log_returns, beta, idio, beta_count, receiver_mom_rank, receiver_vol_rank
    gc.collect()

    print("computing rolling liquidity features", flush=True)
    needed = sorted(set(source) | set(receiver))
    prices = pd.read_csv(paths["prices"], usecols=[
        "Instrument", "Date", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume",
        "in_sp500_that_day"])
    prices = prices.loc[prices.Instrument.isin(needed)].copy()
    prices["Date"] = parse_date(prices.Date, "price Date")
    prices["in_sp500_that_day"] = parse_bool(
        prices.in_sp500_that_day, "prices in_sp500_that_day").to_numpy()
    if prices.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("Duplicate price keys")
    price_membership = prices.pivot(
        index="Date", columns="Instrument", values="in_sp500_that_day").reindex(
        index=sessions, columns=needed).astype("boolean").fillna(False).astype(bool)
    volume = prices.pivot(index="Date", columns="Instrument", values="ACVOL_UNS").reindex(
        index=sessions, columns=needed)
    dollar_volume = prices.pivot(index="Date", columns="Instrument", values="dollar_volume").reindex(
        index=sessions, columns=needed)
    spread = prices.pivot(index="Date", columns="Instrument", values="quoted_spread_bps").reindex(
        index=sessions, columns=needed)
    if (volume.lt(0) & volume.notna()).any(axis=None) or (
            dollar_volume.lt(0) & dollar_volume.notna()).any(axis=None):
        raise RuntimeError("Negative volume or dollar volume")
    volume_med20 = volume.rolling(20, min_periods=15).median()
    dollar_volume_med20 = dollar_volume.rolling(20, min_periods=15).median()
    spread_med20 = spread.rolling(20, min_periods=10).median()
    liquidity_matrices = {
        "volume_median_20_log1p": np.log1p(volume_med20),
        "dollar_volume_median_20_log1p": np.log1p(dollar_volume_med20),
        "spread_median_20_bps": spread_med20,
    }
    for prefix, instruments in [("source", source), ("receiver", receiver)]:
        for name, matrix in liquidity_matrices.items():
            engineered[f"{prefix}_{name}"] = matrix_lookup(
                matrix, instruments, reference, f"{prefix}_{name}")
    receiver_dvol_rank = dollar_volume_med20.where(price_membership).rank(axis=1, pct=True)
    receiver_spread_rank = spread_med20.where(price_membership).rank(axis=1, pct=True)
    engineered["receiver_dollar_volume_20_pct_rank"] = matrix_lookup(
        receiver_dvol_rank, receiver, reference, "receiver_dollar_volume_20_pct_rank")
    engineered["receiver_spread_20_pct_rank"] = matrix_lookup(
        receiver_spread_rank, receiver, reference, "receiver_spread_20_pct_rank")
    lagged_dollar = pd.to_numeric(base.lagged_dollar_volume, errors="coerce")
    lagged_spread = pd.to_numeric(base.lagged_quoted_spread_bps, errors="coerce")
    engineered["receiver_dollar_volume_lag1_log1p"] = np.log1p(lagged_dollar.where(lagged_dollar.ge(0)))
    engineered["receiver_spread_lag1_bps"] = lagged_spread
    del prices, price_membership, volume, dollar_volume, spread
    del volume_med20, dollar_volume_med20, spread_med20, receiver_dvol_rank, receiver_spread_rank
    gc.collect()

    print("building event-level splits and independent eligibility flags", flush=True)
    split = base.announcement_day.map(split_for_day)
    next_boundary = split.map({"training": pd.Timestamp("2021-01-01"),
                               "validation": pd.Timestamp("2023-01-01")})
    boundary_purged = next_boundary.notna() & (
        labels.entry_session.ge(next_boundary) | labels.exit_session.ge(next_boundary))
    event_group = pd.DataFrame({
        "source": base.source, "announcement": base.announcement,
        "period_end": base.period_end, "split": split,
        "boundary_purged": boundary_purged,
    }).groupby(EVENT_KEY, dropna=False)
    if event_group.split.nunique().gt(1).any() or event_group.boundary_purged.nunique().gt(1).any():
        raise RuntimeError("An event crosses split or purge assignments")

    label_complete = parse_bool(labels.label_complete, "label_complete")
    member_at_entry = parse_bool(diagnostics.receiver_in_index_at_entry, "receiver_in_index_at_entry")
    price_row_available = parse_bool(execution.entry_price_row_available, "entry_price_row_available")
    entry_has_close = parse_bool(execution.entry_has_close, "entry_has_close")
    entry_has_volume = parse_bool(execution.entry_has_volume, "entry_has_volume")
    entry_has_quote = parse_bool(execution.entry_has_two_sided_quote, "entry_has_two_sided_quote")
    entry_has_dollar_volume = execution.entry_dollar_volume.notna()
    entry_trade_eligible = member_at_entry & price_row_available & entry_has_close
    capacity_data_available = entry_trade_eligible & entry_has_volume & entry_has_dollar_volume
    direct_spread_available = entry_has_quote

    core_features = [
        "standardized_surprise", "residual_correlation", "network_signal",
        "source_mom_20", "source_vol_20_ann", "source_beta_126",
        "receiver_mom_20", "receiver_vol_20_ann", "receiver_beta_126",
        "market_mom_20", "market_vol_20_ann",
        "receiver_dollar_volume_median_20_log1p", "receiver_spread_median_20_bps",
    ]
    feature_columns = [column for column in engineered.columns if column != "sample_id"]
    feature_core_available = engineered[core_features].notna().all(axis=1)
    feature_missing_count = engineered[feature_columns].isna().sum(axis=1).astype(int)
    core_missing_list = engineered[core_features].isna().apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1)
    split_retained = split.isin(["training", "validation", "test"]) & ~boundary_purged
    supervised_model_eligible = (
        split_retained & label_complete & entry_trade_eligible & feature_core_available)

    entry_reason = semicolon_reasons(base.index, [
        ("receiver_not_index_member_at_entry", ~member_at_entry),
        ("entry_price_row_missing", ~price_row_available),
        ("entry_close_missing", ~entry_has_close),
    ])
    supervised_reason = semicolon_reasons(base.index, [
        ("out_of_window", ~split.isin(["training", "validation", "test"])),
        ("boundary_purged", boundary_purged),
        ("label_incomplete", ~label_complete),
        ("entry_trade_ineligible", ~entry_trade_eligible),
        ("core_feature_missing", ~feature_core_available),
    ])

    metadata = base[[
        "sample_id", "source", "receiver", "announcement", "announcement_day", "period_end",
        "snapshot", "liquidity_reference_session", "actual_source", "actual_selector",
        "formal_entry_rule", "actual", "consensus", "dispersion", "analysts",
        "eps_difference", "graph_last_date", "lagged_dollar_volume_source",
        "lagged_price_adjustments"]].copy()
    metadata["entry_session"] = labels.entry_session
    metadata["exit_session"] = labels.exit_session
    eligibility = pd.DataFrame({
        "sample_id": base.sample_id,
        "split": split,
        "next_split_boundary": next_boundary,
        "boundary_purged": boundary_purged,
        "split_retained": split_retained,
        "receiver_in_index_at_entry": member_at_entry,
        "entry_price_row_available": price_row_available,
        "entry_has_close": entry_has_close,
        "entry_has_volume": entry_has_volume,
        "entry_has_two_sided_quote": entry_has_quote,
        "entry_has_dollar_volume": entry_has_dollar_volume,
        "entry_trade_eligible": entry_trade_eligible,
        "entry_ineligibility_reason": entry_reason,
        "capacity_data_available": capacity_data_available,
        "direct_spread_available": direct_spread_available,
        "supervised_label_available": label_complete,
        "feature_core_available": feature_core_available,
        "feature_missing_count": feature_missing_count,
        "core_feature_missing_list": core_missing_list,
        "supervised_model_eligible": supervised_model_eligible,
        "supervised_ineligibility_reason": supervised_reason,
    })
    targets = labels.copy()

    missing_rows = []
    for split_name in ["training", "validation", "test", "all"]:
        mask = pd.Series(True, index=engineered.index) if split_name == "all" else split.eq(split_name)
        denominator = int(mask.sum())
        for column in feature_columns:
            count = int(engineered.loc[mask, column].isna().sum())
            missing_rows.append({"split": split_name, "feature": column, "rows": denominator,
                                 "missing": count, "missing_fraction": count / denominator if denominator else np.nan})
    feature_missingness = pd.DataFrame(missing_rows)

    split_rows = []
    for split_name in ["training", "validation", "test", "out_of_window", "all"]:
        mask = pd.Series(True, index=base.index) if split_name == "all" else split.eq(split_name)
        split_rows.append({
            "split": split_name,
            "edge_rows": int(mask.sum()),
            "source_events": int(base.loc[mask, EVENT_KEY].drop_duplicates().shape[0]),
            "boundary_purged_rows": int((mask & boundary_purged).sum()),
            "label_available_rows": int((mask & label_complete).sum()),
            "entry_trade_eligible_rows": int((mask & entry_trade_eligible).sum()),
            "core_feature_available_rows": int((mask & feature_core_available).sum()),
            "supervised_model_eligible_rows": int((mask & supervised_model_eligible).sum()),
        })
    split_counts = pd.DataFrame(split_rows)

    forbidden_feature_columns = [column for column in feature_columns if column.startswith("entry_") or
                                 "forward_return" in column or column in {"label_complete", "exit_session"}]
    checks = {
        "all_panel_samples_retained": len(engineered) == len(base) == len(eligibility) == len(targets),
        "sample_ids_unique": all(not frame.sample_id.duplicated().any() for frame in
                                 [engineered, metadata, targets, eligibility]),
        "sample_id_sets_equal": all(set(frame.sample_id) == set(base.sample_id) for frame in
                                    [engineered, metadata, targets, eligibility]),
        "reference_strictly_before_entry": bool((base.liquidity_reference_session < labels.entry_session).all()),
        "features_exclude_entry_and_future_fields": not forbidden_feature_columns,
        "engineered_features_have_no_infinity": bool((
            np.isfinite(engineered[feature_columns].to_numpy(float)) |
            np.isnan(engineered[feature_columns].to_numpy(float))).all()),
        "event_split_single_valued": bool(event_group.split.nunique().le(1).all()),
        "event_purge_single_valued": bool(event_group.boundary_purged.nunique().le(1).all()),
        "eligibility_does_not_delete_rows": len(eligibility) == len(base),
        "supervised_eligibility_components_recompute": bool(np.array_equal(
            supervised_model_eligible.to_numpy(),
            (split_retained & label_complete & entry_trade_eligible & feature_core_available).to_numpy())),
        "rank_features_within_unit_interval": bool(engineered[[
            "receiver_mom_20_pct_rank", "receiver_vol_20_pct_rank",
            "receiver_dollar_volume_20_pct_rank", "receiver_spread_20_pct_rank"]].apply(
                lambda values: values.dropna().between(0, 1).all()).all()),
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    outputs = {
        "model_features": out / "model_features.csv",
        "metadata": out / "metadata.csv",
        "targets": out / "targets.csv",
        "eligibility": out / "eligibility.csv",
        "feature_missingness": out / "feature_missingness.csv",
        "split_counts": out / "split_counts.csv",
    }
    engineered.to_csv(outputs["model_features"], index=False)
    metadata.to_csv(outputs["metadata"], index=False)
    targets.to_csv(outputs["targets"], index=False)
    eligibility.to_csv(outputs["eligibility"], index=False)
    feature_missingness.to_csv(outputs["feature_missingness"], index=False)
    split_counts.to_csv(outputs["split_counts"], index=False)

    after_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if after_hashes != {name: item["sha256"] for name, item in input_hashes.items()}:
        raise RuntimeError("An input changed while building model-ready data")
    summary = {
        "run_id": run_id,
        "panel_run_id": PANEL_RUN_ID,
        "output_dir": rel(out),
        "rows": int(len(base)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "core_features": core_features,
        "config": config,
        "split_counts": split_counts.to_dict(orient="records"),
        "entry_trade_ineligible_rows": int((~entry_trade_eligible).sum()),
        "entry_trade_ineligibility_reason_counts": entry_reason.loc[entry_reason.ne("")].value_counts().to_dict(),
        "incomplete_label_rows_retained": int((~label_complete).sum()),
        "rows_with_any_feature_missing": int(feature_missing_count.gt(0).sum()),
        "preprocessing_fitted": False,
        "rows_physically_deleted": 0,
        "missing_value_policy": config["missing_policy"],
        "checks": checks,
        "input_hashes": input_hashes,
        "output_hashes": {name: sha256_file(path) for name, path in outputs.items()},
        "limitations": [
            "Eligibility flags are a frozen audit layer; no model-specific imputation has been chosen.",
            "Entry close availability is execution evidence and is not included in model features.",
            "Daily bid/ask is an end-of-day spread proxy, not a guaranteed executable quote.",
            "No model, parameter selection, portfolio construction, cost application, or performance claim is made.",
        ],
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "rows": len(base), "feature_count": len(feature_columns),
                      "split_counts": summary["split_counts"],
                      "entry_trade_ineligible_rows": summary["entry_trade_ineligible_rows"],
                      "incomplete_label_rows_retained": summary["incomplete_label_rows_retained"],
                      "rows_with_any_feature_missing": summary["rows_with_any_feature_missing"],
                      "checks": checks}, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError("Model-ready checks failed: " + ", ".join(k for k, v in checks.items() if not v))


if __name__ == "__main__":
    main()
