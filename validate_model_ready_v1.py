"""Independent validation for a frozen model-ready v1 run."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_RUN_ID = "20260909T064151673764Z"
MODEL = ROOT / "data" / "model_ready_v1" / MODEL_RUN_ID
PANEL = ROOT / "data" / "panel_v3" / "20260909T035245922341Z"
RETURNS_PATH = ROOT / "data" / "clean" / "v2" / "returns.csv"
PRICES_PATH = ROOT / "data" / "clean" / "v3" / "20260909T012417705069Z" / "prices.csv"
AUDIT_ROOT = ROOT / "data" / "audit" / "model_ready_v1"
EVENT_KEY = ["source", "announcement", "period_end"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_date(values):
    return pd.to_datetime(values, format="mixed", errors="raise").dt.normalize()


def parse_bool(values, name: str) -> pd.Series:
    normalized = pd.Series(values).astype("string").str.lower()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name}: {invalid}")
    return normalized.eq("true")


def close(left, right, atol=1e-11):
    return bool(np.allclose(np.asarray(left, float), np.asarray(right, float),
                            rtol=1e-9, atol=atol, equal_nan=True))


def split_for_day(day):
    if pd.Timestamp("2015-01-01") <= day <= pd.Timestamp("2020-12-31"):
        return "training"
    if pd.Timestamp("2021-01-01") <= day <= pd.Timestamp("2022-12-31"):
        return "validation"
    if pd.Timestamp("2023-01-01") <= day <= pd.Timestamp("2026-06-30"):
        return "test"
    return "out_of_window"


def direct_return_features(values: np.ndarray, market: np.ndarray) -> dict:
    output = {"ret_1": values[-1] if len(values) else np.nan}
    for window in [5, 20, 60]:
        block = values[-window:]
        output[f"mom_{window}"] = float(np.prod(1 + block) - 1) if len(block) == window and np.isfinite(block).all() else np.nan
    for window, min_obs in [(20, 15), (60, 40)]:
        block = values[-window:]
        good = block[np.isfinite(block)]
        output[f"vol_{window}_ann"] = float(np.std(good, ddof=1) * np.sqrt(252)) if len(good) >= min_obs else np.nan
    x, m = values[-126:], market[-126:]
    valid = np.isfinite(x) & np.isfinite(m)
    x, m = x[valid], m[valid]
    output["beta_obs_126"] = float(len(x)) if len(x) >= 100 else np.nan
    if len(x) >= 100:
        centered_x, centered_m = x - x.mean(), m - m.mean()
        market_ss = float(centered_m @ centered_m)
        if market_ss > 0:
            beta = float(centered_x @ centered_m / market_ss)
            residual = centered_x - beta * centered_m
            output["beta_126"] = beta
            output["idio_vol_126_ann"] = float(np.sqrt((residual @ residual) / (len(x) - 2)) * np.sqrt(252))
        else:
            output["beta_126"] = np.nan
            output["idio_vol_126_ann"] = np.nan
    else:
        output["beta_126"] = np.nan
        output["idio_vol_126_ann"] = np.nan
    return output


def main():
    paths = {
        "model_features": MODEL / "model_features.csv",
        "metadata": MODEL / "metadata.csv",
        "targets": MODEL / "targets.csv",
        "eligibility": MODEL / "eligibility.csv",
        "feature_missingness": MODEL / "feature_missingness.csv",
        "split_counts": MODEL / "split_counts.csv",
        "summary": MODEL / "summary.json",
        "panel_features": PANEL / "features.csv",
        "panel_labels": PANEL / "labels.csv",
        "panel_execution": PANEL / "execution_inputs.csv",
        "panel_diagnostics": PANEL / "diagnostics_ex_post.csv",
        "returns": RETURNS_PATH,
        "prices": PRICES_PATH,
        "validator_code": Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    model = pd.read_csv(paths["model_features"])
    metadata = pd.read_csv(paths["metadata"], low_memory=False)
    targets = pd.read_csv(paths["targets"])
    eligibility = pd.read_csv(paths["eligibility"], low_memory=False)
    panel_features = pd.read_csv(paths["panel_features"], low_memory=False)
    panel_labels = pd.read_csv(paths["panel_labels"])
    panel_execution = pd.read_csv(paths["panel_execution"])
    panel_diagnostics = pd.read_csv(paths["panel_diagnostics"])
    frames = [model, metadata, targets, eligibility]
    checks = {
        "all_model_ready_rows_retained": all(len(frame) == 107532 for frame in frames),
        "all_sample_ids_unique": all(not frame.sample_id.duplicated().any() for frame in frames),
        "all_sample_id_sets_equal": all(set(frame.sample_id) == set(model.sample_id) for frame in frames),
    }
    feature_columns = [column for column in model if column != "sample_id"]
    checks["feature_count_is_45"] = len(feature_columns) == 45 == summary["feature_count"]
    checks["feature_schema_matches_summary"] = feature_columns == summary["feature_columns"]
    checks["no_entry_or_future_feature_fields"] = not [
        column for column in feature_columns if column.startswith("entry_") or
        "forward_return" in column or column in {"label_complete", "exit_session"}]

    metadata["announcement_day"] = parse_date(metadata.announcement_day)
    metadata["liquidity_reference_session"] = parse_date(metadata.liquidity_reference_session)
    metadata["entry_session"] = parse_date(metadata.entry_session)
    metadata["exit_session"] = parse_date(metadata.exit_session)
    expected_split = metadata.announcement_day.map(split_for_day)
    observed_split = eligibility.split.astype(str)
    checks["split_assignment_recomputed"] = bool(expected_split.equals(observed_split))
    next_boundary = expected_split.map({"training": pd.Timestamp("2021-01-01"),
                                        "validation": pd.Timestamp("2023-01-01")})
    expected_purge = next_boundary.notna() & (
        metadata.entry_session.ge(next_boundary) | metadata.exit_session.ge(next_boundary))
    observed_purge = parse_bool(eligibility.boundary_purged, "boundary_purged")
    checks["boundary_purge_recomputed"] = bool(np.array_equal(
        expected_purge.to_numpy(), observed_purge.to_numpy()))
    checks["boundary_purge_is_14_edges_3_events"] = bool(expected_purge.sum() == 14 and
        metadata.loc[expected_purge, EVENT_KEY].drop_duplicates().shape[0] == 3)
    checks["event_split_single_valued"] = bool(pd.DataFrame({
        **{key: metadata[key] for key in EVENT_KEY}, "split": observed_split}).groupby(EVENT_KEY).split.nunique().le(1).all())

    order = model[["sample_id"]]
    panel_features = order.merge(panel_features, on="sample_id", validate="one_to_one")
    panel_labels = order.merge(panel_labels, on="sample_id", validate="one_to_one")
    panel_execution = order.merge(panel_execution, on="sample_id", validate="one_to_one")
    panel_diagnostics = order.merge(panel_diagnostics, on="sample_id", validate="one_to_one")
    expected_member = parse_bool(panel_diagnostics.receiver_in_index_at_entry, "receiver membership")
    expected_price_row = parse_bool(panel_execution.entry_price_row_available, "entry price row")
    expected_close = parse_bool(panel_execution.entry_has_close, "entry close")
    expected_entry_eligible = expected_member & expected_price_row & expected_close
    observed_entry_eligible = parse_bool(eligibility.entry_trade_eligible, "entry trade eligible")
    checks["entry_trade_eligibility_recomputed"] = bool(expected_entry_eligible.equals(observed_entry_eligible))
    checks["entry_trade_ineligible_is_17"] = int((~observed_entry_eligible).sum()) == 17
    expected_label = parse_bool(panel_labels.label_complete, "panel label complete")
    observed_label = parse_bool(eligibility.supervised_label_available, "supervised label available")
    checks["label_availability_recomputed"] = bool(expected_label.equals(observed_label))
    checks["incomplete_labels_are_33"] = int((~observed_label).sum()) == 33

    checks["standardized_surprise_copied"] = close(model.standardized_surprise, panel_features.standardized_surprise)
    checks["residual_correlation_copied"] = close(model.residual_correlation, panel_features.residual_correlation)
    checks["network_interactions_recomputed"] = all([
        close(model.surprise_abs, model.standardized_surprise.abs()),
        close(model.surprise_sign, np.sign(model.standardized_surprise)),
        close(model.abs_residual_correlation, model.residual_correlation.abs()),
        close(model.network_signal, model.standardized_surprise * model.residual_correlation),
        close(model.network_signal_abs, (model.standardized_surprise * model.residual_correlation).abs()),
    ])
    panel_event = panel_features.groupby(EVENT_KEY, sort=False, dropna=False)
    expected_count = panel_event.sample_id.transform("size").astype(float)
    expected_rank = panel_event.residual_correlation.transform(
        lambda values: values.abs().rank(method="first", ascending=False)).astype(float)
    checks["event_neighbor_count_recomputed"] = close(model.event_neighbor_count, expected_count)
    checks["neighbor_rank_recomputed"] = close(model.neighbor_abs_corr_rank, expected_rank)
    checks["snapshot_age_copied"] = close(model.snapshot_age_days, panel_features.snapshot_age_days)
    checks["analyst_transform_recomputed"] = close(
        model.analysts_log1p, np.log1p(pd.to_numeric(panel_features.analysts, errors="coerce")))

    core = summary["core_features"]
    expected_core = model[core].notna().all(axis=1)
    observed_core = parse_bool(eligibility.feature_core_available, "feature core available")
    checks["core_feature_availability_recomputed"] = bool(np.array_equal(
        expected_core.to_numpy(), observed_core.to_numpy()))
    split_retained = expected_split.isin(["training", "validation", "test"]) & ~expected_purge
    expected_supervised = split_retained & expected_label & expected_entry_eligible & expected_core
    observed_supervised = parse_bool(eligibility.supervised_model_eligible, "supervised model eligible")
    checks["supervised_eligibility_recomputed"] = bool(expected_supervised.equals(observed_supervised))
    checks["supervised_eligible_is_107270"] = int(observed_supervised.sum()) == 107270
    expected_missing_count = model[feature_columns].isna().sum(axis=1).astype(int)
    checks["feature_missing_count_recomputed"] = bool(np.array_equal(
        expected_missing_count.to_numpy(), eligibility.feature_missing_count.to_numpy(int)))
    checks["rows_with_any_feature_missing_is_213"] = int(expected_missing_count.gt(0).sum()) == 213
    rank_columns = ["receiver_mom_20_pct_rank", "receiver_vol_20_pct_rank",
                    "receiver_dollar_volume_20_pct_rank", "receiver_spread_20_pct_rank"]
    checks["rank_ranges_valid"] = bool(model[rank_columns].apply(
        lambda values: values.dropna().between(0, 1).all()).all())

    print("independently recomputing deterministic historical-feature sample", flush=True)
    returns = pd.read_csv(paths["returns"], usecols=["Instrument", "Date", "return_decimal"])
    returns["Date"] = parse_date(returns.Date)
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    wide = wide.loc[wide["SPY.P"].notna()]
    sessions = wide.index
    prices = pd.read_csv(paths["prices"], usecols=[
        "Instrument", "Date", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume"])
    prices["Date"] = parse_date(prices.Date)
    price_index = prices.set_index(["Instrument", "Date"]).sort_index()
    sample_positions = np.unique(np.linspace(0, len(model) - 1, 256, dtype=int))
    return_checks, liquidity_checks = [], []
    for position in sample_positions:
        meta = metadata.iloc[position]
        observed = model.iloc[position]
        ref_i = int(sessions.get_loc(meta.liquidity_reference_session))
        market = wide["SPY.P"].iloc[:ref_i + 1].to_numpy(float)
        for prefix, instrument in [("source", meta.source), ("receiver", meta.receiver)]:
            values = wide[instrument].iloc[:ref_i + 1].to_numpy(float)
            expected = direct_return_features(values, market)
            return_checks.extend(close([observed[f"{prefix}_{name}"]], [value])
                                 for name, value in expected.items())
            dates20 = sessions[max(0, ref_i - 19):ref_i + 1]
            try:
                block = price_index.xs(instrument, level=0).reindex(dates20)
            except KeyError:
                block = pd.DataFrame(index=dates20, columns=["ACVOL_UNS", "quoted_spread_bps", "dollar_volume"])
            specs = [
                ("volume_median_20_log1p", "ACVOL_UNS", 15, True),
                ("dollar_volume_median_20_log1p", "dollar_volume", 15, True),
                ("spread_median_20_bps", "quoted_spread_bps", 10, False),
            ]
            for feature, raw, minimum, log_value in specs:
                values_liq = pd.to_numeric(block[raw], errors="coerce").dropna()
                expected_liq = float(values_liq.median()) if len(values_liq) >= minimum else np.nan
                if log_value and np.isfinite(expected_liq):
                    expected_liq = float(np.log1p(expected_liq))
                liquidity_checks.append(close([observed[f"{prefix}_{feature}"]], [expected_liq]))
        market_expected = direct_return_features(market, market)
        for source_name, target_name in [("mom_20", "market_mom_20"), ("mom_60", "market_mom_60"),
                                         ("vol_20_ann", "market_vol_20_ann"),
                                         ("vol_60_ann", "market_vol_60_ann")]:
            return_checks.append(close([observed[target_name]], [market_expected[source_name]]))
    checks["sampled_return_features_recomputed"] = bool(all(return_checks))
    checks["sampled_liquidity_features_recomputed"] = bool(all(liquidity_checks))
    details = {
        "sampled_rows": int(len(sample_positions)),
        "sampled_return_comparisons": int(len(return_checks)),
        "sampled_liquidity_comparisons": int(len(liquidity_checks)),
        "boundary_purged_edges": int(expected_purge.sum()),
        "boundary_purged_events": int(metadata.loc[expected_purge, EVENT_KEY].drop_duplicates().shape[0]),
        "entry_trade_ineligible_rows": int((~observed_entry_eligible).sum()),
        "incomplete_label_rows": int((~observed_label).sum()),
        "rows_with_any_feature_missing": int(expected_missing_count.gt(0).sum()),
    }

    checks["builder_inputs_still_match_recorded_hashes"] = all(
        sha256_file(ROOT / item["path"]) == item["sha256"]
        for item in summary["input_hashes"].values())
    checks["model_ready_outputs_match_recorded_hashes"] = all(
        sha256_file(MODEL / f"{name}.csv") == digest
        for name, digest in summary["output_hashes"].items())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = AUDIT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    boundary_rows = metadata.loc[expected_purge, [
        "sample_id", "source", "receiver", "announcement", "announcement_day", "period_end",
        "entry_session", "exit_session"]].copy()
    boundary_rows["split"] = expected_split.loc[expected_purge].to_numpy()
    boundary_rows["next_split_boundary"] = next_boundary.loc[expected_purge].to_numpy()
    boundary_rows["purge_reason"] = "exit_at_or_after_next_split_boundary"
    boundary_rows.to_csv(out / "boundary_purged_rows.csv", index=False)
    entry_rows = metadata.loc[~observed_entry_eligible, [
        "sample_id", "source", "receiver", "announcement", "announcement_day", "period_end",
        "entry_session"]].copy()
    entry_rows = entry_rows.merge(eligibility[[
        "sample_id", "receiver_in_index_at_entry", "entry_price_row_available", "entry_has_close",
        "entry_ineligibility_reason"]], on="sample_id", validate="one_to_one")
    entry_rows.to_csv(out / "entry_ineligible_rows.csv", index=False)
    feature_rows = metadata.loc[expected_missing_count.gt(0), [
        "sample_id", "source", "receiver", "announcement", "announcement_day", "period_end",
        "liquidity_reference_session"]].copy()
    feature_rows["feature_missing_count"] = expected_missing_count.loc[expected_missing_count.gt(0)].to_numpy()
    feature_rows["missing_features"] = model.loc[expected_missing_count.gt(0), feature_columns].isna().apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1).to_numpy()
    feature_rows.to_csv(out / "feature_missing_rows.csv", index=False)

    audit_outputs = ["boundary_purged_rows.csv", "entry_ineligible_rows.csv", "feature_missing_rows.csv"]
    report = {
        "run_id": run_id,
        "model_ready_run_id": MODEL_RUN_ID,
        "read_only_validation": True,
        "passed": int(sum(checks.values())),
        "total": int(len(checks)),
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "details": details,
        "input_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "audit_output_hashes": {name: sha256_file(out / name) for name in audit_outputs},
    }
    (out / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in [
        "run_id", "model_ready_run_id", "passed", "total", "all_passed", "details"]},
        indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError("Validation failures: " + ", ".join(k for k, value in checks.items() if not value))


if __name__ == "__main__":
    main()
