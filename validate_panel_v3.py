"""Independent, read-only validation of a versioned v3 event/neighbour panel."""
from __future__ import annotations

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
RETURNS = ROOT / "data" / "clean" / "v2" / "returns.csv"
AUDIT_ROOT = ROOT / "data" / "audit" / "panel_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_date(values):
    return pd.to_datetime(values, format="mixed", errors="raise").dt.normalize()


def same_numeric(left, right, atol=1e-12):
    left = pd.to_numeric(pd.Series(left), errors="coerce").to_numpy(float)
    right = pd.to_numeric(pd.Series(right), errors="coerce").to_numpy(float)
    return bool(np.allclose(left, right, rtol=1e-10, atol=atol, equal_nan=True))


def same_text(left, right):
    left = pd.Series(left).fillna("<MISSING>").astype(str).to_numpy()
    right = pd.Series(right).fillna("<MISSING>").astype(str).to_numpy()
    return bool(np.array_equal(left, right))


def main():
    required = {name: PANEL / f"{name}.csv" for name in
                ["events", "features", "labels", "execution_inputs", "diagnostics_ex_post"]}
    required["summary"] = PANEL / "summary.json"
    required["actuals"] = CLEAN / "actuals.csv"
    required["estimates"] = CLEAN / "estimates_weekly.csv"
    required["prices"] = CLEAN / "prices.csv"
    required["returns"] = RETURNS
    required["validator_code"] = Path(__file__).resolve()
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    events = pd.read_csv(required["events"])
    features = pd.read_csv(required["features"], low_memory=False)
    labels = pd.read_csv(required["labels"])
    execution = pd.read_csv(required["execution_inputs"])
    diagnostics = pd.read_csv(required["diagnostics_ex_post"])
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    for frame, columns in [(events, ["announcement_day", "snapshot", "graph_last_date", "entry_session"]),
                           (features, ["announcement_day", "snapshot", "graph_last_date", "liquidity_reference_session"]),
                           (labels, ["entry_session", "exit_session"]),
                           (execution, ["entry_session"])]:
        for column in columns:
            if column in frame:
                frame[column] = parse_date(frame[column])

    sample_frames = {"features": features, "labels": labels, "execution_inputs": execution,
                     "diagnostics_ex_post": diagnostics}
    base_ids = pd.Index(features.sample_id)
    checks = {}
    details = {}
    checks["all_sample_ids_unique"] = all(not frame.sample_id.duplicated().any() for frame in sample_frames.values())
    checks["all_sample_id_sets_equal"] = all(set(frame.sample_id) == set(base_ids) for frame in sample_frames.values())
    details["row_counts"] = {name: int(len(frame)) for name, frame in {"events": events, **sample_frames}.items()}

    merged = features[["sample_id", "source", "receiver", "announcement_day", "period_end",
                       "snapshot", "actual_source", "formal_entry_rule", "liquidity_reference_session"]].merge(
        labels, on="sample_id", how="inner", validate="one_to_one")
    checks["entry_strictly_after_announcement_day"] = bool((merged.entry_session > merged.announcement_day).all())
    checks["liquidity_reference_precedes_entry"] = bool((merged.liquidity_reference_session < merged.entry_session).all())
    checks["snapshot_strictly_before_announcement_day"] = bool((merged.snapshot < merged.announcement_day).all())
    checks["formal_entry_rule_exact"] = set(merged.formal_entry_rule) == {
        "first_market_session_strictly_after_announcement_day"}
    forbidden = [column for column in features.columns if column in labels.columns and column != "sample_id"]
    forbidden += [column for column in features.columns if column.startswith("entry_") or
                  column.startswith("receiver_forward") or column.startswith("benchmark_forward") or
                  column.startswith("forward_") or column == "exit_session"]
    checks["features_exclude_entry_day_and_forward_fields"] = len(set(forbidden)) == 0
    details["forbidden_feature_columns"] = sorted(set(forbidden))

    actuals = pd.read_csv(required["actuals"], usecols=["Instrument", "announcement", "period_end", "actual_source"])
    actuals["announcement_day"] = parse_date(actuals.announcement)
    actuals["period_end"] = parse_date(actuals.period_end)
    actual_key = set(zip(actuals.Instrument, actuals.announcement_day, actuals.period_end, actuals.actual_source))
    panel_actual_key = set(zip(merged.source, merged.announcement_day, parse_date(merged.period_end), merged.actual_source))
    checks["features_trace_to_clean_actuals"] = panel_actual_key.issubset(actual_key)
    study_actuals = actuals.loc[actuals.announcement_day.between("2015-01-01", "2026-06-30", inclusive="both")]
    checks["all_study_actuals_retained_in_events_audit"] = len(events) == len(study_actuals)

    estimates = pd.read_csv(required["estimates"], usecols=["Instrument", "snapshot", "Period End Date"])
    estimates["snapshot"] = parse_date(estimates.snapshot)
    estimates["period_end"] = parse_date(estimates["Period End Date"])
    estimate_key = set(zip(estimates.Instrument, estimates.snapshot, estimates.period_end))
    panel_estimate_key = set(zip(merged.source, merged.snapshot, parse_date(merged.period_end)))
    checks["features_trace_to_exact_period_estimates"] = panel_estimate_key.issubset(estimate_key)

    returns = pd.read_csv(required["returns"], usecols=["Instrument", "Date", "return_decimal"])
    returns["Date"] = parse_date(returns.Date)
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    wide = wide.loc[wide["SPY.P"].notna()]
    sessions, instruments = wide.index, wide.columns
    forward = (1.0 + wide).rolling(5, min_periods=5).apply(np.prod, raw=True).shift(-5) - 1.0
    row_i = sessions.get_indexer(labels.entry_session)
    receiver_for_labels = features.set_index("sample_id").loc[labels.sample_id, "receiver"]
    col_i = instruments.get_indexer(receiver_for_labels)
    valid_keys = (row_i >= 0) & (col_i >= 0)
    expected_receiver = np.full(len(labels), np.nan)
    expected_benchmark = np.full(len(labels), np.nan)
    expected_exit = np.full(len(labels), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    has_horizon = valid_keys & ((row_i + 5) < len(sessions))
    expected_receiver[has_horizon] = forward.to_numpy()[row_i[has_horizon], col_i[has_horizon]]
    benchmark_i = int(instruments.get_loc("SPY.P"))
    expected_benchmark[has_horizon] = forward.to_numpy()[row_i[has_horizon], benchmark_i]
    expected_exit[has_horizon] = sessions.to_numpy()[row_i[has_horizon] + 5]
    expected_complete = np.isfinite(expected_receiver) & np.isfinite(expected_benchmark)
    checks["all_entry_sessions_in_benchmark_calendar"] = bool((row_i >= 0).all())
    checks["exit_session_recomputed"] = bool(pd.Series(labels.exit_session.to_numpy(dtype="datetime64[ns]")).equals(
        pd.Series(expected_exit)))
    checks["receiver_forward_return_recomputed"] = same_numeric(labels.receiver_forward_return, expected_receiver)
    checks["benchmark_forward_return_recomputed"] = same_numeric(labels.benchmark_forward_return, expected_benchmark)
    checks["label_complete_recomputed"] = bool(np.array_equal(labels.label_complete.to_numpy(bool), expected_complete))
    checks["excess_return_recomputed"] = same_numeric(
        labels.forward_benchmark_excess, expected_receiver - expected_benchmark)

    price_cols = ["Instrument", "Date", "TRDPRC_1", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume",
                  "dollar_volume_source", "has_close", "has_volume", "has_two_sided_quote",
                  "in_sp500_that_day", "price_adjustments"]
    prices = pd.read_csv(required["prices"], usecols=price_cols)
    prices["Date"] = parse_date(prices.Date)
    checks["price_keys_unique"] = not prices.duplicated(["Instrument", "Date"]).any()
    execution_join = features[["sample_id", "receiver"]].merge(execution, on="sample_id", validate="one_to_one")
    expected_px = execution_join.merge(prices, left_on=["receiver", "entry_session"],
                                       right_on=["Instrument", "Date"], how="left", indicator=True,
                                       validate="many_to_one")
    checks["entry_price_row_flag_recomputed"] = bool(np.array_equal(
        expected_px.entry_price_row_available.to_numpy(bool), expected_px._merge.eq("both").to_numpy()))
    checks["entry_close_recomputed"] = same_numeric(expected_px.entry_close, expected_px.TRDPRC_1)
    checks["entry_volume_recomputed"] = same_numeric(expected_px.entry_volume, expected_px.ACVOL_UNS)
    checks["entry_spread_recomputed"] = same_numeric(expected_px.entry_quoted_spread_bps,
                                                      expected_px.quoted_spread_bps)
    checks["entry_dollar_volume_recomputed"] = same_numeric(expected_px.entry_dollar_volume,
                                                              expected_px.dollar_volume)
    checks["entry_availability_flags_recomputed"] = all([
        np.array_equal(expected_px.entry_has_close.to_numpy(bool), expected_px.has_close.astype("boolean").fillna(False).to_numpy(bool)),
        np.array_equal(expected_px.entry_has_volume.to_numpy(bool), expected_px.has_volume.astype("boolean").fillna(False).to_numpy(bool)),
        np.array_equal(expected_px.entry_has_two_sided_quote.to_numpy(bool),
                       expected_px.has_two_sided_quote.astype("boolean").fillna(False).to_numpy(bool)),
    ])
    checks["entry_provenance_recomputed"] = (same_text(expected_px.entry_dollar_volume_source,
                                                        expected_px.dollar_volume_source) and
                                              same_text(expected_px.entry_price_adjustments,
                                                        expected_px.price_adjustments))
    lagged_join = features.merge(prices, left_on=["receiver", "liquidity_reference_session"],
                                 right_on=["Instrument", "Date"], how="left", indicator=True,
                                 validate="many_to_one")
    checks["lagged_price_row_flag_recomputed"] = bool(np.array_equal(
        lagged_join.lagged_price_row_available.to_numpy(bool), lagged_join._merge.eq("both").to_numpy()))
    checks["lagged_price_values_recomputed"] = all([
        same_numeric(lagged_join.lagged_close, lagged_join.TRDPRC_1),
        same_numeric(lagged_join.lagged_volume, lagged_join.ACVOL_UNS),
        same_numeric(lagged_join.lagged_quoted_spread_bps, lagged_join.quoted_spread_bps),
        same_numeric(lagged_join.lagged_dollar_volume, lagged_join.dollar_volume),
    ])
    checks["lagged_provenance_recomputed"] = (same_text(lagged_join.lagged_dollar_volume_source,
                                                         lagged_join.dollar_volume_source) and
                                               same_text(lagged_join.lagged_price_adjustments,
                                                         lagged_join.price_adjustments))
    details["entry_missing_counts"] = {
        "price_row": int((~execution.entry_price_row_available).sum()),
        "close": int(execution.entry_close.isna().sum()),
        "volume": int(execution.entry_volume.isna().sum()),
        "quote": int(execution.entry_quoted_spread_bps.isna().sum()),
        "dollar_volume": int(execution.entry_dollar_volume.isna().sum()),
    }
    checks["entry_missing_counts_match_summary"] = details["entry_missing_counts"] == {
        "price_row": summary["entry_price_row_missing"], "close": summary["entry_close_missing"],
        "volume": summary["entry_volume_missing"], "quote": summary["entry_quote_missing"],
        "dollar_volume": summary["entry_dollar_volume_missing"]}
    checks["amcr_provenance_preserved"] = (set(features.loc[features.source.eq("AMCR.N"), "actual_source"]) <=
                                            {"archived_v2_current_vendor_unavailable"})
    checks["evhc_excluded_from_panel"] = not (features.source.eq("EVHC.N^L16") |
                                               features.receiver.eq("EVHC.N^L16")).any()
    checks["builder_output_hashes_match"] = all(
        sha256_file(PANEL / f"{name}.csv") == digest for name, digest in summary["output_hashes"].items())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = AUDIT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    graph_ready = events.graph_status.eq("available")
    event_flow = pd.DataFrame([
        {"stage": "study_actuals", "count": len(events), "disposition": "retained in events.csv",
         "rule": "announcement_day in 2015-01-01 through 2026-06-30"},
        {"stage": "source_not_index_member", "count": int(events.status.eq("source_not_in_index_on_announcement").sum()),
         "disposition": "excluded from edge samples; retained in events.csv",
         "rule": "source must be an index member on announcement calendar day"},
        {"stage": "no_quarter_estimate", "count": int(events.status.eq("no_estimates_for_quarter").sum()),
         "disposition": "excluded from edge samples; retained in events.csv", "rule": "exact source and period_end match"},
        {"stage": "no_strict_pre_event_snapshot", "count": int(events.status.eq("no_preannouncement_snapshot").sum()),
         "disposition": "excluded from edge samples; retained in events.csv", "rule": "snapshot < announcement_day"},
        {"stage": "stale_snapshot", "count": int(events.status.eq("stale_snapshot").sum()),
         "disposition": "excluded from edge samples; retained in events.csv", "rule": "snapshot age <= 14 calendar days"},
        {"stage": "matched_snapshot", "count": int(events.status.eq("matched_snapshot").sum()),
         "disposition": "eligible for graph test", "rule": "strict prior exact-quarter snapshot within 14 days"},
        {"stage": "insufficient_graph_history", "count": int(events.graph_status.eq("insufficient_graph_history").sum()),
         "disposition": "excluded from edge samples; retained in events.csv", "rule": "126 prior sessions and >=100 pair observations"},
        {"stage": "graph_ready", "count": int(graph_ready.sum()), "disposition": "neighbor selection attempted",
         "rule": "graph uses sessions strictly before announcement_day"},
        {"stage": "graph_ready_zero_neighbors", "count": int((graph_ready & events.neighbor_count.fillna(-1).eq(0)).sum()),
         "disposition": "no forced edge; retained in events.csv", "rule": "absolute residual correlation >= 0.30"},
        {"stage": "selected_edges", "count": len(features), "disposition": "retained in all sample tables",
         "rule": "at most five threshold-passing neighbors per event"},
        {"stage": "complete_labels", "count": int(labels.label_complete.sum()), "disposition": "usable label",
         "rule": "all five receiver and SPY total returns available"},
        {"stage": "incomplete_labels", "count": int((~labels.label_complete).sum()),
         "disposition": "retained with label_complete=False; no imputation",
         "rule": "at least one required future return is missing or outside sample"},
    ])
    event_flow.to_csv(out / "event_flow.csv", index=False)
    missing_rows = execution_join[["sample_id", "receiver", "entry_session", "entry_price_row_available",
                                   "entry_close", "entry_volume", "entry_quoted_spread_bps",
                                   "entry_dollar_volume"]].copy()
    missing_rows["missing_price_row"] = ~missing_rows.entry_price_row_available
    missing_rows["missing_close"] = missing_rows.entry_close.isna()
    missing_rows["missing_volume"] = missing_rows.entry_volume.isna()
    missing_rows["missing_quote"] = missing_rows.entry_quoted_spread_bps.isna()
    missing_rows["missing_dollar_volume"] = missing_rows.entry_dollar_volume.isna()
    missing_rows = missing_rows.loc[missing_rows[["missing_price_row", "missing_close", "missing_volume",
                                                 "missing_quote", "missing_dollar_volume"]].any(axis=1)]
    missing_rows.to_csv(out / "entry_missing_rows.csv", index=False)
    details["event_flow_rows"] = int(len(event_flow))
    details["unique_entry_missing_samples"] = int(len(missing_rows))
    report = {"run_id": run_id, "panel_run_id": PANEL_RUN_ID, "read_only_validation": True,
              "passed": int(sum(checks.values())), "total": int(len(checks)), "all_passed": bool(all(checks.values())),
              "checks": checks, "details": details,
              "input_hashes": {name: sha256_file(path) for name, path in required.items()},
              "audit_output_hashes": {"event_flow": sha256_file(out / "event_flow.csv"),
                                      "entry_missing_rows": sha256_file(out / "entry_missing_rows.csv")}}
    (out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**{k: report[k] for k in ["run_id", "panel_run_id", "passed", "total", "all_passed"]},
                      "details": details}, indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError("Validation failures: " + ", ".join(k for k, value in checks.items() if not value))


if __name__ == "__main__":
    main()
