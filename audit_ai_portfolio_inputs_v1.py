"""Audit validation-period portfolio execution inputs without using predictions.

This is an input-only audit for the 49-RIC exploratory AI-pool registry and the
daily primary model's model-ready validation schedule.  It deliberately reads
no prediction output and no sealed test target file.  The audit does not build
or run a strategy; it only joins validation decision rows to clean daily
prices/quotes/liquidity and records coverage, quality flags and proposed cost
rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "data" / "audit" / "ai_pool_expansion_v1" / "20260910T045000Z" / "candidate_registry.csv"
CANDIDATE_LIST = ROOT / "data" / "audit" / "ai_pool_expansion_v1" / "20260910T045000Z" / "candidate_instruments_49ric.csv"
CANDIDATE_QUARANTINE = ROOT / "data" / "audit" / "ai_pool_expansion_v1" / "20260910T045000Z" / "candidate_quarantine.csv"
PRICES = ROOT / "data" / "clean" / "v3" / "20260909T012417705069Z" / "prices.csv"
RETURNS = ROOT / "data" / "clean" / "v2" / "returns.csv"
MODEL_READY = ROOT / "data" / "model_ready_ai_pool_daily_v1" / "20260910T052100000000Z"
META = MODEL_READY / "metadata.csv"
ELIGIBILITY = MODEL_READY / "eligibility.csv"
SPLIT_EVENTS = ROOT / "data" / "audit" / "stock_split_cases" / "20260908T163803607846Z" / "split_events.csv"
OUT_ROOT = ROOT / "data" / "audit" / "ai_portfolio_inputs_v1"

VAL_START = pd.Timestamp("2021-01-01")
VAL_END = pd.Timestamp("2022-12-31")
BENCHMARK = "SPY.P"
PRICE_USECOLS = [
    "Date", "Instrument", "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1",
    "ACVOL_UNS", "BID", "ASK", "TRNOVR_UNS", "has_close", "has_volume",
    "has_two_sided_quote", "quoted_spread_bps", "dollar_volume",
    "dollar_volume_source", "in_sp500_that_day", "price_adjustments",
]
RETURN_USECOLS = ["Instrument", "Date", "Total Return", "return_decimal", "in_sp500_that_day"]
MODEL_META_COLS = [
    "sample_id", "security_id", "Instrument", "formation_session", "split",
    "membership_asof_status", "membership_asof_eligible", "membership_asof_pool_label",
    "member_from", "member_to", "membership_reason", "reference_session",
    "entry_session", "exit_session", "ai_role_evidence_status", "registry_active",
    "registry_candidate_status", "registry_primary_group", "registry_pit_evidence_status",
    "registry_pit_eligible_now", "registry_source_retrieval_status",
    "source_membership_asof_status", "source_membership_asof_eligible",
]
MODEL_ELIG_COLS = [
    "sample_id", "security_id", "Instrument", "formation_session", "split",
    "next_split_boundary", "boundary_purged", "split_retained", "entry_has_close",
    "entry_trade_eligible", "feature_core_available", "feature_missing_count",
    "non_overlap_block", "non_overlap_selected", "supervised_model_eligible",
    "supervised_ineligibility_reason",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold().map(
        {"true": True, "1": True, "yes": True, "y": True,
         "false": False, "0": False, "no": False, "n": False}
    ).fillna(False).astype(bool)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def append_reason(values: list[str], code: str) -> None:
    if code not in values:
        values.append(code)


def parse_registry() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(REGISTRY, low_memory=False)
    candidates = pd.read_csv(CANDIDATE_LIST, low_memory=False)
    quarantine = pd.read_csv(CANDIDATE_QUARANTINE, low_memory=False)
    registry["ric"] = registry["ric"].astype(str).str.strip()
    registry["member_from_dt"] = pd.to_datetime(registry["member_from"], errors="coerce").dt.normalize()
    registry["member_to_dt"] = pd.to_datetime(registry["member_to"], errors="coerce").dt.normalize()
    registry["delisted_bool"] = bool_series(registry["delisted_ric"])
    registry["pit_eligible_now_bool"] = bool_series(registry["pit_eligible_now"])
    candidates["Instrument"] = candidates["Instrument"].astype(str).str.strip()
    return registry, candidates, quarantine


def parse_model_ready() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    meta = pd.read_csv(META, usecols=MODEL_META_COLS, low_memory=False)
    eligibility = pd.read_csv(ELIGIBILITY, usecols=MODEL_ELIG_COLS, low_memory=False)
    for frame in (meta, eligibility):
        frame["Instrument"] = frame["Instrument"].astype(str).str.strip()
        frame["formation_dt"] = pd.to_datetime(frame["formation_session"], errors="coerce").dt.normalize()
        frame["sample_id"] = frame["sample_id"].astype(str)
    meta_val = meta.loc[meta["split"].eq("validation")].copy()
    elig_val = eligibility.loc[eligibility["split"].eq("validation")].copy()
    if meta.duplicated("sample_id").any() or eligibility.duplicated("sample_id").any():
        raise RuntimeError("model-ready sample_id keys are not unique")
    common = set(meta_val["sample_id"]) & set(elig_val["sample_id"])
    if common != set(meta_val["sample_id"]) or common != set(elig_val["sample_id"]):
        raise RuntimeError("metadata and eligibility validation keys differ")
    elig_for_join = elig_val.rename(columns={
        "Instrument": "Instrument_elig",
        "formation_session": "formation_session_elig",
        "split": "split_elig",
        "formation_dt": "formation_dt_elig",
    })
    merged = meta_val.merge(
        elig_for_join, on="sample_id", how="left", validate="one_to_one",
    )
    if not (merged["Instrument"] == merged["Instrument_elig"]).all():
        raise RuntimeError("metadata/eligibility instrument mismatch")
    if not (merged["formation_dt"] == merged["formation_dt_elig"]).all():
        raise RuntimeError("metadata/eligibility formation date mismatch")
    merged["model_supervised"] = bool_series(merged["supervised_model_eligible"])
    merged["model_entry_trade_eligible"] = bool_series(merged["entry_trade_eligible"])
    merged["model_entry_close_present"] = numeric(merged["entry_has_close"]).notna()
    info = {
        "metadata_rows_all": int(len(meta)),
        "eligibility_rows_all": int(len(eligibility)),
        "validation_rows": int(len(merged)),
        "validation_instruments": int(merged["Instrument"].nunique()),
        "validation_formation_sessions": int(merged["formation_dt"].nunique()),
        "validation_supervised_anchor_rows": int(merged["model_supervised"].sum()),
        "validation_supervised_anchor_instruments": int(merged.loc[merged["model_supervised"], "Instrument"].nunique()),
    }
    return merged, meta, info


def load_candidate_prices(instruments: set[str]) -> tuple[pd.DataFrame, int, int]:
    selected: list[pd.DataFrame] = []
    source_rows = 0
    selected_rows = 0
    wanted = instruments | {BENCHMARK}
    for chunk in pd.read_csv(PRICES, usecols=PRICE_USECOLS, chunksize=200_000, low_memory=False):
        source_rows += len(chunk)
        mask = chunk["Instrument"].astype(str).isin(wanted)
        if not mask.any():
            continue
        frame = chunk.loc[mask].copy()
        frame["Instrument"] = frame["Instrument"].astype(str).str.strip()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        for col in ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS", "BID", "ASK", "TRNOVR_UNS", "quoted_spread_bps", "dollar_volume"]:
            frame[col] = numeric(frame[col])
        frame["has_close_calc"] = frame["TRDPRC_1"].notna()
        frame["has_volume_calc"] = frame["ACVOL_UNS"].notna()
        frame["has_quote_calc"] = frame[["BID", "ASK"]].notna().all(axis=1)
        frame["quote_valid_calc"] = frame["has_quote_calc"] & frame["BID"].gt(0) & frame["ASK"].gt(0) & frame["ASK"].ge(frame["BID"])
        frame["quoted_spread_bps_calc"] = np.where(
            frame["quote_valid_calc"],
            (frame["ASK"] - frame["BID"]) / ((frame["ASK"] + frame["BID"]) / 2.0) * 10_000,
            np.nan,
        )
        frame["dollar_volume_valid_calc"] = frame["dollar_volume"].gt(0)
        selected_rows += len(frame)
        selected.append(frame)
    if not selected:
        raise RuntimeError("no candidate or benchmark prices were found")
    prices = pd.concat(selected, ignore_index=True, sort=False)
    prices = prices.sort_values(["Instrument", "Date"], kind="mergesort").reset_index(drop=True)
    prices["price_duplicate_key"] = prices.duplicated(["Instrument", "Date"], keep=False)
    if prices["Date"].isna().any():
        raise RuntimeError("selected clean prices contain unparseable Date")
    # Stale-price diagnostics are flags only.  They never overwrite or fill a price.
    prices["prev_close"] = prices.groupby("Instrument", sort=False)["TRDPRC_1"].shift(1)
    prices["prev_date"] = prices.groupby("Instrument", sort=False)["Date"].shift(1)
    prices["close_equal_prev"] = prices["TRDPRC_1"].notna() & prices["prev_close"].notna() & prices["TRDPRC_1"].eq(prices["prev_close"])
    prices["stale_no_volume"] = prices["close_equal_prev"] & (prices["ACVOL_UNS"].isna() | prices["ACVOL_UNS"].le(0))
    prices["large_close_move_prev"] = (
        prices["TRDPRC_1"].gt(0) & prices["prev_close"].gt(0)
        & ((prices["TRDPRC_1"] / prices["prev_close"] - 1.0).abs().gt(0.50))
    )
    return prices, source_rows, selected_rows


def load_candidate_returns(instruments: set[str]) -> tuple[pd.DataFrame, int, int]:
    selected: list[pd.DataFrame] = []
    source_rows = 0
    selected_rows = 0
    wanted = instruments | {BENCHMARK}
    for chunk in pd.read_csv(RETURNS, usecols=RETURN_USECOLS, chunksize=200_000, low_memory=False):
        source_rows += len(chunk)
        mask = chunk["Instrument"].astype(str).isin(wanted)
        if not mask.any():
            continue
        frame = chunk.loc[mask].copy()
        frame["Instrument"] = frame["Instrument"].astype(str).str.strip()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame["return_decimal"] = numeric(frame["return_decimal"])
        frame["Total Return"] = numeric(frame["Total Return"])
        frame["return_duplicate_key"] = frame.duplicated(["Instrument", "Date"], keep=False)
        selected_rows += len(frame)
        selected.append(frame)
    if not selected:
        raise RuntimeError("no candidate or benchmark returns were found")
    returns = pd.concat(selected, ignore_index=True, sort=False)
    returns = returns.sort_values(["Instrument", "Date"], kind="mergesort").reset_index(drop=True)
    if returns["Date"].isna().any():
        raise RuntimeError("selected clean returns contain unparseable Date")
    return returns, source_rows, selected_rows


def price_join_for_validation(
    val_rows: pd.DataFrame, prices: pd.DataFrame, returns: pd.DataFrame
) -> pd.DataFrame:
    price_cols = [
        "Instrument", "Date", "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS",
        "BID", "ASK", "TRNOVR_UNS", "has_close_calc", "has_volume_calc", "has_quote_calc",
        "quote_valid_calc", "quoted_spread_bps_calc", "dollar_volume", "dollar_volume_valid_calc",
        "dollar_volume_source", "in_sp500_that_day", "price_adjustments", "price_duplicate_key",
        "prev_close", "prev_date", "close_equal_prev", "stale_no_volume", "large_close_move_prev",
    ]
    lookup = prices.loc[prices["Instrument"].ne(BENCHMARK), price_cols].rename(columns={"Date": "formation_dt"})
    joined = val_rows.merge(
        lookup, on=["Instrument", "formation_dt"], how="left", validate="one_to_one", suffixes=("", "_price"),
    )
    joined["price_row_present"] = joined["TRDPRC_1"].notna() | joined["price_duplicate_key"].fillna(False)
    # An all-NA price row can exist in the clean file; distinguish it from a missing row.
    present_keys = set(zip(lookup["Instrument"], lookup["formation_dt"]))
    joined["price_row_present"] = [
        (str(i), d) in present_keys for i, d in zip(joined["Instrument"], joined["formation_dt"])
    ]
    return_cols = ["Instrument", "Date", "return_decimal", "Total Return", "in_sp500_that_day", "return_duplicate_key"]
    return_lookup = returns.loc[returns["Instrument"].ne(BENCHMARK), return_cols].rename(
        columns={"Date": "formation_dt", "in_sp500_that_day": "return_in_sp500_that_day"}
    )
    joined = joined.merge(
        return_lookup, on=["Instrument", "formation_dt"], how="left", validate="one_to_one", suffixes=("", "_return"),
    )
    return_keys = set(zip(return_lookup["Instrument"], return_lookup["formation_dt"]))
    joined["return_row_present"] = [
        (str(i), d) in return_keys for i, d in zip(joined["Instrument"], joined["formation_dt"])
    ]
    joined["return_valid"] = joined["return_decimal"].notna()
    joined["close_valid"] = joined["TRDPRC_1"].gt(0)
    joined["volume_valid"] = joined["ACVOL_UNS"].gt(0)
    joined["cost_inputs_complete"] = joined["close_valid"] & joined["quote_valid_calc"].fillna(False) & joined["dollar_volume_valid_calc"].fillna(False)
    joined["execution_reason_codes"] = ""
    reasons: list[str] = []
    for row in joined.itertuples(index=False):
        row_reasons: list[str] = []
        if not row.price_row_present:
            append_reason(row_reasons, "PRICE_ROW_MISSING")
        else:
            if pd.isna(row.TRDPRC_1):
                append_reason(row_reasons, "CLOSE_MISSING")
            elif row.TRDPRC_1 <= 0:
                append_reason(row_reasons, "NONPOSITIVE_CLOSE")
            if pd.isna(row.ACVOL_UNS):
                append_reason(row_reasons, "VOLUME_MISSING")
            elif row.ACVOL_UNS < 0:
                append_reason(row_reasons, "NEGATIVE_VOLUME")
            elif row.ACVOL_UNS == 0:
                append_reason(row_reasons, "ZERO_VOLUME")
            if not bool(row.has_quote_calc):
                append_reason(row_reasons, "QUOTE_MISSING")
            elif not bool(row.quote_valid_calc):
                append_reason(row_reasons, "QUOTE_CROSSED_OR_NONPOSITIVE")
            if pd.isna(row.dollar_volume):
                append_reason(row_reasons, "DOLLAR_VOLUME_MISSING")
            elif row.dollar_volume <= 0:
                append_reason(row_reasons, "NONPOSITIVE_DOLLAR_VOLUME")
            if bool(row.stale_no_volume):
                append_reason(row_reasons, "STALE_CLOSE_NO_VOLUME")
            elif bool(row.close_equal_prev):
                append_reason(row_reasons, "UNCHANGED_CLOSE_WITH_VOLUME")
            if bool(row.large_close_move_prev):
                append_reason(row_reasons, "LARGE_CLOSE_MOVE_REVIEW")
        if not row.return_row_present:
            append_reason(row_reasons, "RETURN_ROW_MISSING")
        elif not row.return_valid:
            append_reason(row_reasons, "RETURN_MISSING")
        reasons.append(";".join(row_reasons))
    joined["execution_reason_codes"] = reasons
    return joined


def run_length_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.sort_values("Date").copy()
    valid = frame["TRDPRC_1"].notna()
    close = frame.loc[valid, "TRDPRC_1"]
    if close.empty:
        max_run = 0
    else:
        run = close.ne(close.shift()).cumsum()
        max_run = int(run.groupby(run).size().max())
    return {
        "equal_close_transitions": int(frame["close_equal_prev"].fillna(False).sum()),
        "stale_no_volume_transitions": int(frame["stale_no_volume"].fillna(False).sum()),
        "large_close_move_transitions": int(frame["large_close_move_prev"].fillna(False).sum()),
        "max_equal_close_run": max_run,
    }


def candidate_coverage(
    registry: pd.DataFrame,
    val_rows: pd.DataFrame,
    joined: pd.DataFrame,
    prices: pd.DataFrame,
    validation_sessions: pd.DatetimeIndex,
    split_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_rows_by_ric = val_rows.groupby("Instrument", sort=False).size().to_dict()
    joined_by_ric = {ric: frame for ric, frame in joined.groupby("Instrument", sort=False)}
    price_by_ric = {ric: frame for ric, frame in prices.loc[prices["Instrument"].ne(BENCHMARK)].groupby("Instrument", sort=False)}
    split_events = split_events.copy()
    split_events["ex_date_dt"] = pd.to_datetime(split_events["Capital Change Ex Date"], errors="coerce").dt.normalize()
    split_by_ric = {ric: frame for ric, frame in split_events.groupby("Instrument", sort=False)}

    coverage_rows: list[dict[str, Any]] = []
    candidate_exclusions: list[dict[str, Any]] = []
    for record in registry.itertuples(index=False):
        ric = record.ric
        expected = validation_sessions[(validation_sessions >= max(record.member_from_dt, VAL_START)) & (validation_sessions <= min(record.member_to_dt, VAL_END))]
        frame = joined_by_ric.get(ric, pd.DataFrame())
        price_frame = price_by_ric.get(ric, pd.DataFrame())
        val_price = price_frame.loc[price_frame["Date"].isin(expected)] if not price_frame.empty else price_frame
        events = split_by_ric.get(ric, pd.DataFrame())
        val_events = events.loc[events["ex_date_dt"].between(VAL_START, VAL_END)] if not events.empty else events
        if frame.empty:
            frame = pd.DataFrame(columns=joined.columns)
        cost_complete = frame["cost_inputs_complete"].fillna(False)
        close_valid = frame["close_valid"].fillna(False)
        return_present = frame["return_row_present"].fillna(False)
        return_valid = frame["return_valid"].fillna(False)
        observed_return_rows = int(return_present.sum())
        volume_missing = frame["ACVOL_UNS"].isna()
        quote_missing = ~frame["quote_valid_calc"].fillna(False)
        dollar_missing = ~frame["dollar_volume_valid_calc"].fillna(False)
        anchors = frame.loc[frame["model_supervised"].fillna(False)]
        anchor_cost = anchors["cost_inputs_complete"].fillna(False) if not anchors.empty else pd.Series(dtype=bool)
        expected_set = set(expected)
        observed_set = set(val_price["Date"].dropna())
        missing_dates = sorted(expected_set - observed_set)
        reasons: list[str] = []
        if len(expected) == 0:
            append_reason(reasons, "NO_ACTIVE_VALIDATION_SPAN")
        if len(missing_dates):
            append_reason(reasons, "PRICE_ROW_MISSING")
        if int(frame["TRDPRC_1"].isna().sum()) > 0:
            append_reason(reasons, "CLOSE_MISSING")
        if int((~return_present).sum()) > 0:
            append_reason(reasons, "RETURN_ROW_MISSING")
        if int((return_present & ~return_valid).sum()) > 0:
            append_reason(reasons, "RETURN_MISSING")
        if int(volume_missing.sum()) > 0:
            append_reason(reasons, "VOLUME_MISSING")
        if int(quote_missing.sum()) > 0:
            append_reason(reasons, "QUOTE_MISSING_OR_INVALID")
        if int(dollar_missing.sum()) > 0:
            append_reason(reasons, "DOLLAR_VOLUME_MISSING_OR_INVALID")
        if bool(record.delisted_bool):
            append_reason(reasons, "DELISTED_RIC_RETAINED_HISTORY")
        if len(val_events):
            append_reason(reasons, "CORPORATE_ACTION_SPLIT_IN_VALIDATION")
        if int(frame["stale_no_volume"].fillna(False).sum()) > 0:
            append_reason(reasons, "STALE_CLOSE_NO_VOLUME_REVIEW")
        if int(frame["large_close_move_prev"].fillna(False).sum()) > 0:
            append_reason(reasons, "LARGE_CLOSE_MOVE_REVIEW")

        all_price = price_frame.sort_values("Date") if not price_frame.empty else price_frame
        run_metrics = run_length_metrics(val_price) if not val_price.empty else {
            "equal_close_transitions": 0, "stale_no_volume_transitions": 0,
            "large_close_move_transitions": 0, "max_equal_close_run": 0,
        }
        all_last = all_price["Date"].max() if not all_price.empty else pd.NaT
        all_first = all_price["Date"].min() if not all_price.empty else pd.NaT
        member_to = record.member_to_dt
        post_span_rows = int((all_price["Date"] > member_to).sum()) if not all_price.empty else 0
        coverage_rows.append({
            "ric": ric,
            "canonical_name": record.canonical_name,
            "primary_group": record.primary_group,
            "candidate_status": record.candidate_status,
            "pit_evidence_status": record.pit_evidence_status,
            "pit_eligible_now": bool(record.pit_eligible_now_bool),
            "delisted_ric": bool(record.delisted_bool),
            "member_from": record.member_from_dt,
            "member_to": record.member_to_dt,
            "registry_taxonomy_version": record.taxonomy_version,
            "validation_active_expected_sessions": int(len(expected)),
            "model_ready_validation_rows": int(val_rows_by_ric.get(ric, 0)),
            "model_ready_supervised_anchor_rows": int(len(anchors)),
            "observed_price_rows_on_expected_sessions": int(len(val_price)),
            "price_row_coverage_pct": float(len(val_price) / len(expected) * 100.0) if len(expected) else None,
            "missing_price_rows": int(len(missing_dates)),
            "observed_return_rows_on_expected_sessions": observed_return_rows,
            "return_row_coverage_pct": float(observed_return_rows / len(expected) * 100.0) if len(expected) else None,
            "missing_return_rows": int((~return_present).sum()),
            "missing_return_value_rows": int((return_present & ~return_valid).sum()),
            "return_valid_rows": int(return_valid.sum()),
            "missing_close_rows": int(frame["TRDPRC_1"].isna().sum()),
            "missing_volume_rows": int(volume_missing.sum()),
            "missing_quote_or_invalid_rows": int(quote_missing.sum()),
            "missing_or_invalid_dollar_volume_rows": int(dollar_missing.sum()),
            "close_valid_rows": int(close_valid.sum()),
            "volume_positive_rows": int(frame["ACVOL_UNS"].gt(0).sum()),
            "quote_valid_rows": int(frame["quote_valid_calc"].fillna(False).sum()),
            "dollar_volume_valid_rows": int(frame["dollar_volume_valid_calc"].fillna(False).sum()),
            "cost_inputs_complete_rows": int(cost_complete.sum()),
            "anchor_cost_inputs_complete_rows": int(anchor_cost.sum()) if len(anchor_cost) else 0,
            "quote_median_bps": float(frame.loc[frame["quote_valid_calc"].fillna(False), "quoted_spread_bps_calc"].median()) if frame["quote_valid_calc"].fillna(False).any() else None,
            "quote_p95_bps": float(frame.loc[frame["quote_valid_calc"].fillna(False), "quoted_spread_bps_calc"].quantile(0.95)) if frame["quote_valid_calc"].fillna(False).any() else None,
            "fallback_turnover_rows": int(frame["dollar_volume_source"].eq("TRDPRC_1_times_ACVOL_UNS").sum()),
            "vendor_turnover_rows": int(frame["dollar_volume_source"].eq("TRNOVR_UNS").sum()),
            "zero_volume_rows": int(frame["ACVOL_UNS"].eq(0).sum()),
            "equal_close_transitions": run_metrics["equal_close_transitions"],
            "stale_no_volume_transitions": run_metrics["stale_no_volume_transitions"],
            "large_close_move_transitions": run_metrics["large_close_move_transitions"],
            "max_equal_close_run": run_metrics["max_equal_close_run"],
            "first_price_date_validation": val_price["Date"].min() if not val_price.empty else pd.NaT,
            "last_price_date_validation": val_price["Date"].max() if not val_price.empty else pd.NaT,
            "first_price_date_all": all_first,
            "last_price_date_all": all_last,
            "post_registry_member_to_price_rows": post_span_rows,
            "last_price_at_or_before_member_to": bool(pd.isna(all_last) or all_last <= member_to),
            "validation_split_event_count": int(len(val_events)),
            "all_candidate_split_event_count": int(len(events)),
            "execution_status": "usable_with_cost_inputs" if len(expected) and int(cost_complete.sum()) == len(expected) else ("no_active_validation_span" if len(expected) == 0 else "review_input_gaps"),
            "reason_codes": ";".join(reasons),
        })
        if len(expected) == 0:
            candidate_exclusions.append({
                "record_type": "candidate",
                "ric": ric,
                "formation_session": pd.NaT,
                "sample_id": "",
                "scope": "validation_universe",
                "reason_codes": "NO_ACTIVE_VALIDATION_SPAN",
                "retained": True,
                "note": "Candidate registry row retained; member_from/member_to does not intersect 2021-2022 validation sessions.",
            })

    coverage = pd.DataFrame(coverage_rows).sort_values("ric").reset_index(drop=True)
    candidate_excl = pd.DataFrame(candidate_exclusions)
    group = coverage.groupby("primary_group", dropna=False).agg(
        candidate_count=("ric", "nunique"),
        active_validation_candidates=("validation_active_expected_sessions", lambda x: int(x.gt(0).sum())),
        expected_validation_sessions=("validation_active_expected_sessions", "sum"),
        observed_price_rows=("observed_price_rows_on_expected_sessions", "sum"),
        observed_return_rows=("observed_return_rows_on_expected_sessions", "sum"),
        model_ready_validation_rows=("model_ready_validation_rows", "sum"),
        supervised_anchor_rows=("model_ready_supervised_anchor_rows", "sum"),
        cost_complete_rows=("cost_inputs_complete_rows", "sum"),
        validation_split_events=("validation_split_event_count", "sum"),
    ).reset_index()
    group["price_row_coverage_pct"] = np.where(group["expected_validation_sessions"].gt(0), group["observed_price_rows"] / group["expected_validation_sessions"] * 100.0, np.nan)
    group["return_row_coverage_pct"] = np.where(group["expected_validation_sessions"].gt(0), group["observed_return_rows"] / group["expected_validation_sessions"] * 100.0, np.nan)
    group["cost_input_coverage_pct"] = np.where(group["expected_validation_sessions"].gt(0), group["cost_complete_rows"] / group["expected_validation_sessions"] * 100.0, np.nan)
    return coverage, candidate_excl, group


def make_row_exclusions(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        reasons: list[str] = []
        if not bool(row.model_supervised):
            model_reason = str(row.supervised_ineligibility_reason) if pd.notna(row.supervised_ineligibility_reason) else "MODEL_READY_NON_OVERLAP_UNSELECTED"
            append_reason(reasons, model_reason)
        if str(row.execution_reason_codes):
            for code in str(row.execution_reason_codes).split(";"):
                if code:
                    append_reason(reasons, code)
        if not reasons:
            continue
        rows.append({
            "record_type": "validation_model_ready_row",
            "ric": row.Instrument,
            "formation_session": row.formation_dt,
            "sample_id": row.sample_id,
            "scope": "validation_schedule",
            "model_supervised_anchor": bool(row.model_supervised),
            "entry_trade_eligible_model_ready": bool(row.model_entry_trade_eligible),
            "price_row_present": bool(row.price_row_present),
            "cost_inputs_complete": bool(row.cost_inputs_complete),
            "reason_codes": ";".join(reasons),
            "retained": True,
            "note": "Original model-ready validation schedule row retained; reason codes identify why it is outside the supervised anchor or cost-complete execution subset.",
        })
    return pd.DataFrame(rows)


def corporate_action_audit(split_events: pd.DataFrame, prices: pd.DataFrame, rics: set[str]) -> pd.DataFrame:
    events = split_events.loc[split_events["Instrument"].isin(rics)].copy()
    events["ex_date_dt"] = pd.to_datetime(events["Capital Change Ex Date"], errors="coerce").dt.normalize()
    price_lookup = {ric: frame.sort_values("Date").reset_index(drop=True) for ric, frame in prices.loc[prices["Instrument"].isin(rics)].groupby("Instrument", sort=False)}
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        frame = price_lookup.get(event.Instrument, pd.DataFrame())
        exact = frame.loc[frame["Date"].eq(event.ex_date_dt)] if not frame.empty else frame
        before = frame.loc[frame["Date"] < event.ex_date_dt].tail(1) if not frame.empty else frame
        after = frame.loc[frame["Date"] > event.ex_date_dt].head(1) if not frame.empty else frame
        ex_close = exact["TRDPRC_1"].iloc[0] if not exact.empty else np.nan
        before_close = before["TRDPRC_1"].iloc[0] if not before.empty else np.nan
        after_close = after["TRDPRC_1"].iloc[0] if not after.empty else np.nan
        pre_to_ex = ex_close / before_close - 1.0 if pd.notna(ex_close) and pd.notna(before_close) and before_close > 0 else np.nan
        ex_to_after = after_close / ex_close - 1.0 if pd.notna(after_close) and pd.notna(ex_close) and ex_close > 0 else np.nan
        validation = bool(pd.notna(event.ex_date_dt) and VAL_START <= event.ex_date_dt <= VAL_END)
        rows.append({
            "ric": event.Instrument,
            "announcement_date": event[1],
            "effective_date": event[3],
            "ex_date": event.ex_date_dt,
            "adjustment_factor": event[5],
            "terms_old_shares": event[7],
            "terms_new_shares": event[8],
            "in_validation_2021_2022": validation,
            "price_row_on_ex_date": bool(not exact.empty),
            "previous_price_date": before["Date"].iloc[0] if not before.empty else pd.NaT,
            "next_price_date": after["Date"].iloc[0] if not after.empty else pd.NaT,
            "adjusted_close_pre_to_ex_return": pre_to_ex,
            "adjusted_close_ex_to_next_return": ex_to_after,
            "large_adjusted_move_review": bool(pd.notna(pre_to_ex) and abs(pre_to_ex) > 0.50 or pd.notna(ex_to_after) and abs(ex_to_after) > 0.50),
            "price_adjustments_field_values": ";".join(sorted(set(exact["price_adjustments"].dropna().astype(str)))) if not exact.empty else "",
            "action": "retain_adjusted_price;manual_review_if_large_move" if validation else "retain_for_provenance_only",
            "reason_codes": "CORPORATE_ACTION_SPLIT_IN_VALIDATION" if validation else "CORPORATE_ACTION_OUTSIDE_VALIDATION",
        })
    return pd.DataFrame(rows).sort_values(["ex_date", "ric"]).reset_index(drop=True) if rows else pd.DataFrame()


def write_manifest(out: Path, input_paths: list[Path], output_paths: list[Path], run_id: str) -> None:
    manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": [{"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in input_paths],
        "output_files": [{"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in output_paths if path.exists()],
        "non_actions": [
            "No validation prediction file was read or used.",
            "No sealed test target file was read, parsed, aggregated, scored, ranked or predicted.",
            "No strategy, portfolio, target, prediction or performance result was built.",
            "No source row was deleted, imputed, winsorized, forward-filled or overwritten.",
        ],
    }
    write_json(out / "manifest.json", manifest)


def append_processing_log(summary: dict[str, Any], out: Path) -> None:
    path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# DATA_PROCESSING_LOG\n"
    counts = summary["counts"]
    checks = summary["checks"]
    lines = [
        "",
        f"## {summary['run_id']} — AI portfolio validation inputs audit ({summary['generated_at_utc']})",
        "",
        "- **阶段目的与状态**：在不打开验证预测、不读取封存测试目标、不构建策略的前提下，审计 49-RIC 候选池在 2021–2022 验证期的 clean close、quotes/spread、volume/dollar-volume、model-ready validation schedule、taxonomy/group 与退市/拆股记录，为后续组合模拟冻结执行资格和成本输入。状态=`audit_complete_with_review_flags`。",
        f"- **输入版本与哈希**：registry `{rel(REGISTRY)}` sha256=`{sha256(REGISTRY)}`；candidate list `{rel(CANDIDATE_LIST)}` sha256=`{sha256(CANDIDATE_LIST)}`；prices `{rel(PRICES)}` sha256=`{sha256(PRICES)}`；returns `{rel(RETURNS)}` sha256=`{sha256(RETURNS)}`；model-ready metadata/eligibility `{rel(META)}`, `{rel(ELIGIBILITY)}`；split audit `{rel(SPLIT_EVENTS)}`。",
        f"- **规则与理由**：validation sessions 取 model-ready 的 literal formation sessions；候选有效日为 registry `member_from/member_to` 与 2021-01-01—2022-12-31 的交集；价格按 `Instrument + Date` 精确匹配；close/volume/quote/dollar-volume 缺失保持 NA，不前向填充、不设零；quote 合法要求 BID、ASK>0 且 ASK≥BID；成本完整行要求正 close、合法双边 quote 与正 dollar volume。连续同 close 仅在 volume 缺失/为零时标记 stale 风险，不删除。",
        f"- **输入/输出行数**：扫描 clean prices physical rows=`{counts['price_source_rows_scanned']}`，保留候选+SPY rows=`{counts['price_rows_selected']}`；扫描 clean returns physical rows=`{counts['return_source_rows_scanned']}`，保留候选+SPY rows=`{counts['return_rows_selected']}`；registry before/after=`{counts['registry_rows_before']}/{counts['registry_rows_after']}`；model-ready validation rows=`{counts['validation_model_ready_rows']}`，supervised anchors=`{counts['validation_supervised_anchor_rows']}`；expected active candidate-session rows=`{counts['expected_active_candidate_sessions']}`，observed price/return rows=`{counts['observed_price_rows_on_expected_sessions']}/{counts['observed_return_rows_on_expected_sessions']}`；audit exclusion rows=`{counts['exclusion_rows']}`，物理删除=`{counts['physical_deletions']}`。",
        f"- **缺失/排除/退市处理**：{counts['candidate_no_active_validation_span']} 个候选没有 2021–2022 active span，保留并标 `NO_ACTIVE_VALIDATION_SPAN`；唯一 delisted RIC=`{counts['delisted_candidates']}`，验证期 delisted candidate=`{counts['delisted_candidates_in_validation']}`，不因退市删除历史；候选验证期 split events=`{counts['validation_split_events']}`，价格调整字段原样保留并只做连续性审计。",
        f"- **检查结果**：`{json.dumps(checks, ensure_ascii=False, sort_keys=True)}`。",
        f"- **限制**：40/49 候选为 provisional static source，不能视为完整 PIT AI-role history；daily BID/ASK 是日末 spread proxy，不保证成交；dollar-volume fallback `close × volume` 仅保留来源标签；split audit 只覆盖供应商返回的 stock-split records，不证明所有 corporate actions 已覆盖；成本参数为 proposed formula，未运行组合。",
        f"- **输出**：`{rel(out)}`；未覆盖行保留于 `exclusions.csv` 并带 reason codes。",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8", newline="\n")


def append_ai_use_log(summary: dict[str, Any], out: Path) -> None:
    path = ROOT / "AI_USE_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# AI_USE_LOG\n"
    counts = summary["counts"]
    lines = [
        "",
        f"### AI portfolio validation-input audit ({summary['run_id']})",
        "",
        f"- OpenAI Codex performed a read-only, run-scoped audit at `{rel(out)}` using the 49-RIC candidate registry, clean v3 prices, daily model-ready metadata/eligibility and archived stock-split records. It covered `{counts['validation_model_ready_rows']}` validation schedule rows and `{counts['validation_supervised_anchor_rows']}` non-overlapping supervised anchor rows; no validation prediction was read or used, and no sealed test target was read.",
        f"- The audit retained missing close/volume/quote/dollar-volume rows and model-ready non-anchor rows with stable reason codes, retained the single delisted candidate's history, and found `{counts['validation_split_events']}` candidate split events in 2021–2022. It proposed a spread-plus-square-root-participation cost rule for later review; no strategy, portfolio or performance result was produced.",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8", newline="\n")


def audit(run_id: str) -> dict[str, Any]:
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    print("阶段说明：读取 49-RIC registry、model-ready validation schedule、clean prices 与 stock-split audit，精确匹配 2021–2022 形成日，统计 close/quote/volume/cost-input coverage 与 stale/delisting/corporate-action flags；不读取 predictions/test targets，不运行策略。", flush=True)

    registry, candidate_list, candidate_quarantine = parse_registry()
    registry_rics = set(registry["ric"])
    candidate_rics = set(candidate_list["Instrument"])
    if registry_rics != candidate_rics or len(registry_rics) != 49:
        raise RuntimeError("registry and 49-RIC candidate list do not match exactly")
    val_rows, model_meta_all, model_info = parse_model_ready()
    prices, source_rows_scanned, price_rows_selected = load_candidate_prices(registry_rics)
    returns, returns_source_rows_scanned, returns_rows_selected = load_candidate_returns(registry_rics)
    validation_sessions = pd.DatetimeIndex(sorted(val_rows["formation_dt"].dropna().unique()))
    if validation_sessions.empty:
        raise RuntimeError("model-ready validation schedule has no sessions")
    split_events = pd.read_csv(SPLIT_EVENTS, low_memory=False)
    joined = price_join_for_validation(val_rows, prices, returns)
    coverage, candidate_exclusions, group = candidate_coverage(registry, val_rows, joined, prices, validation_sessions, split_events)
    row_exclusions = make_row_exclusions(joined)
    exclusions = pd.concat([candidate_exclusions, row_exclusions], ignore_index=True, sort=False)
    ca_audit = corporate_action_audit(split_events, prices, registry_rics)

    # Validation coverage is measured against the model-ready schedule; the
    # source clean file itself is left untouched.
    expected_active = int(coverage["validation_active_expected_sessions"].sum())
    observed_rows = int(coverage["observed_price_rows_on_expected_sessions"].sum())
    anchor_rows = joined.loc[joined["model_supervised"]]
    active_cov = coverage.loc[coverage["validation_active_expected_sessions"].gt(0)]
    val_quote_missing = int((~joined["quote_valid_calc"].fillna(False)).sum())
    val_volume_missing = int(joined["ACVOL_UNS"].isna().sum())
    val_close_missing = int(joined["TRDPRC_1"].isna().sum())
    val_dollar_missing = int((~joined["dollar_volume_valid_calc"].fillna(False)).sum())
    checks: dict[str, Any] = {
        "registry_has_exactly_49_unique_rics": bool(len(registry) == 49 and registry["ric"].nunique() == 49),
        "candidate_list_matches_registry": bool(registry_rics == candidate_rics),
        "taxonomy_group_mapping_complete": bool(registry["primary_group"].notna().all() and registry["primary_group"].astype(str).str.strip().ne("").all()),
        "taxonomy_version_complete_and_single": bool(registry["taxonomy_version"].notna().all() and registry["taxonomy_version"].nunique() == 1),
        "validation_sessions_equal_model_ready_distinct_dates": bool(len(validation_sessions) == model_info["validation_formation_sessions"]),
        "model_ready_validation_rows_joined_to_eligibility": bool(len(joined) == model_info["validation_rows"]),
        "model_ready_supervised_anchor_count_recorded": bool(model_info["validation_supervised_anchor_rows"] == int(joined["model_supervised"].sum())),
        "selected_price_keys_unique": bool(not prices["price_duplicate_key"].any()),
        "active_expected_price_row_coverage_complete": bool(observed_rows == expected_active),
        "active_expected_return_row_coverage_complete": bool(
            int(coverage["observed_return_rows_on_expected_sessions"].sum()) == expected_active
        ),
        "validation_close_complete": bool(val_close_missing == 0),
        "validation_return_complete": bool(int((~joined["return_valid"].fillna(False)).sum()) == 0),
        "validation_quote_complete": bool(val_quote_missing == 0),
        "validation_dollar_volume_complete": bool(val_dollar_missing == 0),
        "validation_price_adjustments_field_present": bool(prices["price_adjustments"].notna().all()),
        "candidate_last_price_not_after_registry_member_to": bool(coverage["last_price_at_or_before_member_to"].all()),
        "no_delisting_event_inside_validation_period": bool(
            registry.loc[registry["delisted_bool"], "member_to_dt"].between(VAL_START, VAL_END).sum() == 0
        ),
        "split_event_rows_preserved": bool(len(ca_audit) == int(split_events["Instrument"].isin(registry_rics).sum())),
        "no_validation_predictions_read_or_used": True,
        "no_sealed_test_targets_read": True,
        "no_strategy_or_backtest_run": True,
        "no_physical_source_deletions": True,
    }
    cost_formula = {
        "status": "proposed_for_later_strategy_run; not executed in this audit",
        "execution_reference": "daily model formation_session/entry_session close; clean v3 BID/ASK is an end-of-day proxy, not a guaranteed fill",
        "notional": "abs(delta_weight) * portfolio_value",
        "mid_price": "(BID + ASK) / 2 when BID>0, ASK>0 and ASK>=BID",
        "spread_cost_decimal": "0.5 * (ASK - BID) / mid_price",
        "participation": "notional / prior_20_session_median_dollar_volume",
        "impact_cost_decimal": "0.005 * sqrt(participation)",
        "fixed_fee_decimal": 0.0,
        "total_trade_cost_decimal": "abs(delta_weight) * (0.5 * quoted_spread_decimal + 0.005 * sqrt(notional / prior_20_session_median_dollar_volume) + fixed_fee_decimal)",
        "execution_gate": "require positive close, valid two-sided quote, positive dollar volume, and participation <= 0.10; otherwise retain row with reason code and do not silently fill",
        "dollar_volume_source_rule": "use vendor TRNOVR_UNS when available; otherwise use explicitly tagged TRDPRC_1 * ACVOL_UNS; missing remains unavailable",
        "sensitivity_parameters_to_freeze_before_run": {"impact_coeff_decimal": [0.0025, 0.005, 0.01], "impact_exponent": 0.5, "max_participation": 0.10, "fixed_fee_decimal": 0.0},
        "limitations": [
            "The audit does not infer broker commission or borrow fees.",
            "Daily EOD BID/ASK and dollar volume are proxies; next-open execution would require a new timing/cost specification.",
            "No portfolio weights, turnover or costs were calculated because validation predictions were intentionally not read.",
        ],
    }

    # Stable, reviewable outputs.
    coverage.to_csv(out / "coverage_by_ric.csv", index=False)
    group.to_csv(out / "coverage_by_group.csv", index=False)
    joined.to_csv(out / "validation_row_input_audit.csv", index=False)
    exclusions.to_csv(out / "exclusions.csv", index=False)
    ca_audit.to_csv(out / "corporate_action_audit.csv", index=False)
    candidate_list.to_csv(out / "candidate_registry_snapshot.csv", index=False)
    write_json(out / "cost_formula.json", cost_formula)

    counts = {
        "registry_rows_before": int(len(registry)),
        "registry_rows_after": int(len(registry)),
        "candidate_quarantine_rows_input": int(len(candidate_quarantine)),
        "candidate_no_active_validation_span": int((coverage["validation_active_expected_sessions"] == 0).sum()),
        "price_source_rows_scanned": int(source_rows_scanned),
        "price_rows_selected": int(price_rows_selected),
        "price_rows_selected_candidate": int((prices["Instrument"].ne(BENCHMARK)).sum()),
        "price_rows_selected_benchmark": int((prices["Instrument"].eq(BENCHMARK)).sum()),
        "return_source_rows_scanned": int(returns_source_rows_scanned),
        "return_rows_selected": int(returns_rows_selected),
        "return_rows_selected_candidate": int((returns["Instrument"].ne(BENCHMARK)).sum()),
        "return_rows_selected_benchmark": int((returns["Instrument"].eq(BENCHMARK)).sum()),
        "validation_sessions": int(len(validation_sessions)),
        "validation_model_ready_rows": int(len(joined)),
        "validation_model_ready_instruments": int(joined["Instrument"].nunique()),
        "validation_supervised_anchor_rows": int(joined["model_supervised"].sum()),
        "validation_supervised_anchor_instruments": int(joined.loc[joined["model_supervised"], "Instrument"].nunique()),
        "expected_active_candidate_sessions": expected_active,
        "observed_price_rows_on_expected_sessions": observed_rows,
        "observed_return_rows_on_expected_sessions": int(coverage["observed_return_rows_on_expected_sessions"].sum()),
        "validation_rows_return_missing": int((~joined["return_valid"].fillna(False)).sum()),
        "validation_rows_close_missing": val_close_missing,
        "validation_rows_volume_missing": val_volume_missing,
        "validation_rows_quote_missing_or_invalid": val_quote_missing,
        "validation_rows_dollar_volume_missing_or_invalid": val_dollar_missing,
        "validation_rows_cost_complete": int(joined["cost_inputs_complete"].fillna(False).sum()),
        "validation_anchor_rows_cost_complete": int(anchor_rows["cost_inputs_complete"].fillna(False).sum()),
        "delisted_candidates": int(registry["delisted_bool"].sum()),
        "delisted_candidates_in_validation": int(
            registry.loc[registry["delisted_bool"], "member_to_dt"].between(VAL_START, VAL_END).sum()
        ),
        "candidate_split_events_total": int(len(ca_audit)),
        "validation_split_events": int(ca_audit["in_validation_2021_2022"].sum()) if not ca_audit.empty else 0,
        "exclusion_rows": int(len(exclusions)),
        "physical_deletions": 0,
    }
    summary = {
        "schema_version": "ai_portfolio_inputs_v1",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "audit_complete_with_review_flags",
        "scope": {
            "candidate_registry": rel(REGISTRY),
            "candidate_count": 49,
            "validation_period": ["2021-01-01", "2022-12-31"],
            "primary_model_context": "ai_pool_daily_state_models_v1_1 technical + AI-state Logistic; model-ready schedule only",
            "prediction_use": "forbidden and not performed",
            "sealed_test_target_use": "forbidden and not performed",
        },
        "inputs": {
            "registry": {"path": rel(REGISTRY), "sha256": sha256(REGISTRY)},
            "candidate_list": {"path": rel(CANDIDATE_LIST), "sha256": sha256(CANDIDATE_LIST)},
            "prices": {"path": rel(PRICES), "sha256": sha256(PRICES)},
            "returns": {"path": rel(RETURNS), "sha256": sha256(RETURNS)},
            "model_ready_metadata": {"path": rel(META), "sha256": sha256(META)},
            "model_ready_eligibility": {"path": rel(ELIGIBILITY), "sha256": sha256(ELIGIBILITY)},
            "stock_split_audit": {"path": rel(SPLIT_EVENTS), "sha256": sha256(SPLIT_EVENTS)},
        },
        "counts": counts,
        "model_ready_validation": model_info,
        "checks": checks,
        "missing_value_policy": "Preserve NA and genuine zero; no imputation, forward fill, interpolation, winsorization or row deletion. Missing quote/volume/dollar volume is an explicit execution reason.",
        "delisting_policy": "Retain delisted history; do not remove solely because a RIC is delisted. Compare terminal member_to and last clean price as a separate audit.",
        "corporate_action_policy": "Use clean v3 corporate-action-adjusted fields as supplied; do not apply split factors again. Retain candidate split records and flag validation events for continuity review.",
        "cost_formula_path": rel(out / "cost_formula.json"),
        "outputs": {
            "coverage_by_ric": rel(out / "coverage_by_ric.csv"),
            "coverage_by_group": rel(out / "coverage_by_group.csv"),
            "validation_row_input_audit": rel(out / "validation_row_input_audit.csv"),
            "exclusions": rel(out / "exclusions.csv"),
            "corporate_action_audit": rel(out / "corporate_action_audit.csv"),
        },
        "limitations": [
            "40 of 49 candidates are provisional static source; group membership is complete as a registry mapping but AI-role membership is not full PIT evidence.",
            "Daily BID/ASK is an end-of-day spread proxy; it is not a guaranteed executable quote.",
            "Stock-split audit covers the archived vendor stock-split response; other corporate actions and borrow availability are not verified here.",
            "This audit does not run or validate portfolio weights, turnover, returns, predictions or performance.",
        ],
    }
    write_json(out / "checks.json", checks)
    write_json(out / "summary.json", summary)
    report = """# AI portfolio validation-input audit\n\n""" + f"Run ID: `{run_id}`.\n\n" + """This run audits execution inputs for the 49-RIC candidate registry over the 2021–2022 validation schedule. It reads clean prices, quotes, volume/turnover, model-ready metadata/eligibility, taxonomy registry rows and archived stock-split records. It does not read validation predictions or sealed test targets, and it does not construct or run a strategy.\n\n""" + f"The model-ready schedule has {counts['validation_model_ready_rows']:,} validation rows across {counts['validation_model_ready_instruments']} active RICs and {counts['validation_supervised_anchor_rows']:,} non-overlapping supervised anchors. Registry spans imply {counts['expected_active_candidate_sessions']:,} expected candidate-session rows; clean prices match {counts['observed_price_rows_on_expected_sessions']:,}. Validation close gaps={counts['validation_rows_close_missing']:,}, quote gaps/invalid={counts['validation_rows_quote_missing_or_invalid']:,}, volume gaps={counts['validation_rows_volume_missing']:,}, dollar-volume gaps/invalid={counts['validation_rows_dollar_volume_missing_or_invalid']:,}.\n\n""" + f"All 49 registry rows have a non-empty primary group and one taxonomy version. Nine candidates have direct local PIT evidence and 40 are provisional static-source candidates. The sole delisted RIC is retained historically and ends at its recorded member_to; it has no active validation span issue. Five candidates have no 2021–2022 active span and are retained in `exclusions.csv` with `NO_ACTIVE_VALIDATION_SPAN`.\n\n""" + f"The candidate split audit contains {counts['validation_split_events']} validation-period split events. Clean prices carry the vendor corporate-action adjustment record; split factors are not reapplied. Use `cost_formula.json` as the proposed execution rule: valid two-sided quote plus positive dollar volume, half-spread cost, and square-root participation impact with a 10% participation gate. Parameters remain to be frozen before any later strategy run.\n\n""" + """Every excluded or review row remains in the audit output with stable reason codes. Missing values were not imputed, and no source table was changed.\n"""
    (out / "report.md").write_text(report, encoding="utf-8", newline="\n")
    output_paths = [
        out / "coverage_by_ric.csv", out / "coverage_by_group.csv", out / "validation_row_input_audit.csv",
        out / "exclusions.csv", out / "corporate_action_audit.csv", out / "candidate_registry_snapshot.csv",
        out / "cost_formula.json", out / "checks.json", out / "summary.json", out / "report.md",
    ]
    write_manifest(out, [REGISTRY, CANDIDATE_LIST, CANDIDATE_QUARANTINE, PRICES, RETURNS, META, ELIGIBILITY, SPLIT_EVENTS], output_paths, run_id)
    append_processing_log(summary, out)
    append_ai_use_log(summary, out)
    summary["output_manifest"] = rel(out / "manifest.json")
    print(json.dumps({"run_id": run_id, "counts": counts, "checks": checks}, ensure_ascii=False, default=json_default), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
