"""Independent offline validation of v3 clean tables and event readiness."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_ID = "20260909T012417705069Z"
CLEAN = ROOT / "data" / "clean" / "v3" / RUN_ID
AUDIT_SOURCE = ROOT / "data" / "audit" / "clean_v3" / "20260909T015455578916Z" / "summary.json"
OUT_ROOT = ROOT / "data" / "audit" / "validate_clean_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, condition: bool, detail: object = None) -> dict:
    return {"name": name, "passed": bool(condition), "detail": detail}


def match_estimates(actuals: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = {
        key: group.sort_values("snapshot")
        for key, group in estimates.groupby(["Instrument", "period_end"], sort=False)
    }
    for row in actuals.itertuples(index=False):
        group = grouped.get((row.Instrument, row.period_end))
        selected = None
        if group is not None:
            eligible = group.loc[group["snapshot"] < row.announcement_day]
            if not eligible.empty:
                selected = eligible.iloc[-1]
        if selected is None:
            rows.append({
                "Instrument": row.Instrument, "announcement": row.announcement,
                "period_end": row.period_end, "snapshot": pd.NaT,
                "snapshot_age_days": np.nan, "match_status": "no_strict_pre_event_exact_period_snapshot",
                "mean": np.nan, "stddev": np.nan, "analysts": np.nan,
            })
            continue
        age = (row.announcement_day - selected["snapshot"]).days
        rows.append({
            "Instrument": row.Instrument, "announcement": row.announcement,
            "period_end": row.period_end, "snapshot": selected["snapshot"],
            "snapshot_age_days": age,
            "match_status": "matched_within_14_days" if age <= 14 else "snapshot_older_than_14_days",
            "mean": selected["mean"], "stddev": selected["stddev"], "analysts": selected["analysts"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    validation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / validation_id
    out.mkdir(parents=True, exist_ok=False)
    source_summary = json.loads(AUDIT_SOURCE.read_text(encoding="utf-8"))
    checks = []
    for filename, expected in source_summary["outputs"].items():
        path = CLEAN / filename
        checks.append(check(f"output_hash_{filename}", path.exists() and sha256(path) == expected["sha256"]))

    actuals = pd.read_csv(
        CLEAN / "actuals.csv", parse_dates=["announcement", "announcement_day", "period_end", "calc_date"]
    )
    actual_q = pd.read_csv(CLEAN / "actuals_quarantine.csv")
    estimates = pd.read_csv(
        CLEAN / "estimates_weekly.csv",
        usecols=[
            "Instrument", "snapshot", "Period End Date", "Earnings Per Share - Mean",
            "Earnings Per Share - Standard Deviation", "Earnings Per Share - Number of Included Estimates",
        ],
        parse_dates=["snapshot", "Period End Date"],
    ).rename(columns={
        "Period End Date": "period_end", "Earnings Per Share - Mean": "mean",
        "Earnings Per Share - Standard Deviation": "stddev",
        "Earnings Per Share - Number of Included Estimates": "analysts",
    })
    estimate_exclusions = pd.read_csv(CLEAN / "estimates_universe_exclusions.csv")
    price_columns = [
        "Instrument", "Date", "TRDPRC_1", "ACVOL_UNS", "BID", "ASK", "TRNOVR_UNS",
        "dollar_volume", "dollar_volume_source", "has_close", "has_volume",
        "has_two_sided_quote", "in_sp500_that_day", "price_adjustments",
    ]
    prices = pd.read_csv(CLEAN / "prices.csv", usecols=price_columns, parse_dates=["Date"])
    price_q = pd.read_csv(
        CLEAN / "prices_quarantine.csv",
        usecols=["Instrument", "raw_date", *[field for field in ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS", "BID", "ASK", "TRNOVR_UNS"]], "rejection_reason"],
    )

    checks.extend([
        check("actual_row_accounting", len(actuals) + len(actual_q) == source_summary["actuals"]["input_rows"]),
        check("actual_unique_event_key", not actuals.duplicated(["Instrument", "announcement", "period_end"]).any()),
        check("actual_required_complete", actuals[["Instrument", "announcement", "period_end", "eps_actual"]].notna().all().all()),
        check("actual_amcr_fallback_29", int(actuals.actual_source.str.startswith("archived").sum()) == 29),
        check("actual_evch_absent", not actuals.Instrument.eq("EVHC.N^L16").any()),
        check("actual_formal_entry_rule_frozen", actuals.formal_entry_rule.eq("first_market_session_strictly_after_announcement_day").all()),
        check("estimate_rows", len(estimates) == source_summary["estimates"]["v3_clean_rows"]),
        check("estimate_unique_key", not estimates.duplicated(["Instrument", "snapshot", "period_end"]).any()),
        check("estimate_evch_excluded", not estimates.Instrument.eq("EVHC.N^L16").any()),
        check("estimate_exclusion_exact", len(estimate_exclusions) == 161 and set(estimate_exclusions.Instrument) == {"EVHC.N^L16"}),
        check("price_row_accounting", len(prices) + len(price_q) == source_summary["prices"]["input_instrument_date_cells"]),
        check("price_unique_key", not prices.duplicated(["Instrument", "Date"]).any()),
        check("price_dates_valid", prices.Date.notna().all()),
        check("price_quarantine_only_structural_padding", set(price_q.rejection_reason) == {"structural_wide_padding_all_fields_missing"}),
        check("price_quarantine_fields_all_missing", price_q[["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS", "BID", "ASK", "TRNOVR_UNS"]].isna().all().all()),
        check("price_adjustments_explicit", prices.price_adjustments.eq("exchangeCorrection,manualCorrection,CCH,CRE,RPO,RTS").all()),
    ])

    vendor_turnover = prices.loc[prices.dollar_volume_source.eq("TRNOVR_UNS")]
    fallback = prices.loc[prices.dollar_volume_source.eq("TRDPRC_1_times_ACVOL_UNS")]
    checks.extend([
        check("vendor_turnover_copied", np.allclose(vendor_turnover.dollar_volume, vendor_turnover.TRNOVR_UNS, equal_nan=False)),
        check("fallback_turnover_recomputed", np.allclose(fallback.dollar_volume, fallback.TRDPRC_1 * fallback.ACVOL_UNS, rtol=1e-12, atol=1e-6)),
        check("missing_flags_match", (prices.has_close == prices.TRDPRC_1.notna()).all() and (prices.has_volume == prices.ACVOL_UNS.notna()).all()),
    ])

    aapl = prices.loc[
        prices.Instrument.eq("AAPL.OQ") & prices.Date.between("2020-08-27", "2020-09-01"),
        ["Date", "TRDPRC_1"],
    ].sort_values("Date")
    aapl_change = aapl.TRDPRC_1.pct_change(fill_method=None).abs().max()
    checks.append(check("aapl_split_adjusted_close_no_75pct_break", aapl_change < 0.2, float(aapl_change)))

    study_actuals = actuals.loc[actuals.announcement_day.between("2015-01-01", "2026-06-30")].copy()
    matched = match_estimates(study_actuals, estimates)
    matched.to_csv(out / "actual_estimate_match_status.csv", index=False)
    match_counts = matched.match_status.value_counts().to_dict()
    checks.extend([
        check("matched_snapshots_strictly_pre_event", (matched.loc[matched.snapshot.notna(), "snapshot"] < pd.to_datetime(matched.loc[matched.snapshot.notna(), "announcement"]).dt.normalize()).all()),
        check("matched_period_is_exact_by_construction", True, "Grouping key is Instrument + period_end"),
    ])

    spy_sessions = np.sort(prices.loc[prices.Instrument.eq("SPY.P") & prices.TRDPRC_1.notna(), "Date"].unique())
    source_close = prices.loc[prices.TRDPRC_1.notna(), ["Instrument", "Date"]].drop_duplicates()
    close_keys = set(zip(source_close.Instrument, source_close.Date))
    readiness_rows = []
    for row in study_actuals.itertuples(index=False):
        position = np.searchsorted(spy_sessions, np.datetime64(row.announcement_day), side="right")
        entry = pd.Timestamp(spy_sessions[position]) if position < len(spy_sessions) else pd.NaT
        readiness_rows.append({
            "Instrument": row.Instrument, "announcement": row.announcement,
            "entry_date": entry, "spy_session_available": pd.notna(entry),
            "source_close_available": pd.notna(entry) and (row.Instrument, entry) in close_keys,
        })
    readiness = pd.DataFrame(readiness_rows)
    readiness.loc[~readiness.source_close_available].to_csv(out / "source_entry_close_gaps.csv", index=False)
    checks.append(check("all_study_events_have_spy_next_session", readiness.spy_session_available.all()))

    crossed_quotes = int(((prices.ASK < prices.BID) & prices[["ASK", "BID"]].notna().all(axis=1)).sum())
    validation = {
        "validation_id": validation_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_run_id": RUN_ID,
        "checks": checks,
        "passed": int(sum(item["passed"] for item in checks)),
        "failed": [item for item in checks if not item["passed"]],
        "counts": {
            "actuals": int(len(actuals)), "actuals_quarantine": int(len(actual_q)),
            "estimates_weekly": int(len(estimates)), "prices": int(len(prices)),
            "prices_quarantine": int(len(price_q)), "study_actuals": int(len(study_actuals)),
            "actual_estimate_match_status": {str(k): int(v) for k, v in match_counts.items()},
            "standardized_surprise_computable": int((matched["stddev"] > 0).sum()),
            "study_events_with_source_entry_close": int(readiness.source_close_available.sum()),
            "study_events_without_source_entry_close": int((~readiness.source_close_available).sum()),
            "crossed_daily_quotes_retained_for_review": crossed_quotes,
        },
        "non_actions": [
            "No row or field was filled, removed, winsorized, or changed during validation.",
            "Crossed quotes are counted for review and retained.",
            "No graph, feature panel, label, model, portfolio, or backtest was built.",
        ],
    }
    write_path = out / "validation.json"
    write_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, default=str))
    return 1 if validation["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
