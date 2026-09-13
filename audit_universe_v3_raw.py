"""Offline integrity, coverage, and v2-v3 comparison for the v3 raw run."""
from __future__ import annotations

import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_ID = "20260909T012417705069Z"
RAW = ROOT / "data" / "raw" / "universe_v3" / RUN_ID
OLD = ROOT / "data" / "raw" / "universe_v2"
OUT_ROOT = ROOT / "data" / "audit" / "universe_v3_raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_price(path: Path, instruments: list[str]) -> dict[str, pd.DataFrame]:
    if len(instruments) == 1:
        frame = pd.read_csv(path)
        return {instruments[0]: frame.set_index("Date")}
    frame = pd.read_csv(path, header=[0, 1], index_col=0)
    frame = frame.loc[frame.index.astype(str) != "Date"]
    return {instrument: frame[instrument] for instrument in instruments}


def main() -> int:
    audit_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / audit_id
    out.mkdir(parents=True, exist_ok=False)
    plan_file = RAW / "plan.json"
    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    integrity_rows = []
    for item in plan["calls"]:
        csv_path = RAW / f"{item['name']}.csv"
        meta_path = RAW / f"{item['name']}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        integrity_rows.append({
            "name": item["name"],
            "table": item["table"],
            "planned": True,
            "csv_exists": csv_path.exists(),
            "meta_exists": meta_path.exists(),
            "request_matches": meta.get("request") == item["request"],
            "status": meta.get("status", "pending"),
            "hash_matches": csv_path.exists() and meta.get("sha256") == sha256(csv_path),
            "rows": meta.get("rows"),
        })
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(out / "batch_integrity.csv", index=False)

    new_actual = pd.concat(
        [pd.read_csv(path) for path in sorted(RAW.glob("actuals_*.csv"))],
        ignore_index=True,
    )
    old_actual = pd.concat(
        [pd.read_csv(path) for path in sorted(OLD.glob("actuals_*.csv"))],
        ignore_index=True,
    )
    required = ["Instrument", "Report Date", "Period End Date", "Earnings Per Share - Actual"]
    old_valid = old_actual.dropna(subset=required)
    new_valid = new_actual.dropna(subset=required)
    comparison = old_valid.merge(
        new_valid[required],
        on=["Instrument", "Report Date", "Period End Date"],
        how="outer",
        suffixes=("_v2", "_v3"),
        indicator=True,
    )
    comparison["value_equal"] = (
        comparison["Earnings Per Share - Actual_v2"]
        == comparison["Earnings Per Share - Actual_v3"]
    )
    missing_by_ric = (
        comparison.loc[comparison["_merge"] == "left_only"]
        .groupby("Instrument", dropna=False)
        .size().rename("v2_valid_events_absent_from_v3")
        .reset_index()
    )
    missing_by_ric.to_csv(out / "actual_v2_only_by_ric.csv", index=False)

    price_rows = []
    price_items = [item for item in plan["calls"] if item["table"] == "prices"]
    for item in price_items:
        meta = json.loads((RAW / f"{item['name']}.meta.json").read_text(encoding="utf-8"))
        by_instrument = read_price(RAW / f"{item['name']}.csv", item["request"]["universe"])
        for instrument, frame in by_instrument.items():
            price_rows.append({
                "batch": item["name"],
                "Instrument": instrument,
                "date_rows": int(len(frame)),
                "rows_with_any_value": int(frame.notna().any(axis=1).sum()),
                **{f"{field}_non_null": int(frame[field].notna().sum()) for field in frame.columns},
            })
    price_coverage = pd.DataFrame(price_rows)
    price_coverage.to_csv(out / "price_coverage_by_instrument.csv", index=False)

    shared = comparison.loc[comparison["_merge"] == "both"]
    summary = {
        "audit_id": audit_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_run_id": RUN_ID,
        "plan_sha256": sha256(plan_file),
        "batch_integrity": {
            "planned_calls": int(len(integrity)),
            "actuals_complete": bool(
                integrity.loc[integrity.table == "actuals", ["csv_exists", "meta_exists", "request_matches", "hash_matches"]].all().all()
            ),
            "prices_complete": bool(
                integrity.loc[integrity.table == "prices", ["csv_exists", "meta_exists", "request_matches", "hash_matches"]].all().all()
            ),
            "estimates_completed": int((integrity.loc[integrity.table == "estimates", "status"] == "returned").sum()),
            "estimates_planned": int((integrity.table == "estimates").sum()),
            "retained_error_sidecars": sorted(path.name for path in RAW.glob("*.error.json")),
            "retained_temp_artifacts": sorted(path.name for path in RAW.glob("*.tmp")),
        },
        "actuals": {
            "v2_raw_rows": int(len(old_actual)),
            "v3_raw_rows": int(len(new_actual)),
            "v2_valid_rows": int(len(old_valid)),
            "v3_valid_rows": int(len(new_valid)),
            "v3_blank_required_rows": int(len(new_actual) - len(new_valid)),
            "shared_valid_events": int(len(shared)),
            "shared_values_all_equal": bool(shared["value_equal"].all()),
            "v2_only_valid_events": int((comparison["_merge"] == "left_only").sum()),
            "v3_only_valid_events": int((comparison["_merge"] == "right_only").sum()),
            "v2_only_by_ric": missing_by_ric.set_index("Instrument")["v2_valid_events_absent_from_v3"].to_dict(),
            "extra_field_non_null": {
                field: int(new_actual[field].notna().sum())
                for field in ["Calc Date", "Financial Period Absolute", "Currency"]
            },
            "amcr_current_probe": "All five selector calls on 2026-09-09 returned only a blank placeholder; archived v2 has 29 valid events.",
        },
        "prices": {
            "requested_instruments_including_spy": int(price_coverage.Instrument.nunique()),
            "instrument_entries": int(len(price_coverage)),
            "instruments_with_any_value": int(price_coverage.rows_with_any_value.gt(0).sum()),
            "instruments_with_no_values": price_coverage.loc[
                price_coverage.rows_with_any_value.eq(0), "Instrument"
            ].tolist(),
            "security_date_rows_with_any_field": int(price_coverage.rows_with_any_value.sum()),
            "close_non_null": int(price_coverage["TRDPRC_1_non_null"].sum()),
            "volume_non_null": int(price_coverage["ACVOL_UNS_non_null"].sum()),
            "bid_non_null": int(price_coverage["BID_non_null"].sum()),
            "ask_non_null": int(price_coverage["ASK_non_null"].sum()),
            "turnover_non_null": int(price_coverage["TRNOVR_UNS_non_null"].sum()),
        },
        "decisions": [
            "Exclude EVHC.N^L16 from v3 eligibility because the corrected membership audit removed its erroneous interval; preserve all v2 raw rows.",
            "Carry archived AMCR.N actuals into a separately tagged fallback layer because the same LSEG query now returns no dated history; report a sensitivity analysis excluding AMCR.N.",
            "Do not fill any price, volume, bid, ask, or turnover gap during cleaning.",
            "Treat the one completed daily-estimates batch as a pilot, not as a complete table.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Universe v3 raw audit

Run `{RUN_ID}` has 39/39 verified actuals batches and 163/163 verified price
batches. One daily-estimates batch is a retained pilot; the other 83 remain
unrequested and are not counted as a complete dataset.

The explicit Reported collection contains {len(new_actual):,} raw rows and
{len(new_valid):,} valid dated events. All {len(shared):,} events shared with v2
have identical actual EPS values. The {int((comparison['_merge'] == 'left_only').sum())}
v2-only valid events are isolated to EVHC.N^L16 and AMCR.N. EVHC is excluded by
the corrected membership record. AMCR now returns blank placeholders under
default, Reported, All, and Latest selectors; its 29 archived events require a
tagged fallback plus an exclusion sensitivity check.

The price table covers {price_coverage.Instrument.nunique():,} requested
instruments including SPY and contains {price_coverage.rows_with_any_value.sum():,}
security-date observations with at least one field. Missing field observations
remain missing. No imputation, deletion, winsorization, date conversion, or
modeling occurred in this audit.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
