"""Build versioned v3 clean tables with explicit row-accounting decisions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_RUN_ID = "20260909T012417705069Z"
RAW = ROOT / "data" / "raw" / "universe_v3" / RAW_RUN_ID
OLD_RAW = ROOT / "data" / "raw" / "universe_v2"
OLD_CLEAN = ROOT / "data" / "clean" / "v2"
OLD_AUDIT = ROOT / "data" / "audit" / "v2"
SPANS = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
INTERVALS = ROOT / "data" / "audit" / "universe_rebuild" / "membership_intervals.csv"
CLEAN = ROOT / "data" / "clean" / "v3" / RAW_RUN_ID
AUDIT_ROOT = ROOT / "data" / "audit" / "clean_v3"
PRICE_FIELDS = [
    "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS",
    "BID", "ASK", "TRNOVR_UNS",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_actuals() -> tuple[pd.DataFrame, dict]:
    parts = []
    inputs = []
    for path in sorted(RAW.glob("actuals_*.csv")):
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if meta["sha256"] != sha256(path):
            raise RuntimeError(f"Raw actual hash mismatch: {path.name}")
        frame = pd.read_csv(path, keep_default_na=True)
        frame["raw_file"] = str(path.relative_to(ROOT))
        frame["raw_row"] = np.arange(1, len(frame) + 1)
        frame["actual_source"] = "v3_explicit_reported"
        frame["actual_selector"] = "Reported"
        parts.append(frame)
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": meta["sha256"], "rows": len(frame)})

    archived_parts = []
    for path in sorted(OLD_RAW.glob("actuals_*.csv")):
        frame = pd.read_csv(path, keep_default_na=True)
        selected = frame.loc[frame["Instrument"].eq("AMCR.N")].copy()
        if selected.empty:
            continue
        selected["Calc Date"] = pd.NA
        selected["Financial Period Absolute"] = pd.NA
        selected["Currency"] = pd.NA
        selected["raw_file"] = str(path.relative_to(ROOT))
        selected["raw_row"] = selected.index.to_numpy() + 1
        selected["actual_source"] = "archived_v2_current_vendor_unavailable"
        selected["actual_selector"] = "default_archived_not_reclassified"
        archived_parts.append(selected)
        inputs.append({
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "selected_rows": int(len(selected)), "selection": "Instrument == AMCR.N",
        })
    if not archived_parts:
        raise RuntimeError("Archived AMCR fallback rows not found")
    parts.extend(archived_parts)
    raw = pd.concat(parts, ignore_index=True, sort=False)

    raw["announcement"] = pd.to_datetime(raw["Report Date"], format="mixed", errors="coerce")
    raw["period_end"] = pd.to_datetime(raw["Period End Date"], format="mixed", errors="coerce")
    raw["calc_date"] = pd.to_datetime(raw["Calc Date"], format="mixed", errors="coerce")
    raw["eps_actual"] = pd.to_numeric(raw["Earnings Per Share - Actual"], errors="coerce")
    required = ["Instrument", "announcement", "period_end", "eps_actual"]
    reason = pd.Series("", index=raw.index, dtype="object")
    reason.loc[raw[required].isna().any(axis=1)] = "missing_or_unparseable_required_field"

    eligible = reason.eq("")
    keys = ["Instrument", "announcement", "period_end"]
    duplicate = raw.loc[eligible].duplicated(keys, keep=False)
    if duplicate.any():
        duplicate_indices = raw.loc[eligible].index[duplicate]
        reason.loc[duplicate_indices] = "duplicate_event_key_all_quarantined"

    quarantine = raw.loc[reason.ne("")].copy()
    quarantine["rejection_reason"] = reason.loc[reason.ne("")]
    clean = raw.loc[reason.eq("")].copy()
    clean["announcement_day"] = clean["announcement"].dt.normalize()
    clean["formal_entry_rule"] = "first_market_session_strictly_after_announcement_day"
    clean["timezone_status"] = "vendor_clock_naive; intraday label not used for formal entry"
    output_columns = [
        "Instrument", "announcement", "announcement_day", "period_end", "calc_date",
        "eps_actual", "Financial Period Absolute", "Currency", "actual_source",
        "actual_selector", "formal_entry_rule", "timezone_status", "raw_file", "raw_row",
    ]
    clean = clean[output_columns].sort_values(keys).reset_index(drop=True)
    return clean, {
        "input": raw, "quarantine": quarantine, "inputs": inputs,
        "stats": {
            "input_rows": int(len(raw)), "clean_rows": int(len(clean)),
            "quarantine_rows": int(len(quarantine)),
            "quarantine_reasons": quarantine["rejection_reason"].value_counts().to_dict(),
            "clean_instruments": int(clean.Instrument.nunique()),
            "archived_amcr_clean_rows": int(clean.actual_source.str.startswith("archived").sum()),
        },
    }


def load_price_file(path: Path, instruments: list[str]) -> list[pd.DataFrame]:
    frames = []
    if len(instruments) == 1:
        wide = pd.read_csv(path, keep_default_na=True)
        frame = wide.rename(columns={"Date": "raw_date"})
        frame["Instrument"] = instruments[0]
        frame["raw_wide_row"] = np.arange(1, len(frame) + 1)
        return [frame]
    wide = pd.read_csv(path, header=[0, 1], index_col=0, keep_default_na=True)
    wide = wide.loc[wide.index.astype(str) != "Date"]
    for instrument in instruments:
        frame = wide[instrument].copy().reset_index().rename(columns={wide.index.name or "index": "raw_date"})
        if "raw_date" not in frame:
            frame = frame.rename(columns={frame.columns[0]: "raw_date"})
        frame["Instrument"] = instrument
        frame["raw_wide_row"] = np.arange(1, len(frame) + 1)
        frames.append(frame)
    return frames


def membership_flags(frame: pd.DataFrame) -> pd.Series:
    intervals = pd.read_csv(INTERVALS)
    intervals["start"] = pd.to_datetime(intervals["start"], errors="raise")
    intervals["end"] = pd.to_datetime(intervals["end"], errors="raise")
    by_ric = {ric: list(zip(group.start, group.end)) for ric, group in intervals.groupby("ric")}
    result = pd.Series(False, index=frame.index)
    for ric, indices in frame.groupby("Instrument").groups.items():
        dates = frame.loc[indices, "Date"]
        inside = pd.Series(False, index=indices)
        for start, end in by_ric.get(ric, []):
            inside |= dates.between(start, end)
        result.loc[indices] = inside
    return result


def load_prices(plan: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parts = []
    inputs = []
    price_calls = [item for item in plan["calls"] if item["table"] == "prices"]
    for item in price_calls:
        path = RAW / f"{item['name']}.csv"
        meta = json.loads((RAW / f"{item['name']}.meta.json").read_text(encoding="utf-8"))
        if meta["sha256"] != sha256(path) or meta["request"] != item["request"]:
            raise RuntimeError(f"Raw price integrity mismatch: {item['name']}")
        file_frames = load_price_file(path, item["request"]["universe"])
        for frame in file_frames:
            frame["raw_file"] = str(path.relative_to(ROOT))
            parts.append(frame)
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": meta["sha256"], "wide_rows": meta["rows"]})
    raw = pd.concat(parts, ignore_index=True, sort=False)
    raw["Date"] = pd.to_datetime(raw["raw_date"], format="mixed", errors="coerce")
    reason = pd.Series("", index=raw.index, dtype="object")
    reason.loc[raw["Date"].isna()] = "missing_or_unparseable_date"
    parse_failure = pd.Series(False, index=raw.index)
    for field in PRICE_FIELDS:
        original_present = raw[field].notna() & raw[field].astype(str).str.strip().ne("")
        parsed = pd.to_numeric(raw[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        parse_failure |= original_present & parsed.isna()
        raw[field] = parsed
    reason.loc[reason.eq("") & parse_failure] = "unparseable_numeric_value"
    all_missing = raw[PRICE_FIELDS].isna().all(axis=1)
    reason.loc[reason.eq("") & all_missing] = "structural_wide_padding_all_fields_missing"

    eligible = reason.eq("")
    duplicate = raw.loc[eligible].duplicated(["Instrument", "Date"], keep=False)
    if duplicate.any():
        reason.loc[raw.loc[eligible].index[duplicate]] = "duplicate_instrument_date_all_quarantined"
    quarantine = raw.loc[reason.ne("")].copy()
    quarantine["rejection_reason"] = reason.loc[reason.ne("")]
    clean = raw.loc[reason.eq("")].copy()
    clean["has_close"] = clean["TRDPRC_1"].notna()
    clean["has_volume"] = clean["ACVOL_UNS"].notna()
    clean["has_two_sided_quote"] = clean[["BID", "ASK"]].notna().all(axis=1)
    clean["quoted_spread_bps"] = np.where(
        clean["has_two_sided_quote"] & (clean["BID"] + clean["ASK"]).ne(0),
        (clean["ASK"] - clean["BID"]) / ((clean["ASK"] + clean["BID"]) / 2) * 10_000,
        np.nan,
    )
    clean["dollar_volume"] = clean["TRNOVR_UNS"]
    fallback_turnover = clean["dollar_volume"].isna() & clean[["TRDPRC_1", "ACVOL_UNS"]].notna().all(axis=1)
    clean.loc[fallback_turnover, "dollar_volume"] = (
        clean.loc[fallback_turnover, "TRDPRC_1"] * clean.loc[fallback_turnover, "ACVOL_UNS"]
    )
    clean["dollar_volume_source"] = np.select(
        [clean["TRNOVR_UNS"].notna(), fallback_turnover],
        ["TRNOVR_UNS", "TRDPRC_1_times_ACVOL_UNS"],
        default="missing",
    )
    clean["in_sp500_that_day"] = membership_flags(clean)
    clean["price_adjustments"] = "exchangeCorrection,manualCorrection,CCH,CRE,RPO,RTS"
    clean = clean.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    return clean, quarantine, {
        "inputs": inputs,
        "input_instrument_date_cells": int(len(raw)),
        "clean_rows": int(len(clean)),
        "quarantine_rows": int(len(quarantine)),
        "quarantine_reasons": quarantine["rejection_reason"].value_counts().to_dict(),
        "instruments": int(clean.Instrument.nunique()),
        "close_missing_clean_rows": int(clean["TRDPRC_1"].isna().sum()),
        "volume_missing_clean_rows": int(clean["ACVOL_UNS"].isna().sum()),
        "two_sided_quote_missing_clean_rows": int((~clean["has_two_sided_quote"]).sum()),
        "turnover_vendor_rows": int(clean["TRNOVR_UNS"].notna().sum()),
        "turnover_price_times_volume_fallback_rows": int(fallback_turnover.sum()),
        "dollar_volume_missing_rows": int(clean["dollar_volume"].isna().sum()),
    }


def load_weekly_estimates(corrected_rics: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = OLD_CLEAN / "estimates.csv"
    inherited_quarantine = OLD_AUDIT / "estimates_quarantine.csv"
    estimates = pd.read_csv(source)
    excluded = estimates.loc[~estimates["Instrument"].isin(corrected_rics)].copy()
    excluded["exclusion_reason"] = "outside_corrected_v3_universe"
    clean = estimates.loc[estimates["Instrument"].isin(corrected_rics)].copy()
    clean["estimate_frequency"] = "weekly"
    clean["source_clean_version"] = "v2_verified_clean"
    return clean, excluded, {
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "inherited_raw_rows": 463871,
        "inherited_clean_rows": int(len(estimates)),
        "inherited_quarantine_path": str(inherited_quarantine.relative_to(ROOT)),
        "inherited_quarantine_sha256": sha256(inherited_quarantine),
        "inherited_quarantine_rows": int(len(pd.read_csv(inherited_quarantine))),
        "v3_clean_rows": int(len(clean)),
        "v3_universe_exclusion_rows": int(len(excluded)),
        "v3_universe_exclusion_rics": sorted(excluded.Instrument.unique().tolist()),
        "frequency": "weekly",
        "point_in_time_rule": "latest snapshot strictly before announcement day with exact period-end match",
        "missing_policy": "Inherited v2 quarantine; no new fill or interpolation.",
    }


def main() -> int:
    if CLEAN.exists():
        raise RuntimeError(f"Refusing to overwrite existing clean run: {CLEAN}")
    CLEAN.mkdir(parents=True, exist_ok=False)
    audit_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit = AUDIT_ROOT / audit_id
    audit.mkdir(parents=True, exist_ok=False)
    plan = json.loads((RAW / "plan.json").read_text(encoding="utf-8"))
    corrected_rics = set(pd.read_csv(SPANS)["ric"])

    actuals, actual_details = load_actuals()
    actuals.to_csv(CLEAN / "actuals.csv", index=False)
    actual_details["quarantine"].to_csv(CLEAN / "actuals_quarantine.csv", index=False)

    estimates, estimate_exclusions, estimate_stats = load_weekly_estimates(corrected_rics)
    estimates.to_csv(CLEAN / "estimates_weekly.csv", index=False)
    estimate_exclusions.to_csv(CLEAN / "estimates_universe_exclusions.csv", index=False)

    prices, price_quarantine, price_stats = load_prices(plan)
    prices.to_csv(CLEAN / "prices.csv", index=False)
    price_quarantine.to_csv(CLEAN / "prices_quarantine.csv", index=False)

    pilot_path = RAW / "estimates_live_000.csv"
    pilot_meta = json.loads((RAW / "estimates_live_000.meta.json").read_text(encoding="utf-8"))
    if sha256(pilot_path) != pilot_meta["sha256"]:
        raise RuntimeError("Daily estimate pilot hash mismatch")
    write_json(CLEAN / "daily_estimates_pilot_profile.json", {
        "status": "pilot_only_not_core_input", "raw_path": str(pilot_path.relative_to(ROOT)),
        "raw_sha256": pilot_meta["sha256"], "raw_rows": pilot_meta["rows"],
        "completed_batches": 1, "planned_batches": 84,
    })

    outputs = {}
    for path in sorted(CLEAN.iterdir()):
        outputs[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    summary = {
        "audit_id": audit_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_run_id": RAW_RUN_ID,
        "clean_path": str(CLEAN.relative_to(ROOT)),
        "actuals": actual_details["stats"],
        "estimates": estimate_stats,
        "prices": price_stats,
        "outputs": outputs,
        "explicit_processing_decisions": [
            "Current explicit-Reported actual rows with missing required fields are quarantined, never filled.",
            "Archived AMCR.N v2 rows are appended with a visible source/selector tag because current LSEG calls return no dated history.",
            "EVHC.N^L16 is outside the corrected universe; its old raw data remains untouched.",
            "Weekly estimates inherit the verified v2 clean/quarantine split; 161 EVHC rows move to a visible v3 universe-exclusion file.",
            "All-null instrument/date cells created by wide price alignment move to price quarantine as structural padding.",
            "Missing price fields on otherwise valid rows remain NA; none are forward-filled, back-filled, or set to zero.",
            "Dollar volume uses vendor TRNOVR_UNS when present; otherwise price times volume is computed and explicitly tagged.",
            "Daily estimates remain a one-batch pilot and are not mixed with the weekly core table.",
        ],
        "status": "clean_data_ready_for_v3_panel_build; daily-consensus sensitivity remains partial",
    }
    write_json(CLEAN / "summary.json", summary)
    summary["outputs"]["summary.json"] = {"sha256": sha256(CLEAN / "summary.json"), "bytes": (CLEAN / "summary.json").stat().st_size}
    write_json(audit / "summary.json", summary)
    report = f"""# v3 clean-data report

The clean run contains {len(actuals):,} valid actual events,
{len(estimates):,} weekly point-in-time estimate rows, and {len(prices):,}
security-date price/liquidity rows.

Actual input consists of the current explicit-Reported collection plus the
archived AMCR rows, visibly tagged because current vendor calls return no dated
AMCR history. {len(actual_details['quarantine']):,} actual rows are quarantined;
none were filled. The 161 clean v2 estimate rows for EVHC are preserved in a
separate universe-exclusion file because corrected membership removed that RIC.

The wide price files create {len(price_quarantine):,} all-null structural cells;
they are preserved in quarantine rather than silently discarded. Valid rows
keep every field-level missing value. Dollar volume substitutions are tagged by
source and use close times volume only when vendor turnover is absent.

The 32,290-row daily estimate batch remains a pilot and is not a core input.
No feature panel, label, model, portfolio, or backtest was built in this step.
"""
    (audit / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
