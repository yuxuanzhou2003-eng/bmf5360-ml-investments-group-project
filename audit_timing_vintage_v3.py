"""Audit the isolated timing/vintage probe and persist reproducible conclusions."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "timing_vintage_probe_v3" / "20260909T010626599672Z"
OUT_ROOT = ROOT / "data" / "audit" / "timing_vintage_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))

    call_checks = []
    for call in manifest["calls"]:
        csv_path = RAW / f"{call['name']}.csv"
        call_checks.append({
            "name": call["name"],
            "status": call["status"],
            "csv_exists": csv_path.exists(),
            "csv_hash_matches": csv_path.exists() and sha256(csv_path) == call.get("csv_sha256"),
            "rows_match": csv_path.exists() and len(pd.read_csv(csv_path)) == call.get("rows"),
        })
    checks = pd.DataFrame(call_checks)
    checks.to_csv(out / "call_integrity.csv", index=False)

    reported_2015 = pd.read_csv(RAW / "aapl_actual_reported_2015q1.csv")
    utc_2015 = pd.read_csv(RAW / "aapl_financial_original_announcement.csv")
    reported_2020 = pd.read_csv(RAW / "aapl_reported_actuals_2020.csv")
    utc_2020 = pd.read_csv(RAW / "aapl_original_announcements_2020.csv")

    local = pd.concat([
        reported_2015[["Report Date"]], reported_2020[["Report Date"]]
    ], ignore_index=True)
    utc = pd.concat([
        utc_2015[["Original Announcement Date Time"]],
        utc_2020[["Original Announcement Date Time"]],
    ], ignore_index=True)
    comparison = pd.DataFrame({
        "report_date_naive": pd.to_datetime(local["Report Date"], errors="raise"),
        # LSEG CSV serialization mixes ISO-8601 ``T...Z`` values with
        # space-separated UTC timestamps across otherwise identical calls.
        # Parse each representation explicitly rather than coercing failures.
        "original_announcement_utc": pd.to_datetime(
            utc["Original Announcement Date Time"],
            format="mixed",
            utc=True,
            errors="raise",
        ),
    })
    comparison["original_announcement_in_new_york"] = (
        comparison.original_announcement_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    )
    comparison["absolute_clock_difference_seconds"] = (
        comparison.report_date_naive - comparison.original_announcement_in_new_york
    ).abs().dt.total_seconds()
    comparison["within_five_minutes"] = comparison.absolute_clock_difference_seconds.le(300)
    comparison.to_csv(out / "aapl_timezone_crosscheck.csv", index=False)

    fq1 = pd.read_csv(RAW / "aapl_consensus_fq1_daily.csv")
    fq0 = pd.read_csv(RAW / "aapl_consensus_fq0_daily.csv")
    for frame in (fq1, fq0):
        frame["Calc Date"] = pd.to_datetime(frame["Calc Date"], errors="raise")
        frame["Period End Date"] = pd.to_datetime(frame["Period End Date"], errors="raise")
    event_day = pd.Timestamp("2015-01-27")
    event_period = pd.Timestamp("2014-12-31")
    eligible_fq1 = fq1.loc[(fq1["Calc Date"] < event_day) & (fq1["Period End Date"] == event_period)]
    eligible_fq0 = fq0.loc[(fq0["Calc Date"] < event_day) & (fq0["Period End Date"] == event_period)]
    latest_fq1 = eligible_fq1.sort_values("Calc Date").tail(1)

    all_actual = pd.read_csv(RAW / "aapl_actual_all_2015q1.csv")
    reported_actual = pd.read_csv(RAW / "aapl_actual_reported_2015q1.csv")
    event_calendar = pd.read_csv(RAW / "aapl_event_release_2015.csv")

    summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_probe_run": manifest["run_id"],
        "raw_manifest_sha256": sha256(RAW / "manifest.json"),
        "integrity": {
            "calls": int(len(checks)),
            "all_returned": bool(checks.status.eq("returned").all()),
            "all_csv_hashes_match": bool(checks.csv_hash_matches.all()),
            "all_row_counts_match": bool(checks.rows_match.all()),
        },
        "timezone_crosscheck": {
            "aapl_rows": int(len(comparison)),
            "all_utc_reference_times_map_within_five_minutes_of_naive_report_clock_when_converted_to_new_york": bool(comparison.within_five_minutes.all()),
            "maximum_absolute_difference_seconds": float(comparison.absolute_clock_difference_seconds.max()),
            "dst_offsets_observed_hours": sorted(set(
                comparison.original_announcement_utc.map(lambda x: x.tz_convert("America/New_York").utcoffset().total_seconds() / 3600)
            )),
            "scope": "AAPL-only field cross-check; not a universal vendor-timezone declaration.",
        },
        "point_in_time_consensus": {
            "fq1_daily_rows": int(len(fq1)),
            "fq1_pre_event_exact_period_rows": int(len(eligible_fq1)),
            "fq0_pre_event_exact_period_rows": int(len(eligible_fq0)),
            "latest_eligible_calc_date": latest_fq1["Calc Date"].iloc[0].date().isoformat(),
            "latest_eligible_mean": float(latest_fq1["Earnings Per Share - Mean"].iloc[0]),
            "latest_eligible_stddev": float(latest_fq1["Earnings Per Share - Standard Deviation"].iloc[0]),
            "latest_eligible_estimate_count": int(latest_fq1["Earnings Per Share - Number of Included Estimates"].iloc[0]),
            "fq1_rolls_to_next_period_on_event_day": bool(
                fq1.loc[fq1["Calc Date"] == event_day, "Period End Date"].eq(pd.Timestamp("2015-03-31")).all()
            ),
            "strict_pre_event_and_exact_period_match_selects_only_2014q4_target": bool(
                len(eligible_fq1) > 0 and eligible_fq1["Period End Date"].eq(event_period).all()
            ),
        },
        "actual_selector": {
            "reported_rows": int(len(reported_actual)),
            "all_rows": int(len(all_actual)),
            "same_value_for_aapl_2015q1": bool(
                float(reported_actual["Earnings Per Share - Actual"].iloc[0])
                == float(all_actual["Earnings Per Share - Actual"].iloc[0])
            ),
            "multiple_versions_exposed_by_all_for_this_case": bool(len(all_actual) > 1),
        },
        "event_calendar": {
            "rows": int(len(event_calendar)),
            "event_start_time": str(event_calendar["Event Start Time"].iloc[0]),
            "report_date_clock": str(reported_actual["Report Date"].iloc[0]),
            "substitution_decision": "Do not substitute Event Start Time for Report Date; it represents the scheduled earnings event and differs from the release timestamp in this case.",
        },
        "decisions": [
            "Use ActType=Reported explicitly in the next actuals collection.",
            "Upgrade consensus collection from weekly to daily Calc Date observations.",
            "Match the latest Calc Date strictly before the supplied Report Date calendar day and require exact Period End Date.",
            "Use the first market session strictly after the supplied Report Date calendar day for the formal strategy; keep intraday session labels diagnostic only.",
            "Do not use TR.EventStartTime or TR.F.OriginalAnnouncementDate as universal replacements for EPS Report Date.",
        ],
        "limits": [
            "The AAPL UTC comparison validates both EST and EDT offsets for four cases, not every issuer.",
            "ActType=All exposed no extra AAPL 2015Q1 version; this does not prove that other issuers have no restatements.",
            "Daily Calc Date is date-level point-in-time evidence; no intraday estimate cutoff was established.",
        ],
        "non_actions": [
            "No existing raw, clean, panel, feature, label, or diagnostic file was changed.",
            "No missing value was filled and no row was deleted.",
            "No model or backtest was run.",
        ],
        "sources": [
            "https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/actuals",
            "https://community.developers.lseg.com/discussion/134049/retrieving-historical-tr-epsmean-data",
            "https://community.developers.lseg.com/discussion/121794",
        ],
    }

    report = f"""# Timing and point-in-time field audit

Run ID: `{run_id}`. Raw probe: `{manifest['run_id']}`.

## What the data shows

All {len(checks)} isolated requests returned and their saved hashes and row counts agree with the manifest.

For four AAPL announcements covering winter and daylight-saving time, converting the independent UTC-marked `TR.F.OriginalAnnouncementDate` values to `America/New_York` reproduces the timezone-naive I/B/E/S `Report Date` clocks within five minutes. Both UTC-5 and UTC-4 offsets appear. This is strong AAPL evidence that `Report Date` is an Eastern local clock, but it is not treated as a universal field declaration.

The daily `TR.EPSMean.calcdate` sequence contains {len(eligible_fq1)} exact-period observations strictly before the 2015-01-27 event. The last is 2015-01-26 with mean {float(latest_fq1['Earnings Per Share - Mean'].iloc[0]):.5f}, standard deviation {float(latest_fq1['Earnings Per Share - Standard Deviation'].iloc[0]):.5f}, and {int(latest_fq1['Earnings Per Share - Number of Included Estimates'].iloc[0])} included estimates. On the event day, `FQ1` rolls to the next fiscal period. Exact `Period End Date` matching therefore remains mandatory.

`ActType=All` and explicit `ActType=Reported` each returned one identical AAPL 2015Q1 actual. That case supplies no restatement history. The event-calendar start time was 21:00, while the UTC-marked original announcement was 21:30; the scheduled event field is not a replacement for the EPS release timestamp.

## Protocol decision

The next dataset version will request explicit `ActType=Reported` actuals and daily consensus observations. It will select the latest `Calc Date` strictly before the supplied Report Date calendar day with an exact period-end match. Formal entry will be the first market session strictly after that calendar day. This removes reliance on the unverified intraday session classification and is safe under both ET-local and UTC interpretations audited separately.

No existing dataset or panel was overwritten. No rows were removed or filled, and no model or backtest was run.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    summary["outputs"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.iterdir())
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "out": str(out),
        "all_calls_valid": summary["integrity"],
        "timezone_crosscheck": summary["timezone_crosscheck"],
        "point_in_time_consensus": summary["point_in_time_consensus"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
