"""Audit a timezone-robust event-entry rule without changing the saved panel.

The raw I/B/E/S Report Date timestamps are timezone-naive.  This audit compares
the two plausible interpretations used in the prior sensitivity review:

1. the supplied calendar date is already US Eastern local time; and
2. the supplied clock is UTC and must be converted to America/New_York.

For a close-to-close strategy, entering on the first benchmark session strictly
after the later of those two candidate calendar dates is safe under either
interpretation.  Raw values and the existing v2 panel remain untouched.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ACTUALS_PATH = ROOT / "data" / "clean" / "v2" / "actuals.csv"
RETURNS_PATH = ROOT / "data" / "clean" / "v2" / "returns.csv"
EVENTS_PATH = ROOT / "data" / "panel_v2" / "20260908T111211716224Z" / "events.csv"
OUT_ROOT = ROOT / "data" / "audit" / "announcement_time_robustness"
STUDY_START = pd.Timestamp("2015-01-01")
STUDY_END = pd.Timestamp("2026-06-30")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_naive(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, format="mixed", errors="raise")
    if parsed.dt.tz is not None:
        raise ValueError(f"{name} unexpectedly contains timezone-aware values")
    return parsed


def next_session_strictly_after(days: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    positions = sessions.searchsorted(pd.DatetimeIndex(days), side="right")
    result = np.full(len(days), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = positions < len(sessions)
    result[valid] = sessions.to_numpy()[positions[valid]]
    return pd.Series(result, index=days.index)


def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)

    actuals = pd.read_csv(ACTUALS_PATH)
    actuals["announcement"] = parse_naive(actuals["announcement"], "actual announcement")
    actuals = actuals.loc[actuals.announcement.dt.normalize().between(STUDY_START, STUDY_END)].copy()
    actuals["raw_calendar_day"] = actuals.announcement.dt.normalize()
    as_utc = actuals.announcement.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    actuals["utc_interpretation_et_day"] = as_utc.dt.tz_localize(None).dt.normalize()
    actuals["candidate_days_differ"] = actuals.raw_calendar_day.ne(actuals.utc_interpretation_et_day)
    actuals["safe_reference_day"] = actuals[["raw_calendar_day", "utc_interpretation_et_day"]].max(axis=1)
    if not actuals.safe_reference_day.eq(actuals.raw_calendar_day).all():
        raise AssertionError("The later candidate day is not always the supplied calendar day")

    minute = actuals.announcement.dt.hour * 60 + actuals.announcement.dt.minute
    actuals["minute_of_day"] = minute
    actuals["clock_bucket"] = pd.cut(
        minute,
        bins=[-1, 569, 959, 1439],
        labels=["before_0930", "0930_to_1559", "1600_or_later"],
    ).astype("string")

    time_profile = (
        actuals.groupby([actuals.announcement.dt.hour.rename("hour"), "clock_bucket"], dropna=False)
        .size().rename("rows").reset_index()
    )
    time_profile.to_csv(out / "time_of_day_profile.csv", index=False)

    example_mask = (
        ((actuals.Instrument.isin(["AAPL.O", "AAPL.OQ", "MSFT.O", "MSFT.OQ", "INTC.O", "INTC.OQ"]))
         & actuals.raw_calendar_day.between("2015-01-01", "2015-03-31"))
        | ((actuals.Instrument.isin(["AAPL.O", "AAPL.OQ"]))
           & actuals.raw_calendar_day.between("2020-01-01", "2020-08-31"))
    )
    examples = actuals.loc[example_mask, [
        "Instrument", "announcement", "raw_calendar_day", "utc_interpretation_et_day",
        "candidate_days_differ", "announcement_session", "Period End Date",
        "Earnings Per Share - Actual",
    ]].sort_values(["Instrument", "announcement"])
    examples.to_csv(out / "known_clock_examples.csv", index=False)

    returns = pd.read_csv(RETURNS_PATH, usecols=["Instrument", "Date"])
    returns["Date"] = pd.to_datetime(returns.Date, errors="raise")
    sessions = pd.DatetimeIndex(
        returns.loc[returns.Instrument.eq("SPY.P"), "Date"].drop_duplicates().sort_values()
    )
    if len(sessions) == 0:
        raise ValueError("No SPY.P benchmark sessions found")

    events = pd.read_csv(EVENTS_PATH)
    events["announcement"] = parse_naive(events.announcement, "panel event announcement")
    events["existing_entry_close"] = pd.to_datetime(events.entry_close, errors="coerce")
    events["raw_calendar_day"] = events.announcement.dt.normalize()
    event_as_utc = events.announcement.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    events["utc_interpretation_et_day"] = event_as_utc.dt.tz_localize(None).dt.normalize()
    events["safe_reference_day"] = events[["raw_calendar_day", "utc_interpretation_et_day"]].max(axis=1)
    events["robust_entry_close"] = next_session_strictly_after(events.safe_reference_day, sessions)
    events["existing_entry_available"] = events.entry_status.eq("available")
    events["entry_changed"] = ~(
        events.existing_entry_close.eq(events.robust_entry_close)
        | (events.existing_entry_close.isna() & events.robust_entry_close.isna())
    )
    events["available_entry_changed"] = events.existing_entry_available & events.entry_changed
    comparison_columns = [
        "source", "announcement", "announcement_session", "status", "graph_status", "entry_status",
        "raw_calendar_day", "utc_interpretation_et_day", "safe_reference_day",
        "existing_entry_close", "robust_entry_close", "existing_entry_available", "entry_changed",
        "available_entry_changed",
    ]
    events[comparison_columns].to_csv(out / "event_entry_comparison.csv", index=False)

    changed_by_session = (
        events.groupby("announcement_session", dropna=False).entry_changed
        .agg(rows="size", changed="sum").reset_index()
    )
    changed_by_session.to_csv(out / "entry_changes_by_session.csv", index=False)

    summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only audit of a timezone-robust strict-next-session entry rule.",
        "inputs": {
            str(ACTUALS_PATH.relative_to(ROOT)): sha256(ACTUALS_PATH),
            str(RETURNS_PATH.relative_to(ROOT)): sha256(RETURNS_PATH),
            str(EVENTS_PATH.relative_to(ROOT)): sha256(EVENTS_PATH),
        },
        "study_actual_rows": int(len(actuals)),
        "naive_timestamp_rows": int(len(actuals)),
        "candidate_et_calendar_days_differ_if_clock_is_utc": int(actuals.candidate_days_differ.sum()),
        "later_candidate_is_always_supplied_calendar_day": bool(actuals.safe_reference_day.eq(actuals.raw_calendar_day).all()),
        "clock_buckets": {str(k): int(v) for k, v in actuals.clock_bucket.value_counts(dropna=False).items()},
        "panel_event_rows": int(len(events)),
        "panel_events_with_existing_available_entry": int(events.existing_entry_available.sum()),
        "available_entries_changed_under_strict_next_session": int(events.available_entry_changed.sum()),
        "available_entries_unchanged": int((events.existing_entry_available & ~events.entry_changed).sum()),
        "events_without_existing_entry_not_used_as_change_denominator": int((~events.existing_entry_available).sum()),
        "robust_entry_unavailable": int(events.robust_entry_close.isna().sum()),
        "rule": "Treat the supplied Report Date calendar day and the UTC-to-America/New_York candidate day as an ambiguity set; enter at the first SPY.P session strictly after the later candidate day.",
        "proof_observation": "For every row in this US-equity sample, UTC-to-America/New_York is the same or prior calendar day, so the later candidate is the supplied calendar day.",
        "non_actions": [
            "No timezone was assigned to the source field.",
            "No source, clean, panel, feature, label, or diagnostic file was overwritten.",
            "No missing value was filled and no row was deleted.",
            "No model or backtest was run.",
        ],
        "outputs": {},
    }

    report = [
        "# Announcement-time robustness audit",
        "",
        f"Run ID: `{run_id}`.",
        "",
        "The saved I/B/E/S Report Date is timezone-naive. This audit does not label it ET or UTC. "
        "Instead, it constructs both calendar-day interpretations and uses the later day, then enters "
        "on the first benchmark session strictly after that day.",
        "",
        f"- Study actual rows: {len(actuals):,}",
        f"- Rows whose ET calendar date would differ if the supplied clock were UTC: {int(actuals.candidate_days_differ.sum()):,}",
        f"- Events with an existing available entry: {int(events.existing_entry_available.sum()):,}",
        f"- Available entries changed by the conservative rule: {int(events.available_entry_changed.sum()):,}",
        f"- Events without an existing entry, excluded from that denominator: {int((~events.existing_entry_available).sum()):,}",
        f"- Robust entries unavailable at the data boundary: {int(events.robust_entry_close.isna().sum()):,}",
        "",
        "The later candidate day equals the supplied Report Date calendar day for every row. Therefore "
        "strictly waiting until the next session prevents same-day look-ahead under either interpretation. "
        "This is a research-design safeguard, not a claim about the vendor field's true timezone.",
        "",
        "The comparison output retains every event and both entry dates. No saved panel or data table was changed, "
        "and no model or performance calculation was run.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    for path in sorted(out.iterdir()):
        if path.name != "summary.json":
            summary["outputs"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "out": str(out),
        "actual_rows": len(actuals),
        "candidate_day_differences": int(actuals.candidate_days_differ.sum()),
        "existing_available_entries": int(events.existing_entry_available.sum()),
        "changed_available_entries": int(events.available_entry_changed.sum()),
        "robust_entry_unavailable": int(events.robust_entry_close.isna().sum()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
