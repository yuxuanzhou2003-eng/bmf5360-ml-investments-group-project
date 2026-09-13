"""Audit a frozen raw LSEG stock-split run; never edits raw or research data."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/stock_splits_v2/20260908T163414365646Z"
if not RAW.is_absolute():
    RAW = ROOT / RAW
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data/audit/stock_split_cases" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
FEATURES = ROOT / "data/panel_v2/20260908T111211716224Z/features.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = sorted(RAW.glob("batch_*.csv"))
if len(files) != 8:
    raise RuntimeError(f"Expected 8 full-run batches, got {len(files)}")
input_hashes_before = {path.name: sha256(path) for path in files} | {"features.csv": sha256(FEATURES)}
frames = []
for path in files:
    frame = pd.read_csv(path)
    frame["raw_file"] = path.name
    frame["raw_row"] = np.arange(len(frame))
    frames.append(frame)
raw = pd.concat(frames, ignore_index=True)
factor = "Adjustment Factor"
old, new = "Terms Old Shares", "Terms New Shares"
event_mask = raw[[factor, old, new]].notna().any(axis=1)
events = raw.loc[event_mask].copy()
placeholders = raw.loc[~event_mask].copy()
events["event_status"] = "valid_split_event"
placeholders["event_status"] = "no_split_event_returned_in_window"

for column in ["Capital Change Announcement Date", "Capital Change Record Date",
               "Capital Change Effective Date", "Capital Change Ex Date"]:
    events[column] = pd.to_datetime(events[column], errors="raise")
ratio = events[old] / events[new]
factor_match = np.isclose(events[factor], ratio, atol=5e-6, rtol=0)
key = ["Instrument", "Capital Change Announcement Date", "Capital Change Ex Date", factor]

panel = pd.read_csv(
    FEATURES,
    usecols=["sample_id", "source", "announcement", "period_end", "snapshot", "actual",
             "consensus", "dispersion", "analysts", "standardized_surprise"],
)
panel["announcement_day"] = pd.to_datetime(panel["announcement"]).dt.normalize()
windows = [
    ("AAPL_2015_Q1", "AAPL.OQ", "2015-01-01", "2015-03-31"),
    ("AAPL_2020_split_announcement", "AAPL.OQ", "2020-07-01", "2020-09-30"),
    ("TSLA_2020_split_announcement", "TSLA.OQ", "2020-07-01", "2020-09-30"),
    ("NVDA_2021_split_window", "NVDA.OQ", "2021-05-01", "2021-08-31"),
    ("NVDA_2024_split_window", "NVDA.OQ", "2024-05-01", "2024-08-31"),
]
case_parts, window_counts = [], []
for label, ric, start, end in windows:
    selected = panel.loc[panel["source"].eq(ric) & panel["announcement_day"].between(start, end)].copy()
    unique = selected.drop_duplicates(["source", "announcement", "period_end"]).copy()
    unique["case_window"] = label
    case_parts.append(unique)
    window_counts.append({"case_window": label, "source": ric, "edge_rows": len(selected),
                          "unique_source_events": len(unique)})
cases = pd.concat(case_parts, ignore_index=True)

# Current historical EPS may reflect splits after an announcement. This is a
# deterministic arithmetic diagnostic, not a claim about LSEG adjustment rules.
diagnostics = []
for row in cases.itertuples(index=False):
    later = events.loc[
        events["Instrument"].eq(row.source)
        & events["Capital Change Ex Date"].gt(row.announcement_day)
    ]
    multiplier = float((later[new] / later[old]).prod()) if len(later) else 1.0
    original_z = ((row.actual - row.consensus) / row.dispersion
                  if pd.notna(row.dispersion) and row.dispersion != 0 else np.nan)
    common_z = (((row.actual * multiplier) - (row.consensus * multiplier))
                / (row.dispersion * multiplier)
                if pd.notna(row.dispersion) and row.dispersion != 0 else np.nan)
    actual_only_z = (((row.actual * multiplier) - row.consensus) / row.dispersion
                     if pd.notna(row.dispersion) and row.dispersion != 0 else np.nan)
    diagnostics.append({
        "case_window": row.case_window, "source": row.source, "announcement": row.announcement,
        "period_end": row.period_end, "snapshot": row.snapshot, "actual": row.actual,
        "consensus": row.consensus, "dispersion": row.dispersion, "analysts": row.analysts,
        "recorded_standardized_surprise": row.standardized_surprise,
        "later_split_count": len(later), "later_split_share_multiplier": multiplier,
        "hypothetical_common_scale_actual": row.actual * multiplier,
        "hypothetical_common_scale_consensus": row.consensus * multiplier,
        "hypothetical_common_scale_dispersion": row.dispersion * multiplier,
        "recomputed_original_z": original_z, "common_scale_z": common_z,
        "actual_only_scale_z": actual_only_z,
        "interpretation": "arithmetic diagnostic only; not a vendor-vintage conclusion",
    })
diag = pd.DataFrame(diagnostics)

events.to_csv(OUT / "split_events.csv", index=False)
placeholders.to_csv(OUT / "no_split_placeholders.csv", index=False)
pd.DataFrame(window_counts).to_csv(OUT / "case_window_counts.csv", index=False)
diag.to_csv(OUT / "eps_scale_diagnostics.csv", index=False)

checks = {
    "eight_raw_batches": len(files) == 8,
    "all_782_instruments_returned": raw["Instrument"].nunique() == 782,
    "all_valid_events_complete_factor_terms": not events[[factor, old, new]].isna().any().any(),
    "all_valid_events_complete_dates": not events[["Capital Change Announcement Date",
                                                     "Capital Change Record Date",
                                                     "Capital Change Effective Date",
                                                     "Capital Change Ex Date"]].isna().any().any(),
    "factor_equals_old_over_new": bool(factor_match.all()),
    "no_exact_event_duplicates": not events.duplicated().any(),
    "no_event_key_duplicates": not events.duplicated(key).any(),
    "common_scale_preserves_z": bool(np.allclose(diag["recomputed_original_z"], diag["common_scale_z"],
                                                  equal_nan=True, atol=1e-12)),
    "all_rows_preserved_in_audit_outputs": len(events) + len(placeholders) == len(raw),
}
input_hashes_after = {path.name: sha256(path) for path in files} | {"features.csv": sha256(FEATURES)}
checks["inputs_unchanged"] = input_hashes_before == input_hashes_after
summary = {
    "run_id": RUN_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_raw_run": str(RAW.relative_to(ROOT)), "raw_physical_rows": len(raw),
    "returned_instruments": raw["Instrument"].nunique(), "valid_split_events": len(events),
    "split_event_instruments": events["Instrument"].nunique(), "no_split_placeholders": len(placeholders),
    "event_announcement_min": str(events["Capital Change Announcement Date"].min().date()),
    "event_ex_date_max": str(events["Capital Change Ex Date"].max().date()),
    "input_hashes_before": input_hashes_before, "input_hashes_after": input_hashes_after,
    "audit_script_sha256": sha256(Path(__file__)),
    "checks": checks,
    "limitations": [
        "A no-split placeholder is a returned blank event row, not proof that all vendor content is complete.",
        "Corporate-action factors are not applied to actuals, estimates, returns, labels, or positions.",
        "The arithmetic scale diagnostic cannot prove EPS basis or point-in-time vintage.",
        "Case windows are fixed in code and include every source event returned by the current panel in each window.",
    ],
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not all(checks.values()):
    raise RuntimeError({key: value for key, value in checks.items() if not value})
print(json.dumps({key: summary[key] for key in ["run_id", "raw_physical_rows", "valid_split_events",
                                                 "split_event_instruments", "no_split_placeholders"]}, indent=2))
