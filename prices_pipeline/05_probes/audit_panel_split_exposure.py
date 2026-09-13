"""Mark every source event with later LSEG stock splits; no data changes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "data/panel_v2/20260908T111211716224Z/features.csv"
SPLITS = ROOT / "data/audit/stock_split_cases/20260908T163803607846Z/split_events.csv"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data/audit/stock_split_exposure" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = {"features.csv": sha256(FEATURES), "split_events.csv": sha256(SPLITS)}
features = pd.read_csv(
    FEATURES,
    usecols=["source", "announcement", "period_end", "actual", "consensus", "dispersion",
             "eps_difference", "standardized_surprise"],
)
features["announcement_day"] = pd.to_datetime(features["announcement"]).dt.normalize()
splits = pd.read_csv(SPLITS)
splits["ex_date"] = pd.to_datetime(splits["Capital Change Ex Date"])
splits["share_multiplier"] = splits["Terms New Shares"] / splits["Terms Old Shares"]

event_key = ["source", "announcement", "period_end"]
events = features.groupby(event_key, as_index=False).agg(
    edge_rows=("source", "size"), actual=("actual", "first"), consensus=("consensus", "first"),
    dispersion=("dispersion", "first"), eps_difference=("eps_difference", "first"),
    standardized_surprise=("standardized_surprise", "first"), announcement_day=("announcement_day", "first"),
)


def exposure(row) -> pd.Series:
    later = splits.loc[splits["Instrument"].eq(row.source) & splits["ex_date"].gt(row.announcement_day)]
    return pd.Series({
        "later_split_count": len(later),
        "later_split_share_multiplier": float(later["share_multiplier"].prod()) if len(later) else 1.0,
        "later_split_ex_dates": "|".join(later["Capital Change Ex Date"].astype(str)),
        "later_split_ratios": "|".join(
            later["Terms Old Shares"].astype(str) + "->" + later["Terms New Shares"].astype(str)
        ),
    })


events = pd.concat([events, events.apply(exposure, axis=1)], axis=1)
events["split_scale_status"] = np.where(events["later_split_count"].gt(0),
                                         "later_split_observed", "no_later_split_observed")
events["hypothetical_pre_later_split_actual"] = events["actual"] * events["later_split_share_multiplier"]
events["hypothetical_pre_later_split_consensus"] = events["consensus"] * events["later_split_share_multiplier"]
events["hypothetical_pre_later_split_dispersion"] = events["dispersion"] * events["later_split_share_multiplier"]
valid_z = events["dispersion"].notna() & events["dispersion"].ne(0)
events["common_scale_z"] = np.nan
events.loc[valid_z, "common_scale_z"] = (
    (events.loc[valid_z, "hypothetical_pre_later_split_actual"]
     - events.loc[valid_z, "hypothetical_pre_later_split_consensus"])
    / events.loc[valid_z, "hypothetical_pre_later_split_dispersion"]
)
events.to_csv(OUT / "event_split_exposure.csv", index=False)

affected = events["later_split_count"].gt(0)
after = {"features.csv": sha256(FEATURES), "split_events.csv": sha256(SPLITS)}
checks = {
    "all_107532_edges_accounted": int(events["edge_rows"].sum()) == 107532,
    "unique_22523_source_events": len(events) == 22523,
    "event_keys_unique": not events.duplicated(event_key).any(),
    "valid_z_matches_panel": bool(np.allclose(events.loc[valid_z, "common_scale_z"],
                                               events.loc[valid_z, "standardized_surprise"],
                                               rtol=1e-10, atol=1e-12)),
    "zero_dispersion_kept_with_missing_z": bool(
        events.loc[events["dispersion"].eq(0), "standardized_surprise"].isna().all()
    ),
    "inputs_unchanged": before == after,
}
summary = {
    "run_id": RUN_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_events": len(events), "edge_rows": int(events["edge_rows"].sum()),
    "events_with_later_split": int(affected.sum()),
    "edges_with_later_split": int(events.loc[affected, "edge_rows"].sum()),
    "source_rics_with_later_split": int(events.loc[affected, "source"].nunique()),
    "zero_dispersion_events": int(events["dispersion"].eq(0).sum()),
    "zero_dispersion_edges": int(features["dispersion"].eq(0).sum()),
    "multiplier_event_counts": {
        str(key): int(value) for key, value in
        events.loc[affected, "later_split_share_multiplier"].value_counts().sort_index().items()
    },
    "input_hashes_before": before, "input_hashes_after": after,
    "script_sha256": sha256(Path(__file__)), "checks": checks,
    "scope": "Read-only scale-exposure audit. Later-split presence is not proof of vendor EPS adjustment or vintage.",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not all(checks.values()):
    raise RuntimeError({key: value for key, value in checks.items() if not value})
print(json.dumps({key: summary[key] for key in ["run_id", "source_events", "edge_rows",
                                                 "events_with_later_split", "edges_with_later_split",
                                                 "source_rics_with_later_split"]}, indent=2))
