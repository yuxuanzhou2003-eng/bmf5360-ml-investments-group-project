"""Fast training/validation read on the common-analyst graph: is there any signal at all?

This is a deliberately small check, not the model-ready rebuild. It reuses the frozen event-level
surprise and entry sessions from panel v3, computes the 5-session benchmark-excess target for the
NEW edges from the same v2 total-return table the baseline used, and reports rank correlations for
the raw signal `standardized_surprise x jaccard`.

Declared before looking at any number, so the answer cannot be chosen after the fact:
  - training 2015-01-01..2020-12-31, validation 2021-01-01..2022-12-31, test untouched;
  - any edge whose 5-session exit crosses into the next split is purged;
  - three strata reported together -- all edges, cross-sector, same-sector -- because the point of
    the cross-sector stratum is to separate diffusion from same-industry earnings co-movement;
  - the same statistic is reported for the rejected residual-correlation graph as the reference.

Sector labels come from the current TRBC fetch. Point-in-time behaviour is unverified, so they
stratify the report only and never enter a signal, a feature or an edge-selection rule.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / "data" / "analyst_graph_v1" / "20260909T094242433400Z"
PANEL = ROOT / "data" / "panel_v3" / "20260909T035245922341Z"
RETURNS = ROOT / "data" / "clean" / "v2" / "returns.csv"
TRBC = ROOT / "data" / "raw" / "trbc_diagnostic_v1" / "trbc_current.csv"
OUT_ROOT = ROOT / "data" / "analysis" / "analyst_signal_quickcheck_v1"
EVENT_KEY = ["source", "announcement", "period_end"]
HORIZON = 5
SPLITS = {"training": ("2015-01-01", "2020-12-31", "2021-01-01"),
          "validation": ("2021-01-01", "2022-12-31", "2023-01-01")}


def event_ic(frame: pd.DataFrame, signal: str, target: str) -> tuple[float, float, int]:
    usable = frame.loc[frame[signal].notna() & frame[target].notna()]
    if len(usable) < 3:
        return np.nan, np.nan, 0
    pooled = spearmanr(usable[signal], usable[target]).statistic
    values = []
    for _, group in usable.groupby(EVENT_KEY):
        if len(group) >= 3 and group[signal].nunique() > 1 and group[target].nunique() > 1:
            value = spearmanr(group[signal], group[target]).statistic
            if np.isfinite(value):
                values.append(value)
    return float(pooled), (float(np.mean(values)) if values else np.nan), len(values)


def main() -> int:
    events = pd.read_csv(PANEL / "events.csv", low_memory=False)
    events["announcement_day"] = pd.to_datetime(events.announcement_day).dt.normalize()
    events["entry_session"] = pd.to_datetime(events.entry_session, errors="coerce").dt.normalize()
    event_level = events[EVENT_KEY + ["announcement_day", "entry_session", "standardized_surprise"]]

    returns = pd.read_csv(RETURNS, usecols=["Instrument", "Date", "return_decimal"])
    returns["Date"] = pd.to_datetime(returns.Date).dt.normalize()
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    sessions = wide.index
    # Forward cumulative return over the H sessions strictly after the entry session.
    forward = (1 + wide).rolling(HORIZON, min_periods=HORIZON).apply(np.prod, raw=True).shift(-HORIZON) - 1
    spy_forward = forward["SPY.P"]

    def attach_targets(edges: pd.DataFrame, strength_columns: list[str]) -> pd.DataFrame:
        frame = edges.merge(event_level, on=EVENT_KEY, how="inner", validate="many_to_one")
        frame = frame.loc[frame.entry_session.notna()].copy()
        entry_position = sessions.get_indexer(frame.entry_session)
        frame = frame.loc[entry_position >= 0].copy()
        entry_position = entry_position[entry_position >= 0]
        exit_position = entry_position + HORIZON
        valid = exit_position < len(sessions)
        frame["exit_session"] = pd.NaT
        frame.loc[valid, "exit_session"] = sessions[exit_position[valid]]
        receiver_forward = np.full(len(frame), np.nan)
        columns = forward.columns
        receiver_index = columns.get_indexer(frame.receiver)
        matrix = forward.to_numpy()
        known = (receiver_index >= 0)
        receiver_forward[known] = matrix[entry_position[known], receiver_index[known]]
        frame["target"] = receiver_forward - spy_forward.to_numpy()[entry_position]
        frame["split"] = np.select(
            [frame.announcement_day.between(*[pd.Timestamp(SPLITS["training"][index]) for index in (0, 1)]),
             frame.announcement_day.between(*[pd.Timestamp(SPLITS["validation"][index]) for index in (0, 1)])],
            ["training", "validation"], default="excluded")
        limit = frame.split.map({name: pd.Timestamp(bounds[2]) for name, bounds in SPLITS.items()})
        frame["boundary_purged"] = frame.exit_session.notna() & frame.exit_session.ge(limit)
        frame.loc[frame.boundary_purged, "target"] = np.nan
        for column in strength_columns:
            frame[f"signal_{column}"] = frame.standardized_surprise * frame[column]
        return frame.loc[frame.split.ne("excluded")]

    sector = pd.read_csv(TRBC)
    sector_column = next(column for column in sector.columns if "Economic Sector" in column)
    sector_map = sector.set_index("Instrument")[sector_column]

    new_edges = pd.read_csv(GRAPH / "edges.csv")
    new_edges = new_edges.loc[new_edges.reason.eq("ok"),
                              EVENT_KEY + ["receiver", "jaccard_coverage", "jaccard_rec_weighted",
                                           "jaccard_named_only", "common_brokers"]]
    strengths = ["jaccard_coverage", "jaccard_rec_weighted", "jaccard_named_only"]
    new_frame = attach_targets(new_edges, strengths)
    new_frame["same_sector"] = (new_frame.source.map(sector_map) ==
                                new_frame.receiver.map(sector_map))

    old_features = pd.read_csv(PANEL / "features.csv", low_memory=False,
                               usecols=EVENT_KEY + ["receiver", "residual_correlation"])
    old_frame = attach_targets(old_features, ["residual_correlation"])
    old_frame["same_sector"] = (old_frame.source.map(sector_map) ==
                                old_frame.receiver.map(sector_map))

    results = []
    for name, frame, signals in [("analyst_graph", new_frame, [f"signal_{column}" for column in strengths]),
                                 ("rejected_residual_graph", old_frame, ["signal_residual_correlation"])]:
        for stratum, subset in [("all", frame),
                                ("cross_sector", frame.loc[~frame.same_sector]),
                                ("same_sector", frame.loc[frame.same_sector])]:
            for split in ["training", "validation"]:
                part = subset.loc[subset.split.eq(split)]
                for signal in signals:
                    pooled, mean_event, groups = event_ic(part, signal, "target")
                    results.append({
                        "graph": name, "stratum": stratum, "split": split, "signal": signal,
                        "rows": int(part[signal].notna().sum() & part.target.notna().sum()
                                    if False else part.loc[part[signal].notna() &
                                                           part.target.notna()].shape[0]),
                        "events": int(part[EVENT_KEY].drop_duplicates().shape[0]),
                        "pooled_spearman": pooled, "mean_event_spearman": mean_event,
                        "event_groups": groups,
                    })

    table = pd.DataFrame(results)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "quickcheck_metrics.csv", index=False)
    summary = {
        "run_id": run_id,
        "graph_run": GRAPH.name,
        "scope": "exploratory training/validation rank check on raw signals; no model fitted",
        "horizon_sessions": HORIZON,
        "target": "receiver 5-session cumulative total return minus SPY, entry session excluded",
        "edges_new_graph": int(len(new_frame)),
        "edges_rejected_graph": int(len(old_frame)),
        "boundary_purged_new": int(new_frame.boundary_purged.sum()),
        "boundary_purged_old": int(old_frame.boundary_purged.sum()),
        "cross_sector_share_new": round(float((~new_frame.same_sector).mean()), 4),
        "test_policy": "No test rows, targets, predictions or metrics were computed.",
        "sector_caveat": "Current TRBC labels; stratification only, never a feature or selection rule.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    pivot = table.pivot_table(index=["graph", "signal", "stratum"], columns="split",
                              values="mean_event_spearman").round(5)
    print("=== mean event Spearman ===")
    print(pivot.to_string())
    print()
    print("=== pooled Spearman ===")
    print(table.pivot_table(index=["graph", "signal", "stratum"], columns="split",
                            values="pooled_spearman").round(5).to_string())
    print()
    print("=== rows ===")
    print(table.pivot_table(index=["graph", "signal", "stratum"], columns="split",
                            values="rows").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
