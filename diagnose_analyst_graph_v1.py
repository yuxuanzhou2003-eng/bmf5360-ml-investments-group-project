"""Does the common-analyst graph carry information beyond industry and beyond the rejected graph?

The whole reason for switching to common-analyst coverage was that it should be distinguishable
from the same-industry earnings co-movement that v1 already suspected of driving the signal. If
the selected neighbours are simply same-industry firms, the new graph is a TRBC industry graph in
disguise and the pivot buys nothing. This script tests that before any model is fitted:

  1. Same-industry rate among selected neighbours, against the rate expected if neighbours were
     drawn at random from the eligible universe of the same snapshot.
  2. Overlap between the new edge set and the rejected residual-correlation edge set.
  3. Share of edges that exist only because of masked broker tokens.
  4. Full similarity distribution, not just the retained top-5.

TRBC classification is fetched once at the current date. Its point-in-time behaviour is NOT
established (see RELATION_DATA_FEASIBILITY.md), so it is used only as a diagnostic label here and
must never enter a feature or a selection rule.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / "data" / "analyst_graph_v1" / "20260909T094242433400Z"
OLD_PANEL = ROOT / "data" / "panel_v3" / "20260909T035245922341Z"
COVERAGE_RUN = ROOT / "data" / "raw" / "broker_coverage_v1" / "20260909T085010336967Z"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
OUT_ROOT = ROOT / "data" / "analysis" / "analyst_graph_diagnostics_v1"
TRBC_CACHE = ROOT / "data" / "raw" / "trbc_diagnostic_v1" / "trbc_current.csv"
CREDS = dotenv_values(ROOT / ".env")
SECRETS = [str(value) for value in CREDS.values() if value]
RNG_SEED = 20260909


def fetch_trbc(instruments: list[str]) -> pd.DataFrame:
    if TRBC_CACHE.exists():
        return pd.read_csv(TRBC_CACHE)
    TRBC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    import lseg.data as ld
    ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
    frames = []
    for index in range(0, len(instruments), 30):
        chunk = instruments[index:index + 30]
        frames.append(ld.get_data(universe=chunk,
                                  fields=["TR.TRBCEconomicSector", "TR.TRBCIndustryGroup"]))
    frame = pd.concat(frames, ignore_index=True).replace(r"^\s*$", pd.NA, regex=True)
    frame.to_csv(TRBC_CACHE, index=False)
    return frame


def main() -> int:
    edges = pd.read_csv(GRAPH / "edges.csv")
    built = edges.loc[edges.reason.eq("ok")].copy()
    instruments = sorted(set(built.source) | set(built.receiver.dropna()))
    trbc = fetch_trbc(instruments)
    sector_column = next(column for column in trbc.columns if "Economic Sector" in column)
    group_column = next(column for column in trbc.columns if "Industry Group" in column)
    sector = trbc.set_index("Instrument")[sector_column]
    group = trbc.set_index("Instrument")[group_column]

    built["source_sector"] = built.source.map(sector)
    built["receiver_sector"] = built.receiver.map(sector)
    built["source_group"] = built.source.map(group)
    built["receiver_group"] = built.receiver.map(group)
    labelled = built.loc[built.source_sector.notna() & built.receiver_sector.notna()]
    same_sector = float(labelled.source_sector.eq(labelled.receiver_sector).mean())
    same_group = float(labelled.source_group.eq(labelled.receiver_group).mean())

    # Random baseline: for each snapshot, how often would two eligible members share a sector?
    spans = pd.read_csv(SPANS_PATH)
    spans["member_from"] = pd.to_datetime(spans.member_from)
    spans["member_to"] = pd.to_datetime(spans.member_to)
    rng = np.random.default_rng(RNG_SEED)
    sector_baseline, group_baseline = [], []
    for snapshot, count in built.groupby("snapshot").size().items():
        stamp = pd.Timestamp(snapshot)
        members = spans.loc[spans.member_from.le(stamp) & spans.member_to.ge(stamp), "ric"]
        pool = [ric for ric in members if ric in sector.index and pd.notna(sector.get(ric))]
        if len(pool) < 2:
            continue
        draws = min(int(count), 20000)
        left = rng.choice(pool, draws)
        right = rng.choice(pool, draws)
        keep = left != right
        sector_baseline.append(np.mean(sector.loc[left[keep]].to_numpy() ==
                                       sector.loc[right[keep]].to_numpy()))
        group_baseline.append(np.mean(group.loc[left[keep]].to_numpy() ==
                                      group.loc[right[keep]].to_numpy()))

    # Overlap with the rejected residual-correlation graph.
    old = pd.read_csv(OLD_PANEL / "features.csv", low_memory=False)
    old_meta = pd.read_csv(OLD_PANEL / "labels.csv", low_memory=False) if False else None
    old_edges = None
    for candidate in ["events.csv", "features.csv"]:
        frame = pd.read_csv(OLD_PANEL / candidate, low_memory=False)
        if {"source", "receiver"}.issubset(frame.columns):
            old_edges = frame[["source", "announcement", "period_end", "receiver"]]
            break
    if old_edges is None:
        panel_meta = pd.read_csv(ROOT / "data" / "model_ready_v1" / "20260909T064151673764Z" /
                                 "metadata.csv", low_memory=False)
        old_edges = panel_meta[["source", "announcement", "period_end", "receiver"]]
    key = ["source", "announcement", "period_end", "receiver"]
    new_keys = set(map(tuple, built[key].astype(str).to_numpy()))
    old_keys = set(map(tuple, old_edges[key].astype(str).to_numpy()))
    overlap = len(new_keys & old_keys)

    masked_only = built.loc[built.jaccard_named_only.eq(0)]
    report = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "graph_run": GRAPH.name,
        "edges_examined": int(len(built)),
        "edges_with_both_sectors_labelled": int(len(labelled)),
        "same_sector_rate": round(same_sector, 4),
        "same_sector_random_baseline": round(float(np.mean(sector_baseline)), 4),
        "same_industry_group_rate": round(same_group, 4),
        "same_industry_group_random_baseline": round(float(np.mean(group_baseline)), 4),
        "cross_sector_edge_share": round(1 - same_sector, 4),
        "overlap_with_rejected_graph": {
            "new_edges": len(new_keys),
            "old_edges": len(old_keys),
            "shared": overlap,
            "share_of_new": round(overlap / max(1, len(new_keys)), 4),
        },
        "masked_dependency": {
            "edges_with_zero_named_only_jaccard": int(len(masked_only)),
            "share": round(len(masked_only) / max(1, len(built)), 4),
        },
        "strength_correlation": {
            "coverage_vs_rec_weighted": round(float(
                built.jaccard_coverage.corr(built.jaccard_rec_weighted, method="spearman")), 4),
            "coverage_vs_named_only": round(float(
                built.jaccard_coverage.corr(built.jaccard_named_only, method="spearman")), 4),
        },
        "trbc_caveat": "TRBC fetched at the current date; point-in-time behaviour unverified. "
                       "Diagnostic label only, never a feature or a selection rule.",
    }
    out = OUT_ROOT / report["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    labelled.groupby(["source_sector", "receiver_sector"]).size().rename("edges").reset_index() \
            .to_csv(out / "sector_pair_counts.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
