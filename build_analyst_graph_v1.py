"""Build the point-in-time common-analyst coverage graph for the panel v3 event set.

Every rule below was approved by the user on 2026-09-09 and is recorded in RESEARCH_PROTOCOL.md
v0.7. Changing any of them requires re-approval.

  1. Broker identity is the raw token string. `Permission Denied <n>` is used verbatim as an id;
     no name resolution, mapping or substitution.
  2. An edge requires Jaccard similarity over covering brokers AND at least 3 common brokers.
     The absolute floor rejects pairs where a high Jaccard comes from tiny coverage sets; the
     normalisation stops widely covered large caps from linking to everything.
  3. Two strengths are emitted for the SAME edge set, which is selected on coverage only:
     `jaccard_coverage` and `jaccard_rec_weighted` (agreement of the 1-5 ratings the common
     brokers gave to the two companies). This makes "does rating agreement add anything" a
     measurable contrast rather than an assumption.
  4. Snapshots are quarter-end as-of; each event uses the latest snapshot STRICTLY before the
     announcement day.
  5. Snapshots older than 120 days relative to the announcement day are not used.
  6. Announcements before the first snapshot (2015-03-31) get no edge and are retained with an
     explicit reason code. 2015 Q1 is graph warm-up.
  7. At most 5 receivers per event, ranked by the coverage strength.
  8. Splits, boundary purge, feature cut-off, entry rule and labels are inherited unchanged from
     panel v3. Only the relation edge is replaced.
  9. A broker with no usable 1-5 rating (missing, or the 1,523 zero-valued rows) does not enter
     the weighted strength but still counts as coverage.
 10. The 14 approved duplicate groups are resolved by keeping the latest activation date.

An eleventh output is added by the implementation, not requested: `jaccard_named_only`, computed
over named brokers only. The masked tokens are probably real broker ids but one of them shows an
unstable coverage footprint, so whether masking injects noise is measured instead of argued.

No source file is modified. No test-period target, prediction or metric is produced here; the
build covers every split because the graph is an input, not a result.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent
COVERAGE_RUN = ROOT / "data" / "raw" / "broker_coverage_v1" / "20260909T085010336967Z"
PANEL = ROOT / "data" / "panel_v3" / "20260909T035245922341Z"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
OUT_ROOT = ROOT / "data" / "analyst_graph_v1"
BROKER_COLUMN = "Broker Name"
REC_COLUMN = "Standard Rec (1-5) - Broker Estimate"
DATE_COLUMN = "Activation Date"
MASK_PREFIX = "Permission Denied"
MIN_COMMON_BROKERS = 3
MAX_SNAPSHOT_AGE_DAYS = 120
MAX_RECEIVERS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_coverage(plan: dict) -> pd.DataFrame:
    frames = []
    for batch in plan["batches"]:
        frame = pd.read_csv(COVERAGE_RUN / f"{batch['batch_id']}.csv")
        frame["snapshot"] = batch["snapshot"]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data[BROKER_COLUMN].notna()].copy()
    data["activation"] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data["rating"] = pd.to_numeric(data[REC_COLUMN], errors="coerce")
    # Rule 9: zero is outside the documented 1-5 domain and is treated as "no usable rating".
    data.loc[~data.rating.between(1, 5), "rating"] = np.nan
    # Rule 10: keep the latest activation within each duplicate group.
    before = len(data)
    data = (data.sort_values("activation")
                .drop_duplicates(["snapshot", "Instrument", BROKER_COLUMN], keep="last"))
    data["masked"] = data[BROKER_COLUMN].str.startswith(MASK_PREFIX, na=False)
    return data.reset_index(drop=True), before - len(data)


def snapshot_matrices(frame: pd.DataFrame):
    """Return the instrument index and the sparse coverage / rating-indicator matrices."""
    instruments = pd.Index(sorted(frame.Instrument.unique()))
    brokers = pd.Index(sorted(frame[BROKER_COLUMN].unique()))
    rows = instruments.get_indexer(frame.Instrument)
    columns = brokers.get_indexer(frame[BROKER_COLUMN])
    shape = (len(instruments), len(brokers))
    coverage = sparse.csr_matrix((np.ones(len(frame)), (rows, columns)), shape=shape)
    coverage.data[:] = 1.0
    named_mask = ~frame.masked.to_numpy()
    named = sparse.csr_matrix((np.ones(int(named_mask.sum())),
                               (rows[named_mask], columns[named_mask])), shape=shape)
    if named.nnz:
        named.data[:] = 1.0
    rated = {}
    for value in (1, 2, 3, 4, 5):
        selector = (frame.rating.to_numpy() == value)
        if selector.any():
            matrix = sparse.csr_matrix((np.ones(int(selector.sum())),
                                        (rows[selector], columns[selector])), shape=shape)
            matrix.data[:] = 1.0
            rated[value] = matrix
    return instruments, coverage, named, rated


def jaccard(common: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    union = sizes[:, None] + sizes[None, :] - common
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(union > 0, common / union, 0.0)
    return result


def build_snapshot_graph(frame: pd.DataFrame):
    instruments, coverage, named, rated = snapshot_matrices(frame)
    common = np.asarray((coverage @ coverage.T).todense())
    sizes = np.asarray(coverage.sum(axis=1)).ravel()
    coverage_jaccard = jaccard(common, sizes)

    named_common = np.asarray((named @ named.T).todense()) if named.nnz else np.zeros_like(common)
    named_sizes = np.asarray(named.sum(axis=1)).ravel()
    named_jaccard = jaccard(named_common, named_sizes)

    # Rating agreement: sum over common rated brokers of (1 - |r_i - r_j| / 4), expanded over the
    # five discrete rating values so it stays a handful of sparse products.
    agreement = np.zeros_like(common)
    rated_common = np.zeros_like(common)
    for value_i, matrix_i in rated.items():
        for value_j, matrix_j in rated.items():
            product = np.asarray((matrix_i @ matrix_j.T).todense())
            agreement += product * (1.0 - abs(value_i - value_j) / 4.0)
            rated_common += product
    union = sizes[:, None] + sizes[None, :] - common
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_jaccard = np.where(union > 0, agreement / union, 0.0)
    return instruments, common, coverage_jaccard, weighted_jaccard, named_jaccard, rated_common


def main() -> int:
    plan = json.loads((COVERAGE_RUN / "plan.json").read_text(encoding="utf-8"))
    coverage_data, duplicates_removed = load_coverage(plan)
    events = pd.read_csv(PANEL / "events.csv", low_memory=False)
    events["announcement_day"] = pd.to_datetime(events.announcement_day).dt.normalize()
    spans = pd.read_csv(SPANS_PATH)
    spans["member_from"] = pd.to_datetime(spans.member_from)
    spans["member_to"] = pd.to_datetime(spans.member_to)

    snapshots = pd.DatetimeIndex(sorted(pd.to_datetime(coverage_data.snapshot.unique())))
    graphs = {}
    for snapshot in snapshots:
        key = snapshot.strftime("%Y-%m-%d")
        graphs[key] = build_snapshot_graph(coverage_data.loc[coverage_data.snapshot.eq(key)])

    # Assign each event the latest snapshot strictly before its announcement day (rules 4-6).
    position = snapshots.searchsorted(events.announcement_day.to_numpy(), side="left") - 1
    events["snapshot"] = [snapshots[index].strftime("%Y-%m-%d") if index >= 0 else None
                          for index in position]
    events["snapshot_age_days"] = [
        (day - snapshots[index]).days if index >= 0 else np.nan
        for day, index in zip(events.announcement_day, position)]
    events["graph_status"] = "ok"
    events.loc[events.snapshot.isna(), "graph_status"] = "before_first_snapshot"
    events.loc[events.snapshot.notna() & events.snapshot_age_days.gt(MAX_SNAPSHOT_AGE_DAYS),
               "graph_status"] = "snapshot_too_stale"

    edges = []
    for snapshot, group in events.loc[events.graph_status.eq("ok")].groupby("snapshot"):
        instruments, common, coverage_jaccard, weighted_jaccard, named_jaccard, rated_common = graphs[snapshot]
        stamp = pd.Timestamp(snapshot)
        members = spans.loc[spans.member_from.le(stamp) & spans.member_to.ge(stamp), "ric"]
        eligible = instruments.isin(set(members))
        for row in group.itertuples(index=False):
            source_position = instruments.get_indexer([row.source])[0]
            if source_position < 0:
                edges.append({"source": row.source, "announcement": row.announcement,
                              "period_end": row.period_end, "receiver": None,
                              "reason": "source_absent_from_snapshot"})
                continue
            common_row = common[source_position]
            candidates = eligible & (common_row >= MIN_COMMON_BROKERS)
            candidates[source_position] = False
            if not candidates.any():
                edges.append({"source": row.source, "announcement": row.announcement,
                              "period_end": row.period_end, "receiver": None,
                              "reason": "no_candidate_meets_min_common_brokers"})
                continue
            strength = np.where(candidates, coverage_jaccard[source_position], -np.inf)
            chosen = np.argsort(-strength, kind="stable")[:MAX_RECEIVERS]
            chosen = [index for index in chosen if candidates[index]]
            for index in chosen:
                edges.append({
                    "source": row.source,
                    "announcement": row.announcement,
                    "period_end": row.period_end,
                    "receiver": instruments[index],
                    "snapshot": snapshot,
                    "snapshot_age_days": row.snapshot_age_days,
                    "common_brokers": int(common_row[index]),
                    "jaccard_coverage": float(coverage_jaccard[source_position, index]),
                    "jaccard_rec_weighted": float(weighted_jaccard[source_position, index]),
                    "jaccard_named_only": float(named_jaccard[source_position, index]),
                    "common_rated_brokers": int(rated_common[source_position, index]),
                    "reason": "ok",
                })

    edge_frame = pd.DataFrame(edges)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    edge_frame.to_csv(out / "edges.csv", index=False)
    events[["source", "announcement", "period_end", "announcement_day", "snapshot",
            "snapshot_age_days", "graph_status"]].to_csv(out / "event_graph_status.csv", index=False)

    built = edge_frame.loc[edge_frame.reason.eq("ok")]
    summary = {
        "run_id": run_id,
        "coverage_run": COVERAGE_RUN.name,
        "panel_run": PANEL.name,
        "rules": {
            "min_common_brokers": MIN_COMMON_BROKERS,
            "max_snapshot_age_days": MAX_SNAPSHOT_AGE_DAYS,
            "max_receivers": MAX_RECEIVERS,
            "broker_identity": "raw token, masked tokens used verbatim",
            "rating_domain": "1-5; zero and missing treated as no usable rating",
            "duplicate_resolution": "keep latest activation date",
            "selection_strength": "jaccard_coverage",
        },
        "duplicate_rows_removed": int(duplicates_removed),
        "events_total": int(len(events)),
        "events_by_graph_status": events.graph_status.value_counts().to_dict(),
        "events_with_edges": int(built.groupby(["source", "announcement", "period_end"]).ngroups),
        "edges_built": int(len(built)),
        "edges_unbuilt_reasons": edge_frame.loc[edge_frame.reason.ne("ok"), "reason"]
                                            .value_counts().to_dict(),
        "receivers_per_event_mean": float(
            built.groupby(["source", "announcement", "period_end"]).size().mean()) if len(built) else 0.0,
        "strength_profile": {
            column: {"mean": float(built[column].mean()), "median": float(built[column].median()),
                     "min": float(built[column].min()), "max": float(built[column].max())}
            for column in ["jaccard_coverage", "jaccard_rec_weighted", "jaccard_named_only",
                           "common_brokers", "common_rated_brokers"]} if len(built) else {},
        "snapshot_age_profile": {
            "mean": float(events.loc[events.graph_status.eq("ok"), "snapshot_age_days"].mean()),
            "max": float(events.loc[events.graph_status.eq("ok"), "snapshot_age_days"].max())},
        "inputs": {"plan_sha256": sha256_file(COVERAGE_RUN / "plan.json"),
                   "events_sha256": sha256_file(PANEL / "events.csv"),
                   "spans_sha256": sha256_file(SPANS_PATH),
                   "code_sha256": sha256_file(Path(__file__).resolve())},
        "outputs": {name: sha256_file(out / name) for name in ["edges.csv", "event_graph_status.csv"]},
        "note": "Graph construction only. No model, portfolio, backtest or test-period result.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ["run_id", "duplicate_rows_removed", "events_total", "events_by_graph_status",
                       "events_with_edges", "edges_built", "edges_unbuilt_reasons",
                       "receivers_per_event_mean", "strength_profile", "snapshot_age_profile"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
