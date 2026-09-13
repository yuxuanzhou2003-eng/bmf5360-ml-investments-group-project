"""Independent structural and sampled value validation for analyst_graph_v2."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / "data/analyst_graph_v2/20260909T113109814872Z"
COVERAGE = ROOT / "data/raw/broker_coverage_v1/20260909T085010336967Z"
PANEL = ROOT / "data/panel_v3/20260909T035245922341Z"
SPANS = ROOT / "data/audit/universe_rebuild/eligible_spans_2015_2026_corrected.csv"
OUT_ROOT = ROOT / "data/audit/analyst_graph_v2"
KEY = ["source", "announcement", "period_end"]
BROKER = "Broker Name"
RATING = "Standard Rec (1-5) - Broker Estimate"
ACTIVATION = "Activation Date"
MASK_PREFIX = "Permission Denied"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bools(values):
    normalized = pd.Series(values).astype("string").str.lower()
    if normalized.isna().any() or set(normalized.unique()) - {"true", "false"}:
        raise ValueError("invalid boolean values")
    return normalized.eq("true")


def load_coverage(plan):
    frames, manifest = [], {}
    for batch in plan["batches"]:
        csv_path = COVERAGE / f"{batch['batch_id']}.csv"
        observed = sha256(csv_path)
        metadata = json.loads((COVERAGE / f"{batch['batch_id']}.metadata.json").read_text(encoding="utf-8"))
        if observed != metadata["csv_sha256"] or metadata["request_sha256"] != batch["request_sha256"]:
            raise RuntimeError(f"coverage identity failure: {batch['batch_id']}")
        manifest[batch["batch_id"]] = observed
        frame = pd.read_csv(csv_path)
        frame["snapshot"] = batch["snapshot"]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data[BROKER].notna()].copy()
    data["activation"] = pd.to_datetime(data[ACTIVATION], errors="coerce")
    data["rating"] = pd.to_numeric(data[RATING], errors="coerce")
    data.loc[~data.rating.between(1, 5), "rating"] = np.nan
    data = data.sort_values("activation").drop_duplicates(
        ["snapshot", "Instrument", BROKER], keep="last")
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return data, manifest_hash


def expected_status(events, snapshots):
    position = snapshots.searchsorted(events.announcement_day.to_numpy(), side="left") - 1
    snap = pd.Series([snapshots[i] if i >= 0 else pd.NaT for i in position], index=events.index)
    age = (events.announcement_day - snap).dt.days
    status = pd.Series("ok", index=events.index)
    status.loc[~events.source_in_index_on_announcement] = "source_not_member_on_announcement"
    status.loc[events.source_in_index_on_announcement & snap.isna()] = "before_first_snapshot"
    status.loc[events.source_in_index_on_announcement & snap.notna() & age.gt(120)] = "snapshot_too_stale"
    return snap, age, status


def direct_event(event, coverage_by_snapshot, spans):
    snapshot = event.snapshot
    block = coverage_by_snapshot[snapshot]
    maps = {}
    for instrument, group in block.groupby("Instrument"):
        broker_set = set(group[BROKER])
        named = {token for token in broker_set if not str(token).startswith(MASK_PREFIX)}
        ratings = dict(zip(group.loc[group.rating.notna(), BROKER], group.loc[group.rating.notna(), "rating"]))
        maps[instrument] = (broker_set, named, ratings)
    if event.source not in maps:
        return "source_absent_from_snapshot", []
    members = set(spans.loc[spans.member_from.le(event.announcement_day) &
                            spans.member_to.ge(event.announcement_day), "ric"])
    source_brokers, source_named, source_ratings = maps[event.source]
    candidates = []
    for receiver in sorted(set(maps) & members - {event.source}):
        receiver_brokers, receiver_named, receiver_ratings = maps[receiver]
        common_set = source_brokers & receiver_brokers
        common = len(common_set)
        if common < 3:
            continue
        union = len(source_brokers | receiver_brokers)
        common_named = len(source_named & receiver_named)
        named_union = len(source_named | receiver_named)
        agreement, rated_common = 0.0, 0
        for broker in common_set:
            if broker in source_ratings and broker in receiver_ratings:
                agreement += 1.0 - abs(source_ratings[broker] - receiver_ratings[broker]) / 4.0
                rated_common += 1
        candidates.append({
            "receiver": receiver,
            "common_brokers": common,
            "jaccard_coverage": common / union,
            "jaccard_rec_weighted": agreement / union,
            "jaccard_named_only": common_named / named_union if named_union else 0.0,
            "common_rated_brokers": rated_common,
        })
    if not candidates:
        return "no_candidate_meets_min_common_brokers", []
    candidates.sort(key=lambda row: (-row["jaccard_coverage"], row["receiver"]))
    return "ok", candidates[:5]


def main():
    summary = json.loads((GRAPH / "summary.json").read_text(encoding="utf-8"))
    edge = pd.read_csv(GRAPH / "edges.csv")
    status = pd.read_csv(GRAPH / "event_graph_status.csv")
    events = pd.read_csv(PANEL / "events.csv", low_memory=False)
    events["announcement_day"] = pd.to_datetime(events.announcement_day).dt.normalize()
    events["source_in_index_on_announcement"] = bools(events.source_in_index_on_announcement).to_numpy()
    status["announcement_day"] = pd.to_datetime(status.announcement_day).dt.normalize()
    status["snapshot_ts"] = pd.to_datetime(status.snapshot)
    status["source_in_index_on_announcement"] = bools(status.source_in_index_on_announcement).to_numpy()
    spans = pd.read_csv(SPANS)
    spans["member_from"] = pd.to_datetime(spans.member_from)
    spans["member_to"] = pd.to_datetime(spans.member_to)
    plan = json.loads((COVERAGE / "plan.json").read_text(encoding="utf-8"))
    coverage, manifest_hash = load_coverage(plan)
    snapshots = pd.DatetimeIndex(sorted(pd.to_datetime(coverage.snapshot.unique())))

    expected_snap, expected_age, expected_graph_status = expected_status(events, snapshots)
    joined = status.merge(events[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one")
    built = edge.loc[edge.reason.eq("ok")].copy()
    nonbuilt = edge.loc[edge.reason.ne("ok")].copy()
    edge_events = edge[KEY].astype(str).agg("|".join, axis=1)
    counts = built.groupby(KEY).size()
    built_with_event = built.merge(events[KEY + ["announcement_day", "source_in_index_on_announcement"]],
                                   on=KEY, validate="many_to_one")
    built_with_membership = built_with_event.merge(
        spans[["ric", "member_from", "member_to"]], left_on="receiver", right_on="ric", how="left")
    receiver_member = (built_with_membership.member_from.le(built_with_membership.announcement_day) &
                       built_with_membership.member_to.ge(built_with_membership.announcement_day))

    status_expected_frame = events[KEY].copy()
    status_expected_frame["expected_snapshot"] = expected_snap.dt.strftime("%Y-%m-%d")
    status_expected_frame["expected_age"] = expected_age
    status_expected_frame["expected_status"] = expected_graph_status
    observed_status = status.merge(status_expected_frame, on=KEY, validate="one_to_one")
    snapshot_match = observed_status.snapshot.fillna("").eq(observed_status.expected_snapshot.fillna(""))
    age_match = np.isclose(observed_status.snapshot_age_days, observed_status.expected_age,
                           equal_nan=True)

    graph_ok_keys = set(status.loc[status.graph_status.eq("ok"), KEY].astype(str).agg("|".join, axis=1))
    output_keys = set(edge_events)
    nonok_output_keys = set(status.loc[status.graph_status.ne("ok"), KEY].astype(str).agg("|".join, axis=1))

    # Independently recompute every no-edge event and a deterministic sample of 512 built events.
    built_events = status.loc[status.graph_status.eq("ok") &
                              status[KEY].astype(str).agg("|".join, axis=1).isin(output_keys - set(
                                  nonbuilt[KEY].astype(str).agg("|".join, axis=1)))]
    sampled = built_events.sample(n=min(512, len(built_events)), random_state=5360)
    no_edge_events = status.loc[status[KEY].astype(str).agg("|".join, axis=1).isin(set(
        nonbuilt[KEY].astype(str).agg("|".join, axis=1)))]
    review = pd.concat([sampled, no_edge_events], ignore_index=True).drop_duplicates(KEY)
    coverage_by_snapshot = {key: group for key, group in coverage.groupby("snapshot")}
    sampled_selection_ok, sampled_values_ok, no_edge_reason_ok = [], [], []
    edge_groups = {key: group for key, group in edge.groupby(KEY, sort=False)}
    for row in review.itertuples(index=False):
        expected_reason, expected_rows = direct_event(row, coverage_by_snapshot, spans)
        observed = edge_groups[(row.source, row.announcement, row.period_end)]
        if expected_reason != "ok":
            no_edge_reason_ok.append(len(observed) == 1 and observed.iloc[0].reason == expected_reason)
            continue
        observed = observed.loc[observed.reason.eq("ok")]
        sampled_selection_ok.append(observed.receiver.tolist() == [item["receiver"] for item in expected_rows])
        if len(observed) == len(expected_rows):
            expected_table = pd.DataFrame(expected_rows).set_index("receiver").loc[observed.receiver]
            exact_counts = (observed.common_brokers.to_numpy() == expected_table.common_brokers.to_numpy()).all()
            exact_rated = (observed.common_rated_brokers.to_numpy() ==
                           expected_table.common_rated_brokers.to_numpy()).all()
            numeric = all(np.allclose(observed[column], expected_table[column], rtol=0, atol=1e-12)
                          for column in ["jaccard_coverage", "jaccard_rec_weighted", "jaccard_named_only"])
            sampled_values_ok.append(bool(exact_counts and exact_rated and numeric))
        else:
            sampled_values_ok.append(False)

    checks = {
        "summary_output_hash_edges": summary["outputs"]["edges.csv"] == sha256(GRAPH / "edges.csv"),
        "summary_output_hash_status": summary["outputs"]["event_graph_status.csv"] == sha256(GRAPH / "event_graph_status.csv"),
        "coverage_manifest_hash": summary["inputs"]["coverage_csv_manifest_sha256"] == manifest_hash,
        "event_status_exact_set": bool(joined._merge.eq("both").all()),
        "event_status_unique": not status.duplicated(KEY).any(),
        "snapshot_strictly_before_announcement": bool(status.loc[status.snapshot_ts.notna(), "snapshot_ts"].lt(
            status.loc[status.snapshot_ts.notna(), "announcement_day"]).all()),
        "snapshot_assignment_recomputed": bool(snapshot_match.all()),
        "snapshot_age_recomputed": bool(age_match.all()),
        "graph_status_recomputed": bool(observed_status.graph_status.eq(observed_status.expected_status).all()),
        "source_membership_on_all_built_edges": bool(built_with_event.source_in_index_on_announcement.all()),
        "receiver_membership_on_all_built_edges": bool(receiver_member.fillna(False).all()),
        "no_self_edges": bool(built.source.ne(built.receiver).all()),
        "edge_keys_unique": not built.duplicated(KEY + ["receiver"]).any(),
        "maximum_five_receivers": bool(counts.le(5).all()),
        "minimum_three_common_brokers": bool(built.common_brokers.ge(3).all()),
        "nonbuilt_rows_have_missing_receiver": bool(nonbuilt.receiver.isna().all()),
        "only_graph_ok_events_have_output_rows": not bool(nonok_output_keys & output_keys),
        "every_graph_ok_event_has_output_or_reason": graph_ok_keys == output_keys,
        "sampled_receiver_selection_recomputed": bool(sampled_selection_ok) and all(sampled_selection_ok),
        "sampled_edge_values_recomputed": bool(sampled_values_ok) and all(sampled_values_ok),
        "all_no_edge_reasons_recomputed": bool(no_edge_reason_ok) and all(no_edge_reason_ok),
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "run_id": run_id,
        "graph_run": GRAPH.name,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
        "rows": {"events": len(status), "edge_rows": len(edge), "built_edges": len(built),
                 "no_edge_reason_rows": len(nonbuilt), "direct_events_recomputed": len(review)},
        "input_hashes": {"edges": sha256(GRAPH / "edges.csv"),
                         "status": sha256(GRAPH / "event_graph_status.csv"),
                         "summary": sha256(GRAPH / "summary.json"),
                         "coverage_plan": sha256(COVERAGE / "plan.json"),
                         "panel_events": sha256(PANEL / "events.csv"),
                         "spans": sha256(SPANS), "validator": sha256(Path(__file__).resolve())},
        "test_policy": "Graph covariates for all splits were validated; no return target, prediction or metric was computed.",
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_passed"]:
        raise RuntimeError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    main()
