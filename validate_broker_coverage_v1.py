"""Independent completeness and point-in-time validation of the broker coverage collection.

Checks the collected run against its own plan without trusting the collector's bookkeeping:
every planned batch is present, request identity and file hashes verify, no row carries a
recommendation activated after its snapshot date, and the coverage/masking profile is measured
rather than assumed. Writes an audit directory and fails loudly if any check fails.

Usage: validate_broker_coverage_v1.py [run_id]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw" / "broker_coverage_v1"
AUDIT_ROOT = ROOT / "data" / "audit" / "broker_coverage_v1"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
MASK_PREFIX = "Permission Denied"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_hash(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()


def latest_run() -> Path:
    runs = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"no run under {RAW_ROOT}")
    return runs[-1]


def main() -> int:
    run = RAW_ROOT / sys.argv[1] if len(sys.argv) > 1 else latest_run()
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    batches = plan["batches"]
    checks: dict[str, bool] = {}

    present, identity_ok, hash_ok, meta_rows_ok = [], [], [], []
    frames = []
    for batch in batches:
        csv_path = run / f"{batch['batch_id']}.csv"
        meta_path = run / f"{batch['batch_id']}.metadata.json"
        if not (csv_path.exists() and meta_path.exists()):
            present.append(batch["batch_id"])
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        identity_ok.append(meta.get("request_sha256") == batch["request_sha256"] ==
                           request_hash(batch["request"]))
        hash_ok.append(meta.get("csv_sha256") == sha256_file(csv_path))
        frame = pd.read_csv(csv_path)
        meta_rows_ok.append(int(meta.get("rows", -1)) == len(frame))
        frame["snapshot"] = batch["snapshot"]
        frame["batch_id"] = batch["batch_id"]
        frame["instruments_requested"] = batch["instrument_count"]
        frames.append(frame)

    checks["all_planned_batches_present"] = not present
    checks["request_identity_verified"] = bool(identity_ok) and all(identity_ok)
    checks["csv_hashes_verified"] = bool(hash_ok) and all(hash_ok)
    checks["metadata_row_counts_match_files"] = bool(meta_rows_ok) and all(meta_rows_ok)
    # A preserved failure is evidence, not a defect. What matters is that every failed attempt was
    # subsequently resolved by a verified batch, so the requirement is resolution, not absence.
    sidecars = sorted(path.stem.replace(".error", "") for path in run.glob("*.error.json"))
    verified_ids = {batch["batch_id"] for batch in batches
                    if (run / f"{batch['batch_id']}.metadata.json").exists()}
    unresolved = [name for name in sidecars if name not in verified_ids]
    checks["all_preserved_failures_were_resolved"] = not unresolved
    if not frames:
        raise RuntimeError("no batch files could be read")

    data = pd.concat(frames, ignore_index=True)
    # The recommendation column is "Standard Rec (1-5) - Broker Estimate", which also contains
    # "Broker"; match each column exactly rather than by a substring that hits both.
    broker_column = "Broker Name"
    rec_column = next((column for column in data.columns if "Rec (1-5)" in column), None)
    date_column = next((column for column in data.columns if column == "Activation Date"), None)
    data["snapshot_ts"] = pd.to_datetime(data.snapshot)

    checks["expected_snapshot_count"] = data.snapshot.nunique() == plan["snapshot_count"]
    checks["snapshots_match_plan"] = set(data.snapshot) <= {batch["snapshot"] for batch in batches}

    # Point-in-time discipline: no recommendation may be activated after its snapshot date.
    if date_column:
        activation = pd.to_datetime(data[date_column], errors="coerce")
        future = activation.notna() & (activation.dt.normalize() > data.snapshot_ts)
        checks["no_recommendation_activated_after_snapshot"] = not bool(future.any())
        future_rows = int(future.sum())
    else:
        checks["no_recommendation_activated_after_snapshot"] = False
        future_rows = -1

    covered = data.loc[data[broker_column].notna()]
    duplicates = covered.duplicated(["snapshot", "Instrument", broker_column]).sum()
    # Raw vendor rows legitimately contain a small, known set of duplicates: 14 groups, all under
    # the single token "Permission Denied 47272", each holding two activation dates with no rating.
    # Deduplication (keep the latest activation, approved 2026-09-09) belongs to the graph build,
    # not to the collection layer, so this check pins the known profile instead of demanding zero.
    # Any change in the count or in the number of tokens involved fails and needs a fresh decision.
    duplicate_rows = covered[covered.duplicated(["snapshot", "Instrument", broker_column], keep=False)]
    duplicate_tokens = int(duplicate_rows[broker_column].nunique())
    checks["duplicates_match_approved_profile"] = bool(duplicates == 14 and duplicate_tokens == 1)

    masked = covered[broker_column].str.startswith(MASK_PREFIX, na=False)
    values = pd.to_numeric(covered[rec_column], errors="coerce") if rec_column else pd.Series(dtype=float)
    out_of_range = int((values.notna() & ~values.between(1, 5)).sum())
    zero_valued = int(values.eq(0).sum())
    checks["recommendation_values_within_1_to_5_or_zero"] = bool(out_of_range == zero_valued)

    # Membership: every returned instrument must have been an index member on its snapshot date.
    spans = pd.read_csv(SPANS_PATH)
    spans["member_from"] = pd.to_datetime(spans.member_from)
    spans["member_to"] = pd.to_datetime(spans.member_to)
    pairs = data[["Instrument", "snapshot_ts"]].drop_duplicates()
    merged = pairs.merge(spans[["ric", "member_from", "member_to"]], left_on="Instrument",
                         right_on="ric", how="left")
    member_ok = merged.member_from.le(merged.snapshot_ts) & merged.member_to.ge(merged.snapshot_ts)
    checks["all_returned_instruments_were_members"] = bool(member_ok.fillna(False).all())

    # instruments_requested is a per-batch constant carried on every row, so it must be summed
    # once per batch from the plan, never summed across rows.
    requested = pd.DataFrame([(batch["snapshot"], batch["instrument_count"]) for batch in batches],
                             columns=["snapshot", "n"]).groupby("snapshot").n.sum()
    per_snapshot = data.groupby("snapshot").agg(
        rows=("Instrument", "size"),
        instruments_returned=("Instrument", "nunique"),
    )
    per_snapshot["instruments_requested"] = requested
    per_snapshot["instruments_with_coverage"] = covered.groupby("snapshot").Instrument.nunique()
    per_snapshot["coverage_rate"] = (per_snapshot.instruments_with_coverage /
                                     per_snapshot.instruments_requested).round(4)
    per_snapshot["brokers"] = covered.groupby("snapshot")[broker_column].nunique()

    checks["every_snapshot_has_coverage"] = bool((per_snapshot.instruments_with_coverage > 0).all())
    checks["coverage_rate_above_half_everywhere"] = bool((per_snapshot.coverage_rate > 0.5).all())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = AUDIT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    per_snapshot.to_csv(out / "per_snapshot_profile.csv")
    report = {
        "run_id": run_id,
        "collection_run": run.name,
        "validator": "validate_broker_coverage_v1.py",
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "all_passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "missing_batches": present[:20],
        "preserved_failures": sidecars,
        "unresolved_failures": unresolved,
        "profile": {
            "batches": len(batches),
            "total_rows": int(len(data)),
            "rows_with_broker": int(len(covered)),
            "distinct_snapshots": int(data.snapshot.nunique()),
            "distinct_instruments": int(data.Instrument.nunique()),
            "distinct_broker_tokens": int(covered[broker_column].nunique()),
            "masked_row_share": round(float(masked.mean()), 4),
            "masked_token_count": int(covered.loc[masked, broker_column].nunique()),
            "named_token_count": int(covered.loc[~masked, broker_column].nunique()),
            "instruments_without_any_coverage_row": int(
                data.loc[data[broker_column].isna(), "Instrument"].nunique()),
            "rows_activated_after_snapshot": future_rows,
            "duplicate_rows": int(duplicates),
            "duplicate_tokens_involved": duplicate_tokens,
            "recommendation_missing_rows": int(values.isna().sum()),
            "recommendation_zero_rows": zero_valued,
            "recommendation_out_of_range_rows": out_of_range,
            "coverage_rate_min": float(per_snapshot.coverage_rate.min()),
            "coverage_rate_median": float(per_snapshot.coverage_rate.median()),
            "brokers_per_snapshot_median": float(per_snapshot.brokers.median()),
        },
        "plan_sha256": sha256_file(run / "plan.json"),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "test_policy": "Collection only; no graph, model, portfolio or test-period result produced.",
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["run_id", "collection_run", "passed", "total", "all_passed"]},
                     indent=2))
    print(json.dumps(report["checks"], indent=2))
    print(json.dumps(report["profile"], indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])
    return 0


if __name__ == "__main__":
    sys.exit(main())
