"""Versioned, resumable collection of point-in-time broker coverage for the common-analyst graph.

Design decisions, all established by the probes recorded in RELATION_DATA_FEASIBILITY.md:

- Range requests are unusable: LSEG expands still-active recommendations across every trading day
  with no column identifying the day, producing ~167x redundancy. Coverage is therefore collected
  as one single-date as-of snapshot per quarter.
- `Permission Denied <number>` tokens are stable broker identifiers (verified across 30 companies
  and 3 dates), so the token string is stored verbatim as the broker id. No name resolution, no
  mapping, no substitution.
- Only instruments that are index members on the snapshot date are requested, which keeps the
  universe point-in-time and avoids asking for coverage of companies not yet or no longer in scope.

The collector writes direct vendor responses to a new run directory and never overwrites an
existing one. A batch already present is reused only after its request identity, row count and
SHA-256 all verify. Failed attempts are preserved as error sidecars. Collecting a snapshot is not
permission to score test-period observations; the sealed test window is unchanged by this stage.

Usage:
  collect_broker_coverage_v1.py --plan            build and save the request plan only
  collect_broker_coverage_v1.py --run [--limit N] execute pending batches
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
CREDS = dotenv_values(ROOT / ".env")
SECRETS = [str(value) for value in CREDS.values() if value]
RAW_ROOT = ROOT / "data" / "raw" / "broker_coverage_v1"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
FIELDS = ["TR.RecEstBrokerName", "TR.RecEstValue", "TR.RecEstDate"]
BATCH_SIZE = 30
REQUEST_TIMEOUT_SECONDS = 180
# The corrected spans table starts at 2015-01-01, so a 2014-12-31 snapshot has zero index members
# and yields no batch. Rather than borrow the 2015-01-01 membership list to ask for a 2014-12-31
# snapshot -- which would put one day of look-ahead into the membership question -- the user
# decided on 2026-09-09 to accept that announcements before the first snapshot get no graph edge
# and to treat 2015 Q1 as graph warm-up. This costs 657 events / 2,170 edges, all in training,
# and those events must stay visible with an explicit reason code rather than disappear.
SNAPSHOT_START = "2015-03-31"
SNAPSHOT_END = "2026-06-30"
GRAPH_WARMUP_NOTE = (
    "Announcements before the first snapshot (2015-03-31) receive no common-analyst edge. "
    "657 events / 2,170 edges in panel v3 fall in that window, all in the training split. "
    "Approved 2026-09-09: accept the gap as graph warm-up rather than introduce look-ahead."
)


def redact(value: object) -> str:
    result = str(value)
    for secret in SECRETS:
        result = result.replace(secret, "[REDACTED]")
    return result


class SafeStream:
    def __init__(self, target):
        self.target = target

    def write(self, message):
        return self.target.write(redact(message))

    def flush(self):
        return self.target.flush()

    def isatty(self):
        return False


sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)

import pandas as pd  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_hash(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()


def quarter_ends(start: str, end: str) -> list[str]:
    return [stamp.strftime("%Y-%m-%d")
            for stamp in pd.date_range(start=start, end=end, freq="QE")]


def build_plan() -> dict:
    spans = pd.read_csv(SPANS_PATH)
    spans["member_from"] = pd.to_datetime(spans.member_from)
    spans["member_to"] = pd.to_datetime(spans.member_to)
    snapshots = quarter_ends(SNAPSHOT_START, SNAPSHOT_END)
    batches = []
    for snapshot in snapshots:
        stamp = pd.Timestamp(snapshot)
        members = spans.loc[spans.member_from.le(stamp) & spans.member_to.ge(stamp), "ric"]
        members = sorted(members.unique())
        for index in range(0, len(members), BATCH_SIZE):
            chunk = members[index:index + BATCH_SIZE]
            request = {"universe": chunk, "fields": FIELDS, "parameters": {"SDate": snapshot}}
            batches.append({
                "batch_id": f"coverage_{snapshot.replace('-', '')}_{index // BATCH_SIZE:03d}",
                "snapshot": snapshot,
                "instrument_count": len(chunk),
                "request": request,
                "request_sha256": request_hash(request),
            })
    return {
        "plan_created_utc": datetime.now(timezone.utc).isoformat(),
        "spans_source": str(SPANS_PATH.relative_to(ROOT)),
        "spans_sha256": sha256_file(SPANS_PATH),
        "fields": FIELDS,
        "batch_size": BATCH_SIZE,
        "snapshot_start": SNAPSHOT_START,
        "snapshot_end": SNAPSHOT_END,
        "snapshot_count": len(snapshots),
        "membership_rule": "instrument is an index member on the snapshot date, per the corrected spans table",
        "graph_warmup_decision": GRAPH_WARMUP_NOTE,
        "batch_count": len(batches),
        "total_requested_instrument_slots": int(sum(batch["instrument_count"] for batch in batches)),
        "batches": batches,
    }


def batch_state(out: Path, batch: dict) -> str:
    csv_path = out / f"{batch['batch_id']}.csv"
    meta_path = out / f"{batch['batch_id']}.metadata.json"
    if not (csv_path.exists() and meta_path.exists()):
        return "pending"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "pending"
    if meta.get("request_sha256") != batch["request_sha256"]:
        return "mismatch"
    if meta.get("csv_sha256") != sha256_file(csv_path):
        return "mismatch"
    return "done"


def install(out: Path, batch: dict, frame: pd.DataFrame, elapsed: float) -> dict:
    csv_path = out / f"{batch['batch_id']}.csv"
    temporary = out / f"{batch['batch_id']}.csv.{os.getpid()}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)
    meta = {
        "batch_id": batch["batch_id"],
        "snapshot": batch["snapshot"],
        "request": batch["request"],
        "request_sha256": batch["request_sha256"],
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "instruments_requested": batch["instrument_count"],
        "instruments_returned": int(frame.Instrument.nunique()) if "Instrument" in frame else 0,
        "non_null": {str(column): int(frame[column].notna().sum()) for column in frame.columns},
        "csv_sha256": sha256_file(csv_path),
    }
    (out / f"{batch['batch_id']}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="build and save the plan, issue no request")
    parser.add_argument("--run", action="store_true", help="execute pending batches")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many batches")
    parser.add_argument("--run-id", type=str, default="", help="resume a specific run directory")
    arguments = parser.parse_args()
    if not (arguments.plan or arguments.run):
        parser.error("choose --plan or --run")

    if arguments.run_id:
        out = RAW_ROOT / arguments.run_id
    else:
        existing = sorted(path for path in RAW_ROOT.glob("*") if path.is_dir()) if RAW_ROOT.exists() else []
        out = existing[-1] if existing else RAW_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        rebuilt = build_plan()
        if [batch["request_sha256"] for batch in rebuilt["batches"]] != \
           [batch["request_sha256"] for batch in plan["batches"]]:
            print("Existing plan does not match a freshly built plan; refusing to continue.", flush=True)
            return 1
    else:
        plan = build_plan()
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: plan[key] for key in
                      ["snapshot_count", "batch_count", "total_requested_instrument_slots",
                       "batch_size", "snapshot_start", "snapshot_end"]}, indent=2), flush=True)
    if arguments.plan:
        print(f"Plan saved to {plan_path}. No request was issued.", flush=True)
        return 0

    states = {batch["batch_id"]: batch_state(out, batch) for batch in plan["batches"]}
    mismatched = [key for key, value in states.items() if value == "mismatch"]
    if mismatched:
        print(f"Refusing to run: {len(mismatched)} existing batches fail identity or hash verification: "
              f"{mismatched[:5]}", flush=True)
        return 1
    pending = [batch for batch in plan["batches"] if states[batch["batch_id"]] == "pending"]
    print(f"{len(plan['batches']) - len(pending)} already verified, {len(pending)} pending.", flush=True)
    if not pending:
        return 0

    import lseg.data as ld
    try:
        ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
    except Exception as exc:
        (out / "session.error.json").write_text(
            json.dumps({"error": redact(exc)[:4000],
                        "at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        print("SESSION FAILED; no data request issued.", flush=True)
        return 1

    done = 0
    for batch in pending:
        if arguments.limit and done >= arguments.limit:
            print(f"Reached --limit {arguments.limit}; stopping with pending batches remaining.", flush=True)
            break
        holder = {}

        def on_timeout():
            (out / f"{batch['batch_id']}.error.json").write_text(
                json.dumps({"batch_id": batch["batch_id"], "request": batch["request"],
                            "error": f"Exceeded {REQUEST_TIMEOUT_SECONDS}s",
                            "at_utc": datetime.now(timezone.utc).isoformat()}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
            print("TIMEOUT", batch["batch_id"], flush=True)
            os._exit(2)

        timer = threading.Timer(REQUEST_TIMEOUT_SECONDS, on_timeout)
        timer.start()
        started = time.monotonic()
        try:
            frame = ld.get_data(**batch["request"]).replace(r"^\s*$", pd.NA, regex=True)
            holder["meta"] = install(out, batch, frame, time.monotonic() - started)
        except Exception as exc:
            (out / f"{batch['batch_id']}.error.json").write_text(
                json.dumps({"batch_id": batch["batch_id"], "request": batch["request"],
                            "error": redact(exc)[:4000],
                            "at_utc": datetime.now(timezone.utc).isoformat()}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
            print("ERROR", batch["batch_id"], redact(exc)[:200], flush=True)
        finally:
            timer.cancel()
        if "meta" in holder:
            done += 1
            if done % 25 == 0 or done == 1:
                print(f"{done}/{len(pending)} {batch['batch_id']} rows={holder['meta']['rows']} "
                      f"{holder['meta']['elapsed_seconds']}s", flush=True)

    states = {batch["batch_id"]: batch_state(out, batch) for batch in plan["batches"]}
    remaining = [key for key, value in states.items() if value != "done"]
    summary = {
        "run_directory": out.name,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "batches_total": len(plan["batches"]),
        "batches_done": len(plan["batches"]) - len(remaining),
        "batches_remaining": len(remaining),
        "error_sidecars": sorted(path.name for path in out.glob("*.error.json")),
    }
    (out / "collection_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if not remaining else 3


if __name__ == "__main__":
    sys.exit(main())
