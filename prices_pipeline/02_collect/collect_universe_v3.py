"""Versioned, resumable LSEG collection for the formal v3 research dataset.

The collector writes direct vendor responses only to a new run directory.  It
never overwrites v2.  Failed attempts remain as sidecars and successful files
are verified by request identity, row count, and SHA-256 before reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
CREDS = dotenv_values(ROOT / ".env")
SECRETS = [str(value) for value in CREDS.values() if value]
RAW_ROOT = ROOT / "data" / "raw" / "universe_v3"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
WARMUP_START = "2013-11-01"
STUDY_END = "2026-09-07"
FULL_ADJUSTMENTS = [
    "exchangeCorrection", "manualCorrection", "CCH", "CRE", "RPO", "RTS",
]
PRICE_FIELDS = [
    "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS",
    "BID", "ASK", "TRNOVR_UNS",
]


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
import lseg.data as ld  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    # A process-specific temporary name keeps sharded collectors from racing
    # over the same temporary summary path.  The final replace remains atomic.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for retry in range(40):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if retry == 39:
                raise
            time.sleep(0.05)


def groups_from_spans(spans: pd.DataFrame) -> list[tuple[str, list[str], str, str]]:
    work = spans.copy()
    work["fetch_end"] = pd.to_datetime(work["fetch_end"], errors="raise")
    live = work.loc[~work["delisted_ric"].astype(bool), "ric"].tolist()
    dead = work.loc[work["delisted_ric"].astype(bool)].copy()
    dead["end_year"] = dead["fetch_end"].dt.year
    groups = [("live", live, WARMUP_START, STUDY_END)]
    for year, subset in dead.groupby("end_year", sort=True):
        groups.append((
            f"dead{int(year)}",
            subset["ric"].tolist(),
            WARMUP_START,
            subset["fetch_end"].max().strftime("%Y-%m-%d"),
        ))
    return groups


def batched(values: list[str], size: int):
    for offset in range(0, len(values), size):
        yield offset // size, values[offset:offset + size]


def build_plan(spans: pd.DataFrame) -> list[dict]:
    plan: list[dict] = []
    groups = groups_from_spans(spans)
    specs = {
        "actuals": {
            "batch_size": 25,
            "fields": [
                "TR.EPSActValue(ActType=Reported).announcedate",
                "TR.EPSActValue(ActType=Reported).calcdate",
                "TR.EPSActValue(ActType=Reported).periodenddate",
                "TR.EPSActValue(ActType=Reported).fperiod",
                "TR.EPSActValue(ActType=Reported).currency",
                "TR.EPSActValue(ActType=Reported)",
            ],
            "parameters": {"Frq": "FQ", "Period": "FQ0"},
        },
        "estimates": {
            "batch_size": 10,
            "fields": [
                "TR.EPSMean.calcdate", "TR.EPSMean.periodenddate",
                "TR.EPSMean.fperiod", "TR.EPSMean.currency", "TR.EPSMean",
                "TR.EPSStdDev", "TR.EPSNumIncEstimates",
            ],
            "parameters": {"Frq": "D", "Period": "FQ1"},
        },
    }
    for table, spec in specs.items():
        for tag, rics, start, end in groups:
            for batch_number, batch in batched(rics, spec["batch_size"]):
                plan.append({
                    "name": f"{table}_{tag}_{batch_number:03d}",
                    "table": table,
                    "method": "get_data",
                    "request": {
                        "universe": batch,
                        "fields": spec["fields"],
                        "parameters": {
                            "SDate": start, "EDate": end, **spec["parameters"],
                        },
                    },
                })

    price_groups = [("benchmark", ["SPY.P"], WARMUP_START, STUDY_END), *groups]
    for tag, rics, start, end in price_groups:
        for batch_number, batch in batched(rics, 5):
            plan.append({
                "name": f"prices_{tag}_{batch_number:03d}",
                "table": "prices",
                "method": "get_history",
                "request": {
                    "universe": batch,
                    "fields": PRICE_FIELDS,
                    "interval": "1D",
                    "start": start,
                    "end": end,
                    "adjustments": FULL_ADJUSTMENTS,
                },
            })
    return plan


def validate_cached(out: Path, item: dict) -> bool:
    csv_path = out / f"{item['name']}.csv"
    meta_path = out / f"{item['name']}.meta.json"
    if not csv_path.exists() and not meta_path.exists():
        return False
    if not csv_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Incomplete cached pair: {item['name']}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("request") != item["request"]:
        raise RuntimeError(f"Cache request mismatch: {item['name']}")
    if meta.get("sha256") != sha256(csv_path):
        raise RuntimeError(f"Cache checksum mismatch: {item['name']}")
    if meta.get("status") not in {"returned", "empty"}:
        raise RuntimeError(f"Invalid cached status: {item['name']}")
    return True


def execute(out: Path, item: dict) -> dict:
    name = item["name"]
    started = time.monotonic()
    existing_attempts = sorted(out.glob(f"{name}.attempt_*.error.json"))
    attempt = len(existing_attempts) + 1
    record = {
        "name": name,
        "table": item["table"],
        "method": item["method"],
        "request": item["request"],
        "request_sha256": hashlib.sha256(
            json.dumps(item["request"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "attempt": attempt,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    timer = threading.Timer(300, lambda: os._exit(2))
    timer.start()
    try:
        if item["method"] == "get_data":
            frame = ld.get_data(**item["request"])
            include_index = False
        else:
            frame = ld.get_history(**item["request"])
            include_index = True
        csv_path = out / f"{name}.csv"
        temporary = out / f"{name}.csv.tmp"
        frame.to_csv(temporary, index=include_index)
        temporary.replace(csv_path)
        record.update({
            "status": "returned" if len(frame) else "empty",
            "rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "non_null": {str(column): int(frame[column].notna().sum()) for column in frame.columns},
            "sha256": sha256(csv_path),
            "bytes": csv_path.stat().st_size,
            "elapsed_s": round(time.monotonic() - started, 3),
        })
        atomic_json(out / f"{name}.meta.json", record)
        return record
    except Exception as exc:
        record.update({
            "status": "error",
            "error": redact(exc)[:2000],
            "elapsed_s": round(time.monotonic() - started, 3),
        })
        atomic_json(out / f"{name}.attempt_{attempt:02d}.error.json", record)
        return record
    finally:
        timer.cancel()


def write_summary(out: Path, plan: list[dict], run_id: str) -> dict:
    statuses = Counter()
    rows = Counter()
    bytes_by_table = Counter()
    for item in plan:
        meta_path = out / f"{item['name']}.meta.json"
        if not meta_path.exists():
            statuses[(item["table"], "pending")] += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        statuses[(item["table"], meta["status"])] += 1
        rows[item["table"]] += int(meta.get("rows", 0))
        bytes_by_table[item["table"]] += int(meta.get("bytes", 0))
    summary = {
        "run_id": run_id,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha256(out / "plan.json"),
        "by_table_status": {
            table: {status: count for (table_name, status), count in sorted(statuses.items()) if table_name == table}
            for table in ["actuals", "estimates", "prices"]
        },
        "rows_by_table": dict(rows),
        "bytes_by_table": dict(bytes_by_table),
        "policy": {
            "raw": "Direct vendor serialization only; no filling, filtering, deduplication, timezone conversion, or unit conversion.",
            "overwrite": "Existing successful CSV/meta pairs are verified and reused; never overwritten.",
            "failure": "Each failed attempt is retained in a numbered error sidecar.",
        },
    }
    atomic_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", nargs="?", choices=["actuals", "estimates", "prices", "all"], default="all")
    parser.add_argument("--run-id")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require shard_count >= 1 and 0 <= shard_index < shard_count")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = RAW_ROOT / run_id
    spans = pd.read_csv(SPANS_PATH)
    plan = build_plan(spans)
    if out.exists():
        saved = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        if saved["calls"] != plan:
            raise RuntimeError("Saved run plan differs from current generated plan")
    else:
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "plan.json", {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "spans_path": str(SPANS_PATH.relative_to(ROOT)),
            "spans_sha256": sha256(SPANS_PATH),
            "spans_rows": int(len(spans)),
            "excluded_from_v2_union": ["EVHC.N^L16"],
            "exclusion_reason": "Corrected membership audit removed an erroneous same-day join/leave interval; no old raw row was deleted.",
            "script": Path(__file__).name,
            "script_sha256_at_creation": sha256(Path(__file__)),
            "calls": plan,
        })
    selected_all = [item for item in plan if args.part == "all" or item["table"] == args.part]
    selected = [
        item for position, item in enumerate(selected_all)
        if position % args.shard_count == args.shard_index
    ]
    write_summary(out, plan, run_id)

    timer = threading.Timer(150, lambda: os._exit(2))
    timer.start()
    try:
        ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
    finally:
        timer.cancel()

    new_calls = 0
    errors = 0
    try:
        for index, item in enumerate(selected, start=1):
            if validate_cached(out, item):
                print(f"CACHE [{index}/{len(selected)}] {item['name']}", flush=True)
                continue
            if args.max_calls is not None and new_calls >= args.max_calls:
                break
            record = execute(out, item)
            new_calls += 1
            if record["status"] == "error":
                errors += 1
            print(
                f"{record['status'].upper()} [{index}/{len(selected)}] {item['name']} "
                f"rows={record.get('rows', 0)} seconds={record['elapsed_s']}",
                flush=True,
            )
            write_summary(out, plan, run_id)
    finally:
        ld.close_session()
    summary = write_summary(out, plan, run_id)
    print(json.dumps({"out": str(out), "new_calls": new_calls, "errors": errors, "summary": summary}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
