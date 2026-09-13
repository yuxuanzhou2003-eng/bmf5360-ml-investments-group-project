"""Recover only missing/unverified estimates batches from the v2 collector.

The script has two explicit phases:

1. ``--plan`` reconstructs the collector's 84 expected batches and writes a
   run-specific audit plan without opening an LSEG session.
2. ``--fetch PLAN`` executes only the non-verified batches in that plan. Each
   request runs in a short-lived worker process, is bounded by a timeout, and
   is attempted at most twice. Existing raw files and old ``.error.json``
   files are never overwritten.

The worker writes its response under the run audit directory. The parent then
copies a successful response byte-for-byte into the standard raw filename and
creates the matching metadata file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "universe_v2"
AUDIT_ROOT = ROOT / "data" / "audit" / "estimate_recovery"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "universe_spans_2015_2026.csv"
COLLECTOR_PATH = ROOT / "collect_universe_v2.py"

WARMUP_START = "2013-11-01"
STUDY_END = "2026-09-07"
BATCH_SIZE = 10
MAX_ATTEMPTS = 2
ATTEMPT_TIMEOUT_SECONDS = 150
FIELDS = [
    "TR.EPSMean.calcdate",
    "TR.EPSMean.periodenddate",
    "TR.EPSMean",
    "TR.EPSStdDev",
    "TR.EPSNumIncEstimates",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_hash(request: dict[str, Any]) -> str:
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def row_and_instrument_counts(path: Path) -> tuple[int, int | None]:
    """Count physical data rows and distinct Instrument values without edits."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(x) for x in (reader.fieldnames or [])]
        instrument_key = next((x for x in fieldnames if x.lower() == "instrument"), None)
        instruments: set[str] = set()
        rows = 0
        for record in reader:
            rows += 1
            if instrument_key is not None:
                value = record.get(instrument_key)
                if value not in (None, ""):
                    instruments.add(str(value))
        return rows, len(instruments) if instrument_key is not None else None


def read_groups() -> list[tuple[str, list[str], str, str]]:
    """Reproduce the live/dead grouping and file-order batching in the collector."""
    with SPANS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    live: list[str] = []
    dead_by_year: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        ric = str(row.get("ric", ""))
        if not ric:
            continue
        if bool_value(row.get("delisted_ric", "")):
            fetch_end = str(row.get("fetch_end", ""))
            year = int(fetch_end[:4])
            dead_by_year.setdefault(year, []).append(row)
        else:
            live.append(ric)

    groups: list[tuple[str, list[str], str, str]] = [("live", live, WARMUP_START, STUDY_END)]
    for year in sorted(dead_by_year):
        group = dead_by_year[year]
        fetch_end = max(str(row["fetch_end"]) for row in group)
        groups.append((f"dead{year}", [str(row["ric"]) for row in group], WARMUP_START, fetch_end))
    return groups


def build_expected() -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    global_index = 0
    for tag, rics, start_date, end_date in read_groups():
        for offset in range(0, len(rics), BATCH_SIZE):
            batch_index = offset // BATCH_SIZE
            request = {
                "universe": rics[offset : offset + BATCH_SIZE],
                "fields": FIELDS,
                "parameters": {
                    "SDate": start_date,
                    "EDate": end_date,
                    "Frq": "W",
                    "Period": "FQ1",
                },
            }
            expected.append(
                {
                    "global_index": global_index,
                    "tag": tag,
                    "batch_index": batch_index,
                    "name": f"estimates_{tag}_{batch_index:03d}",
                    "request": request,
                    "requested_instrument_count": len(request["universe"]),
                }
            )
            global_index += 1
    return expected


def classify_batch(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item["name"])
    csv_path = OUT / f"{name}.csv"
    meta_path = OUT / f"{name}.meta.json"
    error_path = OUT / f"{name}.error.json"
    result: dict[str, Any] = {
        "name": name,
        "state": "missing",
        "csv_path": rel(csv_path),
        "meta_path": rel(meta_path),
        "error_path": rel(error_path),
        "csv_exists": csv_path.exists(),
        "meta_exists": meta_path.exists(),
        "error_exists": error_path.exists(),
        "requested_instrument_count": item.get("requested_instrument_count"),
    }

    if csv_path.exists() and meta_path.exists():
        checks: list[str] = []
        try:
            metadata = read_json(meta_path)
            physical_rows, observed_instruments = row_and_instrument_counts(csv_path)
            actual_hash = short_hash(csv_path)
            result.update(
                {
                    "meta_status": metadata.get("status"),
                    "meta_rows": metadata.get("rows"),
                    "physical_rows": physical_rows,
                    "observed_instrument_count": observed_instruments,
                    "sha256": actual_hash,
                }
            )
            if metadata.get("status") != "success":
                checks.append("meta_status_not_success")
            if metadata.get("request") != item["request"]:
                checks.append("request_mismatch")
            if metadata.get("sha256") != actual_hash:
                checks.append("sha256_mismatch")
            if metadata.get("rows") != physical_rows:
                checks.append("row_count_mismatch")
            result["checks"] = checks
            result["state"] = "success_verified" if not checks else "unverified"
        except Exception as exc:  # the file is retained and reported as unverified
            result.update({"state": "unverified", "checks": ["metadata_or_csv_read_error"], "error": str(exc)[:400]})
    elif meta_path.exists():
        try:
            metadata = read_json(meta_path)
            result["meta_status"] = metadata.get("status")
            result["state"] = "empty_result" if metadata.get("status") == "empty" else "meta_only"
        except Exception as exc:
            result.update({"state": "unverified", "checks": ["metadata_read_error"], "error": str(exc)[:400]})
    elif csv_path.exists():
        result["state"] = "csv_only"
    elif error_path.exists():
        result["state"] = "error_placeholder"
    return result


def inventory(expected: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, int]]:
    states = {str(item["name"]): classify_batch(item) for item in expected}
    status_counts = dict(Counter(rec["state"] for rec in states.values()))
    totals = {
        "verified_success_rows": sum(int(rec.get("physical_rows", 0) or 0) for rec in states.values() if rec["state"] == "success_verified"),
        "verified_success_instruments": sum(int(rec.get("observed_instrument_count", 0) or 0) for rec in states.values() if rec["state"] == "success_verified"),
        "verified_success_batches": status_counts.get("success_verified", 0),
    }
    return states, status_counts, totals


def make_plan(run_id: str) -> dict[str, Any]:
    expected = build_expected()
    states, status_counts, totals = inventory(expected)
    target_states = {
        "error_placeholder",
        "missing",
        "unverified",
        "partial",
        "empty_result",
        "meta_only",
        "csv_only",
    }
    batches: list[dict[str, Any]] = []
    for item in expected:
        state = states[item["name"]]
        enriched = dict(item)
        enriched["current"] = state
        if state["state"] in target_states:
            batches.append(enriched)

    protected_errors = []
    for item in expected:
        error_path = OUT / f"{item['name']}.error.json"
        if error_path.exists():
            protected_errors.append(
                {
                    "path": rel(error_path),
                    "sha256": short_hash(error_path),
                    "bytes": error_path.stat().st_size,
                }
            )

    plan = {
        "stage": "recovery_plan",
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "script": rel(Path(__file__)),
        "script_sha256": short_hash(Path(__file__)),
        "collector": rel(COLLECTOR_PATH),
        "collector_sha256": short_hash(COLLECTOR_PATH),
        "universe_spans": rel(SPANS_PATH),
        "universe_spans_sha256": short_hash(SPANS_PATH),
        "raw_output_dir": rel(OUT),
        "rules": {
            "source": "collect_universe_v2.py grouping and request shape",
            "batch_size": BATCH_SIZE,
            "fields": FIELDS,
            "warmup_start": WARMUP_START,
            "live_end": STUDY_END,
            "periodicity": "W",
            "period": "FQ1",
            "retry_only_non_verified": True,
            "max_attempts_per_batch": MAX_ATTEMPTS,
            "attempt_timeout_seconds": ATTEMPT_TIMEOUT_SECONDS,
            "no_imputation": True,
            "no_deduplication": True,
            "no_row_or_security_removal": True,
            "old_error_files_preserved": True,
        },
        "expected_batch_count": len(expected),
        "expected_instrument_count": sum(int(item["requested_instrument_count"]) for item in expected),
        "current_status_counts": status_counts,
        "current_verified_totals": totals,
        "target_batch_count": len(batches),
        "target_instrument_count": sum(int(item["requested_instrument_count"]) for item in batches),
        "target_batch_names": [item["name"] for item in batches],
        "protected_error_files": protected_errors,
        "target_batches": batches,
    }
    return plan


def plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        f"# Estimate recovery plan — {plan['run_id']}",
        "",
        f"Created (UTC): `{plan['created_at_utc']}`",
        "",
        "This is a pre-fetch plan generated from the collector's existing grouping and request rules. It does not impute, deduplicate, delete, or exclude securities.",
        "",
        f"- Expected batches: **{plan['expected_batch_count']}**",
        f"- Expected requested instruments: **{plan['expected_instrument_count']}**",
        f"- Current verified successes: **{plan['current_status_counts'].get('success_verified', 0)} batches**; **{plan['current_verified_totals'].get('verified_success_rows', 0)} rows**; **{plan['current_verified_totals'].get('verified_success_instruments', 0)} observed instruments**",
        f"- Current errors/placeholders: **{plan['current_status_counts'].get('error_placeholder', 0)}**",
        f"- Recovery targets: **{plan['target_batch_count']} batches / {plan['target_instrument_count']} requested instruments**",
        "",
        "## Recovery targets",
        "",
        "| Batch | Requested instruments | SDate | EDate | Existing state | RICs |",
        "|---|---:|---|---|---|---|",
    ]
    for item in plan["target_batches"]:
        req = item["request"]
        state = item["current"]["state"]
        rics = ", ".join(req["universe"])
        lines.append(f"| `{item['name']}` | {len(req['universe'])} | {req['parameters']['SDate']} | {req['parameters']['EDate']} | `{state}` | {rics} |")
    lines.extend(
        [
            "",
            "## Protected old error files",
            "",
        ]
    )
    for record in plan["protected_error_files"]:
        lines.append(f"- `{record['path']}` — SHA-256 `{record['sha256']}`, {record['bytes']} bytes")
    lines.extend(
        [
            "",
            "The fetch phase will use the same fields and parameters recorded in the JSON plan, with a 150-second per-attempt timeout and at most two attempts per target. Any failure is written under the run-specific audit directory.",
            "",
        ]
    )
    return "\n".join(lines)


def load_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    if plan.get("stage") != "recovery_plan":
        raise ValueError(f"Not a recovery plan: {path}")
    if not plan.get("target_batches"):
        raise ValueError("Recovery plan has no target batches")
    return plan


def redact_factory() -> Any:
    credentials = dotenv_values(ROOT / ".env")
    secrets = [str(value) for value in credentials.values() if value]

    def redact(value: Any) -> str:
        text = str(value)
        for secret in secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    return redact


def worker_main(request_path: Path, result_path: Path) -> int:
    """Run one LSEG request in an isolated process and return a JSON result."""
    redact = redact_factory()
    credentials = dotenv_values(ROOT / ".env")
    request = read_json(request_path)
    csv_path = result_path.with_suffix(result_path.suffix + ".csv")
    record: dict[str, Any] = {
        "status": "running",
        "request_hash": request_hash(request),
        "started_at_utc": utc_now(),
    }
    started = time.monotonic()
    session_open = False
    try:
        # Imports are kept inside the worker so --plan remains a local-only stage.
        import lseg.data as ld  # type: ignore
        import pandas as pd  # type: ignore

        ld.open_session(name="desktop.workspace", app_key=credentials.get("LSEG_APP_KEY"))
        session_open = True
        frame = ld.get_data(**request)
        if frame is None or not len(frame):
            record.update(status="empty", rows=0, columns=[])
        else:
            frame = frame.replace(r"^\s*$", pd.NA, regex=True)
            temp_csv = csv_path.with_name(csv_path.name + ".tmp")
            frame.to_csv(temp_csv, index=False)
            os.replace(temp_csv, csv_path)
            record.update(
                status="success",
                rows=len(frame),
                columns=[str(column) for column in frame.columns],
                sha256=short_hash(csv_path),
                csv_path=rel(csv_path),
            )
    except Exception as exc:
        record.update(status="error", error=redact(exc)[:1200])
    finally:
        if session_open:
            try:
                ld.close_session()
            except Exception as exc:  # preserve the request result even if close fails
                record["close_error"] = redact(exc)[:400]
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(result_path, record)
    return 0 if record.get("status") in {"success", "empty"} else 1


def run_worker(request_path: Path, result_path: Path, timeout_seconds: int) -> dict[str, Any]:
    redact = redact_factory()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--request-file",
        str(request_path),
        "--result-file",
        str(result_path),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "error": f"Worker exceeded {timeout_seconds} seconds",
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "stdout_tail": redact(str(exc.stdout or ""))[-500:],
            "stderr_tail": redact(str(exc.stderr or ""))[-500:],
        }
    result: dict[str, Any]
    if result_path.exists():
        try:
            result = read_json(result_path)
        except Exception as exc:
            result = {"status": "worker_result_unreadable", "error": str(exc)[:500]}
    else:
        result = {"status": "worker_no_result", "error": "Worker exited without a result file"}
    result["worker_returncode"] = completed.returncode
    result["elapsed_seconds_parent"] = round(time.monotonic() - started, 2)
    # Keep diagnostics bounded and credential-redacted in the run report.
    if completed.stdout:
        result["stdout_tail"] = redact(completed.stdout)[-500:]
    if completed.stderr:
        result["stderr_tail"] = redact(completed.stderr)[-500:]
    return result


def install_success(name: str, request: dict[str, Any], result: dict[str, Any], run_id: str, attempt: int, run_dir: Path) -> dict[str, Any]:
    """Install one new raw file and metadata atomically without overwriting."""
    source = ROOT / str(result.get("csv_path", ""))
    # Worker reports a relative path; refuse unexpected paths.
    try:
        source.resolve().relative_to(run_dir.resolve())
    except ValueError:
        raise RuntimeError("Worker CSV path is outside the recovery run directory")
    if not source.exists():
        raise RuntimeError("Worker CSV path is missing or outside the run directory")
    destination = OUT / f"{name}.csv"
    metadata_path = OUT / f"{name}.meta.json"
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Target occupied before install: {name}")

    physical_rows, observed_instruments = row_and_instrument_counts(source)
    if physical_rows <= 0:
        raise RuntimeError("Worker returned no physical rows")
    raw_hash = short_hash(source)
    if not raw_hash:
        raise RuntimeError("Worker CSV has no SHA-256")

    temp_destination = OUT / f".{name}.{run_id}.csv.tmp"
    temp_metadata = OUT / f".{name}.{run_id}.meta.json.tmp"
    try:
        shutil.copyfile(source, temp_destination)
        if destination.exists():
            raise FileExistsError(f"Target appeared during CSV install: {name}")
        temp_destination.rename(destination)
        if short_hash(destination) != raw_hash:
            raise RuntimeError("Destination SHA-256 changed during install")
        metadata = {
            "name": name,
            "request": request,
            "request_sha256": request_hash(request),
            "retrieved_at_utc": utc_now(),
            "status": "success",
            "rows": physical_rows,
            "columns": result.get("columns", []),
            "sha256": raw_hash,
            "elapsed_s": result.get("elapsed_seconds"),
            "source": "lseg.desktop.workspace",
            "recovery_run_id": run_id,
            "recovery_attempt": attempt,
            "observed_instrument_count": observed_instruments,
            "worker_result": rel(run_dir / "attempts" / f"{name}.attempt{attempt}.result.json"),
        }
        if metadata_path.exists():
            raise FileExistsError(f"Target metadata appeared during install: {name}")
        write_json(temp_metadata, metadata)
        temp_metadata.rename(metadata_path)
    finally:
        if temp_destination.exists():
            temp_destination.unlink()
        if temp_metadata.exists():
            temp_metadata.unlink()

    final_rows, final_instruments = row_and_instrument_counts(destination)
    final_hash = short_hash(destination)
    checks = {
        "request_matches_plan": True,
        "rows_positive": final_rows > 0,
        "row_count_matches_worker": final_rows == physical_rows,
        "sha256_matches_worker": final_hash == raw_hash,
        "metadata_exists": metadata_path.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Post-install check failed: {checks}")
    return {
        "status": "success_installed",
        "rows": final_rows,
        "observed_instrument_count": final_instruments,
        "sha256": final_hash,
        "checks": checks,
    }


def preserve_error_snapshot(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for record in plan.get("protected_error_files", []):
        path = ROOT / str(record["path"])
        snapshots[str(record["path"])] = {
            "path": str(record["path"]),
            "before_exists": path.exists(),
            "before_sha256": short_hash(path),
            "before_bytes": path.stat().st_size if path.exists() else None,
        }
    return snapshots


def verify_error_snapshots(snapshots: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    checked: dict[str, dict[str, Any]] = {}
    for key, record in snapshots.items():
        path = ROOT / key
        after_hash = short_hash(path)
        after_bytes = path.stat().st_size if path.exists() else None
        checked[key] = {
            **record,
            "after_exists": path.exists(),
            "after_sha256": after_hash,
            "after_bytes": after_bytes,
            "unchanged": record["before_exists"] == path.exists() and record["before_sha256"] == after_hash,
        }
    return checked


def fetch_from_plan(plan_path: Path) -> dict[str, Any]:
    plan = load_plan(plan_path)
    run_id = str(plan["run_id"])
    run_dir = plan_path.parent
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    snapshots = preserve_error_snapshot(plan)
    started = utc_now()
    target_records: list[dict[str, Any]] = []

    for item in plan["target_batches"]:
        name = str(item["name"])
        request = item["request"]
        current = classify_batch(item)
        record: dict[str, Any] = {
            "name": name,
            "planned_state": item.get("current", {}).get("state"),
            "fetch_start_utc": utc_now(),
            "request": request,
            "request_sha256": request_hash(request),
            "attempts": [],
        }
        if current["state"] == "success_verified":
            record.update(status="skipped_already_verified", final_state=current["state"])
            target_records.append(record)
            continue
        # An existing CSV or metadata file may have appeared after the plan;
        # do not overwrite it, even if it is unverified.
        if current["csv_exists"] or current["meta_exists"]:
            record.update(status="skipped_target_occupied", final_state=current["state"], current=current)
            target_records.append(record)
            continue

        request_file = attempts_dir / f"{name}.request.json"
        write_json(request_file, request)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result_path = attempts_dir / f"{name}.attempt{attempt}.result.json"
            worker_csv = result_path.with_suffix(result_path.suffix + ".csv")
            if result_path.exists():
                result_path.unlink()
            if worker_csv.exists():
                worker_csv.unlink()
            attempt_started = utc_now()
            child_result = run_worker(request_file, result_path, ATTEMPT_TIMEOUT_SECONDS)
            attempt_record = {
                "attempt": attempt,
                "started_at_utc": attempt_started,
                **child_result,
            }
            record["attempts"].append(attempt_record)
            if child_result.get("status") == "success":
                try:
                    installed = install_success(name, request, child_result, run_id, attempt, run_dir)
                    # ``install_success`` includes its own status field.  Merge
                    # it first so a successful install cannot be mis-recorded
                    # as a failed retry because of a duplicate keyword.
                    record.update(installed)
                    record.update(status="success_installed", final_state="success_verified")
                    break
                except Exception as exc:
                    attempt_record["install_error"] = str(exc)[:800]
            if attempt < MAX_ATTEMPTS:
                time.sleep(2)
        else:
            record.update(status="failed_after_max_attempts", final_state=classify_batch(item)["state"])
        record["fetch_end_utc"] = utc_now()
        target_records.append(record)
        progress = {
            "stage": "recovery_progress",
            "run_id": run_id,
            "plan": rel(plan_path),
            "started_at_utc": started,
            "updated_at_utc": utc_now(),
            "targets_completed": len(target_records),
            "target_count": len(plan["target_batches"]),
            "target_records": target_records,
        }
        write_json(run_dir / "recovery_progress.json", progress)

    expected = build_expected()
    after_states, after_status_counts, after_totals = inventory(expected)
    error_check = verify_error_snapshots(snapshots)
    report = {
        "stage": "recovery_report",
        "run_id": run_id,
        "plan": rel(plan_path),
        "plan_sha256": short_hash(plan_path),
        "script": rel(Path(__file__)),
        "script_sha256": short_hash(Path(__file__)),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "expected_batch_count": len(expected),
        "before_status_counts": plan.get("current_status_counts", {}),
        "after_status_counts": after_status_counts,
        "before_verified_totals": plan.get("current_verified_totals", {}),
        "after_verified_totals": after_totals,
        "target_batch_count": len(plan["target_batches"]),
        "target_records": target_records,
        "remaining_non_verified": [name for name, state in after_states.items() if state["state"] != "success_verified"],
        "old_error_file_checks": error_check,
        "rules": plan.get("rules", {}),
        "limitations": [
            "LSEG API availability and returned rows are recorded outcomes, not completeness proof.",
            "No imputation, deduplication, winsorization, forward fill, row removal, or security exclusion was applied.",
            "Existing success files were checked but never rewritten by this recovery run.",
        ],
    }
    write_json(run_dir / "recovery_report.json", report)
    return report


def verify_existing_run(plan_path: Path) -> dict[str, Any]:
    """Reconcile an already completed run without making any LSEG request."""
    plan = load_plan(plan_path)
    run_dir = plan_path.parent
    expected = build_expected()
    states, status_counts, totals = inventory(expected)
    source_report_path = run_dir / "recovery_report.json"
    source_report = read_json(source_report_path) if source_report_path.exists() else None
    source_records = {str(record.get("name")): record for record in (source_report or {}).get("target_records", [])}
    target_reconciliation: list[dict[str, Any]] = []
    for item in plan["target_batches"]:
        name = str(item["name"])
        current = states[name]
        original = source_records.get(name, {})
        attempts = original.get("attempts", [])
        successful_attempts = [record for record in attempts if record.get("status") == "success"]
        metadata_path = OUT / f"{name}.meta.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        corrected = {
            "name": name,
            "planned_state": item.get("current", {}).get("state"),
            "source_report_status": original.get("status"),
            "source_report_final_state": original.get("final_state"),
            "current_state": current["state"],
            "attempt_count_recorded": len(attempts),
            "successful_attempts_recorded": [record.get("attempt") for record in successful_attempts],
            "installed_recovery_attempt": metadata.get("recovery_attempt"),
            "rows": current.get("physical_rows"),
            "observed_instrument_count": current.get("observed_instrument_count"),
            "sha256": current.get("sha256"),
        }
        if current["state"] == "success_verified" and successful_attempts:
            corrected.update(
                corrected_status=f"success_installed_on_attempt_{metadata.get('recovery_attempt', 'unknown')}",
                correction="The original report raised a duplicate status-key bookkeeping error after installing the file; the post-fetch file and metadata checks are authoritative.",
            )
        else:
            corrected.update(corrected_status="remaining_non_verified")
        target_reconciliation.append(corrected)

    error_snapshots = {}
    for record in plan.get("protected_error_files", []):
        path = ROOT / str(record["path"])
        after_hash = short_hash(path)
        error_snapshots[str(record["path"])] = {
            "before_sha256": record.get("sha256"),
            "before_bytes": record.get("bytes"),
            "after_sha256": after_hash,
            "after_bytes": path.stat().st_size if path.exists() else None,
            "unchanged": bool(path.exists() and after_hash == record.get("sha256")),
        }
    verification = {
        "stage": "post_fetch_verification",
        "run_id": plan["run_id"],
        "plan": rel(plan_path),
        "plan_sha256": short_hash(plan_path),
        "source_report": rel(source_report_path),
        "source_report_sha256": short_hash(source_report_path),
        "verified_at_utc": utc_now(),
        "script": rel(Path(__file__)),
        "script_sha256": short_hash(Path(__file__)),
        "expected_batch_count": len(expected),
        "before_status_counts": plan.get("current_status_counts", {}),
        "after_status_counts": status_counts,
        "before_verified_totals": plan.get("current_verified_totals", {}),
        "after_verified_totals": totals,
        "target_reconciliation": target_reconciliation,
        "remaining_non_verified": [name for name, state in states.items() if state["state"] != "success_verified"],
        "old_error_file_checks": error_snapshots,
        "network_requests_made": 0,
        "limitations": [
            "This reconciliation performs no network request; it verifies the persisted raw files and metadata against the pre-fetch plan.",
            "The original report is preserved verbatim; its bookkeeping error is explicitly retained as source evidence.",
        ],
    }
    write_json(run_dir / "recovery_verification.json", verification)
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Write a run-specific pre-fetch recovery plan")
    mode.add_argument("--fetch", metavar="PLAN", type=Path, help="Execute only targets from a recovery plan")
    mode.add_argument("--verify", metavar="PLAN", type=Path, help="Verify persisted outputs for a completed recovery run")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--request-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if not args.request_file or not args.result_file:
            raise SystemExit("--worker requires --request-file and --result-file")
        return worker_main(args.request_file, args.result_file)
    if args.plan:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = AUDIT_ROOT / run_id
        # Avoid accidental reuse if two plan stages begin in one second.
        suffix = 1
        while run_dir.exists():
            run_dir = AUDIT_ROOT / f"{run_id}_{suffix:02d}"
            suffix += 1
        plan = make_plan(run_dir.name)
        plan_path = run_dir / "recovery_plan.json"
        write_json(plan_path, plan)
        (run_dir / "recovery_plan.md").write_text(plan_markdown(plan), encoding="utf-8")
        print(json.dumps({"plan": rel(plan_path), "status_counts": plan["current_status_counts"], "targets": plan["target_batch_names"]}, ensure_ascii=False))
        return 0
    if args.verify:
        verification = verify_existing_run(args.verify)
        print(
            json.dumps(
                {
                    "run_id": verification["run_id"],
                    "after_status_counts": verification["after_status_counts"],
                    "remaining_non_verified": verification["remaining_non_verified"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if not verification["remaining_non_verified"] else 2
    report = fetch_from_plan(args.fetch)
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "before_status_counts": report["before_status_counts"],
                "after_status_counts": report["after_status_counts"],
                "remaining_non_verified": report["remaining_non_verified"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if not report["remaining_non_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
