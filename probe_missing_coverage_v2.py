"""One-shot, read-only EPS/identity probes for the missing-RIC audit.

Writes only a new run directory under data/raw/missing_coverage_probe/.
No retries, remapping, cleaning, or merging are performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
credentials = dotenv_values(ROOT / ".env")
secrets = [v for v in credentials.values() if v]


def redact(value):
    value = str(value)
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    return value


class SafeStream:
    def __init__(self, target):
        self.target = target

    def write(self, message):
        return self.target.write(redact(message))

    def flush(self):
        self.target.flush()

    def isatty(self):
        return False


sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)

import pandas as pd
import lseg.data as ld


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "missing_coverage_probe" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
records = []

IDENTITY_FIELDS = ["TR.RIC", "TR.CommonName", "TR.OrganizationID", "TR.ISIN"]
ACTUAL_FIELDS = [
    "TR.EPSActValue.announcedate",
    "TR.EPSActValue.periodenddate",
    "TR.EPSActValue",
]
ESTIMATE_FIELDS = [
    "TR.EPSMean.calcdate",
    "TR.EPSMean.periodenddate",
    "TR.EPSMean",
    "TR.EPSStdDev",
    "TR.EPSNumIncEstimates",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> str:
    """Write an auditable request/error sidecar and return its hash."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def nonempty_count(series) -> int:
    count = 0
    for value in series.tolist():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "nat", "none", "<na>"}:
            count += 1
    return count


def write_manifest(session_status=None):
    manifest = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).name),
        "script_sha256": sha256_file(Path(__file__)),
        "session_status": session_status,
        "call_count": len(records),
        "calls": records,
        "rules": {
            "read_only": True,
            "max_calls": 7,
            "no_retries": True,
            "no_clean_merge_or_ric_remap": True,
            "blank_strings_as_missing_for_counts_only": True,
        },
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def probe(label: str, request: dict):
    started = datetime.now(timezone.utc)
    request_path = OUT / f"{label}.request.json"
    request_sha256 = write_json(request_path, request)
    record = {
        "label": label,
        "request": request,
        "request_path": str(request_path.relative_to(ROOT)),
        "request_sha256": request_sha256,
        "started_at_utc": started.isoformat(),
        "status": "running",
    }
    records.append(record)
    try:
        timer = threading.Timer(90, lambda: os._exit(2))
        timer.start()
        try:
            frame = ld.get_data(**request)
        finally:
            timer.cancel()
        if frame is None:
            record.update(status="empty", rows=0, columns=[], field_nonempty={})
        else:
            # Preserve the response itself.  Blank strings are considered only
            # in the aggregate non-empty count; they are never rewritten here.
            csv_path = OUT / f"{label}.csv"
            frame.to_csv(csv_path, index=False)
            record.update(
                status="success" if len(frame) else "empty",
                rows=int(len(frame)),
                columns=[str(c) for c in frame.columns],
                field_nonempty={str(c): nonempty_count(frame[c]) for c in frame.columns},
                csv_path=str(csv_path.relative_to(ROOT)),
                csv_sha256=sha256_file(csv_path),
            )
    except Exception as exc:
        error_path = OUT / f"{label}.error.txt"
        error_path.write_text(redact(exc), encoding="utf-8")
        record.update(
            status="error",
            error=redact(exc)[:1500],
            error_path=str(error_path.relative_to(ROOT)),
            error_sha256=sha256_file(error_path),
        )
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest()
        print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    total_timer = threading.Timer(120, lambda: os._exit(2))
    total_timer.start()
    session_status = "error"
    try:
        try:
            ld.open_session(name="desktop.workspace", app_key=credentials.get("LSEG_APP_KEY"))
            session_status = "connected"
            write_manifest(session_status)
        except Exception as exc:
            session_status = "error"
            session_error_path = OUT / "session_error.json"
            session_error_sha256 = write_json(
                session_error_path,
                {
                    "status": "error",
                    "error": redact(exc)[:1500],
                    "tested_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            records.append(
                {
                    "label": "desktop_workspace_session",
                    "status": "error",
                    "error_path": str(session_error_path.relative_to(ROOT)),
                    "error_sha256": session_error_sha256,
                }
            )
            write_manifest(session_status)
            print("SESSION ERROR; no API calls made", flush=True)
            return

        fdx_universe = ["FDX.N", "FDX", "FDXF.N"]
        dead_common = ["SWY.N^A15", "DD.N^I17", "COV.N^A15"]

        probe(
            "identity_fdx_dd",
            {"universe": ["FDX.N", "FDX", "FDXF.N", "DD.N", "DD.N^I17"], "fields": IDENTITY_FIELDS},
        )
        probe(
            "fdx_actuals_full",
            {
                "universe": fdx_universe,
                "fields": ACTUAL_FIELDS,
                "parameters": {"SDate": "2013-11-01", "EDate": "2026-09-07", "Frq": "FQ", "Period": "FQ0"},
            },
        )
        probe(
            "fdx_estimates_full",
            {
                "universe": fdx_universe,
                "fields": ESTIMATE_FIELDS,
                "parameters": {"SDate": "2013-11-01", "EDate": "2026-09-07", "Frq": "W", "Period": "FQ1"},
            },
        )
        probe(
            "dead_common_actuals_2014",
            {
                "universe": dead_common,
                "fields": ACTUAL_FIELDS,
                "parameters": {"SDate": "2014-01-01", "EDate": "2014-12-31", "Frq": "FQ", "Period": "FQ0"},
            },
        )
        probe(
            "dead_common_estimates_2014",
            {
                "universe": dead_common,
                "fields": ESTIMATE_FIELDS,
                "parameters": {"SDate": "2014-01-01", "EDate": "2014-12-31", "Frq": "W", "Period": "FQ1"},
            },
        )
        probe(
            "day_actuals_2025",
            {
                "universe": ["DAY.N^B26"],
                "fields": ACTUAL_FIELDS,
                "parameters": {"SDate": "2025-01-01", "EDate": "2025-12-31", "Frq": "FQ", "Period": "FQ0"},
            },
        )
        probe(
            "day_estimates_2025",
            {
                "universe": ["DAY.N^B26"],
                "fields": ESTIMATE_FIELDS,
                "parameters": {"SDate": "2025-01-01", "EDate": "2025-12-31", "Frq": "W", "Period": "FQ1"},
            },
        )
    finally:
        if session_status == "connected":
            ld.close_session()
        total_timer.cancel()
        write_manifest(session_status)
        print(json.dumps({"run_id": RUN_ID, "output": str(OUT), "calls": len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
