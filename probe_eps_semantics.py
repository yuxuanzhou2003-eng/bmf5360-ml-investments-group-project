"""Bounded, read-only probe for LSEG I/B/E/S actual date and adjustment semantics.

The script makes one desktop session attempt and at most six small get_data calls.
It preserves raw DataFrame responses, exact request objects, and SHA-256 metadata
under data/raw/semantics_probe/<UTC run id>/. Credentials are loaded locally and
never written to files or stdout.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
CREDS = dotenv_values(ROOT / ".env")
SECRETS = [str(v) for v in CREDS.values() if v]


def redact(value: object) -> str:
    text = str(value)
    for secret in SECRETS:
        text = text.replace(secret, "[REDACTED]")
    return text


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

import lseg.data as ld  # noqa: E402  (redaction is installed first)
import pandas as pd  # noqa: E402


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "semantics_probe" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
RICs = ["AAPL.O", "MSFT.O", "INTC.O"]
PARAMETERS = {"SDate": "2015-01-01", "EDate": "2015-03-31", "Frq": "FQ", "Period": "FQ0"}
records: list[dict] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return sha256(path)


def request_for(fields: list[str]) -> dict:
    return {"universe": RICs, "fields": fields, "parameters": PARAMETERS}


def write_record(name: str, request: dict, started: str, status: str, **extra) -> None:
    record = {
        "name": name,
        "request": request,
        "requested_at_utc": started,
        "status": status,
        **extra,
    }
    records.append(record)
    write_json(OUT / "probe_manifest.json", {"run_id": RUN_ID, "calls": records})


def probe(name: str, request: dict, label: str) -> None:
    started = datetime.now(timezone.utc).isoformat()
    query_path = OUT / f"{name}.request.json"
    query_hash = write_json(query_path, request)
    print(f"START {name}", flush=True)
    try:
        frame = ld.get_data(**request)
        if frame is None:
            write_record(name, request, started, "empty_response", query_sha256=query_hash)
            return
        # Preserve the direct tabular response exactly as serialized by pandas;
        # no missing-value conversion, date parsing, deduplication, or filtering.
        csv_path = OUT / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        json_path = OUT / f"{name}.records.json"
        json_hash = write_json(json_path, frame.to_dict(orient="records"))
        write_record(
            name,
            request,
            started,
            "returned" if len(frame) else "empty",
            label=label,
            rows=int(len(frame)),
            columns=[str(column) for column in frame.columns],
            non_null={str(column): int(frame[column].notna().sum()) for column in frame.columns},
            csv_sha256=sha256(csv_path),
            records_json_sha256=json_hash,
            query_sha256=query_hash,
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    except Exception as exc:
        error_path = OUT / f"{name}.error.txt"
        error_path.write_text(redact(exc), encoding="utf-8")
        write_record(
            name,
            request,
            started,
            "error",
            label=label,
            error=redact(exc)[:2000],
            error_sha256=sha256(error_path),
            query_sha256=query_hash,
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)


base = request_for([
    "TR.EPSActValue.announcedate",
    "TR.EPSActValue.periodenddate",
    "TR.EPSActValue",
])
date_outputs = request_for([
    "TR.EPSActValue.date",
    "TR.EPSActValue.announcedate",
    "TR.EPSActValue.calcdate",
    "TR.EPSActValue.periodenddate",
])
reported = request_for([
    "TR.EPSActValue(ActType=Reported)",
    "TR.EPSActValue(ActType=Reported).announcedate",
    "TR.EPSActValue(ActType=Reported).periodenddate",
])
timezone_probe = request_for([
    # Exploratory only: the official community example documents this suffix
    # for TR.EPSEstDate, not for EPSActValue or its announcement-date output.
    "TR.EPSActValue.origtimezone",
    "TR.EPSActValue.announcedate.origtimezone",
])
adjustment_types = request_for([
    "TR.EPSActValue",
    "TR.EPSActValue(ActType=Reported)",
    "TR.EPSActValue(ActType=Comparable)",
    "TR.EPSActValue(ActType=Restated)",
    "TR.EPSActValue(ActType=GoForward)",
])


def main() -> int:
    # Exactly one session attempt. A session failure is terminal; no alternate
    # credential path or broad retry is attempted.
    session_started = datetime.now(timezone.utc).isoformat()
    session_request = {"session": "desktop.workspace", "app_key_source": ".env:LSEG_APP_KEY"}
    try:
        session = ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
        if session is None:
            write_record("desktop_session", session_request, session_started, "failed")
            return 2
        write_record("desktop_session", session_request, session_started, "connected")
    except Exception as exc:
        error_path = OUT / "desktop_session.error.txt"
        error_path.write_text(redact(exc), encoding="utf-8")
        write_record(
            "desktop_session",
            session_request,
            session_started,
            "error",
            error=redact(exc)[:2000],
            error_sha256=sha256(error_path),
        )
        return 2

    try:
        probe("base_2015q1", base, "documented base actual, announcement and period-end outputs")
        probe("date_outputs_2015q1", date_outputs, "date/calcdate output comparison; exact meaning remains content-dependent")
        probe("reported_2015q1", reported, "reported actual parameter syntax observed in official LSEG Developer Community example")
        probe("origtimezone_exploratory_2015q1", timezone_probe, "unsupported exploratory timezone suffix; evidence only if API returns it")
        probe("adjustment_types_2015q1", adjustment_types, "default versus reported/comparable/restated/go-forward candidate values")
    finally:
        ld.close_session()
        write_json(OUT / "run_metadata.json", {
            "run_id": RUN_ID,
            "started_at_utc": session_started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).name),
            "script_sha256": sha256(Path(__file__)),
            "ric_universe": RICs,
            "parameters": PARAMETERS,
            "max_data_calls": 6,
            "data_calls_attempted": 5,
            "raw_policy": "Direct DataFrame serialization only; no parsing, filtering, deduplication, imputation, or unit/date/timezone conversion.",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
