"""Collect LSEG stock-split corporate actions without altering research tables.

Usage: .venv/Scripts/python.exe collect_stock_splits_v2.py probe|full
Each invocation creates an immutable run folder with requests, direct CSV
responses, hashes, and a manifest. Blank values are preserved as returned.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
CREDS = dotenv_values(ROOT / ".env")
SECRETS = [str(value) for value in CREDS.values() if value]


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

import lseg.data as ld  # noqa: E402
import pandas as pd  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "probe"
if MODE not in {"probe", "full"}:
    raise SystemExit("mode must be probe or full")

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "stock_splits_v2" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
FIELDS = [
    "TR.CAAnnouncementDate",
    "TR.CARecordDate",
    "TR.CAEffectiveDate",
    "TR.CAExDate",
    "TR.CAAdjustmentFactor",
    "TR.CAAdjustmentType",
    "TR.CATermsOldShares",
    "TR.CATermsNewShares",
]
PARAMETERS = {"CAEventType": "SSP", "SDate": "2013-11-01", "EDate": "2026-09-09"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return sha256(path)


coverage_path = ROOT / "data/audit/event_readiness/20260908T020634146977Z/ric_coverage.csv"
if MODE == "probe":
    groups = [("probe", ["AAPL.O", "TSLA.O", "NVDA.O"])]
else:
    coverage = pd.read_csv(coverage_path, usecols=["Instrument"])
    rics = coverage["Instrument"].astype(str).tolist()
    if len(rics) != 782 or coverage["Instrument"].duplicated().any():
        raise RuntimeError("Expected 782 unique RICs in the frozen coverage input")
    batch_size = 100
    groups = [(f"batch_{start // batch_size:03d}", rics[start:start + batch_size])
              for start in range(0, len(rics), batch_size)]

manifest = {
    "run_id": RUN_ID,
    "mode": MODE,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Raw stock-split corporate-action collection; no research-table transformation.",
    "fields": FIELDS,
    "parameters": PARAMETERS,
    "coverage_input": str(coverage_path.relative_to(ROOT)) if MODE == "full" else None,
    "coverage_input_sha256": sha256(coverage_path) if MODE == "full" else None,
    "calls": [],
}
write_json(OUT / "manifest.json", manifest)

ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
try:
    for name, universe in groups:
        request = {"universe": universe, "fields": FIELDS, "parameters": PARAMETERS}
        request_path = OUT / f"{name}.request.json"
        request_hash = write_json(request_path, request)
        record = {"name": name, "universe_count": len(universe), "request_sha256": request_hash}
        try:
            frame = ld.get_data(**request)
            csv_path = OUT / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            record.update(
                status="returned",
                rows=int(len(frame)),
                columns=[str(column) for column in frame.columns],
                non_null={str(column): int(frame[column].notna().sum()) for column in frame.columns},
                csv_sha256=sha256(csv_path),
            )
            print("OK", name, len(frame), flush=True)
        except Exception as exc:
            record.update(status="error", error=redact(exc)[:500])
            write_json(OUT / f"{name}.error.json", record)
            print("ERROR", name, record["error"], flush=True)
        manifest["calls"].append(record)
        write_json(OUT / "manifest.json", manifest)
finally:
    ld.close_session()

manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["returned_calls"] = sum(call["status"] == "returned" for call in manifest["calls"])
manifest["error_calls"] = sum(call["status"] == "error" for call in manifest["calls"])
write_json(OUT / "manifest.json", manifest)
print(OUT, flush=True)
