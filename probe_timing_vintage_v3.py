"""Small, versioned LSEG probe for earnings timing and point-in-time fields.

Every request is isolated so an unsupported field cannot suppress the other
results. Direct DataFrames are serialized without parsing, filling, filtering,
deduplication, or timezone conversion.
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


RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "timing_vintage_probe_v3" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)


CALLS = [
    (
        "aapl_consensus_fq1_daily",
        {
            "universe": ["AAPL.OQ"],
            "fields": [
                "TR.EPSMean.calcdate", "TR.EPSMean.periodenddate", "TR.EPSMean.fperiod",
                "TR.EPSMean.currency", "TR.EPSMean", "TR.EPSStdDev", "TR.EPSNumIncEstimates",
            ],
            "parameters": {"SDate": "2015-01-20", "EDate": "2015-01-30", "Frq": "D", "Period": "FQ1"},
        },
        "Daily calc-date sequence around the 2015-01-27 earnings release.",
    ),
    (
        "aapl_consensus_fq0_daily",
        {
            "universe": ["AAPL.OQ"],
            "fields": [
                "TR.EPSMean.calcdate", "TR.EPSMean.periodenddate", "TR.EPSMean.fperiod",
                "TR.EPSMean.currency", "TR.EPSMean", "TR.EPSStdDev", "TR.EPSNumIncEstimates",
            ],
            "parameters": {"SDate": "2015-01-20", "EDate": "2015-01-30", "Frq": "D", "Period": "FQ0"},
        },
        "Control for fiscal-period rolling across the same event.",
    ),
    (
        "aapl_actual_reported_2015q1",
        {
            "universe": ["AAPL.OQ"],
            "fields": [
                "TR.EPSActValue(ActType=Reported).date",
                "TR.EPSActValue(ActType=Reported).announcedate",
                "TR.EPSActValue(ActType=Reported).calcdate",
                "TR.EPSActValue(ActType=Reported).periodenddate",
                "TR.EPSActValue(ActType=Reported).fperiod",
                "TR.EPSActValue(ActType=Reported).currency",
                "TR.EPSActValue(ActType=Reported)",
            ],
            "parameters": {"SDate": "2015-01-01", "EDate": "2015-03-31", "Frq": "FQ", "Period": "FQ0"},
        },
        "Explicit Reported actual with available date and basis outputs.",
    ),
    (
        "aapl_actual_all_2015q1",
        {
            "universe": ["AAPL.OQ"],
            "fields": [
                "TR.EPSActValue(ActType=All).announcedate",
                "TR.EPSActValue(ActType=All).calcdate",
                "TR.EPSActValue(ActType=All).periodenddate",
                "TR.EPSActValue(ActType=All)",
            ],
            "parameters": {"SDate": "2015-01-01", "EDate": "2015-03-31", "Frq": "FQ", "Period": "FQ0"},
        },
        "Check whether ActType=All exposes multiple versions for this event.",
    ),
    (
        "aapl_event_release_2015",
        {
            "universe": ["AAPL.OQ"],
            "fields": ["TR.EventStartDate", "TR.EventStartTime", "TR.EventType", "TR.EventTitle"],
            "parameters": {"SDate": "2015-01-25", "EDate": "2015-01-29", "EventType": "RES"},
        },
        "Independent event-calendar timestamp candidate using the documented RES filter.",
    ),
    (
        "aapl_financial_original_announcement",
        {
            "universe": ["AAPL.OQ"],
            "fields": ["TR.F.OriginalAnnouncementDate(Period=FQ0,Frq=FQ,SDate=2015-01-01,EDate=2015-03-31)"],
        },
        "UTC-marked financial-statement announcement candidate; may differ from the EPS release event.",
    ),
    (
        "winter_original_announcement_crosscheck",
        {
            "universe": ["AAPL.OQ", "MSFT.OQ", "INTC.OQ"],
            "fields": ["TR.F.OriginalAnnouncementDate(Period=FQ0,Frq=FQ,SDate=2015-01-01,EDate=2015-03-31)"],
        },
        "Winter UTC timestamp cross-check for the three existing Report Date examples.",
    ),
    (
        "aapl_reported_actuals_2020",
        {
            "universe": ["AAPL.OQ"],
            "fields": [
                "TR.EPSActValue(ActType=Reported).announcedate",
                "TR.EPSActValue(ActType=Reported).periodenddate",
                "TR.EPSActValue(ActType=Reported)",
            ],
            "parameters": {"SDate": "2020-01-01", "EDate": "2020-08-31", "Frq": "FQ", "Period": "FQ0"},
        },
        "AAPL local-clock candidates spanning standard and daylight-saving time.",
    ),
    (
        "aapl_original_announcements_2020",
        {
            "universe": ["AAPL.OQ"],
            "fields": ["TR.F.OriginalAnnouncementDate"],
            "parameters": {"SDate": "2020-01-01", "EDate": "2020-08-31", "Frq": "FQ", "Period": "FQ0"},
        },
        "UTC-marked AAPL financial announcement timestamps across DST regimes.",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main() -> int:
    manifest = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Isolated timing and point-in-time field probe; not a transformation.",
        "maximum_calls": len(CALLS),
        "calls": [],
    }
    write_json(OUT / "manifest.json", manifest)
    try:
        ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
        manifest["session"] = "connected"
    except Exception as exc:
        manifest["session"] = "error"
        manifest["session_error"] = redact(exc)[:1000]
        write_json(OUT / "manifest.json", manifest)
        return 2

    try:
        for name, request, note in CALLS:
            request_path = OUT / f"{name}.request.json"
            write_json(request_path, request)
            record = {
                "name": name,
                "note": note,
                "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                "request_sha256": sha256(request_path),
            }
            try:
                frame = ld.get_data(**request)
                csv_path = OUT / f"{name}.csv"
                frame.to_csv(csv_path, index=False)
                record.update({
                    "status": "returned" if len(frame) else "empty",
                    "rows": int(len(frame)),
                    "columns": [str(column) for column in frame.columns],
                    "non_null": {str(column): int(frame[column].notna().sum()) for column in frame.columns},
                    "csv_sha256": sha256(csv_path),
                })
            except Exception as exc:
                error_path = OUT / f"{name}.error.txt"
                error_path.write_text(redact(exc), encoding="utf-8")
                record.update({"status": "error", "error": redact(exc)[:1000], "error_sha256": sha256(error_path)})
            manifest["calls"].append(record)
            write_json(OUT / "manifest.json", manifest)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        ld.close_session()

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["script"] = Path(__file__).name
    manifest["script_sha256"] = sha256(Path(__file__))
    manifest["raw_policy"] = "Direct serialization only; no parsing, filtering, deduplication, filling, or timezone conversion."
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"run_id": RUN_ID, "out": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
