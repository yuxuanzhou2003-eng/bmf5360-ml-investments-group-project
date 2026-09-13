"""Bounded LSEG probe for EPS basis, scale, currency, and vintage candidates.

This creates a new immutable raw folder. It never edits prior raw responses,
clean tables, the panel, or labels. Candidate failures are retained as evidence.
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
OUT = ROOT / "data/raw/eps_field_probe_v2" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
UNIVERSE = ["AAPL.OQ", "NVDA.OQ"]
ACT = {"SDate": "2020-01-01", "EDate": "2024-06-30", "Frq": "FQ", "Period": "FQ0"}
EST = {"SDate": "2024-05-10", "EDate": "2024-05-22", "Frq": "D", "Period": "FQ1"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return sha256(path)


def req(fields, params):
    return {"universe": UNIVERSE, "fields": fields, "parameters": params}


base_actual = ["TR.EPSActValue.announcedate", "TR.EPSActValue.periodenddate", "TR.EPSActValue"]
base_estimate = ["TR.EPSMean.calcdate", "TR.EPSMean.periodenddate", "TR.EPSMean",
                 "TR.EPSStdDev", "TR.EPSNumIncEstimates"]
requests = [
    ("actual_base", req(base_actual, ACT), "control"),
    ("actual_reported", req(["TR.EPSActValue(ActType=Reported).announcedate",
                              "TR.EPSActValue(ActType=Reported).periodenddate",
                              "TR.EPSActValue(ActType=Reported)"], ACT), "supported selector retest"),
    ("actual_restated_candidate", req(["TR.EPSActValue(ActType=Restated)"], ACT), "exploratory"),
    ("actual_goforward_candidate", req(["TR.EPSActValue(ActType=GoForward)"], ACT), "exploratory"),
    ("actual_currency_suffix", req(base_actual + ["TR.EPSActValue.currency"], ACT), "exploratory suffix"),
    ("actual_scale_suffix", req(base_actual + ["TR.EPSActValue.scale"], ACT), "exploratory suffix"),
    ("actual_activation_suffix", req(base_actual + ["TR.EPSActValue.activationdate"], ACT), "exploratory suffix"),
    ("actual_effective_suffix", req(base_actual + ["TR.EPSActValue.effectivedate"], ACT), "exploratory suffix"),
    ("actual_fperiod_suffix", req(base_actual + ["TR.EPSActValue.fperiod"], ACT), "official-example suffix"),
    ("alternative_eps_fields", req(["TR.DilutedEPSExclExtra", "TR.DilutedEPSExclExtra.periodenddate",
                                     "TR.EPSNormalizeddil", "TR.EPSNormalizeddil.periodenddate"], ACT),
     "official-community examples; basis not inferred"),
    ("estimate_base", req(base_estimate, EST), "control"),
    ("estimate_currency_suffix", req(base_estimate + ["TR.EPSMean.currency"], EST), "exploratory suffix"),
    ("estimate_scale_suffix", req(base_estimate + ["TR.EPSMean.scale"], EST), "exploratory suffix"),
    ("estimate_fperiod_suffix", req(base_estimate + ["TR.EPSMean.fperiod"], EST), "exploratory suffix"),
    ("actual_usd_scale0", req(base_actual, ACT | {"Curn": "USD", "Scale": "0"}), "global parameter test"),
    ("actual_usd_scale6", req(base_actual, ACT | {"Curn": "USD", "Scale": "6"}), "global parameter test"),
    ("estimate_usd_scale0", req(base_estimate, EST | {"Curn": "USD", "Scale": "0"}), "global parameter test"),
    ("estimate_usd_scale6", req(base_estimate, EST | {"Curn": "USD", "Scale": "6"}), "global parameter test"),
]

manifest = {
    "run_id": RUN_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Isolated field acceptance and returned-value probe; not a DIB definition.",
    "universe": UNIVERSE, "maximum_calls": len(requests), "calls": [],
}
write_json(OUT / "manifest.json", manifest)
ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
try:
    for name, request, status_note in requests:
        request_path = OUT / f"{name}.request.json"
        request_hash = write_json(request_path, request)
        record = {"name": name, "note": status_note, "request_sha256": request_hash,
                  "requested_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            frame = ld.get_data(**request)
            csv_path = OUT / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            record.update(status="returned", rows=int(len(frame)),
                          columns=[str(column) for column in frame.columns],
                          non_null={str(column): int(frame[column].notna().sum()) for column in frame.columns},
                          csv_sha256=sha256(csv_path))
            print("OK", name, len(frame), flush=True)
        except Exception as exc:
            record.update(status="error", error=redact(exc)[:1000])
            (OUT / f"{name}.error.txt").write_text(record["error"] + "\n", encoding="utf-8")
            record["error_sha256"] = sha256(OUT / f"{name}.error.txt")
            print("ERROR", name, record["error"][:160], flush=True)
        manifest["calls"].append(record)
        write_json(OUT / "manifest.json", manifest)
finally:
    ld.close_session()
manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["returned_calls"] = sum(call["status"] == "returned" for call in manifest["calls"])
manifest["error_calls"] = sum(call["status"] == "error" for call in manifest["calls"])
write_json(OUT / "manifest.json", manifest)
print(OUT, flush=True)
