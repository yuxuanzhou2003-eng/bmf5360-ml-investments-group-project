"""Isolated follow-up for EPS basis candidates after a combined-field API error."""
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


def redact(value):
    text = str(value)
    for secret in SECRETS:
        text = text.replace(secret, "[REDACTED]")
    return text


class SafeStream:
    def __init__(self, target): self.target = target
    def write(self, message): return self.target.write(redact(message))
    def flush(self): return self.target.flush()
    def isatty(self): return False


sys.stdout, sys.stderr = SafeStream(sys.stdout), SafeStream(sys.stderr)
import lseg.data as ld  # noqa: E402

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out = ROOT / "data/raw/eps_alternative_probe" / run_id
out.mkdir(parents=True, exist_ok=False)
params = {"SDate": "2024-05-01", "EDate": "2024-06-30", "Frq": "FQ", "Period": "FQ0"}
universe = ["AAPL.OQ", "NVDA.OQ"]
specs = [
    ("diluted_ex_extra", ["TR.DilutedEPSExclExtra.periodenddate", "TR.DilutedEPSExclExtra"]),
    ("normalized_diluted", ["TR.EPSNormalizeddil.periodenddate", "TR.EPSNormalizeddil"]),
    ("reported_actual_full", ["TR.EPSActValue(ActType=Reported).periodenddate",
                               "TR.EPSActValue(ActType=Reported).currency",
                               "TR.EPSActValue(ActType=Reported).fperiod",
                               "TR.EPSActValue(ActType=Reported)"]),
    ("restated_with_period", ["TR.EPSActValue(ActType=Restated).periodenddate",
                               "TR.EPSActValue(ActType=Restated)"]),
    ("goforward_with_period", ["TR.EPSActValue(ActType=GoForward).periodenddate",
                                "TR.EPSActValue(ActType=GoForward)"]),
]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return digest(path)


manifest = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Isolate fields that caused a combined-request dataframe error.",
            "universe": universe, "parameters": params, "calls": []}
save_json(out / "manifest.json", manifest)
ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
try:
    for name, fields in specs:
        request = {"universe": universe, "fields": fields, "parameters": params}
        request_hash = save_json(out / f"{name}.request.json", request)
        record = {"name": name, "request_sha256": request_hash}
        try:
            frame = ld.get_data(**request)
            path = out / f"{name}.csv"; frame.to_csv(path, index=False)
            record.update(status="returned", rows=len(frame), columns=list(map(str, frame.columns)),
                          non_null={str(c): int(frame[c].notna().sum()) for c in frame.columns},
                          csv_sha256=digest(path))
            print("OK", name, len(frame), flush=True)
        except Exception as exc:
            record.update(status="error", error=redact(exc)[:1000])
            path = out / f"{name}.error.txt"; path.write_text(record["error"] + "\n", encoding="utf-8")
            record["error_sha256"] = digest(path)
            print("ERROR", name, record["error"][:160], flush=True)
        manifest["calls"].append(record); save_json(out / "manifest.json", manifest)
finally:
    ld.close_session()
manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
manifest["returned_calls"] = sum(c["status"] == "returned" for c in manifest["calls"])
manifest["error_calls"] = sum(c["status"] == "error" for c in manifest["calls"])
save_json(out / "manifest.json", manifest)
print(out)
