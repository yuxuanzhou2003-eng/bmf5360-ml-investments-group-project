"""Isolated AMCR actual-selector probe; direct serialization only."""
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
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "amcr_actual_selector_probe_v3" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)


def redact(value: object) -> str:
    result = str(value)
    for secret in SECRETS:
        result = result.replace(secret, "[REDACTED]")
    return result


class SafeStream:
    def __init__(self, target): self.target = target
    def write(self, message): return self.target.write(redact(message))
    def flush(self): return self.target.flush()
    def isatty(self): return False


sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)
import lseg.data as ld  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def fields(selector: str | None, detailed: bool = False) -> list[str]:
    base = "TR.EPSActValue" if selector is None else f"TR.EPSActValue(ActType={selector})"
    result = [f"{base}.announcedate", f"{base}.periodenddate"]
    if detailed:
        result += [f"{base}.calcdate", f"{base}.fperiod", f"{base}.currency"]
    return [*result, base]


CALLS = [
    ("default_minimal", fields(None)),
    ("reported_minimal", fields("Reported")),
    ("reported_detailed", fields("Reported", detailed=True)),
    ("all_minimal", fields("All")),
    ("latest_minimal", fields("Latest")),
]


def main() -> int:
    manifest = {"run_id": RUN_ID, "calls": [], "policy": "Direct response; no fill/filter/dedup/parse."}
    write_json(OUT / "manifest.json", manifest)
    ld.open_session(name="desktop.workspace", app_key=CREDS.get("LSEG_APP_KEY"))
    try:
        for name, requested_fields in CALLS:
            request = {
                "universe": ["AMCR.N"],
                "fields": requested_fields,
                "parameters": {"SDate": "2019-01-01", "EDate": "2026-09-07", "Frq": "FQ", "Period": "FQ0"},
            }
            request_path = OUT / f"{name}.request.json"
            write_json(request_path, request)
            record = {"name": name, "request_sha256": sha256(request_path)}
            try:
                frame = ld.get_data(**request)
                csv_path = OUT / f"{name}.csv"
                frame.to_csv(csv_path, index=False)
                record.update({
                    "status": "returned", "rows": int(len(frame)),
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
    manifest["script_sha256"] = sha256(Path(__file__))
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"run_id": RUN_ID, "out": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
