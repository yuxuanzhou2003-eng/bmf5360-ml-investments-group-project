"""Isolated LSEG probe for executable-price and liquidity fields.

The probe serializes each response directly into a new run directory.  It does
not parse, fill, filter, deduplicate, or merge the vendor response.
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
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUT = ROOT / "data" / "raw" / "price_execution_probe_v3" / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)


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

import lseg.data as ld  # noqa: E402


FIELDS = [
    "TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS",
    "BID", "ASK", "TRNOVR_UNS",
]
FULL_ADJUSTMENTS = [
    "exchangeCorrection", "manualCorrection", "CCH", "CRE", "RPO", "RTS",
]
CALLS = [
    {
        "name": "aapl_split_default",
        "universe": ["AAPL.OQ"],
        "fields": FIELDS,
        "interval": "1D",
        "start": "2020-08-24",
        "end": "2020-09-04",
        "note": "Default adjustment behavior around Apple's 2020 stock split.",
    },
    {
        "name": "aapl_split_explicit_adjusted",
        "universe": ["AAPL.OQ"],
        "fields": FIELDS,
        "interval": "1D",
        "start": "2020-08-24",
        "end": "2020-09-04",
        "adjustments": FULL_ADJUSTMENTS,
        "note": "Explicit price/volume corporate-action adjustments around the split.",
    },
    {
        "name": "aapl_split_unadjusted",
        "universe": ["AAPL.OQ"],
        "fields": FIELDS,
        "interval": "1D",
        "start": "2020-08-24",
        "end": "2020-09-04",
        "adjustments": ["unadjusted"],
        "note": "Unadjusted control around the split.",
    },
    {
        "name": "multi_instrument_history_shape",
        "universe": ["AAPL.OQ", "MSFT.OQ", "SPY.P"],
        "fields": FIELDS,
        "interval": "1D",
        "start": "2015-01-20",
        "end": "2015-02-06",
        "adjustments": FULL_ADJUSTMENTS,
        "note": "Confirm multi-instrument output shape and field coverage.",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    manifest = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Field and adjustment probe for execution-price and liquidity inputs.",
        "calls": [],
        "raw_policy": "Direct serialization; no parsing, filling, filtering, deduplication, or merging.",
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
        for spec in CALLS:
            request = {key: value for key, value in spec.items() if key not in {"name", "note"}}
            name = spec["name"]
            request_path = OUT / f"{name}.request.json"
            write_json(request_path, request)
            record = {
                "name": name,
                "note": spec["note"],
                "request_sha256": sha256(request_path),
                "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                frame = ld.get_history(**request)
                csv_path = OUT / f"{name}.csv"
                frame.to_csv(csv_path)
                record.update({
                    "status": "returned" if len(frame) else "empty",
                    "rows": int(len(frame)),
                    "columns": [str(column) for column in frame.columns],
                    "csv_sha256": sha256(csv_path),
                })
            except Exception as exc:
                error_path = OUT / f"{name}.error.txt"
                error_path.write_text(redact(exc), encoding="utf-8")
                record.update({
                    "status": "error",
                    "error": redact(exc)[:1000],
                    "error_sha256": sha256(error_path),
                })
            manifest["calls"].append(record)
            write_json(OUT / "manifest.json", manifest)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        ld.close_session()

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["script"] = Path(__file__).name
    manifest["script_sha256"] = sha256(Path(__file__))
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"run_id": RUN_ID, "out": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
