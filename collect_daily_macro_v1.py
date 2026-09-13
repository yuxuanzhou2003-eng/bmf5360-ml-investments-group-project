"""Collect daily macro series from the official FRED CSV endpoint.

The collector is deliberately raw-only.  It writes each HTTP response as
bytes, keeps request/provenance metadata, and never parses or transforms a
series.  Parsing and feature construction are implemented separately so that
the source bytes remain an immutable audit anchor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
SERIES = (
    "VIXCLS",
    "DGS3MO",
    "DGS2",
    "DGS10",
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "DTWEXBGS",
)
BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def header_value(headers: object, name: str) -> str | None:
    try:
        value = headers.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None
    return str(value) if value is not None else None


def save_response(out_dir: Path, series: str, attempt: int, payload: bytes, suffix: str) -> str | None:
    if not payload:
        return None
    path = out_dir / f"fred_{series}.attempt{attempt:02d}.{suffix}"
    path.write_bytes(payload)
    return path.name


def fetch_one(
    series: str,
    start: str,
    end: str,
    out_dir: Path,
    timeout: float,
    retries: int,
) -> dict[str, object]:
    url = f"{BASE_URL}?{urlencode({'id': series, 'cosd': start, 'coed': end})}"
    attempts: list[dict[str, object]] = []
    final_file: str | None = None
    for attempt in range(1, retries + 1):
        request_time = utc_now()
        # FRED occasionally stalls when a custom User-Agent/Accept header is
        # supplied from this environment.  The endpoint is public and the
        # default urllib request is sufficient; retaining the exact URL in
        # the manifest preserves source provenance.
        request = Request(url)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
                status = int(getattr(response, "status", response.getcode()))
                content_type = header_value(response.headers, "Content-Type")
                body_file = save_response(out_dir, series, attempt, payload, "csv")
                attempts.append(
                    {
                        "attempt": attempt,
                        "requested_at_utc": request_time,
                        "status_code": status,
                        "status": "returned" if 200 <= status < 300 else "http_error",
                        "content_type": content_type,
                        "bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                        "response_file": body_file,
                    }
                )
                if 200 <= status < 300:
                    final_file = body_file
                    break
        except HTTPError as exc:
            payload = exc.read()
            body_file = save_response(out_dir, series, attempt, payload, "error.bin")
            attempts.append(
                {
                    "attempt": attempt,
                    "requested_at_utc": request_time,
                    "status_code": int(exc.code),
                    "status": "http_error",
                    "content_type": header_value(exc.headers, "Content-Type"),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "response_file": body_file,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        except (URLError, TimeoutError, OSError) as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "requested_at_utc": request_time,
                    "status_code": None,
                    "status": "network_error",
                    "content_type": None,
                    "bytes": 0,
                    "sha256": None,
                    "response_file": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if attempt < retries:
            time.sleep(float(attempt))

    successful = next((item for item in reversed(attempts) if item["status"] == "returned"), None)
    final = successful or (attempts[-1] if attempts else {})
    return {
        "series": series,
        "url": url,
        "source": "FRED official fredgraph.csv endpoint",
        "retrieved_at_utc": final.get("requested_at_utc"),
        "status": final.get("status", "network_error"),
        "status_code": final.get("status_code"),
        "content_type": final.get("content_type"),
        "bytes": final.get("bytes", 0),
        "sha256": final.get("sha256"),
        "response_file": final_file or final.get("response_file"),
        "attempts": attempts,
        "error": final.get("error"),
        "error_type": final.get("error_type"),
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "series",
        "url",
        "source",
        "retrieved_at_utc",
        "status",
        "status_code",
        "content_type",
        "bytes",
        "sha256",
        "response_file",
        "error_type",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2013-11-01")
    parser.add_argument("--end", default="2026-09-07")
    parser.add_argument("--run-id", default=None, help="UTC run id; defaults to current UTC timestamp")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.retries < 1:
        raise ValueError("--retries must be at least one")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = ROOT / "data" / "raw" / "daily_macro_v1" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema": "daily_macro_collection_v1",
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "source": "FRED official fredgraph.csv endpoint",
        "base_url": BASE_URL,
        "series": list(SERIES),
        "start": args.start,
        "end": args.end,
        "timeout_seconds": args.timeout,
        "max_attempts": args.retries,
        "raw_bytes_policy": "save each non-empty response bytes; no parsing or transformation in collection stage",
    }
    (out_dir / "collection_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = [fetch_one(series, args.start, args.end, out_dir, args.timeout, args.retries) for series in SERIES]
    manifest = {
        **config,
        "finished_at_utc": utc_now(),
        "series_results": rows,
        "returned_count": sum(row["status"] == "returned" for row in rows),
        "failed_count": sum(row["status"] != "returned" for row in rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_manifest(out_dir / "manifest.csv", rows)
    print(json.dumps({"run_id": run_id, "out_dir": str(out_dir), "returned_count": manifest["returned_count"], "failed_count": manifest["failed_count"]}, indent=2))


if __name__ == "__main__":
    main()
