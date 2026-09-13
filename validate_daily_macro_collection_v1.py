"""Independently validate a daily macro raw FRED collection run.

This validator reads only the collection manifest and raw response bytes.  It
does not read prices, targets, model outputs, or any downstream feature file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_SERIES = (
    "VIXCLS",
    "DGS3MO",
    "DGS2",
    "DGS10",
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "DTWEXBGS",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_raw(path: Path, series: str) -> dict[str, object]:
    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    expected_fields = ["observation_date", series]
    fields = list(reader.fieldnames or [])
    rows = list(reader)
    dates: list[str] = []
    numeric_count = 0
    missing_count = 0
    invalid_values: list[str] = []
    invalid_dates: list[str] = []
    for row in rows:
        date = (row.get("observation_date") or "").strip()
        value = (row.get(series) or "").strip()
        try:
            datetime.strptime(date, "%Y-%m-%d")
            dates.append(date)
        except ValueError:
            invalid_dates.append(date)
        if value in {"", ".", "NA", "NaN", "nan"}:
            missing_count += 1
        else:
            try:
                numeric = float(value)
                if numeric != numeric or numeric in {float("inf"), float("-inf")}:
                    raise ValueError("non-finite")
                numeric_count += 1
            except (TypeError, ValueError):
                invalid_values.append(value)
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "columns": fields,
        "rows": len(rows),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "sorted_dates": dates == sorted(dates),
        "duplicate_dates": len(dates) - len(set(dates)),
        "invalid_dates": invalid_dates,
        "numeric_count": numeric_count,
        "missing_count": missing_count,
        "invalid_values": invalid_values,
        "expected_columns": expected_fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-run-id", default=None)
    args = parser.parse_args()
    collection_dir = ROOT / "data" / "raw" / "daily_macro_v1" / args.run_id
    manifest_path = collection_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_id = args.audit_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = ROOT / "data" / "audit" / "daily_macro_v1" / audit_id
    audit_dir.mkdir(parents=True, exist_ok=False)

    checks: dict[str, bool] = {}
    checks["manifest_schema"] = manifest.get("schema") == "daily_macro_collection_v1"
    checks["manifest_series_exact"] = tuple(manifest.get("series", [])) == EXPECTED_SERIES
    checks["manifest_returned_count"] = manifest.get("returned_count") == len(EXPECTED_SERIES)
    checks["manifest_failed_count_zero"] = manifest.get("failed_count") == 0
    results = {str(row.get("series")): row for row in manifest.get("series_results", [])}
    checks["manifest_result_series_exact"] = set(results) == set(EXPECTED_SERIES)

    series_profiles: dict[str, dict[str, object]] = {}
    rehash_rows: list[dict[str, object]] = []
    for series in EXPECTED_SERIES:
        row = results.get(series, {})
        response_file = row.get("response_file")
        path = collection_dir / str(response_file) if response_file else collection_dir / "missing"
        exists = path.exists() and path.is_file()
        status_ok = row.get("status") == "returned" and row.get("status_code") == 200
        checks[f"{series}_status_200"] = bool(status_ok)
        checks[f"{series}_response_exists"] = bool(exists)
        if exists:
            actual_sha = sha256_file(path)
            actual_bytes = path.stat().st_size
            checks[f"{series}_sha256_matches_manifest"] = actual_sha == row.get("sha256")
            checks[f"{series}_bytes_matches_manifest"] = actual_bytes == row.get("bytes")
            profile = parse_raw(path, series)
            profile["path"] = str(path.relative_to(ROOT))
            profile["content_type"] = row.get("content_type")
            profile["retrieved_at_utc"] = row.get("retrieved_at_utc")
            series_profiles[series] = profile
            checks[f"{series}_columns_exact"] = profile["columns"] == profile["expected_columns"]
            checks[f"{series}_rows_positive"] = int(profile["rows"]) > 0
            checks[f"{series}_dates_valid"] = len(profile["invalid_dates"]) == 0
            checks[f"{series}_dates_sorted"] = bool(profile["sorted_dates"])
            checks[f"{series}_no_duplicate_dates"] = int(profile["duplicate_dates"]) == 0
            checks[f"{series}_values_valid_or_missing"] = len(profile["invalid_values"]) == 0
            rehash_rows.append({"series": series, "path": profile["path"], "bytes": actual_bytes, "sha256": actual_sha, "status": "verified"})
        else:
            series_profiles[series] = {"path": None, "status": "missing"}
            rehash_rows.append({"series": series, "path": None, "bytes": 0, "sha256": None, "status": "missing"})

    with (audit_dir / "raw_rehash.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["series", "path", "bytes", "sha256", "status"])
        writer.writeheader()
        writer.writerows(rehash_rows)
    summary = {
        "schema": "daily_macro_collection_validation_v1",
        "audit_run_id": audit_id,
        "validated_at_utc": utc_now(),
        "collection_run_id": args.run_id,
        "collection_path": str(collection_dir.relative_to(ROOT)),
        "input_manifest_sha256": sha256_file(manifest_path),
        "series_profiles": series_profiles,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "scope": "raw manifest/bytes only; no prices, targets, models, or downstream feature files read",
        "limitations": [
            "FRED current CSV responses are not a complete vintage database; revision risk remains.",
            "This validation checks transport, bytes, schema, date integrity, and numeric/missing tokens; it does not establish economic correctness or release-time availability.",
        ],
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"audit_run_id": audit_id, "all_checks_pass": summary["all_checks_pass"], "audit_dir": str(audit_dir), "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
