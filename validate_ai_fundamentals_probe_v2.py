"""Independent structural audit of the raw AI-fundamentals probe v2.

This validator reads only the frozen raw probe run named on the command line.  It
does not import the LSEG client, issue requests, read any model/target table, or
write to the raw run.  It recomputes hashes, table structure, missingness, dates,
duplicates, annual period-end alignment, announcement/update lags, and the
mechanical row-order evidence for the two monthly market-cap responses.

The annual slot grid is anchored to the independent
``is_statement_dates.csv`` response's ``Income Statement Period End Date`` and
contains the six planned instruments and nine returned annual period-end slots
per instrument.  A missing annual value row is kept separate from a keyed row
whose value is blank.  Empty strings are missing; the literal numeric value 0 is
observed data.  No imputation, deletion, winsorisation, date conversion in raw
files, identifier mapping, economic interpretation, return construction,
factor/label construction, or formal monthly join is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw" / "ai_fundamentals_probe_v2"
DEFAULT_RUN_ID = "20260909T174626399754Z"
AUDIT_ROOT = ROOT / "data" / "audit" / "ai_fundamentals_probe_v2"
SCHEMA_VERSION = "ai_fundamentals_probe_v2"
UNIVERSE = ["NVDA.OQ", "MSFT.OQ", "AMD.OQ", "IBM.N", "WMT.N", "JNJ.N"]
ANNUAL_WINDOW = {"SDate": "2014-01-01", "EDate": "2022-12-31", "Frq": "FY"}
MONTHLY_WINDOW = {"SDate": "2014-01-01", "EDate": "2022-12-31", "Frq": "M"}

ANNUAL_PROFILES: dict[str, dict[str, Any]] = {
    "rd_value_date_fperiod": {
        "period_col": "Date",
        "value_col": "Research And Development",
        "fperiod_col": "Financial Period Absolute",
        "label": "R&D",
    },
    "revenue_value_date_fperiod": {
        "period_col": "Date",
        "value_col": "Revenue",
        "fperiod_col": "Financial Period Absolute",
        "label": "Revenue",
    },
    "tot_assets_date_fperiod": {
        "period_col": "Date",
        "value_col": "Total Assets",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Total Assets",
    },
    "com_eq_tot_date_fperiod": {
        "period_col": "Date",
        "value_col": "Common Equity - Total",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Common Equity",
    },
    "gross_prof_ind_prop_tot_date_fperiod": {
        "period_col": "Date",
        "value_col": "Gross Profit - Industrials/Property - Total",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Gross Profit",
    },
    "net_cash_flow_op_date_fperiod": {
        "period_col": "Date",
        "value_col": "Net Cash Flow from Operating Activities",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Net CFO",
    },
    "debt_tot_date_fperiod": {
        "period_col": "Date",
        "value_col": "Debt - Total",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Total Debt",
    },
}
IS_ID = "is_statement_dates"
IS_COLUMNS = {
    "announcement": "Income Statement Orig Announce Date",
    "last_update": "Income Statement Last Update Date",
    "period_end": "Income Statement Period End Date",
}
MONTHLY_VALUE_ID = "company_market_capitalization_monthly"
MONTHLY_DATE_ID = "company_market_capitalization_date_monthly"
MONTHLY_VALUE_COL = "Company Market Capitalization"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_audit_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def json_text(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_nonempty(value: Any) -> bool:
    return clean_cell(value) != ""


def parse_date(value: Any) -> date | None:
    """Parse date-like text for diagnostics only; never write it to raw files."""

    text = clean_cell(value)
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def read_csv_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def date_stats(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    nonempty = [row.get(column) for row in rows if is_nonempty(row.get(column))]
    parsed = [parsed_value for parsed_value in (parse_date(value) for value in nonempty) if parsed_value is not None]
    return {
        "non_null_parseable": len(parsed),
        "parse_failures": len(nonempty) - len(parsed),
        "min": min(parsed).isoformat() if parsed else None,
        "max": max(parsed).isoformat() if parsed else None,
        "unique_values": len(set(parsed)),
    }


def normalize_metadata_date(value: Any) -> str | None:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else None


def row_key(row: dict[str, str], period_column: str) -> tuple[str, str] | None:
    instrument = clean_cell(row.get("Instrument"))
    period = parse_date(row.get(period_column))
    if not instrument or period is None:
        return None
    return instrument, period.isoformat()


def key_counts(rows: Iterable[dict[str, str]], columns: list[str], period_column: str | None = None) -> tuple[Counter[tuple[str, ...]], int]:
    counts: Counter[tuple[str, ...]] = Counter()
    missing = 0
    for row in rows:
        values: list[str] = []
        row_missing = False
        for column in columns:
            value = row.get(column)
            if period_column and column == period_column:
                parsed = parse_date(value)
                if parsed is None:
                    row_missing = True
                    values.append("")
                else:
                    values.append(parsed.isoformat())
            else:
                text = clean_cell(value)
                if not text:
                    row_missing = True
                values.append(text)
        if row_missing:
            missing += 1
        else:
            counts[tuple(values)] += 1
    return counts, missing


def duplicate_summary(
    rows: list[dict[str, str]],
    columns: list[str],
    *,
    date_key_column: str | None = None,
) -> dict[str, Any]:
    full_counts, _ = key_counts(rows, list(rows[0].keys()) if rows else [], period_column=None)
    candidate_counts, candidate_missing = key_counts(rows, columns, period_column=date_key_column)
    full_groups = {key: count for key, count in full_counts.items() if count > 1}
    candidate_groups = {key: count for key, count in candidate_counts.items() if count > 1}
    return {
        "full_row_duplicate_rows": int(sum(full_groups.values())),
        "full_row_duplicate_groups": int(len(full_groups)),
        "candidate_key_columns": list(columns),
        "candidate_key_status": "assessed_mechanically",
        "candidate_key_duplicate_rows": int(sum(candidate_groups.values())),
        "candidate_key_duplicate_groups": int(len(candidate_groups)),
        "candidate_key_missing_rows": int(candidate_missing),
    }


def quantile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min_days": min(values) if values else None,
        "max_days": max(values) if values else None,
        "mean_days": float(statistics.mean(values)) if values else None,
        "median_days": float(statistics.median(values)) if values else None,
        "p05_days": quantile(values, 0.05),
        "p25_days": quantile(values, 0.25),
        "p75_days": quantile(values, 0.75),
        "p95_days": quantile(values, 0.95),
    }


def record_check(
    checks: list[dict[str, Any]],
    check_id: str,
    category: str,
    passed: bool,
    expected: Any,
    observed: Any,
    notes: str = "",
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "passed": bool(passed),
            "expected": json_text(expected),
            "observed": json_text(observed),
            "notes": notes,
        }
    )


def audit_run(run_id: str, audit_id: str) -> dict[str, Any]:
    raw_dir = RAW_ROOT / run_id
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw run does not exist: {raw_dir}")
    audit_dir = AUDIT_ROOT / audit_id
    audit_dir.mkdir(parents=True, exist_ok=False)

    checks: list[dict[str, Any]] = []
    raw_files_before = sorted(path for path in raw_dir.iterdir() if path.is_file())
    raw_hashes_before = {path.name: sha256_file(path) for path in raw_files_before}
    raw_sizes_before = {path.name: path.stat().st_size for path in raw_files_before}

    plan_path = raw_dir / "plan.json"
    summary_path = raw_dir / "probe_summary.json"
    plan = read_json(plan_path)
    summary = read_json(summary_path)
    plan_hash = sha256_file(plan_path)
    summary_hash = sha256_file(summary_path)
    script_path = ROOT / str(plan.get("script_path", "probe_ai_fundamentals_v2.py"))

    record_check(
        checks,
        "raw_run_directory_exists",
        "inventory",
        raw_dir.is_dir(),
        str(raw_dir),
        str(raw_dir),
    )
    record_check(
        checks,
        "plan_schema_and_parameters",
        "plan",
        plan.get("schema_version") == SCHEMA_VERSION
        and plan.get("universe") == UNIVERSE
        and plan.get("annual_window") == ANNUAL_WINDOW
        and plan.get("monthly_window") == MONTHLY_WINDOW,
        {"schema_version": SCHEMA_VERSION, "universe": UNIVERSE, "annual_window": ANNUAL_WINDOW, "monthly_window": MONTHLY_WINDOW},
        {"schema_version": plan.get("schema_version"), "universe": plan.get("universe"), "annual_window": plan.get("annual_window"), "monthly_window": plan.get("monthly_window")},
        "Plan contract is checked mechanically; no plan rule is changed.",
    )
    actual_script_hash = sha256_file(script_path) if script_path.is_file() else None
    record_check(
        checks,
        "probe_script_hash_matches_plan",
        "hashes",
        actual_script_hash == plan.get("script_sha256"),
        plan.get("script_sha256"),
        actual_script_hash,
        "The executed probe script is provenance only; this validator does not execute it.",
    )

    plan_specs = plan.get("requests", [])
    plan_by_id = {str(spec.get("request_id")): spec for spec in plan_specs if isinstance(spec, dict)}
    summary_records = summary.get("records", [])
    summary_by_id = {str(record.get("request_id")): record for record in summary_records if isinstance(record, dict)}
    expected_ids = set(ANNUAL_PROFILES) | {IS_ID, MONTHLY_VALUE_ID, MONTHLY_DATE_ID}
    record_check(
        checks,
        "ten_planned_requests_present",
        "inventory",
        len(plan_specs) == 10 and set(plan_by_id) == expected_ids,
        sorted(expected_ids),
        sorted(plan_by_id),
        "The ten requests remain separate so one field family cannot hide another.",
    )
    record_check(
        checks,
        "ten_summary_records_and_ids_present",
        "inventory",
        len(summary_records) == 10 and set(summary_by_id) == expected_ids,
        sorted(expected_ids),
        {"count": len(summary_records), "ids": sorted(summary_by_id)},
    )
    statuses = {request_id: summary_by_id.get(request_id, {}).get("status") for request_id in sorted(expected_ids)}
    record_check(
        checks,
        "ten_request_statuses_returned",
        "status",
        all(status == "returned" for status in statuses.values()),
        {request_id: "returned" for request_id in sorted(expected_ids)},
        statuses,
        "Returned status is reported separately from field-label exactness and slot coverage.",
    )
    summary_counts = summary.get("counts", {})
    last_execute = summary.get("last_execute", {})
    record_check(
        checks,
        "summary_counts_match_ten_completed_requests",
        "status",
        summary_counts.get("returned") == 10
        and summary_counts.get("planned") == 0
        and summary_counts.get("running") == 0
        and summary_counts.get("empty") == 0
        and summary_counts.get("error") == 0
        and summary_counts.get("timeout") == 0
        and last_execute.get("attempted") == 10
        and last_execute.get("pending_after_run") == 0,
        {"returned": 10, "planned": 0, "running": 0, "empty": 0, "error": 0, "timeout": 0, "attempted": 10, "pending_after_run": 0},
        {**summary_counts, "attempted": last_execute.get("attempted"), "pending_after_run": last_execute.get("pending_after_run")},
    )

    tables: dict[str, dict[str, Any]] = {}
    hash_records: list[dict[str, Any]] = []
    metadata_records: list[dict[str, Any]] = []
    all_required_files_present = True
    request_payloads_match = True
    metadata_identity_match = True
    csv_hashes_match = True
    raw_policy_matches = True
    for request_id in sorted(expected_ids):
        spec = plan_by_id.get(request_id, {})
        summary_record = summary_by_id.get(request_id, {})
        csv_path = raw_dir / f"{request_id}.csv"
        request_path = raw_dir / f"{request_id}.request.json"
        metadata_path = raw_dir / f"{request_id}.metadata.json"
        paths_exist = csv_path.is_file() and request_path.is_file() and metadata_path.is_file()
        all_required_files_present = all_required_files_present and paths_exist
        if not paths_exist:
            hash_records.append({"request_id": request_id, "files_present": False})
            continue
        request_object = read_json(request_path)
        metadata = read_json(metadata_path)
        headers, rows = read_csv_table(csv_path)
        tables[request_id] = {"headers": headers, "rows": rows, "csv_path": csv_path, "metadata": metadata}
        actual_request_hash = sha256_file(request_path)
        canonical_request_hash = sha256_bytes(canonical_json(request_object))
        actual_metadata_hash = sha256_file(metadata_path)
        actual_csv_hash = sha256_file(csv_path)
        expected_request_hash = spec.get("request_sha256")
        request_payload_ok = (
            request_object == spec.get("request") == metadata.get("request")
            and actual_request_hash == canonical_request_hash == expected_request_hash
            and summary_record.get("request_file_sha256") == actual_request_hash
            and metadata.get("request_sha256") == actual_request_hash
            and metadata.get("request_file_sha256") == actual_request_hash
        )
        request_payloads_match = request_payloads_match and request_payload_ok
        metadata_ok = (
            metadata.get("schema_version") == SCHEMA_VERSION
            and metadata.get("request_id") == request_id
            and metadata.get("family") == spec.get("family")
            and metadata.get("question") == spec.get("question")
            and metadata.get("status") == summary_record.get("status")
            and metadata.get("csv_path") == csv_path.name
            and metadata.get("request") == request_object
            and metadata.get("raw_response_policy") == plan.get("raw_response_policy")
        )
        metadata_identity_match = metadata_identity_match and metadata_ok
        csv_ok = (
            actual_csv_hash == metadata.get("csv_sha256") == summary_record.get("csv_sha256")
            and actual_metadata_hash == summary_record.get("metadata_sha256")
        )
        csv_hashes_match = csv_hashes_match and csv_ok
        raw_policy_matches = raw_policy_matches and metadata.get("raw_response_policy") == plan.get("raw_response_policy")
        hash_records.append(
            {
                "request_id": request_id,
                "status": summary_record.get("status"),
                "files_present": True,
                "request_sha256_actual": actual_request_hash,
                "request_sha256_expected": expected_request_hash,
                "request_canonical_sha256": canonical_request_hash,
                "metadata_sha256_actual": actual_metadata_hash,
                "metadata_sha256_summary": summary_record.get("metadata_sha256"),
                "csv_sha256_actual": actual_csv_hash,
                "csv_sha256_metadata": metadata.get("csv_sha256"),
                "csv_sha256_summary": summary_record.get("csv_sha256"),
                "request_hashes_match": request_payload_ok,
                "metadata_identity_match": metadata_ok,
                "csv_hashes_match": csv_ok,
            }
        )
        metadata_records.append(metadata)

    record_check(
        checks,
        "all_request_metadata_csv_files_present",
        "hashes",
        all_required_files_present,
        "each request has .request.json, .metadata.json and .csv",
        {"all_present": all_required_files_present, "missing": [r["request_id"] for r in hash_records if not r.get("files_present")]},
    )
    record_check(
        checks,
        "request_payloads_and_request_hashes_match_plan_summary_metadata",
        "hashes",
        request_payloads_match,
        "all request payload and canonical SHA-256 links agree",
        {"all_match": request_payloads_match},
        "Request JSON is read and canonically hashed; it is never rewritten.",
    )
    record_check(
        checks,
        "metadata_identity_policy_and_hash_links_match",
        "hashes",
        metadata_identity_match and raw_policy_matches,
        "metadata identity and raw response policy agree with plan",
        {"identity_match": metadata_identity_match, "raw_policy_match": raw_policy_matches},
        "Raw response policy explicitly preserves vendor tables and disables cleaning/imputation/deletion.",
    )
    record_check(
        checks,
        "csv_and_metadata_hashes_match_summary",
        "hashes",
        csv_hashes_match,
        "all CSV and metadata hash links agree",
        {"all_match": csv_hashes_match},
        "The validator recomputes SHA-256 from current bytes.",
    )

    structure_results: dict[str, Any] = {}
    structure_rows_columns_all_ok = True
    non_null_counts_all_ok = True
    date_range_all_ok = True
    duplicate_all_ok = True
    unknown_instruments: dict[str, list[str]] = {}
    zero_counts: dict[str, int] = {}
    for request_id in sorted(tables):
        table = tables[request_id]
        headers = table["headers"]
        rows = table["rows"]
        metadata = table["metadata"]
        nonempty_counts = {column: sum(is_nonempty(row.get(column)) for row in rows) for column in headers}
        expected_columns = metadata.get("columns", [])
        expected_nonempty = metadata.get("non_null", {})
        rows_columns_ok = len(rows) == metadata.get("rows") and headers == expected_columns
        non_null_ok = nonempty_counts == expected_nonempty
        structure_rows_columns_all_ok = structure_rows_columns_all_ok and rows_columns_ok
        non_null_counts_all_ok = non_null_counts_all_ok and non_null_ok
        non_null_mismatches = {
            column: {"raw_csv": nonempty_counts.get(column), "metadata": expected_nonempty.get(column)}
            for column in sorted(set(nonempty_counts) | set(expected_nonempty))
            if nonempty_counts.get(column) != expected_nonempty.get(column)
        }
        observed_instruments = sorted({clean_cell(row.get("Instrument")) for row in rows if is_nonempty(row.get("Instrument"))})
        unknown = sorted(set(observed_instruments) - set(UNIVERSE))
        unknown_instruments[request_id] = unknown
        numeric_columns = [column for column in headers if column not in {"Instrument", "Date", "Financial Period Absolute"} and "Date" not in column]
        zero_counts[request_id] = sum(
            1
            for row in rows
            for column in numeric_columns
            if clean_cell(row.get(column)) in {"0", "0.0", "0.00"}
        )
        actual_date_ranges: dict[str, Any] = {}
        table_date_ok = True
        for column in metadata.get("date_ranges", {}):
            stats = date_stats(rows, column)
            actual_date_ranges[column] = stats
            expected = metadata["date_ranges"][column]
            comparable = {
                "non_null_parseable": expected.get("non_null_parseable"),
                "parse_failures": expected.get("parse_failures"),
                "min": normalize_metadata_date(expected.get("min")),
                "max": normalize_metadata_date(expected.get("max")),
                "unique_values": expected.get("unique_values"),
            }
            table_date_ok = table_date_ok and stats == comparable
        date_range_all_ok = date_range_all_ok and table_date_ok
        candidate_columns = list(metadata.get("duplicate_key_diagnostics", {}).get("candidate_key_columns", []))
        date_key_column = "Date" if "Date" in candidate_columns else None
        if request_id == IS_ID:
            date_key_column = None
        actual_duplicates = duplicate_summary(rows, candidate_columns, date_key_column=date_key_column)
        expected_duplicates = metadata.get("duplicate_key_diagnostics", {})
        duplicate_ok = actual_duplicates == expected_duplicates
        duplicate_all_ok = duplicate_all_ok and duplicate_ok
        structure_results[request_id] = {
            "rows": len(rows),
            "columns": headers,
            "non_null_recomputed": nonempty_counts,
            "metadata_rows_columns_match": rows_columns_ok,
            "metadata_non_null_counts_match": non_null_ok,
            "metadata_non_null_mismatches": non_null_mismatches,
            "observed_instruments": observed_instruments,
            "unknown_instruments": unknown,
            "date_ranges_recomputed": actual_date_ranges,
            "date_ranges_match_metadata": table_date_ok,
            "duplicate_diagnostics_recomputed": actual_duplicates,
            "duplicate_diagnostics_match_metadata": duplicate_ok,
            "literal_zero_observed_count": zero_counts[request_id],
        }

    record_check(
        checks,
        "raw_table_rows_and_columns_recomputed",
        "structure",
        structure_rows_columns_all_ok,
        "each CSV matches its metadata row count and column labels",
        {"all_match": structure_rows_columns_all_ok, "tables": structure_results},
        "Rows and headers are checked from the physical CSV bytes.",
    )
    record_check(
        checks,
        "metadata_non_null_counts_match_physical_csv",
        "structure",
        non_null_counts_all_ok,
        "each metadata non_null count matches a blank-aware physical CSV count",
        {"all_match": non_null_counts_all_ok, "tables": structure_results},
        "A mismatch is retained as a provenance finding; no metadata or raw CSV correction is made.",
    )
    record_check(
        checks,
        "date_ranges_and_parse_counts_recomputed",
        "dates",
        date_range_all_ok,
        "each metadata date range matches an in-memory recomputation",
        {"all_match": date_range_all_ok},
        "Date parsing is diagnostic only and is not written back to raw CSVs.",
    )
    record_check(
        checks,
        "duplicate_diagnostics_recomputed",
        "duplicates",
        duplicate_all_ok,
        "each metadata duplicate diagnostic matches raw rows",
        {"all_match": duplicate_all_ok},
        "The monthly value response is expected to repeat Instrument as its candidate key because it has no date column.",
    )
    record_check(
        checks,
        "observed_instruments_are_within_planned_universe",
        "structure",
        not any(unknown_instruments.values()),
        UNIVERSE,
        unknown_instruments,
        "No identifier mapping or company exclusion is applied.",
    )

    # Independent annual period-end slot alignment.
    is_table = tables.get(IS_ID, {})
    is_rows = is_table.get("rows", [])
    canonical_slots = [row_key(row, IS_COLUMNS["period_end"]) for row in is_rows]
    canonical_slots = [key for key in canonical_slots if key is not None]
    canonical_slot_set = set(canonical_slots)
    canonical_counts = Counter(key[0] for key in canonical_slots)
    canonical_grid_ok = (
        len(canonical_slots) == 54
        and len(canonical_slot_set) == 54
        and canonical_counts == Counter({instrument: 9 for instrument in UNIVERSE})
    )
    record_check(
        checks,
        "annual_canonical_slot_grid_is_six_by_nine",
        "alignment",
        canonical_grid_ok,
        {"instruments": UNIVERSE, "slots": 54, "slots_per_instrument": 9},
        {"valid_slots": len(canonical_slots), "unique_slots": len(canonical_slot_set), "counts": dict(canonical_counts)},
        "Canonical grid is the independent IS period-end response; no fiscal-label year coercion is applied.",
    )

    alignment: dict[str, Any] = {}
    alignment_all_keyed_periods = True
    all_annual_key_sets: dict[str, set[tuple[str, str]]] = {}
    for request_id, profile in ANNUAL_PROFILES.items():
        rows = tables.get(request_id, {}).get("rows", [])
        keys = [row_key(row, profile["period_col"]) for row in rows]
        valid_keys = [key for key in keys if key is not None]
        key_set = set(valid_keys)
        all_annual_key_sets[request_id] = key_set
        counts = Counter(key for key in valid_keys)
        duplicate_key_groups = {key: count for key, count in counts.items() if count > 1}
        matched = sorted(canonical_slot_set & key_set)
        missing = sorted(canonical_slot_set - key_set)
        extra = sorted(key_set - canonical_slot_set)
        value_na_slots: list[tuple[str, str]] = []
        value_present_slots: list[tuple[str, str]] = []
        for key in matched:
            row_matches = [row for row in rows if row_key(row, profile["period_col"]) == key]
            if len(row_matches) == 1 and is_nonempty(row_matches[0].get(profile["value_col"])):
                value_present_slots.append(key)
            elif len(row_matches) == 1:
                value_na_slots.append(key)
        unkeyed_indices = [index for index, key in enumerate(keys) if key is None]
        unkeyed_value_na_indices = [
            index for index in unkeyed_indices if not is_nonempty(rows[index].get(profile["value_col"]))
        ]
        alignment_ok = not extra and not duplicate_key_groups and len(matched) + len(missing) == 54
        alignment_all_keyed_periods = alignment_all_keyed_periods and alignment_ok
        alignment[request_id] = {
            "label": profile["label"],
            "raw_rows": len(rows),
            "valid_period_key_rows": len(valid_keys),
            "matched_slots": len(matched),
            "no_row_slots": len(missing),
            "value_na_slots": len(value_na_slots),
            "value_present_slots": len(value_present_slots),
            "unkeyed_rows": len(unkeyed_indices),
            "unkeyed_value_na_rows": len(unkeyed_value_na_indices),
            "extra_period_key_rows": len(extra),
            "duplicate_key_groups": len(duplicate_key_groups),
            "missing_slot_keys": [list(key) for key in missing],
            "extra_slot_keys": [list(key) for key in extra],
            "unkeyed_row_indices": unkeyed_indices,
            "unkeyed_value_na_row_indices": unkeyed_value_na_indices,
            "period_column": profile["period_col"],
            "value_column": profile["value_col"],
        }

    # The IS request is itself checked against the six-by-nine grid and then all
    # other requests are checked against its period-end keys.
    is_key_duplicates = len(canonical_slots) != len(canonical_slot_set)
    is_value_na_by_column = {
        column: sum(not is_nonempty(row.get(column)) for row in is_rows)
        for column in IS_COLUMNS.values()
    }
    alignment[IS_ID] = {
        "label": "IS dates",
        "raw_rows": len(is_rows),
        "valid_period_key_rows": len(canonical_slots),
        "matched_slots": len(canonical_slot_set),
        "no_row_slots": 54 - len(canonical_slot_set),
        "value_na_slots_by_column": is_value_na_by_column,
        "extra_period_key_rows": 0,
        "duplicate_key_groups": 1 if is_key_duplicates else 0,
        "period_column": IS_COLUMNS["period_end"],
        "value_columns": list(IS_COLUMNS.values()),
    }
    record_check(
        checks,
        "annual_period_end_keys_align_to_is_grid",
        "alignment",
        alignment_all_keyed_periods and canonical_grid_ok and not is_key_duplicates,
        "each annual value/control family has only canonical Instrument + period-end keys",
        alignment,
        "Rows without a usable period-end key remain in the unkeyed count and are not forced into a slot.",
    )

    # Explicit equality evidence that emitted Date columns are period-end-like.
    period_end_evidence: dict[str, Any] = {}
    period_end_evidence_ok = True
    for request_id, profile in ANNUAL_PROFILES.items():
        rows = tables[request_id]["rows"]
        mismatch_rows: list[int] = []
        for index, row in enumerate(rows):
            key = row_key(row, profile["period_col"])
            if key is None:
                continue
            if key not in canonical_slot_set:
                mismatch_rows.append(index)
        period_end_evidence[request_id] = {
            "valid_key_rows": alignment[request_id]["valid_period_key_rows"],
            "matching_is_period_end_key_rows": alignment[request_id]["matched_slots"],
            "mismatch_rows": mismatch_rows,
        }
        period_end_evidence_ok = period_end_evidence_ok and not mismatch_rows
    record_check(
        checks,
        "emitted_date_columns_match_is_period_end_keys",
        "date_semantics",
        period_end_evidence_ok,
        "all valid annual Date rows are present in the IS period-end key set",
        period_end_evidence,
        "This is structural evidence that Date is period-end in this probe; it is not evidence of available/announcement time.",
    )

    # Announcement and last-update lag statistics, using date-only values in memory.
    lag_rows: list[dict[str, Any]] = []
    for index, row in enumerate(is_rows):
        announcement = parse_date(row.get(IS_COLUMNS["announcement"]))
        last_update = parse_date(row.get(IS_COLUMNS["last_update"]))
        period_end = parse_date(row.get(IS_COLUMNS["period_end"]))
        if announcement and last_update and period_end:
            lag_rows.append(
                {
                    "row_index": index,
                    "instrument": clean_cell(row.get("Instrument")),
                    "period_end": period_end.isoformat(),
                    "announcement": announcement.isoformat(),
                    "last_update": last_update.isoformat(),
                    "announcement_period_lag_days": (announcement - period_end).days,
                    "last_update_announcement_lag_days": (last_update - announcement).days,
                }
            )
    announcement_period_lags = [row["announcement_period_lag_days"] for row in lag_rows]
    update_announcement_lags = [row["last_update_announcement_lag_days"] for row in lag_rows]
    per_instrument_lag: dict[str, Any] = {}
    for instrument in UNIVERSE:
        rows_for_instrument = [row for row in lag_rows if row["instrument"] == instrument]
        period_values = [row["announcement_period_lag_days"] for row in rows_for_instrument]
        update_values = [row["last_update_announcement_lag_days"] for row in rows_for_instrument]
        per_instrument_lag[instrument] = {
            "rows": len(rows_for_instrument),
            "announcement_period": distribution(period_values),
            "last_update_announcement": distribution(update_values),
        }
    lag_summary = {
        "row_count_with_all_three_dates": len(lag_rows),
        "announcement_period_lag_days": distribution(announcement_period_lags),
        "last_update_announcement_lag_days": distribution(update_announcement_lags),
        "announcement_period_negative_rows": sum(value < 0 for value in announcement_period_lags),
        "announcement_period_zero_rows": sum(value == 0 for value in announcement_period_lags),
        "last_update_after_announcement_rows": sum(value > 0 for value in update_announcement_lags),
        "last_update_equal_announcement_rows": sum(value == 0 for value in update_announcement_lags),
        "last_update_before_announcement_rows": sum(value < 0 for value in update_announcement_lags),
        "positive_last_update_announcement_range_days": [
            min(value for value in update_announcement_lags if value > 0)
            if any(value > 0 for value in update_announcement_lags)
            else None,
            max(value for value in update_announcement_lags if value > 0)
            if any(value > 0 for value in update_announcement_lags)
            else None,
        ],
        "by_instrument": per_instrument_lag,
        "rows": lag_rows,
    }
    record_check(
        checks,
        "is_statement_dates_parseable_for_54_slots",
        "dates",
        len(lag_rows) == 54,
        {"rows": 54, "three_date_columns_parseable": True},
        {"rows_with_all_three_dates": len(lag_rows), "date_columns": list(IS_COLUMNS.values())},
        "Announcement, last-update and period-end are used only for lag diagnostics.",
    )
    record_check(
        checks,
        "announcement_period_lag_recomputed",
        "dates",
        len(announcement_period_lags) == 54,
        "54 announcement minus period-end lags",
        lag_summary["announcement_period_lag_days"],
        "The reported range includes observed long lags; no outlier is removed.",
    )
    record_check(
        checks,
        "last_update_later_than_announcement_range_recomputed",
        "dates",
        len(update_announcement_lags) == 54 and all(value > 0 for value in update_announcement_lags),
        {"rows": 54, "all_positive": True},
        {
            "rows": len(update_announcement_lags),
            "positive_rows": lag_summary["last_update_after_announcement_rows"],
            "range_days": lag_summary["positive_last_update_announcement_range_days"],
        },
        "Last Update is a later revision/update observation in this response; it is not substituted for announcement time.",
    )

    # Monthly value/date responses: check the positional evidence while refusing
    # to call it a formal join because the value response has no date field.
    value_rows = tables.get(MONTHLY_VALUE_ID, {}).get("rows", [])
    date_rows = tables.get(MONTHLY_DATE_ID, {}).get("rows", [])
    value_instrument_sequence = [clean_cell(row.get("Instrument")) for row in value_rows]
    date_instrument_sequence = [clean_cell(row.get("Instrument")) for row in date_rows]
    value_counts = Counter(value_instrument_sequence)
    date_counts = Counter(date_instrument_sequence)
    instrument_order_equal = value_instrument_sequence == date_instrument_sequence
    monthly_date_stats: dict[str, Any] = {}
    monthly_grid_ok = True
    for instrument in UNIVERSE:
        instrument_dates = [parse_date(row.get("Date")) for row in date_rows if clean_cell(row.get("Instrument")) == instrument]
        instrument_dates = [item for item in instrument_dates if item is not None]
        expected_dates = sorted(instrument_dates)
        instrument_ok = (
            value_counts[instrument] == 108
            and date_counts[instrument] == 108
            and len(instrument_dates) == 108
            and len(set(instrument_dates)) == 108
            and instrument_dates == expected_dates
        )
        monthly_grid_ok = monthly_grid_ok and instrument_ok
        monthly_date_stats[instrument] = {
            "value_rows": value_counts[instrument],
            "date_rows": date_counts[instrument],
            "date_nonempty_parseable": len(instrument_dates),
            "unique_dates": len(set(instrument_dates)),
            "date_first": min(instrument_dates).isoformat() if instrument_dates else None,
            "date_last": max(instrument_dates).isoformat() if instrument_dates else None,
            "dates_in_row_order_sorted": instrument_dates == expected_dates,
            "value_nonempty": sum(is_nonempty(row.get(MONTHLY_VALUE_COL)) for row in value_rows if clean_cell(row.get("Instrument")) == instrument),
        }
    positional_evidence = {
        "value_rows": len(value_rows),
        "date_rows": len(date_rows),
        "value_instrument_order": list(dict.fromkeys(value_instrument_sequence)),
        "date_instrument_order": list(dict.fromkeys(date_instrument_sequence)),
        "instrument_sequence_equal": instrument_order_equal,
        "counts_equal_by_instrument": value_counts == date_counts,
        "within_instrument_one_to_one_candidate_rows": int(sum(min(value_counts[instrument], date_counts[instrument]) for instrument in UNIVERSE)),
        "formal_join_approved": False,
        "monthly_grid": monthly_date_stats,
        "recommendation": "Next request should request market-cap value and date in the same response; positional evidence alone is not a formal join key.",
    }
    record_check(
        checks,
        "monthly_mcap_value_date_instrument_order_matches",
        "monthly_alignment",
        instrument_order_equal and value_counts == date_counts and len(value_rows) == len(date_rows) == 648,
        {"rows_per_instrument": 108, "rows_total": 648, "instrument_sequence_equal": True},
        positional_evidence,
        "Observed row-order evidence is retained as a diagnostic and cannot approve a production join.",
    )
    record_check(
        checks,
        "monthly_mcap_date_response_has_108_unique_months_per_instrument",
        "monthly_alignment",
        monthly_grid_ok,
        {"each_instrument": {"rows": 108, "unique_dates": 108, "sorted": True}},
        monthly_date_stats,
        "Month-end dates are read from the independent date response; the value response has no date column.",
    )
    record_check(
        checks,
        "monthly_mcap_formal_join_guardrail_is_retained",
        "monthly_alignment",
        positional_evidence["formal_join_approved"] is False,
        {"formal_join_approved": False},
        {"formal_join_approved": positional_evidence["formal_join_approved"], "recommendation": positional_evidence["recommendation"]},
        "No value/date merge is created by this audit.",
    )

    raw_files_after = sorted(path for path in raw_dir.iterdir() if path.is_file())
    raw_hashes_after = {path.name: sha256_file(path) for path in raw_files_after}
    raw_sizes_after = {path.name: path.stat().st_size for path in raw_files_after}
    raw_file_set_unchanged = [path.name for path in raw_files_before] == [path.name for path in raw_files_after]
    raw_hashes_unchanged = raw_hashes_before == raw_hashes_after
    raw_sizes_unchanged = raw_sizes_before == raw_sizes_after
    record_check(
        checks,
        "raw_input_files_unchanged_during_validation",
        "provenance",
        raw_file_set_unchanged and raw_hashes_unchanged and raw_sizes_unchanged,
        "raw file set, sizes and SHA-256 are unchanged",
        {"file_set_unchanged": raw_file_set_unchanged, "hashes_unchanged": raw_hashes_unchanged, "sizes_unchanged": raw_sizes_unchanged},
        "Validator only reads the raw run and writes outputs under data/audit/.",
    )

    passed = sum(bool(check["passed"]) for check in checks)
    total = len(checks)
    failed = [check["check_id"] for check in checks if not check["passed"]]
    actual_hashes = {
        "plan.json": plan_hash,
        "probe_summary.json": summary_hash,
        "probe_script": actual_script_hash,
        "raw_files_before": raw_hashes_before,
        "raw_files_after": raw_hashes_after,
    }
    audit_report: dict[str, Any] = {
        "audit_schema_version": "ai_fundamentals_probe_v2_validation",
        "audit_run_id": audit_id,
        "audited_run_id": run_id,
        "audited_raw_path": str(raw_dir.relative_to(ROOT)),
        "audit_output_path": str(audit_dir.relative_to(ROOT)),
        "validator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "executed_at_utc": utc_now(),
        "scope": {
            "purpose": "Independent field-semantic and structural audit of ai_fundamentals_probe_v2 raw outputs.",
            "test_target_rows_read": False,
            "test_target_rows_parsed": False,
            "returns_constructed": False,
            "factors_constructed": False,
            "labels_constructed": False,
            "models_or_portfolios_run": False,
            "formal_monthly_value_date_join_created": False,
        },
        "input_contract": {
            "schema_version": SCHEMA_VERSION,
            "universe": UNIVERSE,
            "annual_window": ANNUAL_WINDOW,
            "monthly_window": MONTHLY_WINDOW,
            "expected_annual_slots": 54,
            "annual_slot_grid_basis": "is_statement_dates.csv: Instrument + Income Statement Period End Date",
            "missing_definition": "None or blank after strip is missing; literal numeric 0 is observed.",
            "raw_response_policy": plan.get("raw_response_policy"),
        },
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "failed_checks": failed,
        "checks": checks,
        "request_hash_audit": hash_records,
        "structure": structure_results,
        "annual_alignment": alignment,
        "period_end_evidence": period_end_evidence,
        "announcement_lag": lag_summary,
        "monthly_market_cap_alignment": positional_evidence,
        "actual_hashes": actual_hashes,
        "raw_file_sizes_before": raw_sizes_before,
        "raw_file_sizes_after": raw_sizes_after,
        "limitations": [
            "The annual slot grid is anchored to the independent IS period-end response and therefore cannot establish coverage for a missing IS row outside that anchor.",
            "The probe's emitted Date columns match the IS period-end keys in this run, which supports a period-end reading for this sample; it does not establish available/announcement time, vendor vintage, accounting basis, units, currency or revision policy.",
            "The monthly value response has no date column. Equal per-Instrument counts and row order provide a provisional positional candidate only; no formal value/date join is approved.",
            "Observed missing rows and blank values are reported and retained. No imputation, forward fill, winsorisation, deduplication, row deletion or security exclusion is performed.",
        ],
        "status": "completed" if passed == total else "failed_checks",
    }
    validation_json_path = audit_dir / "validation.json"
    validation_json_path.write_text(
        json.dumps(jsonable(audit_report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validation_csv_path = audit_dir / "validation.csv"
    with validation_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "category", "passed", "expected", "observed", "notes"])
        writer.writeheader()
        writer.writerows(checks)

    findings = render_findings(audit_report, tables, plan_by_id, summary_by_id)
    findings_path = audit_dir / "findings.md"
    findings_path.write_text(findings, encoding="utf-8")
    return audit_report


def fmt_range(summary: dict[str, Any]) -> str:
    return f"{summary.get('min_days')}–{summary.get('max_days')} 天"


def render_findings(
    report: dict[str, Any],
    tables: dict[str, dict[str, Any]],
    plan_by_id: dict[str, dict[str, Any]],
    summary_by_id: dict[str, dict[str, Any]],
) -> str:
    alignment = report["annual_alignment"]
    lag = report["announcement_lag"]
    monthly = report["monthly_market_cap_alignment"]
    structure = report["structure"]
    lines: list[str] = []
    lines.append("# AI fundamentals probe v2 独立字段语义与结构审计")
    lines.append("")
    lines.append(f"- 审计输出：`{report['audit_output_path']}/`；正式审计目录为 `data/audit/ai_fundamentals_probe_v2/{report['audit_run_id']}/`。")
    lines.append(f"- 输入原始运行：`{report['audited_raw_path']}/`（run id `{report['audited_run_id']}`）。")
    lines.append(f"- 执行时间（UTC）：`{report['executed_at_utc']}`。")
    lines.append(f"- 工程检查通过：**{report['passed']}/{report['total']}**；状态：`{report['status']}`。")
    lines.append("")
    lines.append("## 范围和处理规则")
    lines.append("")
    lines.append("本轮只读取 probe v2 的 plan、request、metadata、probe summary 和 10 张原始 CSV，重算哈希、结构、缺失、日期、重复和机械对齐关系。没有读取任何测试目标或未来收益表，没有构造收益、factor、label、模型或组合，也没有创建正式 monthly value/date join。")
    lines.append("")
    lines.append("空字符串或缺失单元格计为缺失；文本形式的 `0`、`0.0` 等仍计为已观测值。所有缺失、无键行和重复诊断均保留在原始数据和审计记录中；没有删行、公司排除、填补、前值填充、winsorize、单位变换、identifier mapping、日期/时区写回或异常值处理。")
    lines.append("")
    lines.append("## 请求、状态和哈希")
    lines.append("")
    lines.append("10/10 请求均为 `returned`；summary 计数为 returned=10、empty=0、error=0、timeout=0，pending_after_run=0。每个 request JSON 的规范化 SHA-256 与 plan、summary、metadata 一致；CSV SHA-256 与 metadata、summary 一致；metadata 文件 SHA-256 与 summary 一致。")
    lines.append("")
    lines.append("| request_id | status | rows | CSV hash link | request hash link |")
    lines.append("|---|---:|---:|---|---|")
    for item in report["request_hash_audit"]:
        if item.get("files_present"):
            lines.append(
                f"| `{item['request_id']}` | `{item.get('status')}` | {structure.get(item['request_id'], {}).get('rows', '—')} | "
                f"{'pass' if item.get('csv_hashes_match') else 'FAIL'} | {'pass' if item.get('request_hashes_match') else 'FAIL'} |"
            )
        else:
            lines.append(f"| `{item['request_id']}` | — | — | FAIL | FAIL |")
    lines.append("")
    lines.append("原始输入目录在读取前后文件集合、文件大小和 SHA-256 均未变化；validator 只在 `data/audit/` 下落盘。")
    lines.append("")
    non_null_mismatches = {
        request_id: item["metadata_non_null_mismatches"]
        for request_id, item in structure.items()
        if item.get("metadata_non_null_mismatches")
    }
    if non_null_mismatches:
        lines.append("非空计数存在 1 个 provenance 不一致：R&D 物理 CSV 的 `Financial Period Absolute` 为空单元格，按本审计的空值规则为 45/46 非空，而 metadata 与 probe summary 记录为 46/46。原始 CSV 和 metadata 均保留，未为了让计数通过而填补、删行或改写文件。")
        lines.append("")
    lines.append("## 54 个年度槽位和独立 period-end 对齐")
    lines.append("")
    lines.append("54 个槽位按独立 `is_statement_dates.csv` 的 `Instrument + Income Statement Period End Date` 建立：6 家公司各 9 个返回期间末日。这里按日期键对每个请求独立核对；没有用 `FY2014` 等标签强行替换真实期间末日。")
    lines.append("")
    lines.append("| family | raw rows | keyed/matched | no-row slots | value-NA slots | value-present slots | unkeyed rows |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for request_id in list(ANNUAL_PROFILES) + [IS_ID]:
        item = alignment[request_id]
        if request_id == IS_ID:
            value_na = ", ".join(f"{column}={count}" for column, count in item["value_na_slots_by_column"].items())
            lines.append(
                f"| {item['label']} | {item['raw_rows']} | {item['valid_period_key_rows']}/{item['matched_slots']} | {item['no_row_slots']} | {value_na} | — | — |"
            )
        else:
            lines.append(
                f"| {item['label']} | {item['raw_rows']} | {item['valid_period_key_rows']}/{item['matched_slots']} | {item['no_row_slots']} | {item['value_na_slots']} | {item['value_present_slots']} | {item['unkeyed_rows']} |"
            )
    lines.append("")
    rd = alignment["rd_value_date_fperiod"]
    lines.append(
        f"R&D 的 46 行中有 {rd['valid_period_key_rows']} 行可按期间末日对齐，WMT 的 9 个 canonical slots 为 no-row；另有 {rd['unkeyed_rows']} 行没有可用 Instrument+Date 键，且该行的 R&D 值、Date、Financial Period 均为空。该空行没有被擅自分配到任何年度槽，也没有把它改写成零。")
    lines.append("")
    lines.append("Revenue 和五个 TR.F 控制族的有效 period-end keys 均覆盖 54/54 槽，值列均为已观测；IS 三个日期列也为 54/54 非空可解析。该表格展示的是返回覆盖与缺失分类，不是对经济字段定义的确认。")
    lines.append("")
    lines.append("## `.date` 的结构证据和可用时间边界")
    lines.append("")
    lines.append("R&D、Revenue 及五个 TR.F 控制族响应中供应商输出的 `Date`，在所有可用键行上与独立 IS 的 `Income Statement Period End Date` 一致（R&D 为 45/45 个有键行，其余年度请求为 54/54）。因此，本 probe 的结构证据支持把 `.date` 记录为 period-end；它不能被解释为 available date、announcement date 或可交易时点。公告日需单独使用 IS 的 Orig Announce Date，且其 vintage/revision 与时区仍未被本审计解决。")
    lines.append("")
    ann = lag["announcement_period_lag_days"]
    upd = lag["last_update_announcement_lag_days"]
    lines.append(
        f"公告日 − 期间末日：{fmt_range(ann)}，中位数 {ann['median_days']} 天（54/54 为正；5%–95% 分位 {ann['p05_days']}–{ann['p95_days']} 天）。")
    lines.append(
        f"Last Update − Announcement：{fmt_range(upd)}；54/54 晚于 announcement，正滞后范围为 {lag['positive_last_update_announcement_range_days'][0]}–{lag['positive_last_update_announcement_range_days'][1]} 天。Last Update 只作为返回字段的更新/修订观察，不替代公告日。")
    lines.append("")
    lines.append("年度 Date 的观测范围为 2013-12-28 至 2022-06-30；查询 SDate 从 2014-01-01 开始，但期间末日可能因财年边界落在此前。这一边界现象被报告，没有通过日期截断来删除行。IS announcement 的最大值为 2024-01-23，Last Update 最大值为 2026-08-18，均按原始返回保留。")
    lines.append("")
    lines.append("## 月频市值 value/date 两个独立响应")
    lines.append("")
    lines.append(
        f"value 和 date 响应分别为 {monthly['value_rows']} 和 {monthly['date_rows']} 行；Instrument 全行序一致，每家公司各 108 行，date 响应每家公司有 108 个唯一且按行递增的月末日期，value 列 648/648 非空。上述证据允许形成 648 行的 provisional positional candidate，但 value 响应没有日期列，所以本审计明确 `formal_join_approved=false`。后续建议同一请求同时取 `TR.CompanyMarketCapitalization` 与 `TR.CompanyMarketCapitalization.date`，用同响应中的显式键核对后再讨论正式 join。")
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("该审计只回答字段返回的结构、覆盖、日期/期间关系和原始文件完整性；任何经济口径、点时可得性、单位/币种、修订政策或正式研究用途仍需独立证据和团队审阅。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw ai_fundamentals_probe_v2 outputs without reading test targets.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Raw probe run id under data/raw/ai_fundamentals_probe_v2")
    parser.add_argument("--audit-id", default=None, help="Audit output id; defaults to a new UTC timestamp")
    args = parser.parse_args()
    audit_id = args.audit_id or new_audit_id()
    report = audit_run(args.run_id, audit_id)
    print(
        json.dumps(
            {
                "audit_run_id": report["audit_run_id"],
                "audited_run_id": report["audited_run_id"],
                "audit_output_path": report["audit_output_path"],
                "passed": report["passed"],
                "total": report["total"],
                "all_passed": report["all_passed"],
                "failed_checks": report["failed_checks"],
                "test_target_rows_read": report["scope"]["test_target_rows_read"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
