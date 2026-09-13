"""Independent read-only audit of the AI-fundamentals PIT probe v3.

The validator reads the frozen v3 raw run and the v2 current-probe run for a
like-for-like comparison. It never imports the LSEG client, issues requests,
reads returns, targets, models or portfolios, and never writes to either raw
run. All date parsing, key construction and numeric comparisons happen in
memory for diagnostics. Empty strings are missing; a literal numeric zero is
observed data. No cleaning, filling, imputation, winsorisation, deduplication,
row deletion, company exclusion, identifier mapping or unit conversion occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw" / "ai_fundamentals_pit_probe_v3"
V2_RAW_ROOT = ROOT / "data" / "raw" / "ai_fundamentals_probe_v2"
AUDIT_ROOT = ROOT / "data" / "audit" / "ai_fundamentals_pit_probe_v3"
DEFAULT_RUN_ID = "20260909T175636988821Z"
DEFAULT_V2_RUN_ID = "20260909T174626399754Z"
SCHEMA_VERSION = "ai_fundamentals_pit_probe_v3"
AUDIT_SCHEMA_VERSION = "ai_fundamentals_pit_probe_v3_validation"
UNIVERSE = ["NVDA.OQ", "MSFT.OQ", "AMD.OQ", "IBM.N", "WMT.N", "JNJ.N"]
ANNUAL_WINDOW = {
    "SDate": "2014-01-01",
    "EDate": "2022-12-31",
    "Frq": "FY",
    "ReportingState": "Orig",
}
MONTHLY_WINDOW = {"SDate": "2014-01-01", "EDate": "2022-12-31", "Frq": "M"}

REQUEST_IDS = [
    "legacy_rd_reporting_orig",
    "legacy_revenue_reporting_orig",
    "f_tot_revenue_reporting_orig",
    "f_tot_assets_reporting_orig",
    "f_com_eq_tot_reporting_orig",
    "f_gross_prof_ind_prop_tot_reporting_orig",
    "f_net_cash_flow_op_reporting_orig",
    "f_debt_tot_reporting_orig",
    "is_statement_dates_reporting_orig",
    "business_segments_fy0_reporting_orig",
    "company_market_capitalization_date_monthly",
]

ANNUAL_PROFILES: dict[str, dict[str, str]] = {
    "legacy_rd_reporting_orig": {
        "period_col": "Date",
        "value_col": "Research And Development",
        "fperiod_col": "Financial Period Absolute",
        "label": "legacy R&D",
    },
    "legacy_revenue_reporting_orig": {
        "period_col": "Date",
        "value_col": "Revenue",
        "fperiod_col": "Financial Period Absolute",
        "label": "legacy Revenue",
    },
    "f_tot_revenue_reporting_orig": {
        "period_col": "Date",
        "value_col": "Revenue from Business Activities - Total",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Total Revenue",
    },
    "f_tot_assets_reporting_orig": {
        "period_col": "Date",
        "value_col": "Total Assets",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Total Assets",
    },
    "f_com_eq_tot_reporting_orig": {
        "period_col": "Date",
        "value_col": "Common Equity - Total",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Common Equity",
    },
    "f_gross_prof_ind_prop_tot_reporting_orig": {
        "period_col": "Date",
        "value_col": "Gross Profit - Industrials/Property - Total",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Gross Profit",
    },
    "f_net_cash_flow_op_reporting_orig": {
        "period_col": "Date",
        "value_col": "Net Cash Flow from Operating Activities",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Net CFO",
    },
    "f_debt_tot_reporting_orig": {
        "period_col": "Date",
        "value_col": "Debt - Total",
        "fperiod_col": "Financial Period Absolute",
        "label": "TR.F Total Debt",
    },
}
IS_ID = "is_statement_dates_reporting_orig"
IS_COLUMNS = {
    "announcement": "Income Statement Orig Announce Date",
    "last_update": "Income Statement Last Update Date",
    "period_end": "Income Statement Period End Date",
}
SEGMENT_ID = "business_segments_fy0_reporting_orig"
MCAP_ID = "company_market_capitalization_date_monthly"
SEGMENT_COLUMNS = [
    "Instrument",
    "Segment Code",
    "Segment Name",
    "Financial Period Absolute",
    "Date",
    "Standardized Revenue - Business Segment",
]
EXPECTED_HEADERS = {
    "legacy_rd_reporting_orig": ["Instrument", "Research And Development", "Date", "Financial Period Absolute"],
    "legacy_revenue_reporting_orig": ["Instrument", "Revenue", "Date", "Financial Period Absolute"],
    "f_tot_revenue_reporting_orig": ["Instrument", "Revenue from Business Activities - Total", "Date", "Financial Period Absolute"],
    "f_tot_assets_reporting_orig": ["Instrument", "Total Assets", "Date", "Financial Period Absolute"],
    "f_com_eq_tot_reporting_orig": ["Instrument", "Common Equity - Total", "Date", "Financial Period Absolute"],
    "f_gross_prof_ind_prop_tot_reporting_orig": ["Instrument", "Gross Profit - Industrials/Property - Total", "Date", "Financial Period Absolute"],
    "f_net_cash_flow_op_reporting_orig": ["Instrument", "Net Cash Flow from Operating Activities", "Date", "Financial Period Absolute"],
    "f_debt_tot_reporting_orig": ["Instrument", "Debt - Total", "Date", "Financial Period Absolute"],
    IS_ID: ["Instrument", *IS_COLUMNS.values()],
    SEGMENT_ID: SEGMENT_COLUMNS,
    MCAP_ID: ["Instrument", "Company Market Capitalization", "Date"],
}

# These are the five TR.F families common to v2 and v3. v3's total-revenue
# family is retained in the structural audit but has no v2 counterpart.
V2_COMPARE_PROFILES = [
    ("legacy_rd_reporting_orig", "rd_value_date_fperiod", "Research And Development", "Research And Development"),
    ("legacy_revenue_reporting_orig", "revenue_value_date_fperiod", "Revenue", "Revenue"),
    ("f_tot_assets_reporting_orig", "tot_assets_date_fperiod", "Total Assets", "Total Assets"),
    ("f_com_eq_tot_reporting_orig", "com_eq_tot_date_fperiod", "Common Equity - Total", "Common Equity - Total"),
    ("f_gross_prof_ind_prop_tot_reporting_orig", "gross_prof_ind_prop_tot_date_fperiod", "Gross Profit - Industrials/Property - Total", "Gross Profit - Industrials/Property - Total"),
    ("f_net_cash_flow_op_reporting_orig", "net_cash_flow_op_date_fperiod", "Net Cash Flow from Operating Activities", "Net Cash Flow from Operating Activities"),
    ("f_debt_tot_reporting_orig", "debt_tot_date_fperiod", "Debt - Total", "Debt - Total"),
]


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
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_text(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_nonempty(value: Any) -> bool:
    return clean_cell(value) != ""


def parse_date(value: Any) -> date | None:
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


def parse_number(value: Any) -> float | None:
    text = clean_cell(value).replace(",", "")
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_csv_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def date_stats(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    nonempty = [row.get(column) for row in rows if is_nonempty(row.get(column))]
    parsed = [item for item in (parse_date(value) for value in nonempty) if item is not None]
    return {
        "non_null_parseable": len(parsed),
        "parse_failures": len(nonempty) - len(parsed),
        "min": min(parsed).isoformat() if parsed else None,
        "max": max(parsed).isoformat() if parsed else None,
        "unique_values": len(set(parsed)),
    }


def normalize_date_stats(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for key in ("min", "max"):
        if result.get(key) is not None:
            parsed = parse_date(result[key])
            result[key] = parsed.isoformat() if parsed else None
    return result


def row_key(row: dict[str, str], period_column: str) -> tuple[str, str] | None:
    instrument = clean_cell(row.get("Instrument"))
    period = parse_date(row.get(period_column))
    if not instrument or period is None:
        return None
    return instrument, period.isoformat()


def key_counts(
    rows: Iterable[dict[str, str]],
    columns: list[str],
    date_columns: set[str] | None = None,
) -> tuple[Counter[tuple[str, ...]], int]:
    date_columns = date_columns or set()
    counts: Counter[tuple[str, ...]] = Counter()
    missing = 0
    for row in rows:
        values: list[str] = []
        row_missing = False
        for column in columns:
            text = clean_cell(row.get(column))
            if column in date_columns:
                parsed = parse_date(text)
                if parsed is None:
                    row_missing = True
                    values.append("")
                else:
                    values.append(parsed.isoformat())
            else:
                if not text:
                    row_missing = True
                values.append(text)
        if row_missing:
            missing += 1
        else:
            counts[tuple(values)] += 1
    return counts, missing


def duplicate_summary(rows: list[dict[str, str]], candidate_columns: list[str]) -> dict[str, Any]:
    headers = list(rows[0].keys()) if rows else []
    # Full-row duplicate diagnostics must retain rows containing blank cells.
    # key_counts intentionally treats blanks as missing for candidate-key
    # assessment, so it cannot be reused for the full-row counter.
    full_counts: Counter[tuple[str, ...]] = Counter(
        tuple(clean_cell(row.get(column)) for column in headers) for row in rows
    )
    date_columns = {column for column in candidate_columns if "date" in column.lower()}
    candidate_counts, candidate_missing = key_counts(rows, candidate_columns, date_columns)
    full_groups = {key: count for key, count in full_counts.items() if count > 1}
    candidate_groups = {key: count for key, count in candidate_counts.items() if count > 1}
    return {
        "full_row_duplicate_rows": int(sum(full_groups.values())),
        "full_row_duplicate_groups": int(len(full_groups)),
        "candidate_key_columns": list(candidate_columns),
        "candidate_key_status": "assessed_mechanically",
        "candidate_key_duplicate_rows": int(sum(candidate_groups.values())),
        "candidate_key_duplicate_groups": int(len(candidate_groups)),
        "candidate_key_missing_rows": int(candidate_missing),
    }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def numeric_distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": float(statistics.mean(values)) if values else None,
        "median": float(statistics.median(values)) if values else None,
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
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


def table_value_map(rows: list[dict[str, str]], period_column: str) -> tuple[dict[tuple[str, str], dict[str, str]], list[int], dict[tuple[str, str], int]]:
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    unkeyed: list[int] = []
    counts: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(rows):
        key = row_key(row, period_column)
        if key is None:
            unkeyed.append(index)
            continue
        counts[key] += 1
        if key not in mapping:
            mapping[key] = row
    return mapping, unkeyed, counts


def compare_numeric_family(
    v3_rows: list[dict[str, str]],
    v2_rows: list[dict[str, str]],
    v3_value_col: str,
    v2_value_col: str,
    label: str,
) -> dict[str, Any]:
    v3_map, v3_unkeyed, v3_counts = table_value_map(v3_rows, "Date")
    v2_map, v2_unkeyed, v2_counts = table_value_map(v2_rows, "Date")
    v3_keys = set(v3_map)
    v2_keys = set(v2_map)
    matched_keys = sorted(v3_keys & v2_keys)
    only_v3 = sorted(v3_keys - v2_keys)
    only_v2 = sorted(v2_keys - v3_keys)
    equal_count = 0
    difference_rows: list[dict[str, Any]] = []
    missing_value_rows: list[dict[str, Any]] = []
    absolute_differences: list[float] = []
    relative_differences: list[float] = []
    period_label_differences = 0
    for key in matched_keys:
        v3_row = v3_map[key]
        v2_row = v2_map[key]
        v3_number = parse_number(v3_row.get(v3_value_col))
        v2_number = parse_number(v2_row.get(v2_value_col))
        if v3_number is None or v2_number is None:
            missing_value_rows.append(
                {
                    "instrument": key[0],
                    "date": key[1],
                    "v3_value": v3_row.get(v3_value_col),
                    "v2_value": v2_row.get(v2_value_col),
                }
            )
        elif v3_number == v2_number:
            equal_count += 1
        else:
            signed_difference = v3_number - v2_number
            absolute_difference = abs(signed_difference)
            relative_difference = absolute_difference / abs(v2_number) if v2_number != 0 else None
            absolute_differences.append(absolute_difference)
            if relative_difference is not None:
                relative_differences.append(relative_difference)
            difference_rows.append(
                {
                    "instrument": key[0],
                    "date": key[1],
                    "v3_period": v3_row.get("Financial Period Absolute"),
                    "v2_period": v2_row.get("Financial Period Absolute"),
                    "v3_value": v3_number,
                    "v2_value": v2_number,
                    "signed_v3_minus_v2": signed_difference,
                    "absolute_difference": absolute_difference,
                    "absolute_relative_to_v2": relative_difference,
                }
            )
        if clean_cell(v3_row.get("Financial Period Absolute")) != clean_cell(v2_row.get("Financial Period Absolute")):
            period_label_differences += 1
    return {
        "label": label,
        "v3_rows": len(v3_rows),
        "v2_rows": len(v2_rows),
        "v3_keyed_rows": len(v3_map),
        "v2_keyed_rows": len(v2_map),
        "v3_unkeyed_rows": len(v3_unkeyed),
        "v2_unkeyed_rows": len(v2_unkeyed),
        "matched_keys": len(matched_keys),
        "v3_only_keys": [list(key) for key in only_v3],
        "v2_only_keys": [list(key) for key in only_v2],
        "equal_value_count": equal_count,
        "different_value_count": len(difference_rows),
        "missing_value_count": len(missing_value_rows),
        "period_label_difference_count": period_label_differences,
        "absolute_difference_distribution": numeric_distribution(absolute_differences),
        "relative_difference_distribution": numeric_distribution(relative_differences),
        "difference_examples": difference_rows[:20],
        "missing_value_examples": missing_value_rows[:20],
        "v3_value_column": v3_value_col,
        "v2_value_column": v2_value_col,
        "key_definition": "Instrument + parsed Date; Financial Period Absolute is compared as a label after key matching.",
    }


def compare_is_dates(v3_rows: list[dict[str, str]], v2_rows: list[dict[str, str]]) -> dict[str, Any]:
    def keyed(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], dict[str, str]], list[int]]:
        mapping: dict[tuple[str, str], dict[str, str]] = {}
        unkeyed: list[int] = []
        for index, row in enumerate(rows):
            key = row_key(row, IS_COLUMNS["period_end"])
            if key is None:
                unkeyed.append(index)
            elif key not in mapping:
                mapping[key] = row
        return mapping, unkeyed

    v3_map, v3_unkeyed = keyed(v3_rows)
    v2_map, v2_unkeyed = keyed(v2_rows)
    matched = sorted(set(v3_map) & set(v2_map))
    columns: dict[str, Any] = {}
    total_differences = 0
    for column in IS_COLUMNS.values():
        equal = 0
        differences: list[dict[str, Any]] = []
        missing = 0
        for key in matched:
            v3_value = parse_date(v3_map[key].get(column))
            v2_value = parse_date(v2_map[key].get(column))
            if v3_value is None or v2_value is None:
                missing += 1
            elif v3_value == v2_value:
                equal += 1
            else:
                differences.append(
                    {
                        "instrument": key[0],
                        "period_end": key[1],
                        "v3_value": v3_map[key].get(column),
                        "v2_value": v2_map[key].get(column),
                        "signed_days_v3_minus_v2": (v3_value - v2_value).days,
                    }
                )
        total_differences += len(differences)
        columns[column] = {
            "matched_keys": len(matched),
            "equal_count": equal,
            "different_count": len(differences),
            "missing_count": missing,
            "difference_examples": differences[:20],
        }
    return {
        "v3_rows": len(v3_rows),
        "v2_rows": len(v2_rows),
        "v3_keyed_rows": len(v3_map),
        "v2_keyed_rows": len(v2_map),
        "v3_unkeyed_rows": len(v3_unkeyed),
        "v2_unkeyed_rows": len(v2_unkeyed),
        "matched_keys": len(matched),
        "v3_only_keys": [list(key) for key in sorted(set(v3_map) - set(v2_map))],
        "v2_only_keys": [list(key) for key in sorted(set(v2_map) - set(v3_map))],
        "columns": columns,
        "total_date_differences": total_differences,
        "key_definition": "Instrument + parsed Income Statement Period End Date.",
    }


def build_v2_comparison(tables: dict[str, dict[str, Any]], v2_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "v2_run_path": str(v2_dir.relative_to(ROOT)),
        "v2_annual_parameters": None,
        "v3_annual_parameters": dict(ANNUAL_WINDOW),
        "families": {},
        "is_dates": None,
        "status": "completed",
    }
    v2_plan_path = v2_dir / "plan.json"
    if not v2_plan_path.is_file():
        result["status"] = "missing_v2_plan"
        return result
    v2_plan = read_json(v2_plan_path)
    result["v2_annual_parameters"] = v2_plan.get("annual_window")
    for v3_id, v2_id, v3_col, v2_col in V2_COMPARE_PROFILES:
        v3_rows = tables.get(v3_id, {}).get("rows", [])
        v2_csv = v2_dir / f"{v2_id}.csv"
        if not v2_csv.is_file():
            result["families"][v3_id] = {"label": ANNUAL_PROFILES[v3_id]["label"], "v2_counterpart": v2_id, "status": "missing_v2_csv"}
            result["status"] = "missing_v2_input"
            continue
        _, v2_rows = read_csv_table(v2_csv)
        result["families"][v3_id] = compare_numeric_family(v3_rows, v2_rows, v3_col, v2_col, ANNUAL_PROFILES[v3_id]["label"])
        result["families"][v3_id]["v2_counterpart"] = v2_id
    v3_is_rows = tables.get(IS_ID, {}).get("rows", [])
    v2_is_csv = v2_dir / "is_statement_dates.csv"
    if v2_is_csv.is_file():
        _, v2_is_rows = read_csv_table(v2_is_csv)
        result["is_dates"] = compare_is_dates(v3_is_rows, v2_is_rows)
    else:
        result["is_dates"] = {"status": "missing_v2_csv"}
        result["status"] = "missing_v2_input"

    f_revenue_rows = tables.get("f_tot_revenue_reporting_orig", {}).get("rows", [])
    legacy_revenue_rows = tables.get("legacy_revenue_reporting_orig", {}).get("rows", [])
    if f_revenue_rows and legacy_revenue_rows:
        result["v3_internal_total_revenue_vs_legacy"] = compare_numeric_family(
            f_revenue_rows,
            legacy_revenue_rows,
            "Revenue from Business Activities - Total",
            "Revenue",
            "v3 TR.F Total Revenue vs v3 legacy Revenue",
        )
    else:
        result["v3_internal_total_revenue_vs_legacy"] = {"status": "missing_v3_input"}
    comparable_families = list(result["families"].values())
    value_difference_count = sum(item.get("different_value_count", 0) for item in comparable_families)
    value_pair_count = sum(item.get("equal_value_count", 0) + item.get("different_value_count", 0) for item in comparable_families)
    date_difference_count = result["is_dates"].get("total_date_differences", 0) if isinstance(result["is_dates"], dict) else 0
    result["reporting_state_observation"] = {
        "matched_nonmissing_value_pairs": value_pair_count,
        "different_value_pairs": value_difference_count,
        "different_is_date_cells": date_difference_count,
        "difference_observed": bool(value_difference_count or date_difference_count),
        "interpretation": (
            "Differences are observable in these matched sample cells under the v3 Orig request versus the v2 current probe; "
            "the comparison does not identify the vendor rule or establish a complete vintage definition."
            if value_difference_count or date_difference_count
            else "No difference was observed in matched non-missing sample cells; this does not establish equality outside the sample."
        ),
    }
    return result


def segment_analysis(
    rows: list[dict[str, str]],
    canonical_slots: set[tuple[str, str]],
    revenue_map: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    group_rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_rows[(clean_cell(row.get("Instrument")), clean_cell(row.get("Financial Period Absolute")), clean_cell(row.get("Date")))].append(row)

    null_total_rows = [row for row in rows if not is_nonempty(row.get("Segment Code")) and not is_nonempty(row.get("Segment Name"))]
    named_total_rows = [
        row
        for row in rows
        if re.search(r"\btotal\b", clean_cell(row.get("Segment Name")), flags=re.IGNORECASE)
        or clean_cell(row.get("Segment Code")).lower() in {"total", "tot"}
    ]
    named_uncoded_rows = [row for row in rows if not is_nonempty(row.get("Segment Code")) and is_nonempty(row.get("Segment Name"))]
    coded_named_rows = [row for row in rows if is_nonempty(row.get("Segment Code")) and is_nonempty(row.get("Segment Name"))]

    reconciliation_rows: list[dict[str, Any]] = []
    duplicate_total_groups: list[dict[str, Any]] = []
    total_candidate_groups = 0
    selected_total_groups = 0
    for key, group in sorted(group_rows.items(), key=lambda item: (item[0][0], parse_date(item[0][2]) or date.min, item[0][1])):
        instrument, period_label, raw_date = key
        period_date = parse_date(raw_date)
        canonical_key = (instrument, period_date.isoformat()) if period_date else None
        null_candidates = [row for row in group if not is_nonempty(row.get("Segment Code")) and not is_nonempty(row.get("Segment Name"))]
        if null_candidates:
            total_candidate_groups += 1
        if len(null_candidates) > 1:
            duplicate_total_groups.append(
                {
                    "instrument": instrument,
                    "period": period_label,
                    "date": raw_date,
                    "rows": len(null_candidates),
                    "values": [row.get("Standardized Revenue - Business Segment") for row in null_candidates],
                }
            )
        selected = null_candidates[0] if null_candidates else None
        if selected is not None:
            selected_total_groups += 1
        detail_rows = [row for row in group if row not in null_candidates]
        detail_values = [parse_number(row.get("Standardized Revenue - Business Segment")) for row in detail_rows]
        detail_values_present = [value for value in detail_values if value is not None]
        detail_sum = sum(detail_values_present) if detail_values_present else None
        total_value = parse_number(selected.get("Standardized Revenue - Business Segment")) if selected else None
        f_revenue_row = revenue_map.get(canonical_key) if canonical_key else None
        f_revenue_value = parse_number(f_revenue_row.get("Revenue from Business Activities - Total")) if f_revenue_row else None
        reconciliation_rows.append(
            {
                "instrument": instrument,
                "period": period_label,
                "date": raw_date,
                "canonical_key": list(canonical_key) if canonical_key else None,
                "rows": len(group),
                "null_total_candidate_rows": len(null_candidates),
                "named_uncoded_rows": sum(1 for row in group if not is_nonempty(row.get("Segment Code")) and is_nonempty(row.get("Segment Name"))),
                "selected_null_total": total_value,
                "detail_row_count": len(detail_rows),
                "detail_missing_value_rows": sum(value is None for value in detail_values),
                "detail_sum": detail_sum,
                "detail_sum_minus_selected_total": detail_sum - total_value if detail_sum is not None and total_value is not None else None,
                "selected_total_minus_f_revenue": total_value - f_revenue_value if total_value is not None and f_revenue_value is not None else None,
                "f_revenue_value": f_revenue_value,
            }
        )

    sum_differences = [row["detail_sum_minus_selected_total"] for row in reconciliation_rows if row["detail_sum_minus_selected_total"] is not None]
    total_vs_revenue_differences = [row["selected_total_minus_f_revenue"] for row in reconciliation_rows if row["selected_total_minus_f_revenue"] is not None]
    exact_zero_sum = sum(abs(value) < 1e-9 for value in sum_differences)
    exact_zero_total_revenue = sum(abs(value) < 1e-9 for value in total_vs_revenue_differences)

    name_history: dict[str, Any] = {}
    for instrument in UNIVERSE:
        instrument_groups = [
            (key, value)
            for key, value in group_rows.items()
            if key[0] == instrument
        ]
        instrument_groups.sort(key=lambda item: parse_date(item[0][2]) or date.min)
        periods: list[dict[str, Any]] = []
        changes: list[dict[str, Any]] = []
        previous_names: set[str] | None = None
        previous_period: str | None = None
        for (inst, period_label, raw_date), group in instrument_groups:
            names = sorted({clean_cell(row.get("Segment Name")) for row in group if is_nonempty(row.get("Segment Name"))})
            current_names = set(names)
            periods.append({"period": period_label, "date": raw_date, "named_segment_count": len(names), "names": names})
            if previous_names is not None:
                added = sorted(current_names - previous_names)
                removed = sorted(previous_names - current_names)
                if added or removed:
                    changes.append({"from_period": previous_period, "to_period": period_label, "added": added, "removed": removed})
            previous_names = current_names
            previous_period = period_label
        name_history[instrument] = {
            "period_count": len(periods),
            "unique_nonempty_names": sorted({name for item in periods for name in item["names"]}),
            "periods": periods,
            "change_count": len(changes),
            "changes": changes,
        }

    observed_slots = {
        (clean_cell(row.get("Instrument")), parse_date(row.get("Date")).isoformat())
        for row in rows
        if clean_cell(row.get("Instrument")) and parse_date(row.get("Date")) is not None
    }
    per_instrument_periods = {
        instrument: sorted(
            {
                (clean_cell(row.get("Financial Period Absolute")), parse_date(row.get("Date")).isoformat())
                for row in rows
                if clean_cell(row.get("Instrument")) == instrument and parse_date(row.get("Date")) is not None
            }
        )
        for instrument in UNIVERSE
    }
    return {
        "rows": len(rows),
        "group_count": len(group_rows),
        "observed_instruments": sorted({clean_cell(row.get("Instrument")) for row in rows}),
        "instrument_period_coverage": {
            instrument: {
                "period_count": len(values),
                "periods": [list(value) for value in values],
            }
            for instrument, values in per_instrument_periods.items()
        },
        "canonical_slot_count": len(canonical_slots),
        "observed_instrument_date_slots": len(observed_slots),
        "slots_missing_from_segments": [list(key) for key in sorted(canonical_slots - observed_slots)],
        "extra_segment_slots": [list(key) for key in sorted(observed_slots - canonical_slots)],
        "null_total_candidate_rows": len(null_total_rows),
        "null_total_candidate_groups": total_candidate_groups,
        "selected_null_total_groups": selected_total_groups,
        "named_total_rows_by_total_token": len(named_total_rows),
        "named_uncoded_segment_rows": len(named_uncoded_rows),
        "coded_named_segment_rows": len(coded_named_rows),
        "duplicate_total_groups": duplicate_total_groups,
        "duplicate_total_group_count": len(duplicate_total_groups),
        "duplicate_total_row_count_beyond_one": sum(item["rows"] - 1 for item in duplicate_total_groups),
        "reconciliation": {
            "definition": "For each group, select the first blank-code/blank-name row as a structural null-total candidate; sum all other rows without cleaning or classification.",
            "group_count": len(reconciliation_rows),
            "groups_with_numeric_total": sum(row["selected_null_total"] is not None for row in reconciliation_rows),
            "groups_with_numeric_detail_sum": sum(row["detail_sum"] is not None for row in reconciliation_rows),
            "sum_minus_selected_total_distribution": numeric_distribution(sum_differences),
            "sum_minus_selected_total_zero_count": exact_zero_sum,
            "sum_minus_selected_total_nonzero_count": len(sum_differences) - exact_zero_sum,
            "selected_total_minus_f_revenue_distribution": numeric_distribution(total_vs_revenue_differences),
            "selected_total_minus_f_revenue_zero_count": exact_zero_total_revenue,
            "selected_total_minus_f_revenue_nonzero_count": len(total_vs_revenue_differences) - exact_zero_total_revenue,
            "rows": reconciliation_rows,
        },
        "segment_name_history": name_history,
        "keyword_classification_frozen": False,
        "keyword_classification_statement": "This run demonstrates retrievable segment structure and raw name history only; no AI keyword taxonomy or classification rule is frozen.",
    }


def mcap_analysis(rows: list[dict[str, str]]) -> dict[str, Any]:
    expected_months = [f"{year:04d}-{month:02d}" for year in range(2014, 2023) for month in range(1, 13)]
    key_counts_value: Counter[tuple[str, str]] = Counter()
    per_instrument: dict[str, Any] = {}
    for row in rows:
        instrument = clean_cell(row.get("Instrument"))
        parsed = parse_date(row.get("Date"))
        if instrument and parsed:
            key_counts_value[(instrument, parsed.isoformat())] += 1
    duplicate_keys = {key: count for key, count in key_counts_value.items() if count > 1}
    for instrument in UNIVERSE:
        dates = [parse_date(row.get("Date")) for row in rows if clean_cell(row.get("Instrument")) == instrument]
        parsed_dates = [value for value in dates if value is not None]
        observed_months = sorted({value.strftime("%Y-%m") for value in parsed_dates})
        per_instrument[instrument] = {
            "rows": sum(clean_cell(row.get("Instrument")) == instrument for row in rows),
            "parseable_dates": len(parsed_dates),
            "unique_dates": len(set(parsed_dates)),
            "first_date": min(parsed_dates).isoformat() if parsed_dates else None,
            "last_date": max(parsed_dates).isoformat() if parsed_dates else None,
            "missing_months": sorted(set(expected_months) - set(observed_months)),
            "extra_months": sorted(set(observed_months) - set(expected_months)),
            "months_match_expected": observed_months == expected_months,
        }
    return {
        "rows": len(rows),
        "key_definition": "Instrument + parsed Date",
        "parseable_key_rows": sum(key_counts_value.values()),
        "unique_key_count": len(key_counts_value),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": {"|".join(key): count for key, count in duplicate_keys.items()},
        "expected_month_count_per_instrument": len(expected_months),
        "expected_month_first": expected_months[0],
        "expected_month_last": expected_months[-1],
        "per_instrument": per_instrument,
    }


def audit_run(run_id: str, audit_id: str, v2_run_id: str) -> dict[str, Any]:
    raw_dir = RAW_ROOT / run_id
    v2_dir = V2_RAW_ROOT / v2_run_id
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw v3 run does not exist: {raw_dir}")
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
    probe_script_path = ROOT / str(plan.get("script_path", "probe_ai_fundamentals_pit_v3.py"))
    probe_script_hash = sha256_file(probe_script_path) if probe_script_path.is_file() else None

    record_check(checks, "raw_run_directory_exists", "inventory", raw_dir.is_dir(), str(raw_dir), str(raw_dir))
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
        "Plan is checked mechanically; no plan rule is changed.",
    )
    record_check(checks, "probe_script_hash_matches_plan", "hashes", probe_script_hash == plan.get("script_sha256"), plan.get("script_sha256"), probe_script_hash, "Probe script is provenance only; it is not executed by this validator.")

    plan_specs = plan.get("requests", [])
    plan_by_id = {str(spec.get("request_id")): spec for spec in plan_specs if isinstance(spec, dict)}
    summary_records = summary.get("records", [])
    summary_by_id = {str(record.get("request_id")): record for record in summary_records if isinstance(record, dict)}
    record_check(checks, "eleven_planned_requests_present", "inventory", len(plan_specs) == 11 and [spec.get("request_id") for spec in plan_specs] == REQUEST_IDS, REQUEST_IDS, [spec.get("request_id") for spec in plan_specs], "All v3 request families remain independent.")
    record_check(checks, "eleven_summary_records_present", "inventory", len(summary_records) == 11 and set(summary_by_id) == set(REQUEST_IDS), {"count": 11, "ids": REQUEST_IDS}, {"count": len(summary_records), "ids": sorted(summary_by_id)}, "Summary inventory is checked without rewriting it.")
    statuses = {request_id: summary_by_id.get(request_id, {}).get("status") for request_id in REQUEST_IDS}
    record_check(checks, "eleven_request_statuses_returned", "status", all(status == "returned" for status in statuses.values()), {request_id: "returned" for request_id in REQUEST_IDS}, statuses, "Returned status is separate from coverage and semantic interpretation.")
    summary_counts = summary.get("counts", {})
    last_execute = summary.get("last_execute", {})
    record_check(
        checks,
        "summary_counts_match_eleven_completed_requests",
        "status",
        summary_counts.get("returned") == 11
        and summary_counts.get("planned") == 0
        and summary_counts.get("running") == 0
        and summary_counts.get("empty") == 0
        and summary_counts.get("error") == 0
        and summary_counts.get("timeout") == 0
        and last_execute.get("attempted") == 11
        and last_execute.get("pending_after_run") == 0,
        {"returned": 11, "planned": 0, "running": 0, "empty": 0, "error": 0, "timeout": 0, "attempted": 11, "pending_after_run": 0},
        {**summary_counts, "attempted": last_execute.get("attempted"), "pending_after_run": last_execute.get("pending_after_run")},
    )

    tables: dict[str, dict[str, Any]] = {}
    hash_records: list[dict[str, Any]] = []
    all_files_present = True
    request_hashes_ok = True
    metadata_identity_ok = True
    csv_hashes_ok = True
    policy_ok = True
    for request_id in REQUEST_IDS:
        spec = plan_by_id.get(request_id, {})
        summary_record = summary_by_id.get(request_id, {})
        csv_path = raw_dir / f"{request_id}.csv"
        request_path = raw_dir / f"{request_id}.request.json"
        metadata_path = raw_dir / f"{request_id}.metadata.json"
        present = csv_path.is_file() and request_path.is_file() and metadata_path.is_file()
        all_files_present = all_files_present and present
        if not present:
            hash_records.append({"request_id": request_id, "files_present": False})
            request_hashes_ok = False
            metadata_identity_ok = False
            csv_hashes_ok = False
            policy_ok = False
            continue
        request_object = read_json(request_path)
        metadata = read_json(metadata_path)
        headers, rows = read_csv_table(csv_path)
        tables[request_id] = {"headers": headers, "rows": rows, "metadata": metadata, "csv_path": csv_path}
        actual_request_hash = sha256_file(request_path)
        canonical_request_hash = sha256_bytes(canonical_json(request_object))
        actual_metadata_hash = sha256_file(metadata_path)
        actual_csv_hash = sha256_file(csv_path)
        expected_request_hash = spec.get("request_sha256")
        request_ok = (
            request_object == spec.get("request") == metadata.get("request")
            and actual_request_hash == canonical_request_hash == expected_request_hash
            and metadata.get("request_sha256") == actual_request_hash
            and metadata.get("request_file_sha256") == actual_request_hash
            and summary_record.get("request_file_sha256") == actual_request_hash
        )
        metadata_ok = (
            metadata.get("schema_version") == SCHEMA_VERSION
            and metadata.get("request_id") == request_id
            and metadata.get("family") == spec.get("family")
            and metadata.get("frequency") == spec.get("frequency")
            and metadata.get("status") == summary_record.get("status")
            and metadata.get("csv_path") == csv_path.name
            and metadata.get("request") == request_object
        )
        csv_ok = actual_csv_hash == metadata.get("csv_sha256") == summary_record.get("csv_sha256") and actual_metadata_hash == summary_record.get("metadata_sha256")
        policy_match = metadata.get("raw_response_policy") == plan.get("raw_response_policy")
        request_hashes_ok = request_hashes_ok and request_ok
        metadata_identity_ok = metadata_identity_ok and metadata_ok
        csv_hashes_ok = csv_hashes_ok and csv_ok
        policy_ok = policy_ok and policy_match
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
                "request_hashes_match": request_ok,
                "metadata_identity_match": metadata_ok,
                "raw_policy_match": policy_match,
                "csv_hashes_match": csv_ok,
            }
        )
    record_check(checks, "all_request_metadata_csv_files_present", "hashes", all_files_present, "each request has request JSON, metadata JSON and CSV", {"all_present": all_files_present, "missing": [item["request_id"] for item in hash_records if not item.get("files_present")]})
    record_check(checks, "request_payload_hashes_match_plan_summary_metadata", "hashes", request_hashes_ok, "all request payload and SHA-256 links agree", {"all_match": request_hashes_ok}, "Request objects are read and canonically hashed; they are never rewritten.")
    record_check(checks, "metadata_identity_and_policy_match", "hashes", metadata_identity_ok and policy_ok, "metadata identity and raw-response policy agree", {"identity_match": metadata_identity_ok, "raw_policy_match": policy_ok}, "Missing values and raw-response handling remain as collected.")
    record_check(checks, "csv_and_metadata_hashes_match_summary", "hashes", csv_hashes_ok, "all CSV and metadata SHA-256 links agree", {"all_match": csv_hashes_ok}, "CSV bytes are only hashed, never rewritten.")

    structure: dict[str, Any] = {}
    structure_ok = True
    date_range_ok = True
    duplicate_ok = True
    expected_instrument_ok = True
    for request_id in REQUEST_IDS:
        table = tables.get(request_id)
        if not table:
            structure_ok = date_range_ok = duplicate_ok = expected_instrument_ok = False
            continue
        headers = table["headers"]
        rows = table["rows"]
        metadata = table["metadata"]
        non_null = {header: sum(is_nonempty(row.get(header)) for row in rows) for header in headers}
        metadata_dates = metadata.get("date_ranges", {})
        actual_dates = {column: date_stats(rows, column) for column in metadata_dates}
        normalized_metadata_dates = {column: normalize_date_stats(value) for column, value in metadata_dates.items()}
        candidate_columns = list(metadata.get("duplicate_key_diagnostics", {}).get("candidate_key_columns", []))
        actual_duplicates = duplicate_summary(rows, candidate_columns)
        metadata_rows_columns_match = (
            headers == metadata.get("columns") == EXPECTED_HEADERS[request_id]
            and len(rows) == metadata.get("rows")
        )
        metadata_non_null = metadata.get("non_null", {})
        metadata_non_null_counts_valid = (
            isinstance(metadata_non_null, dict)
            and set(metadata_non_null) == set(headers)
            and all(isinstance(value, int) and 0 <= value <= len(rows) for value in metadata_non_null.values())
        )
        # The probe's metadata uses the source dataframe's notna semantics.
        # CSV round-tripping cannot recover whether an empty cell originated as
        # a literal empty string or a missing value, so raw-memory non-null is
        # reported separately rather than compared as if it were identical.
        table_ok = metadata_rows_columns_match and metadata_non_null_counts_valid
        dates_ok = actual_dates == normalized_metadata_dates
        dups_ok = actual_duplicates == metadata.get("duplicate_key_diagnostics", {})
        observed_instruments = sorted({clean_cell(row.get("Instrument")) for row in rows})
        unknown = sorted(set(observed_instruments) - set(UNIVERSE))
        instruments_ok = not unknown
        structure_ok = structure_ok and table_ok
        date_range_ok = date_range_ok and dates_ok
        duplicate_ok = duplicate_ok and dups_ok
        expected_instrument_ok = expected_instrument_ok and instruments_ok
        zero_counts = {header: sum(is_nonempty(row.get(header)) and parse_number(row.get(header)) == 0 for row in rows) for header in headers}
        structure[request_id] = {
            "rows": len(rows),
            "columns": headers,
            "expected_columns": EXPECTED_HEADERS[request_id],
            "non_null_recomputed": non_null,
            "metadata_rows_columns_match": metadata_rows_columns_match,
            "metadata_non_null_counts_valid": metadata_non_null_counts_valid,
            "metadata_non_null_reported": metadata_non_null,
            "raw_memory_non_null_recomputed": non_null,
            "raw_memory_missing_recomputed": {header: len(rows) - count for header, count in non_null.items()},
            "observed_instruments": observed_instruments,
            "unknown_instruments": unknown,
            "date_ranges_recomputed": actual_dates,
            "date_ranges_match_metadata": dates_ok,
            "duplicate_diagnostics_recomputed": actual_duplicates,
            "duplicate_diagnostics_match_metadata": dups_ok,
            "literal_zero_observed_by_column": zero_counts,
        }
    record_check(checks, "raw_table_rows_columns_and_non_null_recomputed", "structure", structure_ok, "each CSV matches metadata rows/columns and reports both non-null conventions", {"all_match": structure_ok, "tables": structure}, "Metadata non-null follows probe dataframe semantics; raw-memory blank handling and date parsed-valid counts are reported separately.")
    record_check(checks, "date_ranges_and_parse_counts_recomputed", "dates", date_range_ok, "metadata date diagnostics recompute from raw CSV", {"all_match": date_range_ok}, "Date parsing is diagnostic only and never written back.")
    record_check(checks, "duplicate_diagnostics_recomputed", "duplicates", duplicate_ok, "metadata duplicate diagnostics recompute from raw CSV", {"all_match": duplicate_ok}, "Duplicates are reported and retained, including duplicated segment totals.")
    record_check(checks, "observed_instruments_are_within_universe", "structure", expected_instrument_ok, UNIVERSE, {request_id: item.get("unknown_instruments", []) for request_id, item in structure.items()}, "No identifier mapping or company exclusion is applied.")

    is_rows = tables.get(IS_ID, {}).get("rows", [])
    canonical_slots_list = [row_key(row, IS_COLUMNS["period_end"]) for row in is_rows]
    canonical_slots = {key for key in canonical_slots_list if key is not None}
    canonical_counts = Counter(key[0] for key in canonical_slots_list if key is not None)
    canonical_grid_ok = len(canonical_slots_list) == 54 and len(canonical_slots) == 54 and canonical_counts == Counter({instrument: 9 for instrument in UNIVERSE})
    record_check(checks, "annual_canonical_slot_grid_is_six_by_nine", "alignment", canonical_grid_ok, {"slots": 54, "instruments": UNIVERSE, "slots_per_instrument": 9}, {"valid_slots": len(canonical_slots_list), "unique_slots": len(canonical_slots), "counts": dict(canonical_counts)}, "IS period-end dates anchor the mechanical grid; fiscal labels are not coerced.")

    annual_alignment: dict[str, Any] = {}
    annual_alignment_ok = canonical_grid_ok
    for request_id, profile in ANNUAL_PROFILES.items():
        rows = tables.get(request_id, {}).get("rows", [])
        keys = [row_key(row, profile["period_col"]) for row in rows]
        valid_keys = [key for key in keys if key is not None]
        key_set = set(valid_keys)
        counts = Counter(valid_keys)
        duplicate_groups = {key: count for key, count in counts.items() if count > 1}
        matched = sorted(canonical_slots & key_set)
        missing = sorted(canonical_slots - key_set)
        extra = sorted(key_set - canonical_slots)
        value_na = []
        value_present = []
        for key in matched:
            matched_rows = [row for row in rows if row_key(row, profile["period_col"]) == key]
            if len(matched_rows) == 1 and is_nonempty(matched_rows[0].get(profile["value_col"])):
                value_present.append(key)
            elif len(matched_rows) == 1:
                value_na.append(key)
        unkeyed_indices = [index for index, key in enumerate(keys) if key is None]
        alignment_item = {
            "label": profile["label"],
            "raw_rows": len(rows),
            "valid_period_key_rows": len(valid_keys),
            "matched_slots": len(matched),
            "no_row_slots": len(missing),
            "value_na_slots": len(value_na),
            "value_present_slots": len(value_present),
            "unkeyed_rows": len(unkeyed_indices),
            "extra_period_key_rows": len(extra),
            "duplicate_key_groups": len(duplicate_groups),
            "missing_slot_keys": [list(key) for key in missing],
            "extra_slot_keys": [list(key) for key in extra],
            "unkeyed_row_indices": unkeyed_indices,
            "period_column": profile["period_col"],
            "value_column": profile["value_col"],
        }
        annual_alignment[request_id] = alignment_item
        annual_alignment_ok = annual_alignment_ok and not extra and not duplicate_groups and len(matched) + len(missing) == 54
    is_na_by_column = {column: sum(not is_nonempty(row.get(column)) for row in is_rows) for column in IS_COLUMNS.values()}
    annual_alignment[IS_ID] = {
        "label": "IS dates",
        "raw_rows": len(is_rows),
        "valid_period_key_rows": len(canonical_slots_list),
        "matched_slots": len(canonical_slots),
        "no_row_slots": 54 - len(canonical_slots),
        "value_na_slots_by_column": is_na_by_column,
        "duplicate_key_groups": int(len(canonical_slots_list) != len(canonical_slots)),
        "period_column": IS_COLUMNS["period_end"],
        "value_columns": list(IS_COLUMNS.values()),
    }
    record_check(checks, "annual_period_end_keys_align_to_is_grid", "alignment", annual_alignment_ok, "each annual family uses only IS Instrument + period-end keys", annual_alignment, "Rows without a usable key remain unkeyed and are not assigned to a slot.")

    period_end_evidence: dict[str, Any] = {}
    period_end_ok = True
    for request_id, profile in ANNUAL_PROFILES.items():
        rows = tables.get(request_id, {}).get("rows", [])
        mismatch_rows = [index for index, row in enumerate(rows) if row_key(row, profile["period_col"]) is not None and row_key(row, profile["period_col"]) not in canonical_slots]
        period_end_evidence[request_id] = {"valid_key_rows": annual_alignment[request_id]["valid_period_key_rows"], "matching_is_period_end_key_rows": annual_alignment[request_id]["matched_slots"], "mismatch_rows": mismatch_rows}
        period_end_ok = period_end_ok and not mismatch_rows
    record_check(checks, "annual_date_columns_match_is_period_end_keys", "date_semantics", period_end_ok, "all valid annual Date rows are in the IS period-end key set", period_end_evidence, "This supports a period-end reading for this sample; it is not available/announcement time evidence.")

    lag_rows: list[dict[str, Any]] = []
    for index, row in enumerate(is_rows):
        announcement = parse_date(row.get(IS_COLUMNS["announcement"]))
        last_update = parse_date(row.get(IS_COLUMNS["last_update"]))
        period_end = parse_date(row.get(IS_COLUMNS["period_end"]))
        if announcement and last_update and period_end:
            lag_rows.append({"row_index": index, "instrument": clean_cell(row.get("Instrument")), "period_end": period_end.isoformat(), "announcement": announcement.isoformat(), "last_update": last_update.isoformat(), "announcement_period_lag_days": (announcement - period_end).days, "last_update_announcement_lag_days": (last_update - announcement).days})
    announcement_lags = [row["announcement_period_lag_days"] for row in lag_rows]
    update_lags = [row["last_update_announcement_lag_days"] for row in lag_rows]
    lag_summary = {
        "row_count_with_all_three_dates": len(lag_rows),
        "announcement_period_lag_days": numeric_distribution([float(value) for value in announcement_lags]),
        "last_update_announcement_lag_days": numeric_distribution([float(value) for value in update_lags]),
        "announcement_period_negative_rows": sum(value < 0 for value in announcement_lags),
        "announcement_period_zero_rows": sum(value == 0 for value in announcement_lags),
        "last_update_after_announcement_rows": sum(value > 0 for value in update_lags),
        "last_update_equal_announcement_rows": sum(value == 0 for value in update_lags),
        "last_update_before_announcement_rows": sum(value < 0 for value in update_lags),
        "rows": lag_rows,
    }
    record_check(checks, "is_statement_dates_three_columns_parseable", "dates", len(lag_rows) == 54, 54, {"rows_with_all_three_dates": len(lag_rows)}, "Date arithmetic is in memory only; no timezone or raw-date rewrite is applied.")

    segment_rows = tables.get(SEGMENT_ID, {}).get("rows", [])
    revenue_map, _, _ = table_value_map(tables.get("f_tot_revenue_reporting_orig", {}).get("rows", []), "Date")
    segments = segment_analysis(segment_rows, canonical_slots, revenue_map)
    segment_coverage_ok = (
        segments["rows"] == 269
        and segments["group_count"] == 54
        and segments["observed_instruments"] == sorted(UNIVERSE)
        and not segments["slots_missing_from_segments"]
        and not segments["extra_segment_slots"]
        and all(item["period_count"] == 9 for item in segments["instrument_period_coverage"].values())
    )
    record_check(checks, "business_segments_269_rows_six_by_nine_coverage", "business_segments", segment_coverage_ok, {"rows": 269, "groups": 54, "instruments": UNIVERSE, "periods_per_instrument": 9}, {"rows": segments["rows"], "groups": segments["group_count"], "coverage": segments["instrument_period_coverage"], "missing_slots": segments["slots_missing_from_segments"], "extra_slots": segments["extra_segment_slots"]}, "Coverage is structural; no segment classification is applied.")
    record_check(checks, "business_segments_named_and_null_total_rows_classified", "business_segments", segments["null_total_candidate_groups"] == 54 and segments["selected_null_total_groups"] == 54, {"null_total_candidate_groups": 54, "selected_null_total_groups": 54}, {"null_total_candidate_rows": segments["null_total_candidate_rows"], "null_total_candidate_groups": segments["null_total_candidate_groups"], "selected_null_total_groups": segments["selected_null_total_groups"], "named_total_rows_by_total_token": segments["named_total_rows_by_total_token"], "named_uncoded_segment_rows": segments["named_uncoded_segment_rows"]}, "Blank-code/blank-name rows are structural null-total candidates; named rows with blank codes remain detail candidates.")
    record_check(checks, "business_segments_duplicate_totals_are_retained_and_reported", "business_segments", segments["duplicate_total_group_count"] == 4 and segments["duplicate_total_row_count_beyond_one"] == 4, {"duplicate_total_groups": 4, "duplicate_rows_beyond_one": 4}, {"duplicate_total_group_count": segments["duplicate_total_group_count"], "duplicate_rows_beyond_one": segments["duplicate_total_row_count_beyond_one"], "groups": segments["duplicate_total_groups"]}, "The duplicate total observation is retained as evidence; no deduplication is performed.")
    recon = segments["reconciliation"]
    record_check(checks, "business_segments_sum_total_reconciliation_recomputed", "business_segments", recon["group_count"] == 54 and recon["groups_with_numeric_total"] == 54 and recon["groups_with_numeric_detail_sum"] == 54, {"groups": 54, "numeric_totals": 54, "numeric_detail_sums": 54}, {"groups": recon["group_count"], "numeric_totals": recon["groups_with_numeric_total"], "numeric_detail_sums": recon["groups_with_numeric_detail_sum"], "sum_minus_total": recon["sum_minus_selected_total_distribution"], "total_minus_f_revenue": recon["selected_total_minus_f_revenue_distribution"]}, "Reconciliation is a diagnostic distribution, not a license to drop or adjust rows.")
    record_check(checks, "business_segments_name_history_recomputed", "business_segments", sum(item["period_count"] for item in segments["segment_name_history"].values()) == 54, {"instrument_periods": 54}, {"instrument_periods": sum(item["period_count"] for item in segments["segment_name_history"].values()), "change_counts": {key: item["change_count"] for key, item in segments["segment_name_history"].items()}, "keyword_classification_frozen": segments["keyword_classification_frozen"]}, "Name-set transitions are reported literally; AI keyword classification remains unfrozen.")

    mcap = mcap_analysis(tables.get(MCAP_ID, {}).get("rows", []))
    mcap_ok = mcap["rows"] == 648 and mcap["unique_key_count"] == 648 and mcap["duplicate_key_count"] == 0 and all(item["months_match_expected"] and item["rows"] == 108 and item["unique_dates"] == 108 for item in mcap["per_instrument"].values())
    record_check(checks, "combined_mcap_instrument_date_keys_unique", "market_cap", mcap["unique_key_count"] == 648 and mcap["duplicate_key_count"] == 0, {"rows": 648, "unique_keys": 648, "duplicate_keys": 0}, {"rows": mcap["rows"], "unique_keys": mcap["unique_key_count"], "duplicate_keys": mcap["duplicate_key_count"]}, "The combined response has an explicit Instrument + Date key.")
    record_check(checks, "combined_mcap_monthly_date_coverage", "market_cap", mcap_ok, {"rows_per_instrument": 108, "months": "2014-01 through 2022-12"}, mcap["per_instrument"], "Month coverage is checked by year-month; vendor month-end day strings are retained.")

    v2_comparison = build_v2_comparison(tables, v2_dir)
    comparison_ok = v2_comparison["status"] == "completed" and len(v2_comparison["families"]) == len(V2_COMPARE_PROFILES) and all("different_value_count" in item for item in v2_comparison["families"].values()) and isinstance(v2_comparison.get("is_dates"), dict) and "total_date_differences" in v2_comparison["is_dates"]
    record_check(checks, "v2_current_probe_comparison_recomputed", "v2_comparison", comparison_ok, {"families": len(V2_COMPARE_PROFILES), "is_date_comparison": True}, {"status": v2_comparison.get("status"), "families": len(v2_comparison.get("families", {})), "reporting_state_observation": v2_comparison.get("reporting_state_observation")}, "Comparison uses Instrument + parsed Date/period-end keys and reports differences without assigning a semantic cause.")

    raw_files_after = sorted(path for path in raw_dir.iterdir() if path.is_file())
    raw_hashes_after = {path.name: sha256_file(path) for path in raw_files_after}
    raw_sizes_after = {path.name: path.stat().st_size for path in raw_files_after}
    raw_unchanged = [path.name for path in raw_files_before] == [path.name for path in raw_files_after] and raw_hashes_before == raw_hashes_after and raw_sizes_before == raw_sizes_after
    record_check(checks, "raw_input_files_unchanged_during_validation", "provenance", raw_unchanged, "raw file set, sizes and SHA-256 unchanged", {"file_set_unchanged": [path.name for path in raw_files_before] == [path.name for path in raw_files_after], "hashes_unchanged": raw_hashes_before == raw_hashes_after, "sizes_unchanged": raw_sizes_before == raw_sizes_after}, "Validator only writes under data/audit/.")

    passed = sum(bool(item["passed"]) for item in checks)
    total = len(checks)
    failed = [item["check_id"] for item in checks if not item["passed"]]
    report: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_run_id": audit_id,
        "audited_run_id": run_id,
        "audited_raw_path": str(raw_dir.relative_to(ROOT)),
        "v2_compared_run_id": v2_run_id,
        "v2_compared_raw_path": str(v2_dir.relative_to(ROOT)),
        "audit_output_path": str(audit_dir.relative_to(ROOT)),
        "validator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "executed_at_utc": utc_now(),
        "scope": {
            "purpose": "Independent structural, provenance, segment and v2 comparison audit of the v3 AI-fundamentals PIT probe.",
            "test_target_rows_read": False,
            "test_target_rows_parsed": False,
            "returns_constructed": False,
            "factors_constructed": False,
            "labels_constructed": False,
            "models_or_portfolios_run": False,
            "cleaning_or_imputation": False,
            "row_deletion_or_company_exclusion": False,
            "quarantine_path": None,
        },
        "input_contract": {
            "schema_version": SCHEMA_VERSION,
            "universe": UNIVERSE,
            "annual_window": ANNUAL_WINDOW,
            "monthly_window": MONTHLY_WINDOW,
            "expected_request_count": 11,
            "expected_annual_slots": 54,
            "missing_definition": "None or blank after strip is missing; literal numeric 0 is observed.",
            "raw_response_policy": plan.get("raw_response_policy"),
        },
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "failed_checks": failed,
        "checks": checks,
        "request_hash_audit": hash_records,
        "structure": structure,
        "annual_alignment": annual_alignment,
        "period_end_evidence": period_end_evidence,
        "announcement_lag": lag_summary,
        "business_segments": segments,
        "combined_market_cap": mcap,
        "v2_comparison": v2_comparison,
        "actual_hashes": {
            "plan.json": plan_hash,
            "probe_summary.json": summary_hash,
            "probe_script": probe_script_hash,
            "raw_files_before": raw_hashes_before,
            "raw_files_after": raw_hashes_after,
        },
        "raw_file_sizes_before": raw_sizes_before,
        "raw_file_sizes_after": raw_sizes_after,
        "limitations": [
            "ReportingState=Orig differences are observable in matched sample cells versus v2's current-probe request, but this audit cannot identify the vendor rule or establish a complete original-vintage definition.",
            "Annual Date values match IS period-end keys in this sample; that does not establish announcement time, available time, vendor vintage, timezone, accounting basis, units or currency.",
            "Blank-code/blank-name segment rows are treated as structural null-total candidates for a transparent diagnostic. Named blank-code rows are retained as detail candidates; no economic segment taxonomy is asserted.",
            "Business-segment sums and total reconciliation differences are reported as distributions. Duplicate total rows are retained; no deduplication or reconciliation adjustment is performed.",
            "The business-segment run proves current structural retrieval and name history only. AI keyword classification has not been frozen.",
            "The combined market-cap response has explicit Instrument + Date keys and 108 monthly observations per instrument for 2014-01 through 2022-12; this is coverage evidence, not a point-in-time or currency/unit conclusion.",
            "No return, test-target, label, factor, model, portfolio or performance artifact was read or generated.",
        ],
        "status": "completed" if passed == total else "failed_checks",
    }

    (audit_dir / "validation.json").write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (audit_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check_id", "category", "passed", "expected", "observed", "notes"])
        writer.writeheader()
        writer.writerows(checks)
    (audit_dir / "findings.md").write_text(render_findings(report), encoding="utf-8")
    return report


def fmt_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_findings(report: dict[str, Any]) -> str:
    comparison = report["v2_comparison"]
    segments = report["business_segments"]
    recon = segments["reconciliation"]
    mcap = report["combined_market_cap"]
    rd_structure = report["structure"]["legacy_rd_reporting_orig"]
    segment_structure = report["structure"][SEGMENT_ID]
    name_change_text = ", ".join(
        f"{key}={value['change_count']}" for key, value in segments["segment_name_history"].items()
    )
    lines = [
        "# AI fundamentals PIT probe v3 独立验证",
        "",
        f"- 审计输出：`{report['audit_output_path']}/`；输入运行：`{report['audited_raw_path']}/`。",
        f"- 执行时间（UTC）：`{report['executed_at_utc']}`。",
        f"- 机械检查通过：**{report['passed']}/{report['total']}**；状态：`{report['status']}`。",
        "",
        "## 范围与处理边界",
        "",
        "本轮只读取 v3 的 plan、request、metadata、probe summary 和 11 张原始 CSV，并读取 v2 current probe 的对应原始 CSV 做 Instrument + period/date 比较。没有读取或生成测试收益、标签、factor、模型、组合或性能结果。",
        "",
        "空字符串或缺失单元格计为缺失；文本形式的 `0`、`0.0` 等仍计为已观测值。日期只在内存中解析为诊断键，没有写回原始 CSV。没有清洗、填补、插值、前值填充、winsorize、去重、删行、公司排除、identifier mapping、单位/币种变换或时区写回；没有需要 quarantine 的行，quarantine 为 `None`。",
        "",
        f"为避免把 probe metadata 的 dataframe `notna` 口径误解为 CSV 损坏，报告同时保留 raw-memory 非空和 parsed-valid：R&D 的 Financial Period metadata non-null={rd_structure['metadata_non_null_reported'].get('Financial Period Absolute')}/{rd_structure['rows']}、raw-memory non-null={rd_structure['raw_memory_non_null_recomputed'].get('Financial Period Absolute')}/{rd_structure['rows']}；business-segment 的 Segment Code metadata non-null={segment_structure['metadata_non_null_reported'].get('Segment Code')}/{segment_structure['rows']}、raw-memory non-null={segment_structure['raw_memory_non_null_recomputed'].get('Segment Code')}/{segment_structure['rows']}，Segment Name 对应为 {segment_structure['metadata_non_null_reported'].get('Segment Name')}/{segment_structure['rows']} 与 {segment_structure['raw_memory_non_null_recomputed'].get('Segment Name')}/{segment_structure['rows']}。Date 的 parsed-valid 计数另列在 JSON 的 `date_ranges_recomputed`。",
        "",
        "## 请求、哈希和原始行完整性",
        "",
        f"v3 为 11/11 请求 `returned`，summary 为 returned={report['input_contract']['expected_request_count']}、empty=0、error=0、timeout=0、pending_after_run=0。每个 request JSON、metadata JSON、CSV 的 SHA-256 链接都按 plan、summary 和 metadata 重新计算。",
        "",
        "| request_id | status | rows | request hash | CSV/metadata hash |",
        "|---|---:|---:|---|---|",
    ]
    for item in report["request_hash_audit"]:
        lines.append(f"| `{item['request_id']}` | `{item.get('status', '—')}` | {report['structure'].get(item['request_id'], {}).get('rows', '—')} | {'pass' if item.get('request_hashes_match') else 'FAIL'} | {'pass' if item.get('csv_hashes_match') else 'FAIL'} |")
    lines.extend([
        "",
        "验证前后 v3 原始运行的文件集合、大小和 SHA-256 均未变化；validator 只在 `data/audit/ai_fundamentals_pit_probe_v3/` 下写入输出。",
        "",
        "## 年度结构与 v2 current probe 比较",
        "",
        "独立 `is_statement_dates_reporting_orig.csv` 给出 6 家公司各 9 个期间末日，共 54 个 Instrument + period-end 槽位。R&D 的 46 行中有 45 个可用键，另 1 行无键；其他年度数值族按期间末日核对到这组 canonical keys。",
        "",
        "| family | v3 rows | v3 keyed/matched | no-row | value NA | value present |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for request_id, item in report["annual_alignment"].items():
        if request_id == IS_ID:
            lines.append(f"| {item['label']} | {item['raw_rows']} | {item['valid_period_key_rows']}/{item['matched_slots']} | {item['no_row_slots']} | {sum(item['value_na_slots_by_column'].values())} | — |")
        else:
            lines.append(f"| {item['label']} | {item['raw_rows']} | {item['valid_period_key_rows']}/{item['matched_slots']} | {item['no_row_slots']} | {item['value_na_slots']} | {item['value_present_slots']} |")
    lines.extend([
        "",
        "下表按 Instrument + parsed Date 对 v3 `ReportingState=Orig` 请求与 v2 current probe 做匹配。值差异仅使用两边都可解析且非空的配对；`abs diff` 是 v3−v2 的绝对值，relative 是相对 v2 的绝对值。",
        "",
        "| family | matched keys | equal | different | max abs diff | max relative diff |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for request_id, item in comparison["families"].items():
        lines.append(f"| {item['label']} | {item.get('matched_keys', '—')} | {item.get('equal_value_count', '—')} | {item.get('different_value_count', '—')} | {fmt_number(item.get('absolute_difference_distribution', {}).get('max'))} | {fmt_number(item.get('relative_difference_distribution', {}).get('max'))} |")
    is_dates = comparison.get("is_dates", {})
    lines.extend([
        "",
        f"IS dates 比较匹配 {is_dates.get('matched_keys', '—')} 个 period-end keys。Orig Announce Date 差异 {is_dates.get('columns', {}).get(IS_COLUMNS['announcement'], {}).get('different_count', '—')} 个，Last Update Date 差异 {is_dates.get('columns', {}).get(IS_COLUMNS['last_update'], {}).get('different_count', '—')} 个，Period End Date 差异 {is_dates.get('columns', {}).get(IS_COLUMNS['period_end'], {}).get('different_count', '—')} 个。",
        "",
        f"因此，在这个有限样本里 ReportingState=Orig 的结果与 v2 current probe 存在可观察差异：matched non-missing value pairs 中 {comparison.get('reporting_state_observation', {}).get('different_value_pairs', '—')} 个值配对不同，IS date cells 中 {comparison.get('reporting_state_observation', {}).get('different_is_date_cells', '—')} 个不同。这个结果只说明样本中差异可见，不能单独说明差异由哪条 vendor 规则造成，也不能证明完整的 original-vintage 语义。",
        "",
        "## Business segments：269 行",
        "",
        f"分部响应有 {segments['rows']} 行、{segments['group_count']} 个 Instrument + period/date 组，6 家公司均为 9 个期间。按 `Segment Code` 与 `Segment Name` 同时为空定义结构性 null-total candidate：{segments['null_total_candidate_rows']} 行覆盖 {segments['null_total_candidate_groups']} 组；含 `total` token 的 named-total 行为 {segments['named_total_rows_by_total_token']}，另有 {segments['named_uncoded_segment_rows']} 个 blank-code 但有名字的行，均保留为 named segment candidate，不被擅自当成 total。",
        "",
        f"发现 {segments['duplicate_total_group_count']} 个重复 total 组，额外重复 {segments['duplicate_total_row_count_beyond_one']} 行（见 JSON 逐组记录）；原始行没有去重。以每组第一个 null-total candidate 为结构诊断 total，named/detail 行求和减 total 的分布为 count={recon['sum_minus_selected_total_distribution']['count']}、zero={recon['sum_minus_selected_total_zero_count']}、nonzero={recon['sum_minus_selected_total_nonzero_count']}、min={fmt_number(recon['sum_minus_selected_total_distribution']['min'])}、max={fmt_number(recon['sum_minus_selected_total_distribution']['max'])}。结构 total 减 `TR.F.TotRevenue` 的 zero={recon['selected_total_minus_f_revenue_zero_count']}、nonzero={recon['selected_total_minus_f_revenue_nonzero_count']}。这些是观察分布，不是清洗指令。",
        "",
        f"逐 instrument 的非空 Segment Name 历史变化事件数：{name_change_text}。当前结果只证明结构和名称历史可取；AI keyword 分类规则尚未冻结。",
        "",
        "## Combined market cap",
        "",
        f"combined response 共 {mcap['rows']} 行，Instrument + Date 可解析键 {mcap['unique_key_count']} 个且重复键 {mcap['duplicate_key_count']} 个。每家公司 {mcap['expected_month_count_per_instrument']} 行、{mcap['expected_month_count_per_instrument']} 个唯一日期，覆盖 2014-01 至 2022-12 的月份标签。",
        "",
        "## 限制",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently validate the raw AI-fundamentals PIT probe v3.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="v3 raw run id under data/raw/ai_fundamentals_pit_probe_v3")
    parser.add_argument("--v2-run-id", default=DEFAULT_V2_RUN_ID, help="v2 raw run id under data/raw/ai_fundamentals_probe_v2")
    parser.add_argument("--audit-id", default=None, help="Audit output id; defaults to a new UTC timestamp")
    args = parser.parse_args()
    audit_id = args.audit_id or new_audit_id()
    report = audit_run(args.run_id, audit_id, args.v2_run_id)
    print(json.dumps({"audit_run_id": report["audit_run_id"], "audited_run_id": report["audited_run_id"], "audit_output_path": report["audit_output_path"], "passed": report["passed"], "total": report["total"], "all_passed": report["all_passed"], "failed_checks": report["failed_checks"], "test_target_rows_read": report["scope"]["test_target_rows_read"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
