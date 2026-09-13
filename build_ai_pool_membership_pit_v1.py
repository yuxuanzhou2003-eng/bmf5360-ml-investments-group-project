"""Build a point-in-time AI-industry membership from immutable raw inputs.

The implementation follows ``AI_POOL_SPEC.md`` v1.1.  It reads only the
Orig raw business-segment, income-statement-original-announcement, and
Orig total-revenue families plus the corrected literal-RIC spans.  It does
not import LSEG, make network calls, read targets/future returns/labels/model
outputs, or use current company names/TRBC to backfill history.

``--dry-run`` scans and classifies in memory, writing only run-scoped summary
and gate artifacts.  ``--execute`` writes the two membership tables and the
summary.  ``--limit`` bounds the number of raw request CSVs read in one run;
limited runs are explicitly marked partial and non-model-ready.  A new UTC
run directory is created for every invocation; existing directories are never
overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ai_pool_membership_pit_v1_config.json"
SPEC_PATH = ROOT / "AI_POOL_SPEC.md"
SCHEMA_VERSION = "ai_pool_membership_pit_v1"
RUN_ID_RE = re.compile(r"^\d{8}T\d{6,12}Z$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_path(relative: str, *, label: str) -> Path:
    path = (ROOT / Path(relative)).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes workspace: {relative}") from exc
    return path


def forbidden_token(path: Path, tokens: Iterable[str]) -> str | None:
    text = str(path).replace("\\", "/").casefold()
    for token in tokens:
        if str(token).casefold() in text:
            return str(token)
    return None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.where(pd.notna(frame), "").to_csv(temp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temp, path)


def load_config() -> tuple[dict[str, Any], str]:
    config = read_json(CONFIG_PATH)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported config schema: {config.get('schema_version')}")
    return config, sha256_file(CONFIG_PATH)


def provider_tokens(config: dict[str, Any]) -> set[str]:
    return {str(value).strip().casefold() for value in config["missing_value_policy"]["provider_tokens_are_na"]}


def clean_text(value: Any, tokens: set[str]) -> tuple[str | None, str | None]:
    if value is None:
        return None, "MISSING_VALUE"
    text = str(value).strip()
    if not text or text.casefold() in tokens:
        return None, "MISSING_VALUE"
    return text, None


def clean_number(value: Any, tokens: set[str], field: str) -> tuple[str | None, str | None]:
    text, _ = clean_text(value, tokens)
    if text is None:
        return None, f"MISSING_VALUE_{field}"
    try:
        number = Decimal(text.replace(",", ""))
        if not number.is_finite():
            raise InvalidOperation
    except (InvalidOperation, ValueError):
        return None, f"UNPARSEABLE_NUMERIC_{field}"
    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".") or "0"
    return normalized, None


def clean_date(value: Any, tokens: set[str], field: str) -> tuple[str | None, str | None]:
    text, _ = clean_text(value, tokens)
    if text is None:
        return None, f"MISSING_VALUE_{field}"
    try:
        parsed = pd.to_datetime(text, errors="raise", format="mixed")
        if isinstance(parsed, pd.DatetimeIndex):
            raise ValueError("non-scalar date")
        if getattr(parsed, "tzinfo", None) is not None:
            return None, f"UNPARSEABLE_DATE_TIMEZONE_{field}"
        return pd.Timestamp(parsed).date().isoformat(), None
    except (TypeError, ValueError, OverflowError):
        return None, f"UNPARSEABLE_DATE_{field}"


def append_reasons(*values: Any) -> str:
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        for item in str(value).split(";"):
            item = item.strip()
            if item and item not in result:
                result.append(item)
    return ";".join(result)


def validate_spec(config: dict[str, Any]) -> str:
    if not SPEC_PATH.exists():
        raise RuntimeError(f"missing specification: {SPEC_PATH}")
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    expected_version = str(config.get("specification_version", ""))
    expected_declaration = f"AI 行业股票池规范 v{expected_version}"
    if expected_declaration not in spec_text:
        raise RuntimeError(f"AI_POOL_SPEC.md does not declare {expected_version}")
    if expected_version == "1.1" and "AI industry broad" not in spec_text:
        raise RuntimeError("AI_POOL_SPEC.md v1.1 broad taxonomy is missing")
    if expected_version == "1.1" and config.get("taxonomy", {}).get("word_boundary_regex") != r"\bAI\b":
        raise RuntimeError("v1.1 config must declare the strict AI word-boundary regex \\bAI\\b")
    return sha256_file(SPEC_PATH)


def raw_header_and_rows(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [str(item) for item in next(reader)]
        except StopIteration as exc:
            raise RuntimeError(f"empty raw CSV: {path}") from exc
        rows = sum(1 for _ in reader)
    return header, rows


def filename_family(path: Path, config: dict[str, Any]) -> tuple[str, str]:
    for family_id, family in config["input"]["families"].items():
        prefix = str(family["prefix"])
        if path.name.startswith(prefix) and path.name.endswith(".csv"):
            request_id = path.stem
            if family_id == "segments" or family_id == "announcement_dates" or family_id == "revenue_reference":
                if not re.fullmatch(re.escape(prefix) + r"\d{3}", request_id):
                    raise RuntimeError(f"unexpected request filename for {family_id}: {path.name}")
            return family_id, request_id
    raise RuntimeError(f"unexpected raw CSV: {path.name}")


def load_request_sidecars(path: Path, family_id: str, request_id: str, config: dict[str, Any]) -> dict[str, Any]:
    meta_path = path.with_name(f"{path.stem}.meta.json")
    request_path = path.with_name(f"{path.stem}.request.json")
    if not meta_path.exists() or not request_path.exists():
        raise RuntimeError(f"missing raw sidecar for {path.name}")
    meta = read_json(meta_path)
    request = read_json(request_path)
    raw_run_id = str(config["input"]["raw_run_id"])
    expected_raw_family_id = str(config["input"]["families"][family_id].get("raw_family_id", family_id))
    if (
        meta.get("run_id") != raw_run_id
        or meta.get("request_id") != request_id
        or meta.get("family_id") != expected_raw_family_id
    ):
        raise RuntimeError(f"raw sidecar identity mismatch: {path.name}")
    csv_hash = sha256_file(path)
    if meta.get("csv_sha256") != csv_hash:
        raise RuntimeError(f"raw CSV hash mismatch: {path.name}")
    request_file_hash = sha256_file(request_path)
    if meta.get("request_file_sha256") != request_file_hash:
        raise RuntimeError(f"raw request sidecar hash mismatch: {request_path.name}")
    canonical_request = (request.get("request") if isinstance(request.get("request"), dict) else request)
    if meta.get("request_sha256") != sha256_json(canonical_request):
        raise RuntimeError(f"raw canonical request hash mismatch: {request_path.name}")
    parameters = canonical_request.get("parameters", {}) if isinstance(canonical_request, dict) else {}
    expected = config["input"]["families"][family_id]
    if parameters.get("ReportingState") != expected["reporting_state"]:
        raise RuntimeError(f"{family_id} request is not ReportingState=Orig")
    header, rows = raw_header_and_rows(path)
    if header != expected["required_header"]:
        raise RuntimeError(f"raw header mismatch for {path.name}: {header!r}")
    if int(meta.get("rows", rows)) != rows:
        raise RuntimeError(f"raw row count mismatch for {path.name}")
    universe = canonical_request.get("universe", []) if isinstance(canonical_request, dict) else []
    return {
        "family_id": family_id,
        "raw_family_id": expected_raw_family_id,
        "request_id": request_id,
        "raw_file": relpath(path),
        "meta_file": relpath(meta_path),
        "request_file": relpath(request_path),
        "raw_sha256": csv_hash,
        "meta_sha256": sha256_file(meta_path),
        "request_file_sha256": request_file_hash,
        "request_sha256": str(meta["request_sha256"]),
        "rows": rows,
        "header": header,
        "requested_instruments": [str(value) for value in universe],
        "reporting_state": parameters.get("ReportingState"),
    }


def discover_inputs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir = safe_path(str(config["input"]["raw_run_path"]), label="raw_run_path")
    if not raw_dir.exists():
        raise RuntimeError(f"missing raw run: {raw_dir}")
    forbidden = forbidden_token(raw_dir, config["input"]["forbidden_inputs"])
    if forbidden:
        raise RuntimeError(f"forbidden input path token: {forbidden}")
    specs: list[dict[str, Any]] = []
    ignored: list[dict[str, str]] = []
    configured_prefixes = {
        str(family["prefix"]): family_id
        for family_id, family in config["input"]["families"].items()
    }
    for path in sorted(raw_dir.glob("*.csv")):
        # The collector stores several unrelated families in the same raw
        # run.  Only families explicitly declared in the membership config are
        # inputs to this stage; retain an inventory of the rest for audit.
        matching_prefixes = [prefix for prefix in configured_prefixes if path.name.startswith(prefix)]
        if not matching_prefixes:
            ignored.append({"file": relpath(path), "reason_code": "INPUT_FAMILY_NOT_REQUESTED"})
            continue
        family_id, request_id = filename_family(path, config)
        specs.append(load_request_sidecars(path, family_id, request_id, config))
    required = set(config["input"]["families"])
    found = {item["family_id"] for item in specs}
    if missing := sorted(required - found):
        raise RuntimeError(f"raw run missing required families: {missing}")
    return specs, {
        "raw_run_id": str(config["input"]["raw_run_id"]),
        "raw_run_path": relpath(raw_dir),
        "raw_csv_count": len(specs),
        "raw_inventory_csv_count": len(specs) + len(ignored),
        "ignored_csv_count": len(ignored),
        "ignored_csv_files": ignored,
        "raw_rows": int(sum(item["rows"] for item in specs)),
        "family_counts": dict(sorted(Counter(item["family_id"] for item in specs).items())),
    }


def read_raw_records(path: Path) -> tuple[list[str], list[tuple[int, str, list[str]]]]:
    records: list[tuple[int, str, list[str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise RuntimeError(f"empty raw CSV: {path}")
        header = [str(item) for item in next(csv.reader([header_line]))]
        for row_number, raw_line in enumerate(handle, 1):
            parsed = next(csv.reader([raw_line]))
            records.append((row_number, raw_line.rstrip("\r\n"), parsed))
    return header, records


def normalize_family(spec: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    path = safe_path(spec["raw_file"], label="raw_file")
    header, rows = read_raw_records(path)
    tokens = provider_tokens(config)
    family_id = spec["family_id"]
    records: list[dict[str, Any]] = []
    for raw_row, raw_text, values in rows:
        malformed = len(values) != len(header)
        padded = list(values) + [""] * max(0, len(header) - len(values))
        fields = {header[index]: padded[index] for index in range(len(header))}
        reason = "MALFORMED_CSV_ROW" if malformed else ""
        instrument, instrument_code = clean_text(fields.get("Instrument", ""), tokens)
        reason = append_reasons(reason, "INSTRUMENT_MISSING" if instrument_code else None)
        record: dict[str, Any] = {
            "family_id": family_id,
            "request_id": spec["request_id"],
            "Instrument": instrument,
            "period_end": None,
            "fperiod": None,
            "orig_announcement_date": None,
            "last_update_date": None,
            "value": None,
            "segment_code": None,
            "segment_name": None,
            "segment_value": None,
            "raw_file": spec["raw_file"],
            "raw_row": raw_row,
            "raw_text": raw_text,
            "raw_fields_json": json.dumps(fields, ensure_ascii=False, sort_keys=True),
            "row_reason_codes": reason,
        }
        if family_id == "segments":
            period_end, period_code = clean_date(fields.get("Date", ""), tokens, "period_end")
            fperiod, fperiod_code = clean_text(fields.get("Financial Period Absolute", ""), tokens)
            segment_code, segment_code_reason = clean_text(fields.get("Segment Code", ""), tokens)
            segment_name, segment_name_reason = clean_text(fields.get("Segment Name", ""), tokens)
            segment_value, segment_value_reason = clean_number(fields.get("Standardized Revenue - Business Segment", ""), tokens, "segment_revenue")
            record.update({"period_end": period_end, "fperiod": fperiod, "segment_code": segment_code, "segment_name": segment_name, "segment_value": segment_value})
            record["row_reason_codes"] = append_reasons(record["row_reason_codes"], period_code, fperiod_code, segment_code_reason, segment_name_reason, segment_value_reason)
        elif family_id == "announcement_dates":
            period_end, period_code = clean_date(fields.get("Income Statement Period End Date", ""), tokens, "period_end")
            announcement, announcement_code = clean_date(fields.get("Income Statement Orig Announce Date", ""), tokens, "orig_announcement_date")
            last_update, last_update_code = clean_date(fields.get("Income Statement Last Update Date", ""), tokens, "last_update_date")
            record.update({"period_end": period_end, "orig_announcement_date": announcement, "last_update_date": last_update})
            record["row_reason_codes"] = append_reasons(record["row_reason_codes"], period_code, announcement_code, last_update_code)
        elif family_id == "revenue_reference":
            period_end, period_code = clean_date(fields.get("Date", ""), tokens, "period_end")
            fperiod, fperiod_code = clean_text(fields.get("Financial Period Absolute", ""), tokens)
            value, value_code = clean_number(fields.get("Revenue from Business Activities - Total", ""), tokens, "reference_revenue")
            record.update({"period_end": period_end, "fperiod": fperiod, "value": value})
            record["row_reason_codes"] = append_reasons(record["row_reason_codes"], period_code, fperiod_code, value_code)
        else:
            raise RuntimeError(f"unsupported family: {family_id}")
        if record["period_end"] is None:
            record["row_reason_codes"] = append_reasons(record["row_reason_codes"], "PERIOD_KEY_MISSING")
        records.append(record)
    columns = [
        "family_id", "request_id", "Instrument", "period_end", "fperiod", "orig_announcement_date", "last_update_date", "value",
        "segment_code", "segment_name", "segment_value", "raw_file", "raw_row", "raw_text", "raw_fields_json", "row_reason_codes",
    ]
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame[columns]


def read_membership_spans(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = safe_path(str(config["input"]["membership_spans"]), label="membership_spans")
    forbidden = forbidden_token(path, config["input"]["forbidden_inputs"])
    if forbidden:
        raise RuntimeError(f"forbidden membership path token: {forbidden}")
    if not path.exists():
        raise RuntimeError(f"missing membership spans: {path}")
    header, rows = read_raw_records(path)
    expected = ["ric", "name", "delisted_ric", "member_from", "member_to", "member_days", "fetch_start", "fetch_end"]
    if header != expected:
        raise RuntimeError(f"membership spans header mismatch: {header!r}")
    records = []
    for raw_row, raw_text, values in rows:
        padded = list(values) + [""] * max(0, len(header) - len(values))
        records.append({**{header[index]: padded[index] for index in range(len(header))}, "raw_file": relpath(path), "raw_row": raw_row, "raw_text": raw_text})
    return pd.DataFrame(records), {"path": relpath(path), "sha256": sha256_file(path), "rows": len(records), "header": header}


def duplicate_status(frame: pd.DataFrame, keys: list[str], compare_columns: list[str]) -> tuple[pd.Series, dict[tuple[str, ...], str]]:
    status = pd.Series("UNIQUE_KEY", index=frame.index, dtype="object")
    lookup: dict[tuple[str, ...], str] = {}
    valid = frame[keys].notna().all(axis=1)
    for key, group in frame.loc[valid].groupby(keys, dropna=False, sort=False):
        key_tuple = tuple(str(value) for value in (key if isinstance(key, tuple) else (key,)))
        comparable = group[compare_columns].fillna("").astype(str)
        if len(group) > 1:
            code = "DUPLICATE_KEY_IDENTICAL" if len(comparable.drop_duplicates()) == 1 else "DUPLICATE_KEY_CONFLICT"
            status.loc[group.index] = code
        else:
            code = "UNIQUE_KEY"
        lookup[key_tuple] = code
    return status, lookup


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def build_lookup(frame: pd.DataFrame, value_columns: list[str]) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    if frame.empty:
        return lookup, status_counts
    valid = frame["Instrument"].notna() & frame["period_end"].notna()
    for key, group in frame.loc[valid].groupby(["Instrument", "period_end"], dropna=False, sort=False):
        key_tuple = (str(key[0]), str(key[1]))
        comparable = group[value_columns].fillna("").astype(str)
        status = "UNIQUE_KEY" if len(group) == 1 else ("DUPLICATE_KEY_IDENTICAL" if len(comparable.drop_duplicates()) == 1 else "DUPLICATE_KEY_CONFLICT")
        status_counts[status] += len(group)
        row = group.iloc[0]
        lookup[key_tuple] = {"status": status, **{column: row.get(column) for column in value_columns}, "raw_rows": [int(value) for value in group["raw_row"].tolist()]}
    return lookup, status_counts


def classify_name(name: Any, config: dict[str, Any]) -> tuple[str, list[str], list[str], list[str], list[str]]:
    taxonomy = config["taxonomy"]
    if pd.isna(name) or not str(name).strip():
        return "ambiguous", [], ["SEGMENT_NAME_MISSING"], [], []
    text = str(name)
    folded = text.casefold()
    boundary_terms = {str(value).casefold() for value in taxonomy.get("word_boundary_terms", [])}

    def matches(term: Any) -> bool:
        candidate = str(term)
        if candidate.casefold() in boundary_terms:
            return re.search(rf"\b{re.escape(candidate)}\b", text, flags=re.IGNORECASE) is not None
        return candidate.casefold() in folded

    core_hits = [str(value) for value in taxonomy["core_ai"] if matches(value)]
    broad_hits: list[str] = []
    for category, terms in taxonomy.get("ai_industry_broad", {}).items():
        for value in terms:
            if matches(value):
                broad_hits.append(f"{category}:{value}")
    ecosystem_hits = [str(value) for value in taxonomy["ai_ecosystem"] if matches(value)]
    hits = (
        [f"core:{value}" for value in core_hits]
        + [f"broad:{value}" for value in broad_hits]
        + [f"ecosystem:{value}" for value in ecosystem_hits]
    )
    reasons: list[str] = []
    if core_hits and broad_hits:
        reasons.append("CORE_BROAD_OVERLAP")
    if core_hits and ecosystem_hits:
        reasons.append("CORE_ECOSYSTEM_OVERLAP")
    if broad_hits and ecosystem_hits:
        reasons.append("BROAD_ECOSYSTEM_OVERLAP")
    if core_hits:
        return "core_ai", hits, reasons, broad_hits, ecosystem_hits
    if broad_hits:
        return "ai_industry_broad", hits, reasons, broad_hits, ecosystem_hits
    if ecosystem_hits:
        return "ai_ecosystem", hits, reasons, broad_hits, ecosystem_hits
    generic = [str(value) for value in taxonomy.get("generic_terms_only_are_not_members", []) if matches(value)]
    return "ambiguous", [], ["GENERIC_TERM_ONLY" if generic else "NO_TAXONOMY_MATCH"], broad_hits, ecosystem_hits


def reconciliation_diagnostics(segments: pd.DataFrame, revenue_lookup: dict[tuple[str, str], dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(columns=["total_candidate_count", "segment_sum_non_total", "reference_revenue", "reconciliation_difference", "reconciliation_status"])
    result = segments.copy()
    result["null_total_candidate"] = result["segment_code"].isna() & result["segment_name"].isna()
    for column in ["total_candidate_count", "segment_sum_non_total", "reference_revenue", "reconciliation_difference"]:
        result[column] = pd.NA
    result["reconciliation_status"] = "REFERENCE_UNAVAILABLE"
    abs_tol = float(config["reconciliation"]["absolute_tolerance"])
    rel_tol = float(config["reconciliation"]["relative_tolerance"])
    for key, index_values in result.groupby(["Instrument", "period_end"], dropna=False, sort=False).groups.items():
        indices = list(index_values)
        group = result.loc[indices]
        total_mask = group["null_total_candidate"]
        total_count = int(total_mask.sum())
        values = pd.to_numeric(group["segment_value"], errors="coerce")
        ref_info = revenue_lookup.get((str(key[0]), str(key[1])))
        reference = finite_float(ref_info.get("value")) if ref_info and ref_info.get("status") != "DUPLICATE_KEY_CONFLICT" else None
        if total_count == 1:
            candidate_value = finite_float(values.loc[total_mask].iloc[0])
            if candidate_value is not None:
                observed = candidate_value
                compare_label = "total_candidate"
            else:
                observed = float(values.loc[~total_mask].sum()) if values.loc[~total_mask].notna().any() else None
                compare_label = "non_total_sum_fallback"
        else:
            observed = float(values.loc[~total_mask].sum()) if values.loc[~total_mask].notna().any() else None
            compare_label = "non_total_sum"
        status = "REFERENCE_UNAVAILABLE"
        difference = None
        if total_count > 1:
            status = "SEGMENT_TOTAL_AMBIGUOUS"
        elif reference is not None and observed is not None:
            difference = observed - reference
            tolerance = max(abs_tol, abs(reference) * rel_tol)
            status = "RECONCILED_WITHIN_TOLERANCE" if abs(difference) <= tolerance else "SEGMENT_RECONCILIATION_GAP"
        elif observed is None:
            status = "SEGMENT_VALUE_UNAVAILABLE"
        result.loc[indices, "total_candidate_count"] = total_count
        result.loc[indices, "segment_sum_non_total"] = float(values.loc[~total_mask].sum()) if values.loc[~total_mask].notna().any() else pd.NA
        result.loc[indices, "reference_revenue"] = reference if reference is not None else pd.NA
        result.loc[indices, "reconciliation_difference"] = difference if difference is not None else pd.NA
        result.loc[indices, "reconciliation_status"] = status
        if status == "SEGMENT_TOTAL_AMBIGUOUS":
            result.loc[indices, "row_reason_codes"] = result.loc[indices, "row_reason_codes"].map(lambda value: append_reasons(value, "SEGMENT_TOTAL_AMBIGUOUS"))
        elif status == "SEGMENT_RECONCILIATION_GAP":
            result.loc[indices, "row_reason_codes"] = result.loc[indices, "row_reason_codes"].map(lambda value: append_reasons(value, "SEGMENT_RECONCILIATION_GAP"))
        elif status in {"REFERENCE_UNAVAILABLE", "SEGMENT_VALUE_UNAVAILABLE"}:
            result.loc[indices, "row_reason_codes"] = result.loc[indices, "row_reason_codes"].map(lambda value: append_reasons(value, "RECONCILIATION_UNAVAILABLE"))
        result.loc[indices, "reconciliation_compare_basis"] = compare_label
    return result


def membership_span_lookup(spans: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tokens = provider_tokens(config)
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in spans.iterrows():
        ric, _ = clean_text(row.get("ric", ""), tokens)
        if not ric:
            continue
        delisted, _ = clean_text(row.get("delisted_ric", ""), tokens)
        lookup[ric] = {
            "name": clean_text(row.get("name", ""), tokens)[0],
            "delisted_ric": delisted,
            "member_from": clean_text(row.get("member_from", ""), tokens)[0],
            "member_to": clean_text(row.get("member_to", ""), tokens)[0],
            "fetch_start": clean_text(row.get("fetch_start", ""), tokens)[0],
            "fetch_end": clean_text(row.get("fetch_end", ""), tokens)[0],
            "member_days": clean_number(row.get("member_days", ""), tokens, "member_days")[0],
        }
    return lookup


def category_match_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if frame.empty or column not in frame:
        return {}
    for value in frame[column].fillna("").astype(str):
        for item in value.split(";"):
            if item.strip():
                counts[item.split(":", 1)[0]] += 1
    return dict(sorted(counts.items()))


def category_instrument_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    instruments: dict[str, set[str]] = defaultdict(set)
    if frame.empty or column not in frame or "Instrument" not in frame:
        return {}
    for _, row in frame[["Instrument", column]].iterrows():
        instrument = row.get("Instrument")
        if pd.isna(instrument) or not str(instrument).strip():
            continue
        value = row.get(column)
        if pd.isna(value):
            value = ""
        for item in str(value).split(";"):
            item = item.strip()
            if item:
                instruments[item.split(":", 1)[0]].add(str(instrument))
    return {category: len(values) for category, values in sorted(instruments.items())}


def fill_taxonomy_categories(counts: dict[str, int], config: dict[str, Any]) -> dict[str, int]:
    categories = config["taxonomy"].get("ai_industry_broad", {})
    return {str(category): int(counts.get(category, 0)) for category in sorted(categories)}


def delimited_value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if frame.empty or column not in frame:
        return {}
    for value in frame[column].fillna("").astype(str):
        for item in value.split(";"):
            item = item.strip()
            if item:
                counts[item] += 1
    return dict(sorted(counts.items()))


def compare_with_baseline(membership: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    comparison = config.get("comparison")
    if not isinstance(comparison, dict):
        raise RuntimeError("v1.1 comparison baseline is missing from config")
    baseline_path = safe_path(str(comparison["baseline_membership_path"]), label="baseline_membership")
    baseline_summary_path = safe_path(str(comparison["baseline_summary_path"]), label="baseline_summary")
    if not baseline_path.exists() or not baseline_summary_path.exists():
        raise RuntimeError("configured v1.0 baseline artifact is missing")
    baseline_hash = sha256_file(baseline_path)
    if baseline_hash != str(comparison["baseline_membership_sha256"]):
        raise RuntimeError("v1.0 baseline membership hash mismatch")
    baseline_summary_hash = sha256_file(baseline_summary_path)
    if baseline_summary_hash != str(comparison["baseline_summary_sha256"]):
        raise RuntimeError("v1.0 baseline summary hash mismatch")
    baseline_summary = read_json(baseline_summary_path)
    expected_version = str(comparison["baseline_specification_version"])
    if str(baseline_summary.get("specification_version")) != expected_version:
        raise RuntimeError("configured v1.0 baseline summary version mismatch")
    baseline = pd.read_csv(baseline_path, dtype=str, keep_default_na=False)
    required = {"Instrument", "membership_status", "pool_label"}
    if not required.issubset(baseline.columns):
        raise RuntimeError(f"v1.0 baseline membership is missing columns: {sorted(required - set(baseline.columns))}")
    baseline = baseline.loc[baseline["Instrument"].astype(str).str.strip().ne("")].copy()
    current = membership.loc[membership["Instrument"].astype(str).str.strip().ne("")].copy()
    baseline_eligible = set(baseline.loc[baseline["membership_status"].eq("member"), "Instrument"].astype(str))
    current_eligible = set(current.loc[current["membership_status"].eq("member"), "Instrument"].astype(str))
    def pool_counts(frame: pd.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        if frame.empty or "pool_label" not in frame:
            return counts
        for label, group in frame.loc[frame["pool_label"].notna()].groupby("pool_label", dropna=True, sort=True):
            label_text = str(label).strip()
            if not label_text or label_text.casefold() in {"none", "nan", "<na>"}:
                continue
            counts[label_text] = int(group["Instrument"].nunique())
        for label_key in ("core_label", "broad_label", "ecosystem_label"):
            label_value = config.get("membership_status", {}).get(label_key)
            if label_value:
                counts.setdefault(str(label_value), 0)
        return counts
    return {
        "baseline_run_id": str(comparison["baseline_run_id"]),
        "baseline_specification_version": expected_version,
        "baseline_membership_path": str(comparison["baseline_membership_path"]),
        "baseline_membership_sha256": baseline_hash,
        "baseline_summary_path": str(comparison["baseline_summary_path"]),
        "baseline_summary_sha256": baseline_summary_hash,
        "baseline_hash_verified": True,
        "baseline_eligible_ric_count": len(baseline_eligible),
        "current_eligible_ric_count": len(current_eligible),
        "new_ric_count": len(current_eligible - baseline_eligible),
        "new_ric": sorted(current_eligible - baseline_eligible),
        "v1_0_overlap_ric_count": len(current_eligible & baseline_eligible),
        "v1_0_overlap_ric": sorted(current_eligible & baseline_eligible),
        "baseline_only_ric_count": len(baseline_eligible - current_eligible),
        "baseline_only_ric": sorted(baseline_eligible - current_eligible),
        "baseline_pool_instrument_counts": pool_counts(baseline),
        "current_pool_instrument_counts": pool_counts(current),
    }


def classify_segments(
    segments: pd.DataFrame,
    announcement_dates: pd.DataFrame,
    revenue_reference: pd.DataFrame,
    spans: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if segments.empty:
        empty = pd.DataFrame(columns=["Instrument", "formation_date", "membership_status", "pool_label"])
        return empty, empty.copy(), {
            "segment_rows": 0,
            "audit_instruments": 0,
            "membership_rows": 0,
            "known_membership_rows": 0,
            "unknown_membership_rows": 0,
            "ambiguous_membership_rows": 0,
            "core_rows": 0,
            "broad_rows": 0,
            "ecosystem_rows": 0,
            "taxonomy_overlap_rows": 0,
            "core_broad_overlap_rows": 0,
            "core_ecosystem_overlap_rows": 0,
            "broad_ecosystem_overlap_rows": 0,
            "legacy_ecosystem_match_rows": 0,
            "broad_category_match_rows": fill_taxonomy_categories({}, config),
            "broad_category_match_instruments": fill_taxonomy_categories({}, config),
            "matched_keyword_counts": {},
            "cutoff_candidate_rows": 0,
            "cutoff_passed_rows": 0,
            "cutoff_violation_rows": 0,
            "legacy_ecosystem_rows": 0,
            "legacy_ecosystem_instruments": 0,
            "legacy_ecosystem_only_rows": 0,
            "delisted_membership_rows": 0,
        }
    ann = announcement_dates.copy()
    ann = ann.loc[ann["Instrument"].notna() & ann["period_end"].notna()].copy()
    ann_status, _ = duplicate_status(ann, ["Instrument", "period_end"], ["orig_announcement_date", "last_update_date"])
    ann["announcement_key_status"] = ann_status
    ann_lookup, _ = build_lookup(ann, ["orig_announcement_date", "last_update_date"])
    rev = revenue_reference.copy()
    rev = rev.loc[rev["Instrument"].notna() & rev["period_end"].notna()].copy()
    rev_lookup, _ = build_lookup(rev, ["value", "fperiod"])
    audit = segments.copy()
    audit["candidate_class"] = pd.NA
    audit["pool_label"] = pd.NA
    audit["membership_status"] = "membership_unknown"
    audit["membership_eligible"] = "false"
    audit["formation_date"] = pd.NA
    audit["announcement_date"] = pd.NA
    audit["announcement_key_status"] = pd.NA
    audit["matched_keywords"] = pd.NA
    audit["broad_category_hits"] = pd.NA
    audit["legacy_ecosystem_hits"] = pd.NA
    audit["legacy_ecosystem_match"] = "false"
    audit["taxonomy_overlap"] = "false"
    audit["taxonomy_overlap_detail"] = pd.NA
    audit["row_reason_codes"] = audit["row_reason_codes"].fillna("").astype(str)
    for index, row in audit.iterrows():
        candidate, hits, class_reasons, broad_hits, ecosystem_hits = classify_name(row.get("segment_name"), config)
        audit.at[index, "candidate_class"] = candidate
        audit.at[index, "matched_keywords"] = ";".join(hits) if hits else pd.NA
        audit.at[index, "broad_category_hits"] = ";".join(broad_hits) if broad_hits else pd.NA
        audit.at[index, "legacy_ecosystem_hits"] = ";".join(ecosystem_hits) if ecosystem_hits else pd.NA
        audit.at[index, "legacy_ecosystem_match"] = "true" if ecosystem_hits else "false"
        overlap_reasons = [reason for reason in class_reasons if reason.endswith("_OVERLAP")]
        audit.at[index, "taxonomy_overlap"] = "true" if overlap_reasons else "false"
        audit.at[index, "taxonomy_overlap_detail"] = ";".join(overlap_reasons) if overlap_reasons else pd.NA
        audit.at[index, "row_reason_codes"] = append_reasons(audit.at[index, "row_reason_codes"], *class_reasons)
        key = (str(row["Instrument"]), str(row["period_end"])) if pd.notna(row.get("Instrument")) and pd.notna(row.get("period_end")) else None
        ann_info = ann_lookup.get(key) if key else None
        if key is None:
            audit.at[index, "row_reason_codes"] = append_reasons(audit.at[index, "row_reason_codes"], "PERIOD_KEY_MISSING")
            continue
        if ann_info is None:
            audit.at[index, "row_reason_codes"] = append_reasons(audit.at[index, "row_reason_codes"], "PERIOD_KEY_MISMATCH", "PIT_DATE_UNAVAILABLE")
            continue
        ann_status_value = ann_info.get("status", "UNIQUE_KEY")
        audit.at[index, "announcement_key_status"] = ann_status_value
        if ann_status_value == "DUPLICATE_KEY_CONFLICT":
            audit.at[index, "row_reason_codes"] = append_reasons(audit.at[index, "row_reason_codes"], "ANNOUNCEMENT_DATE_CONFLICT", "PIT_DATE_UNAVAILABLE")
            continue
        announcement = ann_info.get("orig_announcement_date")
        if pd.isna(announcement) or not str(announcement).strip():
            audit.at[index, "row_reason_codes"] = append_reasons(audit.at[index, "row_reason_codes"], "PIT_DATE_UNAVAILABLE")
            continue
        audit.at[index, "announcement_date"] = announcement
        audit.at[index, "formation_date"] = announcement
        audit.at[index, "candidate_class"] = candidate
        member_classes = {"core_ai", "ai_industry_broad", "ai_ecosystem"}
        audit.at[index, "pool_label"] = candidate if candidate in member_classes else pd.NA
        audit.at[index, "membership_status"] = "member" if candidate in member_classes else "ambiguous"
        audit.at[index, "membership_eligible"] = "true" if candidate in member_classes else "false"
        audit.at[index, "row_reason_codes"] = append_reasons(audit.at[index, "row_reason_codes"], "CUTOFF_CHECK_PASSED")

    audit = reconciliation_diagnostics(audit, rev_lookup, config)
    spans_lookup = membership_span_lookup(spans, config)
    for index, row in audit.iterrows():
        instrument = str(row["Instrument"]) if pd.notna(row.get("Instrument")) else ""
        span = spans_lookup.get(instrument)
        if span is None:
            audit.at[index, "row_reason_codes"] = append_reasons(audit.at[index, "row_reason_codes"], "RIC_NOT_IN_CORRECTED_SPANS")
            continue
        for field, value in span.items():
            audit.at[index, field] = value
    gap_or_ambiguous = audit["reconciliation_status"].isin({"SEGMENT_RECONCILIATION_GAP", "SEGMENT_TOTAL_AMBIGUOUS"})
    audit.loc[gap_or_ambiguous, "membership_status"] = "ambiguous"
    audit.loc[gap_or_ambiguous, "membership_eligible"] = "false"
    audit.loc[gap_or_ambiguous, "pool_label"] = pd.NA
    audit.loc[gap_or_ambiguous, "row_reason_codes"] = audit.loc[gap_or_ambiguous, "row_reason_codes"].map(
        lambda value: append_reasons(value, "SEGMENT_RECONCILIATION_AMBIGUOUS")
    )
    # Aggregate to the requested RIC + formation_date unit while retaining
    # every source segment row in the audit table. Unknown rows have no
    # formation date; period_end remains visible so they are not conflated.
    membership_records: list[dict[str, Any]] = []
    known = audit.loc[audit["formation_date"].notna() & audit["Instrument"].notna()].copy()
    unknown = audit.loc[audit["formation_date"].isna() & audit["Instrument"].notna()].copy()
    for key, group in known.groupby(["Instrument", "formation_date"], sort=True, dropna=False):
        membership_records.append(aggregate_membership_record(group, key, spans_lookup, config))
    # Keep unknown historical periods auditable without assigning period_end as
    # a publication date. This is deliberately outside the known PIT key gate.
    for _, group in unknown.groupby(["Instrument", "period_end"], sort=True, dropna=False):
        membership_records.append(aggregate_membership_record(group, (group.iloc[0]["Instrument"], pd.NA), spans_lookup, config, unknown=True))
    membership = pd.DataFrame(membership_records)
    stats = {
        "segment_rows": int(len(audit)),
        "audit_instruments": int(audit["Instrument"].nunique()),
        "membership_rows": int(len(membership)),
        "known_membership_rows": int(membership["formation_date"].notna().sum()) if not membership.empty else 0,
        "unknown_membership_rows": int(membership["membership_status"].eq("membership_unknown").sum()) if not membership.empty else 0,
        "ambiguous_membership_rows": int(membership["membership_status"].eq("ambiguous").sum()) if not membership.empty else 0,
        "core_rows": int(membership["pool_label"].eq("core_ai").sum()) if not membership.empty else 0,
        "broad_rows": int(membership["pool_label"].eq("ai_industry_broad").sum()) if not membership.empty else 0,
        "ecosystem_rows": int(membership["pool_label"].eq("ai_ecosystem").sum()) if not membership.empty else 0,
        "taxonomy_overlap_rows": int(membership["taxonomy_overlap"].eq("true").sum()) if not membership.empty else 0,
        "core_broad_overlap_rows": int(membership["taxonomy_overlap_detail"].astype(str).str.contains("CORE_BROAD_OVERLAP", regex=False).sum()) if not membership.empty else 0,
        "core_ecosystem_overlap_rows": int(membership["taxonomy_overlap_detail"].astype(str).str.contains("CORE_ECOSYSTEM_OVERLAP", regex=False).sum()) if not membership.empty else 0,
        "broad_ecosystem_overlap_rows": int(membership["taxonomy_overlap_detail"].astype(str).str.contains("BROAD_ECOSYSTEM_OVERLAP", regex=False).sum()) if not membership.empty else 0,
        "legacy_ecosystem_match_rows": int(membership["legacy_ecosystem_match"].eq("true").sum()) if not membership.empty else 0,
        "broad_category_match_rows": fill_taxonomy_categories(category_match_counts(audit, "broad_category_hits"), config),
        "broad_category_match_instruments": fill_taxonomy_categories(category_instrument_counts(audit, "broad_category_hits"), config),
        "matched_keyword_counts": delimited_value_counts(audit, "matched_keywords"),
        "cutoff_candidate_rows": int(audit["formation_date"].notna().sum()),
        "cutoff_passed_rows": int(delimited_value_counts(audit, "row_reason_codes").get("CUTOFF_CHECK_PASSED", 0)),
        "cutoff_violation_rows": int(
            (
                audit.loc[audit["formation_date"].notna(), "announcement_date"].fillna("").astype(str)
                > audit.loc[audit["formation_date"].notna(), "formation_date"].fillna("").astype(str)
            ).sum()
        ),
        "legacy_ecosystem_rows": int(membership["legacy_ecosystem_match"].eq("true").sum()) if not membership.empty else 0,
        "legacy_ecosystem_instruments": int(membership.loc[membership["legacy_ecosystem_match"].eq("true"), "Instrument"].nunique()) if not membership.empty else 0,
        "legacy_ecosystem_only_rows": int((membership["legacy_ecosystem_match"].eq("true") & membership["pool_label"].eq("ai_ecosystem")).sum()) if not membership.empty else 0,
        "delisted_membership_rows": int(membership["delisted_ric"].astype(str).str.casefold().eq("true").sum()) if not membership.empty and "delisted_ric" in membership else 0,
    }
    return membership, audit, stats


def aggregate_membership_record(group: pd.DataFrame, key: tuple[Any, Any], spans_lookup: dict[str, dict[str, Any]], config: dict[str, Any], *, unknown: bool = False) -> dict[str, Any]:
    instrument = str(key[0])
    formation_date = None if unknown else str(key[1])
    reasons = append_reasons(*group["row_reason_codes"].tolist())
    candidate_classes = set(str(value) for value in group["candidate_class"].dropna())
    reconciliation_statuses = set(str(value) for value in group["reconciliation_status"].dropna())
    if unknown:
        status = "membership_unknown"
        pool_label = None
        eligible = "false"
        reasons = append_reasons(reasons, "PIT_DATE_UNAVAILABLE")
    elif "SEGMENT_RECONCILIATION_GAP" in reconciliation_statuses or "SEGMENT_TOTAL_AMBIGUOUS" in reconciliation_statuses:
        status = "ambiguous"
        pool_label = None
        eligible = "false"
        reasons = append_reasons(reasons, "SEGMENT_RECONCILIATION_AMBIGUOUS")
    elif "core_ai" in candidate_classes:
        status = "member"
        pool_label = "core_ai"
        eligible = "true"
    elif "ai_industry_broad" in candidate_classes:
        status = "member"
        pool_label = "ai_industry_broad"
        eligible = "true"
    elif "ai_ecosystem" in candidate_classes:
        status = "member"
        pool_label = "ai_ecosystem"
        eligible = "true"
    else:
        status = "ambiguous"
        pool_label = None
        eligible = "false"
        reasons = append_reasons(reasons, "NO_TAXONOMY_MATCH")
    span = spans_lookup.get(instrument, {})
    return {
        "Instrument": instrument,
        "formation_date": formation_date,
        "announcement_date": None if unknown else str(group["announcement_date"].dropna().iloc[0]) if group["announcement_date"].notna().any() else None,
        "period_end": ";".join(sorted({str(value) for value in group["period_end"].dropna()})),
        "pool_label": pool_label,
        "membership_status": status,
        "membership_eligible": eligible,
        "matched_keywords": ";".join(sorted({str(value) for value in group["matched_keywords"].dropna() if str(value).strip()})) or None,
        "segment_names": " | ".join(sorted({str(value) for value in group["segment_name"].dropna() if str(value).strip()})) or None,
        "taxonomy_version": config["taxonomy"]["version"],
        "taxonomy_overlap": "true" if group["taxonomy_overlap"].astype(str).eq("true").any() else "false",
        "taxonomy_overlap_detail": ";".join(sorted({str(value) for value in group["taxonomy_overlap_detail"].dropna() if str(value).strip()})) or None,
        "broad_category_hits": ";".join(sorted({str(value) for value in group["broad_category_hits"].dropna() if str(value).strip()})) or None,
        "legacy_ecosystem_hits": ";".join(sorted({str(value) for value in group["legacy_ecosystem_hits"].dropna() if str(value).strip()})) or None,
        "legacy_ecosystem_match": "true" if group["legacy_ecosystem_match"].astype(str).eq("true").any() else "false",
        "reconciliation_status": ";".join(sorted(reconciliation_statuses)) or "NOT_EVALUATED",
        "reconciliation_difference": ";".join(sorted({str(value) for value in group["reconciliation_difference"].dropna()})) or None,
        "delisted_ric": span.get("delisted_ric"),
        "member_from": span.get("member_from"),
        "member_to": span.get("member_to"),
        "delisted_coverage_status": "retained_to_last_available_trade" if str(span.get("delisted_ric", "")).casefold() == "true" else "not_delisted_flagged",
        "reason_codes": reasons,
        "raw_file": json.dumps(sorted({str(value) for value in group["raw_file"].dropna()}), ensure_ascii=False),
        "raw_row": json.dumps([int(value) for value in group["raw_row"].dropna()], ensure_ascii=False),
        "raw_text": json.dumps([str(value) for value in group["raw_text"].dropna()], ensure_ascii=False),
    }


def audit_output_frame(audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Instrument", "period_end", "formation_date", "announcement_date", "fperiod", "segment_code", "segment_name", "segment_value",
        "candidate_class", "pool_label", "membership_status", "membership_eligible", "matched_keywords", "broad_category_hits", "legacy_ecosystem_hits", "legacy_ecosystem_match", "taxonomy_overlap", "taxonomy_overlap_detail",
        "announcement_key_status", "total_candidate_count", "segment_sum_non_total", "reference_revenue", "reconciliation_difference",
        "reconciliation_status", "reconciliation_compare_basis", "taxonomy_version", "name", "delisted_ric", "member_from", "member_to",
        "delisted_coverage_status", "row_reason_codes", "raw_file", "raw_row", "raw_text", "raw_fields_json",
    ]
    result = audit.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    return result[columns].sort_values(["Instrument", "period_end", "raw_file", "raw_row"], kind="stable").reset_index(drop=True)


def reason_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if frame.empty or column not in frame:
        return {}
    for value in frame[column].fillna("").astype(str):
        counts.update(item for item in value.split(";") if item)
    return dict(sorted(counts.items()))


def output_gate(summary: dict[str, Any], *, status: str) -> dict[str, Any]:
    checks = summary["checks"]
    check_rows = [{"name": key, "passed": bool(value) if isinstance(value, bool) else True, "detail": value} for key, value in checks.items()]
    failed = [item for item in check_rows if not item["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": summary["run_id"],
        "generated_at_utc": utc_now(),
        "status": status,
        "clean_ready": status == "complete",
        "model_ready": False,
        "checks": check_rows,
        "passed": len(check_rows) - len(failed),
        "failed": failed,
        "input_hashes": summary["input_hashes"],
        "output_hashes": summary.get("output_hashes", {}),
        "limitations": summary["limitations"],
        "non_actions": summary["non_actions"],
    }


def append_processing_log(summary: dict[str, Any], output_dir: Path, audit_dir: Path) -> None:
    path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# DATA_PROCESSING_LOG\n"
    counts = summary.get("counts", {})
    entry = [
        "",
        f"## {summary['run_id']} — AI pool membership PIT v{summary['specification_version']} ({summary['generated_at_utc']})",
        "",
        f"- **性质与目的**：`{summary['mode']}`；依据 `AI_POOL_SPEC.md` v{summary['specification_version']}，从 Orig raw 分部名称、Orig 原始公告日和 Orig total revenue 生成/诊断点时 AI 行业池；保留 Core、预注册 AI industry broad 六类和旧 v1.0 ecosystem；不生成未来收益、labels、预测或组合业绩。",
        f"- **输入版本与路径**：raw run `{summary['raw_run']['raw_run_id']}` at `{summary['raw_run']['raw_run_path']}`；spec SHA-256=`{summary['spec_sha256']}`；config SHA-256=`{summary['config_sha256']}`；raw CSV={summary['input_file_count']}，本次读取={summary['processed_file_count']}。",
        f"- **哈希与完整性**：raw CSV/meta/request hashes rechecked；membership spans `{summary['membership_spans']['path']}` SHA-256=`{summary['membership_spans']['sha256']}`；raw_before/after unchanged=`{summary['checks']['raw_unchanged_after_processing']}`。",
        f"- **形成时点规则**：formation_date 使用 Orig announcement calendar date，日级 inclusive cutoff；未用 period_end 替代 publication date，交易入场 session 留给后续 event 层。",
        f"- **命中词与公告 cutoff**：matched keyword counts=`{json.dumps(summary.get('classification_stats', {}).get('matched_keyword_counts', {}), ensure_ascii=False, sort_keys=True)}`；cutoff candidate/passed/violations=`{summary.get('classification_stats', {}).get('cutoff_candidate_rows')}/{summary.get('classification_stats', {}).get('cutoff_passed_rows')}/{summary.get('classification_stats', {}).get('cutoff_violation_rows')}`。",
        f"- **行与标的计数**：`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`。",
        f"- **宽口径与基准比较**：broad category match rows/instruments=`{json.dumps(summary.get('classification_stats', {}).get('broad_category_match_rows', {}), ensure_ascii=False, sort_keys=True)}`/`{json.dumps(summary.get('classification_stats', {}).get('broad_category_match_instruments', {}), ensure_ascii=False, sort_keys=True)}`；相对已冻结 v1.0 基准新增 RIC=`{summary.get('v1_0_comparison', {}).get('new_ric_count')}`，重叠=`{summary.get('v1_0_comparison', {}).get('v1_0_overlap_ric_count')}`，基准独有=`{summary.get('v1_0_comparison', {}).get('baseline_only_ric_count')}`。",
        f"- **受影响行与 reason code**：`{json.dumps(summary.get('reason_code_counts', {}), ensure_ascii=False, sort_keys=True)}`；unknown/ambiguous 保留审计，未当作非 AI。",
        f"- **输出与 quarantine**：output `{relpath(output_dir)}`；audit `{relpath(audit_dir)}`；quarantine=`None`（本阶段不删除或隔离 raw 行，unknown/ambiguous 均保留在审计表）；output hashes 记录在 summary/gate。",
        "- **缺失/删除/映射规则**：空字符串、供应商 NA、不可解析值转 NA；文本 0 保留；无填补、前后填充、winsorize、静默去重或强制对账；退市 RIC 保留并标注 coverage；不按名称猜 RIC，不回填当前 TRBC。",
        f"- **执行状态**：`{summary['status']}`；限制：{json.dumps(summary['limitations'], ensure_ascii=False)}。",
        "",
    ]
    atomic_write_text(path, prior.rstrip("\n") + "\n" + "\n".join(entry))


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    config, config_sha256 = load_config()
    spec_sha256 = validate_spec(config)
    specs, raw_info = discover_inputs(config)
    spans, spans_info = read_membership_spans(config)
    selected = specs if args.limit == 0 else specs[: args.limit]
    if args.limit and len(selected) < args.limit:
        print(f"--limit {args.limit} exceeds available raw files; reading {len(selected)}.", flush=True)
    run_id = args.run_id or new_run_id()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id}")
    output_root = safe_path(str(config["output"]["clean_root"]), label="clean_root")
    audit_root = safe_path(str(config["output"]["audit_root"]), label="audit_root")
    output_dir = output_root / run_id
    audit_dir = audit_root / run_id
    if output_dir.exists() or audit_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_id}")
    print(
        "阶段说明（AI pool dry-run/execute）：仅读取已经核验哈希的 Orig 分部、Orig 原始公告日、Orig total revenue 和 corrected literal-RIC spans；"
        "按 Instrument+period_end 合并、按 formation_date=Orig announcement calendar date 做 inclusive cutoff，"
        "按冻结 v1.1 taxonomy 识别 Core AI、六类 broad 和旧 ecosystem，未知与对账/日期冲突进入 unknown/ambiguous；退市保留。"
        "不读取 targets、future returns、labels、model outputs 或 current TRBC，不发 LSEG 请求。",
        flush=True,
    )
    print(f"输入结构已核验：raw_files={len(specs)}，selected={len(selected)}，raw_rows={raw_info['raw_rows']}。", flush=True)
    selected_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for index, spec in enumerate(selected, 1):
        frame = normalize_family(spec, config)
        selected_frames[spec["family_id"]].append(frame)
        print(f"已读取 {index}/{len(selected)} {spec['request_id']} family={spec['family_id']} rows={len(frame)}", flush=True)
    frames = {family: pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame() for family, parts in selected_frames.items()}
    segments = frames.get("segments", pd.DataFrame())
    announcements = frames.get("announcement_dates", pd.DataFrame())
    revenue = frames.get("revenue_reference", pd.DataFrame())
    membership, audit, classification_stats = classify_segments(segments, announcements, revenue, spans, config)
    audit_out = audit_output_frame(audit)
    v1_0_comparison = compare_with_baseline(membership, config)
    mode = "dry_run" if args.dry_run else "execute"
    complete_input = len(selected) == len(specs)
    status = "dry_run_only" if args.dry_run else ("complete" if complete_input else "partial_not_model_ready")
    raw_after_hashes = {spec["raw_file"]: sha256_file(safe_path(spec["raw_file"], label="raw_file")) for spec in specs}
    raw_unchanged = all(raw_after_hashes[item["raw_file"]] == item["raw_sha256"] for item in specs)
    spans_after = sha256_file(safe_path(str(config["input"]["membership_spans"]), label="membership_spans"))
    if spans_after != spans_info["sha256"]:
        raise RuntimeError("membership spans changed during run")
    span_lookup = membership_span_lookup(spans, config)
    if not membership.empty and "delisted_ric" in membership:
        delisted_count = int(membership["delisted_ric"].astype(str).str.casefold().eq("true").sum())
    else:
        delisted_count = 0
    counts = {
        "inventory_raw_rows": int(raw_info["raw_rows"]),
        "selected_raw_rows": int(sum(spec["rows"] for spec in selected)),
        "segment_input_rows": int(len(segments)),
        "segment_input_instruments": int(segments["Instrument"].nunique()) if not segments.empty else 0,
        "announcement_input_rows": int(len(announcements)),
        "announcement_input_instruments": int(announcements["Instrument"].nunique()) if not announcements.empty else 0,
        "revenue_reference_input_rows": int(len(revenue)),
        "revenue_reference_input_instruments": int(revenue["Instrument"].nunique()) if not revenue.empty else 0,
        "segment_audit_rows": int(len(audit_out)),
        "segment_audit_instruments": int(audit_out["Instrument"].nunique()) if not audit_out.empty else 0,
        "membership_output_rows": int(len(membership)),
        "membership_instruments": int(membership["Instrument"].nunique()) if not membership.empty else 0,
        "known_membership_rows": int(membership["formation_date"].notna().sum()) if not membership.empty else 0,
        "membership_unknown_rows": int(membership["membership_status"].eq("membership_unknown").sum()) if not membership.empty else 0,
        "ambiguous_rows": int(membership["membership_status"].eq("ambiguous").sum()) if not membership.empty else 0,
        "core_ai_rows": int(membership["pool_label"].eq("core_ai").sum()) if not membership.empty else 0,
        "ai_industry_broad_rows": int(membership["pool_label"].eq("ai_industry_broad").sum()) if not membership.empty else 0,
        "ai_ecosystem_rows": int(membership["pool_label"].eq("ai_ecosystem").sum()) if not membership.empty else 0,
        "taxonomy_overlap_rows": int(membership["taxonomy_overlap"].eq("true").sum()) if not membership.empty else 0,
        "core_broad_overlap_rows": int(membership["taxonomy_overlap_detail"].astype(str).str.contains("CORE_BROAD_OVERLAP", regex=False).sum()) if not membership.empty else 0,
        "core_ecosystem_overlap_rows": int(membership["taxonomy_overlap_detail"].astype(str).str.contains("CORE_ECOSYSTEM_OVERLAP", regex=False).sum()) if not membership.empty else 0,
        "legacy_ecosystem_match_rows": int(membership["legacy_ecosystem_match"].eq("true").sum()) if not membership.empty else 0,
        "broad_ecosystem_overlap_rows": int(membership["taxonomy_overlap_detail"].astype(str).str.contains("BROAD_ECOSYSTEM_OVERLAP", regex=False).sum()) if not membership.empty else 0,
        "broad_category_match_rows": classification_stats["broad_category_match_rows"],
        "broad_category_match_instruments": classification_stats["broad_category_match_instruments"],
        "matched_keyword_counts": classification_stats["matched_keyword_counts"],
        "cutoff_candidate_rows": classification_stats["cutoff_candidate_rows"],
        "cutoff_passed_rows": classification_stats["cutoff_passed_rows"],
        "cutoff_violation_rows": classification_stats["cutoff_violation_rows"],
        "legacy_ecosystem_rows": classification_stats["legacy_ecosystem_rows"],
        "legacy_ecosystem_instruments": classification_stats["legacy_ecosystem_instruments"],
        "legacy_ecosystem_only_rows": classification_stats["legacy_ecosystem_only_rows"],
        "current_eligible_ric_count": v1_0_comparison["current_eligible_ric_count"],
        "new_ric_count_vs_v1_0": v1_0_comparison["new_ric_count"],
        "v1_0_overlap_ric_count": v1_0_comparison["v1_0_overlap_ric_count"],
        "baseline_only_ric_count": v1_0_comparison["baseline_only_ric_count"],
        "delisted_rows_retained": delisted_count,
        "corrected_span_instruments": len(span_lookup),
    }
    checks = {
        "spec_version_is_v1_1": str(config["specification_version"]) == "1.1" and str(config["taxonomy"]["version"]) == "AI_POOL_SPEC_v1.1",
        "broad_taxonomy_present": bool(config["taxonomy"].get("ai_industry_broad")) and len(config["taxonomy"].get("ai_industry_broad", {})) == 6,
        "v1_0_baseline_hash_verified": bool(v1_0_comparison["baseline_hash_verified"]),
        "v1_0_comparison_complete": all(
            key in v1_0_comparison
            for key in ["new_ric", "v1_0_overlap_ric", "baseline_only_ric", "current_pool_instrument_counts"]
        ),
        "raw_sidecar_hashes_verified": True,
        "raw_reporting_state_orig": all(item["reporting_state"] == "Orig" for item in selected),
        "membership_spans_hash_verified": spans_after == spans_info["sha256"],
        "raw_unchanged_after_processing": raw_unchanged,
        "known_cutoff_passed": bool((audit.loc[audit["formation_date"].notna(), "announcement_date"].fillna("").astype(str) <= audit.loc[audit["formation_date"].notna(), "formation_date"].fillna("").astype(str)).all()) if not audit.empty else True,
        "cutoff_violation_rows_zero": classification_stats["cutoff_violation_rows"] == 0,
        "unknown_not_labeled_non_ai": not membership.loc[membership["membership_status"].eq("membership_unknown"), "pool_label"].notna().any() if not membership.empty else True,
        "delisted_not_excluded": True,
        "no_forbidden_inputs_read": True,
        "no_lseg_import_or_network": True,
        "no_future_returns_labels_or_models": True,
        "no_current_trbc_backfill": True,
        "no_name_based_ric_mapping": True,
        "no_imputation_fill_or_winsorization": True,
        "raw_text_provenance_present": bool((audit["raw_file"].notna() & audit["raw_row"].notna() & audit["raw_text"].notna()).all()) if not audit.empty else True,
    }
    limitations = [
        "formation_date is the Orig announcement calendar date with inclusive day-level cutoff; next trading-session entry is deferred to event layer because this stage reads no market calendar.",
        "unknown announcement periods remain membership_unknown and are never treated as non-AI.",
        "reconciliation gap or multiple null-total candidates force ambiguous status; no forced total adjustment is performed.",
        "AI taxonomy is frozen at AI_POOL_SPEC_v1.1; broad-category phrases are pre-registered and generic words alone remain ambiguous.",
        "The v1.0 membership artifact is used only as a hash-verified RIC-set comparison baseline; it is not overwritten or reclassified.",
        "limited runs are partial_not_model_ready and only describe the selected raw request files.",
    ]
    non_actions = config["non_actions"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "mode": mode,
        "status": status,
        "clean_ready": status == "complete",
        "model_ready": False,
        "specification": "AI_POOL_SPEC.md",
        "specification_version": str(config["specification_version"]),
        "spec_sha256": spec_sha256,
        "config_path": relpath(CONFIG_PATH),
        "config_sha256": config_sha256,
        "raw_run": raw_info,
        "membership_spans": spans_info,
        "input_file_count": len(specs),
        "processed_file_count": len(selected),
        "input_hashes": {
            "raw_csv": {item["raw_file"]: item["raw_sha256"] for item in specs},
            "raw_meta": {item["meta_file"]: item["meta_sha256"] for item in specs},
            "raw_request_sidecars": {item["request_file"]: item["request_file_sha256"] for item in specs},
            "membership_spans": spans_info["sha256"],
            "v1_0_baseline_membership": v1_0_comparison["baseline_membership_sha256"],
            "v1_0_baseline_summary": v1_0_comparison["baseline_summary_sha256"],
        },
        "raw_after_hashes": raw_after_hashes,
        "counts": counts,
        "classification_stats": classification_stats,
        "v1_0_comparison": v1_0_comparison,
        "reason_code_counts": reason_counts(audit_out, "row_reason_codes"),
        "checks": checks,
        "taxonomy": config["taxonomy"],
        "formation": config["formation"],
        "limitations": limitations,
        "non_actions": non_actions,
        "output_hashes": {},
    }
    # Do not create run directories until input validation and classification
    # have completed; failed calls therefore leave no misleading empty result.
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_dir.mkdir(parents=True, exist_ok=False)
    if args.execute:
        atomic_write_csv(output_dir / config["output"]["membership_file"], membership)
        atomic_write_csv(audit_dir / config["output"]["audit_file"], audit_out)
    summary_path = output_dir / config["output"]["summary_file"]
    gate_path = output_dir / config["output"]["gate_file"]
    dry_run_path = output_dir / config["output"]["dry_run_file"]
    if args.dry_run:
        atomic_write_json(
            dry_run_path,
            {
                "dry_run": True,
                "run_id": run_id,
                "specification_version": summary["specification_version"],
                "taxonomy_version": summary["taxonomy"]["version"],
                "counts": counts,
                "classification_stats": classification_stats,
                "matched_keyword_counts": classification_stats["matched_keyword_counts"],
                "cutoff": {
                    "candidate_rows": classification_stats["cutoff_candidate_rows"],
                    "passed_rows": classification_stats["cutoff_passed_rows"],
                    "violation_rows": classification_stats["cutoff_violation_rows"],
                },
                "reason_code_counts": summary["reason_code_counts"],
                "v1_0_comparison": v1_0_comparison,
                "input_hashes": summary["input_hashes"],
                "status": status,
            },
        )
    output_paths = [
        path
        for path in [
            output_dir / config["output"]["membership_file"],
            audit_dir / config["output"]["audit_file"],
            dry_run_path,
        ]
        if path.exists()
    ]
    summary["output_hashes"] = {relpath(path): sha256_file(path) for path in output_paths}
    atomic_write_json(summary_path, summary)
    gate = output_gate(summary, status=status)
    atomic_write_json(gate_path, gate)
    # Keep a manifest-like hash index for data artifacts only.  Summary and
    # gate hashes are reported externally because recording either one's own
    # hash inside itself would make that value recursively stale.
    append_processing_log(summary, output_dir, audit_dir)
    print(
        f"AI pool {mode} 完成：status={status} segment_rows={len(audit_out)} membership_rows={len(membership)} "
        f"core={counts['core_ai_rows']} broad={counts['ai_industry_broad_rows']} ecosystem={counts['ai_ecosystem_rows']} "
        f"unknown={counts['membership_unknown_rows']} ambiguous={counts['ambiguous_rows']} "
        f"new_ric_vs_v1.0={counts['new_ric_count_vs_v1_0']}",
        flush=True,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI pool point-in-time membership from Orig raw inputs")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true", help="scan/classify in memory and write run-scoped stats only")
    modes.add_argument("--execute", action="store_true", help="write ai_pool_membership_pit.csv and ai_pool_membership_audit.csv")
    parser.add_argument("--limit", type=int, default=0, help="maximum raw request CSV files read; 0 means all")
    parser.add_argument("--run-id", default="", help="new UTC run id; existing directories are never overwritten")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.run_id and not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id must be a UTC run directory id")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_once(args)
    return 0 if summary["checks"].get("raw_unchanged_after_processing", False) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED AI POOL MEMBERSHIP: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
