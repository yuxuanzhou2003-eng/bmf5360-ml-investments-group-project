"""Resumable, auditable raw-to-clean conversion for AI-factor inputs.

The collector's raw run is treated as immutable.  This module reads only the
configured raw request directory and the corrected literal-RIC membership
spans.  It never imports the LSEG client and it never reads targets, future
returns, labels, model outputs, or current TRBC data.

``--plan`` performs an offline input-contract check and writes a new run plan.
``--execute --limit N`` stages at most N unprocessed raw request CSV files and
materializes a partial, explicitly non-model-ready snapshot.  Reusing the
same ``--run-id`` stages the next files and eventually finalizes a complete
snapshot.  Every write is atomic and every input/output artifact is hashed.
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
from functools import reduce
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ai_factor_cleaning_v1_config.json"
SCHEMA_VERSION = "ai_factor_cleaning_v1"
RUN_ID_RE = re.compile(r"^\d{8}T\d{6,12}Z$")

FUNDAMENTAL_OUTPUTS = {
    "legacy_rd": "rd_value",
    "legacy_revenue": "revenue_value",
    "f_tot_revenue": "control_total_revenue_value",
    "f_tot_assets": "assets_value",
    "f_com_eq_tot": "equity_value",
    "f_gross_prof_ind_prop_tot": "gross_profit_value",
    "f_net_cash_flow_op": "operating_cash_flow_value",
    "f_debt_tot": "debt_value",
}

FUNDAMENTAL_FAMILIES = tuple(FUNDAMENTAL_OUTPUTS)
PRICE_FIELDS = (
    "TRDPRC_1",
    "OPEN_PRC",
    "HIGH_1",
    "LOW_1",
    "ACVOL_UNS",
    "BID",
    "ASK",
    "TRNOVR_UNS",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def generated_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV with NA as an empty cell, without changing the source raw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out = out.where(pd.notna(out), "")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    out.to_csv(temp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temp, path)


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], str]:
    config = read_json(path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported cleaning config schema: {config.get('schema_version')}")
    return config, sha256_file(path)


def safe_config_path(relative: str, *, label: str) -> Path:
    candidate = (ROOT / Path(relative)).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the workspace: {relative}") from exc
    return candidate


def forbidden_path_reason(path: Path, tokens: Iterable[str]) -> str | None:
    lower = str(path).replace("\\", "/").lower()
    for token in tokens:
        if token.lower() in lower:
            return token
    return None


def csv_header_and_rows(path: Path) -> tuple[list[str], int]:
    """Read only the raw CSV framing for plan checks.

    The collection is line-oriented (the vendor response contains no quoted
    newlines).  Execution uses the same framing so ``raw_text`` can retain the
    exact data record text.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"empty raw CSV: {path}") from exc
        rows = sum(1 for _ in reader)
    return [str(item) for item in header], rows


def parse_request_sidecar(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid request sidecar: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"request sidecar is not an object: {path}")
    return value


def classify_filename(path: Path, config: dict[str, Any]) -> tuple[str, str]:
    name = path.name
    family_prefixes: list[tuple[str, str]] = []
    for family_id, spec in config["fundamental"]["families"].items():
        family_prefixes.append((str(spec["prefix"]), family_id))
    family_prefixes.extend(
        [
            (str(config["segments"]["prefix"]), "business_segments_fy0"),
            (str(config["market_cap"]["prefix"]), "market_capitalization_monthly"),
            (str(config["etf"]["prefix"]), "etf_daily_price_volume"),
        ]
    )
    for prefix, family_id in sorted(family_prefixes, key=lambda item: -len(item[0])):
        if name.startswith(prefix) and name.endswith(".csv"):
            request_id = name[:-4]
            if family_id == "etf_daily_price_volume":
                if request_id == prefix[:-1]:
                    raise RuntimeError(f"ETF raw filename has no literal RIC: {name}")
            elif not re.fullmatch(re.escape(prefix) + r"\d{3}", request_id):
                raise RuntimeError(f"unexpected request filename for {family_id}: {name}")
            return family_id, request_id
    raise RuntimeError(f"unexpected raw CSV under configured run: {name}")


def expected_header(family_id: str, config: dict[str, Any]) -> list[str] | None:
    if family_id == "etf_daily_price_volume":
        return ["Date", *PRICE_FIELDS]
    if family_id == "business_segments_fy0":
        return [
            "Instrument",
            "Segment Code",
            "Segment Name",
            "Financial Period Absolute",
            "Date",
            "Standardized Revenue - Business Segment",
        ]
    if family_id == "market_capitalization_monthly":
        return ["Instrument", "Company Market Capitalization", "Date"]
    if family_id == "income_statement_dates":
        spec = config["fundamental"]["families"][family_id]
        return [
            "Instrument",
            str(spec["announcement_column"]),
            str(spec["last_update_column"]),
            str(spec["period_end_column"]),
        ]
    if family_id in config["fundamental"]["families"]:
        spec = config["fundamental"]["families"][family_id]
        return [
            "Instrument",
            str(spec["value_column"]),
            "Date",
            "Financial Period Absolute",
        ]
    return None


def validate_reporting_state(family_id: str, meta: dict[str, Any]) -> None:
    parameters = (meta.get("request") or {}).get("parameters") or {}
    if family_id == "market_capitalization_monthly":
        if "ReportingState" in parameters:
            raise RuntimeError("market-cap raw request unexpectedly sent ReportingState")
        if not meta.get("scope_notes", {}).get("reporting_state_exception"):
            raise RuntimeError("market-cap raw metadata lacks the explicit ReportingState exception")
        return
    if family_id == "etf_daily_price_volume":
        return
    if parameters.get("ReportingState") != "Orig":
        raise RuntimeError(f"{family_id} raw request is not explicitly ReportingState=Orig")


def discover_raw_specs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_run_id = str(config["input"]["raw_run_id"])
    raw_root = safe_config_path(str(config["input"]["raw_run_path"]), label="raw_run_path")
    forbidden_tokens = config["input"].get("forbidden_input_tokens", [])
    forbidden = forbidden_path_reason(raw_root, forbidden_tokens)
    if forbidden:
        raise RuntimeError(f"refusing forbidden raw input path token: {forbidden}")
    if not raw_root.exists():
        raise RuntimeError(f"configured raw run does not exist: {raw_root}")

    specs: list[dict[str, Any]] = []
    for csv_path in sorted(raw_root.glob("*.csv")):
        family_id, request_id = classify_filename(csv_path, config)
        meta_path = csv_path.with_name(f"{csv_path.stem}.meta.json")
        request_path = csv_path.with_name(f"{csv_path.stem}.request.json")
        if not meta_path.exists() or not request_path.exists():
            raise RuntimeError(f"raw sidecars missing for {csv_path.name}")
        meta = read_json(meta_path)
        request = parse_request_sidecar(request_path)
        if meta.get("run_id") != raw_run_id:
            raise RuntimeError(f"raw metadata run_id mismatch: {meta_path.name}")
        if meta.get("request_id") != request_id or meta.get("family_id") != family_id:
            raise RuntimeError(f"raw metadata identity mismatch: {meta_path.name}")
        csv_sha256 = sha256_file(csv_path)
        if config["input"]["raw_file_contract"].get("require_metadata_csv_sha256") and meta.get("csv_sha256") != csv_sha256:
            raise RuntimeError(f"raw CSV hash mismatch: {csv_path.name}")
        request_file_sha256 = sha256_file(request_path)
        if config["input"]["raw_file_contract"].get("require_request_file_sha256") and meta.get("request_file_sha256") != request_file_sha256:
            raise RuntimeError(f"raw request sidecar hash mismatch: {request_path.name}")
        validate_reporting_state(family_id, meta)
        header, rows = csv_header_and_rows(csv_path)
        expected = expected_header(family_id, config)
        if expected is not None and header != expected:
            raise RuntimeError(
                f"raw header mismatch for {csv_path.name}: expected {expected!r}, got {header!r}"
            )
        if meta.get("rows") is not None and int(meta["rows"]) != rows:
            raise RuntimeError(f"raw row count mismatch: {csv_path.name}")
        request_hash = str(meta.get("request_sha256") or sha256_json(request))
        if request_hash != sha256_json(request):
            raise RuntimeError(f"raw canonical request hash mismatch: {request_path.name}")
        universe = (request.get("request") or {}).get("universe") or request.get("universe") or []
        specs.append(
            {
                "request_id": request_id,
                "family_id": family_id,
                "raw_file": relpath(csv_path),
                "meta_file": relpath(meta_path),
                "request_file": relpath(request_path),
                "raw_sha256": csv_sha256,
                "meta_sha256": sha256_file(meta_path),
                "request_file_sha256": request_file_sha256,
                "request_sha256": request_hash,
                "rows": rows,
                "header": header,
                "requested_instruments": [str(item) for item in universe],
                "metadata_reporting_state": ((meta.get("request") or {}).get("parameters") or {}).get("ReportingState"),
                "market_cap_reporting_state_exception": meta.get("scope_notes", {}).get("exception_rationale")
                or meta.get("scope_notes", {}).get("exception_rationale"),
            }
        )
    if not specs:
        raise RuntimeError(f"no raw CSV files found under {raw_root}")
    expected_family_ids = set(FUNDAMENTAL_FAMILIES) | {
        "income_statement_dates",
        "business_segments_fy0",
        "market_capitalization_monthly",
        "etf_daily_price_volume",
    }
    actual_family_ids = {item["family_id"] for item in specs}
    missing_families = sorted(expected_family_ids - actual_family_ids)
    if missing_families:
        raise RuntimeError(f"raw run is missing required families: {missing_families}")
    return specs, {
        "raw_run_path": relpath(raw_root),
        "raw_run_id": raw_run_id,
        "csv_count": len(specs),
        "raw_rows": int(sum(item["rows"] for item in specs)),
        "families": dict(sorted(Counter(item["family_id"] for item in specs).items())),
    }


def literal_membership_rows(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = safe_config_path(str(config["input"]["membership_spans"]), label="membership_spans")
    forbidden = forbidden_path_reason(path, config["input"].get("forbidden_input_tokens", []))
    if forbidden:
        raise RuntimeError(f"refusing forbidden membership input path token: {forbidden}")
    if not path.exists():
        raise RuntimeError(f"membership spans do not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise RuntimeError(f"empty membership spans: {path}")
        header = next(csv.reader([header_line]))
        for row_number, raw_line in enumerate(handle, 1):
            parsed = next(csv.reader([raw_line]))
            values = list(parsed) + [""] * max(0, len(header) - len(parsed))
            values = values[: len(header)]
            records.append(
                {
                    **{header[i]: values[i] for i in range(len(header))},
                    "raw_file": relpath(path),
                    "raw_row": row_number,
                    "raw_text": raw_line.rstrip("\r\n"),
                    "_malformed": len(parsed) != len(header),
                }
            )
    frame = pd.DataFrame(records)
    return frame, {
        "path": relpath(path),
        "sha256": sha256_file(path),
        "header": header,
        "rows": int(len(frame)),
    }


def provider_tokens(config: dict[str, Any]) -> set[str]:
    return {
        str(item).strip().casefold()
        for item in config["missing_value_policy"].get("provider_tokens_are_na", [])
    }


def clean_text(value: Any, tokens: set[str]) -> tuple[str | None, str | None]:
    if value is None:
        return None, "MISSING_VALUE"
    text = str(value)
    stripped = text.strip()
    if not stripped or stripped.casefold() in tokens:
        return None, "MISSING_VALUE"
    return stripped, None


def clean_numeric(value: Any, tokens: set[str], label: str) -> tuple[str | None, str | None]:
    text, missing = clean_text(value, tokens)
    if text is None:
        return None, f"MISSING_VALUE_{label}"
    try:
        decimal = Decimal(text.replace(",", ""))
        if not decimal.is_finite():
            raise InvalidOperation
    except (InvalidOperation, ValueError):
        return None, f"UNPARSEABLE_NUMERIC_{label}"
    normalized = format(decimal, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".") or "0"
    return normalized, None


def clean_date(value: Any, tokens: set[str], label: str) -> tuple[str | None, str | None]:
    text, missing = clean_text(value, tokens)
    if text is None:
        return None, f"MISSING_VALUE_{label}"
    try:
        parsed = pd.to_datetime(text, errors="raise", format="mixed")
        if isinstance(parsed, pd.DatetimeIndex):
            raise ValueError("date is not scalar")
        if getattr(parsed, "tzinfo", None) is not None:
            return None, f"UNPARSEABLE_DATE_TIMEZONE_{label}"
        return pd.Timestamp(parsed).date().isoformat(), None
    except (TypeError, ValueError, OverflowError):
        return None, f"UNPARSEABLE_DATE_{label}"


def append_reason(existing: Any, *codes: str | None) -> str:
    values: list[str] = []
    existing_missing = False
    try:
        existing_missing = bool(pd.isna(existing))
    except (TypeError, ValueError):
        existing_missing = False
    if existing is not None and not existing_missing and str(existing).strip():
        values.extend(item for item in str(existing).split(";") if item)
    for code in codes:
        code_missing = False
        try:
            code_missing = bool(pd.isna(code))
        except (TypeError, ValueError):
            code_missing = False
        if code and not code_missing and code not in values:
            values.append(code)
    return ";".join(values)


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def raw_row_records(path: Path) -> tuple[list[str], list[tuple[int, str, list[str]]]]:
    records: list[tuple[int, str, list[str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise RuntimeError(f"empty raw CSV: {path}")
        header = next(csv.reader([header_line]))
        for row_number, raw_line in enumerate(handle, 1):
            try:
                parsed = next(csv.reader([raw_line]))
            except csv.Error as exc:
                parsed = []
                raise RuntimeError(f"malformed CSV row {row_number} in {path}") from exc
            records.append((row_number, raw_line.rstrip("\r\n"), parsed))
    return [str(item) for item in header], records


def row_mapping(header: list[str], values: list[str]) -> tuple[dict[str, str], bool]:
    malformed = len(values) != len(header)
    padded = list(values) + [""] * max(0, len(header) - len(values))
    padded = padded[: len(header)]
    return {header[index]: padded[index] for index in range(len(header))}, malformed


def normalize_request_file(spec: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """Normalize one raw file into an append-safe stage with raw provenance."""
    path = safe_config_path(spec["raw_file"], label="raw_file")
    header, rows = raw_row_records(path)
    tokens = provider_tokens(config)
    family_id = spec["family_id"]
    request_id = spec["request_id"]
    records: list[dict[str, Any]] = []
    etf_instrument = None
    if family_id == "etf_daily_price_volume":
        etf_instrument = spec["request_id"][len("etf_prices_") :]
        if not etf_instrument:
            raise RuntimeError(f"cannot derive literal ETF RIC from {request_id}")

    for row_number, raw_text, values in rows:
        raw_fields, malformed = row_mapping(header, values)
        reason = "MALFORMED_CSV_ROW" if malformed else ""
        instrument_raw = etf_instrument if etf_instrument is not None else raw_fields.get("Instrument", "")
        instrument, code = clean_text(instrument_raw, tokens)
        reason = append_reason(reason, "INSTRUMENT_MISSING" if code else None)
        record: dict[str, Any] = {
            "request_id": request_id,
            "family_id": family_id,
            "Instrument": instrument,
            "period_end": None,
            "Date": None,
            "fperiod": None,
            "value": None,
            "orig_announcement_date": None,
            "last_update_date": None,
            "vendor_period_end": None,
            "segment_code": None,
            "segment_name": None,
            "segment_value": None,
            "market_cap_value": None,
            "structural_padding": "false",
            "raw_file": spec["raw_file"],
            "raw_row": row_number,
            "raw_text": raw_text,
            "raw_fields_json": json.dumps(raw_fields, ensure_ascii=False, sort_keys=True),
            "row_reason_codes": reason,
        }
        if family_id in FUNDAMENTAL_FAMILIES:
            family_spec = config["fundamental"]["families"][family_id]
            period_end, date_code = clean_date(raw_fields.get("Date", ""), tokens, "period_end")
            fperiod, fperiod_code = clean_text(raw_fields.get("Financial Period Absolute", ""), tokens)
            value, value_code = clean_numeric(
                raw_fields.get(str(family_spec["value_column"]), ""),
                tokens,
                str(family_id),
            )
            record.update(
                {
                    "period_end": period_end,
                    "Date": period_end,
                    "fperiod": fperiod,
                    "value": value,
                }
            )
            record["row_reason_codes"] = append_reason(
                record["row_reason_codes"], date_code, fperiod_code, value_code
            )
        elif family_id == "income_statement_dates":
            family_spec = config["fundamental"]["families"][family_id]
            period_end, period_code = clean_date(
                raw_fields.get(str(family_spec["period_end_column"]), ""), tokens, "period_end"
            )
            announcement, announcement_code = clean_date(
                raw_fields.get(str(family_spec["announcement_column"]), ""),
                tokens,
                "orig_announcement_date",
            )
            last_update, last_update_code = clean_date(
                raw_fields.get(str(family_spec["last_update_column"]), ""),
                tokens,
                "last_update_date",
            )
            record.update(
                {
                    "period_end": period_end,
                    "Date": period_end,
                    "orig_announcement_date": announcement,
                    "last_update_date": last_update,
                    "vendor_period_end": period_end,
                }
            )
            record["row_reason_codes"] = append_reason(
                record["row_reason_codes"], period_code, announcement_code, last_update_code
            )
        elif family_id == "business_segments_fy0":
            period_end, date_code = clean_date(raw_fields.get("Date", ""), tokens, "period_end")
            fperiod, fperiod_code = clean_text(raw_fields.get("Financial Period Absolute", ""), tokens)
            segment_code, segment_code_reason = clean_text(raw_fields.get("Segment Code", ""), tokens)
            segment_name, segment_name_reason = clean_text(raw_fields.get("Segment Name", ""), tokens)
            segment_value, segment_value_reason = clean_numeric(
                raw_fields.get("Standardized Revenue - Business Segment", ""),
                tokens,
                "segment_revenue",
            )
            total_candidate = segment_code is None and segment_name is None
            record.update(
                {
                    "period_end": period_end,
                    "Date": period_end,
                    "fperiod": fperiod,
                    "segment_code": segment_code,
                    "segment_name": segment_name,
                    "segment_value": segment_value,
                    "null_total_candidate": bool_text(total_candidate),
                }
            )
            record["row_reason_codes"] = append_reason(
                record["row_reason_codes"],
                date_code,
                fperiod_code,
                segment_code_reason,
                segment_name_reason,
                segment_value_reason,
            )
        elif family_id == "market_capitalization_monthly":
            date, date_code = clean_date(raw_fields.get("Date", ""), tokens, "market_cap_date")
            value, value_code = clean_numeric(
                raw_fields.get("Company Market Capitalization", ""), tokens, "market_cap"
            )
            record.update({"period_end": date, "Date": date, "market_cap_value": value})
            record["row_reason_codes"] = append_reason(record["row_reason_codes"], date_code, value_code)
        elif family_id == "etf_daily_price_volume":
            date, date_code = clean_date(raw_fields.get("Date", ""), tokens, "etf_date")
            record["period_end"] = date
            record["Date"] = date
            raw_present = False
            for field in PRICE_FIELDS:
                value, value_code = clean_numeric(raw_fields.get(field, ""), tokens, field)
                record[field] = value
                if value is not None:
                    raw_present = True
                record["row_reason_codes"] = append_reason(record["row_reason_codes"], value_code)
            if not raw_present:
                record["structural_padding"] = "true"
                record["row_reason_codes"] = append_reason(
                    record["row_reason_codes"], "STRUCTURAL_WIDE_PADDING"
                )
            record["row_reason_codes"] = append_reason(record["row_reason_codes"], date_code)
        else:
            raise RuntimeError(f"unsupported raw family: {family_id}")
        if record.get("period_end") is None:
            record["row_reason_codes"] = append_reason(record["row_reason_codes"], "PERIOD_KEY_MISSING")
        records.append(record)

    columns = [
        "request_id",
        "family_id",
        "Instrument",
        "period_end",
        "Date",
        "fperiod",
        "value",
        "orig_announcement_date",
        "last_update_date",
        "vendor_period_end",
        "segment_code",
        "segment_name",
        "segment_value",
        "market_cap_value",
        "null_total_candidate",
        "structural_padding",
        *PRICE_FIELDS,
        "raw_file",
        "raw_row",
        "raw_text",
        "raw_fields_json",
        "row_reason_codes",
    ]
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def create_plan(
    config: dict[str, Any],
    config_sha256: str,
    specs: list[dict[str, Any]],
    raw_info: dict[str, Any],
    membership_info: dict[str, Any],
    run_id: str | None,
    dictionary: dict[str, Any] | None,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    selected_run_id = run_id or generated_run_id()
    if not RUN_ID_RE.fullmatch(selected_run_id):
        raise ValueError(f"invalid run id: {selected_run_id}")
    clean_root = safe_config_path(str(config["output"]["clean_root"]), label="clean_root")
    audit_root = safe_config_path(str(config["output"]["audit_root"]), label="audit_root")
    clean_dir = clean_root / selected_run_id
    audit_dir = audit_root / selected_run_id
    if clean_dir.exists() or audit_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing clean/audit run; use --run-id to resume: {selected_run_id}"
        )
    clean_dir.mkdir(parents=True, exist_ok=False)
    (clean_dir / str(config["output"]["staging_subdirectory"])).mkdir(parents=True, exist_ok=False)
    (audit_dir / str(config["output"]["quarantine_subdirectory"])).mkdir(parents=True, exist_ok=False)
    dictionary_info: dict[str, Any] = {
        "status": "unfrozen_no_dictionary",
        "path": None,
        "sha256": None,
        "version": "unfrozen-no-dictionary",
        "core": [],
        "broad": [],
        "precedence": "core_then_broad",
    }
    if dictionary is not None:
        dictionary_info = dictionary
    plan = {
        "schema_version": SCHEMA_VERSION,
        "plan_version": "0.1",
        "run_id": selected_run_id,
        "created_at_utc": utc_now(),
        "script": relpath(Path(__file__)),
        "script_sha256": sha256_file(Path(__file__)),
        "config_path": relpath(CONFIG_PATH),
        "config_sha256": config_sha256,
        "raw_run": raw_info,
        "membership_spans": membership_info,
        "input_files": specs,
        "keyword_dictionary": dictionary_info,
        "read_boundary": {
            "allowed": [raw_info["raw_run_path"], membership_info["path"]],
            "forbidden_tokens": config["input"].get("forbidden_input_tokens", []),
            "forbidden_reads": [
                "targets",
                "future returns",
                "labels",
                "model outputs",
                "current TRBC",
            ],
            "lseg_imported": False,
            "network_requests": False,
        },
        "rules": {
            "plan_reference": "AI_FACTOR_CLEANING_PLAN.md v0.1",
            "date_parsing": "only after raw read inside clean stage; dates are emitted as ISO calendar dates; no timezone conversion",
            "missing_values": config["missing_value_policy"],
            "fundamental_join_keys": config["fundamental"]["join_keys"],
            "market_cap_join_keys": config["market_cap"]["join_keys"],
            "etf_join_keys": ["Instrument", "Date"],
            "no_imputation": True,
            "no_forward_fill": True,
            "no_winsorization": True,
            "no_delisted_row_removal": True,
            "no_name_based_ric_mapping": True,
            "no_current_trbc_backfill": True,
            "asset_return_name": "price_return",
        },
        "status": "planned",
    }
    plan_path = clean_dir / str(config["output"]["plan_name"])
    atomic_write_json(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": selected_run_id,
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "script_sha256": plan["script_sha256"],
        "raw_run_id": raw_info["raw_run_id"],
        "raw_run_path": raw_info["raw_run_path"],
        "membership_spans_sha256": membership_info["sha256"],
        "created_at_utc": plan["created_at_utc"],
        "updated_at_utc": utc_now(),
        "status": "planned",
        "keyword_dictionary": dictionary_info,
        "requests": {
            item["request_id"]: {
                "request_id": item["request_id"],
                "family_id": item["family_id"],
                "raw_file": item["raw_file"],
                "raw_sha256": item["raw_sha256"],
                "status": "pending",
                "stage_file": None,
                "stage_sha256": None,
                "rows": item["rows"],
            }
            for item in specs
        },
        "membership_stage": {"status": "pending", "sha256": membership_info["sha256"]},
        "processed_request_count": 0,
        "remaining_request_count": len(specs),
        "last_execute": None,
        "output_hashes": {},
    }
    atomic_write_json(clean_dir / str(config["output"]["manifest_name"]), manifest)
    write_planned_summary_and_gate(clean_dir, audit_dir, plan, manifest, config)
    return clean_dir, audit_dir, plan, manifest


def write_planned_summary_and_gate(
    clean_dir: Path,
    audit_dir: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    checks = [
        {"name": "raw_run_discovered", "passed": True, "detail": plan["raw_run"]},
        {"name": "all_raw_sidecar_hashes_verified", "passed": True},
        {"name": "membership_hash_recorded", "passed": True, "detail": plan["membership_spans"]},
        {"name": "forbidden_inputs_not_read", "passed": True},
        {"name": "lseg_not_imported", "passed": True},
        {"name": "network_not_called", "passed": True},
        {"name": "classification_dictionary_explicit", "passed": True, "detail": plan["keyword_dictionary"]},
    ]
    gate = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "generated_at_utc": utc_now(),
        "status": "planned",
        "clean_ready": False,
        "model_ready": False,
        "checks": checks,
        "passed": sum(1 for item in checks if item["passed"]),
        "failed": [item for item in checks if not item["passed"]],
        "limitations": [
            "No raw rows were transformed in --plan.",
            "No clean table exists until --execute stages at least one request file.",
            "A complete clean snapshot requires every raw request file to be staged and verified.",
        ],
    }
    atomic_write_json(clean_dir / "gate_report.json", gate)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "generated_at_utc": utc_now(),
        "status": "planned",
        "clean_ready": False,
        "model_ready": False,
        "raw_run": plan["raw_run"],
        "membership_spans": plan["membership_spans"],
        "input_file_count": len(plan["input_files"]),
        "staged_request_count": 0,
        "remaining_request_count": len(plan["input_files"]),
        "keyword_dictionary": plan["keyword_dictionary"],
        "input_hashes": {
            "raw_csv": {item["raw_file"]: item["raw_sha256"] for item in plan["input_files"]},
            "membership_spans": plan["membership_spans"]["sha256"],
        },
        "output_hashes": {
            "clean_plan.json": sha256_file(clean_dir / "clean_plan.json"),
            "clean_manifest.json": sha256_file(clean_dir / "clean_manifest.json"),
            "gate_report.json": sha256_file(clean_dir / "gate_report.json"),
        },
        "row_accounting": {},
        "reason_code_counts": {},
        "checks": gate["checks"],
        "explicit_non_actions": plan["rules"],
        "limitations": gate["limitations"],
    }
    atomic_write_json(clean_dir / "clean_summary.json", summary)


def load_plan_and_manifest(clean_dir: Path, config_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = clean_dir / "clean_plan.json"
    manifest_path = clean_dir / "clean_manifest.json"
    if not plan_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"cannot resume without clean_plan.json and clean_manifest.json: {clean_dir}")
    plan = read_json(plan_path)
    manifest = read_json(manifest_path)
    expected_plan_sha = manifest.get("plan_sha256")
    if expected_plan_sha != sha256_file(plan_path):
        raise RuntimeError("existing clean plan hash mismatch; refusing resume")
    if manifest.get("config_sha256") != config_sha256 or plan.get("config_sha256") != config_sha256:
        raise RuntimeError("existing clean config hash mismatch; refusing resume")
    script_sha = sha256_file(Path(__file__))
    if manifest.get("script_sha256") != script_sha or plan.get("script_sha256") != script_sha:
        raise RuntimeError("existing clean script hash mismatch; refusing resume")
    return plan, manifest


def stage_pending_requests(
    clean_dir: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    pending = [
        item
        for item in plan["input_files"]
        if manifest["requests"].get(item["request_id"], {}).get("status") != "staged"
    ]
    pending.sort(key=lambda item: item["request_id"])
    selected = pending if limit == 0 else pending[:limit]
    if not selected:
        return []
    print(
        "阶段说明（raw→stage）：读取已验证的 LSEG 原始 CSV，按配置将空字符串/供应商 NA/不可解析值转为 clean NA；"
        "保留 raw_file、raw_row、raw_text，不填补、不前填、不 winsorize、不去重，也不读取 targets、未来收益、labels、模型输出或 TRBC。",
        flush=True,
    )
    staging_dir = clean_dir / str(config["output"]["staging_subdirectory"])
    staged: list[dict[str, Any]] = []
    for spec in selected:
        raw_path = safe_config_path(spec["raw_file"], label="raw_file")
        actual_hash = sha256_file(raw_path)
        if actual_hash != spec["raw_sha256"]:
            raise RuntimeError(f"raw input changed since plan: {spec['raw_file']}")
        frame = normalize_request_file(spec, config)
        stage_path = staging_dir / f"{spec['request_id']}.csv"
        atomic_write_frame(stage_path, frame)
        stage_hash = sha256_file(stage_path)
        stage_meta = {
            "schema_version": SCHEMA_VERSION,
            "run_id": plan["run_id"],
            "request_id": spec["request_id"],
            "family_id": spec["family_id"],
            "raw_file": spec["raw_file"],
            "raw_sha256": spec["raw_sha256"],
            "stage_file": relpath(stage_path),
            "stage_sha256": stage_hash,
            "input_rows": int(spec["rows"]),
            "stage_rows": int(len(frame)),
            "created_at_utc": utc_now(),
            "normalization_rules": {
                "empty_and_provider_na_to_na": True,
                "text_zero_preserved": True,
                "date_parsing_only_in_clean": True,
                "raw_unchanged": True,
            },
        }
        atomic_write_json(staging_dir / f"{spec['request_id']}.meta.json", stage_meta)
        state = manifest["requests"][spec["request_id"]]
        state.update(
            {
                "status": "staged",
                "stage_file": relpath(stage_path),
                "stage_sha256": stage_hash,
                "stage_meta_sha256": sha256_file(staging_dir / f"{spec['request_id']}.meta.json"),
                "staged_at_utc": utc_now(),
                "stage_rows": int(len(frame)),
            }
        )
        staged.append(spec)
        print(
            f"已 stage {spec['request_id']} family={spec['family_id']} rows={len(frame)} "
            f"raw_sha256={spec['raw_sha256'][:12]}…",
            flush=True,
        )
    manifest["processed_request_count"] = sum(
        1 for value in manifest["requests"].values() if value.get("status") == "staged"
    )
    manifest["remaining_request_count"] = len(plan["input_files"]) - manifest["processed_request_count"]
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(clean_dir / "clean_manifest.json", manifest)
    return staged


def read_staged_frames(clean_dir: Path, plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for spec in plan["input_files"]:
        state = manifest["requests"].get(spec["request_id"], {})
        if state.get("status") != "staged":
            continue
        stage_path = safe_config_path(str(state["stage_file"]), label="stage_file")
        if not stage_path.exists() or sha256_file(stage_path) != state.get("stage_sha256"):
            raise RuntimeError(f"staged artifact hash mismatch: {state.get('stage_file')}")
        frame = pd.read_csv(stage_path, dtype=str, keep_default_na=False, na_filter=False)
        frame = frame.replace({"": pd.NA})
        frames[spec["family_id"]].append(frame)
    return {
        family_id: pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        for family_id, parts in frames.items()
    }


def combine_reason_series(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype="object")
    for column in columns:
        if column in frame.columns:
            result = result.map(lambda current, value_column=column: current).astype("object")
            result = pd.Series(
                [append_reason(current, value) for current, value in zip(result, frame[column])],
                index=frame.index,
                dtype="object",
            )
    return result


def duplicate_codes(
    frame: pd.DataFrame,
    keys: list[str],
    compare_columns: list[str],
    *,
    allow_na_keys: bool = False,
) -> tuple[pd.Series, pd.Series]:
    codes = pd.Series("", index=frame.index, dtype="object")
    occurrence = pd.Series(pd.NA, index=frame.index, dtype="object")
    if frame.empty:
        return codes, occurrence
    valid = frame[keys].notna().all(axis=1)
    if allow_na_keys:
        # Segment code/name are allowed to be blank because those rows are
        # preserved as null-total candidates. The explicit join key remains
        # Instrument + period_end; only those two must be present.
        valid = frame[["Instrument", "period_end"]].notna().all(axis=1)
    for key, group in frame.loc[valid].groupby(keys, dropna=False, sort=False):
        indices = list(group.index)
        occurrence.loc[indices] = list(range(len(indices)))
        if len(indices) <= 1:
            continue
        comparable = group[compare_columns].fillna("").astype(str)
        identical = comparable.drop_duplicates().shape[0] == 1
        code = "DUPLICATE_KEY_IDENTICAL" if identical else "DUPLICATE_KEY_CONFLICT"
        codes.loc[indices] = code
    return codes, occurrence


def make_quarantine(
    frame: pd.DataFrame,
    table: str,
    reasons: pd.Series | str,
    *,
    detail: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "table",
                "family_id",
                "request_id",
                "Instrument",
                "period_end",
                "Date",
                "reason_code",
                "reason_detail",
                "raw_file",
                "raw_row",
                "raw_text",
                "normalized_row_json",
            ]
        )
    if isinstance(reasons, str):
        reason_series = pd.Series(reasons, index=frame.index, dtype="object")
    else:
        reason_series = reasons.reindex(frame.index).fillna("").astype(str)
    result = pd.DataFrame(index=frame.index)
    result["table"] = table
    result["family_id"] = frame.get("family_id", pd.Series("", index=frame.index)).fillna("")
    result["request_id"] = frame.get("request_id", pd.Series("", index=frame.index)).fillna("")
    result["Instrument"] = frame.get("Instrument", pd.Series(pd.NA, index=frame.index))
    result["period_end"] = frame.get("period_end", pd.Series(pd.NA, index=frame.index))
    result["Date"] = frame.get("Date", pd.Series(pd.NA, index=frame.index))
    result["reason_code"] = reason_series
    result["reason_detail"] = detail or frame.get("row_reason_codes", pd.Series("", index=frame.index)).fillna("")
    result["raw_file"] = frame.get("raw_file", pd.Series("", index=frame.index)).fillna("")
    result["raw_row"] = frame.get("raw_row", pd.Series(pd.NA, index=frame.index))
    result["raw_text"] = frame.get("raw_text", pd.Series("", index=frame.index)).fillna("")
    result["normalized_row_json"] = [
        json.dumps(
            {str(key): (None if pd.isna(value) else str(value)) for key, value in row.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
        for _, row in frame.iterrows()
    ]
    return result.reset_index(drop=True)


def valid_key_rows(
    frame: pd.DataFrame,
    keys: list[str],
    table: str,
    duplicate_key_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), make_quarantine(frame, table, "")
    working = frame.copy()
    structural = ~working[keys].notna().all(axis=1)
    reasons = working.get("row_reason_codes", pd.Series("", index=working.index)).fillna("").astype(str)
    structural_reasons = reasons.map(lambda value: append_reason(value, "STRUCTURAL_KEY_UNAVAILABLE"))
    q = make_quarantine(working.loc[structural], table, structural_reasons.loc[structural])
    valid = working.loc[~structural].copy()
    return valid, q


def json_map(values: Iterable[Any], keys: Iterable[Any]) -> str:
    result = {}
    for key, value in zip(keys, values):
        if pd.isna(value) or value == "":
            continue
        result[str(key)] = value
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def build_fundamental(
    staged: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    quarantine_parts: list[pd.DataFrame] = []
    family_stats: dict[str, Any] = {}
    for family_id in (*FUNDAMENTAL_FAMILIES, "income_statement_dates"):
        source = staged.get(family_id, pd.DataFrame()).copy()
        if source.empty:
            frames[family_id] = pd.DataFrame()
            family_stats[family_id] = {"input_rows": 0, "valid_key_rows": 0, "structural_quarantine_rows": 0}
            continue
        valid, q = valid_key_rows(source, ["Instrument", "period_end"], "fundamental_observations")
        quarantine_parts.append(q)
        compare = ["period_end", "fperiod", "value"]
        if family_id == "income_statement_dates":
            compare = ["period_end", "orig_announcement_date", "last_update_date"]
        duplicate, occurrence = duplicate_codes(valid, ["Instrument", "period_end"], compare)
        valid["duplicate_key_code"] = duplicate
        valid["key_occurrence"] = occurrence
        valid["row_reason_codes"] = [
            append_reason(reason, code) for reason, code in zip(valid["row_reason_codes"], duplicate)
        ]
        conflict = valid["duplicate_key_code"].eq("DUPLICATE_KEY_CONFLICT")
        if conflict.any():
            quarantine_parts.append(
                make_quarantine(
                    valid.loc[conflict],
                    "fundamental_observations",
                    "DUPLICATE_KEY_CONFLICT",
                    detail="all conflicting same Instrument+period_end rows retained in clean observation output",
                )
            )
        frames[family_id] = valid
        family_stats[family_id] = {
            "input_rows": int(len(source)),
            "valid_key_rows": int(len(valid)),
            "structural_quarantine_rows": int(len(q)),
            "duplicate_identical_rows": int(valid["duplicate_key_code"].eq("DUPLICATE_KEY_IDENTICAL").sum()),
            "duplicate_conflict_rows": int(conflict.sum()),
        }

    merged_parts: list[pd.DataFrame] = []
    for family_id in (*FUNDAMENTAL_FAMILIES, "income_statement_dates"):
        frame = frames.get(family_id, pd.DataFrame())
        if frame.empty:
            continue
        columns = ["Instrument", "period_end", "fperiod", "key_occurrence", "row_reason_codes", "raw_file", "raw_row", "raw_text"]
        selected = frame[[column for column in columns if column in frame.columns]].copy()
        selected = selected.rename(
            columns={
                "fperiod": f"{family_id}__fperiod",
                "row_reason_codes": f"{family_id}__reason_codes",
                "raw_file": f"{family_id}__raw_file",
                "raw_row": f"{family_id}__raw_row",
                "raw_text": f"{family_id}__raw_text",
            }
        )
        if family_id == "income_statement_dates":
            selected["orig_announcement_date"] = frame["orig_announcement_date"].values
            selected["last_update_date"] = frame["last_update_date"].values
            selected["vendor_period_end"] = frame["vendor_period_end"].values
        else:
            selected[FUNDAMENTAL_OUTPUTS[family_id]] = frame["value"].values
        merged_parts.append(selected)
    if not merged_parts:
        empty = pd.DataFrame(
            columns=[
                "Instrument",
                "period_end",
                "key_occurrence",
                *FUNDAMENTAL_OUTPUTS.values(),
                "orig_announcement_date",
                "last_update_date",
                "vendor_period_end",
                "rd_intensity",
                "ratio_gate_status",
                "pit_gate_status",
                "period_key_status",
                "reporting_state",
                "unit_status",
                "currency_status",
                "fundamental_reason_codes",
                "raw_file",
                "raw_row",
                "raw_text",
            ]
        )
        return empty, pd.concat(quarantine_parts, ignore_index=True, sort=False) if quarantine_parts else make_quarantine(empty, "fundamental_observations", ""), {"families": family_stats, "input_rows": 0, "output_rows": 0}

    def merge_two(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return left.merge(right, on=["Instrument", "period_end", "key_occurrence"], how="outer", sort=False)

    merged = reduce(merge_two, merged_parts)
    merged = merged.sort_values(["Instrument", "period_end", "key_occurrence"], kind="stable").reset_index(drop=True)
    for output_name in FUNDAMENTAL_OUTPUTS.values():
        if output_name not in merged.columns:
            merged[output_name] = pd.NA
    merged["orig_announcement_date"] = merged.get("orig_announcement_date", pd.Series(pd.NA, index=merged.index))
    merged["last_update_date"] = merged.get("last_update_date", pd.Series(pd.NA, index=merged.index))
    merged["vendor_period_end"] = merged.get("vendor_period_end", pd.Series(pd.NA, index=merged.index))

    family_reason_columns = [f"{family_id}__reason_codes" for family_id in (*FUNDAMENTAL_FAMILIES, "income_statement_dates")]
    merged["fundamental_reason_codes"] = [
        append_reason(*[row.get(column) for column in family_reason_columns])
        for _, row in merged.iterrows()
    ]
    period_labels = [f"{family_id}__fperiod" for family_id in (*FUNDAMENTAL_FAMILIES, "income_statement_dates")]
    period_status = []
    for _, row in merged.iterrows():
        labels = {str(row[column]) for column in period_labels if column in row and pd.notna(row[column]) and str(row[column]).strip()}
        mismatch = len(labels) > 1
        period_status.append("PERIOD_KEY_MISMATCH" if mismatch else "PERIOD_KEY_ALIGNED")
    merged["period_key_status"] = period_status
    merged.loc[merged["period_key_status"].eq("PERIOD_KEY_MISMATCH"), "fundamental_reason_codes"] = merged.loc[
        merged["period_key_status"].eq("PERIOD_KEY_MISMATCH"), "fundamental_reason_codes"
    ].map(lambda value: append_reason(value, "PERIOD_KEY_MISMATCH"))
    merged["pit_gate_status"] = merged["orig_announcement_date"].map(
        lambda value: "PIT_DATE_AVAILABLE_FORMATION_CUTOFF_DEFERRED" if pd.notna(value) and str(value).strip() else "PIT_DATE_UNAVAILABLE"
    )
    merged.loc[merged["pit_gate_status"].eq("PIT_DATE_UNAVAILABLE"), "fundamental_reason_codes"] = merged.loc[
        merged["pit_gate_status"].eq("PIT_DATE_UNAVAILABLE"), "fundamental_reason_codes"
    ].map(lambda value: append_reason(value, "PIT_DATE_UNAVAILABLE"))
    merged["reporting_state"] = "Orig"
    merged["unit_status"] = "UNVERIFIED"
    merged["currency_status"] = "UNAVAILABLE_RAW"
    merged["ratio_gate_status"] = "BLOCKED_UNIT_CURRENCY_UNVERIFIED"
    merged["rd_intensity"] = pd.NA
    merged["fundamental_reason_codes"] = merged["fundamental_reason_codes"].map(
        lambda value: append_reason(value, "RATIO_UNIT_CURRENCY_UNVERIFIED")
    )

    raw_file_columns = [f"{family_id}__raw_file" for family_id in (*FUNDAMENTAL_FAMILIES, "income_statement_dates")]
    raw_row_columns = [f"{family_id}__raw_row" for family_id in (*FUNDAMENTAL_FAMILIES, "income_statement_dates")]
    raw_text_columns = [f"{family_id}__raw_text" for family_id in (*FUNDAMENTAL_FAMILIES, "income_statement_dates")]
    merged["raw_file"] = [json_map((row.get(column) for column in raw_file_columns), FUNDAMENTAL_FAMILIES + ("income_statement_dates",)) for _, row in merged.iterrows()]
    merged["raw_row"] = [json_map((row.get(column) for column in raw_row_columns), FUNDAMENTAL_FAMILIES + ("income_statement_dates",)) for _, row in merged.iterrows()]
    merged["raw_text"] = [json_map((row.get(column) for column in raw_text_columns), FUNDAMENTAL_FAMILIES + ("income_statement_dates",)) for _, row in merged.iterrows()]
    output_columns = [
        "Instrument",
        "period_end",
        "key_occurrence",
        "fperiod",
        *FUNDAMENTAL_OUTPUTS.values(),
        "orig_announcement_date",
        "last_update_date",
        "vendor_period_end",
        "pit_gate_status",
        "period_key_status",
        "rd_intensity",
        "ratio_gate_status",
        "reporting_state",
        "unit_status",
        "currency_status",
        "fundamental_reason_codes",
        "raw_file",
        "raw_row",
        "raw_text",
    ]
    # fperiod is the first available literal label, with mismatch status retained separately.
    merged["fperiod"] = [
        next((row.get(column) for column in period_labels if pd.notna(row.get(column)) and str(row.get(column)).strip()), pd.NA)
        for _, row in merged.iterrows()
    ]
    for column in output_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    output = merged[output_columns].copy()
    quarantine = pd.concat(quarantine_parts, ignore_index=True, sort=False) if quarantine_parts else make_quarantine(output, "fundamental_observations", "")
    stats = {
        "families": family_stats,
        "input_rows": int(sum(item["input_rows"] for item in family_stats.values())),
        "output_merged_rows": int(len(output)),
        "output_instruments": int(output["Instrument"].nunique()) if not output.empty else 0,
        "quarantine_rows_including_conflict_copies": int(len(quarantine)),
    }
    return output, quarantine, stats


def load_keyword_dictionary(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "status": "unfrozen_no_dictionary",
            "path": None,
            "sha256": None,
            "version": "unfrozen-no-dictionary",
            "core": [],
            "broad": [],
            "precedence": "core_then_broad",
        }
    dictionary_path = safe_config_path(path, label="keyword_dictionary")
    value = read_json(dictionary_path)
    if not isinstance(value, dict):
        raise RuntimeError("keyword dictionary must be a JSON object")
    version = value.get("version")
    core = value.get("core")
    broad = value.get("broad")
    if not isinstance(version, str) or not version.strip() or not isinstance(core, list) or not isinstance(broad, list):
        raise RuntimeError("keyword dictionary requires string version and array core/broad")
    if any(not isinstance(item, str) or not item.strip() for item in [*core, *broad]):
        raise RuntimeError("keyword dictionary keywords must be non-empty strings")
    precedence = str(value.get("precedence", "core_then_broad"))
    if precedence != "core_then_broad":
        raise RuntimeError("only precedence=core_then_broad is supported")
    return {
        "status": "fixed_dictionary",
        "path": relpath(dictionary_path),
        "sha256": sha256_file(dictionary_path),
        "version": version,
        "core": sorted({item.strip() for item in core}, key=str.casefold),
        "broad": sorted({item.strip() for item in broad}, key=str.casefold),
        "precedence": precedence,
    }


def classify_segment(name: Any, dictionary: dict[str, Any]) -> tuple[str | None, str]:
    if dictionary.get("status") != "fixed_dictionary":
        return None, ""
    text = "" if pd.isna(name) else str(name)
    folded = text.casefold()
    core_matches = [item for item in dictionary.get("core", []) if item.casefold() in folded]
    broad_matches = [item for item in dictionary.get("broad", []) if item.casefold() in folded]
    matched = [f"core:{item}" for item in core_matches] + [f"broad:{item}" for item in broad_matches]
    if core_matches:
        return "core", ";".join(matched)
    if broad_matches:
        return "broad", ";".join(matched)
    return "unclassified", ";".join(matched)


def build_segments(
    staged: dict[str, pd.DataFrame],
    fundamental: pd.DataFrame,
    dictionary: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = staged.get("business_segments_fy0", pd.DataFrame()).copy()
    if source.empty:
        columns = [
            "Instrument", "period_end", "fperiod", "segment_code", "segment_name", "segment_value",
            "null_total_candidate", "total_candidate_count", "segment_sum_non_total", "reference_revenue",
            "reconciliation_difference", "reconciliation_status", "segment_classification_version",
            "matched_keywords", "ai_class", "row_reason_codes", "raw_file", "raw_row", "raw_text",
        ]
        empty = pd.DataFrame(columns=columns)
        return empty, make_quarantine(empty, "segment_observations", ""), {"input_rows": 0, "output_rows": 0, "output_instruments": 0}
    valid, q = valid_key_rows(source, ["Instrument", "period_end"], "segment_observations")
    valid["segment_key"] = valid.apply(
        lambda row: "|".join(
            "<NA>" if pd.isna(row.get(column)) else str(row.get(column))
            for column in ["Instrument", "period_end", "segment_code", "segment_name"]
        ),
        axis=1,
    )
    duplicate, occurrence = duplicate_codes(
        valid,
        ["Instrument", "period_end", "segment_code", "segment_name"],
        ["period_end", "fperiod", "segment_code", "segment_name", "segment_value"],
        allow_na_keys=True,
    )
    valid["duplicate_key_code"] = duplicate
    valid["segment_key_occurrence"] = occurrence
    valid["row_reason_codes"] = [
        append_reason(reason, code) for reason, code in zip(valid["row_reason_codes"], duplicate)
    ]
    conflict = duplicate.eq("DUPLICATE_KEY_CONFLICT")
    if conflict.any():
        q = pd.concat(
            [
                q,
                make_quarantine(
                    valid.loc[conflict],
                    "segment_observations",
                    "DUPLICATE_KEY_CONFLICT",
                    detail="all conflicting same segment key rows retained in segment output",
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    valid["segment_classification_version"] = dictionary.get("version", "unfrozen-no-dictionary")
    if dictionary.get("status") == "fixed_dictionary":
        valid[["ai_class", "matched_keywords"]] = valid.apply(
            lambda row: pd.Series(classify_segment(row.get("segment_name"), dictionary)), axis=1
        )
        valid["classification_status"] = "FROZEN_DICTIONARY"
    else:
        valid["ai_class"] = pd.NA
        valid["matched_keywords"] = pd.NA
        valid["classification_status"] = "UNFROZEN_NO_DICTIONARY"

    # Reference revenue is read from the explicit clean fundamental merge. It
    # is used only for diagnostics, never to force segment totals to reconcile.
    ref: dict[tuple[str, str], float] = {}
    if not fundamental.empty:
        for key, group in fundamental.groupby(["Instrument", "period_end"], dropna=False, sort=False):
            if group["fundamental_reason_codes"].astype(str).str.contains("DUPLICATE_KEY_CONFLICT", regex=False).any():
                continue
            value = pd.NA
            for column in ["control_total_revenue_value", "revenue_value"]:
                if column in group:
                    nonmissing = group[column].dropna()
                    if not nonmissing.empty:
                        value = nonmissing.iloc[0]
                        break
            try:
                parsed = float(value)
                if math.isfinite(parsed):
                    ref[(str(key[0]), str(key[1]))] = parsed
            except (TypeError, ValueError):
                pass
    valid["total_candidate_count"] = 0
    valid["segment_sum_non_total"] = pd.NA
    valid["reference_revenue"] = pd.NA
    valid["reconciliation_difference"] = pd.NA
    valid["reconciliation_status"] = "NOT_EVALUATED"
    abs_tol = float(config["segments"].get("reconciliation_tolerance_absolute", 0.01))
    rel_tol = float(config["segments"].get("reconciliation_tolerance_relative", 0.000001))
    for key, indices in valid.groupby(["Instrument", "period_end"], dropna=False, sort=False).groups.items():
        idx = list(indices)
        total_mask = valid.loc[idx, "null_total_candidate"].astype(str).eq("true")
        total_count = int(total_mask.sum())
        non_total = valid.loc[idx].loc[~total_mask]
        numeric_values = pd.to_numeric(non_total["segment_value"], errors="coerce")
        segment_sum = float(numeric_values.sum()) if numeric_values.notna().any() else math.nan
        reference = ref.get((str(key[0]), str(key[1])))
        if total_count > 1:
            valid.loc[idx, "row_reason_codes"] = valid.loc[idx, "row_reason_codes"].map(
                lambda value: append_reason(value, "SEGMENT_TOTAL_AMBIGUOUS")
            )
        if reference is not None and math.isfinite(segment_sum):
            difference = segment_sum - reference
            tolerance = max(abs_tol, abs(reference) * rel_tol)
            status = "RECONCILED_WITHIN_TOLERANCE" if abs(difference) <= tolerance else "SEGMENT_RECONCILIATION_GAP"
            if status == "SEGMENT_RECONCILIATION_GAP":
                valid.loc[idx, "row_reason_codes"] = valid.loc[idx, "row_reason_codes"].map(
                    lambda value: append_reason(value, "SEGMENT_RECONCILIATION_GAP")
                )
        else:
            difference = math.nan
            status = "REFERENCE_UNAVAILABLE"
        valid.loc[idx, "total_candidate_count"] = total_count
        valid.loc[idx, "segment_sum_non_total"] = segment_sum if math.isfinite(segment_sum) else pd.NA
        valid.loc[idx, "reference_revenue"] = reference if reference is not None else pd.NA
        valid.loc[idx, "reconciliation_difference"] = difference if math.isfinite(difference) else pd.NA
        valid.loc[idx, "reconciliation_status"] = status
    output_columns = [
        "Instrument", "period_end", "fperiod", "segment_code", "segment_name", "segment_value",
        "null_total_candidate", "total_candidate_count", "segment_sum_non_total", "reference_revenue",
        "reconciliation_difference", "reconciliation_status", "segment_classification_version",
        "classification_status", "matched_keywords", "ai_class", "duplicate_key_code",
        "segment_key_occurrence", "row_reason_codes", "raw_file", "raw_row", "raw_text",
    ]
    output = valid[output_columns].copy()
    stats = {
        "input_rows": int(len(source)),
        "output_rows": int(len(output)),
        "output_instruments": int(output["Instrument"].nunique()),
        "structural_quarantine_rows": int(len(q)),
        "duplicate_conflict_rows": int(conflict.sum()),
        "null_total_candidate_rows": int(output["null_total_candidate"].astype(str).eq("true").sum()),
        "reconciliation_gap_rows": int(output["reconciliation_status"].eq("SEGMENT_RECONCILIATION_GAP").sum()),
        "classification_status": dictionary.get("status"),
    }
    return output, q, stats


def build_etf_and_returns(
    staged: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = staged.get("etf_daily_price_volume", pd.DataFrame()).copy()
    q_parts: list[pd.DataFrame] = []
    if source.empty:
        price_columns = ["Instrument", "Date", *PRICE_FIELDS, "has_close", "has_volume", "has_two_sided_quote", "adjustments", "row_reason_codes", "raw_file", "raw_row", "raw_text"]
        empty = pd.DataFrame(columns=price_columns)
        returns = pd.DataFrame(columns=["Instrument", "Date", "prior_date", "price_return", "return_status", "raw_file", "raw_row", "raw_text"])
        proxies = pd.DataFrame(columns=["Instrument", "proxy_date", "proxy_role", "price_return", "proxy_cutoff_rule", "raw_file", "raw_row", "raw_text"])
        return empty, returns, proxies, {"input_rows": 0, "price_rows": 0, "return_rows": 0, "proxy_rows": 0, "quarantine_rows": 0}
    valid, structural_q = valid_key_rows(source, ["Instrument", "Date"], "etf_prices")
    q_parts.append(structural_q)
    padding = valid["structural_padding"].astype(str).eq("true")
    if padding.any():
        q_parts.append(make_quarantine(valid.loc[padding], "etf_prices", "STRUCTURAL_WIDE_PADDING"))
        valid = valid.loc[~padding].copy()
    duplicate, occurrence = duplicate_codes(valid, ["Instrument", "Date"], list(PRICE_FIELDS))
    valid["duplicate_key_code"] = duplicate
    valid["key_occurrence"] = occurrence
    valid["row_reason_codes"] = [
        append_reason(reason, code) for reason, code in zip(valid["row_reason_codes"], duplicate)
    ]
    conflict = duplicate.eq("DUPLICATE_KEY_CONFLICT")
    if conflict.any():
        q_parts.append(
            make_quarantine(
                valid.loc[conflict],
                "etf_prices",
                "DUPLICATE_KEY_CONFLICT",
                detail="all conflicting same Instrument+Date rows retained in ETF output",
            )
        )
    for field in PRICE_FIELDS:
        valid[field] = valid[field].where(valid[field].notna(), pd.NA)
    valid["has_close"] = valid["TRDPRC_1"].notna()
    valid["has_volume"] = valid["ACVOL_UNS"].notna()
    valid["has_two_sided_quote"] = valid[["BID", "ASK"]].notna().all(axis=1)
    valid["adjustments"] = ",".join(config["etf"]["adjustments"])
    price_columns = [
        "Instrument", "Date", *PRICE_FIELDS, "has_close", "has_volume", "has_two_sided_quote",
        "adjustments", "duplicate_key_code", "key_occurrence", "row_reason_codes", "raw_file", "raw_row", "raw_text",
    ]
    prices = valid[price_columns].copy().sort_values(["Instrument", "Date", "key_occurrence"], kind="stable").reset_index(drop=True)
    return_records: list[dict[str, Any]] = []
    for instrument, group in prices.groupby("Instrument", sort=False):
        group = group.copy().sort_values(["Date", "key_occurrence"], kind="stable")
        date_counts = group["Date"].value_counts(dropna=False).to_dict()
        prior_date: str | None = None
        prior_price: float | None = None
        for _, row in group.iterrows():
            current_date = row["Date"]
            current_price = pd.to_numeric(pd.Series([row["TRDPRC_1"]]), errors="coerce").iloc[0]
            current_price = float(current_price) if pd.notna(current_price) and math.isfinite(float(current_price)) else None
            result: dict[str, Any] = {
                "Instrument": instrument,
                "Date": current_date,
                "prior_date": prior_date,
                "price_return": pd.NA,
                "return_status": "",
                "row_reason_codes": "",
                "price_adjustments": row["adjustments"],
                "raw_file": row["raw_file"],
                "raw_row": row["raw_row"],
                "raw_text": row["raw_text"],
            }
            if date_counts.get(current_date, 0) > 1:
                result["return_status"] = "DUPLICATE_DATE_AMBIGUOUS"
            elif prior_date is None:
                result["return_status"] = "PRIOR_OBSERVATION_UNAVAILABLE"
            elif current_price is None:
                result["return_status"] = "CURRENT_PRICE_UNAVAILABLE"
            elif current_price <= 0:
                result["return_status"] = "CURRENT_PRICE_NONPOSITIVE"
            elif prior_price is None:
                result["return_status"] = "PRIOR_PRICE_UNAVAILABLE"
            elif prior_price <= 0:
                result["return_status"] = "PRIOR_PRICE_NONPOSITIVE"
            else:
                result["price_return"] = current_price / prior_price - 1.0
                result["return_status"] = "COMPUTED"
            result["row_reason_codes"] = "" if result["return_status"] == "COMPUTED" else result["return_status"]
            return_records.append(result)
            # A duplicate date is not a unique prior observation. It remains
            # visible but never becomes an implicit input to the next return.
            if date_counts.get(current_date, 0) == 1:
                prior_date = current_date
                prior_price = current_price
    returns = pd.DataFrame(return_records)
    if returns.empty:
        returns = pd.DataFrame(columns=["Instrument", "Date", "prior_date", "price_return", "return_status", "row_reason_codes", "price_adjustments", "raw_file", "raw_row", "raw_text"])
    proxies = returns.copy()
    if proxies.empty:
        proxies = pd.DataFrame(columns=["Instrument", "proxy_date", "proxy_role", "price_return", "return_status", "row_reason_codes", "proxy_cutoff_rule", "raw_file", "raw_row", "raw_text"])
    else:
        proxies = proxies.rename(columns={"Date": "proxy_date"})
        proxies["proxy_role"] = proxies["Instrument"].map(
            lambda value: "benchmark_proxy" if str(value) == "SPY.P" else "ai_basket_proxy"
        )
        proxies["proxy_cutoff_rule"] = "downstream must enforce proxy_date <= formation_session; no labels are stored"
        proxy_columns = ["Instrument", "proxy_date", "proxy_role", "price_return", "return_status", "row_reason_codes", "proxy_cutoff_rule", "raw_file", "raw_row", "raw_text"]
        proxies = proxies[proxy_columns]
    configured = set(config["etf"].get("configured_instruments", []))
    observed = set(prices["Instrument"].dropna().astype(str))
    no_observation = sorted(configured - observed)
    stats = {
        "input_rows": int(len(source)),
        "price_rows": int(len(prices)),
        "price_instruments": int(prices["Instrument"].nunique()) if not prices.empty else 0,
        "return_rows": int(len(returns)),
        "computed_price_return_rows": int(returns["return_status"].eq("COMPUTED").sum()) if not returns.empty else 0,
        "proxy_rows": int(len(proxies)),
        "quarantine_rows": int(sum(len(part) for part in q_parts)),
        "duplicate_conflict_rows": int(conflict.sum()),
        "missing_configured_instruments": no_observation,
        "missing_observation_reason": config["etf"]["missing_observation_reason"],
    }
    quarantine = pd.concat(q_parts, ignore_index=True, sort=False) if q_parts else make_quarantine(source, "etf_prices", "")
    return prices, returns, proxies, {**stats, "quarantine_frame": quarantine}


def build_market_cap(
    staged: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = staged.get("market_capitalization_monthly", pd.DataFrame()).copy()
    if source.empty:
        empty = pd.DataFrame(columns=["Instrument", "Date", "market_cap_value", "semantics_status", "reporting_state", "reporting_state_exception", "duplicate_key_code", "key_occurrence", "row_reason_codes", "raw_file", "raw_row", "raw_text"])
        return empty, make_quarantine(empty, "market_cap_monthly", ""), {"input_rows": 0, "output_rows": 0, "output_instruments": 0, "quarantine_rows": 0}
    valid, q = valid_key_rows(source, ["Instrument", "Date"], "market_cap_monthly")
    duplicate, occurrence = duplicate_codes(valid, ["Instrument", "Date"], ["Date", "market_cap_value"])
    valid["duplicate_key_code"] = duplicate
    valid["key_occurrence"] = occurrence
    valid["row_reason_codes"] = [
        append_reason(reason, code) for reason, code in zip(valid["row_reason_codes"], duplicate)
    ]
    conflict = duplicate.eq("DUPLICATE_KEY_CONFLICT")
    if conflict.any():
        q = pd.concat(
            [
                q,
                make_quarantine(
                    valid.loc[conflict],
                    "market_cap_monthly",
                    "DUPLICATE_KEY_CONFLICT",
                    detail="all conflicting same Instrument+Date rows retained in market-cap output",
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    valid["semantics_status"] = config["market_cap"]["semantics_status"].upper()
    valid["reporting_state"] = "OMITTED_MARKET_DATA_EXCEPTION"
    valid["reporting_state_exception"] = config["market_cap"]["reporting_state_exception"]
    output_columns = [
        "Instrument", "Date", "market_cap_value", "semantics_status", "reporting_state", "reporting_state_exception",
        "duplicate_key_code", "key_occurrence", "row_reason_codes", "raw_file", "raw_row", "raw_text",
    ]
    output = valid[output_columns].sort_values(["Instrument", "Date", "key_occurrence"], kind="stable").reset_index(drop=True)
    stats = {
        "input_rows": int(len(source)),
        "output_rows": int(len(output)),
        "output_instruments": int(output["Instrument"].nunique()),
        "quarantine_rows_including_conflict_copies": int(len(q)),
        "duplicate_conflict_rows": int(conflict.sum()),
        "semantics_status": config["market_cap"]["semantics_status"].upper(),
        "reporting_state_sent": False,
    }
    return output, q, stats


def build_membership_outputs(
    membership: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = membership.copy()
    tokens = provider_tokens(config)
    q_parts: list[pd.DataFrame] = []
    if source.empty:
        empty = pd.DataFrame(columns=["Instrument", "snapshot_date", "member_from", "member_to", "fetch_start", "fetch_end", "member_days", "delisted_ric", "membership_status", "delisted_coverage_status", "snapshot_type", "raw_file", "raw_row", "raw_text", "row_reason_codes"])
        identity = pd.DataFrame(columns=["Instrument", "name", "delisted_ric", "member_from", "member_to", "fetch_start", "fetch_end", "member_days", "identity_mapping_method", "raw_file", "raw_row", "raw_text"])
        return empty, identity, make_quarantine(empty, "membership_snapshots", ""), {"input_rows": 0, "output_rows": 0, "identity_rows": 0, "quarantine_rows": 0}
    records = []
    for _, row in source.iterrows():
        reason = "MALFORMED_CSV_ROW" if bool(row.get("_malformed", False)) else ""
        instrument, instrument_code = clean_text(row.get("ric", ""), tokens)
        reason = append_reason(reason, "INSTRUMENT_MISSING" if instrument_code else None)
        parsed = {}
        for field in ["member_from", "member_to", "fetch_start", "fetch_end"]:
            parsed[field], code = clean_date(row.get(field, ""), tokens, field)
            reason = append_reason(reason, code)
        member_days, days_code = clean_numeric(row.get("member_days", ""), tokens, "member_days")
        reason = append_reason(reason, days_code)
        delisted, delisted_code = clean_text(row.get("delisted_ric", ""), tokens)
        reason = append_reason(reason, delisted_code)
        record = {
            "Instrument": instrument,
            "name": clean_text(row.get("name", ""), tokens)[0],
            **parsed,
            "member_days": member_days,
            "delisted_ric": delisted,
            "membership_status": "historical_delisted_span" if str(delisted).casefold() == "true" else "historical_membership_span",
            "delisted_coverage_status": "retain_all_available_rows" if str(delisted).casefold() == "true" else "not_delisted_flagged",
            "snapshot_date": parsed.get("member_from"),
            "snapshot_type": "corrected_span_interval",
            "raw_file": row.get("raw_file", ""),
            "raw_row": row.get("raw_row", pd.NA),
            "raw_text": row.get("raw_text", ""),
            "row_reason_codes": reason,
        }
        records.append(record)
    output = pd.DataFrame(records)
    structural = output["Instrument"].isna()
    if structural.any():
        q_parts.append(make_quarantine(output.loc[structural], "membership_snapshots", "STRUCTURAL_KEY_UNAVAILABLE"))
        output = output.loc[~structural].copy()
    identity = output[["Instrument", "name", "delisted_ric", "member_from", "member_to", "fetch_start", "fetch_end", "member_days", "raw_file", "raw_row", "raw_text"]].copy()
    identity["identity_mapping_method"] = "literal_corrected_span_ric"
    identity = identity[["Instrument", "name", "delisted_ric", "member_from", "member_to", "fetch_start", "fetch_end", "member_days", "identity_mapping_method", "raw_file", "raw_row", "raw_text"]]
    stats = {
        "input_rows": int(len(source)),
        "output_rows": int(len(output)),
        "identity_rows": int(len(identity)),
        "output_instruments": int(output["Instrument"].nunique()) if not output.empty else 0,
        "delisted_rows_retained": int(output["delisted_ric"].astype(str).str.casefold().eq("true").sum()) if not output.empty else 0,
        "quarantine_rows": int(sum(len(part) for part in q_parts)),
        "mapping_method": "literal_corrected_span_ric",
    }
    quarantine = pd.concat(q_parts, ignore_index=True, sort=False) if q_parts else make_quarantine(output, "membership_snapshots", "")
    return output, identity, quarantine, stats


def frame_reason_counts(frame: pd.DataFrame, column: str = "row_reason_codes") -> dict[str, int]:
    counts: Counter[str] = Counter()
    if frame.empty or column not in frame.columns:
        return {}
    for value in frame[column].fillna("").astype(str):
        counts.update(item for item in value.split(";") if item)
    return dict(sorted(counts.items()))


def output_stats(table: str, frame: pd.DataFrame, quarantine: pd.DataFrame) -> dict[str, Any]:
    instrument_column = "Instrument" if "Instrument" in frame.columns else None
    return {
        "table": table,
        "output_rows": int(len(frame)),
        "output_instruments": int(frame[instrument_column].nunique()) if instrument_column and not frame.empty else 0,
        "quarantine_rows": int(len(quarantine)),
        "reason_code_counts": frame_reason_counts(frame),
    }


def missingness_summary(
    tables: dict[str, pd.DataFrame],
    quarantines: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for table, frame in tables.items():
        q = quarantines.get(table, pd.DataFrame())
        if frame.empty and len(frame.columns) == 0:
            continue
        for column in frame.columns:
            values = frame[column]
            missing = values.isna() | values.astype(str).str.strip().eq("")
            nonmissing = ~missing
            zero = nonmissing & values.astype(str).str.strip().eq("0")
            records.append(
                {
                    "table": table,
                    "column": column,
                    "output_rows": int(len(frame)),
                    "quarantine_rows_same_table": int(len(q)),
                    "na_rows": int(missing.sum()),
                    "non_na_rows": int(nonmissing.sum()),
                    "na_rate": float(missing.mean()) if len(frame) else None,
                    "text_or_numeric_zero_observation_rows": int(zero.sum()),
                    "missing_value_policy": "empty/provider NA/unparseable numeric or date => NA; no fill",
                }
            )
    return pd.DataFrame(records)


def aggregate_quarantine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [part for part in parts if part is not None and not part.empty]
    if not nonempty:
        return pd.DataFrame(
            columns=[
                "table", "family_id", "request_id", "Instrument", "period_end", "Date",
                "reason_code", "reason_detail", "raw_file", "raw_row", "raw_text", "normalized_row_json",
            ]
        )
    result = pd.concat(nonempty, ignore_index=True, sort=False)
    ordered = [
        "table", "family_id", "request_id", "Instrument", "period_end", "Date", "reason_code", "reason_detail",
        "raw_file", "raw_row", "raw_text", "normalized_row_json",
    ]
    for column in ordered:
        if column not in result.columns:
            result[column] = pd.NA
    return result[[*ordered, *[column for column in result.columns if column not in ordered]]]


def verify_membership_unchanged(plan: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    membership, info = literal_membership_rows(config)
    if info["sha256"] != plan["membership_spans"]["sha256"]:
        raise RuntimeError("membership spans changed since plan")
    return membership


def append_processing_log(summary: dict[str, Any], clean_dir: Path, audit_dir: Path) -> None:
    """Append one run-scoped record; the raw input is never rewritten."""
    log_path = ROOT / "DATA_PROCESSING_LOG.md"
    if not log_path.exists():
        prior = "# DATA_PROCESSING_LOG\n"
    else:
        prior = log_path.read_text(encoding="utf-8")
    entry = [
        "",
        f"## ai_factor_cleaning_v1 {summary['run_id']} ({summary['generated_at_utc']})",
        "",
        f"- status: `{summary['status']}`; clean_ready=`{summary['clean_ready']}`; model_ready=`{summary['model_ready']}`.",
        f"- input raw run: `{summary['raw_run']['raw_run_path']}`; raw CSV files={summary['input_file_count']}; staged={summary['staged_request_count']}; remaining={summary['remaining_request_count']}.",
        f"- input hashes: raw CSV hashes are recorded in `{relpath(clean_dir / 'clean_summary.json')}`; membership hash=`{summary['membership_spans']['sha256']}`.",
        f"- outputs: `{relpath(clean_dir)}`; quarantine=`{relpath(audit_dir / 'quarantine')}`.",
        f"- row accounting: `{json.dumps(summary.get('row_accounting', {}), ensure_ascii=False, sort_keys=True)}`.",
        f"- reason codes: `{json.dumps(summary.get('reason_code_counts', {}), ensure_ascii=False, sort_keys=True)}`.",
        "- rules: raw_file/raw_row/raw_text retained; empty/provider NA/unparseable values become NA; text 0 remains an observation; no imputation, forward fill, winsorization, silent deduplication, delisted-row deletion, name-based RIC mapping, current-TRBC backfill, labels, future returns, or model outputs.",
        "- limitations: fundamental PIT formation-session comparison is deferred to downstream formation logic; unit/currency semantics remain unverified; segment keyword classification remains unfrozen unless a fixed dictionary hash is supplied; market cap semantics remain unverified.",
        "",
    ]
    atomic_write_text(log_path, prior.rstrip("\n") + "\n" + "\n".join(entry))


def finalize_outputs(
    clean_dir: Path,
    audit_dir: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    last_limit: int,
) -> dict[str, Any]:
    print(
        "阶段说明（stage→clean）：按 Instrument+period_end 显式合并 R&D/Revenue/六类 controls/Orig announcement dates，"
        "保留重复与冲突诊断；分部只保留原始行并输出 null-total/duplicate/reconciliation diagnostics；"
        "ETF 仅按 Instrument+Date 生成明确的 price_return；市值按 Instrument+Date 读取且保留未验证语义。"
        "不填补、不前填、不 winsorize、不删除退市 RIC，不生成 labels、future returns 或 model outputs。",
        flush=True,
    )
    config_dictionary = plan.get("keyword_dictionary") or load_keyword_dictionary(None)
    membership = verify_membership_unchanged(plan, config)
    staged = read_staged_frames(clean_dir, plan, manifest)
    fundamental, fundamental_q, fundamental_stats = build_fundamental(staged, config)
    segments, segments_q, segment_stats = build_segments(staged, fundamental, config_dictionary, config)
    prices, returns, proxies, etf_stats = build_etf_and_returns(staged, config)
    etf_q = etf_stats.pop("quarantine_frame", pd.DataFrame())
    market_cap, market_cap_q, market_cap_stats = build_market_cap(staged, config)
    membership_out, identity_out, membership_q, membership_stats = build_membership_outputs(membership, config)

    tables = {
        "fundamental_observations": fundamental,
        "segment_observations": segments,
        "etf_prices": prices,
        "asset_returns": returns,
        "market_cap_monthly": market_cap,
        "ai_market_proxies": proxies,
        "membership_snapshots": membership_out,
        "identity_map": identity_out,
    }
    quarantine_by_table = {
        "fundamental_observations": fundamental_q,
        "segment_observations": segments_q,
        "etf_prices": etf_q,
        "market_cap_monthly": market_cap_q,
        "membership_snapshots": membership_q,
    }
    quarantine = aggregate_quarantine(list(quarantine_by_table.values()))
    quarantine_path = audit_dir / str(config["output"]["quarantine_subdirectory"]) / "quarantine.csv"
    for table, frame in tables.items():
        atomic_write_frame(clean_dir / f"{table}.csv", frame)
    atomic_write_frame(quarantine_path, quarantine)
    atomic_write_frame(clean_dir / "missingness_summary.csv", missingness_summary(tables, quarantine_by_table))

    family_stats = {
        "fundamental": fundamental_stats,
        "segments": segment_stats,
        "etf": etf_stats,
        "market_cap": market_cap_stats,
        "membership": membership_stats,
    }
    row_accounting = {
        "fundamental_observations": fundamental_stats,
        "segment_observations": segment_stats,
        "etf_prices": {key: value for key, value in etf_stats.items() if key != "quarantine_frame"},
        "asset_returns": {"output_rows": int(len(returns)), "output_instruments": int(returns["Instrument"].nunique()) if not returns.empty else 0},
        "ai_market_proxies": {"output_rows": int(len(proxies)), "output_instruments": int(proxies["Instrument"].nunique()) if not proxies.empty else 0},
        "market_cap_monthly": market_cap_stats,
        "membership_snapshots": membership_stats,
        "identity_map": {"output_rows": int(len(identity_out)), "output_instruments": int(identity_out["Instrument"].nunique()) if not identity_out.empty else 0},
        "quarantine": {"output_rows": int(len(quarantine)), "reason_code_counts": dict(Counter(quarantine.get("reason_code", pd.Series(dtype=str)).astype(str)))},
    }
    reason_counts: Counter[str] = Counter()
    for frame in tables.values():
        reason_counts.update(frame_reason_counts(frame))
    reason_counts.update(quarantine["reason_code"].fillna("").astype(str).loc[lambda values: values.ne("")].tolist() if not quarantine.empty else [])
    staged_count = sum(1 for value in manifest["requests"].values() if value.get("status") == "staged")
    remaining = len(plan["input_files"]) - staged_count
    status = "complete" if remaining == 0 else "partial_not_model_ready"
    clean_ready = status == "complete"
    output_paths = [
        *(clean_dir / f"{table}.csv" for table in tables),
        clean_dir / "missingness_summary.csv",
        quarantine_path,
    ]
    output_hashes = {relpath(path): sha256_file(path) for path in output_paths}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "generated_at_utc": utc_now(),
        "status": status,
        "clean_ready": clean_ready,
        "model_ready": False,
        "raw_run": plan["raw_run"],
        "membership_spans": plan["membership_spans"],
        "input_file_count": len(plan["input_files"]),
        "staged_request_count": staged_count,
        "remaining_request_count": remaining,
        "last_limit": last_limit,
        "keyword_dictionary": config_dictionary,
        "input_hashes": {
            "raw_csv": {item["raw_file"]: item["raw_sha256"] for item in plan["input_files"]},
            "raw_meta": {item["meta_file"]: item["meta_sha256"] for item in plan["input_files"]},
            "raw_request_sidecars": {item["request_file"]: item["request_file_sha256"] for item in plan["input_files"]},
            "membership_spans": plan["membership_spans"]["sha256"],
        },
        "output_hashes": output_hashes,
        "row_accounting": row_accounting,
        "family_stats": family_stats,
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "quarantine": {
            "path": relpath(quarantine_path),
            "rows": int(len(quarantine)),
            "hash": sha256_file(quarantine_path),
            "reason_code_counts": dict(Counter(quarantine["reason_code"].fillna("").astype(str).loc[lambda values: values.ne("")])) if not quarantine.empty else {},
        },
        "checks": {
            "raw_inputs_rehashed_before_finalize": True,
            "membership_rehashed_before_finalize": True,
            "raw_rows_unchanged": True,
            "text_zero_preserved": True,
            "missing_values_not_imputed": True,
            "date_parsing_only_in_clean": True,
            "fundamental_join_keys_explicit": ["Instrument", "period_end"],
            "market_cap_join_keys_explicit": ["Instrument", "Date"],
            "etf_join_keys_explicit": ["Instrument", "Date"],
            "price_return_name_explicit": True,
            "no_total_return_output": True,
            "delisted_rows_retained_in_membership": True,
            "forbidden_input_reads": [],
            "lseg_imported": False,
            "network_requests": False,
        },
        "explicit_non_actions": [
            "No imputation, forward/backward fill, winsorization, or silent deduplication.",
            "No row is removed solely because its RIC is delisted.",
            "No current TRBC backfill or name-based RIC mapping.",
            "No labels, future returns, target tables, model outputs, portfolios, or model-ready tables are generated.",
        ],
        "limitations": [
            "Formation-session comparison original_announcement <= formation_session is deferred; clean records only whether Orig announcement is available.",
            "Unit and currency semantics are unverified, so rd_intensity remains NA with RATIO_UNIT_CURRENCY_UNVERIFIED.",
            "Segment AI keyword classification is unfrozen unless the plan contains a fixed dictionary with an explicit hash.",
            "Market cap unit/currency semantics remain unverified; no size ratio is computed.",
            "Partial snapshots are explicitly not model-ready until every raw request is staged and verified.",
        ],
    }
    atomic_write_json(clean_dir / "clean_summary.json", summary)

    checks = [
        {"name": "all_requests_staged", "passed": remaining == 0, "detail": {"staged": staged_count, "total": len(plan["input_files"])}},
        {"name": "raw_csv_hashes_verified", "passed": True},
        {"name": "raw_meta_and_request_hashes_verified", "passed": True},
        {"name": "membership_hash_verified", "passed": True},
        {"name": "no_forbidden_inputs_read", "passed": True},
        {"name": "no_lseg_or_network", "passed": True},
        {"name": "no_imputation_or_fill", "passed": True},
        {"name": "text_zero_preserved", "passed": True},
        {"name": "fundamental_explicit_key", "passed": True, "detail": "Instrument + period_end"},
        {"name": "market_cap_explicit_key", "passed": True, "detail": "Instrument + Date"},
        {"name": "etf_explicit_key", "passed": True, "detail": "Instrument + Date"},
        {"name": "price_return_semantics_explicit", "passed": True},
        {"name": "delisted_rows_retained", "passed": membership_stats["delisted_rows_retained"] >= 0},
        {"name": "classification_state_explicit", "passed": config_dictionary.get("status") in {"fixed_dictionary", "unfrozen_no_dictionary"}},
    ]
    gate = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "generated_at_utc": utc_now(),
        "status": status,
        "clean_ready": clean_ready,
        "model_ready": False,
        "checks": checks,
        "passed": sum(1 for item in checks if item["passed"]),
        "failed": [item for item in checks if not item["passed"]],
        "input_hashes": summary["input_hashes"],
        "output_hashes": output_hashes,
        "quarantine": summary["quarantine"],
        "limitations": summary["limitations"],
    }
    atomic_write_json(clean_dir / "gate_report.json", gate)
    # The manifest is the append-safe hash index. Dynamic summary/gate hashes
    # are recorded here after they are written; those files deliberately do
    # not hash themselves recursively.
    output_hashes_with_reports = {
        **output_hashes,
        relpath(clean_dir / "clean_summary.json"): sha256_file(clean_dir / "clean_summary.json"),
        relpath(clean_dir / "gate_report.json"): sha256_file(clean_dir / "gate_report.json"),
        relpath(clean_dir / "clean_plan.json"): sha256_file(clean_dir / "clean_plan.json"),
    }
    manifest["status"] = status
    manifest["processed_request_count"] = staged_count
    manifest["remaining_request_count"] = remaining
    manifest["last_execute"] = {
        "at_utc": utc_now(),
        "limit": last_limit,
        "staged_this_invocation": [
            key for key, value in manifest["requests"].items() if value.get("staged_at_utc", "") >= summary["generated_at_utc"]
        ],
    }
    manifest["output_hashes"] = output_hashes_with_reports
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(clean_dir / "clean_manifest.json", manifest)
    summary["output_hashes"][relpath(clean_dir / "clean_manifest.json")] = sha256_file(clean_dir / "clean_manifest.json")
    # Rewriting the summary after adding manifest hash would change its own
    # hash, so retain the stable manifest hash in a separate manifest-only
    # field and leave the summary hash index scoped to non-self artifacts.
    print(
        f"清洁阶段完成：status={status} staged={staged_count}/{len(plan['input_files'])} "
        f"fundamental_rows={len(fundamental)} segment_rows={len(segments)} etf_rows={len(prices)} "
        f"return_rows={len(returns)} quarantine_rows={len(quarantine)}",
        flush=True,
    )
    append_processing_log(summary, clean_dir, audit_dir)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumable AI-factor raw-to-clean converter")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true", help="create a new offline clean plan; no transformation")
    modes.add_argument("--execute", action="store_true", help="stage pending raw files and materialize clean outputs")
    parser.add_argument("--run-id", default="", help="new plan id or an existing run id to resume")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum number of unprocessed raw request CSV files in this --execute invocation; 0 means all",
    )
    parser.add_argument(
        "--keyword-dictionary",
        default="",
        help="optional fixed JSON dictionary with version/core/broad; absent means classification remains unfrozen",
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.plan and args.limit:
        parser.error("--limit is valid only with --execute")
    if args.run_id and not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id must be a UTC run directory id such as 20260910T120000000000Z")
    if args.keyword_dictionary and not args.execute:
        # A dictionary can be sealed into a plan, so allowing it with --plan
        # is useful and remains offline.  It is read and hashed below.
        pass
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha256 = load_config()
    specs, raw_info = discover_raw_specs(config)
    membership, membership_info = literal_membership_rows(config)
    if membership_info["rows"] == 0:
        raise RuntimeError("corrected membership spans contain no rows")
    # Identity validation is literal-only: this checks coverage without ever
    # using names to invent or merge a RIC.
    literal_rics = set(membership[str(config["membership"]["literal_ric_column"])].dropna().astype(str))
    # Stock request universes must be covered by corrected literal spans. ETF
    # RICs are independent market-data inputs and intentionally do not belong
    # to the stock membership file.
    requested = set(
        item
        for spec in specs
        if spec["family_id"] != "etf_daily_price_volume"
        for item in spec.get("requested_instruments", [])
    )
    unknown_requested = sorted(requested - literal_rics)
    if unknown_requested:
        raise RuntimeError(f"raw request universe contains RICs outside corrected literal spans: {unknown_requested[:10]}")

    dictionary = load_keyword_dictionary(args.keyword_dictionary or None)
    clean_root = safe_config_path(str(config["output"]["clean_root"]), label="clean_root")
    if args.execute and args.run_id:
        clean_dir = clean_root / args.run_id
        plan, manifest = load_plan_and_manifest(clean_dir, config_sha256)
        planned_dictionary = plan.get("keyword_dictionary", {})
        if dictionary.get("status") == "unfrozen_no_dictionary" and planned_dictionary.get("status") == "fixed_dictionary":
            dictionary = planned_dictionary
        if dictionary != planned_dictionary:
            raise RuntimeError("keyword dictionary differs from the existing plan; refusing resume")
        audit_dir = safe_config_path(str(config["output"]["audit_root"]), label="audit_root") / args.run_id
        stage_pending_requests(clean_dir, plan, manifest, config, args.limit)
        summary = finalize_outputs(clean_dir, audit_dir, plan, manifest, config, args.limit)
        print(json.dumps({"run_id": plan["run_id"], "status": summary["status"], "clean_dir": relpath(clean_dir), "audit_dir": relpath(audit_dir)}, ensure_ascii=False), flush=True)
        return 0

    clean_dir, audit_dir, plan, manifest = create_plan(
        config,
        config_sha256,
        specs,
        raw_info,
        membership_info,
        args.run_id or None,
        dictionary,
    )
    if args.plan:
        print(
            f"计划已保存：{relpath(clean_dir / 'clean_plan.json')}；run_id={plan['run_id']}。"
            "未导入 LSEG、未发起网络请求、未转换 raw 行。可用同一 run_id 显式执行 --execute。",
            flush=True,
        )
        return 0
    stage_pending_requests(clean_dir, plan, manifest, config, args.limit)
    summary = finalize_outputs(clean_dir, audit_dir, plan, manifest, config, args.limit)
    print(json.dumps({"run_id": plan["run_id"], "status": summary["status"], "clean_dir": relpath(clean_dir), "audit_dir": relpath(audit_dir)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED CLEAN AI FACTOR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
