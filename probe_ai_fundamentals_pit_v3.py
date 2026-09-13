"""Read-only LSEG point-in-time fundamentals probe.

This script tests the mechanical availability and response shape of a small set of
fundamental fields.  It is intentionally an observation probe, not a data-building
pipeline: no values are cleaned, filled, imputed, winsorized, deduplicated, filtered,
or converted into returns, labels, features, portfolios, or model inputs.

The two modes are explicit::

    python probe_ai_fundamentals_pit_v3.py --plan
    python probe_ai_fundamentals_pit_v3.py --execute --run-id <UTC-run-id>

``--plan`` writes only a versioned ``plan.json`` and never imports or calls the LSEG
client.  ``--execute`` is the only mode that can open the local ``desktop.workspace``
session and issue requests.  Requests are independent, each request has a 90-second
watchdog, and a run can be resumed using the same run id.  Completed request/CSV
artifacts are verified by request and CSV SHA-256 hashes before they are skipped.

The raw vendor frame is written as returned.  Date parsing and duplicate checks are
diagnostics on the in-memory response only; they never alter the raw CSV.  Error
sidecars are retained and redacted.  This probe does not read test targets and makes
no semantic or point-in-time conclusion about any field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw" / "ai_fundamentals_pit_probe_v3"
SESSION_NAME = "desktop.workspace"
REQUEST_TIMEOUT_SECONDS = 90
UNIVERSE = ["NVDA.OQ", "MSFT.OQ", "AMD.OQ", "IBM.N", "WMT.N", "JNJ.N"]
ANNUAL_PARAMETERS = {
    "SDate": "2014-01-01",
    "EDate": "2022-12-31",
    "Frq": "FY",
    "ReportingState": "Orig",
}
MONTHLY_PARAMETERS = {"SDate": "2014-01-01", "EDate": "2022-12-31", "Frq": "M"}
SCHEMA_VERSION = "ai_fundamentals_pit_probe_v3"
TERMINAL_STATUSES = {"returned", "empty", "error", "timeout"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


# Credentials are read only from the repository-local .env.  Request objects do not
# contain credentials; this list is used solely to redact process output and errors.
CREDS = dotenv_values(ROOT / ".env")
SECRETS = sorted(
    {str(value) for value in CREDS.values() if value not in (None, "")},
    key=len,
    reverse=True,
)


def redact(value: object) -> str:
    result = str(value)
    for secret in SECRETS:
        result = result.replace(secret, "[REDACTED]")
    return result


class SafeStream:
    """Redact local credential values from normal output and third-party warnings."""

    def __init__(self, target: Any):
        self.target = target

    def write(self, message: str) -> int:
        return self.target.write(redact(message))

    def flush(self) -> None:
        self.target.flush()

    def isatty(self) -> bool:
        return False


sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    """Stable request identity bytes; request objects contain no credentials."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def request_sha256(request: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(request))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Install a sidecar atomically in the same directory as its final path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def atomic_write_request(path: Path, request: dict[str, Any]) -> str:
    payload = canonical_json(request)
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def atomic_write_frame(path: Path, frame: Any) -> str:
    """Serialize the vendor frame unchanged, then atomically install the CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        # Calling to_csv serializes the returned frame; no replace/fill/parse/filter
        # operation is applied to the frame before serialization.
        frame.to_csv(temporary, index=False)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def annual_spec(
    request_id: str,
    family: str,
    fields: list[str],
    question: str,
    scope_notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = {
        "universe": list(UNIVERSE),
        "fields": list(fields),
        "parameters": dict(ANNUAL_PARAMETERS),
    }
    spec: dict[str, Any] = {
        "request_id": request_id,
        "family": family,
        "question": question,
        "frequency": "FY",
        "request": request,
        "request_sha256": request_sha256(request),
        "independent_request": True,
    }
    if scope_notes:
        spec["scope_notes"] = scope_notes
    return spec


def monthly_spec(request_id: str, family: str, fields: list[str], question: str) -> dict[str, Any]:
    request = {
        "universe": list(UNIVERSE),
        "fields": list(fields),
        "parameters": dict(MONTHLY_PARAMETERS),
    }
    return {
        "request_id": request_id,
        "family": family,
        "question": question,
        "frequency": "M",
        "request": request,
        "request_sha256": request_sha256(request),
        "independent_request": True,
    }


def request_specs() -> list[dict[str, Any]]:
    """Build independent requests so one unsupported field does not suppress others."""

    segment_base = "TR.F.BUSTotRevBizActiv(Period=FY0)"
    segment_scope_notes = {
        "field_level_period": "FY0",
        "request_level_frequency": "FY",
        "request_level_reporting_state": "Orig",
        "scope_interaction": (
            "Field-level Period=FY0 is retained alongside request-level Frq=FY for observation; "
            "any interaction is recorded rather than resolved."
        ),
        "independence_rule": "Keep this historical business-segment request separate from every other request family.",
    }
    specs = [
        annual_spec(
            "legacy_rd_reporting_orig",
            "legacy_rd",
            [
                "TR.ResearchAndDevelopment",
                "TR.ResearchAndDevelopment.date",
                "TR.ResearchAndDevelopment.fperiod",
            ],
            "Observe response structure for legacy R&D value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "legacy_revenue_reporting_orig",
            "legacy_revenue",
            ["TR.Revenue", "TR.Revenue.date", "TR.Revenue.fperiod"],
            "Observe response structure for legacy revenue value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "f_tot_revenue_reporting_orig",
            "f_tot_revenue",
            ["TR.F.TotRevenue", "TR.F.TotRevenue.date", "TR.F.TotRevenue.fperiod"],
            "Observe response structure for total-revenue value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "f_tot_assets_reporting_orig",
            "f_tot_assets",
            ["TR.F.TotAssets", "TR.F.TotAssets.date", "TR.F.TotAssets.fperiod"],
            "Observe response structure for total-assets value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "f_com_eq_tot_reporting_orig",
            "f_com_eq_tot",
            ["TR.F.ComEqTot", "TR.F.ComEqTot.date", "TR.F.ComEqTot.fperiod"],
            "Observe response structure for common-equity value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "f_gross_prof_ind_prop_tot_reporting_orig",
            "f_gross_prof_ind_prop_tot",
            [
                "TR.F.GrossProfIndPropTot",
                "TR.F.GrossProfIndPropTot.date",
                "TR.F.GrossProfIndPropTot.fperiod",
            ],
            "Observe response structure for gross-profit value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "f_net_cash_flow_op_reporting_orig",
            "f_net_cash_flow_op",
            ["TR.F.NetCashFlowOp", "TR.F.NetCashFlowOp.date", "TR.F.NetCashFlowOp.fperiod"],
            "Observe response structure for operating-net-cash-flow value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "f_debt_tot_reporting_orig",
            "f_debt_tot",
            ["TR.F.DebtTot", "TR.F.DebtTot.date", "TR.F.DebtTot.fperiod"],
            "Observe response structure for total-debt value, date, and fiscal-period fields.",
        ),
        annual_spec(
            "is_statement_dates_reporting_orig",
            "income_statement_dates",
            [
                "TR.ISOriginalAnnouncementDate",
                "TR.ISStatementLastUpdatedDate",
                "TR.ISPeriodEndDate",
            ],
            "Observe response structure for statement announcement, update, and period-end date fields.",
        ),
        annual_spec(
            "business_segments_fy0_reporting_orig",
            "business_segments_historical",
            [
                f"{segment_base}.segmentCode",
                f"{segment_base}.segmentName",
                f"{segment_base}.fperiod",
                f"{segment_base}.date",
                segment_base,
            ],
            "Observe response structure for the official historical business-segment field set.",
            scope_notes=segment_scope_notes,
        ),
        monthly_spec(
            "company_market_capitalization_date_monthly",
            "market_capitalization_monthly",
            ["TR.CompanyMarketCapitalization", "TR.CompanyMarketCapitalization.date"],
            "Observe response structure when market capitalization and its date suffix share one monthly request.",
        ),
    ]
    return specs


def build_plan() -> dict[str, Any]:
    specs = request_specs()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__)),
        "purpose": (
            "Small read-only LSEG point-in-time field availability and response-shape probe; "
            "only mechanical structure, non-null, date, and duplicate diagnostics are recorded."
        ),
        "universe": list(UNIVERSE),
        "annual_window": dict(ANNUAL_PARAMETERS),
        "monthly_window": dict(MONTHLY_PARAMETERS),
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "session": SESSION_NAME,
        "raw_response_policy": {
            "preserve_vendor_table": True,
            "cleaning": False,
            "blank_replacement": False,
            "missing_value_imputation": False,
            "winsorization": False,
            "deduplication": False,
            "row_deletion": False,
            "row_filtering": False,
            "date_or_timezone_conversion_in_raw": False,
            "unit_or_currency_conversion": False,
        },
        "research_use_policy": {
            "returns": False,
            "labels": False,
            "features": False,
            "portfolios": False,
            "models": False,
            "test_targets_read": False,
            "statement": "Probe artifacts are retained for field and response-shape inspection only.",
        },
        "summary_scope": ["structure", "non_null_counts", "date_diagnostics", "duplicate_diagnostics"],
        "semantic_boundary": (
            "Availability, non-null counts, returned dates, and duplicate diagnostics are mechanical observations; "
            "they do not verify economic meaning, accounting basis, units, currency, vintage, or point-in-time use."
        ),
        "requests": specs,
    }


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run id must be a single safe directory name")
    return run_id


def run_directory(run_id: str | None) -> Path:
    return RAW_ROOT / validate_run_id(run_id or new_run_id())


def load_or_create_plan(out: Path) -> dict[str, Any]:
    plan_path = out / "plan.json"
    fresh_plan = build_plan()
    if not plan_path.exists():
        atomic_write_json(plan_path, fresh_plan)
        return fresh_plan

    try:
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read existing plan: {redact(exc)}") from exc

    expected = [item["request_sha256"] for item in fresh_plan["requests"]]
    observed = [item.get("request_sha256") for item in existing.get("requests", [])]
    if existing.get("schema_version") != SCHEMA_VERSION or observed != expected:
        raise RuntimeError("existing plan does not match the current script; refusing to continue")
    if existing.get("script_sha256") != fresh_plan.get("script_sha256"):
        raise RuntimeError("existing plan was made by different script bytes; refusing to continue")
    return existing


def initial_summary(plan: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "purpose": plan["purpose"],
        "universe": plan["universe"],
        "annual_window": plan["annual_window"],
        "monthly_window": plan["monthly_window"],
        "request_timeout_seconds": plan["request_timeout_seconds"],
        "summary_scope": plan["summary_scope"],
        "semantic_boundary": plan["semantic_boundary"],
        "records": [
            {
                "request_id": spec["request_id"],
                "family": spec["family"],
                "request_sha256": spec["request_sha256"],
                "status": "planned",
            }
            for spec in plan["requests"]
        ],
    }


def save_summary(out: Path, summary: dict[str, Any]) -> None:
    summary["updated_at_utc"] = utc_now()
    atomic_write_json(out / "probe_summary.json", summary)


def ensure_summary(out: Path, plan: dict[str, Any]) -> dict[str, Any]:
    summary_path = out / "probe_summary.json"
    if not summary_path.exists():
        summary = initial_summary(plan, out.name)
        save_summary(out, summary)
        return summary
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read existing summary: {redact(exc)}") from exc
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("existing summary schema does not match the current script")
    if summary.get("run_id") != out.name:
        raise RuntimeError("existing summary run id does not match its directory")
    return summary


def summary_record(summary: dict[str, Any], request_id: str) -> dict[str, Any]:
    for record in summary.get("records", []):
        if record.get("request_id") == request_id:
            return record
    record = {"request_id": request_id, "status": "planned"}
    summary.setdefault("records", []).append(record)
    return record


def field_label(column: object) -> str:
    return str(column)


def column_occurrence_labels(columns: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    labels: list[str] = []
    for column in columns:
        base = field_label(column)
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        labels.append(base if occurrence == 0 else f"{base}__duplicate_{occurrence}")
    return labels


def is_date_column(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return any(
        token in normalized
        for token in ("date", "datetime", "timestamp", "announcement", "updated", "periodend")
    )


def is_period_column(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return "fperiod" in normalized or normalized.endswith("period") or "fiscalperiod" in normalized


def is_instrument_column(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized in {"instrument", "ric", "ticker", "symbol", "identifier"} or normalized.endswith(
        "instrument"
    )


def parse_date_diagnostics(frame: Any, labels: list[str]) -> dict[str, Any]:
    """Parse dates only for diagnostics; the returned frame is not changed."""

    ranges: dict[str, Any] = {}
    for position, label in enumerate(labels):
        if not is_date_column(label):
            continue
        try:
            series = frame.iloc[:, position]
            parsed = __import__("pandas").to_datetime(series, errors="coerce", utc=True)
            valid = parsed.notna()
            item: dict[str, Any] = {
                "non_null_parseable": int(valid.sum()),
                "parse_failures": int((~valid & series.notna()).sum()),
            }
            if bool(valid.any()):
                item["min"] = str(parsed[valid].min())
                item["max"] = str(parsed[valid].max())
                item["unique_values"] = int(parsed[valid].nunique())
            else:
                item["min"] = None
                item["max"] = None
                item["unique_values"] = 0
            ranges[label] = item
        except Exception as exc:
            ranges[label] = {"diagnostic_error": redact(exc)[:1000]}
    return ranges


def duplicate_key_diagnostics(frame: Any, labels: list[str]) -> dict[str, Any]:
    """Report mechanical duplicate diagnostics without removing or rewriting rows."""

    full_duplicate_mask = frame.duplicated(keep=False)
    full_duplicate_rows = int(full_duplicate_mask.sum())
    try:
        full_duplicate_groups = int(frame.loc[full_duplicate_mask].drop_duplicates().shape[0])
    except Exception:
        full_duplicate_groups = None
    result: dict[str, Any] = {
        "full_row_duplicate_rows": full_duplicate_rows,
        "full_row_duplicate_groups": full_duplicate_groups,
        "candidate_key_columns": [],
        "candidate_key_status": "not_assessed",
    }
    instrument = next((label for label in labels if is_instrument_column(label)), None)
    temporal = [label for label in labels if is_date_column(label) or is_period_column(label)]
    key_columns: list[str] = []
    if instrument:
        key_columns.append(instrument)
    key_columns.extend(label for label in temporal if label not in key_columns)
    result["candidate_key_columns"] = key_columns
    if not key_columns:
        result["candidate_key_reason"] = "no instrument/date/period labels were returned"
        return result

    positions = [labels.index(label) for label in key_columns]
    try:
        key_frame = frame.iloc[:, positions]
        duplicate_mask = key_frame.duplicated(keep=False)
        missing_mask = key_frame.isna().any(axis=1)
        try:
            duplicate_groups = int(key_frame.loc[duplicate_mask].drop_duplicates().shape[0])
        except Exception:
            duplicate_groups = None
        result.update(
            {
                "candidate_key_status": "assessed_mechanically",
                "candidate_key_duplicate_rows": int(duplicate_mask.sum()),
                "candidate_key_duplicate_groups": duplicate_groups,
                "candidate_key_missing_rows": int(missing_mask.sum()),
            }
        )
    except Exception as exc:
        result["candidate_key_status"] = "diagnostic_error"
        result["candidate_key_error"] = redact(exc)[:1000]
    return result


def analyse_frame(frame: Any) -> dict[str, Any]:
    labels = column_occurrence_labels(frame.columns)
    non_null: dict[str, int] = {}
    diagnostics: dict[str, Any] = {}
    for position, label in enumerate(labels):
        try:
            series = frame.iloc[:, position]
            non_null[label] = int(series.notna().sum())
        except Exception as exc:
            non_null[label] = -1
            diagnostics[label] = {"non_null_diagnostic_error": redact(exc)[:1000]}
    return {
        "rows": int(len(frame)),
        "columns": labels,
        "non_null": non_null,
        "date_ranges": parse_date_diagnostics(frame, labels),
        "duplicate_key_diagnostics": duplicate_key_diagnostics(frame, labels),
        "diagnostic_errors": diagnostics,
    }


def classify_failure(error: object) -> str:
    text = redact(error).lower()
    if any(token in text for token in ("permission", "permission denied", "not authorized", "forbidden", "unauthor")):
        return "permission_or_entitlement"
    if any(
        token in text
        for token in (
            "unable to resolve",
            "unable to collect",
            "unrecognized",
            "invalid field",
            "field not found",
            "unknown field",
            "field name",
            "invalid parameter",
        )
    ):
        return "field_resolution_or_parameter"
    if any(token in text for token in ("timed out", "timeout", "gateway time-out", "readtimeout")):
        return "timeout_or_gateway"
    if any(
        token in text
        for token in (
            "connection",
            "network",
            "http 4",
            "http 5",
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "server error",
            "too many requests",
            "transport",
        )
    ):
        return "transport_or_service"
    if any(token in text for token in ("session", "app key", "authentication", "login", "credential")):
        return "session_or_authentication"
    return "unknown_error"


def existing_artifact_state(out: Path, spec: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    request_id = spec["request_id"]
    csv_path = out / f"{request_id}.csv"
    metadata_path = out / f"{request_id}.metadata.json"
    request_path = out / f"{request_id}.request.json"
    error_path = out / f"{request_id}.error.json"
    if error_path.exists() and not (csv_path.exists() or metadata_path.exists()):
        try:
            sidecar = json.loads(error_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return "mismatch", {"reason": "error_sidecar_unreadable", "error": redact(exc)[:1000]}
        if sidecar.get("request_sha256") != spec["request_sha256"]:
            return "mismatch", {"reason": "error_sidecar_request_identity_mismatch"}
        return "terminal_error", sidecar
    if not any(path.exists() for path in (csv_path, metadata_path, request_path)):
        return "pending", None
    if not (csv_path.exists() and metadata_path.exists() and request_path.exists()):
        return "mismatch", {"reason": "incomplete_artifact_set"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "mismatch", {"reason": "artifact_json_unreadable", "error": redact(exc)[:1000]}
    request_hash = request_sha256(request)
    csv_hash = sha256_file(csv_path)
    if request_hash != spec["request_sha256"] or metadata.get("request_sha256") != spec["request_sha256"]:
        return "mismatch", {"reason": "request_identity_mismatch", "request_sha256": request_hash}
    if metadata.get("csv_sha256") != csv_hash:
        return "mismatch", {"reason": "csv_hash_mismatch", "csv_sha256": csv_hash}
    if metadata.get("request_file_sha256") not in (None, request_hash):
        return "mismatch", {"reason": "request_file_hash_mismatch", "request_sha256": request_hash}
    return "verified", metadata


def error_sidecar(
    out: Path,
    spec: dict[str, Any],
    status: str,
    error: object,
    elapsed: float | None = None,
) -> dict[str, Any]:
    error_text = redact(error)[:4000]
    failure_type = "timeout" if status == "timeout" else classify_failure(error_text)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "request_id": spec["request_id"],
        "family": spec["family"],
        "request": spec["request"],
        "request_sha256": spec["request_sha256"],
        "status": status,
        "failure_type": failure_type,
        "error": error_text,
        "at_utc": utc_now(),
    }
    if elapsed is not None:
        sidecar["elapsed_seconds"] = round(elapsed, 2)
    path = out / f"{spec['request_id']}.error.json"
    atomic_write_json(path, sidecar)
    sidecar["error_sha256"] = sha256_file(path)
    return sidecar


def record_empty(record: dict[str, Any], spec: dict[str, Any], elapsed: float) -> None:
    record.update(
        {
            "status": "empty",
            "rows": 0,
            "columns": [],
            "non_null": {},
            "date_ranges": {},
            "duplicate_key_diagnostics": {"status": "not_available", "reason": "no_frame_returned"},
            "elapsed_seconds": round(elapsed, 2),
            "finished_at_utc": utc_now(),
        }
    )


def run_one_request(
    out: Path,
    plan: dict[str, Any],
    summary: dict[str, Any],
    spec: dict[str, Any],
    ld: Any,
) -> bool:
    request_id = spec["request_id"]
    record = summary_record(summary, request_id)
    request_path = out / f"{request_id}.request.json"
    request_file_hash = atomic_write_request(request_path, spec["request"])
    if request_file_hash != spec["request_sha256"]:
        record.update(
            {
                "status": "error",
                "failure_type": "request_identity_write_failure",
                "request_file_sha256": request_file_hash,
                "finished_at_utc": utc_now(),
            }
        )
        save_summary(out, summary)
        return False

    record.update(
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "request_sha256": spec["request_sha256"],
            "request_file_sha256": request_file_hash,
        }
    )
    save_summary(out, summary)
    print(f"START {request_id}", flush=True)
    started = time.monotonic()

    def on_timeout() -> None:
        elapsed = time.monotonic() - started
        sidecar = error_sidecar(
            out,
            spec,
            "timeout",
            f"Request exceeded {REQUEST_TIMEOUT_SECONDS} seconds; availability remains unconfirmed.",
            elapsed,
        )
        record.update(
            {
                "status": "timeout",
                "failure_type": "timeout",
                "error": sidecar["error"],
                "elapsed_seconds": round(elapsed, 2),
                "finished_at_utc": utc_now(),
                "error_sha256": sidecar["error_sha256"],
            }
        )
        save_summary(out, summary)
        print(f"TIMEOUT {request_id}", flush=True)
        os._exit(2)

    timer = threading.Timer(REQUEST_TIMEOUT_SECONDS, on_timeout)
    timer.daemon = True
    timer.start()
    try:
        # Keep the vendor DataFrame untouched until atomically serialized.
        frame = ld.get_data(**spec["request"])
        elapsed = time.monotonic() - started
        if frame is None:
            record_empty(record, spec, elapsed)
            save_summary(out, summary)
            print(f"EMPTY {request_id} rows=0", flush=True)
            return True

        csv_path = out / f"{request_id}.csv"
        csv_hash = atomic_write_frame(csv_path, frame)
        analysis = analyse_frame(frame)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "family": spec["family"],
            "frequency": spec["frequency"],
            "request": spec["request"],
            "request_sha256": spec["request_sha256"],
            "request_file_sha256": request_file_hash,
            "status": "returned" if len(frame) else "empty",
            "collected_at_utc": utc_now(),
            "elapsed_seconds": round(elapsed, 2),
            "csv_path": csv_path.name,
            "csv_sha256": csv_hash,
            "raw_response_policy": plan["raw_response_policy"],
            "scope_notes": spec.get("scope_notes", {}),
            "requested_fields": list(spec["request"]["fields"]),
            **analysis,
        }
        metadata_path = out / f"{request_id}.metadata.json"
        atomic_write_json(metadata_path, metadata)
        metadata_hash = sha256_file(metadata_path)
        record.update(
            {
                **analysis,
                "status": metadata["status"],
                "request_sha256": spec["request_sha256"],
                "request_file_sha256": request_file_hash,
                "csv_sha256": csv_hash,
                "metadata_sha256": metadata_hash,
                "elapsed_seconds": round(elapsed, 2),
                "finished_at_utc": utc_now(),
            }
        )
        save_summary(out, summary)
        print(f"{metadata['status'].upper()} {request_id} rows={len(frame)}", flush=True)
        return True
    except Exception as exc:
        elapsed = time.monotonic() - started
        sidecar = error_sidecar(out, spec, "error", exc, elapsed)
        record.update(
            {
                "status": "error",
                "failure_type": sidecar["failure_type"],
                "error": sidecar["error"],
                "error_sha256": sidecar["error_sha256"],
                "elapsed_seconds": round(elapsed, 2),
                "finished_at_utc": utc_now(),
            }
        )
        save_summary(out, summary)
        print(f"ERROR {request_id} type={sidecar['failure_type']}", flush=True)
        return False
    finally:
        timer.cancel()


def execute(out: Path, plan: dict[str, Any], summary: dict[str, Any], limit: int) -> int:
    states: dict[str, str] = {}
    mismatches: list[str] = []
    for spec in plan["requests"]:
        state, metadata = existing_artifact_state(out, spec)
        states[spec["request_id"]] = state
        if state == "mismatch":
            mismatches.append(spec["request_id"])
        elif state == "terminal_error" and metadata is not None:
            record = summary_record(summary, spec["request_id"])
            record.update(
                {
                    "status": metadata.get("status", "error"),
                    "request_sha256": spec["request_sha256"],
                    "failure_type": metadata.get("failure_type", "unknown_error"),
                    "error": redact(metadata.get("error", ""))[:4000],
                    "error_sha256": sha256_file(out / f"{spec['request_id']}.error.json"),
                    "resumed_from_verified_error_sidecar": True,
                }
            )
        elif state == "verified" and metadata is not None:
            record = summary_record(summary, spec["request_id"])
            record.update(
                {
                    "status": metadata.get("status", "returned"),
                    "request_sha256": spec["request_sha256"],
                    "request_file_sha256": metadata.get("request_file_sha256"),
                    "csv_sha256": metadata.get("csv_sha256"),
                    "metadata_sha256": sha256_file(out / f"{spec['request_id']}.metadata.json"),
                    "rows": metadata.get("rows"),
                    "columns": metadata.get("columns", []),
                    "non_null": metadata.get("non_null", {}),
                    "date_ranges": metadata.get("date_ranges", {}),
                    "duplicate_key_diagnostics": metadata.get("duplicate_key_diagnostics", {}),
                    "resumed_from_verified_artifact": True,
                }
            )
    save_summary(out, summary)
    if mismatches:
        for request_id in mismatches:
            record = summary_record(summary, request_id)
            record.update(
                {
                    "status": "error",
                    "failure_type": "artifact_identity_or_hash_mismatch",
                    "error": "Existing request/CSV/metadata artifacts failed verification; no overwrite issued.",
                }
            )
        save_summary(out, summary)
        print(f"REFUSING TO RUN: {len(mismatches)} artifact set(s) failed identity/hash checks.", flush=True)
        return 1

    pending = [spec for spec in plan["requests"] if states[spec["request_id"]] == "pending"]
    terminal_ids = {
        spec["request_id"]
        for spec in plan["requests"]
        if summary_record(summary, spec["request_id"]).get("status") in TERMINAL_STATUSES
    }
    pending = [spec for spec in pending if spec["request_id"] not in terminal_ids]
    if not pending:
        print("All requests are already terminal or verified; no request issued.", flush=True)
        return 0

    # Importing the client is deliberately inside --execute after the no-network plan path.
    import lseg.data as ld

    try:
        ld.open_session(name=SESSION_NAME, app_key=CREDS.get("LSEG_APP_KEY"))
    except Exception as exc:
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "failure_type": "session_or_authentication",
            "error": redact(exc)[:4000],
            "at_utc": utc_now(),
        }
        atomic_write_json(out / "session.error.json", sidecar)
        summary["session"] = sidecar
        save_summary(out, summary)
        print("SESSION FAILED; no field request issued.", flush=True)
        return 1

    succeeded = 0
    attempted = 0
    try:
        for spec in pending:
            if limit and attempted >= limit:
                print(f"Reached --limit {limit}; pending requests remain.", flush=True)
                break
            attempted += 1
            if run_one_request(out, plan, summary, spec, ld):
                succeeded += 1
    finally:
        try:
            ld.close_session()
        except Exception as exc:
            summary["close_session_error"] = redact(exc)[:1000]

    summary["counts"] = {
        status: sum(record.get("status") == status for record in summary.get("records", []))
        for status in ["planned", "running", "returned", "empty", "error", "timeout"]
    }
    summary["last_execute"] = {
        "at_utc": utc_now(),
        "attempted": attempted,
        "succeeded_or_empty": succeeded,
        "pending_after_run": sum(
            record.get("status") in {"planned", "running"} for record in summary.get("records", [])
        ),
    }
    save_summary(out, summary)
    return 0 if not any(record.get("status") in {"planned", "running"} for record in summary["records"]) else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only LSEG point-in-time fundamentals probe")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="write plan.json only; issue no network request")
    mode.add_argument("--execute", action="store_true", help="execute pending requests explicitly")
    parser.add_argument("--run-id", default="", help="safe UTC run directory name to create or resume")
    parser.add_argument("--limit", type=int, default=0, help="execute at most N pending requests")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.limit and args.plan:
        parser.error("--limit is valid only with --execute")

    try:
        out = run_directory(args.run_id or None)
        out.mkdir(parents=True, exist_ok=True)
        plan = load_or_create_plan(out)
        if args.plan:
            print(f"Plan saved to {out / 'plan.json'}. No request was issued.", flush=True)
            return 0
        summary = ensure_summary(out, plan)
        return execute(out, plan, summary, args.limit)
    except Exception as exc:
        print(f"FAILED BEFORE REQUEST: {redact(exc)}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
