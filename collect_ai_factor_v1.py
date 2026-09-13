"""Collect raw AI-factor inputs from LSEG with explicit planning and execution modes.

The collector deliberately keeps the two stages separate::

    python collect_ai_factor_v1.py --plan
    python collect_ai_factor_v1.py --execute --run-id <UTC-run-id> [--limit N]

``--plan`` reads only the checked-in configuration and the corrected membership
spans file.  It writes one versioned ``plan.json`` and never imports the LSEG
client.  ``--execute`` is the only mode that imports ``lseg.data`` and opens a
session.  Successful request artifacts are reused only after the request JSON,
CSV hash, and metadata identity checks pass.  Failed attempts remain as numbered
error sidecars and may be retried by a later explicit execution.

Every vendor response is serialized directly.  This file does not parse dates,
replace blanks, impute values, deduplicate, filter, convert units/timezones, build
ratios/scores, or create returns and labels.  The corrected spans file supplies
literal RICs; no identity is inferred from names, tickers, suffixes, or issuer
guesses.  Delisted rows remain in the requested historical universe and are
marked in the plan.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - keeps the no-network plan path portable
    def dotenv_values(_path: Path) -> dict[str, str]:
        return {}


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ai_factor_collection_v1_config.json"
RAW_ROOT = ROOT / "data" / "raw" / "ai_factor_v1"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"
SCHEMA_VERSION = "ai_factor_collection_v1"
SESSION_NAME = "desktop.workspace"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TERMINAL_SUCCESS_STATUSES = {"returned", "empty"}
RETRYABLE_STATUSES = {"pending", "error", "timeout"}


# Credentials are used only by --execute when opening the local session.  They are
# never put into request objects, plans, metadata, manifests, or summaries.
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
    """Redact local credential values from output and third-party warnings."""

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


try:
    import pandas as pd  # noqa: E402
except ImportError as exc:  # pragma: no cover - a deployment dependency check
    raise RuntimeError("pandas is required to read the fixed membership spans input") from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run id must be a single safe UTC directory name")
    return run_id


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    """Stable bytes for request/config identity hashes."""

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
    """Write and install a file atomically in its final directory."""

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
    atomic_write_bytes(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8") + b"\n",
    )


def atomic_write_request(path: Path, request: dict[str, Any]) -> str:
    payload = canonical_json(request)
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def atomic_write_frame(path: Path, frame: Any, include_index: bool) -> str:
    """Serialize the returned vendor frame without transforming its values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=include_index)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", ""}:
        return False
    raise ValueError(f"cannot interpret delisted_ric value {value!r} as boolean")


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        config = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load collection config: {redact(exc)}") from exc
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("collection config schema_version does not match the collector")
    if config.get("config_version") != "1.0.0":
        raise RuntimeError("unsupported collection config_version")
    required = {"input", "windows", "fundamental_requests", "etf_history", "raw_response_policy"}
    missing = sorted(required - set(config))
    if missing:
        raise RuntimeError(f"collection config is missing keys: {missing}")
    return config, sha256_bytes(raw)


def load_spans(path: Path = SPANS_PATH) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], str]:
    """Read and validate the fixed spans source without changing it on disk."""

    config, _ = load_config()
    required = list(config["input"]["required_columns"])
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"cannot read corrected spans input: {redact(exc)}") from exc
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"corrected spans input is missing required columns: {missing}")
    if frame.empty:
        raise RuntimeError("corrected spans input has no rows")
    if frame["ric"].eq("").any():
        raise RuntimeError("corrected spans input contains an empty literal RIC")
    duplicate_mask = frame["ric"].duplicated(keep=False)
    if bool(duplicate_mask.any()):
        duplicates = sorted(frame.loc[duplicate_mask, "ric"].unique().tolist())
        raise RuntimeError(
            "corrected spans input contains duplicate literal RIC rows; refusing to deduplicate: "
            f"{duplicates[:10]}"
        )

    # Date parsing is used only to validate the membership source and to compute
    # diagnostics for the plan.  It never touches a vendor response or rewrites the
    # source CSV.
    member_from = pd.to_datetime(frame["member_from"], errors="coerce", format="mixed")
    member_to = pd.to_datetime(frame["member_to"], errors="coerce", format="mixed")
    if bool(member_from.isna().any() or member_to.isna().any()):
        raise RuntimeError("corrected spans input contains an empty or unparseable membership date")
    if bool((member_from > member_to).any()):
        raise RuntimeError("corrected spans input contains member_from after member_to")

    members: list[dict[str, Any]] = []
    for position, row in frame.iterrows():
        raw_delisted = str(row["delisted_ric"])
        delisted = normalize_bool(raw_delisted)
        members.append(
            {
                "source_row": int(position) + 2,
                "ric": str(row["ric"]),
                "name": str(row["name"]),
                "delisted_ric": delisted,
                "delisted_ric_raw": raw_delisted,
                "member_from": str(row["member_from"]),
                "member_to": str(row["member_to"]),
                "member_days": str(row["member_days"]),
                "fetch_start": str(row["fetch_start"]),
                "fetch_end": str(row["fetch_end"]),
                "span_status": "valid",
                "membership_status": "delisted_historical_span" if delisted else "live_historical_span",
                "identity_status": "raw_ric_unmapped",
            }
        )

    source_hash = sha256_file(path)
    delisted_count = sum(bool(item["delisted_ric"]) for item in members)
    span_counts = {
        "rows": int(len(frame)),
        "distinct_literal_rics": int(frame["ric"].nunique(dropna=False)),
        "live_rows": int(len(frame) - delisted_count),
        "delisted_rows": int(delisted_count),
        "valid_span_rows": int(len(frame)),
        "invalid_span_rows": 0,
        "duplicate_literal_ric_rows": 0,
    }
    return frame, members, span_counts, source_hash


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def chunks(values: list[Any], size: int) -> Iterable[tuple[int, list[Any]]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for offset in range(0, len(values), size):
        yield offset // size, values[offset:offset + size]


def parameters_for(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if "parameters" in spec:
        return dict(spec["parameters"])
    reference = spec.get("parameters_from")
    if reference == "windows.fundamental.parameters":
        return dict(config["windows"]["fundamental"]["parameters"])
    raise RuntimeError(f"unsupported parameter source for {spec.get('family_id')}: {reference}")


def member_batch_status(batch: list[dict[str, Any]]) -> dict[str, Any]:
    delisted = sum(bool(member["delisted_ric"]) for member in batch)
    statuses = sorted({str(member["span_status"]) for member in batch})
    return {
        "status": "included_from_corrected_spans",
        "span_status_values": statuses,
        "members": int(len(batch)),
        "live_members": int(len(batch) - delisted),
        "delisted_members": int(delisted),
        "rows_retained": int(len(batch)),
        "company_exclusion_applied": False,
    }


def build_plan(
    config: dict[str, Any],
    members: list[dict[str, Any]],
    spans_sha256: str,
    config_sha256: str,
    script_sha256: str,
    run_id: str,
    include_spy: bool = False,
) -> dict[str, Any]:
    """Build deterministic request specs and embed the exact member spans per batch."""

    batch_size = int(config["batching"]["stock_batch_size"])
    requests: list[dict[str, Any]] = []
    stock_rics = [member["ric"] for member in members]
    stock_timeout = int(config["timeouts_seconds"]["fundamental"])

    for family in config["fundamental_requests"]:
        family_id = str(family["family_id"])
        prefix = str(family["request_id_prefix"])
        fields = list(family["fields"])
        if family.get("method") != "get_data":
            raise RuntimeError(f"fundamental family {family_id} must use get_data")
        if not fields:
            raise RuntimeError(f"fundamental family {family_id} has no fields")
        parameters = parameters_for(config, family)
        for batch_number, batch in chunks(members, batch_size):
            request = {
                "universe": [member["ric"] for member in batch],
                "fields": fields,
                "parameters": parameters,
            }
            request_id = f"{prefix}_{batch_number:03d}"
            requests.append(
                {
                    "request_id": request_id,
                    "kind": "fundamental",
                    "family_id": family_id,
                    "group": family["group"],
                    "method": "get_data",
                    "batch_number": int(batch_number),
                    "batch_size": int(len(batch)),
                    "timeout_seconds": stock_timeout,
                    "fields": fields,
                    "parameters": parameters,
                    "request": request,
                    "request_sha256": request_sha256(request),
                    "request_file_sha256": request_sha256(request),
                    "members": batch,
                    "member_span_status": member_batch_status(batch),
                    "scope_notes": {
                        "source_spans_sha256": spans_sha256,
                        "identity_policy": config["input"]["ric_policy"],
                        "delisted_policy": config["input"]["membership_policy"],
                        "independent_request": bool(family.get("independent_request", True)),
                        **(
                            {
                                "reporting_state_exception": True,
                                "exception_rationale": family["exception_rationale"],
                                "reporting_state_sent": False,
                            }
                            if family.get("reporting_state_exception")
                            else {}
                        ),
                        **({"field_period": family["field_period"]} if "field_period" in family else {}),
                    },
                }
            )

    etf = config["etf_history"]
    etf_rics = list(etf["universe"])
    if include_spy:
        etf_rics.extend(str(ric) for ric in etf.get("optional_reconciliation_universe", []))
    if len(set(etf_rics)) != len(etf_rics):
        raise RuntimeError("ETF configuration contains duplicate literal RICs")
    etf_timeout = int(config["timeouts_seconds"]["etf_history"])
    history_window = config["windows"]["etf_history"]
    for batch_number, batch in chunks(etf_rics, int(config["batching"]["etf_batch_size"])):
        ric = str(batch[0])
        request = {
            "universe": batch,
            "fields": list(etf["fields"]),
            "interval": str(etf["interval"]),
            "start": str(history_window["start"]),
            "end": str(history_window["end"]),
            "adjustments": list(etf["adjustments"]),
        }
        requests.append(
            {
                "request_id": f"{etf['request_id_prefix']}_{ric}",
                "kind": "etf_history",
                "family_id": etf["family_id"],
                "group": etf["group"],
                "method": "get_history",
                "batch_number": int(batch_number),
                "batch_size": 1,
                "timeout_seconds": etf_timeout,
                "fields": list(etf["fields"]),
                "parameters": {
                    "interval": str(etf["interval"]),
                    "start": str(history_window["start"]),
                    "end": str(history_window["end"]),
                    "adjustments": list(etf["adjustments"]),
                },
                "request": request,
                "request_sha256": request_sha256(request),
                "request_file_sha256": request_sha256(request),
                "members": [],
                "member_span_status": {
                    "status": "fixed_etf_config",
                    "span_status_values": [],
                    "members": 0,
                    "live_members": 0,
                    "delisted_members": 0,
                    "rows_retained": 0,
                    "company_exclusion_applied": False,
                },
                "instrument": ric,
                "scope_notes": {
                    "identity_policy": "Use the fixed ETF RIC literal from configuration; do not infer an alternative listing.",
                    "independent_request": True,
                    "adjustments_source": "probe_price_execution_v3.FULL_ADJUSTMENTS",
                    "optional_spy_reconciliation": bool(ric == "SPY.P"),
                },
            }
        )

    fundamental_window = config["windows"]["fundamental"]
    return {
        "schema_version": SCHEMA_VERSION,
        "config_version": config["config_version"],
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "script": relative_path(Path(__file__)),
        "script_sha256": script_sha256,
        "config_path": relative_path(CONFIG_PATH),
        "config_sha256": config_sha256,
        "spans_path": relative_path(SPANS_PATH),
        "spans_sha256": spans_sha256,
        "span_counts": {
            "rows": int(len(members)),
            "distinct_literal_rics": int(len(members)),
            "live_rows": int(sum(not member["delisted_ric"] for member in members)),
            "delisted_rows": int(sum(member["delisted_ric"] for member in members)),
        },
        "fundamental_window": {
            "start": str(fundamental_window["start"]),
            "end": str(fundamental_window["end"]),
            "parameters": dict(fundamental_window["parameters"]),
        },
        "etf_history_window": {
            "start": str(history_window["start"]),
            "end": str(history_window["end"]),
            "interval": str(history_window["interval"]),
        },
        "stock_batch_size": batch_size,
        "etf_batch_size": int(config["batching"]["etf_batch_size"]),
        "include_spy_reconciliation": bool(include_spy),
        "requests": requests,
        "request_count": int(len(requests)),
        "expected_outputs": {
            "request": "<request_id>.request.json",
            "raw_csv": "<request_id>.csv",
            "metadata": "<request_id>.meta.json",
            "error_sidecar": "<request_id>.attempt_<n>.error.json plus <request_id>.error.json pointer",
            "execute_manifest": "collector_manifest.json",
            "execute_summary": "summary.json",
        },
        "raw_response_policy": config["raw_response_policy"],
        "test_seal_policy": config["test_seal_policy"],
        "identity_policy": config["input"]["ric_policy"],
        "delisted_policy": config["input"]["membership_policy"],
        "execution_policy": {
            "plan_imports_lseg": False,
            "execute_imports_lseg_only_after_plan": True,
            "request_timeout_seconds": {
                "fundamental": int(config["timeouts_seconds"]["fundamental"]),
                "etf_history": int(config["timeouts_seconds"]["etf_history"]),
            },
            "session_open_timeout_seconds": int(config["timeouts_seconds"].get("session_open", 90)),
            "reporting_state_rule": config.get("execution_policy", {}).get("reporting_state_rule"),
            "parameter_exceptions": config.get("execution_policy", {}).get("parameter_exceptions", {}),
            "raw_stage_boundary": config.get("execution_policy", {}).get("raw_stage_boundary"),
            "resume_requires": config["failure_policy"]["reuse_requires"],
            "failed_attempts_retained": True,
        },
        "request_order": "config family order, stock batches in corrected spans file order, then independent ETF RIC requests",
    }


def run_directory(run_id: str | None, raw_root: Path = RAW_ROOT) -> Path:
    return raw_root / validate_run_id(run_id or new_run_id())


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON artifact {path.name}: {redact(exc)}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact {path.name} is not an object")
    return value


def plan_identity(plan: dict[str, Any]) -> list[tuple[str, str]]:
    return [(str(item.get("request_id")), str(item.get("request_sha256"))) for item in plan.get("requests", [])]


def load_or_create_plan(
    out: Path,
    config: dict[str, Any],
    members: list[dict[str, Any]],
    spans_sha256: str,
    config_sha256: str,
    include_spy: bool,
) -> dict[str, Any]:
    fresh = build_plan(
        config=config,
        members=members,
        spans_sha256=spans_sha256,
        config_sha256=config_sha256,
        script_sha256=sha256_file(Path(__file__)),
        run_id=out.name,
        include_spy=include_spy,
    )
    plan_path = out / "plan.json"
    if not plan_path.exists():
        atomic_write_json(plan_path, fresh)
        return fresh
    existing = read_json(plan_path)
    checks = {
        "schema_version": existing.get("schema_version") == SCHEMA_VERSION,
        "run_id": existing.get("run_id") == out.name,
        "script_sha256": existing.get("script_sha256") == fresh["script_sha256"],
        "config_sha256": existing.get("config_sha256") == config_sha256,
        "spans_sha256": existing.get("spans_sha256") == spans_sha256,
        "include_spy_reconciliation": existing.get("include_spy_reconciliation") == bool(include_spy),
        "request_identity": plan_identity(existing) == plan_identity(fresh),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"existing plan does not match current config/input/script: {failed}")
    return existing


def occurrence_labels(columns: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    labels: list[str] = []
    for column in columns:
        base = str(column)
        occurrence = seen.get(base, 0)
        seen[base] = occurrence + 1
        labels.append(base if occurrence == 0 else f"{base}__duplicate_{occurrence}")
    return labels


def frame_analysis(frame: Any) -> dict[str, Any]:
    labels = occurrence_labels(frame.columns)
    dataframe_notna_counts: dict[str, int] = {}
    for position, label in enumerate(labels):
        try:
            # pandas.notna intentionally reports an empty string as present.  The
            # serialized CSV diagnostic below is kept separate so this semantic
            # difference is visible instead of being hidden under ``non_null``.
            dataframe_notna_counts[label] = int(frame.iloc[:, position].notna().sum())
        except Exception as exc:
            dataframe_notna_counts[label] = -1
            dataframe_notna_counts[f"{label}__diagnostic_error"] = redact(exc)[:1000]
    return {
        "rows": int(len(frame)),
        "columns": labels,
        "dataframe_notna_counts": dataframe_notna_counts,
        "diagnostics": {
            "date_parsing": False,
            "duplicate_check": False,
            "reason": "raw response is not parsed or changed; dataframe_notna_counts and serialized_nonempty_counts are separate diagnostics",
        },
    }


def serialized_nonempty_counts(
    path: Path,
    data_labels: list[str],
    row_count: int,
    include_index: bool,
) -> dict[str, int]:
    """Count physically non-empty CSV cells without converting text values.

    This is a post-write diagnostic only.  ``csv.reader`` preserves text ``"0"``
    as non-empty while an emitted empty cell remains empty.  Selecting the final
    ``row_count`` records avoids counting one or more header rows when a vendor
    frame has MultiIndex columns.
    """

    labels = (["__serialized_index__"] if include_index else []) + list(data_labels)
    counts = {label: 0 for label in labels}
    if not path.exists() or row_count <= 0 or not labels:
        return counts
    try:
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        data_rows = rows[-row_count:]
        for row in data_rows:
            for position, label in enumerate(labels):
                if position < len(row) and row[position] != "":
                    counts[label] += 1
    except Exception as exc:
        counts["__diagnostic_error__"] = redact(exc)[:1000]
    return counts


def artifact_paths(out: Path, spec: dict[str, Any]) -> dict[str, Path]:
    request_id = spec["request_id"]
    return {
        "request": out / f"{request_id}.request.json",
        "csv": out / f"{request_id}.csv",
        "metadata": out / f"{request_id}.meta.json",
        "error": out / f"{request_id}.error.json",
    }


def existing_artifact_state(out: Path, spec: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    paths = artifact_paths(out, spec)
    has_request = paths["request"].exists()
    has_csv = paths["csv"].exists()
    has_metadata = paths["metadata"].exists()
    has_error = paths["error"].exists()
    expected_request_file_hash = spec.get("request_file_sha256", spec["request_sha256"])

    # Validate the request file before considering a failure sidecar.  This keeps a
    # tampered request from being treated as a retryable failure merely because an
    # old error sidecar happens to be present.
    request: dict[str, Any] | None = None
    physical_request_hash: str | None = None
    if has_request:
        try:
            physical_request_hash = sha256_file(paths["request"])
            request = read_json(paths["request"])
        except (OSError, RuntimeError) as exc:
            return "mismatch", {"reason": "unreadable_request_artifact", "error": redact(exc)[:1000]}
        canonical_request_hash = request_sha256(request)
        request_checks = {
            "request_exact": request == spec["request"],
            "canonical_request_sha256": canonical_request_hash == spec["request_sha256"],
            "request_file_sha256": physical_request_hash == expected_request_file_hash,
        }
        if not all(request_checks.values()):
            return "mismatch", {
                "reason": "request_identity_or_physical_hash_failure",
                "checks": request_checks,
                "observed_request_sha256": canonical_request_hash,
                "observed_request_file_sha256": physical_request_hash,
            }

    sidecar: dict[str, Any] | None = None
    if has_error:
        try:
            sidecar = read_json(paths["error"])
        except RuntimeError as exc:
            return "mismatch", {"reason": "unreadable_error_sidecar", "error": redact(exc)[:1000]}
        sidecar_checks = {
            "status": sidecar.get("status") in {"error", "timeout"},
            "canonical_request_sha256": sidecar.get("request_sha256") == spec["request_sha256"],
            "request_exact": sidecar.get("request") in (None, spec["request"]),
            "request_file_sha256": sidecar.get("request_file_sha256") == physical_request_hash,
        }
        if not all(sidecar_checks.values()):
            return "mismatch", {
                "reason": "error_sidecar_identity_failure",
                "checks": sidecar_checks,
            }

    # A request + error sidecar with no success artifacts is a retryable failure.
    # This branch deliberately precedes the partial-success check, fixing the
    # interrupted-request state that previously looked like a success mismatch.
    if sidecar is not None and not has_csv and not has_metadata:
        return "failed", sidecar

    # Any CSV or metadata implies an attempted success.  Partial success artifacts
    # are blocked, even if an error sidecar exists, because they may be corrupted or
    # incomplete and must never be overwritten silently.
    if has_csv or has_metadata:
        if not has_request or not has_metadata:
            return "mismatch", {"reason": "incomplete_request_csv_metadata_set"}
        try:
            metadata = read_json(paths["metadata"])
        except RuntimeError as exc:
            return "mismatch", {"reason": "unreadable_metadata_artifact", "error": redact(exc)[:1000]}
        no_frame_returned = bool(metadata.get("no_frame_returned")) and metadata.get("csv_path") is None
        if not no_frame_returned and not has_csv:
            return "mismatch", {"reason": "missing_csv_for_returned_frame"}
        csv_hash = sha256_file(paths["csv"]) if has_csv else None
        metadata_checks = {
            "metadata_request_exact": metadata.get("request") == spec["request"],
            "metadata_canonical_request_sha256": metadata.get("request_sha256") == spec["request_sha256"],
            "metadata_request_file_sha256": metadata.get("request_file_sha256") == physical_request_hash,
            "metadata_csv_sha256": metadata.get("csv_sha256") == csv_hash,
            "metadata_status": metadata.get("status") in TERMINAL_SUCCESS_STATUSES,
            "metadata_no_frame_contract": no_frame_returned == (metadata.get("csv_sha256") is None),
        }
        if not all(metadata_checks.values()):
            return "mismatch", {
                "reason": "request_csv_metadata_identity_or_hash_failure",
                "checks": metadata_checks,
                "observed_request_file_sha256": physical_request_hash,
                "observed_csv_sha256": csv_hash,
            }
        return "verified", metadata

    # A valid request file without a response is an interrupted but still pending
    # request.  Its physical and canonical hashes were already checked above.
    return "pending", None


def classify_failure(error: object) -> str:
    text = redact(error).lower()
    if any(token in text for token in ("permission", "forbidden", "not authorized", "unauthor")):
        return "permission_or_entitlement"
    if any(token in text for token in ("unable to resolve", "unrecognized", "invalid field", "field not found", "unknown field", "invalid parameter")):
        return "field_resolution_or_parameter"
    if any(token in text for token in ("timed out", "timeout", "gateway time-out", "readtimeout")):
        return "timeout_or_gateway"
    if any(token in text for token in ("connection", "network", "http 4", "http 5", "429", "server error", "transport")):
        return "transport_or_service"
    if any(token in text for token in ("session", "app key", "authentication", "credential", "login")):
        return "session_or_authentication"
    return "unknown_error"


def next_attempt_number(out: Path, request_id: str) -> int:
    pattern = re.compile(rf"^{re.escape(request_id)}\.attempt_(\d+)\.error\.json$")
    numbers = []
    for path in out.glob(f"{request_id}.attempt_*.error.json"):
        match = pattern.fullmatch(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return (max(numbers) + 1) if numbers else 1


def write_error_sidecar(
    out: Path,
    plan: dict[str, Any],
    spec: dict[str, Any],
    status: str,
    error: object,
    elapsed_seconds: float | None,
    attempt: int | None = None,
) -> dict[str, Any]:
    if status not in {"error", "timeout"}:
        raise ValueError(f"unsupported error sidecar status: {status}")
    attempt_number = int(attempt or next_attempt_number(out, spec["request_id"]))
    request_path = artifact_paths(out, spec)["request"]
    request_file_hash = sha256_file(request_path) if request_path.exists() else None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "request_id": spec["request_id"],
        "family_id": spec["family_id"],
        "group": spec["group"],
        "method": spec["method"],
        "request": spec["request"],
        "request_sha256": spec["request_sha256"],
        "request_file_sha256": request_file_hash,
        "status": status,
        "failure_type": "timeout" if status == "timeout" else classify_failure(error),
        "error": redact(error)[:4000],
        "attempt": attempt_number,
        "at_utc": utc_now(),
        "timeout_seconds": spec.get("timeout_seconds"),
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    attempt_path = out / f"{spec['request_id']}.attempt_{attempt_number:03d}.error.json"
    atomic_write_json(attempt_path, payload)
    payload["attempt_error_path"] = attempt_path.name
    payload["attempt_error_sha256"] = sha256_file(attempt_path)
    # The unnumbered sidecar is a latest-failure pointer; numbered sidecars retain
    # every attempt and therefore preserve failure history.
    atomic_write_json(out / f"{spec['request_id']}.error.json", payload)
    return payload


def ensure_manifest(out: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = out / "collector_manifest.json"
    if path.exists():
        manifest = read_json(path)
        checks = {
            "schema_version": manifest.get("schema_version") == SCHEMA_VERSION,
            "run_id": manifest.get("run_id") == plan["run_id"],
            "plan_sha256": manifest.get("plan_sha256") == sha256_file(out / "plan.json"),
            "config_sha256": manifest.get("config_sha256") == plan["config_sha256"],
            "script_sha256": manifest.get("script_sha256") == plan["script_sha256"],
            "spans_path": manifest.get("spans_path") == plan["spans_path"],
            "spans_sha256": manifest.get("spans_sha256") == plan["spans_sha256"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"existing collector manifest identity mismatch: {checks}")
        return manifest
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "plan_sha256": sha256_file(out / "plan.json"),
        "config_path": plan["config_path"],
        "config_sha256": plan["config_sha256"],
        "script": plan["script"],
        "script_sha256": plan["script_sha256"],
        "spans_path": plan["spans_path"],
        "spans_sha256": plan["spans_sha256"],
        "span_counts": plan["span_counts"],
        "credential_redaction": {
            "enabled": True,
            "credentials_in_request_objects": False,
            "credentials_in_artifacts": False,
            "stdout_stderr_redacted": True,
        },
        "raw_response_policy": plan["raw_response_policy"],
        "test_seal_policy": plan["test_seal_policy"],
        "count_semantics": {
            "dataframe_notna_counts": "notna() on the in-memory vendor DataFrame; empty strings count as present",
            "serialized_nonempty_counts": "physical CSV cells whose csv.reader value is not the empty string; text 0 counts as present",
        },
        "requests": {},
        "session": {"status": "not_opened"},
        "status": "planned",
    }


def save_manifest(out: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(out / "collector_manifest.json", manifest)


def sync_manifest_records(out: Path, plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    records = manifest.setdefault("requests", {})
    for spec in plan["requests"]:
        request_id = spec["request_id"]
        state, artifact = existing_artifact_state(out, spec)
        record = records.setdefault(
            request_id,
            {
                "request_id": request_id,
                "family_id": spec["family_id"],
                "group": spec["group"],
                "method": spec["method"],
                "request_sha256": spec["request_sha256"],
                "status": "planned",
            },
        )
        if state == "verified" and artifact is not None:
            record.update(
                {
                    "status": artifact.get("status", "returned"),
                    "request_sha256": spec["request_sha256"],
                    "request_file_sha256": artifact.get("request_file_sha256"),
                    "csv_sha256": artifact.get("csv_sha256"),
                    "metadata_sha256": sha256_file(out / f"{request_id}.meta.json"),
                    "rows": artifact.get("rows", 0),
                    "columns": artifact.get("columns", []),
                    "dataframe_notna_counts": artifact.get("dataframe_notna_counts", {}),
                    "serialized_nonempty_counts": artifact.get("serialized_nonempty_counts", {}),
                    "resumed_from_verified_artifact": True,
                }
            )
        elif state == "failed" and artifact is not None:
            record.update(
                {
                    "status": artifact.get("status", "error"),
                    "request_sha256": spec["request_sha256"],
                    "failure_type": artifact.get("failure_type"),
                    "error": redact(artifact.get("error", ""))[:4000],
                    "error_sha256": artifact.get("attempt_error_sha256"),
                    "attempt": artifact.get("attempt"),
                }
            )
        elif state == "mismatch":
            record.update(
                {
                    "status": "error",
                    "failure_type": "artifact_identity_or_hash_mismatch",
                    "error": "Existing request/CSV/metadata artifacts failed verification; no overwrite issued.",
                }
            )
        else:
            record.setdefault("status", "planned")


def initial_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "plan_sha256": None,
        "config_path": plan["config_path"],
        "config_sha256": plan["config_sha256"],
        "script": plan["script"],
        "script_sha256": plan["script_sha256"],
        "spans_path": plan["spans_path"],
        "spans_sha256": plan["spans_sha256"],
        "span_counts": plan["span_counts"],
        "test_seal_policy": plan["test_seal_policy"],
        "raw_response_policy": plan["raw_response_policy"],
        "count_semantics": {
            "dataframe_notna_counts": "notna() on the in-memory vendor DataFrame; empty strings count as present",
            "serialized_nonempty_counts": "physical CSV cells whose csv.reader value is not the empty string; text 0 counts as present",
        },
        "status": "planned",
        "counts": {},
        "rows_by_kind": {},
        "instruments_by_kind": {},
        "requests": [],
        "errors": [],
    }


def update_summary(out: Path, plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    summary = initial_summary(plan)
    summary["plan_sha256"] = sha256_file(out / "plan.json")
    status_counts = Counter()
    rows_by_kind = Counter()
    instrument_sets: dict[str, set[str]] = {"fundamental": set(), "etf_history": set()}
    request_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    mismatches = 0
    for spec in plan["requests"]:
        state, artifact = existing_artifact_state(out, spec)
        status = "pending"
        if state == "verified" and artifact is not None:
            status = str(artifact.get("status", "returned"))
            rows_by_kind[spec["kind"]] += int(artifact.get("rows", 0) or 0)
            if spec["kind"] == "etf_history":
                instrument_sets["etf_history"].update(spec["request"].get("universe", []))
            else:
                instrument_sets["fundamental"].update(spec["request"].get("universe", []))
            request_rows.append(
                {
                    "request_id": spec["request_id"],
                    "family_id": spec["family_id"],
                    "group": spec["group"],
                    "kind": spec["kind"],
                    "status": status,
                    "rows": int(artifact.get("rows", 0) or 0),
                    "csv_sha256": artifact.get("csv_sha256"),
                    "metadata_sha256": sha256_file(out / f"{spec['request_id']}.meta.json"),
                }
            )
        elif state == "failed" and artifact is not None:
            status = str(artifact.get("status", "error"))
            error_item = {
                "request_id": spec["request_id"],
                "family_id": spec["family_id"],
                "group": spec["group"],
                "status": status,
                "failure_type": artifact.get("failure_type"),
                "error": redact(artifact.get("error", ""))[:4000],
                "attempt": artifact.get("attempt"),
                "error_sha256": artifact.get("attempt_error_sha256"),
            }
            errors.append(error_item)
            request_rows.append({**error_item, "kind": spec["kind"]})
        elif state == "mismatch":
            status = "artifact_mismatch"
            mismatches += 1
            errors.append(
                {
                    "request_id": spec["request_id"],
                    "family_id": spec["family_id"],
                    "status": status,
                    "failure_type": "artifact_identity_or_hash_mismatch",
                    "error": "Existing success artifacts failed verification; execution refused for this request.",
                }
            )
            request_rows.append(
                {
                    "request_id": spec["request_id"],
                    "family_id": spec["family_id"],
                    "group": spec["group"],
                    "kind": spec["kind"],
                    "status": status,
                }
            )
        status_counts[status] += 1

    summary["counts"] = dict(sorted(status_counts.items()))
    summary["rows_by_kind"] = dict(rows_by_kind)
    summary["instruments_by_kind"] = {key: len(value) for key, value in instrument_sets.items()}
    summary["requests"] = request_rows
    summary["errors"] = errors
    summary["status"] = (
        "blocked_artifact_mismatch" if mismatches else
        "complete" if not any(status in RETRYABLE_STATUSES or status == "pending" for status in status_counts) else
        "partial"
    )
    summary["test_targets_read"] = False
    summary["future_returns_generated"] = False
    summary["labels_generated"] = False
    summary["feature_or_factor_generated"] = False
    summary["failed_attempts_retained"] = len(errors)
    summary["manifest_sha256"] = sha256_file(out / "collector_manifest.json") if (out / "collector_manifest.json").exists() else None
    atomic_write_json(out / "summary.json", summary)
    return summary


def persist_running_record(out: Path, manifest: dict[str, Any], record: dict[str, Any]) -> None:
    manifest.setdefault("requests", {})[record["request_id"]] = record
    save_manifest(out, manifest)


def execute_one_request(
    out: Path,
    plan: dict[str, Any],
    manifest: dict[str, Any],
    spec: dict[str, Any],
    ld: Any,
) -> tuple[str, dict[str, Any] | None]:
    request_id = spec["request_id"]
    paths = artifact_paths(out, spec)
    attempt = next_attempt_number(out, request_id)
    request_file_hash = atomic_write_request(paths["request"], spec["request"])
    if request_file_hash != spec["request_sha256"]:
        sidecar = write_error_sidecar(
            out,
            plan,
            spec,
            "error",
            "request identity write failure",
            0.0,
            attempt,
        )
        return "error", sidecar

    record: dict[str, Any] = {
        "request_id": request_id,
        "family_id": spec["family_id"],
        "group": spec["group"],
        "kind": spec["kind"],
        "method": spec["method"],
        "request_sha256": spec["request_sha256"],
        "request_file_sha256": request_file_hash,
        "attempt": attempt,
        "status": "running",
        "started_at_utc": utc_now(),
        "timeout_seconds": spec["timeout_seconds"],
    }
    persist_running_record(out, manifest, record)
    print(f"START {request_id} ({spec['kind']}, timeout={spec['timeout_seconds']}s)", flush=True)
    started = time.monotonic()
    finished = threading.Event()

    def on_timeout() -> None:
        if finished.is_set():
            return
        elapsed = time.monotonic() - started
        sidecar = write_error_sidecar(
            out,
            plan,
            spec,
            "timeout",
            f"request exceeded {spec['timeout_seconds']} seconds",
            elapsed,
            attempt,
        )
        record.update(
            {
                "status": "timeout",
                "failure_type": "timeout",
                "error": sidecar["error"],
                "error_sha256": sidecar["attempt_error_sha256"],
                "elapsed_seconds": round(elapsed, 3),
                "finished_at_utc": utc_now(),
            }
        )
        persist_running_record(out, manifest, record)
        print(f"TIMEOUT {request_id}", flush=True)
        # A blocked client call cannot be cancelled through the LSEG Python API.
        # Terminating the process prevents a late response from being installed as
        # successful; the numbered timeout sidecar is sufficient for resume/audit.
        os._exit(2)

    timer = threading.Timer(float(spec["timeout_seconds"]), on_timeout)
    timer.daemon = True
    timer.start()
    try:
        if spec["method"] == "get_data":
            frame = ld.get_data(**spec["request"])
            include_index = False
        elif spec["method"] == "get_history":
            frame = ld.get_history(**spec["request"])
            include_index = True
        else:
            raise RuntimeError(f"unsupported request method {spec['method']}")
        finished.set()
        elapsed = time.monotonic() - started
        if frame is None:
            # No vendor frame means no raw CSV exists.  Metadata explicitly records
            # the absence, rather than fabricating columns or values.
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "run_id": plan["run_id"],
                "request_id": request_id,
                "family_id": spec["family_id"],
                "group": spec["group"],
                "kind": spec["kind"],
                "method": spec["method"],
                "request": spec["request"],
                "request_sha256": spec["request_sha256"],
                "request_file_sha256": request_file_hash,
                "status": "empty",
                "no_frame_returned": True,
                "csv_path": None,
                "csv_sha256": None,
                "rows": 0,
                "columns": [],
                "dataframe_notna_counts": {},
                "serialized_nonempty_counts": {},
                "count_semantics": {
                    "dataframe_notna_counts": "notna() on the in-memory vendor DataFrame; empty strings count as present",
                    "serialized_nonempty_counts": "physical CSV cells whose csv.reader value is not the empty string; text 0 counts as present",
                },
                "collected_at_utc": utc_now(),
                "started_at_utc": record["started_at_utc"],
                "elapsed_seconds": round(elapsed, 3),
                "raw_response_policy": plan["raw_response_policy"],
                "scope_notes": spec.get("scope_notes", {}),
                "requested_instrument_count": len(spec["request"].get("universe", [])),
            }
        else:
            analysis = frame_analysis(frame)
            csv_hash = atomic_write_frame(paths["csv"], frame, include_index=include_index)
            serialized_counts = serialized_nonempty_counts(
                paths["csv"],
                analysis["columns"],
                analysis["rows"],
                include_index=include_index,
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "run_id": plan["run_id"],
                "request_id": request_id,
                "family_id": spec["family_id"],
                "group": spec["group"],
                "kind": spec["kind"],
                "method": spec["method"],
                "request": spec["request"],
                "request_sha256": spec["request_sha256"],
                "request_file_sha256": request_file_hash,
                "status": "returned" if len(frame) else "empty",
                "no_frame_returned": False,
                "csv_path": paths["csv"].name,
                "csv_sha256": csv_hash,
                "csv_bytes": paths["csv"].stat().st_size,
                "collected_at_utc": utc_now(),
                "started_at_utc": record["started_at_utc"],
                "elapsed_seconds": round(elapsed, 3),
                "raw_response_policy": plan["raw_response_policy"],
                "scope_notes": spec.get("scope_notes", {}),
                "requested_instrument_count": len(spec["request"].get("universe", [])),
                "serialized_nonempty_counts": serialized_counts,
                "count_semantics": {
                    "dataframe_notna_counts": "notna() on the in-memory vendor DataFrame; empty strings count as present",
                    "serialized_nonempty_counts": "physical CSV cells whose csv.reader value is not the empty string; text 0 counts as present",
                },
                **analysis,
            }
        atomic_write_json(paths["metadata"], metadata)
        metadata_hash = sha256_file(paths["metadata"])
        record.update(
            {
                "status": metadata["status"],
                "rows": metadata["rows"],
                "columns": metadata["columns"],
                "dataframe_notna_counts": metadata["dataframe_notna_counts"],
                "serialized_nonempty_counts": metadata["serialized_nonempty_counts"],
                "csv_sha256": metadata["csv_sha256"],
                "metadata_sha256": metadata_hash,
                "elapsed_seconds": metadata["elapsed_seconds"],
                "finished_at_utc": utc_now(),
            }
        )
        persist_running_record(out, manifest, record)
        print(f"{metadata['status'].upper()} {request_id} rows={metadata['rows']}", flush=True)
        return str(metadata["status"]), metadata
    except Exception as exc:
        finished.set()
        elapsed = time.monotonic() - started
        sidecar = write_error_sidecar(out, plan, spec, "error", exc, elapsed, attempt)
        record.update(
            {
                "status": "error",
                "failure_type": sidecar["failure_type"],
                "error": sidecar["error"],
                "error_sha256": sidecar["attempt_error_sha256"],
                "elapsed_seconds": round(elapsed, 3),
                "finished_at_utc": utc_now(),
            }
        )
        persist_running_record(out, manifest, record)
        print(f"ERROR {request_id} type={sidecar['failure_type']}", flush=True)
        return "error", sidecar
    finally:
        finished.set()
        timer.cancel()


def open_session_with_watchdog(out: Path, plan: dict[str, Any], manifest: dict[str, Any], ld: Any) -> bool:
    timeout = int(
        plan.get("execution_policy", {}).get("session_open_timeout_seconds", 90)
    )
    finished = threading.Event()

    def on_timeout() -> None:
        if finished.is_set():
            return
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "run_id": plan["run_id"],
            "status": "timeout",
            "failure_type": "session_or_authentication",
            "error": f"session open exceeded {timeout} seconds",
            "at_utc": utc_now(),
        }
        atomic_write_json(out / "session.error.json", sidecar)
        manifest["session"] = sidecar
        save_manifest(out, manifest)
        os._exit(2)

    timer = threading.Timer(timeout, on_timeout)
    timer.daemon = True
    timer.start()
    try:
        ld.open_session(name=SESSION_NAME, app_key=CREDS.get("LSEG_APP_KEY"))
        finished.set()
        manifest["session"] = {"status": "connected", "opened_at_utc": utc_now()}
        save_manifest(out, manifest)
        return True
    except Exception as exc:
        finished.set()
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "run_id": plan["run_id"],
            "status": "error",
            "failure_type": "session_or_authentication",
            "error": redact(exc)[:4000],
            "at_utc": utc_now(),
        }
        atomic_write_json(out / "session.error.json", sidecar)
        manifest["session"] = sidecar
        save_manifest(out, manifest)
        print("SESSION FAILED; no field request issued.", flush=True)
        return False
    finally:
        finished.set()
        timer.cancel()


def execute(
    out: Path,
    plan: dict[str, Any],
    config: dict[str, Any],
    limit: int,
) -> int:
    manifest = ensure_manifest(out, plan)
    sync_manifest_records(out, plan, manifest)
    save_manifest(out, manifest)
    summary = update_summary(out, plan, manifest)
    if summary["status"] == "blocked_artifact_mismatch":
        print("REFUSING TO RUN: existing artifact identity/hash verification failed.", flush=True)
        return 1

    states: dict[str, str] = {}
    mismatches: list[str] = []
    pending: list[dict[str, Any]] = []
    for spec in plan["requests"]:
        state, _artifact = existing_artifact_state(out, spec)
        states[spec["request_id"]] = state
        if state == "mismatch":
            mismatches.append(spec["request_id"])
        elif state in {"pending", "failed"}:
            pending.append(spec)
    if mismatches:
        print(f"REFUSING TO RUN: {len(mismatches)} artifact set(s) failed identity/hash checks.", flush=True)
        return 1
    if not pending:
        manifest["status"] = "complete"
        save_manifest(out, manifest)
        update_summary(out, plan, manifest)
        print("All requests are already verified or have no retryable artifacts.", flush=True)
        return 0

    print(
        "阶段说明（采集执行）：读取 corrected spans 中的原始 RIC，按 25 只股票批次和单个 ETF 请求调用 LSEG；"
        "只保存供应商原始响应及元数据，不清洗、不去重、不填补、不转日期/时区/单位，不生成收益、标签或因子；"
        "失败与退市标记均保留。",
        flush=True,
    )
    # Importing the LSEG client is deliberately delayed until explicit --execute,
    # after the plan and all identity checks have completed.
    try:
        import lseg.data as ld  # noqa: PLC0415
    except Exception as exc:
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "run_id": plan["run_id"],
            "status": "error",
            "failure_type": "lseg_import",
            "error": redact(exc)[:4000],
            "at_utc": utc_now(),
        }
        atomic_write_json(out / "session.error.json", sidecar)
        manifest["session"] = sidecar
        manifest["status"] = "error"
        save_manifest(out, manifest)
        update_summary(out, plan, manifest)
        print("LSEG CLIENT IMPORT FAILED; no field request issued.", flush=True)
        return 1

    if not open_session_with_watchdog(out, plan, manifest, ld):
        manifest["status"] = "error"
        save_manifest(out, manifest)
        update_summary(out, plan, manifest)
        return 1

    attempted = 0
    successes = 0
    errors = 0
    try:
        for spec in pending:
            if limit and attempted >= limit:
                print(f"Reached --limit {limit}; pending requests remain.", flush=True)
                break
            attempted += 1
            status, _artifact = execute_one_request(out, plan, manifest, spec, ld)
            if status in TERMINAL_SUCCESS_STATUSES:
                successes += 1
            elif status in {"error", "timeout"}:
                errors += 1
            manifest["status"] = "running"
            save_manifest(out, manifest)
            update_summary(out, plan, manifest)
    finally:
        try:
            ld.close_session()
            manifest["session"]["closed_at_utc"] = utc_now()
        except Exception as exc:
            manifest["close_session_error"] = redact(exc)[:1000]
        manifest["last_execute"] = {
            "at_utc": utc_now(),
            "attempted": attempted,
            "successful_or_empty": successes,
            "errors": errors,
            "limit": limit,
        }
        save_manifest(out, manifest)

    final_summary = update_summary(out, plan, manifest)
    manifest["status"] = final_summary["status"]
    save_manifest(out, manifest)
    final_summary = update_summary(out, plan, manifest)
    print(
        json.dumps(
            {
                "run_id": plan["run_id"],
                "attempted": attempted,
                "successful_or_empty": successes,
                "errors": errors,
                "summary": final_summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if errors:
        return 1
    return 0 if final_summary["status"] == "complete" else 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Versioned raw AI-factor input collector")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true", help="write one versioned plan.json; never import LSEG")
    modes.add_argument("--execute", action="store_true", help="execute pending requests explicitly")
    parser.add_argument("--run-id", default="", help="safe UTC run directory name to create or resume")
    parser.add_argument("--limit", type=int, default=0, help="execute at most N pending/failed requests")
    parser.add_argument(
        "--include-spy",
        action="store_true",
        help="include optional SPY.P history as an independent reconciliation request",
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.plan and args.limit:
        parser.error("--limit is valid only with --execute")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha256 = load_config()
    _spans_frame, members, _span_counts, spans_sha256 = load_spans(SPANS_PATH)
    raw_root = ROOT / Path(config["output"]["raw_root"])
    out = run_directory(args.run_id or None, raw_root=raw_root)
    out.mkdir(parents=True, exist_ok=True)
    plan = load_or_create_plan(
        out=out,
        config=config,
        members=members,
        spans_sha256=spans_sha256,
        config_sha256=config_sha256,
        include_spy=bool(args.include_spy),
    )
    if args.plan:
        print(f"计划已保存至 {out / 'plan.json'}；未导入 LSEG、未发起网络请求。", flush=True)
        return 0
    return execute(out, plan, config, args.limit)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED BEFORE REQUEST: {redact(exc)}", flush=True)
        raise SystemExit(1)
