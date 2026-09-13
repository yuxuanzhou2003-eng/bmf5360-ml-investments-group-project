"""Independent validator for the analyst-graph model-ready v1 layer.

The validator is deliberately independent of the builder.  It checks the
frozen graph edge set, point-in-time membership, event splits, eligibility,
development-period targets, and exact clean-price provenance.  Test-period
future targets are sealed: the validator only checks that the forward-return
columns are missing, ``label_available`` is false, and the reason is
``test_target_sealed``; it never loads or recomputes test-period target values.

Usage::

    python validate_analyst_graph_model_ready_v1.py
    python validate_analyst_graph_model_ready_v1.py --run-dir data/.../<run>

The newest model-ready run is selected when ``--run-dir`` is omitted.  One
run-specific JSON report and mismatch extracts, when needed, are written to
``data/audit/analyst_graph_model_ready_v1/<audit_run>/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "data" / "analyst_graph_model_ready_v1"
AUDIT_ROOT = ROOT / "data" / "audit" / "analyst_graph_model_ready_v1"
GRAPH_ROOT = ROOT / "data" / "analyst_graph_v2"
PANEL_ROOT = ROOT / "data" / "panel_v3"
SPANS_DEFAULT = ROOT / "data" / "audit" / "universe_rebuild" / "eligible_spans_2015_2026_corrected.csv"

EXPECTED_ROWS = 111_128
EVENT_KEY = ["source", "announcement", "period_end"]
EDGE_KEY = EVENT_KEY + ["receiver"]
TARGET_REASON = "test_target_sealed"
MAX_GRAPH_AGE_DAYS = 120
HORIZON = 5

TABLE_ALIASES = {
    "features": ("model_features.csv", "features.csv"),
    "metadata": ("metadata.csv",),
    "targets": ("targets.csv", "labels.csv"),
    "eligibility": ("eligibility.csv",),
    "execution": ("execution_inputs.csv",),
    "feature_missingness": ("feature_missingness.csv",),
    "split_counts": ("split_counts.csv",),
}

IDENTITY_COLUMNS = {
    "sample_id", "source", "receiver", "announcement", "announcement_day", "period_end",
    "snapshot", "snapshot_age_days", "common_brokers", "jaccard_coverage",
    "jaccard_rec_weighted", "jaccard_named_only", "common_rated_brokers",
    "graph_snapshot", "graph_snapshot_age_days", "graph_common_brokers",
    "graph_jaccard_coverage", "graph_jaccard_rec_weighted", "graph_jaccard_named_only",
    "graph_common_rated_brokers", "liquidity_reference_session", "entry_session",
    "exit_session", "split", "next_split_boundary", "boundary_purged", "split_retained",
}

GRAPH_FIELDS = [
    "snapshot", "snapshot_age_days", "common_brokers", "jaccard_coverage",
    "jaccard_rec_weighted", "jaccard_named_only", "common_rated_brokers",
]

TARGET_FIELDS = [
    "entry_session", "exit_session", "label_complete", "receiver_forward_return",
    "benchmark_forward_return", "forward_benchmark_excess",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def jsonable(value: Any) -> Any:
    """Convert pandas/numpy/path values into values accepted by json.dumps."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def parse_date(values: Iterable[Any], name: str, allow_missing: bool = True) -> pd.Series:
    parsed = pd.to_datetime(pd.Series(values), format="mixed", errors="coerce")
    # Dates in this project are deliberately vendor-naive.  Normalising a
    # timezone-aware series would silently change the stated announcement day.
    if getattr(parsed.dt, "tz", None) is not None:
        raise ValueError(f"{name} contains timezone-aware values")
    parsed = parsed.dt.normalize()
    if not allow_missing and parsed.isna().any():
        raise ValueError(f"{name} contains missing or invalid dates")
    return parsed


def parse_bool(values: Iterable[Any], name: str, allow_missing: bool = False) -> pd.Series:
    series = pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        result = series.astype("boolean")
    else:
        normalized = series.astype("string").str.strip().str.lower()
        invalid = sorted(set(normalized.dropna()) - {"true", "false"})
        if invalid:
            raise ValueError(f"{name} contains invalid booleans: {invalid[:10]}")
        result = normalized.eq("true").astype("boolean")
        result[normalized.isna()] = pd.NA
    if not allow_missing and result.isna().any():
        raise ValueError(f"{name} contains missing booleans")
    return result


def values_close(left: Iterable[Any], right: Iterable[Any], atol: float = 1e-11) -> bool:
    try:
        return bool(np.allclose(
            pd.to_numeric(pd.Series(left), errors="coerce").to_numpy(float),
            pd.to_numeric(pd.Series(right), errors="coerce").to_numpy(float),
            rtol=1e-9, atol=atol, equal_nan=True,
        ))
    except (TypeError, ValueError):
        return False


def exact_series_equal(left: pd.Series, right: pd.Series, kind: str = "string") -> bool:
    """Compare two aligned series, retaining NA=NA semantics."""
    if kind == "numeric":
        return values_close(left, right)
    if kind == "date":
        a = parse_date(left, "left date")
        b = parse_date(right, "right date")
        return bool((a.eq(b) | (a.isna() & b.isna())).all())
    if kind == "bool":
        try:
            a = parse_bool(left, "left bool", allow_missing=True)
            b = parse_bool(right, "right bool", allow_missing=True)
            return bool((a.eq(b) | (a.isna() & b.isna())).all())
        except ValueError:
            return False
    a = pd.Series(left).astype("string")
    b = pd.Series(right).astype("string")
    return bool((a.eq(b) | (a.isna() & b.isna())).all())


def latest_run(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    if not candidates:
        # Some interrupted builder runs may not have a summary yet.  Keeping
        # them out of auto-selection avoids validating a partial directory.
        raise FileNotFoundError(f"No completed runs with summary.json under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_path(value: Any, default: Path | None = None) -> Path | None:
    """Resolve a summary path/ID without allowing a missing value to crash a check."""
    if isinstance(value, dict):
        value = value.get("path") or value.get("file") or value.get("directory") or value.get("run_id")
    if value is None:
        return default
    text = str(value)
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    candidates = [ROOT / candidate]
    if default is not None:
        candidates.append(default / candidate)
    for item in candidates:
        if item.exists():
            return item
    return candidates[0]


def hash_entries(summary: dict, section: str) -> dict[str, Any]:
    values = summary.get(section, {})
    return values if isinstance(values, dict) else {}


def resolve_summary_input(summary: dict, names: Iterable[str], default: Path | None = None) -> Path | None:
    hashes = summary.get("input_hashes", {})
    inputs = summary.get("inputs", {})
    for name in names:
        for container in (hashes, inputs):
            if isinstance(container, dict) and name in container:
                return resolve_path(container[name], default)
    # Allow a summary to record an ID instead of an input-hash object.
    for name in names:
        if name in summary:
            return resolve_path(summary[name], default)
    return default


def discover_graph_run(summary: dict) -> Path:
    candidates: list[Path] = []
    for key in ("graph_run_dir", "graph_run", "graph_run_id", "analyst_graph_run_id", "analyst_graph_v2_run"):
        value = summary.get(key)
        path = resolve_path(value, GRAPH_ROOT)
        if path is not None and path.is_dir() and (path / "edges.csv").exists():
            candidates.append(path)
    for section in (summary.get("input_hashes", {}), summary.get("inputs", {})):
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if "graph" not in str(key).lower():
                continue
            path = resolve_path(value, GRAPH_ROOT)
            if path is None:
                continue
            if path.name in {"edges.csv", "event_graph_status.csv", "summary.json"}:
                path = path.parent
            if path.is_dir() and (path / "edges.csv").exists():
                candidates.append(path)
    if candidates:
        # De-duplicate while retaining the first explicit resolution.
        return list(dict.fromkeys(p.resolve() for p in candidates))[0]
    return latest_run(GRAPH_ROOT)


def discover_panel_run(summary: dict) -> Path | None:
    value = summary.get("panel_run_id") or summary.get("panel_run")
    path = resolve_path(value, PANEL_ROOT)
    if path is not None and path.is_dir() and (path / "features.csv").exists():
        return path
    for section in (summary.get("input_hashes", {}), summary.get("inputs", {})):
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if "panel" not in str(key).lower():
                continue
            path = resolve_path(value, PANEL_ROOT)
            if path is None:
                continue
            if path.name.endswith(".csv"):
                path = path.parent
            if path.is_dir() and (path / "features.csv").exists():
                return path
    try:
        return latest_run(PANEL_ROOT)
    except FileNotFoundError:
        return None


def table_path(run_dir: Path, logical: str, summary: dict) -> Path | None:
    # Prefer explicitly recorded output paths, then the stable compatibility
    # aliases used by model-ready v1 and panel v3.
    output_hashes = summary.get("output_hashes", {})
    if isinstance(output_hashes, dict):
        for name, item in output_hashes.items():
            stem = Path(str(name)).name
            if stem in TABLE_ALIASES.get(logical, ()) or f"{stem}.csv" in TABLE_ALIASES.get(logical, ()):
                path = resolve_path(item, run_dir)
                if path is not None and path.exists():
                    return path
    for name in TABLE_ALIASES.get(logical, ()):
        path = run_dir / name
        if path.exists():
            return path
    return None


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_spans(path: Path) -> pd.DataFrame:
    spans = read_csv(path)
    ric = "ric" if "ric" in spans else ("Instrument" if "Instrument" in spans else None)
    start = "member_from" if "member_from" in spans else ("start" if "start" in spans else None)
    end = "member_to" if "member_to" in spans else ("end" if "end" in spans else None)
    if not all((ric, start, end)):
        raise ValueError("corrected spans must contain ric, member_from/member_to")
    result = spans[[ric, start, end]].rename(columns={ric: "ric", start: "member_from", end: "member_to"}).copy()
    result["ric"] = result["ric"].astype("string")
    result["member_from"] = parse_date(result["member_from"], "membership member_from", allow_missing=False)
    result["member_to"] = parse_date(result["member_to"], "membership member_to", allow_missing=False)
    if (result["member_to"] < result["member_from"]).any():
        raise ValueError("membership end precedes start")
    return result


def membership_flags(spans: pd.DataFrame, instruments: Iterable[Any], days: Iterable[Any]) -> pd.Series:
    cache: dict[pd.Timestamp, set[str]] = {}
    instruments = pd.Series(instruments).astype("string")
    days = parse_date(days, "membership day")
    output = []
    for instrument, day in zip(instruments, days):
        if pd.isna(instrument) or pd.isna(day):
            output.append(False)
            continue
        if day not in cache:
            cache[day] = set(spans.loc[
                spans.member_from.le(day) & spans.member_to.ge(day), "ric"
            ].astype(str))
        output.append(str(instrument) in cache[day])
    return pd.Series(output, index=instruments.index, dtype="boolean")


def normalise_key_frame(frame: pd.DataFrame, ids: pd.Series, name: str) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    if "sample_id" not in frame:
        return pd.DataFrame(index=range(len(ids))), [f"{name} missing sample_id"]
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        errors.append(f"{name} sample_id is missing or duplicated")
    indexed = frame.set_index("sample_id", drop=False)
    missing = set(ids.astype(str)) - set(indexed.index.astype(str))
    extra = set(indexed.index.astype(str)) - set(ids.astype(str))
    if missing:
        errors.append(f"{name} missing {len(missing)} sample_id values")
    if extra:
        errors.append(f"{name} has {len(extra)} extra sample_id values")
    # Reindexing with strings avoids pandas treating long hexadecimal IDs as
    # mixed numeric/object values after a malformed CSV read.
    out = indexed.reindex(ids.astype(str)).copy()
    out["sample_id"] = ids.astype(str).to_numpy()
    return out.reset_index(drop=True), errors


def locate_column(frames: Iterable[pd.DataFrame], aliases: Iterable[str]) -> tuple[pd.Series | None, str | None]:
    aliases = list(aliases)
    for frame in frames:
        for alias in aliases:
            if alias in frame:
                return frame[alias], alias
    return None, None


def aliases_for(field: str) -> list[str]:
    aliases = [field, f"graph_{field}", f"analyst_graph_{field}"]
    if field == "common_brokers":
        aliases += ["common_analyst_brokers", "graph_common_analyst_brokers"]
    elif field in {"jaccard_coverage", "jaccard_rec_weighted", "jaccard_named_only"}:
        aliases += [f"broker_{field}", f"analyst_{field}"]
    return aliases


def first_present(mapping: dict[str, pd.DataFrame], names: Iterable[str]) -> tuple[pd.Series | None, str | None]:
    return locate_column(mapping.values(), names)


def split_for_day(day: pd.Timestamp) -> str:
    if pd.Timestamp("2015-01-01") <= day <= pd.Timestamp("2020-12-31"):
        return "training"
    if pd.Timestamp("2021-01-01") <= day <= pd.Timestamp("2022-12-31"):
        return "validation"
    if pd.Timestamp("2023-01-01") <= day <= pd.Timestamp("2026-06-30"):
        return "test"
    return "out_of_window"


def split_series(days: pd.Series) -> pd.Series:
    return days.map(lambda value: split_for_day(value) if pd.notna(value) else "out_of_window").astype("string")


def event_key_strings(frame: pd.DataFrame, key: list[str]) -> pd.Series:
    return frame[key].astype("string").fillna("<NA>").agg("|".join, axis=1)


def compare_field(observed: pd.Series | None, expected: pd.Series, kind: str, name: str,
                  mismatches: dict[str, pd.DataFrame], identity: pd.DataFrame) -> bool:
    if observed is None:
        mismatches[name] = identity[["sample_id"]].assign(reason=f"missing output field: {name}")
        return False
    observed = pd.Series(observed).reset_index(drop=True)
    expected = pd.Series(expected).reset_index(drop=True)
    if kind == "date":
        equal = parse_date(observed, f"observed {name}").eq(parse_date(expected, f"expected {name}"))
        equal = equal | (parse_date(observed, f"observed {name}").isna() & parse_date(expected, f"expected {name}").isna())
    elif kind == "bool":
        try:
            a, b = parse_bool(observed, f"observed {name}", allow_missing=True), parse_bool(expected, f"expected {name}", allow_missing=True)
            equal = a.eq(b) | (a.isna() & b.isna())
        except ValueError:
            equal = pd.Series(False, index=observed.index)
    elif kind == "numeric":
        a = pd.to_numeric(observed, errors="coerce").to_numpy(float)
        b = pd.to_numeric(expected, errors="coerce").to_numpy(float)
        equal = pd.Series(np.isclose(a, b, rtol=1e-9, atol=1e-11, equal_nan=True))
    else:
        a, b = observed.astype("string"), expected.astype("string")
        equal = a.eq(b) | (a.isna() & b.isna())
    if not bool(equal.all()):
        bad = identity.loc[~equal.to_numpy(), ["sample_id"]].copy()
        bad["field"] = name
        bad["observed"] = observed.loc[~equal].astype("string").to_numpy()
        bad["expected"] = expected.loc[~equal].astype("string").to_numpy()
        mismatches[name] = bad
    return bool(equal.all())


def infer_identity(frames: dict[str, pd.DataFrame], ids: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Build one aligned identity frame from whichever main table carries keys."""
    errors: list[str] = []
    identity = pd.DataFrame({"sample_id": ids.astype(str)})
    for field in ["source", "receiver", "announcement", "announcement_day", "period_end"]:
        found: pd.Series | None = None
        first_name: str | None = None
        for name, frame in frames.items():
            if field in frame:
                found, first_name = frame[field], name
                break
        if found is None:
            errors.append(f"identity field missing: {field}")
            identity[field] = pd.NA
            continue
        aligned, align_errors = normalise_key_frame(frame[["sample_id", field]], ids, first_name or field)
        errors.extend(align_errors)
        identity[field] = aligned[field].to_numpy()
        # If a second main table also carries the field, require exact copy.
        for name, frame in frames.items():
            if name == first_name or field not in frame:
                continue
            other, other_errors = normalise_key_frame(frame[["sample_id", field]], ids, name)
            errors.extend(other_errors)
            if not exact_series_equal(aligned[field], other[field], "string"):
                errors.append(f"{field} differs between {first_name} and {name}")
    if identity["announcement_day"].isna().all() and identity["announcement"].notna().any():
        identity["announcement_day"] = parse_date(identity["announcement"], "announcement").to_numpy()
    return identity, errors


def find_graph_field(frames: dict[str, pd.DataFrame], field: str) -> tuple[pd.Series | None, str | None]:
    return first_present(frames, aliases_for(field))


def load_returns_for_development(path: Path, receivers: set[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load only the instruments needed for dev-period target checks.

    No model-ready test target is read here.  The caller supplies only
    training/validation receivers; SPY is used solely to define the calendar
    and the benchmark return.
    """
    usecols = ["Instrument", "Date", "return_decimal"]
    frame = read_csv(path, usecols=usecols)
    frame["Instrument"] = frame["Instrument"].astype("string")
    frame = frame.loc[frame.Instrument.isin(set(receivers) | {"SPY.P"})].copy()
    frame["Date"] = parse_date(frame["Date"], "returns Date", allow_missing=False)
    frame["return_decimal"] = pd.to_numeric(frame["return_decimal"], errors="coerce")
    if frame.duplicated(["Instrument", "Date"]).any():
        raise ValueError("returns has duplicate Instrument/Date keys")
    spy = frame.loc[frame.Instrument.eq("SPY.P") & frame.return_decimal.notna()]
    sessions = pd.Series(pd.DatetimeIndex(sorted(spy.Date.unique())))
    return frame, sessions


def development_target_expectations(frame: pd.DataFrame, sessions: pd.Series,
                                     identity: pd.DataFrame, development_mask: pd.Series) -> dict[str, pd.Series]:
    """Recompute entry/exit and five-session total-return excess for dev rows.

    ``identity`` is filtered before any return lookup.  Consequently this
    function cannot compute a target for a test row, even if one is malformed
    in an output file.
    """
    n = len(identity)
    entry = pd.Series(pd.NaT, index=identity.index, dtype="datetime64[ns]")
    exit_ = pd.Series(pd.NaT, index=identity.index, dtype="datetime64[ns]")
    receiver_ret = pd.Series(np.nan, index=identity.index, dtype=float)
    benchmark_ret = pd.Series(np.nan, index=identity.index, dtype=float)
    excess = pd.Series(np.nan, index=identity.index, dtype=float)
    complete = pd.Series(False, index=identity.index, dtype="boolean")

    dev_index = identity.index[development_mask.to_numpy()]
    if len(dev_index) == 0:
        return {"entry_session": entry, "exit_session": exit_, "receiver_forward_return": receiver_ret,
                "benchmark_forward_return": benchmark_ret, "forward_benchmark_excess": excess,
                "label_complete": complete}

    day = parse_date(identity.loc[dev_index, "announcement_day"], "development announcement_day", allow_missing=False)
    session_values = pd.DatetimeIndex(sessions)
    entry_indices = session_values.searchsorted(day.to_numpy(), side="right")
    valid_entry = entry_indices < len(session_values)
    entry_values = pd.Series(pd.NaT, index=dev_index, dtype="datetime64[ns]")
    entry_values.loc[valid_entry] = session_values[entry_indices[valid_entry]].to_numpy()
    entry.loc[dev_index] = entry_values.to_numpy()
    exit_indices = entry_indices + HORIZON
    valid_exit = valid_entry & (exit_indices < len(session_values))
    exit_values = pd.Series(pd.NaT, index=dev_index, dtype="datetime64[ns]")
    exit_values.loc[valid_exit] = session_values[exit_indices[valid_exit]].to_numpy()
    exit_.loc[dev_index] = exit_values.to_numpy()

    valid_target_rows = dev_index[valid_exit]
    if len(valid_target_rows):
        # Each development row requests exactly five future sessions.  The
        # MultiIndex lookup is exact; it never forward-fills a missing return.
        row_positions = np.flatnonzero(development_mask.to_numpy())[valid_exit]
        future_positions = entry_indices[valid_exit][:, None] + np.arange(1, HORIZON + 1)[None, :]
        future_dates = session_values.to_numpy()[future_positions]
        receiver_values = identity.loc[valid_target_rows, "receiver"].astype(str).to_numpy()[:, None]
        receiver_instruments = np.repeat(receiver_values, HORIZON, axis=1).ravel()
        date_values = future_dates.ravel()
        lookup_index = pd.MultiIndex.from_arrays([receiver_instruments, date_values], names=["Instrument", "Date"])
        indexed = frame.set_index(["Instrument", "Date"])["return_decimal"]
        receiver_matrix = indexed.reindex(lookup_index).to_numpy(float).reshape(-1, HORIZON)
        benchmark_index = pd.MultiIndex.from_arrays([np.repeat("SPY.P", len(date_values)), date_values], names=["Instrument", "Date"])
        benchmark_matrix = indexed.reindex(benchmark_index).to_numpy(float).reshape(-1, HORIZON)
        receiver_ok = np.isfinite(receiver_matrix).all(axis=1)
        benchmark_ok = np.isfinite(benchmark_matrix).all(axis=1)
        receiver_values = np.where(receiver_ok, np.prod(1.0 + receiver_matrix, axis=1) - 1.0, np.nan)
        benchmark_values = np.where(benchmark_ok, np.prod(1.0 + benchmark_matrix, axis=1) - 1.0, np.nan)
        both = receiver_ok & benchmark_ok
        excess_values = np.where(both, receiver_values - benchmark_values, np.nan)
        complete_values = pd.Series(both, index=valid_target_rows, dtype="boolean")
        receiver_ret.loc[valid_target_rows] = receiver_values
        benchmark_ret.loc[valid_target_rows] = benchmark_values
        excess.loc[valid_target_rows] = excess_values
        complete.loc[valid_target_rows] = complete_values

    return {"entry_session": entry, "exit_session": exit_, "receiver_forward_return": receiver_ret,
            "benchmark_forward_return": benchmark_ret, "forward_benchmark_excess": excess,
            "label_complete": complete}


def development_entry_exit_expectations(sessions: pd.Series, identity: pd.DataFrame,
                                        development_mask: pd.Series) -> dict[str, pd.Series]:
    """Derive development entry/exit sessions without reading any return value.

    This is used before boundary purge.  It allows the validator to identify
    validation rows whose five-session horizon would cross 2023-01-01 and to
    verify that those rows were sealed before a future-return lookup occurred.
    Test rows are excluded by ``development_mask`` and therefore never enter
    this calculation.
    """
    entry = pd.Series(pd.NaT, index=identity.index, dtype="datetime64[ns]")
    exit_ = pd.Series(pd.NaT, index=identity.index, dtype="datetime64[ns]")
    dev_index = identity.index[development_mask.to_numpy()]
    if len(dev_index) == 0:
        return {"entry_session": entry, "exit_session": exit_}
    days = parse_date(identity.loc[dev_index, "announcement_day"], "development announcement_day", allow_missing=False)
    session_values = pd.DatetimeIndex(sessions)
    entry_indices = session_values.searchsorted(days.to_numpy(), side="right")
    valid_entry = entry_indices < len(session_values)
    entry_values = pd.Series(pd.NaT, index=dev_index, dtype="datetime64[ns]")
    entry_values.loc[valid_entry] = session_values[entry_indices[valid_entry]].to_numpy()
    entry.loc[dev_index] = entry_values.to_numpy()
    exit_indices = entry_indices + HORIZON
    valid_exit = valid_entry & (exit_indices < len(session_values))
    exit_values = pd.Series(pd.NaT, index=dev_index, dtype="datetime64[ns]")
    exit_values.loc[valid_exit] = session_values[exit_indices[valid_exit]].to_numpy()
    exit_.loc[dev_index] = exit_values.to_numpy()
    return {"entry_session": entry, "exit_session": exit_}


def resolve_clean_path(summary: dict, names: Iterable[str], fallback_root: Path, filename: str) -> Path:
    path = resolve_summary_input(summary, names)
    if path is not None and path.exists():
        return path
    candidates = [p / filename for p in sorted(fallback_root.glob("*/"), key=lambda p: p.name, reverse=True)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    direct = fallback_root / filename
    return direct


def price_field_aliases(prefix: str, field: str) -> list[str]:
    return [f"{prefix}_{field}", f"{prefix}_price_{field}", f"{prefix}{field}"]


def price_expectations(price_path: Path, receivers: pd.Series, sessions: pd.Series,
                       mask: pd.Series, target_frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return exact clean-price fields for development rows only."""
    n = len(target_frame)
    result: dict[str, pd.Series] = {}
    for field in ["price_row_available", "close", "volume", "quoted_spread_bps", "dollar_volume",
                  "dollar_volume_source", "has_close", "has_volume", "has_two_sided_quote",
                  "in_sp500_that_day", "price_adjustments"]:
        result[field] = pd.Series(np.nan, index=target_frame.index, dtype=object)
    dev_index = target_frame.index[mask.to_numpy()]
    if len(dev_index) == 0:
        return result
    usecols = ["Instrument", "Date", "TRDPRC_1", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume",
               "dollar_volume_source", "has_close", "has_volume", "has_two_sided_quote",
               "in_sp500_that_day", "price_adjustments"]
    prices = read_csv(price_path, usecols=usecols)
    prices["Instrument"] = prices["Instrument"].astype("string")
    prices["Date"] = parse_date(prices["Date"], "price Date", allow_missing=False)
    prices = prices.loc[prices.Instrument.isin(set(receivers.astype(str)))].copy()
    if prices.duplicated(["Instrument", "Date"]).any():
        raise ValueError("prices has duplicate Instrument/Date keys")
    index = prices.set_index(["Instrument", "Date"])
    key = pd.MultiIndex.from_arrays([
        receivers.loc[dev_index].astype(str).to_numpy(),
        parse_date(sessions.loc[dev_index], "entry session", allow_missing=True).to_numpy(),
    ], names=["Instrument", "Date"])
    found = index.reindex(key)
    found.index = dev_index
    for output, source in {
        "price_row_available": None, "close": "TRDPRC_1", "volume": "ACVOL_UNS",
        "quoted_spread_bps": "quoted_spread_bps", "dollar_volume": "dollar_volume",
        "dollar_volume_source": "dollar_volume_source", "has_close": "has_close",
        "has_volume": "has_volume", "has_two_sided_quote": "has_two_sided_quote",
        "in_sp500_that_day": "in_sp500_that_day", "price_adjustments": "price_adjustments",
    }.items():
        if output == "price_row_available":
            # Availability is defined by exact instrument/date key presence,
            # independently of whether the row's close is missing.  Clean
            # provenance can retain an all-NA row, which is still a row.
            result[output].loc[dev_index] = index.index.get_indexer(key) >= 0
        else:
            result[output].loc[dev_index] = found[source].to_numpy()
    return result


def mismatch_extract(path: Path, name: str, frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    output = path / name
    frame.to_csv(output, index=False)
    return name


def forbidden_feature_columns(columns: Iterable[str]) -> list[str]:
    forbidden: list[str] = []
    for column in columns:
        lower = str(column).lower()
        # ``residual_correlation`` and its old derived network features are
        # intentionally rejected in this graph v2 model-ready layer.
        is_old_residual = "residual_correlation" in lower or lower in {
            "abs_residual_correlation", "network_signal", "network_signal_abs",
            "neighbor_abs_corr_rank",
        }
        is_current_trbc = "trbc" in lower or "industry" in lower or "sector" in lower
        is_future = any(token in lower for token in (
            "entry_", "entryday", "entry_day", "forward", "target", "label", "exit_",
            "ex_post", "expost", "diagnostic", "overlap", "actuals_through",
        ))
        if is_old_residual or is_current_trbc or is_future:
            forbidden.append(str(column))
    return forbidden


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate analyst graph model-ready v1 output")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="model-ready run directory; defaults to newest completed run")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run(MODEL_ROOT).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    errors: list[str] = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # keep a run-specific audit even for malformed summary JSON
            errors.append(f"summary.json unreadable: {exc}")
    else:
        errors.append("summary.json missing")

    table_frames: dict[str, pd.DataFrame] = {}
    table_paths: dict[str, Path] = {}
    for logical in ("features", "metadata", "targets", "eligibility", "execution", "feature_missingness", "split_counts"):
        path = table_path(run_dir, logical, summary)
        if path is None:
            if logical in {"features", "metadata", "targets", "eligibility"}:
                errors.append(f"required table missing: {logical}")
            continue
        table_paths[logical] = path
        try:
            table_frames[logical] = read_csv(path)
        except Exception as exc:
            errors.append(f"{logical} unreadable: {exc}")

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "run_dir": rel_path(run_dir),
        "expected_rows": EXPECTED_ROWS,
        "table_rows": {name: len(frame) for name, frame in table_frames.items()},
        "errors": errors,
    }
    mismatches: dict[str, pd.DataFrame] = {}

    required = ["features", "metadata", "targets", "eligibility"]
    checks["four_main_tables_present"] = all(name in table_frames for name in required)
    checks["four_main_tables_have_111128_rows"] = bool(
        checks["four_main_tables_present"] and all(len(table_frames[name]) == EXPECTED_ROWS for name in required)
    )

    ids = pd.Series([], dtype="string")
    if "features" in table_frames and "sample_id" in table_frames["features"]:
        ids = table_frames["features"]["sample_id"].astype("string")
    elif "metadata" in table_frames and "sample_id" in table_frames["metadata"]:
        ids = table_frames["metadata"]["sample_id"].astype("string")
    checks["all_main_sample_ids_unique"] = bool(ids.size and ids.notna().all() and not ids.duplicated().any())
    aligned: dict[str, pd.DataFrame] = {}
    alignment_errors: list[str] = []
    if ids.size:
        for name in required + ["execution"]:
            if name not in table_frames:
                continue
            aligned[name], table_errors = normalise_key_frame(table_frames[name], ids, name)
            alignment_errors.extend(table_errors)
    checks["all_main_sample_id_sets_equal"] = bool(ids.size and not alignment_errors)
    details["alignment_errors"] = alignment_errors

    identity, identity_errors = infer_identity(aligned, ids) if ids.size else (pd.DataFrame(), ["no sample_id anchor"])
    details["identity_errors"] = identity_errors
    checks["identity_fields_available_and_copied"] = bool(ids.size and not identity_errors)

    # The feature schema is checked before any data-dependent feature test.
    feature_frame = aligned.get("features", pd.DataFrame())
    summary_features = summary.get("feature_columns")
    if isinstance(summary_features, list):
        feature_columns = [str(value) for value in summary_features]
        schema_present = set(feature_columns).issubset(set(feature_frame.columns))
        checks["feature_schema_matches_summary"] = bool(schema_present and
                                                         [c for c in feature_columns if c not in feature_frame] == [])
        observed_non_id = [c for c in feature_frame.columns if c != "sample_id" and c in feature_columns]
        checks["feature_schema_order_matches_summary"] = observed_non_id == feature_columns
    else:
        feature_columns = [c for c in feature_frame.columns if c != "sample_id" and c not in IDENTITY_COLUMNS]
        checks["feature_schema_matches_summary"] = True
        checks["feature_schema_order_matches_summary"] = True
    forbidden = forbidden_feature_columns(feature_columns)
    checks["feature_whitelist_excludes_entry_future_label_exit_expost_old_residual_current_trbc"] = not forbidden
    checks["engineered_feature_values_have_no_infinity"] = True
    if feature_columns and set(feature_columns).issubset(feature_frame.columns):
        numeric = feature_frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        checks["engineered_feature_values_have_no_infinity"] = bool(
            (np.isfinite(numeric) | np.isnan(numeric)).all()
        )
    details["feature_columns"] = feature_columns
    details["forbidden_feature_columns"] = forbidden


    # Resolve source versions before value checks.  Every path is retained in
    # the report, even when a later check cannot run.
    graph_dir: Path | None = None
    graph_edges = pd.DataFrame()
    graph_status = pd.DataFrame()
    graph_summary: dict[str, Any] = {}
    spans_path = resolve_summary_input(summary, ("membership_intervals", "corrected_spans", "spans"), SPANS_DEFAULT)
    returns_path = resolve_clean_path(summary, ("returns_clean", "returns", "clean_returns"), ROOT / "data" / "clean" / "v2", "returns.csv")
    prices_path = resolve_clean_path(summary, ("prices_clean", "prices", "clean_prices"), ROOT / "data" / "clean" / "v3", "prices.csv")
    try:
        graph_dir = discover_graph_run(summary)
        graph_edges = read_csv(graph_dir / "edges.csv")
        graph_status = read_csv(graph_dir / "event_graph_status.csv")
        graph_summary = json.loads((graph_dir / "summary.json").read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"graph input unavailable: {exc}")
    details["graph_run_dir"] = rel_path(graph_dir) if graph_dir else None
    details["returns_path"] = rel_path(returns_path) if returns_path else None
    details["prices_path"] = rel_path(prices_path) if prices_path else None
    details["spans_path"] = rel_path(spans_path) if spans_path else None

    if "reason" in graph_edges:
        graph_ok = graph_edges.loc[graph_edges["reason"].astype("string").eq("ok")].copy()
    else:
        graph_ok = pd.DataFrame()
    graph_ok_ids = graph_ok[EDGE_KEY].astype("string").fillna("<NA>").agg("|".join, axis=1) if set(EDGE_KEY).issubset(graph_ok.columns) else pd.Series([], dtype="string")
    checks["graph_v2_ok_edges_are_111128"] = len(graph_ok) == EXPECTED_ROWS
    checks["graph_ok_edge_keys_unique"] = bool(len(graph_ok) and not graph_ok.duplicated(EDGE_KEY).any())
    graph_output_hash_checks: dict[str, bool] = {}
    if graph_dir is not None and isinstance(graph_summary.get("outputs"), dict):
        for name, expected_hash in graph_summary["outputs"].items():
            graph_file = graph_dir / Path(str(name)).name
            if graph_file.exists():
                try:
                    graph_output_hash_checks[str(name)] = sha256_file(graph_file) == str(expected_hash)
                except OSError:
                    graph_output_hash_checks[str(name)] = False
            else:
                graph_output_hash_checks[str(name)] = False
    checks["graph_v2_recorded_output_hashes_match"] = bool(
        graph_output_hash_checks and all(graph_output_hash_checks.values())
    )

    output_edge = identity[EDGE_KEY].astype("string").fillna("<NA>").agg("|".join, axis=1) if set(EDGE_KEY).issubset(identity.columns) else pd.Series([], dtype="string")
    checks["each_row_strictly_corresponds_to_graph_v2_reason_ok_edge"] = bool(
        len(output_edge) == EXPECTED_ROWS and len(graph_ok_ids) == EXPECTED_ROWS and
        set(output_edge) == set(graph_ok_ids) and not output_edge.duplicated().any()
    )
    if len(graph_ok) and len(identity):
        graph_join = graph_ok.set_index(EDGE_KEY).reindex(
            pd.MultiIndex.from_frame(identity[EDGE_KEY])
        ).reset_index(drop=True)
        for field in GRAPH_FIELDS:
            observed, observed_name = find_graph_field(aligned, field)
            if field == "snapshot":
                expected = graph_join[field]
                checks[f"graph_{field}_copied_without_drift"] = compare_field(
                    observed, expected, "date", observed_name or field, mismatches, identity)
            else:
                expected = pd.to_numeric(graph_join[field], errors="coerce")
                checks[f"graph_{field}_copied_without_drift"] = compare_field(
                    observed, expected, "numeric", observed_name or field, mismatches, identity)
    else:
        for field in GRAPH_FIELDS:
            checks[f"graph_{field}_copied_without_drift"] = False

    # Graph time and structural constraints are verified from the graph v2
    # edge file, then repeated on the model-ready rows through the copied keys.
    if len(graph_ok):
        graph_day = parse_date(graph_ok["announcement"], "graph announcement", allow_missing=False) if "announcement" in graph_ok else pd.Series(dtype="datetime64[ns]")
        graph_snapshot_day = parse_date(graph_ok["snapshot"], "graph snapshot", allow_missing=False) if "snapshot" in graph_ok else pd.Series(dtype="datetime64[ns]")
        graph_age = pd.to_numeric(graph_ok["snapshot_age_days"], errors="coerce") if "snapshot_age_days" in graph_ok else pd.Series(np.nan, index=graph_ok.index)
        checks["snapshot_strictly_before_announcement"] = bool((graph_snapshot_day < graph_day).all())
        checks["snapshot_age_at_most_120_days"] = bool(graph_age.notna().all() and graph_age.le(MAX_GRAPH_AGE_DAYS).all())
        checks["common_brokers_at_least_3"] = bool(pd.to_numeric(graph_ok["common_brokers"], errors="coerce").ge(3).all()) if "common_brokers" in graph_ok else False
        counts = graph_ok.groupby(EVENT_KEY, dropna=False).size()
        checks["maximum_5_receivers_per_event"] = bool(counts.le(5).all())
        checks["no_self_edges"] = bool((graph_ok.source.astype(str) != graph_ok.receiver.astype(str)).all())
    else:
        checks.update({
            "snapshot_strictly_before_announcement": False,
            "snapshot_age_at_most_120_days": False,
            "common_brokers_at_least_3": False,
            "maximum_5_receivers_per_event": False,
            "no_self_edges": False,
        })

    # Corrected span membership is independently evaluated from dates in the
    # output.  It is intentionally not copied from graph status or eligibility.
    spans = pd.DataFrame()
    if spans_path is not None and spans_path.exists():
        try:
            spans = load_spans(spans_path)
        except Exception as exc:
            errors.append(f"corrected spans unavailable: {exc}")
    checks["corrected_spans_shape_and_ev_hc_exclusion"] = bool(
        len(spans) == 781 and spans.ric.nunique() == 781 and
        "EVHC.N^L16" not in set(spans.ric.astype(str)) and
        "HC.N" not in set(spans.ric.astype(str))
    ) if len(spans) else False
    announcement_days = parse_date(identity["announcement_day"], "output announcement_day") if len(identity) else pd.Series(dtype="datetime64[ns]")
    if len(identity) and announcement_days.isna().any() and identity["announcement"].notna().any():
        announcement_days = parse_date(identity["announcement"], "output announcement", allow_missing=False)
    # Estimate provenance is metadata, but its point-in-time date is still
    # checked independently when the builder persists the newly explicit
    # estimate_snapshot/estimate_age fields.
    estimate_snapshot, estimate_snapshot_name = first_present(aligned, [
        "estimate_snapshot", "consensus_snapshot", "estimate_asof",
    ])
    estimate_age, estimate_age_name = first_present(aligned, [
        "estimate_snapshot_age_days", "estimate_age_days", "consensus_snapshot_age_days",
    ])
    if estimate_snapshot is not None:
        estimate_dates = parse_date(estimate_snapshot, "estimate snapshot")
        checks["estimate_snapshot_strictly_before_announcement"] = bool(
            estimate_dates.lt(announcement_days).fillna(False).all()
        ) if len(identity) else False
        if estimate_age is not None:
            expected_estimate_age = (announcement_days - estimate_dates).dt.days.astype(float)
            checks["estimate_snapshot_age_recomputed"] = values_close(
                pd.to_numeric(estimate_age, errors="coerce"), expected_estimate_age
            )
        else:
            checks["estimate_snapshot_age_recomputed"] = True
    else:
        checks["estimate_snapshot_strictly_before_announcement"] = True
        checks["estimate_snapshot_age_recomputed"] = True
    # Derive the split masks before membership and target checks.  The masks
    # are based solely on the announcement calendar day and contain no target
    # or price information.
    expected_split = split_series(announcement_days) if len(identity) else pd.Series(dtype="string")
    development_mask = expected_split.isin(["training", "validation"])
    test_mask = expected_split.eq("test")
    if len(spans) and len(identity):
        source_announcement_expected = membership_flags(spans, identity.source, announcement_days)
        receiver_announcement_expected = membership_flags(spans, identity.receiver, announcement_days)
        source_observed, source_name = find_graph_field(aligned, "source_in_index_on_announcement")
        if source_observed is None:
            source_observed, source_name = first_present(aligned, ["source_in_index_on_announcement", "source_announcement_member"])
        receiver_observed, receiver_name = first_present(aligned, [
            "receiver_in_index_on_announcement", "receiver_announcement_member",
        ])
        checks["source_announcement_membership_recomputed"] = compare_field(
            source_observed, source_announcement_expected, "bool", source_name or "source_in_index_on_announcement", mismatches, identity)
        checks["receiver_announcement_membership_recomputed"] = compare_field(
            receiver_observed, receiver_announcement_expected, "bool", receiver_name or "receiver_in_index_on_announcement", mismatches, identity)
        entry_series, entry_name = first_present(aligned, ["entry_session"])
        if entry_series is None and "targets" in aligned:
            entry_series, entry_name = first_present({"targets": aligned["targets"]}, ["entry_session"])
        # Keep test entry dates out of the independent membership calculation.
        # A sealed target table may contain no entry date at all; only
        # development rows are needed for this check.
        entry_series_dev = pd.Series(pd.NaT, index=identity.index, dtype="datetime64[ns]")
        if entry_series is not None:
            entry_series_dev.loc[development_mask] = parse_date(
                entry_series.loc[development_mask], "development entry_session", allow_missing=True
            ).to_numpy()
        receiver_entry_expected = membership_flags(spans, identity.receiver, entry_series_dev)
        receiver_entry_observed, receiver_entry_name = first_present(aligned, [
            "receiver_in_index_at_entry", "receiver_in_index_on_entry", "receiver_entry_member",
        ])
        # Entry membership is a development eligibility input.  Test target
        # rows are sealed, so do not inspect any test entry date while
        # independently recomputing this flag.
        checks["receiver_entry_membership_recomputed"] = compare_field(
            receiver_entry_observed.loc[development_mask] if receiver_entry_observed is not None else None,
            receiver_entry_expected.loc[development_mask], "bool", receiver_entry_name or "receiver_in_index_at_entry",
            mismatches, identity.loc[development_mask].reset_index(drop=True))
        source_entry_observed, source_entry_name = first_present(aligned, [
            "source_in_index_at_entry", "source_entry_member",
        ])
        if source_entry_observed is None:
            checks["source_entry_membership_recomputed_if_present"] = True
            details["source_entry_membership"] = "no source entry flag in output; receiver entry flag is the execution eligibility field"
        else:
            source_entry_expected = membership_flags(spans, identity.source, entry_series_dev)
            checks["source_entry_membership_recomputed_if_present"] = compare_field(
                source_entry_observed.loc[development_mask], source_entry_expected.loc[development_mask], "bool",
                source_entry_name or "source_in_index_at_entry", mismatches,
                identity.loc[development_mask].reset_index(drop=True))
    else:
        checks.update({
            "source_announcement_membership_recomputed": False,
            "receiver_announcement_membership_recomputed": False,
            "receiver_entry_membership_recomputed": False,
            "source_entry_membership_recomputed_if_present": False,
        })

    # Independent split and event-level purge recomputation.  For test rows,
    # only the announcement date is used; no target entry/exit value is read.
    elig = aligned.get("eligibility", pd.DataFrame())
    observed_split = elig["split"].astype("string") if "split" in elig else None
    checks["split_assignment_recomputed"] = bool(
        observed_split is not None and exact_series_equal(observed_split, expected_split, "string")
    )
    if len(identity):
        event_split = pd.DataFrame({**{key: identity[key] for key in EVENT_KEY}, "split": expected_split})
        split_nunique = event_split.groupby(EVENT_KEY, dropna=False).split.nunique(dropna=False)
        checks["event_split_single_valued"] = bool(split_nunique.le(1).all())
    else:
        checks["event_split_single_valued"] = False

    details["development_rows"] = int(development_mask.sum())
    details["test_rows"] = int(test_mask.sum())

    # The target table is checked first for test sealing.  Values from test
    # rows are never passed to a return or price calculation below.
    targets = aligned.get("targets", pd.DataFrame())
    target_reason_columns = [c for c in targets.columns if str(c).lower() in {
        "reason", "target_reason", "label_reason", "target_status", "label_status",
    } or str(c).lower().endswith(("_reason", "_status"))]
    # The builder intentionally keeps split and entry/exit timing in targets.
    # Only the three forward return value columns are sealed; label_available
    # is a boolean availability flag and must be false rather than NaN.
    target_value_aliases = {
        "receiver_forward_return", "benchmark_forward_return", "forward_benchmark_excess",
        "target_receiver_forward_return", "target_benchmark_forward_return",
        "target_forward_benchmark_excess", "target_excess_return", "excess_return",
    }
    seal_fields = [c for c in targets.columns if c in target_value_aliases]
    label_seal_col = next((c for c in ("label_available", "label_complete", "target_available") if c in targets), None)
    if len(test_mask) and len(targets):
        test_targets = targets.loc[test_mask.to_numpy(), seal_fields] if seal_fields else pd.DataFrame(index=targets.index[test_mask.to_numpy()])
        checks["test_target_value_fields_all_nan"] = bool(test_targets.isna().all(axis=None))
        reason_checks = []
        for col in target_reason_columns:
            reason_checks.append(targets.loc[test_mask.to_numpy(), col].astype("string").eq(TARGET_REASON).all())
        checks["test_target_reason_is_test_target_sealed"] = bool(reason_checks and all(reason_checks))
        if label_seal_col is not None:
            test_label = parse_bool(
                targets.loc[test_mask.to_numpy(), label_seal_col],
                f"test {label_seal_col}", allow_missing=True,
            )
            checks["test_target_label_available_false"] = bool(
                test_label.notna().all() and test_label.eq(False).all()
            )
        else:
            checks["test_target_label_available_false"] = False
    else:
        checks["test_target_value_fields_all_nan"] = False
        checks["test_target_reason_is_test_target_sealed"] = False
        checks["test_target_label_available_false"] = False
    details["target_reason_columns"] = target_reason_columns
    details["target_seal_fields"] = seal_fields

    # Build the development session calendar and identify boundary-purged
    # rows before looking up any return.  A validation event whose exit would
    # be on/after 2023-01-01 is therefore excluded from the target lookup.
    return_frame = pd.DataFrame()
    sessions = pd.Series([], dtype="datetime64[ns]")
    expected_entry_exit: dict[str, pd.Series] = {
        "entry_session": pd.Series(pd.NaT, index=identity.index, dtype="datetime64[ns]"),
        "exit_session": pd.Series(pd.NaT, index=identity.index, dtype="datetime64[ns]"),
    }
    expected_purge = pd.Series(False, index=identity.index, dtype="boolean")
    target_calc_mask = development_mask.copy()
    if development_mask.any() and returns_path is not None and returns_path.exists():
        try:
            dev_receivers = set(identity.loc[development_mask, "receiver"].astype(str))
            return_frame, sessions = load_returns_for_development(returns_path, dev_receivers)
            expected_entry_exit = development_entry_exit_expectations(sessions, identity, development_mask)
            boundaries = expected_split.map({
                "training": pd.Timestamp("2021-01-01"),
                "validation": pd.Timestamp("2023-01-01"),
            })
            dev_boundaries = boundaries.loc[development_mask]
            expected_purge.loc[development_mask] = (
                expected_entry_exit["entry_session"].loc[development_mask].ge(dev_boundaries.to_numpy()) |
                expected_entry_exit["exit_session"].loc[development_mask].ge(dev_boundaries.to_numpy())
            ).astype("boolean").to_numpy()
            target_calc_mask = development_mask & ~expected_purge
        except Exception as exc:
            errors.append(f"development session calendar failed: {exc}")
            target_calc_mask = development_mask.copy()

    # Boundary-purged rows are retained in the master tables but have their
    # future target sealed before any lookup.  Their reason is deliberately
    # distinct from an incomplete/missing-return target.
    boundary_reason_checks = []
    if expected_purge.any() and len(targets):
        boundary_targets = targets.loc[expected_purge.to_numpy(), seal_fields] if seal_fields else pd.DataFrame(index=targets.index[expected_purge.to_numpy()])
        checks["boundary_purged_target_value_fields_all_nan"] = bool(boundary_targets.isna().all(axis=None))
        for col in target_reason_columns:
            boundary_reason_checks.append(
                targets.loc[expected_purge.to_numpy(), col].astype("string").eq("boundary_purged").all()
            )
        checks["boundary_purged_target_reason_is_boundary_purged"] = bool(
            boundary_reason_checks and all(boundary_reason_checks)
        )
        if label_seal_col is not None:
            boundary_label = parse_bool(
                targets.loc[expected_purge.to_numpy(), label_seal_col],
                f"boundary {label_seal_col}", allow_missing=True,
            )
            checks["boundary_purged_target_label_available_false"] = bool(
                boundary_label.notna().all() and boundary_label.eq(False).all()
            )
        else:
            checks["boundary_purged_target_label_available_false"] = False
    else:
        # The count itself is still recorded; if no boundary rows are expected
        # the sealing invariant is vacuously true, while a missing reason
        # column remains a structural failure whenever rows need sealing.
        checks["boundary_purged_target_value_fields_all_nan"] = True
        checks["boundary_purged_target_reason_is_boundary_purged"] = True
        checks["boundary_purged_target_label_available_false"] = True

    # Recompute only training and validation targets.  This intentionally
    # does not inspect targets.loc[test_mask] beyond the NaN/reason assertions.
    target_checks = []
    expected_targets: dict[str, pd.Series] = {}
    if target_calc_mask.any() and len(targets) and not return_frame.empty:
        try:
            # This mask excludes every boundary-purged validation row and all
            # test rows.  No future return is requested outside it.
            expected_targets = development_target_expectations(
                return_frame, sessions, identity, target_calc_mask
            )
            # Entry/exit are timing fields retained for every development row,
            # including boundary-purged rows.  Return values and availability
            # remain NaN/false for the purged rows.
            expected_targets["entry_session"] = expected_entry_exit["entry_session"]
            expected_targets["exit_session"] = expected_entry_exit["exit_session"]
            for field, expected in expected_targets.items():
                if field not in targets:
                    # Support target_foo aliases while preserving a strict
                    # check that all three returns are actually persisted.
                    aliases = [f"target_{field}", field.replace("forward_", "target_")]
                    if field == "forward_benchmark_excess":
                        aliases += ["target_excess_return", "excess_return", "target_benchmark_excess"]
                    if field == "label_complete":
                        aliases += ["target_available", "label_available"]
                    observed_name = next((x for x in aliases if x in targets), None)
                else:
                    observed_name = field
                if observed_name is None:
                    target_checks.append(False)
                    continue
                target_checks.append(compare_field(
                    targets.loc[development_mask.to_numpy(), observed_name],
                    expected.loc[development_mask],
                    "bool" if field == "label_complete" else "date" if field.endswith("session") else "numeric",
                    observed_name, mismatches, identity.loc[development_mask].reset_index(drop=True),
                ))
            details["development_target_rows_recomputed"] = int(target_calc_mask.sum())
            details["development_sessions"] = int(len(sessions))
        except Exception as exc:
            errors.append(f"development target recomputation failed: {exc}")
            target_checks = [False]
    else:
        target_checks = [False]
    checks["development_entry_exit_forward_returns_excess_recomputed"] = bool(target_checks and all(target_checks))
    if target_reason_columns and expected_targets:
        dev_reason_checks = []
        # A complete development target has an explicit ``ok`` reason.  An
        # incomplete development target must carry a non-empty reason, but
        # the exact vendor-missing subtype is implementation-specific.
        complete_dev = expected_targets["label_complete"].loc[development_mask]
        for col in target_reason_columns:
            observed_reason = targets.loc[development_mask.to_numpy(), col].astype("string").fillna("").str.strip()
            complete_values = complete_dev.fillna(False).astype(bool).to_numpy()
            complete_reason = observed_reason.isin({"ok", "complete", "label_complete"}).to_numpy()
            nonempty_reason = observed_reason.ne("").to_numpy()
            # Apply each rule only to the rows to which it belongs.  Testing
            # the conjunction over the full vector would incorrectly require
            # incomplete rows to say ``ok`` and complete rows to carry an
            # incomplete reason at the same time.
            dev_reason_checks.append(bool(complete_reason[complete_values].all()))
            dev_reason_checks.append(bool(nonempty_reason[~complete_values].all()))
        checks["development_target_reason_matches_completeness"] = bool(dev_reason_checks and all(dev_reason_checks))
    else:
        checks["development_target_reason_matches_completeness"] = False

    # Purge uses independent calendar sessions only for development rows.  A
    # test row is known to be retained by split policy, so no test exit date is
    # computed or accessed.
    purge_observed = elig["boundary_purged"] if "boundary_purged" in elig else None
    checks["event_level_boundary_purge_recomputed"] = bool(
        purge_observed is not None and exact_series_equal(purge_observed, expected_purge, "bool")
    )
    if len(identity):
        purge_group = pd.DataFrame({**{key: identity[key] for key in EVENT_KEY}, "purged": expected_purge})
        purge_nunique = purge_group.groupby(EVENT_KEY, dropna=False).purged.nunique(dropna=False)
        checks["event_boundary_purge_single_valued"] = bool(purge_nunique.le(1).all())
    else:
        checks["event_boundary_purge_single_valued"] = False
    if expected_purge.any():
        purge_rows = identity.loc[expected_purge.to_numpy(), EVENT_KEY + ["sample_id"]].copy()
        purge_rows["split"] = expected_split.loc[expected_purge].to_numpy()
        purge_rows["purge_reason"] = "entry_or_exit_at_or_after_next_split_boundary"
        mismatches["boundary_purged_rows"] = purge_rows
    details["boundary_purged_rows"] = int(expected_purge.sum())
    details["boundary_purged_events"] = int(identity.loc[expected_purge.to_numpy(), EVENT_KEY].drop_duplicates().shape[0]) if len(identity) else 0

    # Entry eligibility and missingness consistency.  Entry flags are compared
    # from exact clean price keys and corrected spans; target completeness is a
    # separate condition and never makes a row entry-ineligible.
    expected_entry_eligible = pd.Series(pd.NA, index=identity.index, dtype="boolean")
    receiver_entry_member_for_eligibility = pd.Series(pd.NA, index=identity.index, dtype="boolean")
    price_expected_for_eligibility: dict[str, pd.Series] = {}
    if len(identity) and len(spans) and prices_path is not None and prices_path.exists():
        try:
            # Read target entry dates only from development rows.  Test target
            # values remain sealed even if a malformed file happens to contain
            # them; the test assertion above is the only permitted inspection.
            # Use the independently derived development calendar.  This also
            # covers boundary-purged rows whose target table is intentionally
            # all-NaN; no sealed target entry date is consulted.
            target_entry = expected_entry_exit["entry_session"].copy()
            receiver_entry_member = membership_flags(spans, identity.receiver, target_entry)
            # Recompute clean price flags only for development rows.  Test
            # rows are passed as an all-NaT session series and therefore never
            # cause a future price row to be looked up.
            dev_entry = target_entry.where(development_mask, pd.NaT)
            development_receivers = identity.receiver.where(development_mask, pd.NA)
            price_expected = price_expectations(prices_path, development_receivers, dev_entry, development_mask, identity)
            expected_entry_eligible = receiver_entry_member & parse_bool(
                price_expected["price_row_available"].fillna(False), "entry price row", allow_missing=False
            ).astype("boolean") & parse_bool(
                price_expected["has_close"].fillna(False), "entry close", allow_missing=False
            ).astype("boolean")
            receiver_entry_member_for_eligibility = receiver_entry_member
            price_expected_for_eligibility = price_expected
            entry_observed, entry_name = first_present(aligned, ["entry_trade_eligible"])
            checks["entry_trade_eligibility_recomputed"] = compare_field(
                entry_observed.loc[development_mask] if entry_observed is not None else None,
                expected_entry_eligible.loc[development_mask], "bool", entry_name or "entry_trade_eligible",
                mismatches, identity.loc[development_mask].reset_index(drop=True))
            checks["receiver_entry_membership_used_for_eligibility"] = bool(
                entry_observed is not None and exact_series_equal(
                    first_present(aligned, ["receiver_in_index_at_entry"])[0].loc[development_mask]
                    if first_present(aligned, ["receiver_in_index_at_entry"])[0] is not None
                    else pd.Series(pd.NA, index=development_mask[development_mask].index),
                    receiver_entry_member.loc[development_mask], "bool")
            )
            price_mismatch_frame = identity[["sample_id"]].copy()
            price_compared = 0
            for field, kind in [
                ("price_row_available", "bool"), ("close", "numeric"), ("volume", "numeric"),
                ("quoted_spread_bps", "numeric"), ("dollar_volume", "numeric"),
                ("dollar_volume_source", "string"), ("has_close", "bool"), ("has_volume", "bool"),
                ("has_two_sided_quote", "bool"), ("in_sp500_that_day", "bool"),
                ("price_adjustments", "string"),
            ]:
                observed, observed_name = first_present(aligned, price_field_aliases("entry", field))
                if observed is None:
                    continue
                price_compared += 1
                compare_field(
                    observed.loc[development_mask] if observed is not None else None,
                    price_expected[field].loc[development_mask], kind, observed_name or f"entry_{field}",
                    mismatches, identity.loc[development_mask].reset_index(drop=True)
                )
            checks["exact_entry_price_provenance_recomputed"] = bool(price_compared >= 1 and all(
                name not in mismatches for name in [
                    x for x in mismatches if x.startswith("entry_")
                ]
            ))
            details["entry_price_fields_compared"] = price_compared
        except Exception as exc:
            errors.append(f"entry eligibility/price recomputation failed: {exc}")
            checks["entry_trade_eligibility_recomputed"] = False
            checks["receiver_entry_membership_used_for_eligibility"] = False
            checks["exact_entry_price_provenance_recomputed"] = False
    else:
        checks["entry_trade_eligibility_recomputed"] = False
        checks["receiver_entry_membership_used_for_eligibility"] = False
        checks["exact_entry_price_provenance_recomputed"] = False

    # Once the independent SPY calendar is available, replace any provisional
    # entry-membership comparison that used a copied metadata date.  This
    # keeps boundary-purged rows correct even when their target table has been
    # sealed and has no entry_session value.
    if len(spans) and expected_entry_exit and development_mask.any():
        independent_entry_member = membership_flags(
            spans, identity.receiver, expected_entry_exit["entry_session"]
        )
        receiver_entry_observed, receiver_entry_name = first_present(aligned, [
            "receiver_in_index_at_entry", "receiver_in_index_on_entry", "receiver_entry_member",
        ])
        checks["receiver_entry_membership_recomputed"] = compare_field(
            receiver_entry_observed.loc[development_mask] if receiver_entry_observed is not None else None,
            independent_entry_member.loc[development_mask], "bool",
            receiver_entry_name or "receiver_in_index_at_entry", mismatches,
            identity.loc[development_mask].reset_index(drop=True)
        )
        source_entry_observed, source_entry_name = first_present(aligned, [
            "source_in_index_at_entry", "source_entry_member",
        ])
        if source_entry_observed is not None:
            independent_source_entry_member = membership_flags(
                spans, identity.source, expected_entry_exit["entry_session"]
            )
            checks["source_entry_membership_recomputed_if_present"] = compare_field(
                source_entry_observed.loc[development_mask],
                independent_source_entry_member.loc[development_mask], "bool",
                source_entry_name or "source_in_index_at_entry", mismatches,
                identity.loc[development_mask].reset_index(drop=True)
            )

    # Label, core-feature and supervised eligibility consistency.  We do not
    # assume a specific reason-string order; the component booleans are the
    # independent source of truth.
    label_available, label_name = first_present(aligned, ["supervised_label_available", "label_available", "target_available"])
    label_expected = pd.Series(False, index=identity.index, dtype="boolean")
    if expected_targets:
        label_expected.loc[development_mask] = expected_targets["label_complete"].loc[development_mask]
    checks["label_availability_recomputed"] = compare_field(
        label_available, label_expected, "bool", label_name or "supervised_label_available", mismatches, identity)

    if feature_columns and set(feature_columns).issubset(feature_frame.columns):
        feature_missing_count = feature_frame[feature_columns].isna().sum(axis=1).astype(int)
        checks["feature_missing_count_recomputed"] = compare_field(
            first_present(aligned, ["feature_missing_count"])[0], feature_missing_count, "numeric",
            "feature_missing_count", mismatches, identity)
        core = summary.get("core_features")
        if not isinstance(core, list) or not core:
            core = [c for c in feature_columns if c in {"standardized_surprise", "jaccard_coverage", "jaccard_rec_weighted", "source_mom_20", "receiver_mom_20"}]
        core = [c for c in core if c in feature_frame]
        feature_core_expected = feature_frame[core].notna().all(axis=1) if core else pd.Series(False, index=identity.index)
        checks["feature_core_availability_recomputed"] = compare_field(
            first_present(aligned, ["feature_core_available", "core_feature_available"])[0], feature_core_expected, "bool",
            "feature_core_available", mismatches, identity)
        details["core_features"] = core
        details["rows_with_any_feature_missing"] = int(feature_missing_count.gt(0).sum())
    else:
        feature_missing_count = pd.Series(np.nan, index=identity.index)
        checks["feature_missing_count_recomputed"] = False
        checks["feature_core_availability_recomputed"] = False

    split_retained_expected = expected_split.isin(["training", "validation", "test"]) & ~expected_purge
    checks["split_retained_recomputed"] = compare_field(
        first_present(aligned, ["split_retained"])[0], split_retained_expected, "bool",
        "split_retained", mismatches, identity)
    entry_eligible_observed, _ = first_present(aligned, ["entry_trade_eligible"])
    entry_eligible_series = parse_bool(entry_eligible_observed, "entry_trade_eligible", allow_missing=True) if entry_eligible_observed is not None else pd.Series(pd.NA, index=identity.index, dtype="boolean")
    core_observed, _ = first_present(aligned, ["feature_core_available", "core_feature_available"])
    core_series = parse_bool(core_observed, "feature_core_available", allow_missing=True) if core_observed is not None else pd.Series(pd.NA, index=identity.index, dtype="boolean")
    supervised_expected = split_retained_expected & label_expected & entry_eligible_series.fillna(False) & core_series.fillna(False)
    supervised_observed, supervised_name = first_present(aligned, ["supervised_model_eligible", "model_eligible"])
    checks["supervised_eligibility_recomputed"] = compare_field(
        supervised_observed, supervised_expected, "bool", supervised_name or "supervised_model_eligible", mismatches, identity)

    # Reason strings must agree with the flags, while accepting a documented
    # semicolon-separated order from either implementation.
    entry_reason, entry_reason_name = first_present(aligned, ["entry_ineligibility_reason", "entry_reason"])
    if entry_reason is not None and entry_eligible_observed is not None:
        entry_bool = parse_bool(entry_eligible_observed, "entry eligibility", allow_missing=True).fillna(False)
        reason_blank = entry_reason.astype("string").fillna("").str.strip().eq("")
        checks["entry_reason_matches_eligibility"] = bool((reason_blank == entry_bool).all())
    else:
        checks["entry_reason_matches_eligibility"] = False
    supervised_reason, supervised_reason_name = first_present(aligned, ["supervised_ineligibility_reason", "supervised_reason"])
    if supervised_reason is not None and supervised_observed is not None:
        supervised_bool = parse_bool(supervised_observed, "supervised eligibility", allow_missing=True).fillna(False)
        reason_blank = supervised_reason.astype("string").fillna("").str.strip().eq("")
        checks["supervised_reason_matches_eligibility"] = bool((reason_blank == supervised_bool).all())
    else:
        checks["supervised_reason_matches_eligibility"] = False

    if (entry_reason is not None and not expected_entry_eligible.isna().all()
            and price_expected_for_eligibility and len(spans) and len(identity)):
        # Reconstruct the canonical semicolon reason order on development
        # rows.  Boundary/test target sealing is independent of entry flags.
        dev_expected_reasons = pd.Series("", index=identity.index, dtype="string")
        component_rules = [
            ("source_not_index_member_on_announcement", ~source_announcement_expected),
            ("receiver_not_index_member_on_announcement", ~receiver_announcement_expected),
            ("receiver_not_index_member_at_entry", ~receiver_entry_member_for_eligibility.fillna(False)),
            ("entry_price_row_missing", ~parse_bool(price_expected_for_eligibility["price_row_available"].fillna(False), "price row", allow_missing=False)),
            ("entry_close_missing", ~parse_bool(price_expected_for_eligibility["has_close"].fillna(False), "entry close", allow_missing=False)),
        ]
        for reason, failed in component_rules:
            failed = failed.fillna(False).astype(bool) & development_mask
            dev_expected_reasons.loc[failed] = dev_expected_reasons.loc[failed].map(
                lambda value: reason if not value else f"{value};{reason}"
            )
        observed_entry_reasons = entry_reason.astype("string").fillna("").str.strip()
        checks["entry_reason_components_recomputed"] = bool(
            observed_entry_reasons.loc[development_mask].reset_index(drop=True).equals(
                dev_expected_reasons.loc[development_mask].reset_index(drop=True)
            )
        )
    else:
        checks["entry_reason_components_recomputed"] = False

    # If the builder persisted feature_missingness.csv, independently rebuild
    # its complete matrix without fitting or using any test distribution.
    if "feature_missingness" in table_frames and feature_columns and set(feature_columns).issubset(feature_frame.columns):
        miss = aligned.get("feature_missingness", table_frames["feature_missingness"])
        expected_rows = []
        for split_name in ["training", "validation", "test", "out_of_window", "all"]:
            mask = pd.Series(True, index=identity.index) if split_name == "all" else expected_split.eq(split_name)
            denominator = int(mask.sum())
            for feature in feature_columns:
                count = int(feature_frame.loc[mask, feature].isna().sum())
                expected_rows.append({"split": split_name, "feature": feature, "rows": denominator,
                                      "missing": count, "missing_fraction": count / denominator if denominator else np.nan})
        expected_missing = pd.DataFrame(expected_rows)
        observed_missing = miss.copy()
        key = ["split", "feature"]
        if set(key).issubset(observed_missing.columns):
            observed_missing = observed_missing.sort_values(key).reset_index(drop=True)
            expected_missing = expected_missing.sort_values(key).reset_index(drop=True)
            checks["feature_missingness_table_recomputed"] = bool(
                len(observed_missing) == len(expected_missing) and
                all(exact_series_equal(observed_missing[col], expected_missing[col], "numeric" if col != "split" and col != "feature" else "string")
                    for col in ["split", "feature", "rows", "missing", "missing_fraction"] if col in observed_missing and col in expected_missing)
            )
        else:
            checks["feature_missingness_table_recomputed"] = False
    elif "feature_missingness" not in table_frames:
        checks["feature_missingness_table_recomputed"] = True
        details["feature_missingness_table"] = "not persisted; row-level missingness was checked"
    else:
        checks["feature_missingness_table_recomputed"] = False

    # Hashes are checked last, after all source paths are resolved.  A summary
    # may use either ``input_hashes``/``output_hashes`` or the older ``inputs``
    # spelling; both are accepted, but the digest itself must match.
    hash_results: dict[str, bool] = {}
    for section_name, section, base_dir in [
        ("input_hashes", summary.get("input_hashes", {}), ROOT),
        ("output_hashes", summary.get("output_hashes", {}), run_dir),
    ]:
        if not isinstance(section, dict) or not section:
            hash_results[section_name] = False
            continue
        section_ok = True
        for name, item in section.items():
            expected_hash = item.get("sha256") if isinstance(item, dict) else str(item)
            if section_name == "output_hashes" and not isinstance(item, dict):
                # Existing model-ready summaries use keys such as
                # ``model_features`` with the digest as the value.
                key_path = Path(str(name))
                path = key_path if key_path.is_absolute() else base_dir / key_path
                if not path.exists() and not path.suffix:
                    path = path.with_suffix(".csv")
                if not path.exists() and not key_path.suffix:
                    path = (key_path if key_path.is_absolute() else base_dir / key_path).with_suffix(".json")
            else:
                path = resolve_path(item, base_dir)
            if path is None or not path.exists():
                section_ok = False
                hash_results[f"{section_name}:{name}"] = False
                continue
            try:
                observed_hash = sha256_file(path)
                ok = observed_hash == expected_hash
            except Exception:
                ok = False
            hash_results[f"{section_name}:{name}"] = ok
            section_ok = section_ok and ok
        hash_results[section_name] = section_ok
    # The output summary itself is self-describing and therefore cannot hash
    # itself without a recursive convention; all persisted CSV outputs are
    # still required to be represented in output_hashes.
    expected_output_names = {p.name for logical, p in table_paths.items() if logical in required}
    recorded_output_names: set[str] = set()
    if isinstance(summary.get("output_hashes", {}), dict):
        for name in summary["output_hashes"]:
            key_name = Path(str(name)).name
            if not key_name.endswith(".csv") and (run_dir / f"{key_name}.csv").exists():
                key_name += ".csv"
            recorded_output_names.add(key_name)
    hash_results["output_hashes_cover_four_main_tables"] = expected_output_names.issubset(recorded_output_names)
    checks["input_hashes_match_recorded_versions"] = bool(hash_results.get("input_hashes", False))
    checks["output_hashes_match_recorded_files"] = bool(hash_results.get("output_hashes", False) and hash_results["output_hashes_cover_four_main_tables"])
    details["hash_results"] = hash_results
    details["graph_output_hash_checks"] = graph_output_hash_checks

    # Include source hashes that the validator actually used, so an audit can
    # be interpreted even when a builder summary omitted one of them.
    used_hashes: dict[str, str] = {}
    for name, path in [("validator", Path(__file__).resolve()), ("run_summary", summary_path),
                       ("graph_edges", graph_dir / "edges.csv" if graph_dir else None),
                       ("graph_status", graph_dir / "event_graph_status.csv" if graph_dir else None),
                       ("graph_summary", graph_dir / "summary.json" if graph_dir else None),
                       ("spans", spans_path), ("returns", returns_path), ("prices", prices_path)]:
        if path is not None and path.exists():
            try:
                used_hashes[name] = sha256_file(path)
            except OSError:
                pass

    # Write only run-specific audit artefacts.  The report is written even on
    # failures, so a malformed or partial builder run remains auditable.
    audit_run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = AUDIT_ROOT / audit_run
    audit_dir.mkdir(parents=True, exist_ok=False)
    audit_outputs: list[str] = []
    for name, frame in mismatches.items():
        output_name = {
            "boundary_purged_rows": "boundary_purged_rows.csv",
        }.get(name, f"mismatch_{name}.csv")
        written = mismatch_extract(audit_dir, output_name, frame)
        if written:
            audit_outputs.append(written)
    details["mismatch_counts"] = {name: int(len(frame)) for name, frame in mismatches.items()}
    details["audit_outputs"] = audit_outputs
    details["errors"] = errors
    checks["no_unhandled_validation_errors"] = not errors
    report = {
        "run_id": audit_run,
        "model_ready_run_id": run_dir.name,
        "read_only_validation": True,
        "test_target_policy": "test rows are sealed; forward returns must be NaN, label_available must be false, reason must be test_target_sealed; no test future target is computed or read",
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": int(len(checks)),
        "all_passed": bool(all(checks.values())),
        "checks": {name: bool(value) for name, value in checks.items()},
        "details": jsonable(details),
        "used_input_hashes": used_hashes,
        "audit_output_hashes": {name: sha256_file(audit_dir / name) for name in audit_outputs},
    }
    (audit_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["run_id", "model_ready_run_id", "passed", "total", "all_passed"]}, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        failed = [name for name, value in checks.items() if not value]
        print("Validation failures: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
