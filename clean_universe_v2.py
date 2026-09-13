"""Clean the 2015-2026 universe pull with auditable, append-safe outputs.

The module is import-safe: processing starts only from ``main()``. Rows that
cannot be used are retained in a quarantine file with a reason and raw
location. No imputation, winsorisation, or outlier deletion is performed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "universe_v2"
CLEAN = ROOT / "data" / "clean" / "v2"
AUDIT = ROOT / "data" / "audit" / "v2"
INTERVALS = ROOT / "data" / "audit" / "universe_rebuild" / "membership_intervals.csv"
TABLES = ("returns", "actuals", "estimates")


def utc_now() -> str:
    """Return a stable, explicit UTC timestamp for an audit record."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    """Hash a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _metadata(path: Path) -> dict:
    meta_path = path.with_name(path.name.replace(".csv", ".meta.json"))
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid_metadata", "meta_path": relative_path(meta_path)}


def load(prefix: str, extra_paths=(), return_manifest: bool = False):
    """Load raw CSV files and, optionally, return an exact input manifest.

    The default return value remains a DataFrame for compatibility with small
    exploratory callers. A manifest hashes every input CSV and records its
    actual loaded row count, metadata status, and retrieval timestamp.
    """
    paths = sorted(RAW.glob(prefix + "_*.csv"))
    for extra in extra_paths or ():
        extra = Path(extra)
        if extra.exists() and extra not in paths:
            paths.append(extra)
    if not paths:
        raise RuntimeError("No raw files for " + prefix)

    frames = []
    files = []
    for path in paths:
        hash_before_load = sha256_file(path)
        frame = pd.read_csv(path).replace(r"^\s*$", pd.NA, regex=True)
        hash_after_load = sha256_file(path)
        if hash_before_load != hash_after_load:
            raise RuntimeError(f"Raw input changed while loading: {path}")
        frame["raw_file"] = path.name
        frame["raw_row"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
        rec = _metadata(path)
        meta_path = path.with_name(path.name.replace(".csv", ".meta.json"))
        files.append(
            {
                "path": relative_path(path),
                "name": path.stem,
                "sha256": hash_before_load,
                "rows": int(len(frame)),
                "status": rec.get("status"),
                "retrieved_at_utc": rec.get("retrieved_at_utc"),
                "metadata_sha256": sha256_file(meta_path) if meta_path.exists() else None,
            }
        )

    frame = pd.concat(frames, ignore_index=True)
    manifest = {"files": files}
    manifest["sha256"] = sha256_json(files)
    manifest["rows"] = int(len(frame))
    return (frame, manifest) if return_manifest else frame


def assert_manifest_unchanged(manifest):
    """Fail before writing outputs if any raw input changed during a run."""
    changed = []
    for record in manifest.get("files", []):
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists() or sha256_file(path) != record.get("sha256"):
            changed.append(record.get("path"))
    if changed:
        raise RuntimeError("Raw input changed during cleaning: " + ", ".join(changed))


def _terminal_mask(df: pd.DataFrame, valid: pd.Series, pad_key):
    terminal = pd.Series(False, index=df.index)
    if not pad_key:
        return terminal
    inst_col, date_col = pad_key
    valid_idx = df.index[valid]
    last = df.loc[valid, date_col].groupby(df.loc[valid, inst_col]).transform("max")
    terminal.loc[valid_idx] = df.loc[valid, date_col].eq(last).to_numpy()
    return terminal


def clean(
    name: str,
    df: pd.DataFrame,
    dates,
    numeric,
    required,
    keys,
    positive=(),
    pad_key=None,
    audit_dir=None,
    return_details: bool = False,
):
    """Normalize and classify one table without silently choosing conflicts.

    A terminal-day duplicate is retained only when all source business fields
    are identical. If values under the same key conflict, every eligible row
    in that key group is quarantined as ``conflicting_duplicate_key``.
    """
    df = df.copy()
    original = [c for c in df.columns if c not in ("raw_file", "raw_row")]
    required = list(required)
    keys = list(keys)
    missing_columns = [c for c in required + keys if c not in df.columns]
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {missing_columns}")

    for col in dates:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    reason = pd.Series("", index=df.index, dtype="object")
    reason.loc[df[required].isna().any(axis=1)] = "missing_or_unparseable_required_field"
    for col in positive:
        reason.loc[(reason == "") & (df[col] < 0)] = "negative_" + col

    valid = reason.eq("")
    valid_idx = df.index[valid]
    if len(valid_idx):
        work = df.loc[valid_idx, original].copy()
        work["_business_signature"] = pd.util.hash_pandas_object(work, index=False)
        signatures = work.groupby(keys, dropna=False, sort=False)["_business_signature"].transform("nunique")
        conflicting = pd.Series(False, index=df.index)
        conflicting.loc[valid_idx] = signatures.to_numpy() > 1
        reason.loc[conflicting] = "conflicting_duplicate_key"

        duplicate = pd.Series(False, index=df.index)
        duplicate.loc[valid_idx] = df.loc[valid_idx].duplicated(keys, keep="first").to_numpy()
        same_key_repeat = duplicate & valid & ~conflicting
        terminal = _terminal_mask(df, valid, pad_key)
        reason.loc[same_key_repeat & terminal] = "last_day_padding"
        reason.loc[same_key_repeat & ~terminal] = "exact_duplicate"

    bad = df.loc[reason.ne("")].copy()
    bad["rejection_reason"] = reason.loc[reason.ne("")]
    bad["table"] = name
    if audit_dir is not None:
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        bad.to_csv(audit_dir / f"{name}_quarantine.csv", index=False)

    good = df.loc[reason.eq("")].copy().sort_values(keys).reset_index(drop=True)
    quarantined = df.loc[reason.ne("")]
    optional = [c for c in original if c not in required]
    stats = {
        "input_rows": int(len(df)),
        "clean_rows": int(len(good)),
        "quarantined_rows": int(len(bad)),
        "reasons": {str(k): int(v) for k, v in bad["rejection_reason"].value_counts().items()},
        "unique_key": bool(not good.duplicated(keys).any()),
        "instrument_counts": {
            "input": int(df["Instrument"].nunique()) if "Instrument" in df else None,
            "clean": int(good["Instrument"].nunique()) if "Instrument" in good else None,
            "quarantined": int(quarantined["Instrument"].nunique()) if "Instrument" in quarantined else None,
        },
        "missing_required_counts": {c: int(df[c].isna().sum()) for c in required if c in df},
        "missing_optional_counts": {
            c: {
                "input": int(df[c].isna().sum()),
                "clean": int(good[c].isna().sum()),
                "quarantined": int(quarantined[c].isna().sum()),
            }
            for c in optional
        },
    }
    if stats["input_rows"] != stats["clean_rows"] + stats["quarantined_rows"]:
        raise RuntimeError("Row accounting mismatch in " + name)
    if return_details:
        return good, bad, stats
    return good


def tag_membership(df: pd.DataFrame, inst_col="Instrument", date_col="Date"):
    iv = pd.read_csv(INTERVALS)
    iv["start"] = pd.to_datetime(iv.start)
    iv["end"] = pd.to_datetime(iv.end)
    spells = {ric: list(zip(group.start, group.end)) for ric, group in iv.groupby("ric")}
    df = df.reset_index(drop=True)
    flag = np.zeros(len(df), dtype=bool)
    for ric, idx in df.groupby(inst_col).groups.items():
        runs = spells.get(ric)
        if not runs:
            continue
        dates_for_ric = df.loc[idx, date_col]
        mask = np.zeros(len(dates_for_ric), dtype=bool)
        for start, end in runs:
            mask |= ((dates_for_ric >= start) & (dates_for_ric <= end)).values
        flag[np.asarray(idx)] = mask
    df["in_sp500_that_day"] = flag
    return df


def _archive_previous(audit_dir: Path, clean_dir: Path, run_dir: Path):
    """Copy existing audit/output files before any current-run write."""
    candidates = []
    if audit_dir.exists():
        candidates.extend(path for path in audit_dir.iterdir() if path.is_file())
    if clean_dir.exists():
        candidates.extend(path for path in clean_dir.iterdir() if path.is_file())
    archived = []
    for path in candidates:
        destination = run_dir / "previous" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        archived.append({"path": relative_path(path), "sha256": sha256_file(path)})
    (run_dir / "archive_manifest.json").write_text(
        json.dumps({"archived_at_utc": utc_now(), "files": archived}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return archived


def _load_prior_report(path: Path):
    if not path.exists():
        return {}, None
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path)


def _prepare_report(prior: dict, parts, run_id: str, run_timestamp: str, prior_hash):
    report = copy.deepcopy(prior)
    report["report_version"] = max(int(report.get("report_version", 0) or 0), 3)
    report["run_id"] = run_id
    report["report_timestamp_utc"] = run_timestamp
    report["processed_tables"] = list(parts)
    report["retained_unprocessed_tables"] = [table for table in TABLES if table not in parts and table in report]
    for table in report["retained_unprocessed_tables"]:
        entry = report[table]
        if not isinstance(entry, dict):
            continue
        source_timestamp = entry.get("processed_at_utc") or entry.get("timestamp_utc")
        source_input = entry.get("input_hash") or entry.get("input_sha256")
        source_output = entry.get("output_hash") or entry.get("output_sha256")
        source_quarantine = entry.get("quarantine_hash") or entry.get("quarantine_sha256")
        entry.update(
            {
                "status": "retained_unprocessed",
                "processed_in_run": False,
                "retained_at_utc": run_timestamp,
                "retained_from_report_sha256": prior_hash,
                "retained_source_timestamp_utc": source_timestamp,
                "retained_source_input_hash": source_input,
                "retained_source_output_hash": source_output,
                "retained_source_quarantine_hash": source_quarantine,
            }
        )
    return report


def _finish_table_stats(stats, manifest, output_path: Path, quarantine_path: Path):
    output_hash = sha256_file(output_path)
    quarantine_hash = sha256_file(quarantine_path)
    stats.update(
        {
            "input_hash": manifest["sha256"],
            "input_sha256": manifest["sha256"],
            "input_files": manifest["files"],
            "output_path": relative_path(output_path),
            "output_hash": output_hash,
            "output_sha256": output_hash,
            "quarantine_path": relative_path(quarantine_path),
            "quarantine_hash": quarantine_hash,
            "quarantine_sha256": quarantine_hash,
            "status": "processed",
            "processed_in_run": True,
        }
    )
    return stats


def run_clean(parts):
    parts = list(parts)
    unknown = sorted(set(parts) - set(TABLES))
    if unknown:
        raise ValueError("Unknown table(s): " + ", ".join(unknown))
    AUDIT.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    run_timestamp = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = AUDIT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    prior, prior_hash = _load_prior_report(AUDIT / "quality_report.json")
    _archive_previous(AUDIT, CLEAN, run_dir)
    report = _prepare_report(prior, parts, run_id, run_timestamp, prior_hash)

    if "returns" in parts:
        benchmark = RAW / "benchmark_returns.csv"
        r, manifest = load("returns", extra_paths=[benchmark] if benchmark.exists() else [], return_manifest=True)
        r, bad, stats = clean(
            "returns", r, ["Date"], ["Total Return"],
            ["Instrument", "Date", "Total Return"], ["Instrument", "Date"],
            pad_key=("Instrument", "Date"), return_details=True, audit_dir=AUDIT,
        )
        assert_manifest_unchanged(manifest)
        r["return_decimal"] = r["Total Return"] / 100
        if (r.return_decimal < -1).any():
            raise RuntimeError("Return below -100%; check units")
        last_valid = r.groupby("Instrument").Date.max().rename("last_valid_session").reset_index()
        last_valid["delisted_ric"] = last_valid.Instrument.str.contains("^", regex=False)
        last_valid_path = AUDIT / "instrument_last_valid_session.csv"
        last_valid.to_csv(last_valid_path, index=False)
        r = tag_membership(r)
        output_path = CLEAN / "returns.csv"
        r.to_csv(output_path, index=False)
        extreme_path = AUDIT / "extreme_returns_review.csv"
        r.loc[r.return_decimal.abs() > 0.2].to_csv(extreme_path, index=False)
        quarantine_path = AUDIT / "returns_quarantine.csv"
        stats = _finish_table_stats(stats, manifest, output_path, quarantine_path)
        stats.update(
            {
                "processed_at_utc": run_timestamp,
                "timestamp_utc": run_timestamp,
                "instruments": int(r.Instrument.nunique()),
                "sessions": int(r.Date.nunique()),
                "rows_inside_sp500": int(r.in_sp500_that_day.sum()),
                "extreme_abs_gt_20pct": int((r.return_decimal.abs() > 0.2).sum()),
                "auxiliary_outputs": [
                    {"path": relative_path(last_valid_path), "sha256": sha256_file(last_valid_path), "rows": int(len(last_valid))},
                    {"path": relative_path(extreme_path), "sha256": sha256_file(extreme_path), "rows": int((r.return_decimal.abs() > 0.2).sum())},
                ],
            }
        )
        report["returns"] = stats

    if "actuals" in parts:
        a, manifest = load("actuals", return_manifest=True)
        a = a.rename(columns={"Report Date": "announcement"})
        a, bad, stats = clean(
            "actuals", a, ["announcement", "Period End Date"], ["Earnings Per Share - Actual"],
            ["Instrument", "announcement", "Period End Date", "Earnings Per Share - Actual"],
            ["Instrument", "announcement", "Period End Date"], return_details=True, audit_dir=AUDIT,
        )
        assert_manifest_unchanged(manifest)
        minutes = a.announcement.dt.hour * 60 + a.announcement.dt.minute
        a["announcement_session"] = np.select(
            [minutes < 9 * 60 + 30, minutes < 16 * 60], ["pre_market", "intraday"], default="post_market"
        )
        output_path = CLEAN / "actuals.csv"
        a.to_csv(output_path, index=False)
        quarantine_path = AUDIT / "actuals_quarantine.csv"
        stats = _finish_table_stats(stats, manifest, output_path, quarantine_path)
        stats.update(
            {
                "processed_at_utc": run_timestamp,
                "timestamp_utc": run_timestamp,
                "companies": int(a.Instrument.nunique()),
                "session_split": {str(k): int(v) for k, v in a.announcement_session.value_counts().items()},
                "hour_histogram": {str(k): int(v) for k, v in a.announcement.dt.hour.value_counts().sort_index().items()},
            }
        )
        report["actuals"] = stats

    if "estimates" in parts:
        e, manifest = load("estimates", return_manifest=True)
        e = e.rename(columns={"Calc Date": "snapshot"})
        e, bad, stats = clean(
            "estimates", e, ["snapshot", "Period End Date"],
            ["Earnings Per Share - Mean", "Earnings Per Share - Standard Deviation",
             "Earnings Per Share - Number of Included Estimates"],
            ["Instrument", "snapshot", "Period End Date", "Earnings Per Share - Mean"],
            ["Instrument", "snapshot", "Period End Date"],
            positive=["Earnings Per Share - Standard Deviation", "Earnings Per Share - Number of Included Estimates"],
            return_details=True, audit_dir=AUDIT,
        )
        assert_manifest_unchanged(manifest)
        output_path = CLEAN / "estimates.csv"
        e.to_csv(output_path, index=False)
        quarantine_path = AUDIT / "estimates_quarantine.csv"
        stats = _finish_table_stats(stats, manifest, output_path, quarantine_path)
        stats.update({"processed_at_utc": run_timestamp, "timestamp_utc": run_timestamp,
                      "companies": int(e.Instrument.nunique())})
        report["estimates"] = stats

    report["dataset_status"] = "universe_v2_engineering_clean"
    report["universe_source"] = (
        "Backward reconstruction from 2026-09-07 anchor plus 751 index change records; "
        "see data/audit/universe_rebuild/"
    )
    (AUDIT / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report


def main():
    parts = sys.argv[1:] or list(TABLES)
    report = run_clean(parts)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
