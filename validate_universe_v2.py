"""Independent checks on persisted v2 artifacts.

Importing this module performs no I/O. ``validate()`` returns a complete check
report, including explicit failures for missing required tables and audit
entries; ``main()`` persists that report.
"""
from __future__ import annotations

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
REBUILD = ROOT / "data" / "audit" / "universe_rebuild"
REQUIRED_TABLES = ("returns", "actuals", "estimates")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _check_key(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _read_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _read_quality(path: Path):
    if not path.exists():
        return None, "quality_report_missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "quality_report_unreadable"
    return value, None


def _entry_hash(entry: dict, *names):
    for name in names:
        value = entry.get(name)
        if value:
            return value
    return None


def _instrument_count(frame):
    return int(frame["Instrument"].nunique()) if frame is not None and "Instrument" in frame else None


def validate(raw_dir=RAW, clean_dir=CLEAN, audit_dir=AUDIT, rebuild_dir=REBUILD):
    """Run independent checks and return a complete persisted-report payload."""
    raw_dir, clean_dir, audit_dir, rebuild_dir = map(Path, (raw_dir, clean_dir, audit_dir, rebuild_dir))
    checks = {}
    notes = {}

    quality, quality_error = _read_quality(audit_dir / "quality_report.json")
    checks["quality_report_present_and_readable"] = quality_error is None and isinstance(quality, dict)
    if quality_error:
        notes["quality_report_error"] = quality_error
        quality = {}

    successful_raw_checks = []
    for meta_path in sorted(raw_dir.glob("*.meta.json")):
        key = _check_key(meta_path.name.removesuffix(".meta.json"))
        try:
            rec = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checks[f"successful_raw_metadata_readable_{key}"] = False
            continue
        if rec.get("status") != "success":
            continue
        data_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".csv")
        expected_hash = rec.get("sha256")
        ok = bool(expected_hash) and data_path.exists()
        actual_hash = sha256_file(data_path) if data_path.exists() else None
        ok = ok and actual_hash == expected_hash
        check_name = f"successful_raw_hash_matches_{key}"
        checks[check_name] = bool(ok)
        successful_raw_checks.append(bool(ok))
        if not expected_hash:
            notes.setdefault("successful_raw_hash_missing", []).append(meta_path.name)
        if not data_path.exists():
            notes.setdefault("successful_raw_data_missing", []).append(relative_path(data_path))
    checks["all_successful_raw_hashes_present_and_match"] = bool(successful_raw_checks) and all(successful_raw_checks)

    table_frames = {}
    quarantine_frames = {}
    for table in REQUIRED_TABLES:
        entry = quality.get(table) if isinstance(quality, dict) else None
        entry_ok = isinstance(entry, dict)
        checks[f"quality_entry_present_{table}"] = entry_ok
        entry_timestamp = (
            entry.get("processed_at_utc") or entry.get("retained_source_timestamp_utc")
            or entry.get("timestamp_utc")
        ) if entry_ok else None
        entry_input_hash = _entry_hash(entry, "input_hash", "input_sha256") if entry_ok else None
        entry_output_hash = _entry_hash(entry, "output_hash", "output_sha256") if entry_ok else None
        entry_quarantine_hash = _entry_hash(entry, "quarantine_hash", "quarantine_sha256") if entry_ok else None
        checks[f"quality_entry_has_explicit_provenance_{table}"] = bool(
            entry_ok and entry.get("status") in ("processed", "retained_unprocessed")
            and entry_timestamp and entry_input_hash and entry_output_hash and entry_quarantine_hash
        )
        clean_path = clean_dir / f"{table}.csv"
        quarantine_path = audit_dir / f"{table}_quarantine.csv"
        clean_frame = _read_csv(clean_path)
        quarantine_frame = _read_csv(quarantine_path)
        table_frames[table] = clean_frame
        quarantine_frames[table] = quarantine_frame
        checks[f"persisted_clean_output_present_{table}"] = clean_frame is not None
        checks[f"persisted_quarantine_output_present_{table}"] = quarantine_frame is not None

        if not entry_ok:
            for suffix in (
                "row_conservation_in_quality",
                "business_key_uniqueness_in_quality",
                "persisted_clean_row_count_matches_quality",
                "persisted_quarantine_row_count_matches_quality",
                "persisted_clean_hash_matches_quality",
                "persisted_quarantine_hash_matches_quality",
                "quality_instrument_counts_present",
                "persisted_clean_instrument_count_matches_quality",
                "persisted_quarantine_instrument_count_matches_quality",
                "quality_missing_optional_counts_present",
            ):
                checks[f"{suffix}_{table}"] = False
            continue

        input_rows = entry.get("input_rows")
        clean_rows = entry.get("clean_rows")
        quarantined_rows = entry.get("quarantined_rows")
        checks[f"quality_row_conservation_{table}"] = (
            isinstance(input_rows, (int, float))
            and isinstance(clean_rows, (int, float))
            and isinstance(quarantined_rows, (int, float))
            and input_rows == clean_rows + quarantined_rows
        )
        checks[f"quality_business_key_uniqueness_{table}"] = bool(entry.get("unique_key") is True)
        checks[f"persisted_clean_row_count_matches_quality_{table}"] = (
            clean_frame is not None and clean_rows is not None and len(clean_frame) == int(clean_rows)
        )
        checks[f"persisted_quarantine_row_count_matches_quality_{table}"] = (
            quarantine_frame is not None and quarantined_rows is not None and len(quarantine_frame) == int(quarantined_rows)
        )

        input_hash = _entry_hash(entry, "input_hash", "input_sha256")
        output_hash = _entry_hash(entry, "output_hash", "output_sha256")
        quarantine_hash = _entry_hash(entry, "quarantine_hash", "quarantine_sha256")
        checks[f"quality_input_hash_present_{table}"] = bool(input_hash)
        checks[f"persisted_clean_hash_matches_quality_{table}"] = (
            clean_frame is not None and bool(output_hash) and sha256_file(clean_path) == output_hash
        )
        checks[f"persisted_quarantine_hash_matches_quality_{table}"] = (
            quarantine_frame is not None and bool(quarantine_hash) and sha256_file(quarantine_path) == quarantine_hash
        )

        instrument_counts = entry.get("instrument_counts")
        counts_ok = isinstance(instrument_counts, dict) and all(
            key in instrument_counts for key in ("input", "clean", "quarantined")
        )
        checks[f"quality_instrument_counts_present_{table}"] = counts_ok
        checks[f"persisted_clean_instrument_count_matches_quality_{table}"] = (
            counts_ok and clean_frame is not None and instrument_counts["clean"] == _instrument_count(clean_frame)
        )
        checks[f"persisted_quarantine_instrument_count_matches_quality_{table}"] = (
            counts_ok and quarantine_frame is not None and instrument_counts["quarantined"] == _instrument_count(quarantine_frame)
        )

        optional_counts = entry.get("missing_optional_counts")
        optional_ok = isinstance(optional_counts, dict)
        checks[f"quality_missing_optional_counts_present_{table}"] = optional_ok
        if optional_ok:
            for column, expected in optional_counts.items():
                field_key = _check_key(column)
                field_ok = (
                    isinstance(expected, dict)
                    and clean_frame is not None
                    and quarantine_frame is not None
                    and column in clean_frame
                    and column in quarantine_frame
                    and expected.get("clean") == int(clean_frame[column].isna().sum())
                    and expected.get("quarantined") == int(quarantine_frame[column].isna().sum())
                    and expected.get("input") == expected.get("clean") + expected.get("quarantined")
                )
                checks[f"persisted_missing_optional_count_matches_{table}_{field_key}"] = bool(field_ok)

    # Returns: these checks state exactly what they recompute. They do not
    # claim that a zero-heavy series proves absence of imputation.
    r = table_frames["returns"]
    if r is not None:
        try:
            r["Date"] = pd.to_datetime(r["Date"], format="mixed")
            checks["returns_instrument_date_keys_are_unique"] = not r.duplicated(["Instrument", "Date"]).any()
            last_rows = r[r.Date == r.groupby("Instrument").Date.transform("max")]
            checks["returns_each_instrument_has_one_surviving_terminal_date_row"] = (
                not last_rows.empty and last_rows.groupby("Instrument").size().max() == 1
            )
            checks["returns_return_decimal_respects_minus_100pct_floor"] = (
                "return_decimal" in r and bool((r.return_decimal >= -1).all())
            )
        except (KeyError, TypeError, ValueError):
            checks["returns_instrument_date_keys_are_unique"] = False
            checks["returns_each_instrument_has_one_surviving_terminal_date_row"] = False
            checks["returns_return_decimal_respects_minus_100pct_floor"] = False
    else:
        checks["returns_instrument_date_keys_are_unique"] = False
        checks["returns_each_instrument_has_one_surviving_terminal_date_row"] = False
        checks["returns_return_decimal_respects_minus_100pct_floor"] = False

    if r is not None and "return_decimal" in r:
        extreme_path = audit_dir / "extreme_returns_review.csv"
        extreme = _read_csv(extreme_path)
        expected_extreme = int((r.return_decimal.abs() > 0.2).sum())
        checks["returns_extreme_observations_match_review_file"] = (
            extreme is not None and len(extreme) == expected_extreme
        )
        bench = r[r.Instrument == "SPY.P"].sort_values("Date")
        if len(bench):
            window = bench[(bench.Date >= "2015-01-01") & (bench.Date <= "2017-12-29")]
            cumulative = (1 + window.return_decimal).prod() - 1
            checks["benchmark_2015_2017_cumulative_return_in_expected_range"] = 0.34 < cumulative < 0.42
            notes["spy_2015_2017_cumulative"] = round(float(cumulative), 4)

    iv_path = rebuild_dir / "membership_intervals.csv"
    iv = _read_csv(iv_path)
    if r is not None and iv is not None and {"ric", "start", "end"}.issubset(iv.columns):
        try:
            iv["start"] = pd.to_datetime(iv.start)
            iv["end"] = pd.to_datetime(iv.end)
            spells = {key: list(zip(group.start, group.end)) for key, group in iv.groupby("ric")}
            sample = r.sample(min(20000, len(r)), random_state=0) if len(r) else r
            recomputed = np.array([
                any(start <= date <= end for start, end in spells.get(inst, []))
                for inst, date in zip(sample.Instrument, sample.Date)
            ])
            checks["returns_membership_flags_recomputed_from_intervals"] = bool(
                len(sample) and (recomputed == sample.in_sp500_that_day.values).all()
            )
        except (KeyError, TypeError, ValueError):
            checks["returns_membership_flags_recomputed_from_intervals"] = False
    else:
        checks["returns_membership_flags_recomputed_from_intervals"] = False

    last_valid = _read_csv(audit_dir / "instrument_last_valid_session.csv")
    if last_valid is not None and iv is not None:
        try:
            last_valid["last_valid_session"] = pd.to_datetime(last_valid.last_valid_session)
            dead = last_valid[last_valid.delisted_ric]
            checks["delisted_ric_last_sessions_stay_inside_study_window"] = bool(
                (dead.last_valid_session <= pd.Timestamp("2026-09-07")).all()
            )
            checks["delisted_ric_last_sessions_do_not_outlive_live_data"] = bool(
                (dead.last_valid_session < last_valid.last_valid_session.max()).all()
            )
            leave = iv.groupby("ric").end.max() if "end" in iv else pd.Series(dtype="datetime64[ns]")
            merged = dead.merge(leave.rename("index_exit"), left_on="Instrument", right_index=True, how="left")
            gap = (merged.last_valid_session - merged.index_exit).dt.days
            notes["delisted_last_session_vs_index_exit_days"] = (
                {"min": int(gap.min()), "median": float(gap.median()), "max": int(gap.max()),
                 "share_within_15d": round(float((gap.abs() <= 15).mean()), 4)}
                if len(gap.dropna()) else {}
            )
        except (KeyError, TypeError, ValueError):
            checks["delisted_ric_last_sessions_stay_inside_study_window"] = False
            checks["delisted_ric_last_sessions_do_not_outlive_live_data"] = False
    else:
        checks["delisted_ric_last_sessions_stay_inside_study_window"] = False
        checks["delisted_ric_last_sessions_do_not_outlive_live_data"] = False

    a = table_frames["actuals"]
    if a is not None:
        try:
            a["announcement"] = pd.to_datetime(a["announcement"], format="mixed")
            minutes = a.announcement.dt.hour * 60 + a.announcement.dt.minute
            intraday = int(((minutes >= 9 * 60 + 30) & (minutes < 16 * 60)).sum())
            checks["actuals_intraday_announcement_share_below_2pct"] = bool(len(a) and intraday / len(a) < 0.02)
            checks["actuals_session_labels_match_minute_boundaries"] = bool(
                "announcement_session" in a and (a.announcement_session == np.select(
                    [minutes < 9 * 60 + 30, minutes < 16 * 60], ["pre_market", "intraday"], default="post_market"
                )).all()
            )
            notes["intraday_announcements"] = intraday
            notes["intraday_share"] = round(float(intraday / len(a)), 4) if len(a) else None
            notes["announcement_hours"] = {str(k): int(v) for k, v in a.announcement.dt.hour.value_counts().sort_index().items()}
        except (KeyError, TypeError, ValueError):
            checks["actuals_intraday_announcement_share_below_2pct"] = False
            checks["actuals_session_labels_match_minute_boundaries"] = False

    e = table_frames["estimates"]
    if e is not None and a is not None:
        try:
            e["Period End Date"] = pd.to_datetime(e["Period End Date"], format="mixed")
            a["Period End Date"] = pd.to_datetime(a["Period End Date"], format="mixed")
            keys_a = set(zip(a.Instrument, a["Period End Date"]))
            keys_e = set(zip(e.Instrument, e["Period End Date"]))
            coverage = len(keys_a & keys_e) / len(keys_a) if keys_a else 0
            checks["estimates_actual_quarter_key_coverage_above_80pct"] = coverage > 0.80
            notes["estimate_quarter_coverage"] = round(float(coverage), 4)
        except (KeyError, TypeError, ValueError):
            checks["estimates_actual_quarter_key_coverage_above_80pct"] = False

    result = {
        "validated_at_utc": utc_now(),
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "failed": [key for key, value in checks.items() if not value],
        "checks": {key: bool(value) for key, value in checks.items()},
        "notes": notes,
        "scope": "Engineering integrity of the reconstructed universe. Not proof of point-in-time correctness or of any predictive content.",
    }
    return result


def _archive_previous_validation(audit_dir: Path, run_id: str):
    previous = audit_dir / "validation.json"
    if not previous.exists():
        return None
    destination = audit_dir / "runs" / f"validation_{run_id}" / "previous" / relative_path(previous)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(previous, destination)
    return {"path": relative_path(previous), "sha256": sha256_file(previous), "archive_path": relative_path(destination)}


def run_validation(raw_dir=RAW, clean_dir=CLEAN, audit_dir=AUDIT, rebuild_dir=REBUILD):
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = _archive_previous_validation(audit_dir, run_id)
    result = validate(raw_dir, clean_dir, audit_dir, rebuild_dir)
    result["run_id"] = run_id
    if archive:
        result["previous_validation_archive"] = archive
    (audit_dir / "validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return result


def main():
    result = run_validation()
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2, ensure_ascii=False, default=str))
    if result["failed"]:
        raise SystemExit("Validation failed: " + ", ".join(result["failed"]))


if __name__ == "__main__":
    main()
