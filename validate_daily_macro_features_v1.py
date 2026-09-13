"""Independently validate the lagged daily macro feature build.

The validator reconstructs source as-of values from raw FRED CSV bytes and
reconstructs SPY sessions from the raw benchmark date column.  It does not
read labels, returns, predictions, metrics, or model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SERIES = ("VIXCLS", "DGS3MO", "DGS2", "DGS10", "BAMLH0A0HYM2", "BAMLC0A0CM", "DTWEXBGS")
MISSING_TOKENS = {"", ".", "NA", "NaN", "nan"}
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2020-12-31")
VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(path: Path, series: str) -> pd.DataFrame:
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    rows: list[dict[str, object]] = []
    for source_row, row in enumerate(reader, start=1):
        raw = (row.get(series) or "").strip()
        value = np.nan if raw in MISSING_TOKENS else float(raw)
        rows.append(
            {
                "observation_date": pd.Timestamp(row["observation_date"]).normalize(),
                "value_raw": raw,
                "value_numeric": value,
                "provider_missing": bool(pd.isna(value)),
                "source_row": source_row,
            }
        )
    return pd.DataFrame(rows).sort_values("observation_date", kind="stable").reset_index(drop=True)


def allclose(left: pd.Series, right: pd.Series, tol: float = 1e-9) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(a, b, equal_nan=True, atol=tol, rtol=tol))


def rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    def rank_last(x: np.ndarray) -> float:
        return float(np.sum(x <= x[-1]) / len(x))

    return values.rolling(window, min_periods=window).apply(rank_last, raw=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-run-id", required=True)
    parser.add_argument("--audit-run-id", default=None)
    args = parser.parse_args()
    clean_dir = ROOT / "data" / "clean" / "daily_macro_v1" / args.clean_run_id
    config_path = clean_dir / "build_config.json"
    macro_path = clean_dir / "macro_features.csv"
    train_valid_path = clean_dir / "macro_features_train_valid.csv"
    if not config_path.exists() or not macro_path.exists():
        raise FileNotFoundError(f"clean build files not found under {clean_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_dir = ROOT / "data" / "raw" / "daily_macro_v1" / str(config["raw_collection_run_id"])
    raw_manifest_path = raw_dir / "manifest.json"
    spy_path = ROOT / str(config["spy_benchmark_path"])
    macro = pd.read_csv(macro_path, keep_default_na=True)
    raw_token_columns = {f"{series}_source_value_raw": "string" for series in SERIES}
    # Keep a second, string-preserving view for exact source-token checks;
    # pandas' numeric inference would otherwise turn e.g. ``19.20`` into
    # ``19.2`` and make an unchanged token look different.
    macro_raw_tokens = pd.read_csv(macro_path, keep_default_na=False, dtype=raw_token_columns)
    train_valid = pd.read_csv(train_valid_path, keep_default_na=True)
    macro["spy_session"] = pd.to_datetime(macro["spy_session"], format="%Y-%m-%d", errors="coerce")
    macro["reference_spy_session"] = pd.to_datetime(macro["reference_spy_session"], format="%Y-%m-%d", errors="coerce")
    for series in SERIES:
        for suffix in ("source_observation_date", "value_observation_date"):
            col = f"{series}_{suffix}"
            macro[col] = pd.to_datetime(macro[col], format="%Y-%m-%d", errors="coerce")

    audit_id = args.audit_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_dir = ROOT / "data" / "audit" / "daily_macro_v1" / audit_id
    audit_dir.mkdir(parents=True, exist_ok=False)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    checks["macro_rows_positive"] = len(macro) > 0
    checks["macro_spy_session_non_null"] = bool(macro["spy_session"].notna().all())
    checks["macro_spy_session_sorted_unique"] = bool(macro["spy_session"].is_monotonic_increasing and not macro["spy_session"].duplicated().any())
    checks["macro_no_target_like_columns"] = not any(any(token in col.casefold() for token in ("target", "label", "return", "prediction", "metric", "y_true", "y_pred")) for col in macro.columns)

    spy_source = pd.read_csv(spy_path, usecols=["Date"])
    spy_sessions = pd.to_datetime(spy_source["Date"], format="%Y-%m-%d", errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    checks["macro_rows_equal_spy_unique_sessions"] = len(macro) == len(spy_sessions)
    checks["macro_sessions_equal_spy_sessions"] = bool(macro["spy_session"].reset_index(drop=True).equals(spy_sessions))
    expected_reference = macro["spy_session"].shift(1)
    checks["reference_is_previous_spy_session"] = bool(macro["reference_spy_session"].equals(expected_reference))
    checks["reference_strictly_prior"] = bool((macro.loc[1:, "reference_spy_session"] < macro.loc[1:, "spy_session"]).all())
    checks["first_session_has_no_reference"] = bool(pd.isna(macro.loc[0, "reference_spy_session"]))

    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    checks["raw_manifest_complete"] = tuple(raw_manifest.get("series", [])) == SERIES and raw_manifest.get("returned_count") == len(SERIES)
    source_tables: dict[str, pd.DataFrame] = {}
    raw_hash_checks: dict[str, bool] = {}
    for series in SERIES:
        manifest_row = next((row for row in raw_manifest.get("series_results", []) if row.get("series") == series), None)
        source_path = raw_dir / str(manifest_row["response_file"]) if manifest_row else raw_dir / "missing"
        exists = source_path.exists()
        raw_hash_checks[series] = bool(exists and sha256_file(source_path) == manifest_row.get("sha256")) if manifest_row else False
        source_tables[series] = parse_source(source_path, series) if exists else pd.DataFrame()
    checks["raw_hashes_match_collection_manifest"] = all(raw_hash_checks.values())
    details["raw_hash_checks"] = raw_hash_checks

    value_checks: dict[str, dict[str, bool]] = {}
    reason_mismatch_rows: list[dict[str, object]] = []
    session_positions = {date: i for i, date in enumerate(spy_sessions)}
    for series in SERIES:
        source = source_tables[series]
        value_col = f"{series}_value"
        value_date_col = f"{series}_value_observation_date"
        source_date_col = f"{series}_source_observation_date"
        source_raw_col = f"{series}_source_value_raw"
        source_numeric_col = f"{series}_source_value_numeric"
        source_missing_col = f"{series}_source_missing"
        source_row_col = f"{series}_source_row"
        value_source_row_col = f"{series}_value_source_row"
        first_col = f"{series}_first_available_date"
        age_calendar_col = f"{series}_carry_age_calendar_days"
        age_sessions_col = f"{series}_carry_age_spy_sessions"
        reason_col = f"{series}_reason_code"
        source_dates = source["observation_date"].to_numpy(dtype="datetime64[ns]")
        source_values = source["value_numeric"].to_numpy(dtype=float)
        valid_source = source[source["value_numeric"].notna()].reset_index(drop=True)
        expected_values: list[float] = []
        expected_value_dates: list[pd.Timestamp | pd.NaT] = []
        expected_latest_dates: list[pd.Timestamp | pd.NaT] = []
        expected_latest_raw: list[str | None] = []
        expected_latest_numeric: list[float] = []
        expected_latest_missing: list[bool | None] = []
        expected_latest_rows: list[float] = []
        expected_valid_rows: list[float] = []
        expected_reasons: list[str] = []
        for i, reference in enumerate(macro["reference_spy_session"]):
            if pd.isna(reference):
                expected_latest_dates.append(pd.NaT)
                expected_latest_raw.append(None)
                expected_latest_numeric.append(np.nan)
                expected_latest_missing.append(None)
                expected_latest_rows.append(np.nan)
                expected_values.append(np.nan)
                expected_value_dates.append(pd.NaT)
                expected_valid_rows.append(np.nan)
                expected_reasons.append("lag_no_prior_spy_session")
                continue
            all_idx = int(np.searchsorted(source_dates, np.datetime64(reference), side="right") - 1)
            valid_idx = int(np.searchsorted(valid_source["observation_date"].to_numpy(dtype="datetime64[ns]"), np.datetime64(reference), side="right") - 1)
            if all_idx < 0:
                expected_latest_dates.append(pd.NaT)
                expected_latest_raw.append(None)
                expected_latest_numeric.append(np.nan)
                expected_latest_missing.append(None)
                expected_latest_rows.append(np.nan)
            else:
                latest = source.iloc[all_idx]
                expected_latest_dates.append(latest["observation_date"])
                expected_latest_raw.append(str(latest["value_raw"]))
                expected_latest_numeric.append(float(latest["value_numeric"]) if pd.notna(latest["value_numeric"]) else np.nan)
                expected_latest_missing.append(bool(latest["provider_missing"]))
                expected_latest_rows.append(float(latest["source_row"]))
            if valid_idx < 0:
                expected_values.append(np.nan)
                expected_value_dates.append(pd.NaT)
                expected_valid_rows.append(np.nan)
            else:
                valid_row = valid_source.iloc[valid_idx]
                expected_values.append(float(valid_row["value_numeric"]))
                expected_value_dates.append(valid_row["observation_date"])
                expected_valid_rows.append(float(valid_row["source_row"]))
            latest_date = expected_latest_dates[-1]
            value_date = expected_value_dates[-1]
            latest_missing = expected_latest_missing[-1]
            if pd.isna(latest_date):
                expected_reasons.append("no_prior_source_observation")
            elif pd.isna(value_date):
                expected_reasons.append("supplier_missing_before_first_valid_observation")
            elif latest_missing:
                expected_reasons.append("carried_forward_across_supplier_missing")
            elif latest_date == value_date and latest_date == reference:
                expected_reasons.append("source_observed_on_reference_session")
            else:
                expected_reasons.append("carried_forward_from_observation")
        expected_value_series = pd.Series(expected_values)
        expected_value_dates_series = pd.Series(expected_value_dates)
        expected_latest_dates_series = pd.Series(expected_latest_dates)
        checks_for_series = {
            "value_asof_matches_raw": allclose(macro[value_col], expected_value_series),
            "value_date_asof_matches_raw": bool(macro[value_date_col].reset_index(drop=True).equals(expected_value_dates_series)),
            "latest_source_date_matches_raw": bool(macro[source_date_col].reset_index(drop=True).equals(expected_latest_dates_series)),
            "latest_source_raw_matches_raw": macro_raw_tokens[source_raw_col].astype(str).tolist() == [x or "" for x in expected_latest_raw],
            "latest_source_numeric_matches_raw": allclose(macro[source_numeric_col], pd.Series(expected_latest_numeric)),
            "latest_source_missing_matches_raw": macro[source_missing_col].astype("string").fillna("<NA>").tolist() == ["<NA>" if x is None else str(bool(x)) for x in expected_latest_missing],
            "latest_source_row_matches_raw": allclose(macro[source_row_col], pd.Series(expected_latest_rows)),
            "value_source_row_matches_raw": allclose(macro[value_source_row_col], pd.Series(expected_valid_rows)),
            "reason_codes_match_raw": macro[reason_col].fillna("").astype(str).tolist() == expected_reasons,
            "no_source_date_after_reference": bool((macro.loc[macro[source_date_col].notna(), source_date_col] <= macro.loc[macro[source_date_col].notna(), "reference_spy_session"]).all()),
            "no_value_date_after_reference": bool((macro.loc[macro[value_date_col].notna(), value_date_col] <= macro.loc[macro[value_date_col].notna(), "reference_spy_session"]).all()),
            "age_calendar_non_negative": bool((pd.to_numeric(macro[age_calendar_col], errors="coerce").dropna() >= 0).all()),
            "age_sessions_non_negative": bool((pd.to_numeric(macro[age_sessions_col], errors="coerce").dropna() >= 0).all()),
            "first_available_date_correct": macro[first_col].dropna().astype(str).eq(valid_source["observation_date"].min().date().isoformat()).all() if not valid_source.empty else macro[first_col].isna().all(),
        }
        expected_age_calendar: list[float] = []
        expected_age_sessions: list[float] = []
        for reference, value_date in zip(macro["reference_spy_session"], expected_value_dates):
            if pd.isna(reference) or pd.isna(value_date):
                expected_age_calendar.append(np.nan)
                expected_age_sessions.append(np.nan)
            else:
                expected_age_calendar.append(float((reference - value_date).days))
                last_session_position = int(np.searchsorted(spy_sessions.to_numpy(dtype="datetime64[ns]"), np.datetime64(value_date), side="right") - 1)
                expected_age_sessions.append(float(session_positions[reference] - last_session_position))
        checks_for_series["age_calendar_matches"] = allclose(macro[age_calendar_col], pd.Series(expected_age_calendar))
        checks_for_series["age_sessions_matches"] = allclose(macro[age_sessions_col], pd.Series(expected_age_sessions))
        value_checks[series] = checks_for_series
        for i, (actual, expected) in enumerate(zip(macro[reason_col].fillna(""), expected_reasons)):
            if actual != expected:
                reason_mismatch_rows.append({"series": series, "row_index": i, "spy_session": macro.loc[i, "spy_session"].date().isoformat(), "actual": actual, "expected": expected})
    checks.update({f"{series}_{name}": value for series, values in value_checks.items() for name, value in values.items()})

    # Recompute engineered features from the independently checked lagged values.
    feature_checks: dict[str, bool] = {}
    for series, prefix, horizons in [
        ("VIXCLS", "vix", (1, 5, 20)),
        ("DGS3MO", "dgs3mo", (5, 20)),
        ("DGS2", "dgs2", (5, 20)),
        ("DGS10", "dgs10", (5, 20)),
        ("BAMLH0A0HYM2", "high_yield_oas", (5, 20)),
        ("BAMLC0A0CM", "investment_grade_oas", (5, 20)),
        ("DTWEXBGS", "broad_dollar", (5, 20, 60)),
    ]:
        levels = macro[f"{series}_value"]
        feature_checks[f"{prefix}_level"] = allclose(macro[f"{prefix}_level"], levels)
        for horizon in horizons:
            feature_checks[f"{prefix}_change_{horizon}d"] = allclose(macro[f"{prefix}_change_{horizon}d"], levels.diff(horizon))
        if series == "VIXCLS":
            feature_checks["vix_percentile_252d"] = allclose(macro["vix_percentile_252d"], rolling_percentile(levels, 252))
    feature_checks["term_spread_10y_2y"] = allclose(macro["term_spread_10y_2y"], macro["DGS10_value"] - macro["DGS2_value"])
    feature_checks["term_spread_10y_3m"] = allclose(macro["term_spread_10y_3m"], macro["DGS10_value"] - macro["DGS3MO_value"])
    checks.update({f"feature_{name}": value for name, value in feature_checks.items()})

    train_dates = pd.to_datetime(train_valid["spy_session"], format="%Y-%m-%d", errors="coerce")
    checks["train_validation_window_exact"] = bool(((train_dates >= TRAIN_START) & (train_dates <= VALIDATION_END)).all())
    expected_train_valid = macro[(macro["spy_session"] >= TRAIN_START) & (macro["spy_session"] <= VALIDATION_END)].copy()
    checks["train_validation_rows_exact"] = len(train_valid) == len(expected_train_valid)
    checks["train_validation_sessions_exact"] = bool(train_valid["spy_session"].tolist() == expected_train_valid["spy_session"].dt.strftime("%Y-%m-%d").tolist())
    for series in ("BAMLH0A0HYM2", "BAMLC0A0CM"):
        checks[f"{series}_training_coverage_unavailable"] = bool(macro.loc[(macro["spy_session"] >= TRAIN_START) & (macro["spy_session"] <= TRAIN_END), f"{series}_value"].isna().all())
        checks[f"{series}_validation_coverage_unavailable"] = bool(macro.loc[(macro["spy_session"] >= VALIDATION_START) & (macro["spy_session"] <= VALIDATION_END), f"{series}_value"].isna().all())
    definitions = json.loads((clean_dir / "feature_definitions.json").read_text(encoding="utf-8"))
    excluded = set(definitions.get("credit_spread_feature_columns_excluded_from_current_train_validation", []))
    checks["credit_features_excluded_from_recommended_list"] = excluded.isdisjoint(set(definitions.get("recommended_train_validation_feature_columns", [])))
    checks["credit_exclusion_reason_explicit"] = "TRAINING_COVERAGE_UNAVAILABLE" in str(definitions.get("credit_spread_exclusion_reason", ""))

    if reason_mismatch_rows:
        pd.DataFrame(reason_mismatch_rows).to_csv(audit_dir / "reason_mismatches.csv", index=False)
    else:
        pd.DataFrame(columns=["series", "row_index", "spy_session", "actual", "expected"]).to_csv(audit_dir / "reason_mismatches.csv", index=False)
    output_hashes = {name: sha256_file(clean_dir / name) for name in ("macro_features.csv", "macro_features_train_valid.csv", "fred_observations.csv", "spy_sessions.csv", "feature_definitions.json", "clean_summary.json") if (clean_dir / name).exists()}
    # Pandas ``Series.all`` and NumPy comparisons can produce np.bool_;
    # normalize every check before serializing the independently produced
    # audit record.
    checks = {name: bool(value) for name, value in checks.items()}
    summary = {
        "schema": "daily_macro_features_validation_v1",
        "audit_run_id": audit_id,
        "validated_at_utc": utc_now(),
        "clean_run_id": args.clean_run_id,
        "clean_path": str(clean_dir.relative_to(ROOT)),
        "input_scope": "raw FRED manifest/bytes and raw SPY benchmark dates plus clean macro outputs; no labels, returns, targets, predictions, metrics, or model artifacts",
        "counts": {"macro_rows": int(len(macro)), "train_validation_rows": int(len(train_valid)), "spy_unique_sessions": int(len(spy_sessions)), "reason_mismatch_rows": int(len(reason_mismatch_rows))},
        "source_profiles": {series: {"raw_rows": int(len(source_tables[series])), "numeric_rows": int(source_tables[series]["value_numeric"].notna().sum()), "missing_rows": int(source_tables[series]["value_numeric"].isna().sum())} for series in SERIES},
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "feature_checks": feature_checks,
        "output_sha256": output_hashes,
        "limitations": [
            "This verifies deterministic alignment and arithmetic, not historical FRED release timestamps or vintage correctness.",
            "FRED current CSV source remains subject to revisions; the one-SPY-session lag is a conservative implementation rule.",
        ],
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"audit_run_id": audit_id, "all_checks_pass": summary["all_checks_pass"], "checks": len(checks), "audit_dir": str(audit_dir)}, indent=2))


if __name__ == "__main__":
    main()
