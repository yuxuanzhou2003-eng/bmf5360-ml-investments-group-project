"""Build lagged, auditable daily macro features from a raw FRED run.

Only a raw FRED collection and the raw SPY benchmark session dates are read.
The output retains source tokens and as-of metadata so downstream model code
can decide how to handle missingness without silently changing the research
assumptions.
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
SERIES = (
    "VIXCLS",
    "DGS3MO",
    "DGS2",
    "DGS10",
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "DTWEXBGS",
)
SOURCE_MISSING_TOKENS = {"", ".", "NA", "NaN", "nan"}
RAW_BENCHMARK_DEFAULT = ROOT / "data" / "raw" / "universe_v3" / "20260909T012417705069Z" / "prices_benchmark_000.csv"
TRAIN_START = pd.Timestamp("2015-01-01")
VALIDATION_END = pd.Timestamp("2022-12-31")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_numeric(raw: str) -> float:
    value = (raw or "").strip()
    if value in SOURCE_MISSING_TOKENS:
        return np.nan
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"non-finite FRED value: {raw!r}")
    return parsed


def load_fred(path: Path, series: str) -> pd.DataFrame:
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    expected = ["observation_date", series]
    if list(reader.fieldnames or []) != expected:
        raise ValueError(f"{path} columns {reader.fieldnames!r} != {expected!r}")
    rows: list[dict[str, object]] = []
    for source_row, row in enumerate(reader, start=1):
        date_raw = (row.get("observation_date") or "").strip()
        value_raw = (row.get(series) or "").strip()
        date = pd.to_datetime(date_raw, format="%Y-%m-%d", errors="coerce")
        if pd.isna(date):
            raise ValueError(f"invalid observation date in {path}: {date_raw!r}")
        value = parse_numeric(value_raw)
        rows.append(
            {
                "series": series,
                "observation_date": date.normalize(),
                "value_raw": value_raw,
                "value_numeric": value,
                "provider_missing": bool(pd.isna(value)),
                "source_file": str(path.relative_to(ROOT)),
                "source_row": source_row,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"empty FRED source: {path}")
    result = result.sort_values("observation_date", kind="stable").reset_index(drop=True)
    if result["observation_date"].duplicated().any():
        raise ValueError(f"duplicate FRED observation dates: {path}")
    return result


def load_spy_sessions(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_csv(path, usecols=["Date"])
    raw_rows = len(source)
    source["spy_session"] = pd.to_datetime(source["Date"], format="%Y-%m-%d", errors="coerce").dt.normalize()
    invalid = source[source["spy_session"].isna()].copy()
    invalid["reason_code"] = "invalid_spy_session_date"
    valid = source[source["spy_session"].notna()].copy()
    duplicate = valid[valid["spy_session"].duplicated(keep=False)].copy()
    duplicate["reason_code"] = "duplicate_spy_session_date"
    valid = valid.drop_duplicates("spy_session", keep="first").sort_values("spy_session", kind="stable")
    sessions = valid[["spy_session"]].reset_index(drop=True)
    profile = {
        "raw_rows": int(raw_rows),
        "valid_unique_sessions": int(len(sessions)),
        "invalid_rows": int(len(invalid)),
        "duplicate_rows": int(len(duplicate)),
        "date_min": sessions["spy_session"].min().date().isoformat() if not sessions.empty else None,
        "date_max": sessions["spy_session"].max().date().isoformat() if not sessions.empty else None,
        "invalid_rows_frame": invalid,
        "duplicate_rows_frame": duplicate,
    }
    return sessions, profile


def effective_source_asof(
    sessions: pd.DataFrame, source: pd.DataFrame, series: str
) -> pd.DataFrame:
    """Return latest source row and latest valid observation as of t-1 session."""
    out = sessions.copy()
    out["reference_spy_session"] = out["spy_session"].shift(1)
    lookup = out[["spy_session", "reference_spy_session"]].dropna(subset=["reference_spy_session"])
    lookup = lookup.sort_values("reference_spy_session", kind="stable")
    all_source = source.sort_values("observation_date", kind="stable")
    latest = pd.merge_asof(
        lookup,
        all_source[["observation_date", "value_raw", "value_numeric", "provider_missing", "source_row"]],
        left_on="reference_spy_session",
        right_on="observation_date",
        direction="backward",
        allow_exact_matches=True,
    )
    latest = latest.rename(
        columns={
            "observation_date": f"{series}_source_observation_date",
            "value_raw": f"{series}_source_value_raw",
            "value_numeric": f"{series}_source_value_numeric",
            "provider_missing": f"{series}_source_missing",
            "source_row": f"{series}_source_row",
        }
    )
    valid_source = all_source[all_source["value_numeric"].notna()].copy()
    valid = pd.merge_asof(
        lookup,
        valid_source[["observation_date", "value_numeric", "source_row"]],
        left_on="reference_spy_session",
        right_on="observation_date",
        direction="backward",
        allow_exact_matches=True,
    )
    valid = valid.rename(
        columns={
            "observation_date": f"{series}_value_observation_date",
            "value_numeric": f"{series}_value",
            "source_row": f"{series}_value_source_row",
        }
    )
    first_available = valid_source["observation_date"].min()
    first_available_text = first_available.date().isoformat() if pd.notna(first_available) else None
    latest = latest.merge(valid, on=["spy_session", "reference_spy_session"], how="left", sort=False)
    latest = out[["spy_session", "reference_spy_session"]].merge(latest, on=["spy_session", "reference_spy_session"], how="left", sort=False)
    latest[f"{series}_first_available_date"] = first_available_text
    latest[f"{series}_source_missing"] = latest[f"{series}_source_missing"].astype("boolean")

    sessions_index = pd.Series(np.arange(len(sessions), dtype=int), index=sessions["spy_session"])
    source_dates = latest[f"{series}_value_observation_date"]
    ref_dates = latest["reference_spy_session"]
    valid_mask = source_dates.notna() & ref_dates.notna()
    latest[f"{series}_carry_age_calendar_days"] = np.nan
    latest.loc[valid_mask, f"{series}_carry_age_calendar_days"] = (
        ref_dates[valid_mask] - source_dates[valid_mask]
    ).dt.days.astype(float)
    source_positions = np.searchsorted(
        sessions["spy_session"].to_numpy(dtype="datetime64[ns]"),
        source_dates[valid_mask].to_numpy(dtype="datetime64[ns]"),
        side="right",
    ) - 1
    ref_positions = ref_dates[valid_mask].map(sessions_index).to_numpy(dtype=float)
    latest[f"{series}_carry_age_spy_sessions"] = np.nan
    latest.loc[valid_mask, f"{series}_carry_age_spy_sessions"] = ref_positions - source_positions

    reason: list[str] = []
    for row in latest.itertuples(index=False):
        ref = getattr(row, "reference_spy_session")
        value_date = getattr(row, f"{series}_value_observation_date")
        source_date = getattr(row, f"{series}_source_observation_date")
        source_missing = getattr(row, f"{series}_source_missing")
        if pd.isna(ref):
            reason.append("lag_no_prior_spy_session")
        elif pd.isna(source_date):
            reason.append("no_prior_source_observation")
        elif pd.isna(value_date):
            reason.append("supplier_missing_before_first_valid_observation")
        elif bool(source_missing):
            reason.append("carried_forward_across_supplier_missing")
        elif source_date == value_date and source_date == ref:
            reason.append("source_observed_on_reference_session")
        else:
            reason.append("carried_forward_from_observation")
    latest[f"{series}_reason_code"] = reason
    return latest


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        last = values[-1]
        return float(np.sum(values <= last) / len(values))

    return series.rolling(window, min_periods=window).apply(rank_last, raw=True)


def add_features(aligned: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, object]]]:
    result = aligned.copy()
    feature_columns: list[str] = []
    definitions: dict[str, dict[str, object]] = {}
    for series in SERIES:
        level_name = f"{series}_value"
        if series == "VIXCLS":
            prefix = "vix"
            horizons = (1, 5, 20)
        elif series == "DGS3MO":
            prefix = "dgs3mo"
            horizons = (5, 20)
        elif series == "DGS2":
            prefix = "dgs2"
            horizons = (5, 20)
        elif series == "DGS10":
            prefix = "dgs10"
            horizons = (5, 20)
        elif series == "BAMLH0A0HYM2":
            prefix = "high_yield_oas"
            horizons = (5, 20)
        elif series == "BAMLC0A0CM":
            prefix = "investment_grade_oas"
            horizons = (5, 20)
        elif series == "DTWEXBGS":
            prefix = "broad_dollar"
            horizons = (5, 20, 60)
        else:  # pragma: no cover
            raise AssertionError(series)
        level_feature = f"{prefix}_level"
        result[level_feature] = result[level_name]
        feature_columns.append(level_feature)
        definitions[level_feature] = {"source_series": series, "operation": "lagged carried level", "units": "source native units"}
        for horizon in horizons:
            change_name = f"{prefix}_change_{horizon}d"
            result[change_name] = result[level_name].diff(horizon)
            feature_columns.append(change_name)
            definitions[change_name] = {
                "source_series": series,
                "operation": f"current lagged carried level minus {horizon} SPY sessions prior",
                "units": "source native units",
            }
        if series == "VIXCLS":
            percentile_name = "vix_percentile_252d"
            result[percentile_name] = rolling_percentile(result[level_name], 252)
            feature_columns.append(percentile_name)
            definitions[percentile_name] = {
                "source_series": series,
                "operation": "252-session rolling percentile rank including current lagged value",
                "units": "0-1 rank",
            }

    result["term_spread_10y_2y"] = result["DGS10_value"] - result["DGS2_value"]
    result["term_spread_10y_3m"] = result["DGS10_value"] - result["DGS3MO_value"]
    feature_columns += ["term_spread_10y_2y", "term_spread_10y_3m"]
    definitions["term_spread_10y_2y"] = {"source_series": ["DGS10", "DGS2"], "operation": "DGS10_value - DGS2_value", "units": "percentage points"}
    definitions["term_spread_10y_3m"] = {"source_series": ["DGS10", "DGS3MO"], "operation": "DGS10_value - DGS3MO_value", "units": "percentage points"}
    return result, feature_columns, definitions


def write_fred_observations(source_tables: dict[str, pd.DataFrame], path: Path) -> None:
    frames = [source_tables[series] for series in SERIES]
    observations = pd.concat(frames, ignore_index=True)
    observations["observation_date"] = observations["observation_date"].dt.strftime("%Y-%m-%d")
    observations["value_numeric"] = observations["value_numeric"].where(observations["value_numeric"].notna(), np.nan)
    observations.to_csv(path, index=False, na_rep="")


def write_quarantine(profile: dict[str, object], path: Path) -> None:
    frames = []
    for key in ("invalid_rows_frame", "duplicate_rows_frame"):
        frame = profile[key]
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=["Date", "spy_session", "reason_code"]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run-id", required=True)
    parser.add_argument("--spy-path", default=str(RAW_BENCHMARK_DEFAULT))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    raw_dir = ROOT / "data" / "raw" / "daily_macro_v1" / args.raw_run_id
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"raw manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("series", [])) != SERIES or manifest.get("returned_count") != len(SERIES):
        raise ValueError("raw manifest does not contain the complete expected seven-series run")
    spy_path = Path(args.spy_path)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = ROOT / "data" / "clean" / "daily_macro_v1" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    source_tables: dict[str, pd.DataFrame] = {}
    source_profiles: dict[str, dict[str, object]] = {}
    for series in SERIES:
        manifest_row = next(item for item in manifest["series_results"] if item.get("series") == series)
        source_path = raw_dir / str(manifest_row["response_file"])
        if sha256_file(source_path) != manifest_row.get("sha256"):
            raise ValueError(f"raw hash mismatch before cleaning: {source_path}")
        table = load_fred(source_path, series)
        source_tables[series] = table
        source_profiles[series] = {
            "rows": int(len(table)),
            "instruments": 1,
            "date_min": table["observation_date"].min().date().isoformat(),
            "date_max": table["observation_date"].max().date().isoformat(),
            "numeric_rows": int(table["value_numeric"].notna().sum()),
            "provider_missing_rows": int(table["provider_missing"].sum()),
            "first_valid_date": table.loc[table["value_numeric"].notna(), "observation_date"].min().date().isoformat(),
            "raw_path": str((raw_dir / str(manifest_row["response_file"])).relative_to(ROOT)),
            "raw_sha256": manifest_row.get("sha256"),
        }

    sessions, spy_profile = load_spy_sessions(spy_path)
    if sessions.empty:
        raise ValueError("no valid SPY sessions")
    for series in SERIES:
        aligned_piece = effective_source_asof(sessions, source_tables[series], series)
        drop_cols = ["reference_spy_session"]
        if series == SERIES[0]:
            aligned = aligned_piece.copy()
        else:
            aligned = aligned.merge(aligned_piece.drop(columns=drop_cols), on="spy_session", how="left", sort=False)
    aligned["spy_session_index"] = np.arange(len(aligned), dtype=int)
    aligned, feature_columns, definitions = add_features(aligned)

    # Keep source trace columns first, then engineered features for simple downstream use.
    source_trace_columns = ["spy_session", "reference_spy_session", "spy_session_index"]
    for series in SERIES:
        source_trace_columns.extend(
            [
                f"{series}_source_observation_date",
                f"{series}_source_value_raw",
                f"{series}_source_value_numeric",
                f"{series}_source_missing",
                f"{series}_source_row",
                f"{series}_value_observation_date",
                f"{series}_value_source_row",
                f"{series}_value",
                f"{series}_first_available_date",
                f"{series}_carry_age_calendar_days",
                f"{series}_carry_age_spy_sessions",
                f"{series}_reason_code",
            ]
        )
    ordered_columns = source_trace_columns + feature_columns
    macro = aligned[ordered_columns].copy()
    for col in macro.columns:
        if col.endswith("_date") or col in {"spy_session", "reference_spy_session"}:
            if pd.api.types.is_datetime64_any_dtype(macro[col]):
                macro[col] = macro[col].dt.strftime("%Y-%m-%d")
    macro.to_csv(out_dir / "macro_features.csv", index=False, na_rep="")
    train_valid = macro.copy()
    session_dt = pd.to_datetime(train_valid["spy_session"], format="%Y-%m-%d")
    train_valid = train_valid[(session_dt >= TRAIN_START) & (session_dt <= VALIDATION_END)].copy()
    train_valid.to_csv(out_dir / "macro_features_train_valid.csv", index=False, na_rep="")

    write_fred_observations(source_tables, out_dir / "fred_observations.csv")
    sessions_out = sessions.copy()
    sessions_out["spy_session"] = sessions_out["spy_session"].dt.strftime("%Y-%m-%d")
    sessions_out.to_csv(out_dir / "spy_sessions.csv", index=False)
    quarantine_path = out_dir / "spy_sessions_quarantine.csv"
    write_quarantine(spy_profile, quarantine_path)

    recommended = [feature for feature in feature_columns if not feature.startswith("high_yield_oas") and not feature.startswith("investment_grade_oas")]
    credit_columns = [feature for feature in feature_columns if feature.startswith("high_yield_oas") or feature.startswith("investment_grade_oas")]
    definitions_payload = {
        "schema": "daily_macro_features_v1",
        "lag_rule": "for SPY session t, as-of reference_spy_session=t-1; source dates must be <= reference_spy_session",
        "carry_rule": "carry the latest valid source observation forward only after its observation date; provider raw missing tokens remain NA and are separately flagged",
        "change_rule": "simple difference in source native units over prior SPY sessions; no percent conversion",
        "percentile_rule": "VIX 252-session rolling percentile rank including the current lagged value; requires 252 observations",
        "feature_columns": feature_columns,
        "recommended_train_validation_feature_columns": recommended,
        "credit_spread_feature_columns_excluded_from_current_train_validation": credit_columns,
        "credit_spread_exclusion_reason": "TRAINING_COVERAGE_UNAVAILABLE: BAMLH0A0HYM2 and BAMLC0A0CM first valid source observation is 2023-09-11, after the 2015-2020 training and 2021-2022 validation windows",
        "definitions": definitions,
    }
    (out_dir / "feature_definitions.json").write_text(json.dumps(definitions_payload, indent=2), encoding="utf-8")

    input_hashes = {"raw_manifest": sha256_file(manifest_path), "spy_benchmark": sha256_file(spy_path)}
    for series in SERIES:
        input_hashes[f"raw_{series}"] = source_profiles[series]["raw_sha256"]
    config = {
        "schema": "daily_macro_feature_build_v1",
        "run_id": run_id,
        "built_at_utc": utc_now(),
        "raw_collection_run_id": args.raw_run_id,
        "raw_collection_path": str(raw_dir.relative_to(ROOT)),
        "spy_benchmark_path": str(spy_path.relative_to(ROOT)) if spy_path.is_relative_to(ROOT) else str(spy_path),
        "input_sha256": input_hashes,
        "training_window": [TRAIN_START.date().isoformat(), "2020-12-31"],
        "validation_window": ["2021-01-01", VALIDATION_END.date().isoformat()],
        "test_target_policy": "no test target or model result was read",
        "missingness_policy": "no imputation; provider missing raw tokens retained; carried aligned value only from prior valid observation",
        "units_policy": "retain source native units; DGS/OAS levels and changes are percentage points as supplied by FRED; dollar/VIX values retain source units",
    }
    (out_dir / "build_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    source_coverage = {}
    for series in SERIES:
        value_col = f"{series}_value"
        train_mask = (pd.to_datetime(macro["spy_session"]) >= TRAIN_START) & (pd.to_datetime(macro["spy_session"]) <= pd.Timestamp("2020-12-31"))
        validation_mask = (pd.to_datetime(macro["spy_session"]) >= pd.Timestamp("2021-01-01")) & (pd.to_datetime(macro["spy_session"]) <= VALIDATION_END)
        source_coverage[series] = {
            "first_valid_source_date": source_profiles[series]["first_valid_date"],
            "train_rows": int(train_mask.sum()),
            "train_rows_with_value": int(macro.loc[train_mask, value_col].notna().sum()),
            "validation_rows": int(validation_mask.sum()),
            "validation_rows_with_value": int(macro.loc[validation_mask, value_col].notna().sum()),
            "status": "TRAINING_COVERAGE_UNAVAILABLE" if series in {"BAMLH0A0HYM2", "BAMLC0A0CM"} else "available_with_source_missingness",
        }
    summary = {
        "schema": "daily_macro_features_summary_v1",
        "run_id": run_id,
        "built_at_utc": config["built_at_utc"],
        "status": "built_with_credit_spread_training_coverage_unavailable",
        "inputs": config,
        "counts": {
            "spy_raw_rows": spy_profile["raw_rows"],
            "spy_valid_unique_sessions": spy_profile["valid_unique_sessions"],
            "spy_invalid_rows_quarantined": spy_profile["invalid_rows"],
            "spy_duplicate_rows_quarantined": spy_profile["duplicate_rows"],
            "macro_output_rows": int(len(macro)),
            "macro_train_validation_rows": int(len(train_valid)),
            "macro_output_instruments": 1,
            "source_rows_total": int(sum(profile["rows"] for profile in source_profiles.values())),
        },
        "source_profiles": source_profiles,
        "spy_profile": {key: value for key, value in spy_profile.items() if not key.endswith("_frame")},
        "source_coverage": source_coverage,
        "feature_columns": feature_columns,
        "recommended_train_validation_feature_columns": recommended,
        "excluded_credit_spread_features": credit_columns,
        "quarantine": str(quarantine_path.relative_to(ROOT)),
        "deletions": {"source_rows": 0, "spy_rows_from_valid_output": int(spy_profile["invalid_rows"] + spy_profile["duplicate_rows"]), "reason": "invalid/duplicate SPY session keys excluded from the unique alignment key and retained in quarantine; no source observation row deleted"},
        "checks_expected_for_independent_validation": [
            "raw FRED hashes remain equal to collection manifest",
            "macro rows equal unique valid SPY sessions",
            "reference_spy_session is strictly prior for all non-first sessions",
            "no source/value observation date exceeds reference_spy_session",
            "carry ages are non-negative and reason codes agree",
            "credit spread train/validation values are absent and flagged TRAINING_COVERAGE_UNAVAILABLE",
        ],
        "limitations": [
            "FRED current CSV is not a complete vintage database; revision risk remains.",
            "The endpoint does not expose release timestamps in this raw CSV; the one-SPY-session lag is a conservative timing rule, not a release-time proof.",
            "Credit spread series begin in 2023-09 and are retained for later exploratory use but excluded from the current train/validation feature recommendation.",
        ],
    }
    (out_dir / "clean_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "out_dir": str(out_dir), "macro_rows": len(macro), "train_validation_rows": len(train_valid), "recommended_features": len(recommended), "excluded_credit_features": len(credit_columns)}, indent=2))


if __name__ == "__main__":
    main()
