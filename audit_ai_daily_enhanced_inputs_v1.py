"""Independent audit of the inputs for the enhanced daily AI model.

This script is intentionally read-only with respect to all input data.  It
does not open targets_test_sealed.csv, targets_dev.csv, model-run artifacts,
or the model runner.  It writes only a run-scoped audit artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
MISSING_RATE_THRESHOLD = 0.05
ROOT = Path(__file__).resolve().parent
BASE_RUN = "20260910T052100000000Z"
MACRO_RUN = "20260910T062700000000Z"
AI_RUN = "20260910T144000000000Z"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_dates(frame: pd.DataFrame, column: str) -> None:
    if column in frame.columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    category: str,
    status: str,
    blocking: bool,
    rows_checked: int,
    failures: int,
    details: str,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "category": category,
            "status": status,
            "blocking": bool(blocking),
            "rows_checked": int(rows_checked),
            "failures": int(failures),
            "details": details,
        }
    )


def date_range(series: pd.Series) -> list[Any]:
    values = pd.to_datetime(series, errors="coerce").dropna()
    if values.empty:
        return [None, None]
    return [values.min().strftime("%Y-%m-%d"), values.max().strftime("%Y-%m-%d")]


def nonfinite_count(frame: pd.DataFrame, columns: list[str]) -> int:
    count = 0
    for column in columns:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna().to_numpy(dtype=float)
        count += int((~np.isfinite(values)).sum())
    return count


def same_or_both_na(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def previous_session_check(
    frame: pd.DataFrame,
    formation_column: str,
    reference_column: str,
    previous_map: dict[pd.Timestamp, pd.Timestamp],
    allow_first_without_reference: bool = False,
) -> tuple[int, int, int]:
    expected = frame[formation_column].map(previous_map)
    if allow_first_without_reference:
        missing_expected = expected.isna()
        matches = same_or_both_na(frame[reference_column], expected)
        return int(len(frame)), int((~matches).sum()), int(missing_expected.sum())
    valid_expected = expected.notna()
    matches = frame[reference_column].eq(expected)
    invalid = int((valid_expected & ~matches).sum() + (~valid_expected).sum())
    return int(len(frame)), invalid, int((~valid_expected).sum())


def source_asof_counts(macro: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    reference = macro["reference_spy_session"]
    for series in ["VIXCLS", "DGS3MO", "DGS2", "DGS10", "BAMLH0A0HYM2", "BAMLC0A0CM", "DTWEXBGS"]:
        for suffix in ["source_observation_date", "value_observation_date"]:
            column = f"{series}_{suffix}"
            if column not in macro.columns:
                continue
            values = pd.to_datetime(macro[column], errors="coerce")
            valid = values.notna() & reference.notna()
            counts[column] = int((values[valid].to_numpy() > reference[valid].to_numpy()).sum())
    return counts


def missing_profile(
    frame: pd.DataFrame,
    feature_columns: list[str],
    source: str,
    scope: str,
    classification_fn,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        if feature not in frame.columns:
            rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "anchor_scope": scope,
                    "rows": int(len(frame)),
                    "missing": int(len(frame)),
                    "missing_fraction": 1.0,
                    "all_missing": True,
                    "threshold_5pct_pass": False,
                    "classification": "missing_column",
                    "rationale": "feature column absent after join",
                }
            )
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        missing = int(values.isna().sum())
        fraction = float(values.isna().mean()) if len(frame) else 1.0
        classification, rationale = classification_fn(feature, fraction, bool(values.isna().all()))
        rows.append(
            {
                "source": source,
                "feature": feature,
                "anchor_scope": scope,
                "rows": int(len(frame)),
                "missing": missing,
                "missing_fraction": fraction,
                "all_missing": bool(values.isna().all()),
                "threshold_5pct_pass": bool(fraction <= MISSING_RATE_THRESHOLD),
                "classification": classification,
                "rationale": rationale,
            }
        )
    return rows


def main() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    started = utc_now()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[0-9TZ]+", run_id):
        raise ValueError("run id must contain only digits, T, and Z")

    base = ROOT / "data" / "model_ready_ai_pool_daily_v1" / BASE_RUN
    macro_root = ROOT / "data" / "clean" / "daily_macro_v1" / MACRO_RUN
    ai_root = ROOT / "data" / "clean" / "ai_daily_macro_features_v1" / AI_RUN
    base_config_path = ROOT / "ai_pool_daily_baseline_model_ready_v1_config.json"
    macro_defs_path = macro_root / "feature_definitions.json"
    macro_config_path = macro_root / "build_config.json"
    ai_config_path = ROOT / "ai_daily_macro_feature_v1_config.json"
    spec_path = ROOT / "AI_DAILY_MACRO_FEATURE_SPEC.md"

    paths: dict[str, Path] = {
        "base_model_features": base / "model_features.csv",
        "base_metadata": base / "metadata.csv",
        "base_eligibility": base / "eligibility.csv",
        "base_split_counts": base / "split_counts.csv",
        "base_feature_missingness": base / "feature_missingness.csv",
        "base_summary": base / "summary.json",
        "base_config": base_config_path,
        "macro_features": macro_root / "macro_features.csv",
        "macro_features_train_valid": macro_root / "macro_features_train_valid.csv",
        "macro_spy_sessions": macro_root / "spy_sessions.csv",
        "macro_clean_summary": macro_root / "clean_summary.json",
        "macro_build_config": macro_config_path,
        "macro_feature_definitions": macro_defs_path,
        "ai_state_features": ai_root / "ai_state_features.csv",
        "ai_pool_member_coverage": ai_root / "pool_member_coverage.csv",
        "ai_etf_coverage": ai_root / "etf_coverage.csv",
        "ai_summary": ai_root / "summary.json",
        "ai_independent_validation": ai_root / "independent_validation.json",
        "ai_config": ai_config_path,
        "ai_spec": spec_path,
        "audit_script": Path(__file__).resolve(),
    }
    absent = [name for name, path in paths.items() if not path.exists()]
    if absent:
        raise FileNotFoundError(f"missing configured input: {absent}")

    # The target files are intentionally absent from paths.  Reading the test
    # seal or development targets is outside this audit's scope.
    model_features = pd.read_csv(paths["base_model_features"], low_memory=False)
    metadata = pd.read_csv(paths["base_metadata"], low_memory=False)
    eligibility = pd.read_csv(paths["base_eligibility"], low_memory=False)
    split_counts = pd.read_csv(paths["base_split_counts"])
    feature_missingness = pd.read_csv(paths["base_feature_missingness"])
    macro = pd.read_csv(paths["macro_features"], low_memory=False)
    macro_train_valid = pd.read_csv(paths["macro_features_train_valid"], low_memory=False)
    spy_sessions = pd.read_csv(paths["macro_spy_sessions"])
    ai_state = pd.read_csv(paths["ai_state_features"], low_memory=False)
    pool_coverage = pd.read_csv(paths["ai_pool_member_coverage"], low_memory=False)
    etf_coverage = pd.read_csv(paths["ai_etf_coverage"], low_memory=False)
    base_summary = json.loads(paths["base_summary"].read_text(encoding="utf-8"))
    macro_summary = json.loads(paths["macro_clean_summary"].read_text(encoding="utf-8"))
    ai_summary = json.loads(paths["ai_summary"].read_text(encoding="utf-8"))
    ai_validation = json.loads(paths["ai_independent_validation"].read_text(encoding="utf-8"))
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    macro_definitions = json.loads(macro_defs_path.read_text(encoding="utf-8"))
    ai_config = json.loads(ai_config_path.read_text(encoding="utf-8"))

    for frame, columns in [
        (model_features, ["formation_session"]),
        (metadata, ["formation_session", "reference_session"]),
        (eligibility, ["formation_session"]),
        (macro, ["spy_session", "reference_spy_session"]),
        (macro_train_valid, ["spy_session"]),
        (spy_sessions, ["spy_session"]),
        (ai_state, ["formation_session", "reference_session"]),
        (pool_coverage, ["formation_session", "reference_session"]),
        (etf_coverage, ["formation_session", "reference_session"]),
    ]:
        for column in columns:
            parse_dates(frame, column)

    checks: list[dict[str, Any]] = []

    # Key uniqueness and one-to-one master joins.
    key_specs = [
        ("base_model_features_sample_id_unique", model_features, ["sample_id"]),
        ("base_model_features_security_date_unique", model_features, ["security_id", "formation_session"]),
        ("base_metadata_sample_id_unique", metadata, ["sample_id"]),
        ("base_metadata_security_date_unique", metadata, ["security_id", "formation_session"]),
        ("base_eligibility_sample_id_unique", eligibility, ["sample_id"]),
        ("macro_session_unique", macro, ["spy_session"]),
        ("macro_train_valid_session_unique", macro_train_valid, ["spy_session"]),
        ("ai_state_session_unique", ai_state, ["formation_session"]),
        ("ai_pool_coverage_key_unique", pool_coverage, ["formation_session", "Instrument"]),
        ("ai_etf_coverage_key_unique", etf_coverage, ["formation_session", "Instrument"]),
    ]
    for check_id, frame, key in key_specs:
        duplicate_rows = int(frame.duplicated(key).sum())
        add_check(
            checks,
            check_id,
            "key_uniqueness",
            "PASS" if duplicate_rows == 0 else "FAIL",
            True,
            len(frame),
            duplicate_rows,
            f"duplicate_rows={duplicate_rows}; key={key}",
        )

    sample_sets = {
        "model_features": set(model_features["sample_id"]),
        "metadata": set(metadata["sample_id"]),
        "eligibility": set(eligibility["sample_id"]),
    }
    master_set_diff = sum(len(sample_sets["model_features"] ^ sample_sets[name]) for name in ["metadata", "eligibility"])
    add_check(
        checks,
        "master_sample_id_sets_match",
        "key_uniqueness",
        "PASS" if master_set_diff == 0 else "FAIL",
        True,
        len(model_features),
        master_set_diff,
        f"symmetric_difference_sum={master_set_diff}; source tables=model_features,metadata,eligibility",
    )
    identifier_mismatch = int((model_features["security_id"].astype(str) != model_features["Instrument"].astype(str)).sum())
    identifier_mismatch += int((metadata["security_id"].astype(str) != metadata["Instrument"].astype(str)).sum())
    add_check(
        checks,
        "literal_security_identifier_consistency",
        "key_uniqueness",
        "PASS" if identifier_mismatch == 0 else "FAIL",
        True,
        len(model_features) + len(metadata),
        identifier_mismatch,
        f"security_id_vs_Instrument_mismatches={identifier_mismatch}; no name-based mapping applied",
    )

    # Calendar and split separation checks.
    spy_dates = sorted(spy_sessions["spy_session"].dropna().unique())
    previous_map = {spy_dates[index]: spy_dates[index - 1] for index in range(1, len(spy_dates))}
    macro_dates_ordered = sorted(macro["spy_session"].dropna().unique())
    macro_previous_map = {
        macro_dates_ordered[index]: macro_dates_ordered[index - 1]
        for index in range(1, len(macro_dates_ordered))
    }
    for check_id, frame, formation, reference, mapping, first_ok in [
        ("metadata_reference_strict_prior_spy_session", metadata, "formation_session", "reference_session", previous_map, False),
        ("macro_reference_strict_prior_spy_session", macro, "spy_session", "reference_spy_session", macro_previous_map, True),
        ("ai_state_reference_strict_prior_spy_session", ai_state, "formation_session", "reference_session", previous_map, False),
        ("pool_coverage_reference_strict_prior_spy_session", pool_coverage, "formation_session", "reference_session", previous_map, False),
        ("etf_coverage_reference_strict_prior_spy_session", etf_coverage, "formation_session", "reference_session", previous_map, False),
    ]:
        rows, failures, first_or_missing = previous_session_check(frame, formation, reference, mapping, first_ok)
        add_check(
            checks,
            check_id,
            "date_alignment",
            "PASS" if failures == 0 else "FAIL",
            True,
            rows,
            failures,
            f"reference_mismatches={failures}; no_reference_rows={first_or_missing}; strict prior rule",
        )

    ai_lag_failures = int((ai_state["reference_lag_sessions"] != 1).sum())
    add_check(
        checks,
        "ai_reference_lag_sessions_equal_one",
        "date_alignment",
        "PASS" if ai_lag_failures == 0 else "FAIL",
        True,
        len(ai_state),
        ai_lag_failures,
        "reference_lag_sessions must equal 1",
    )
    source_asof = source_asof_counts(macro)
    source_asof_failures = sum(source_asof.values())
    add_check(
        checks,
        "macro_source_observations_not_after_reference",
        "date_alignment",
        "PASS" if source_asof_failures == 0 else "FAIL",
        True,
        len(macro) * 14,
        source_asof_failures,
        f"future_source_date_counts={source_asof}; source/value observation dates <= reference when present",
    )

    model_by_date = model_features.groupby("formation_session")["split"].nunique(dropna=False)
    ai_by_date = ai_state.groupby("formation_session")["split"].nunique(dropna=False)
    split_ambiguity = int((model_by_date > 1).sum() + (ai_by_date > 1).sum())
    add_check(
        checks,
        "formation_date_has_single_split",
        "split_separation",
        "PASS" if split_ambiguity == 0 else "FAIL",
        True,
        len(model_by_date) + len(ai_by_date),
        split_ambiguity,
        f"ambiguous_model_or_ai_dates={split_ambiguity}",
    )
    split_windows = {
        "training": (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31")),
        "validation": (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31")),
        "test": (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-06-30")),
    }
    window_failures = 0
    window_details = []
    for split, (start, end) in split_windows.items():
        for name, frame, column in [("model", model_features, "formation_session"), ("ai", ai_state, "formation_session")]:
            subset = frame.loc[frame["split"].eq(split), column]
            bad = int((~subset.between(start, end)).sum())
            window_failures += bad
            window_details.append(f"{name}_{split}_bad={bad}")
    add_check(
        checks,
        "model_and_ai_split_windows",
        "split_separation",
        "PASS" if window_failures == 0 else "FAIL",
        True,
        len(model_features) + len(ai_state),
        window_failures,
        "; ".join(window_details),
    )
    model_dates = set(model_features["formation_session"])
    model_dev_dates = set(model_features.loc[model_features["split"].isin(["training", "validation"]), "formation_session"])
    model_test_dates = set(model_features.loc[model_features["split"].eq("test"), "formation_session"])
    macro_dates = set(macro["spy_session"])
    macro_tv_dates = set(macro_train_valid["spy_session"])
    ai_dates = set(ai_state["formation_session"])
    macro_tv_diff = len(model_dev_dates ^ macro_tv_dates)
    macro_test_overlap = len(model_test_dates & macro_tv_dates)
    full_macro_missing = len(model_dates - macro_dates)
    ai_date_diff = len(model_dates ^ ai_dates)
    add_check(
        checks,
        "macro_train_validation_date_set_matches_development",
        "split_separation",
        "PASS" if macro_tv_diff == 0 else "FAIL",
        True,
        len(model_dev_dates) + len(macro_tv_dates),
        macro_tv_diff,
        f"symmetric_difference={macro_tv_diff}; macro_train_valid_test_overlap={macro_test_overlap}",
    )
    add_check(
        checks,
        "macro_train_validation_excludes_test_dates",
        "split_separation",
        "PASS" if macro_test_overlap == 0 else "FAIL",
        True,
        len(macro_train_valid),
        macro_test_overlap,
        "train/validation macro subset contains no model test formation date",
    )
    add_check(
        checks,
        "full_macro_and_ai_cover_model_formation_dates",
        "split_separation",
        "PASS" if full_macro_missing + ai_date_diff == 0 else "FAIL",
        True,
        len(model_dates) * 2,
        full_macro_missing + ai_date_diff,
        f"macro_missing_dates={full_macro_missing}; ai_date_set_difference={ai_date_diff}",
    )

    # Feature columns and no target/future columns in feature inputs.
    base_feature_columns = list(base_config["control_features"]["stock_controls"]) + list(base_config["control_features"]["spy_state_controls"])
    macro_feature_columns = list(macro_definitions["feature_columns"])
    ai_feature_columns = [
        column
        for column in ai_state.columns
        if (
            (column.startswith("etf_") and any(token in column for token in ["momentum", "volatility", "max_drawdown"]))
            or column.startswith("soxx_relative_spy_return_")
            or column.startswith("spy_momentum_")
            or column
            in {
                "spy_return_1",
                "pool_breadth_up_1",
                "pool_breadth_above_ma20",
                "pool_breadth_above_ma60",
                "pool_cross_sectional_dispersion_1",
                "pool_equal_weight_return_1",
                "pool_equal_weight_relative_spy_return_1",
            }
        )
    ]
    missing_feature_columns = (
        [column for column in base_feature_columns if column not in model_features.columns]
        + [column for column in macro_feature_columns if column not in macro.columns]
        + [column for column in ai_feature_columns if column not in ai_state.columns]
    )
    add_check(
        checks,
        "configured_feature_columns_present",
        "feature_schema",
        "PASS" if not missing_feature_columns else "FAIL",
        True,
        len(base_feature_columns) + len(macro_feature_columns) + len(ai_feature_columns),
        len(missing_feature_columns),
        f"missing_columns={missing_feature_columns}",
    )
    forbidden_pattern = re.compile(r"(^|_)(y|label|target|forward|future)(_|$)", re.IGNORECASE)
    forbidden_feature_names = {
        source: sorted(column for column in columns if forbidden_pattern.search(column))
        for source, columns in {
            "base_model_features": model_features.columns,
            "macro_features": macro_feature_columns,
            "ai_state_features": ai_feature_columns,
        }.items()
    }
    forbidden_count = sum(len(columns) for columns in forbidden_feature_names.values())
    add_check(
        checks,
        "feature_tables_have_no_target_or_future_named_columns",
        "leakage_guard",
        "PASS" if forbidden_count == 0 else "FAIL",
        True,
        len(model_features.columns) + len(macro_feature_columns) + len(ai_feature_columns),
        forbidden_count,
        f"forbidden_feature_names={forbidden_feature_names}",
    )
    nonfinite_features = (
        nonfinite_count(model_features, base_feature_columns)
        + nonfinite_count(macro, macro_feature_columns)
        + nonfinite_count(ai_state, ai_feature_columns)
    )
    add_check(
        checks,
        "numeric_feature_values_are_finite_when_present",
        "feature_schema",
        "PASS" if nonfinite_features == 0 else "FAIL",
        True,
        len(model_features) * len(base_feature_columns) + len(macro) * len(macro_feature_columns) + len(ai_state) * len(ai_feature_columns),
        nonfinite_features,
        "NA values are allowed; +/-inf is not",
    )

    # Build the primary non-overlap anchor frame.  The join uses only
    # development eligibility and feature tables; it never needs target values.
    anchors = eligibility.loc[
        eligibility["split"].isin(["training", "validation"])
        & eligibility["non_overlap_selected"].astype(bool)
    ].copy()
    anchor_ids = set(anchors["sample_id"])
    anchor_model = model_features.loc[model_features["sample_id"].isin(anchor_ids), ["sample_id", "security_id", "formation_session", "split"] + base_feature_columns].copy()
    ai_join_columns = ["formation_session", "reference_session", "split"] + ai_feature_columns
    ai_join = ai_state[ai_join_columns].rename(columns={column: f"ai__{column}" for column in ai_feature_columns})
    ai_join = ai_join.rename(columns={"reference_session": "ai__reference_session", "split": "ai__split"})
    macro_join_columns = ["spy_session", "reference_spy_session"] + macro_feature_columns
    macro_join = macro_train_valid[macro_join_columns].rename(columns={"spy_session": "formation_session", "reference_spy_session": "macro__reference_session", **{column: f"macro__{column}" for column in macro_feature_columns}})
    anchor_frame = anchor_model.merge(anchors[["sample_id", "non_overlap_selected"]], on="sample_id", how="inner", validate="one_to_one")
    anchor_frame = anchor_frame.merge(ai_join, on="formation_session", how="left", validate="many_to_one")
    anchor_frame = anchor_frame.merge(macro_join, on="formation_session", how="left", validate="many_to_one")
    anchor_join_failures = int(len(anchors) - len(anchor_frame))
    anchor_ai_missing_dates = int(anchor_frame["ai__reference_session"].isna().sum())
    anchor_macro_missing_dates = int(anchor_frame["macro__reference_session"].isna().sum())
    add_check(
        checks,
        "primary_non_overlap_anchor_join_is_complete",
        "anchor_integrity",
        "PASS" if anchor_join_failures + anchor_ai_missing_dates + anchor_macro_missing_dates == 0 else "FAIL",
        True,
        len(anchors),
        anchor_join_failures + anchor_ai_missing_dates + anchor_macro_missing_dates,
        f"anchors={len(anchors)}; ai_missing_anchor_dates={anchor_ai_missing_dates}; macro_missing_anchor_dates={anchor_macro_missing_dates}",
    )
    anchor_duplicate_keys = int(anchor_frame.duplicated(["security_id", "formation_session"]).sum())
    add_check(
        checks,
        "primary_non_overlap_anchor_security_date_unique",
        "anchor_integrity",
        "PASS" if anchor_duplicate_keys == 0 else "FAIL",
        True,
        len(anchor_frame),
        anchor_duplicate_keys,
        "anchor key is literal security_id + formation_session",
    )

    # Exact and formula-level control overlap.
    base_ai_exact = sorted(set(base_feature_columns) & set(ai_feature_columns))
    base_macro_exact = sorted(set(base_feature_columns) & set(macro_feature_columns))
    duplicate_details = []
    if base_ai_exact:
        duplicate_details.append(f"exact_base_ai={base_ai_exact}")
    if base_macro_exact:
        duplicate_details.append(f"exact_base_macro={base_macro_exact}")
    add_check(
        checks,
        "feature_name_overlap_reviewed",
        "duplicate_controls",
        "WARN" if (base_ai_exact or base_macro_exact) else "PASS",
        False,
        len(base_feature_columns) + len(macro_feature_columns) + len(ai_feature_columns),
        len(base_ai_exact) + len(base_macro_exact),
        "; ".join(duplicate_details) if duplicate_details else "no exact feature-name overlap",
    )
    formula_overlap: dict[str, Any] = {}
    for base_name, ai_name in [
        ("spy_momentum_5", "spy_momentum_5"),
        ("spy_momentum_20", "spy_momentum_20"),
        ("spy_momentum_60", "spy_momentum_60"),
        ("spy_momentum_1", "spy_return_1"),
    ]:
        ai_col = f"ai__{ai_name}"
        if base_name not in anchor_frame.columns or ai_col not in anchor_frame.columns:
            continue
        left = pd.to_numeric(anchor_frame[base_name], errors="coerce")
        right = pd.to_numeric(anchor_frame[ai_col], errors="coerce")
        diff = (left - right).abs()
        formula_overlap[f"{base_name}__{ai_name}"] = {
            "comparable_rows": int(diff.notna().sum()),
            "nonzero_rows_at_1e-12": int((diff > 1e-12).sum()),
            "max_abs_difference": float(diff.max(skipna=True)) if diff.notna().any() else None,
        }
    formula_issue = int(sum(item["nonzero_rows_at_1e-12"] for item in formula_overlap.values()))
    add_check(
        checks,
        "formula_equivalent_spy_controls_reviewed",
        "duplicate_controls",
        "WARN" if formula_issue else "PASS",
        False,
        len(anchor_frame),
        formula_issue,
        f"comparisons={formula_overlap}; ai_spy_return_1 uses close-derived return while baseline spy_momentum_1 uses returns input",
    )

    # Missingness profiles on the primary anchor set, split by training,
    # validation, and pooled anchors.
    profiles: list[dict[str, Any]] = []
    anchor_scopes = [("training", anchor_frame.loc[anchor_frame["split"].eq("training")]), ("validation", anchor_frame.loc[anchor_frame["split"].eq("validation")]), ("all_anchors", anchor_frame)]

    def base_classification(feature: str, fraction: float, all_missing: bool) -> tuple[str, str]:
        if all_missing or fraction > MISSING_RATE_THRESHOLD:
            return "exclude_high_missing", "baseline control exceeds 5% anchor missingness"
        return "baseline_control", "configured controls-only baseline field"

    def macro_classification(feature: str, fraction: float, all_missing: bool) -> tuple[str, str]:
        if all_missing:
            return "exclude_all_missing", "credit-spread source begins after the development window"
        if fraction > MISSING_RATE_THRESHOLD:
            return "exclude_high_missing", "macro feature exceeds 5% anchor missingness"
        return "safe_candidate", "strictly lagged macro feature with usable anchor coverage"

    def ai_classification(feature: str, fraction: float, all_missing: bool) -> tuple[str, str]:
        if feature in base_ai_exact:
            return "exclude_duplicate_control", "exact name overlap with baseline SPY control"
        if feature == "spy_return_1":
            return "exclude_duplicate_control", "one-session SPY control is formula-equivalent in intent but source differs"
        if all_missing:
            return "exclude_all_missing", "AI/ETF state unavailable on every primary anchor"
        if fraction > MISSING_RATE_THRESHOLD:
            return "exclude_high_missing", "ETF history/inception coverage exceeds 5% anchor missingness"
        return "safe_candidate", "strictly prior-session AI industry state with usable anchor coverage"

    for scope, frame in anchor_scopes:
        base_frame = frame[base_feature_columns]
        macro_frame = frame[[f"macro__{column}" for column in macro_feature_columns]].rename(columns={f"macro__{column}": column for column in macro_feature_columns})
        ai_frame = frame[[f"ai__{column}" for column in ai_feature_columns]].rename(columns={f"ai__{column}": column for column in ai_feature_columns})
        profiles.extend(missing_profile(base_frame, base_feature_columns, "base", scope, base_classification))
        profiles.extend(missing_profile(macro_frame, macro_feature_columns, "macro", scope, macro_classification))
        profiles.extend(missing_profile(ai_frame, ai_feature_columns, "ai", scope, ai_classification))
    missingness = pd.DataFrame(profiles)

    pooled = missingness.loc[missingness["anchor_scope"].eq("all_anchors")]
    macro_all_missing = sorted(pooled.loc[(pooled["source"] == "macro") & pooled["all_missing"], "feature"].tolist())
    ai_all_missing = sorted(pooled.loc[(pooled["source"] == "ai") & pooled["all_missing"], "feature"].tolist())
    macro_high_missing = sorted(pooled.loc[(pooled["source"] == "macro") & (pooled["missing_fraction"] > MISSING_RATE_THRESHOLD), "feature"].tolist())
    ai_high_missing = sorted(pooled.loc[(pooled["source"] == "ai") & (pooled["missing_fraction"] > MISSING_RATE_THRESHOLD), "feature"].tolist())
    base_high_missing = sorted(pooled.loc[(pooled["source"] == "base") & (pooled["missing_fraction"] > MISSING_RATE_THRESHOLD), "feature"].tolist())
    add_check(
        checks,
        "macro_anchor_all_missing_features_identified",
        "missingness",
        "FAIL" if macro_all_missing else "PASS",
        False,
        len(anchor_frame) * len(macro_feature_columns),
        len(macro_all_missing),
        f"all_missing_features={macro_all_missing}; exclude from development candidates",
    )
    add_check(
        checks,
        "ai_anchor_high_missing_features_identified",
        "missingness",
        "FAIL" if ai_high_missing else "PASS",
        False,
        len(anchor_frame) * len(ai_feature_columns),
        len(ai_high_missing),
        f"over_5pct_features={ai_high_missing}; ETF history availability is retained as NA",
    )
    add_check(
        checks,
        "base_anchor_missingness_within_threshold",
        "missingness",
        "FAIL" if base_high_missing else "PASS",
        False,
        len(anchor_frame) * len(base_feature_columns),
        len(base_high_missing),
        f"over_5pct_features={base_high_missing}; no new imputation applied",
    )

    # Membership and audit-only fields are deliberately not part of the safe
    # model lists.  Preserve the upstream static-registry caveat.
    registry_status = metadata["registry_pit_evidence_status"].value_counts(dropna=False).to_dict()
    registry_instruments = metadata.groupby("registry_pit_evidence_status")["Instrument"].nunique(dropna=False).to_dict()
    provisional_rows = int((metadata["registry_pit_evidence_status"] == "provisional_static_source").sum())
    add_check(
        checks,
        "membership_static_registry_limitation_disclosed",
        "membership_selection",
        "WARN" if provisional_rows else "PASS",
        False,
        len(metadata),
        provisional_rows,
        f"registry_status_rows={registry_status}; registry_status_instruments={registry_instruments}; provisional rows are not full historical PIT evidence",
    )
    add_check(
        checks,
        "test_target_values_not_read_by_audit",
        "test_separation",
        "PASS",
        True,
        0,
        0,
        "targets_test_sealed.csv is intentionally absent from read paths; no test targets, labels, returns, or model runs opened",
    )
    add_check(
        checks,
        "audit_does_not_apply_imputation_or_row_deletion",
        "processing_policy",
        "PASS",
        True,
        len(model_features) + len(macro) + len(ai_state),
        0,
        "read-only audit; no imputation, forward/backward fill, winsorization, deduplication, unit change, timezone change, or input row deletion",
    )

    # Safe lists require <=5% missingness in both split-specific anchor scopes.
    def safe_after_split_gate(source: str, columns: list[str]) -> list[str]:
        safe: list[str] = []
        for column in columns:
            rows = missingness.loc[(missingness["source"] == source) & (missingness["feature"] == column)]
            split_rows = rows.loc[rows["anchor_scope"].isin(["training", "validation"])]
            if rows.empty or len(split_rows) != 2:
                continue
            if bool((split_rows["missing_fraction"] <= MISSING_RATE_THRESHOLD).all()):
                safe.append(column)
        return safe

    base_safe = safe_after_split_gate("base", base_feature_columns)
    macro_safe = safe_after_split_gate("macro", macro_feature_columns)
    ai_safe = safe_after_split_gate("ai", ai_feature_columns)
    ai_safe = [column for column in ai_safe if column not in base_ai_exact and column != "spy_return_1"]
    macro_excluded = [column for column in macro_feature_columns if column not in macro_safe]
    ai_excluded = [column for column in ai_feature_columns if column not in ai_safe]
    ai_diagnostic_columns = [
        column
        for column in ai_state.columns
        if column not in ["formation_session", "reference_session", "split"] + ai_feature_columns
    ]
    base_audit_only = [column for column in ["feature_missing_count", "feature_missing_list"] if column in model_features.columns]
    macro_audit_only = [column for column in macro.columns if column not in macro_feature_columns and column not in ["spy_session", "reference_spy_session"]]
    derived_relationships = [
        "macro.term_spread_10y_2y = dgs10_level - dgs2_level",
        "macro.term_spread_10y_3m = dgs10_level - dgs3mo_level",
        "ai.soxx_relative_spy_return_{5,20,60} = ai.etf_soxx_momentum_{5,20,60} - ai.spy_momentum_{5,20,60}; source is close-derived",
        "ai.pool_equal_weight_relative_spy_return_1 = ai.pool_equal_weight_return_1 - ai.spy_return_1",
    ]

    feature_inventory_rows: list[dict[str, Any]] = []
    inventory_groups = [
        ("base", base_feature_columns, base_safe),
        ("macro", macro_feature_columns, macro_safe),
        ("ai", ai_feature_columns, ai_safe),
    ]
    for source, columns, safe_columns in inventory_groups:
        source_rows = missingness.loc[(missingness["source"] == source) & (missingness["anchor_scope"].eq("all_anchors"))].set_index("feature")
        for column in columns:
            profile = source_rows.loc[column] if column in source_rows.index else {}
            reason = str(profile.get("rationale", "")) if isinstance(profile, dict) else str(profile["rationale"])
            classification = str(profile.get("classification", "")) if isinstance(profile, dict) else str(profile["classification"])
            feature_inventory_rows.append(
                {
                    "source": source,
                    "feature": column,
                    "numeric_feature": True,
                    "recommended_incremental": column in safe_columns,
                    "anchor_missing_fraction_all": float(profile.get("missing_fraction", np.nan)) if isinstance(profile, dict) else float(profile["missing_fraction"]),
                    "classification": classification,
                    "rationale": reason,
                    "exact_base_overlap": column in base_ai_exact or column in base_macro_exact,
                    "formula_or_derived_relationship": column in {"spy_return_1", "soxx_relative_spy_return_5", "soxx_relative_spy_return_20", "soxx_relative_spy_return_60", "pool_equal_weight_relative_spy_return_1", "term_spread_10y_2y", "term_spread_10y_3m"},
                }
            )
    for source, columns, rationale in [
        ("base_audit_only", base_audit_only, "missingness audit metadata; do not pass as a model feature"),
        ("macro_audit_only", macro_audit_only, "raw/provenance/carry/reason metadata; retained for audit, not model input"),
        ("ai_diagnostic_only", ai_diagnostic_columns, "date, coverage, count, boolean, or reason-code diagnostic; not a primary economic feature"),
    ]:
        for column in columns:
            feature_inventory_rows.append(
                {
                    "source": source,
                    "feature": column,
                    "numeric_feature": False,
                    "recommended_incremental": False,
                    "anchor_missing_fraction_all": np.nan,
                    "classification": "exclude_audit_or_metadata",
                    "rationale": rationale,
                    "exact_base_overlap": False,
                    "formula_or_derived_relationship": False,
                }
            )
    feature_inventory = pd.DataFrame(feature_inventory_rows)

    # Hash only the explicitly read inputs.  The sealed test target path is
    # neither hashed nor opened.
    input_hashes = {
        name: {"path": rel(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if name != "audit_script"
    }
    script_hash = sha256_file(paths["audit_script"])

    output_dir = ROOT / "data" / "audit" / "ai_daily_enhanced_inputs_v1" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    checks_path = output_dir / "checks.csv"
    missingness_path = output_dir / "anchor_feature_missingness.csv"
    inventory_path = output_dir / "feature_inventory.csv"
    summary_path = output_dir / "summary.json"
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    feature_inventory.to_csv(inventory_path, index=False)

    blocking_failures = checks_df.loc[(checks_df["status"] == "FAIL") & checks_df["blocking"].astype(bool), "check_id"].tolist()
    status_counts = checks_df["status"].value_counts().to_dict()
    overall_status = "PASS" if not blocking_failures else "FAIL_BLOCKING"
    summary = {
        "schema_version": "ai_daily_enhanced_inputs_audit_v1",
        "audit_script_version": SCRIPT_VERSION,
        "audit_script_sha256": script_hash,
        "run_id": run_id,
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "status": overall_status,
        "overall_pass": not blocking_failures,
        "blocking_failures": blocking_failures,
        "check_status_counts": status_counts,
        "input_versions": {
            "base_model_ready": {"run_id": BASE_RUN, "path": rel(base)},
            "macro_clean": {"run_id": MACRO_RUN, "path": rel(macro_root)},
            "ai_state_clean": {"run_id": AI_RUN, "path": rel(ai_root)},
        },
        "input_hashes": input_hashes,
        "input_rows_and_instruments": {
            "base_model_features": {"rows": len(model_features), "instruments": model_features["Instrument"].nunique(), "formation_sessions": model_features["formation_session"].nunique(), "date_range": date_range(model_features["formation_session"])},
            "base_metadata": {"rows": len(metadata), "instruments": metadata["Instrument"].nunique(), "formation_sessions": metadata["formation_session"].nunique(), "date_range": date_range(metadata["formation_session"])},
            "base_eligibility": {"rows": len(eligibility), "instruments": eligibility["Instrument"].nunique(), "formation_sessions": eligibility["formation_session"].nunique(), "date_range": date_range(eligibility["formation_session"])},
            "macro_features": {"rows": len(macro), "instruments": 1, "formation_sessions": macro["spy_session"].nunique(), "date_range": date_range(macro["spy_session"])},
            "macro_features_train_valid": {"rows": len(macro_train_valid), "instruments": 1, "formation_sessions": macro_train_valid["spy_session"].nunique(), "date_range": date_range(macro_train_valid["spy_session"])},
            "ai_state_features": {"rows": len(ai_state), "instruments": ai_state["formation_session"].nunique(), "formation_sessions": ai_state["formation_session"].nunique(), "date_range": date_range(ai_state["formation_session"])},
            "ai_pool_member_coverage": {"rows": len(pool_coverage), "instruments": pool_coverage["Instrument"].nunique(), "formation_sessions": pool_coverage["formation_session"].nunique(), "date_range": date_range(pool_coverage["formation_session"])},
            "ai_etf_coverage": {"rows": len(etf_coverage), "instruments": etf_coverage["Instrument"].nunique(), "formation_sessions": etf_coverage["formation_session"].nunique(), "date_range": date_range(etf_coverage["formation_session"])},
        },
        "primary_non_overlap_anchors": {
            "rows": len(anchor_frame),
            "training_rows": int((anchor_frame["split"] == "training").sum()),
            "validation_rows": int((anchor_frame["split"] == "validation").sum()),
            "instruments": anchor_frame["security_id"].nunique(),
            "formation_sessions": anchor_frame["formation_session"].nunique(),
            "selection_rule": "eligibility.split in {training, validation} and non_overlap_selected == true",
            "missing_rate_threshold": MISSING_RATE_THRESHOLD,
        },
        "safe_feature_lists": {
            "baseline_controls_verified": base_safe,
            "macro_recommended_incremental": macro_safe,
            "ai_recommended_incremental": ai_safe,
            "combined_recommended_incremental": macro_safe + ai_safe,
            "macro_excluded_from_development": macro_excluded,
            "ai_excluded_from_incremental": ai_excluded,
            "exact_base_ai_overlap_excluded": base_ai_exact,
            "exact_base_macro_overlap": base_macro_exact,
        },
        "missingness_summary": {
            "base_over_5pct_all_anchors": base_high_missing,
            "macro_all_missing_all_anchors": macro_all_missing,
            "macro_over_5pct_all_anchors": macro_high_missing,
            "ai_all_missing_all_anchors": ai_all_missing,
            "ai_over_5pct_all_anchors": ai_high_missing,
        },
        "duplicate_or_derived_controls": {
            "exact_name_overlap_base_ai": base_ai_exact,
            "exact_name_overlap_base_macro": base_macro_exact,
            "formula_comparisons_on_anchors": formula_overlap,
            "derived_relationships_to_review": derived_relationships,
            "recommendation": "Do not add duplicate SPY controls; reconcile the close-derived AI spy_return_1/relative-return source with the returns-derived baseline before any combined model interpretation.",
        },
        "potential_leakage_or_lookahead_flags": {
            "target_like_feature_names": forbidden_feature_names,
            "metadata_target_reason_present_but_excluded": "target_reason exists in metadata/eligibility for audit only and was not joined as a model feature",
            "base_audit_only_fields": base_audit_only,
            "macro_audit_only_field_count": len(macro_audit_only),
            "ai_diagnostic_only_fields": ai_diagnostic_columns,
            "membership_registry_status_rows": registry_status,
            "membership_registry_status_instruments": registry_instruments,
            "membership_caveat": "provisional_static_source rows use exploratory static registry spans and are not full historical point-in-time AI-role evidence",
        },
        "processing_transparency": {
            "purpose": "Audit enhanced daily feature inputs before model integration without fitting models or reading test target values.",
            "rules": [
                "Use literal security_id/Instrument keys; no name mapping or issuer merge.",
                "Require formation date to map to the immediately prior SPY session for base metadata, macro, AI state, pool coverage, and ETF coverage.",
                "Require macro source and effective observation dates to be no later than the macro reference session.",
                "Use only non-overlap training/validation anchors for primary missingness screening.",
                "Use <=5% missingness in both training and validation anchor subsets as a conservative safe-candidate gate.",
                "Keep source missing values as NA; no imputation, winsorization, forward fill, backward fill, deduplication, unit conversion, timezone conversion, row deletion, or company exclusion was performed.",
                "Do not read targets_test_sealed.csv, targets_dev.csv, model runs, predictions, or metrics.",
            ],
            "row_effect": {"input_rows_removed": 0, "input_instruments_removed": 0, "quarantine": "none; audit-only exclusions are reason-coded in feature_inventory.csv"},
            "units_and_time": "Date-only comparisons; no timezone conversion or unit changes. Source units remain as supplied by upstream clean tables.",
        },
        "upstream_status_evidence": {
            "base_model_ready_declared_status": base_summary.get("status"),
            "base_model_ready_declared_checks": base_summary.get("checks", {}),
            "macro_clean_declared_status": macro_summary.get("status"),
            "ai_state_declared_status": ai_summary.get("status"),
            "ai_independent_validation_status": ai_validation.get("status"),
            "ai_independent_validation_failed_checks": ai_validation.get("counts", {}).get("failed_checks"),
        },
        "artifact_files": {
            "summary": rel(summary_path),
            "checks": rel(checks_path),
            "anchor_feature_missingness": rel(missingness_path),
            "feature_inventory": rel(inventory_path),
        },
        "limitations": [
            "This is an input audit only; no model was fitted and no model performance conclusion is made.",
            "Current FRED clean inputs are not a complete historical vintage database; release-time and revision risk remains an upstream limitation.",
            "The expanded AI pool includes provisional static registry evidence for historical membership; safe alignment does not make the resulting backtest a full-PIT unbiased AI-role study.",
            "Fields with high or all-missing anchors are excluded from the recommended incremental list; no imputation decision is implied.",
        ],
    }
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": overall_status, "output_dir": rel(output_dir), "blocking_failures": blocking_failures, "safe_macro": len(macro_safe), "safe_ai": len(ai_safe), "anchors": len(anchor_frame)}, ensure_ascii=False))
    return output_dir


if __name__ == "__main__":
    main()
