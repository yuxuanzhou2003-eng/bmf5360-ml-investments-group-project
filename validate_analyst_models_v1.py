"""Independent audit of the frozen analyst-model training/validation run.

The validator deliberately does not parse any row of the model-ready targets table.  It
uses the target column saved in the validation-only predictions output for independent
validation metrics, while the sealed test split is checked through eligibility/metadata
structure, output absence and the targets header only.  It does not load or score any
model artifact and does not modify the runner or model-ready data.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
RUN_ID = "20260909T172214155509Z"
RUN = ROOT / "data" / "model_runs" / "analyst_v1" / RUN_ID
MODEL_READY_ID = "20260909T125156092817Z"
MODEL_READY = ROOT / "data" / "analyst_graph_model_ready_v1" / MODEL_READY_ID
CONFIG_PATH = ROOT / "analyst_model_v1_config.json"
AUDIT_ROOT = ROOT / "data" / "audit" / "analyst_model_v1"
EVENT_KEY = ["source", "announcement", "period_end"]
NETWORK_FAMILIES = ["analyst_ridge", "analyst_hgb", "two_stage_ridge"]
MODEL_NAMES = [
    "training_weighted_mean",
    "simple_rec_weighted_signal",
    "receiver_only_ridge",
    "context_ridge",
    "analyst_ridge",
    "analyst_hgb",
    "two_stage_ridge",
]
KEY_METRICS = [
    "event_weighted_mse",
    "pooled_spearman",
    "mean_event_spearman",
    "event_spearman_groups",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    """Convert pandas/numpy values into stable JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    return value


def json_text(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_bool(values: pd.Series, name: str) -> pd.Series:
    normalized = pd.Series(values, index=values.index).astype("string").str.strip().str.lower()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name}: invalid boolean values {invalid}")
    return normalized.eq("true")


def safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if len(left_array) < 3 or len(right_array) < 3:
        return np.nan
    if len(np.unique(left_array)) < 2 or len(np.unique(right_array)) < 2:
        return np.nan
    value = spearmanr(left_array, right_array).statistic
    return float(value) if np.isfinite(value) else np.nan


def event_ic_records(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """Independently calculate one within-event Spearman value per event."""
    rows: list[dict[str, Any]] = []
    needed = EVENT_KEY + ["announcement_day", "target", prediction_column]
    for key, group in frame[needed].groupby(EVENT_KEY, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        target_unique = int(group["target"].nunique(dropna=True))
        prediction_unique = int(group[prediction_column].nunique(dropna=True))
        if len(group) < 3:
            value, eligible, reason = np.nan, False, "fewer_than_3_rows"
        elif target_unique < 2:
            value, eligible, reason = np.nan, False, "constant_target"
        elif prediction_unique < 2:
            value, eligible, reason = np.nan, False, "constant_prediction"
        else:
            value = safe_spearman(group["target"], group[prediction_column])
            eligible = bool(np.isfinite(value))
            reason = "ok" if eligible else "nonfinite_spearman"
        announcement_day = pd.Timestamp(group["announcement_day"].iloc[0]).normalize()
        row = dict(zip(EVENT_KEY, key))
        row.update(
            {
                "announcement_day": announcement_day,
                "announcement_month": announcement_day.strftime("%Y-%m"),
                "rows": int(len(group)),
                "target_unique": target_unique,
                "prediction_unique": prediction_unique,
                "event_ic": float(value) if np.isfinite(value) else np.nan,
                "ic_eligible": bool(eligible),
                "ic_reason": reason,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def independent_metrics(frame: pd.DataFrame, prediction_column: str) -> tuple[dict[str, Any], pd.DataFrame]:
    prediction = frame[prediction_column].to_numpy(float)
    target = frame["target"].to_numpy(float)
    counts = frame.groupby(EVENT_KEY, dropna=False, sort=False)["sample_id"].transform("size")
    weights = 1.0 / counts.to_numpy(float)
    error = prediction - target
    event_table = event_ic_records(frame, prediction_column)
    eligible_ics = event_table.loc[event_table["ic_eligible"], "event_ic"].to_numpy(float)
    pooled = safe_spearman(target, prediction)
    weighted_mse = float(np.average(error**2, weights=weights))
    result = {
        "rows": int(len(frame)),
        "events": int(frame[EVENT_KEY].drop_duplicates().shape[0]),
        "event_weighted_mse": weighted_mse,
        "pooled_spearman": float(pooled) if np.isfinite(pooled) else np.nan,
        "mean_event_spearman": float(np.mean(eligible_ics)) if len(eligible_ics) else np.nan,
        "event_spearman_groups": int(len(eligible_ics)),
    }
    return result, event_table


def values_match(expected: Any, observed: Any, *, integer: bool = False) -> bool:
    if integer:
        try:
            return int(expected) == int(observed)
        except (TypeError, ValueError):
            return False
    try:
        if pd.isna(expected) and pd.isna(observed):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.isclose(float(expected), float(observed), rtol=1e-11, atol=1e-13, equal_nan=True))
    except (TypeError, ValueError):
        return expected == observed


def resolve_root_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def expected_candidate_counts(config: dict[str, Any]) -> dict[str, int]:
    alphas = list(config["hyperparameters"]["ridge_alphas"])
    hgb = config["hyperparameters"]["hist_gradient_boosting_grid"]
    return {
        "training_weighted_mean": 1,
        "simple_rec_weighted_signal": 1,
        "receiver_only_ridge": len(alphas),
        "context_ridge": len(alphas),
        "analyst_ridge": len(alphas),
        "analyst_hgb": len(hgb["learning_rate"]) * len(hgb["max_leaf_nodes"]) * len(hgb["l2_regularization"]),
        "two_stage_ridge": len(alphas) ** 2,
    }


def make_audit_directory() -> tuple[str, Path]:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        audit_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = AUDIT_ROOT / audit_id
        try:
            output.mkdir(parents=False, exist_ok=False)
            return audit_id, output
        except FileExistsError:
            continue


def main() -> int:
    audit_id, audit_dir = make_audit_directory()
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, expected: Any, observed: Any, reason: str = "") -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "expected": jsonable(expected),
                "observed": jsonable(observed),
                "failure_reason": "" if passed else reason,
            }
        )

    run_summary_path = RUN / "summary.json"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    selected = json.loads((RUN / "selected_hyperparameters.json").read_text(encoding="utf-8"))

    declared_outputs = list(config["output"]["files"])
    output_paths = {name: RUN / name for name in declared_outputs}
    output_paths.update(
        {
            "models_directory": RUN / "models",
        }
    )
    required_models = list(config["models"])
    model_paths = {name: RUN / "models" / f"{name}.joblib" for name in required_models}

    existing_outputs = [name for name, path in output_paths.items() if path.exists()]
    missing_outputs = [name for name, path in output_paths.items() if not path.exists()]
    record(
        "output_files_complete",
        not missing_outputs,
        sorted(output_paths),
        sorted(existing_outputs),
        f"Missing declared run outputs: {missing_outputs}",
    )
    empty_outputs = [name for name, path in output_paths.items() if path.is_file() and path.stat().st_size == 0]
    record(
        "output_files_nonempty",
        not empty_outputs,
        [],
        empty_outputs,
        f"Empty output files: {empty_outputs}",
    )
    missing_models = [name for name, path in model_paths.items() if not path.exists()]
    record(
        "declared_model_artifacts_complete",
        not missing_models and (RUN / "models").is_dir(),
        required_models,
        [name for name, path in model_paths.items() if path.exists()],
        f"Missing model artifacts: {missing_models}",
    )

    # Hashes are computed on bytes for provenance.  The targets file is never parsed below.
    recorded_input_hashes = run_summary.get("input_hashes", {})
    actual_input_hashes: dict[str, str | None] = {}
    input_hash_mismatches: list[str] = []
    for name, item in recorded_input_hashes.items():
        path = resolve_root_path(str(item["path"]))
        if path.exists():
            actual = sha256_file(path)
            actual_input_hashes[name] = actual
            if actual != item.get("sha256"):
                input_hash_mismatches.append(name)
        else:
            actual_input_hashes[name] = None
            input_hash_mismatches.append(name)
    record(
        "recorded_input_hashes_match",
        not input_hash_mismatches,
        {name: item.get("sha256") for name, item in recorded_input_hashes.items()},
        actual_input_hashes,
        f"Input hash/path mismatch or missing input: {input_hash_mismatches}",
    )
    config_hash = sha256_file(CONFIG_PATH) if CONFIG_PATH.exists() else None
    record(
        "config_hash_matches_run_record",
        bool(config_hash and recorded_input_hashes.get("config", {}).get("sha256") == config_hash),
        recorded_input_hashes.get("config", {}).get("sha256"),
        config_hash,
        "The run's recorded config SHA-256 does not match the current config.",
    )
    runner_path = ROOT / "run_analyst_models_v1.py"
    runner_hash = sha256_file(runner_path) if runner_path.exists() else None
    record(
        "runner_code_hash_matches_run_record",
        bool(runner_hash and recorded_input_hashes.get("runner_code", {}).get("sha256") == runner_hash),
        recorded_input_hashes.get("runner_code", {}).get("sha256"),
        runner_hash,
        "The run's recorded runner SHA-256 does not match the current runner.",
    )

    output_hash_mismatches: list[str] = []
    actual_output_hashes: dict[str, str | None] = {}
    for name, expected_hash in run_summary.get("output_hashes", {}).items():
        path = RUN / f"{name}.csv"
        if not path.exists():
            path = RUN / f"{name}.json"
        if path.exists():
            actual = sha256_file(path)
            actual_output_hashes[name] = actual
            if actual != expected_hash:
                output_hash_mismatches.append(name)
        else:
            actual_output_hashes[name] = None
            output_hash_mismatches.append(name)
    record(
        "recorded_run_output_hashes_match",
        not output_hash_mismatches,
        run_summary.get("output_hashes", {}),
        actual_output_hashes,
        f"Run output hash mismatch or missing output: {output_hash_mismatches}",
    )

    model_hash_mismatches: list[str] = []
    actual_model_hashes: dict[str, str | None] = {}
    for name, expected_hash in run_summary.get("model_hashes", {}).items():
        path = RUN / "models" / f"{name}.joblib"
        if path.exists():
            actual = sha256_file(path)
            actual_model_hashes[name] = actual
            if actual != expected_hash:
                model_hash_mismatches.append(name)
        else:
            actual_model_hashes[name] = None
            model_hash_mismatches.append(name)
    record(
        "recorded_model_hashes_match",
        not model_hash_mismatches,
        run_summary.get("model_hashes", {}),
        actual_model_hashes,
        f"Model artifact hash mismatch or missing artifact: {model_hash_mismatches}",
    )

    # Structural output checks.
    predictions = pd.read_csv(RUN / "validation_predictions.csv")
    reported_metrics = pd.read_csv(RUN / "validation_metrics.csv")
    cv = pd.read_csv(RUN / "cv_results.csv")
    cv_summary = pd.read_csv(RUN / "cv_candidate_summary.csv")
    per_event = pd.read_csv(RUN / "per_event_ic.csv")
    bootstrap_table = pd.read_csv(RUN / "bootstrap_delta_mean_event_ic.csv")
    eligibility_audit = pd.read_csv(RUN / "eligibility_audit.csv", low_memory=False)
    bootstrap_summary = json.loads((RUN / "bootstrap_summary.json").read_text(encoding="utf-8"))
    feature_sets = json.loads((RUN / "feature_sets.json").read_text(encoding="utf-8"))

    required_prediction_columns = ["sample_id", *EVENT_KEY, "announcement_day", "target"]
    prediction_columns = [f"prediction_{name}" for name in MODEL_NAMES]
    expected_prediction_columns = required_prediction_columns + prediction_columns
    record(
        "validation_predictions_columns_complete",
        predictions.columns.tolist() == expected_prediction_columns,
        expected_prediction_columns,
        predictions.columns.tolist(),
        "Validation predictions contain missing, extra or reordered required columns.",
    )
    record(
        "validation_metrics_columns_complete",
        set(["model", *KEY_METRICS]).issubset(reported_metrics.columns),
        ["model", *KEY_METRICS],
        reported_metrics.columns.tolist(),
        "Validation metrics are missing a required model or metric column.",
    )
    record(
        "cv_columns_complete",
        set(["stage", "model", "fold", "params", "train_rows", "train_events", "validation_rows", "validation_events"]).issubset(cv.columns),
        ["stage", "model", "fold", "params", "train_rows", "train_events", "validation_rows", "validation_events"],
        cv.columns.tolist(),
        "CV output is missing a required fold/count column.",
    )
    record(
        "bootstrap_columns_complete",
        set(["replicate", "delta_mean_event_ic"]).issubset(bootstrap_table.columns),
        ["replicate", "delta_mean_event_ic"],
        bootstrap_table.columns.tolist(),
        "Bootstrap output is missing replicate or delta values.",
    )
    actual_config = config
    record(
        "run_summary_config_matches_current_config",
        run_summary.get("config") == actual_config,
        actual_config,
        run_summary.get("config"),
        "Embedded run config differs from the current declared config.",
    )
    record(
        "run_points_to_fixed_model_ready_run",
        run_summary.get("model_ready_run_id") == MODEL_READY_ID
        and config.get("model_ready_run_id") == MODEL_READY_ID,
        MODEL_READY_ID,
        {"summary": run_summary.get("model_ready_run_id"), "config": config.get("model_ready_run_id")},
        "Run/config does not point to the fixed model-ready run.",
    )

    # Read only non-target model-ready tables.  targets.csv is opened only for its header.
    eligibility = pd.read_csv(MODEL_READY / "eligibility.csv", low_memory=False)
    metadata = pd.read_csv(
        MODEL_READY / "metadata.csv",
        usecols=["sample_id", *EVENT_KEY, "announcement_day", "exit_session", "split"],
        low_memory=False,
    )
    target_header = MODEL_READY.joinpath("targets.csv").open("r", encoding="utf-8", newline="").readline().rstrip("\r\n").split(",")
    target_column = config["target"]
    record(
        "targets_header_contains_declared_target",
        target_column in target_header,
        target_column,
        target_header,
        "Declared target column is absent from targets.csv header.",
    )
    record(
        "test_target_rows_were_not_parsed",
        True,
        "No model-ready targets data rows parsed",
        "Header only; validation target came from validation_predictions.csv",
        "",
    )

    eligibility_unique = not eligibility.sample_id.duplicated().any()
    metadata_unique = not metadata.sample_id.duplicated().any()
    record(
        "eligibility_and_metadata_sample_ids_unique",
        eligibility_unique and metadata_unique,
        {"eligibility_unique": True, "metadata_unique": True},
        {"eligibility_unique": eligibility_unique, "metadata_unique": metadata_unique},
        "Duplicate sample_id would make counts or split joins ambiguous.",
    )
    eligibility["eligible_bool"] = parse_bool(eligibility["supervised_model_eligible"], "supervised_model_eligible")
    audit_ids = set(eligibility_audit.get("sample_id", pd.Series(dtype=str)))
    eligible_development_ids = set(eligibility.loc[eligibility.eligible_bool & eligibility.split.isin(["training", "validation"]), "sample_id"])
    audit_included_ids = set(eligibility_audit.loc[eligibility_audit.get("runner_included", pd.Series(dtype=bool)).astype(bool), "sample_id"]) if "runner_included" in eligibility_audit else set()
    record(
        "eligibility_audit_rows_and_inclusion_complete",
        len(eligibility_audit) == len(eligibility)
        and len(audit_ids) == len(eligibility)
        and "runner_included" in eligibility_audit.columns
        and audit_included_ids == eligible_development_ids,
        {"rows": len(eligibility), "unique_sample_ids": len(eligibility), "runner_included_ids": len(eligible_development_ids)},
        {"rows": len(eligibility_audit), "unique_sample_ids": len(audit_ids), "runner_included_ids": len(audit_included_ids)},
        "Eligibility audit does not retain one row per model-ready sample or the expected development inclusion set.",
    )
    metadata["announcement_day"] = pd.to_datetime(metadata["announcement_day"], errors="coerce").dt.normalize()
    metadata["exit_session"] = pd.to_datetime(metadata["exit_session"], errors="coerce").dt.normalize()
    joined = eligibility[["sample_id", "split", "eligible_bool"]].merge(
        metadata, on=["sample_id", "split"], how="left", validate="one_to_one"
    )
    missing_metadata = int(joined["announcement_day"].isna().sum() + joined["exit_session"].isna().sum())
    record(
        "eligibility_metadata_join_complete",
        len(joined) == len(eligibility) and missing_metadata == 0,
        {"rows": len(eligibility), "missing_date_values": 0},
        {"rows": len(joined), "missing_date_values": missing_metadata},
        "Eligibility rows are missing metadata or event dates.",
    )

    test_ids = set(eligibility.loc[eligibility.split.eq("test"), "sample_id"])
    training = joined.loc[joined.eligible_bool & joined.split.eq("training")].copy()
    validation = joined.loc[joined.eligible_bool & joined.split.eq("validation")].copy()
    development = joined.loc[joined.eligible_bool & joined.split.isin(["training", "validation"])].copy()
    test_eligible_count = int((joined.split.eq("test") & joined.eligible_bool).sum())
    independently_counted = {
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "training_events": int(training[EVENT_KEY].drop_duplicates().shape[0]),
        "validation_events": int(validation[EVENT_KEY].drop_duplicates().shape[0]),
        "test_rows": int(joined.split.eq("test").sum()),
        "test_supervised_eligible_rows": test_eligible_count,
    }
    expected_counts = {
        "training_rows": int(run_summary.get("training_rows", -1)),
        "validation_rows": int(run_summary.get("validation_rows", -1)),
        "training_events": int(run_summary.get("training_events", -1)),
        "validation_events": int(run_summary.get("validation_events", -1)),
        "test_rows": int(run_summary.get("processing_record", {}).get("row_counts", {}).get("sealed_test_rows", -1)),
        "test_supervised_eligible_rows": int(config.get("expected_supervised_model_eligible_rows", {}).get("test", -1)),
    }
    record(
        "training_validation_counts_match_frozen_eligibility",
        independently_counted == expected_counts,
        expected_counts,
        independently_counted,
        "Independent eligibility/metadata counts differ from the run record.",
    )
    record(
        "test_split_is_sealed_by_eligibility",
        test_eligible_count == 0,
        0,
        test_eligible_count,
        "At least one test row is marked supervised_model_eligible; test target must remain sealed.",
    )

    # Validation output identity and sealed-test output checks.
    pred_ids = set(predictions.sample_id)
    valid_ids = set(validation.sample_id)
    duplicate_prediction_ids = int(predictions.sample_id.duplicated().sum())
    record(
        "validation_prediction_ids_exactly_match_eligible_validation",
        duplicate_prediction_ids == 0 and pred_ids == valid_ids,
        {"rows": len(valid_ids), "duplicate_ids": 0},
        {"rows": len(predictions), "unique_ids": len(pred_ids), "duplicate_ids": duplicate_prediction_ids, "test_intersection": len(pred_ids & test_ids)},
        "Validation predictions do not exactly cover eligible validation IDs or contain test IDs.",
    )
    pred_days = pd.to_datetime(predictions["announcement_day"], errors="coerce").dt.normalize()
    validation_date_ok = bool(pred_days.notna().all() and pred_days.between("2021-01-01", "2022-12-31").all())
    record(
        "validation_predictions_are_validation_period_only",
        validation_date_ok,
        "announcement_day in 2021-01-01..2022-12-31",
        {"min": str(pred_days.min()), "max": str(pred_days.max()), "rows_outside": int((~pred_days.between("2021-01-01", "2022-12-31")).sum())},
        "A prediction row falls outside the declared validation date range.",
    )
    test_token_columns = [column for column in predictions.columns if "test" in column.lower()]
    record(
        "no_test_prediction_columns_or_rows",
        not test_token_columns and not (pred_ids & test_ids),
        {"test_columns": [], "test_rows": 0},
        {"test_columns": test_token_columns, "test_row_intersection": len(pred_ids & test_ids)},
        "Validation output contains a test prediction column or test sample ID.",
    )
    metrics_test_tokens = [column for column in reported_metrics.columns if "test" in column.lower()]
    record(
        "no_test_metrics_or_target_outputs",
        not metrics_test_tokens and "target_test" not in predictions.columns and "test_target" not in predictions.columns,
        {"metric_test_columns": [], "test_target_columns": []},
        {"metric_test_columns": metrics_test_tokens, "test_target_columns": [c for c in predictions.columns if "target" in c.lower() and "test" in c.lower()]},
        "A test metric or test target output column is present.",
    )
    run_named_test_artifacts = [str(path.relative_to(RUN)) for path in RUN.rglob("*") if "test" in path.name.lower()]
    record(
        "no_test_named_run_artifacts",
        not run_named_test_artifacts,
        [],
        run_named_test_artifacts,
        "A run artifact name indicates test scoring or prediction.",
    )
    record(
        "validation_targets_and_predictions_finite",
        bool(np.isfinite(predictions["target"].to_numpy(float)).all() and np.isfinite(predictions[prediction_columns].to_numpy(float)).all()),
        "finite validation target and all model predictions",
        {"target_finite": bool(np.isfinite(predictions["target"].to_numpy(float)).all()), "predictions_finite": bool(np.isfinite(predictions[prediction_columns].to_numpy(float)).all())},
        "Validation target or prediction contains NaN or infinity.",
    )

    # Rebuild every rolling training fold from eligibility and metadata, without target values.
    folds = config["internal_training_folds"]
    fold_names = [fold["name"] for fold in folds]
    cv_fold_count_observed = cv["fold"].dropna().astype(str).unique().tolist()
    record(
        "internal_folds_are_chronological",
        all(pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"]) for fold in folds)
        and all(pd.Timestamp(left["validation_end"]) < pd.Timestamp(right["validation_start"]) for left, right in zip(folds, folds[1:])),
        True,
        {"folds": fold_names, "observed_cv_folds": cv_fold_count_observed},
        "Declared rolling folds are not strictly chronological.",
    )
    expected_fold_counts: dict[str, dict[str, int]] = {}
    leakage_failures: list[str] = []
    fold_count_failures: list[str] = []
    for fold in folds:
        name = fold["name"]
        start = pd.Timestamp(fold["validation_start"])
        train_end = pd.Timestamp(fold["train_end"])
        valid_end = pd.Timestamp(fold["validation_end"])
        fold_train = training.loc[(training.announcement_day <= train_end) & (training.exit_session < start)].copy()
        fold_valid = training.loc[training.announcement_day.between(start, valid_end, inclusive="both")].copy()
        expected_fold_counts[name] = {
            "train_rows": int(len(fold_train)),
            "train_events": int(fold_train[EVENT_KEY].drop_duplicates().shape[0]),
            "validation_rows": int(len(fold_valid)),
            "validation_events": int(fold_valid[EVENT_KEY].drop_duplicates().shape[0]),
        }
        sub = cv.loc[cv["fold"].astype("string").eq(name)]
        expected_candidates = expected_candidate_counts(config)
        counts_by_model = sub["model"].value_counts().to_dict()
        if counts_by_model != expected_candidates:
            fold_count_failures.append(name)
        if not sub.empty:
            if not sub["stage"].astype("string").eq("training_cv").all():
                leakage_failures.append(f"{name}:non_training_cv_stage")
            if not sub["train_rows"].eq(len(fold_train)).all() or not sub["validation_rows"].eq(len(fold_valid)).all():
                fold_count_failures.append(f"{name}:row_counts")
            if not sub["train_events"].eq(fold_train[EVENT_KEY].drop_duplicates().shape[0]).all() or not sub["validation_events"].eq(fold_valid[EVENT_KEY].drop_duplicates().shape[0]).all():
                fold_count_failures.append(f"{name}:event_counts")
        train_ids = set(fold_train.sample_id)
        valid_ids_fold = set(fold_valid.sample_id)
        train_events = set(map(tuple, fold_train[EVENT_KEY].to_numpy()))
        valid_events = set(map(tuple, fold_valid[EVENT_KEY].to_numpy()))
        if train_ids & valid_ids_fold:
            leakage_failures.append(f"{name}:sample_id_overlap")
        if train_events & valid_events:
            leakage_failures.append(f"{name}:event_overlap")
        if not bool((fold_train.announcement_day <= train_end).all() and (fold_train.exit_session < start).all()):
            leakage_failures.append(f"{name}:train_after_boundary_or_exit_leak")
        if not bool(fold_valid.announcement_day.between(start, valid_end, inclusive="both").all()):
            leakage_failures.append(f"{name}:validation_outside_date_range")
        if train_ids & test_ids or valid_ids_fold & test_ids:
            leakage_failures.append(f"{name}:test_id_in_cv")
    record(
        "cv_rows_cover_expected_candidate_grid",
        not fold_count_failures and set(cv["fold"].astype(str)) == set(fold_names),
        {"folds": fold_names, "candidate_counts_by_fold": expected_candidate_counts(config)},
        {"cv_rows": int(len(cv)), "fold_counts": cv.groupby(["fold", "model"], dropna=False).size().to_dict(), "fold_sizes": expected_fold_counts},
        f"CV candidate/fold count mismatch: {sorted(set(fold_count_failures))}",
    )
    record(
        "cv_time_splits_have_no_leakage",
        not leakage_failures,
        "train announcement <= train_end and exit_session < validation_start; disjoint sample/event IDs; no test IDs",
        {"failures": sorted(set(leakage_failures)), "fold_sizes": expected_fold_counts},
        f"Time-fold leakage or split violation: {sorted(set(leakage_failures))}",
    )
    record(
        "cv_uses_training_split_only",
        set(cv["stage"].astype(str)) == {"training_cv"} and set(cv["fold"].astype(str)) == set(fold_names),
        {"stage": ["training_cv"], "folds": fold_names},
        {"stage": sorted(set(cv["stage"].astype(str))), "folds": sorted(set(cv["fold"].astype(str)))},
        "CV contains an unexpected stage or fold.",
    )

    # Confirmatory/exploratory role labels.
    nominee = str(selected.get("network_family_nominee", run_summary.get("network_family_nominee", "")))
    expected_roles = {name: ("confirmatory" if name in {nominee, "context_ridge"} else "exploratory") for name in MODEL_NAMES}
    metric_roles = dict(zip(reported_metrics.get("model", pd.Series(dtype=str)), reported_metrics.get("comparison_role", pd.Series(dtype=str))))
    metric_nominee_flags = dict(zip(reported_metrics.get("model", pd.Series(dtype=str)), reported_metrics.get("cv_nominated_network_family", pd.Series(dtype=bool))))
    role_failures = [name for name in MODEL_NAMES if metric_roles.get(name) != expected_roles[name]]
    nominee_flag_failures = [name for name in MODEL_NAMES if bool(metric_nominee_flags.get(name, False)) != (name == nominee)]
    per_event_roles = per_event.groupby("model", dropna=False)["comparison_role"].first().to_dict() if not per_event.empty else {}
    per_event_role_failures = [name for name in MODEL_NAMES if per_event_roles.get(name) != expected_roles[name]]
    record(
        "confirmatory_and_exploratory_roles_are_correct",
        nominee in NETWORK_FAMILIES and not role_failures and not per_event_role_failures and not nominee_flag_failures,
        {"nominee": nominee, "roles": expected_roles, "nominee_flag": {name: name == nominee for name in MODEL_NAMES}},
        {"metric_roles": metric_roles, "per_event_roles": per_event_roles, "metric_nominee_flags": metric_nominee_flags},
        f"Role-label failures: metric={role_failures}, per_event={per_event_role_failures}, nominee_flags={nominee_flag_failures}",
    )

    # Independent validation metrics from the saved validation-only predictions.
    recomputed_metrics: dict[str, dict[str, Any]] = {}
    recomputed_event_tables: dict[str, pd.DataFrame] = {}
    metric_row_failures: list[str] = []
    prediction_model_names = [column.removeprefix("prediction_") for column in predictions.columns if column.startswith("prediction_")]
    record(
        "prediction_models_match_declared_models",
        set(prediction_model_names) == set(MODEL_NAMES),
        MODEL_NAMES,
        prediction_model_names,
        "Prediction columns do not cover exactly the declared models.",
    )
    for name in MODEL_NAMES:
        result, event_table = independent_metrics(predictions, f"prediction_{name}")
        recomputed_metrics[name] = result
        recomputed_event_tables[name] = event_table
        stored_rows = reported_metrics.loc[reported_metrics["model"].astype(str).eq(name)]
        stored = stored_rows.iloc[0].to_dict() if len(stored_rows) == 1 else {}
        model_failures = []
        for metric in KEY_METRICS:
            expected_value = result[metric]
            observed_value = stored.get(metric, np.nan)
            matched = values_match(expected_value, observed_value, integer=metric == "event_spearman_groups")
            if not matched:
                model_failures.append(metric)
        metric_row_failures.extend(f"{name}:{metric}" for metric in model_failures)
        record(
            f"validation_metrics_recomputed_{name}",
            not model_failures,
            {metric: result[metric] for metric in KEY_METRICS},
            {metric: stored.get(metric, np.nan) for metric in KEY_METRICS},
            f"Stored {name} metric mismatch: {model_failures}",
        )
    record(
        "all_key_validation_metrics_recomputed",
        not metric_row_failures,
        "pooled Spearman, mean event Spearman and event-weighted MSE/group count match independently",
        {"failures": metric_row_failures, "models": recomputed_metrics},
        f"Independent validation metric mismatches: {metric_row_failures}",
    )

    # Event-level output consistency and independent nominee/context paired bootstrap.
    expected_per_event_rows = len(predictions[EVENT_KEY].drop_duplicates()) * len(MODEL_NAMES)
    per_event_models_ok = set(per_event.get("model", pd.Series(dtype=str)).astype(str)) == set(MODEL_NAMES)
    record(
        "per_event_ic_output_complete",
        len(per_event) == expected_per_event_rows and per_event_models_ok,
        {"rows": expected_per_event_rows, "models": MODEL_NAMES},
        {"rows": len(per_event), "models": sorted(set(per_event.get("model", pd.Series(dtype=str)).astype(str)))},
        "per_event_ic does not contain one row per event/model pair.",
    )
    event_value_failures: list[str] = []
    for name, independent_table in recomputed_event_tables.items():
        stored_table = per_event.loc[per_event["model"].astype(str).eq(name)].copy()
        if len(stored_table) != len(independent_table):
            event_value_failures.append(f"{name}:row_count")
            continue
        left = independent_table[EVENT_KEY + ["rows", "target_unique", "prediction_unique", "event_ic", "ic_eligible", "ic_reason"]].copy()
        right = stored_table[EVENT_KEY + ["rows", "target_unique", "prediction_unique", "event_ic", "ic_eligible", "ic_reason"]].copy()
        merged = left.merge(right, on=EVENT_KEY, how="outer", suffixes=("_independent", "_stored"), indicator=True)
        if (merged["_merge"] != "both").any():
            event_value_failures.append(f"{name}:event_keys")
            continue
        for _, row in merged.iterrows():
            if int(row["rows_independent"]) != int(row["rows_stored"]) or int(row["target_unique_independent"]) != int(row["target_unique_stored"]) or int(row["prediction_unique_independent"]) != int(row["prediction_unique_stored"]):
                event_value_failures.append(f"{name}:event_counts")
                break
            if not values_match(row["event_ic_independent"], row["event_ic_stored"]) or bool(row["ic_eligible_independent"]) != bool(row["ic_eligible_stored"]) or row["ic_reason_independent"] != row["ic_reason_stored"]:
                event_value_failures.append(f"{name}:event_ic_values")
                break
    record(
        "per_event_ic_recomputed_from_validation_predictions",
        not event_value_failures,
        "event IC/count/reason values independently reproduced",
        {"failures": sorted(set(event_value_failures))},
        f"Per-event IC mismatch: {sorted(set(event_value_failures))}",
    )

    if nominee not in recomputed_event_tables or "context_ridge" not in recomputed_event_tables:
        paired = pd.DataFrame()
    else:
        nominee_table = recomputed_event_tables[nominee][EVENT_KEY + ["announcement_month", "event_ic", "ic_eligible"]].rename(
            columns={"event_ic": "nominee_ic", "ic_eligible": "nominee_ic_eligible", "announcement_month": "nominee_month"}
        )
        context_table = recomputed_event_tables["context_ridge"][EVENT_KEY + ["announcement_month", "event_ic", "ic_eligible"]].rename(
            columns={"event_ic": "context_ic", "ic_eligible": "context_ic_eligible", "announcement_month": "context_month"}
        )
        paired = nominee_table.merge(context_table, on=EVENT_KEY, how="inner", validate="one_to_one")
        paired = paired.loc[paired.nominee_ic_eligible & paired.context_ic_eligible].copy()
        paired["delta_mean_event_ic"] = paired.nominee_ic - paired.context_ic
        paired["announcement_month"] = paired.nominee_month
    pair_count = int(len(paired))
    pair_month_counts = paired.groupby("announcement_month", sort=True).size().astype(int).to_dict() if not paired.empty else {}
    observed_delta = float(paired.delta_mean_event_ic.mean()) if pair_count else np.nan
    seed = int(config["bootstrap"]["seed"])
    replications = int(config["bootstrap"]["replications"])
    bootstrap_values = np.array([], dtype=float)
    if pair_count and pair_month_counts:
        month_values = {month: group.delta_mean_event_ic.to_numpy(float) for month, group in paired.groupby("announcement_month", sort=True)}
        months = np.array(sorted(month_values), dtype=object)
        rng = np.random.default_rng(seed)
        bootstrap_values = np.empty(replications, dtype=float)
        for index in range(replications):
            sampled = rng.choice(months, size=len(months), replace=True)
            bootstrap_values[index] = float(np.concatenate([month_values[month] for month in sampled]).mean())
    independent_bootstrap_summary = {
        "comparison": f"{nominee} minus context_ridge",
        "nominee": nominee,
        "context_model": "context_ridge",
        "observed_delta_mean_event_ic": observed_delta,
        "paired_event_count": pair_count,
        "announcement_month_block_count": len(pair_month_counts),
        "paired_event_counts_by_month": {str(key): int(value) for key, value in pair_month_counts.items()},
        "seed": seed,
        "replications": replications,
        "bootstrap_ci_lower": float(np.quantile(bootstrap_values, 0.025)) if len(bootstrap_values) else np.nan,
        "bootstrap_ci_upper": float(np.quantile(bootstrap_values, 0.975)) if len(bootstrap_values) else np.nan,
    }
    stored_bootstrap_values = bootstrap_table["delta_mean_event_ic"].to_numpy(float) if "delta_mean_event_ic" in bootstrap_table else np.array([], dtype=float)
    bootstrap_values_match = len(stored_bootstrap_values) == len(bootstrap_values) and bool(np.allclose(stored_bootstrap_values, bootstrap_values, rtol=1e-11, atol=1e-13, equal_nan=True))
    stored_bootstrap_fields = {key: bootstrap_summary.get(key) for key in independent_bootstrap_summary}
    bootstrap_field_failures = [
        key for key in ["comparison", "nominee", "context_model", "paired_event_count", "announcement_month_block_count", "seed", "replications"]
        if not values_match(independent_bootstrap_summary[key], stored_bootstrap_fields.get(key), integer=key in {"paired_event_count", "announcement_month_block_count", "seed", "replications"})
    ]
    for key in ["observed_delta_mean_event_ic", "bootstrap_ci_lower", "bootstrap_ci_upper"]:
        if not values_match(independent_bootstrap_summary[key], stored_bootstrap_fields.get(key)):
            bootstrap_field_failures.append(key)
    if independent_bootstrap_summary["paired_event_counts_by_month"] != bootstrap_summary.get("paired_event_counts_by_month"):
        bootstrap_field_failures.append("paired_event_counts_by_month")
    record(
        "paired_nominee_context_event_delta_recomputed",
        nominee == "analyst_hgb" and not bootstrap_field_failures,
        independent_bootstrap_summary,
        stored_bootstrap_fields,
        f"Paired nominee/context delta summary mismatch: {bootstrap_field_failures}",
    )
    record(
        "announcement_month_block_bootstrap_recomputed",
        bootstrap_values_match and len(bootstrap_table) == replications and np.isfinite(stored_bootstrap_values).all(),
        {"replications": replications, "seed": seed, "first_three": bootstrap_values[:3].tolist(), "ci": [independent_bootstrap_summary["bootstrap_ci_lower"], independent_bootstrap_summary["bootstrap_ci_upper"]]},
        {"rows": len(bootstrap_table), "seed": bootstrap_summary.get("seed"), "first_three": stored_bootstrap_values[:3].tolist(), "ci": [bootstrap_summary.get("bootstrap_ci_lower"), bootstrap_summary.get("bootstrap_ci_upper")]},
        "Stored announcement-month bootstrap values differ from the independent fixed-seed rerun.",
    )

    # Compare high-level runner claims only as a consistency aid; all substantive checks above are independent.
    runner_claims = run_summary.get("checks", {})
    false_runner_claims = sorted(name for name, value in runner_claims.items() if value is not True)
    record(
        "runner_summary_has_no_false_check_claims",
        not false_runner_claims,
        [],
        false_runner_claims,
        f"Runner summary contains false/non-true claims: {false_runner_claims}",
    )

    passed = int(sum(bool(row["passed"]) for row in checks))
    total = int(len(checks))
    failed = [row for row in checks if not row["passed"]]
    audit_report = {
        "audit_run_id": audit_id,
        "audited_model_run_id": RUN_ID,
        "audited_model_run_path": str(RUN.relative_to(ROOT)),
        "model_ready_run_id": MODEL_READY_ID,
        "validator_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "purpose": "Independent audit of output integrity, hashes, counts, time-fold leakage, validation metrics and nominee/context block bootstrap.",
            "test_target_rows_parsed": False,
            "test_future_returns_read_or_scored": False,
            "test_checks": ["sealed test eligibility count", "test sample IDs absent from validation outputs", "no test-named prediction/metric artifacts", "targets.csv header only"],
        },
        "passed": passed,
        "total": total,
        "all_passed": bool(passed == total),
        "failed_checks": failed,
        "checks": checks,
        "independently_counted_rows_and_events": independently_counted,
        "expected_fold_counts": expected_fold_counts,
        "recomputed_validation_metrics": recomputed_metrics,
        "independent_paired_bootstrap": independent_bootstrap_summary,
        "actual_input_hashes": actual_input_hashes,
        "actual_output_hashes": actual_output_hashes,
        "actual_model_hashes": actual_model_hashes,
        "limitations": [
            "Validation metrics are independently recalculated from the validation_predictions.csv target column and saved predictions.",
            "No model refit or model artifact reload is required for the requested audit; model artifacts are checked for presence and recorded hashes.",
            "The sealed test target file is hashed and its header inspected, but its data rows and future-return values are not parsed.",
            "Announcement-month bootstrap is reproduced exactly as a block resampling aid, not an IID significance test.",
        ],
        "status": "completed" if passed == total else "failed_checks",
    }
    (audit_dir / "validation.json").write_text(
        json.dumps(jsonable(audit_report), indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame(checks).to_csv(audit_dir / "validation.csv", index=False)
    print(
        json.dumps(
            {
                "audit_run_id": audit_id,
                "audited_model_run_id": RUN_ID,
                "passed": passed,
                "total": total,
                "all_passed": passed == total,
                "failed_checks": [row["check"] for row in failed],
                "independent_paired_delta": independent_bootstrap_summary,
                "recomputed_validation_metrics": recomputed_metrics,
                "audit_dir": str(audit_dir),
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
