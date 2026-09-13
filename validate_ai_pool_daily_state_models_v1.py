"""Independently validate a training/validation-only AI-pool state-model run.

This validator reads development labels (including ``exit_session``) only.  It
never opens ``targets_test_sealed.csv`` and does not create predictions or
metrics.  The checks are intentionally independent of the runner's in-memory
objects: source row counts and purged-fold counts are recomputed from the
persisted development inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ai_pool_daily_state_models_v1_config.json"
AUDIT_ROOT = ROOT / "data" / "audit" / "ai_pool_daily_state_models_v1"
TARGET_SEALED_NAME = "targets_test_sealed.csv"
EXPECTED_FEATURE_SETS = [
    "technical_only",
    "technical_plus_macro",
    "technical_plus_ai_state",
    "full_state",
]
TARGET_COLUMNS = {
    "y",
    "forward_excess_return",
    "stock_forward_return",
    "benchmark_forward_return",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (set, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_value(x) for x in value]
    return value


def check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    detail: str,
    classification: str = "metadata_or_validator_semantics",
) -> None:
    checks.append({
        "check_id": check_id,
        "passed": bool(passed),
        "observed": json_value(observed),
        "expected": json_value(expected),
        "detail": detail,
        "failure_classification": classification if not passed else "none",
    })


def independent_primary_sample(model_ready: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features_path = model_ready / "model_features.csv"
    eligibility_path = model_ready / "eligibility.csv"
    targets_path = model_ready / "targets_dev.csv"
    features = pd.read_csv(
        features_path,
        usecols=["sample_id", "Instrument", "formation_session", "split"],
        low_memory=False,
    )
    eligibility = pd.read_csv(
        eligibility_path,
        usecols=[
            "sample_id",
            "boundary_purged",
            "entry_trade_eligible",
            "feature_core_available",
            "non_overlap_selected",
            "label_complete",
        ],
        low_memory=False,
    )
    # Development labels are permitted for this audit.  The sealed test file
    # is deliberately not part of required inputs or any read operation.
    targets = pd.read_csv(
        targets_path,
        usecols=["sample_id", "split", "exit_session", "y", "forward_excess_return", "label_complete"],
        low_memory=False,
    )
    if not set(targets["split"].astype(str)).issubset({"training", "validation"}):
        raise RuntimeError("targets_dev contains a split outside training/validation")
    for frame, name in [(features, "model_features"), (eligibility, "eligibility"), (targets, "targets_dev")]:
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{name} has invalid sample_id keys")
    if set(features["sample_id"]) != set(eligibility["sample_id"]):
        raise RuntimeError("model_features and eligibility sample sets differ")
    data = features.merge(targets, on="sample_id", how="inner", validate="one_to_one", suffixes=("", "_target"))
    data = data.merge(eligibility, on="sample_id", how="inner", validate="one_to_one", suffixes=("", "_eligibility"))
    data["formation_session"] = pd.to_datetime(data["formation_session"], format="mixed", errors="raise").dt.normalize()
    data["exit_session"] = pd.to_datetime(data["exit_session"], format="mixed", errors="raise").dt.normalize()
    data["split"] = data["split"].astype(str)
    if not data["split"].eq(data["split_target"].astype(str)).all():
        raise RuntimeError("feature and development target split labels differ")
    for column in [
        "boundary_purged",
        "entry_trade_eligible",
        "feature_core_available",
        "non_overlap_selected",
        "label_complete",
        "label_complete_eligibility",
    ]:
        if column in data:
            data[column] = bool_series(data[column])
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["forward_excess_return"] = pd.to_numeric(data["forward_excess_return"], errors="coerce")
    eligible = (
        data["label_complete"]
        & ~data["boundary_purged"]
        & data["entry_trade_eligible"]
        & data["feature_core_available"]
        & data["non_overlap_selected"]
        & data["y"].notna()
        & data["forward_excess_return"].notna()
    )
    train = data.loc[data["split"].eq("training") & eligible].sort_values(
        ["formation_session", "Instrument", "sample_id"], kind="stable"
    ).copy()
    validation = data.loc[data["split"].eq("validation") & eligible].sort_values(
        ["formation_session", "Instrument", "sample_id"], kind="stable"
    ).copy()
    expected = {
        "model_ready_master_rows": int(len(features)),
        "model_ready_master_instruments": int(features["Instrument"].nunique()),
        "model_ready_training_rows": int(features["split"].eq("training").sum()),
        "model_ready_validation_rows": int(features["split"].eq("validation").sum()),
        "model_ready_test_rows": int(features["split"].eq("test").sum()),
        "development_target_rows_loaded": int(len(targets)),
        "primary_training_rows": int(len(train)),
        "primary_validation_rows": int(len(validation)),
        "primary_training_instruments": int(train["Instrument"].nunique()),
        "primary_validation_instruments": int(validation["Instrument"].nunique()),
        "primary_training_sessions": int(train["formation_session"].nunique()),
        "primary_validation_sessions": int(validation["formation_session"].nunique()),
    }
    return train, validation, expected


def independent_fold_specs(train: pd.DataFrame, years: list[int], horizon: int) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for year in years:
        valid = train.loc[train["formation_session"].dt.year.eq(year)].copy()
        if valid.empty:
            continue
        start = pd.Timestamp(valid["formation_session"].min())
        candidate = train.loc[train["formation_session"] < start].copy()
        fit = candidate.loc[candidate["exit_session"] < start].copy()
        if fit.empty:
            continue
        folds.append({
            "fold_id": f"train_to_{year - 1}_validate_{year}",
            "validation_year": year,
            "validation_start": start.strftime("%Y-%m-%d"),
            "validation_end": pd.Timestamp(valid["formation_session"].max()).strftime("%Y-%m-%d"),
            "purge_rule": "fit rows require exit_session < validation_start",
            "last_fit_exit": pd.Timestamp(fit["exit_session"].max()).strftime("%Y-%m-%d"),
            "candidate_rows_before_purge": int(len(candidate)),
            "purged_rows": int(len(candidate) - len(fit)),
            "purge_sessions": horizon,
            "fit_rows": int(len(fit)),
            "validation_rows": int(len(valid)),
        })
    return folds


def run(
    run_id: str,
    audit_id: str | None = None,
    run_root: Path | str | None = None,
    audit_root: Path | str | None = None,
) -> dict[str, Any]:
    root_value = Path(run_root) if run_root is not None else Path("data/model_runs/ai_pool_daily_state_models_v1")
    if not root_value.is_absolute():
        root_value = ROOT / root_value
    run_dir = root_value / run_id
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(rel(summary_path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    check(checks, "summary_run_id", summary.get("run_id") == run_id, summary.get("run_id"), run_id, "summary identifies the requested model run")
    check(checks, "summary_complete_status", summary.get("status") == "complete_training_validation_only", summary.get("status"), "complete_training_validation_only", "runner status is complete and development-only")
    check(checks, "sealed_target_not_opened", summary.get("inputs", {}).get("sealed_test_targets", {}).get("opened") is False and summary.get("checks", {}).get("test_targets_not_opened") is True, summary.get("inputs", {}).get("sealed_test_targets"), {"opened": False}, "sealed test target is declared unopened")
    check(checks, "no_test_outputs", summary.get("counts", {}).get("test_prediction_rows") == 0 and summary.get("counts", {}).get("test_metric_rows") == 0 and summary.get("checks", {}).get("test_predictions_zero") is True and summary.get("checks", {}).get("test_metrics_zero") is True, {"prediction_rows": summary.get("counts", {}).get("test_prediction_rows"), "metric_rows": summary.get("counts", {}).get("test_metric_rows")}, {"prediction_rows": 0, "metric_rows": 0}, "run contains no test predictions or metrics")

    recorded_inputs = summary.get("inputs", {}).get("files", {})
    recorded_config_path = resolve_recorded_path(recorded_inputs.get("config", {}).get("path", str(CONFIG_PATH)))
    config = json.loads(recorded_config_path.read_text(encoding="utf-8"))
    for name, record in recorded_inputs.items():
        path = resolve_recorded_path(record["path"])
        exists = path.exists()
        actual = sha256_file(path) if exists else None
        check(
            checks,
            f"input_hash_{name}",
            exists and actual == record.get("sha256"),
            actual,
            record.get("sha256"),
            f"persisted input hash unchanged for {name}",
            "source_or_artifact_integrity",
        )
    runner_record = recorded_inputs.get("runner", {})
    config_record = recorded_inputs.get("config", {})
    runner_path = resolve_recorded_path(runner_record.get("path", "run_ai_pool_daily_state_models_v1.py"))
    check(checks, "runner_hash_chain", runner_record.get("sha256") == sha256_file(runner_path), runner_record.get("sha256"), sha256_file(runner_path), "runner hash recorded in summary matches current script")
    check(checks, "config_hash_chain", config_record.get("sha256") == sha256_file(recorded_config_path), config_record.get("sha256"), sha256_file(recorded_config_path), "config hash recorded in summary matches current config")

    output_hashes = summary.get("outputs", {}).get("output_hashes", {})
    for recorded_path, expected_hash in output_hashes.items():
        path = resolve_recorded_path(recorded_path)
        exists = path.exists()
        actual = sha256_file(path) if exists else None
        check(checks, f"output_hash_{Path(recorded_path).name}", exists and actual == expected_hash, actual, expected_hash, f"model-run output hash matches for {recorded_path}", "source_or_artifact_integrity")

    feature_sets = json.loads((run_dir / "feature_sets.json").read_text(encoding="utf-8"))
    check(checks, "feature_set_names", list(feature_sets) == EXPECTED_FEATURE_SETS, list(feature_sets), EXPECTED_FEATURE_SETS, "all four registered feature groups are present")
    check(checks, "feature_set_columns_unique", all(len(cols) == len(set(cols)) for cols in feature_sets.values()), {key: len(cols) for key, cols in feature_sets.items()}, "no duplicate columns in any effective feature set", "stable technical-first deduplication")
    all_features = {column for columns in feature_sets.values() for column in columns}
    excluded_credit = set(config["excluded_credit_spread_features"])
    check(checks, "credit_spread_excluded", not bool(all_features & excluded_credit), sorted(all_features & excluded_credit), [], "six credit-spread columns are absent from effective features")
    check(checks, "target_columns_excluded", not bool(all_features & TARGET_COLUMNS), sorted(all_features & TARGET_COLUMNS), [], "target and forward-return columns are absent from features")
    alias = summary.get("feature_alias_exclusions", {})
    dropped_dupes = set(alias.get("technical_duplicate_columns_dropped_from_ai_state", []))
    dropped_alias = {item.get("dropped"): item.get("kept") for item in alias.get("numeric_alias_columns_dropped_from_ai_state", [])}
    expected_dupes = set(config.get("feature_alias_exclusions", {}).get("technical_duplicates", []))
    configured_aliases = config.get("feature_alias_exclusions", {}).get("numeric_aliases", {})
    expected_aliases = {
        key: value for key, value in configured_aliases.items()
        if key in set(config.get("ai_state_features", []))
    }
    upstream_exclusions = set(alias.get("upstream_safe_list_exclusions", [])) | set(config.get("feature_alias_exclusions", {}).get("upstream_safe_list_exclusions", []))
    effective_ai_features = set(feature_sets["technical_plus_ai_state"]) - set(config["technical_features"])
    check(checks, "technical_duplicate_exclusions", dropped_dupes == expected_dupes and not dropped_dupes & effective_ai_features, sorted(dropped_dupes), sorted(expected_dupes), "technical duplicates are explicitly excluded before AI merge")
    numeric_alias_ok = dropped_alias == expected_aliases and "spy_return_1" not in all_features
    if "spy_return_1" not in set(config.get("ai_state_features", [])):
        numeric_alias_ok = numeric_alias_ok and "spy_return_1" in upstream_exclusions
    check(checks, "numeric_alias_exclusion", numeric_alias_ok, {"dropped": dropped_alias, "upstream_exclusions_contains_spy_return_1": "spy_return_1" in upstream_exclusions}, {"dropped": expected_aliases, "spy_return_1_excluded": True}, "spy_return_1 alias is excluded in favor of technical spy_momentum_1")
    check(checks, "effective_ai_feature_count", alias.get("effective_ai_state_feature_count") == len(set(feature_sets["technical_plus_ai_state"]) - set(config["technical_features"])), alias.get("effective_ai_state_feature_count"), "39 effective AI-state features", "effective AI feature count agrees with feature sets")

    metrics = pd.read_csv(run_dir / "validation_metrics.csv")
    predictions = pd.read_csv(run_dir / "validation_predictions.csv", low_memory=False)
    expected_model_keys = [f"{feature_set}/{model}" for feature_set in EXPECTED_FEATURE_SETS for model in ["logistic", "random_forest"]]
    observed_model_keys = metrics["model_key"].astype(str).tolist()
    check(checks, "eight_metric_rows", sorted(observed_model_keys) == sorted(expected_model_keys) and len(metrics) == 8, observed_model_keys, expected_model_keys, "four feature sets each have logistic and RF validation metrics")
    expected_validation_rows = int(summary["counts"]["primary_validation_rows"])
    check(checks, "validation_prediction_row_count", len(predictions) == expected_validation_rows and int(summary["counts"]["validation_prediction_rows"]) == len(predictions), len(predictions), expected_validation_rows, "validation predictions cover the persisted primary validation sample")
    check(checks, "validation_only_predictions", set(predictions["split"].astype(str)) == {"validation"}, sorted(predictions["split"].astype(str).unique()), ["validation"], "prediction table contains validation rows only")
    probability_columns = [column for column in predictions.columns if column.startswith("p_up_")]
    check(checks, "eight_probability_columns", len(probability_columns) == 8 and all(predictions[column].notna().all() for column in probability_columns), probability_columns, "8 finite p_up columns", "all model validation probabilities are present and finite")

    tuning = json.loads((run_dir / "rf_tuning_evidence.json").read_text(encoding="utf-8"))
    config_grid = config["random_forest_tuning_grid"]
    tuning_years = config["tuning"]["fold_validation_years"]
    train, validation, independent_counts = independent_primary_sample(
        ROOT / "data" / "model_ready_ai_pool_daily_v1" / config["model_ready_run_id"], config
    )
    for key, expected in independent_counts.items():
        observed = summary["counts"].get(key)
        check(checks, f"independent_count_{key}", observed == expected, observed, expected, f"recomputed {key} from persisted development inputs", "source_or_artifact_integrity")
    check(checks, "independent_primary_anchor_keys_unique", not train.duplicated(["Instrument", "formation_session"]).any() and not validation.duplicated(["Instrument", "formation_session"]).any(), True, True, "recomputed primary anchors are unique per instrument/session", "source_or_artifact_integrity")
    independent_folds = independent_fold_specs(train, tuning_years, int(config["horizon_sessions"]))
    check(checks, "tuning_feature_set_names", set(tuning) == set(EXPECTED_FEATURE_SETS), sorted(tuning), EXPECTED_FEATURE_SETS, "RF tuning evidence covers all feature sets")
    for feature_set in EXPECTED_FEATURE_SETS:
        item = tuning.get(feature_set, {})
        evidence = item.get("evidence", [])
        candidate_ok = len(evidence) == len(config_grid)
        fold_ok = True
        candidate_match = True
        spec_ok = len(item.get("folds", [])) == len(independent_folds)
        for recorded, expected in zip(item.get("folds", []), independent_folds):
            spec_ok = spec_ok and all(recorded.get(field) == expected.get(field) for field in [
                "fold_id", "validation_year", "validation_start", "validation_end", "purge_rule",
                "last_fit_exit", "candidate_rows_before_purge", "purged_rows", "purge_sessions",
            ])
        for candidate, params in zip(evidence, config_grid):
            candidate_match = candidate_match and candidate.get("params") == params and len(candidate.get("folds", [])) == len(independent_folds)
            for recorded, expected in zip(candidate.get("folds", []), independent_folds):
                fold_ok = fold_ok and all(recorded.get(field) == expected.get(field) for field in [
                    "fold_id", "validation_year", "validation_start", "purge_rule",
                    "last_fit_exit", "candidate_rows_before_purge", "purged_rows",
                    "fit_rows", "validation_rows",
                ])
                fold_ok = fold_ok and pd.Timestamp(recorded.get("last_fit_exit")) < pd.Timestamp(recorded.get("validation_start"))
        check(checks, f"tuning_grid_{feature_set}", candidate_ok and candidate_match, {"candidate_count": len(evidence), "params_match": candidate_match}, {"candidate_count": len(config_grid), "params_match": True}, "RF candidate grid is the pre-registered grid")
        check(checks, f"tuning_purge_{feature_set}", fold_ok and spec_ok, {"candidate_evidence": [candidate.get("folds", []) for candidate in evidence], "fold_specs": item.get("folds", [])}, independent_folds, "RF evidence reproduces exit-session purged expanding folds and no label window crosses validation start", "source_or_artifact_integrity")
        selected = item.get("selected_params")
        check(checks, f"tuning_selected_{feature_set}", selected in config_grid, selected, "one registered candidate", "selected RF parameters are from the registered training-only grid")

    # Recheck the session-key joins without reading any target data from the
    # sealed test file.  Structural ETF NA values remain valid after a key hit.
    model_ready = ROOT / "data" / "model_ready_ai_pool_daily_v1" / config["model_ready_run_id"]
    macro_dir = ROOT / "data" / "clean" / "daily_macro_v1" / config["macro_run_id"]
    ai_dir = ROOT / "data" / "clean" / "ai_daily_macro_features_v1" / config["ai_state_run_id"]
    formation_keys = set(pd.to_datetime(pd.concat([train["formation_session"], validation["formation_session"]])).dt.normalize())
    macro_keys = set(pd.to_datetime(pd.read_csv(macro_dir / "macro_features_train_valid.csv", usecols=["spy_session"], low_memory=False)["spy_session"]).dt.normalize())
    ai_keys = set(pd.to_datetime(pd.read_csv(ai_dir / "ai_state_features.csv", usecols=["formation_session", "split"], low_memory=False).query("split in ['training','validation']")["formation_session"]).dt.normalize())
    check(checks, "macro_session_key_coverage", formation_keys <= macro_keys, len(formation_keys - macro_keys), 0, "all primary formation sessions have a macro key", "source_or_artifact_integrity")
    check(checks, "ai_state_session_key_coverage", formation_keys <= ai_keys, len(formation_keys - ai_keys), 0, "all primary formation sessions have an AI-state key", "source_or_artifact_integrity")
    safe_config = config.get("ai_state_safe_feature_audit")
    if safe_config:
        safe_summary_path = resolve_recorded_path(summary["inputs"]["files"]["ai_state_safe_feature_audit"]["path"])
        safe_summary = json.loads(safe_summary_path.read_text(encoding="utf-8"))
        cursor: Any = safe_summary
        for part in safe_config["feature_list_name"].split("."):
            cursor = cursor[part]
        check(checks, "safe_ai_feature_list_matches_audit", list(cursor) == config["ai_state_features"] and summary.get("feature_alias_exclusions", {}).get("effective_ai_state_feature_count") == len(cursor), {"config_features": len(config["ai_state_features"]), "audited_features": len(cursor)}, {"config_features": len(cursor), "audited_features": len(cursor)}, "coverage-safe AI features exactly match the independently audited safe list", "source_or_artifact_integrity")

    audit_path = run_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_checks = audit.get("checks", {})
    check(checks, "runner_audit_flags", all(value is True for value in audit_checks.values()), {key: value for key, value in audit_checks.items() if value is not True}, "all runner audit checks true", "runner audit agrees with independent validation")
    check(checks, "runner_audit_test_seal", audit.get("test_targets_opened") is False and audit.get("test_predictions_written") is False and audit.get("raw_inputs_modified") is False, audit, {"test_targets_opened": False, "test_predictions_written": False, "raw_inputs_modified": False}, "runner audit seals test and input mutation policy")
    processing = (ROOT / "DATA_PROCESSING_LOG.md").read_text(encoding="utf-8") if (ROOT / "DATA_PROCESSING_LOG.md").exists() else ""
    ai_log = (ROOT / "AI_USE_LOG.md").read_text(encoding="utf-8") if (ROOT / "AI_USE_LOG.md").exists() else ""
    check(checks, "processing_log_appended", run_id in processing, run_id in processing, True, "DATA_PROCESSING_LOG contains this run")
    check(checks, "ai_use_log_appended", run_id in ai_log, run_id in ai_log, True, "AI_USE_LOG contains this run")

    failures = [item for item in checks if not item["passed"]]
    status = "PASS" if not failures else "FAIL"
    audit_id = audit_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_root_value = Path(audit_root) if audit_root is not None else AUDIT_ROOT
    if not audit_root_value.is_absolute():
        audit_root_value = ROOT / audit_root_value
    out = audit_root_value / audit_id
    out.mkdir(parents=True, exist_ok=False)
    validation = {
        "schema_version": "ai_pool_daily_state_models_v1_validation",
        "validation_id": audit_id,
        "validated_at_utc": utc_now(),
        "run_id": run_id,
        "run_path": rel(run_dir),
        "status": status,
        "scope": "independent read-only validation of training/validation artifacts; sealed test target rows were not opened",
        "checks": checks,
        "failure_counts_by_classification": {
            "source_or_artifact_integrity": sum(item["failure_classification"] == "source_or_artifact_integrity" for item in failures),
            "metadata_or_validator_semantics": sum(item["failure_classification"] == "metadata_or_validator_semantics" for item in failures),
            "runner_audit_agreement": sum(item["failure_classification"] == "runner audit agrees with independent validation" for item in failures),
        },
        "sealed_test_policy": {
            "path": rel(model_ready / TARGET_SEALED_NAME),
            "opened": False,
            "rows_read": 0,
            "hash_computed": False,
        },
        "limitations": [
            "The validator verifies persisted development artifacts and source hashes; it does not inspect sealed test target rows or make any test performance claim.",
            "Input hash equality verifies that the runner did not mutate listed source files between the runner read and this validation.",
            "Structural pre-listing missingness is validated through the run summary and effective feature schema; imputer fitting remains inside persisted model pipelines.",
        ],
    }
    (out / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(checks).to_csv(out / "validation.csv", index=False)
    lines = [
        f"# Independent validation — {run_id}",
        "",
        f"- **Validation status**: `{status}`",
        f"- **Validation time (UTC)**: `{validation['validated_at_utc']}`",
        f"- **Model run**: `{rel(run_dir)}`",
        "- **Scope**: persisted training/validation artifacts, source hash chain, primary sample reconstruction, and training-only RF purge evidence.",
        "- **Sealed test policy**: `targets_test_sealed.csv` was not opened, read, hashed, or aggregated; test rows read = 0.",
        "",
        "## Check results",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| `{item['check_id']}` | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |")
    lines.extend(["", "## Findings", ""])
    if failures:
        for item in failures:
            lines.append(f"- **{item['check_id']}** (`{item['failure_classification']}`): observed `{item['observed']}`, expected `{item['expected']}`. {item['detail']}.")
    else:
        lines.append("- No failures. The persisted run passed all independent checks within the stated scope.")
    lines.extend(["", "## Processing limits", "", *[f"- {item}" for item in validation["limitations"]], ""])
    (out / "findings.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"validation_id": audit_id, "run_id": run_id, "status": status, "out": rel(out), "checks": len(checks), "failures": len(failures)}, ensure_ascii=False, indent=2), flush=True)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--audit-id", default=None)
    parser.add_argument("--run-root", default=None, help="model-run root relative to the workspace or absolute")
    parser.add_argument("--audit-root", default=None, help="audit root relative to the workspace or absolute")
    args = parser.parse_args()
    run(args.run_id, args.audit_id, args.run_root, args.audit_root)
