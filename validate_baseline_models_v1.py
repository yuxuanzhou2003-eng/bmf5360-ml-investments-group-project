"""Independent validation of the training/validation-only baseline run."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
RUN_ID = "20260909T070830647279Z"
RUN = ROOT / "data" / "model_runs" / "baseline_v1" / RUN_ID
MODEL_READY = ROOT / "data" / "model_ready_v1" / "20260909T064151673764Z"
AUDIT_ROOT = ROOT / "data" / "audit" / "baseline_v1"
EVENT_KEY = ["source", "announcement", "period_end"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(values):
    values = pd.Series(values).astype("string").str.lower()
    if set(values.dropna()) - {"true", "false"} or values.isna().any():
        raise ValueError("Invalid boolean")
    return values.eq("true")


def event_weights(frame):
    return (1 / frame.groupby(EVENT_KEY).sample_id.transform("size")).to_numpy(float)


def recompute_metrics(frame, prediction):
    y = frame.target.to_numpy(float)
    weight = event_weights(frame)
    error = prediction - y
    event_ics = []
    for _, group in frame.assign(prediction=prediction).groupby(EVENT_KEY):
        if len(group) >= 3 and group.target.nunique() > 1 and group.prediction.nunique() > 1:
            value = spearmanr(group.target, group.prediction).statistic
            if np.isfinite(value):
                event_ics.append(value)
    pooled = spearmanr(y, prediction).statistic if len(np.unique(prediction)) > 1 else np.nan
    return {
        "event_weighted_mse": np.average(error ** 2, weights=weight),
        "event_weighted_rmse": np.sqrt(np.average(error ** 2, weights=weight)),
        "event_weighted_mae": np.average(np.abs(error), weights=weight),
        "event_weighted_direction_accuracy": np.average(np.sign(prediction) == np.sign(y), weights=weight),
        "weighted_r2": r2_score(y, prediction, sample_weight=weight),
        "pooled_spearman": pooled,
        "mean_event_spearman": np.mean(event_ics) if event_ics else np.nan,
        "event_spearman_groups": len(event_ics),
    }


def main():
    paths = {
        "summary": RUN / "summary.json", "cv": RUN / "cv_results.csv",
        "metrics": RUN / "validation_metrics.csv", "predictions": RUN / "validation_predictions.csv",
        "params": RUN / "selected_hyperparameters.json", "feature_sets": RUN / "feature_sets.json",
        "features": MODEL_READY / "model_features.csv", "metadata": MODEL_READY / "metadata.csv",
        "targets": MODEL_READY / "targets.csv", "eligibility": MODEL_READY / "eligibility.csv",
        "validator_code": Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    cv = pd.read_csv(paths["cv"])
    reported = pd.read_csv(paths["metrics"])
    predictions = pd.read_csv(paths["predictions"])
    selected = json.loads(paths["params"].read_text(encoding="utf-8"))
    feature_sets = json.loads(paths["feature_sets"].read_text(encoding="utf-8"))
    checks = {
        "cv_has_69_candidate_fold_rows": len(cv) == 69,
        "cv_has_only_three_training_folds": set(cv.fold) == {"validate_2018", "validate_2019", "validate_2020"},
        "validation_predictions_are_2021_2022_only": bool(pd.to_datetime(
            predictions.announcement_day).dt.normalize().between("2021-01-01", "2022-12-31").all()),
        "no_test_prediction_column_or_rows": not any("test" in column.lower() for column in predictions.columns),
    }
    expected_selected = {}
    for model_name, group in cv.groupby("model"):
        means = group.groupby("params").event_weighted_mse.mean()
        expected_selected[model_name] = json.loads(means.idxmin())
    checks["selected_hyperparameters_minimize_training_cv_mse"] = all(
        expected_selected[name] == selected[name] for name in expected_selected)

    features = pd.read_csv(paths["features"])
    metadata = pd.read_csv(paths["metadata"], low_memory=False)
    targets = pd.read_csv(paths["targets"], usecols=["sample_id", "forward_benchmark_excess"])
    eligibility = pd.read_csv(paths["eligibility"], low_memory=False)
    data = features.merge(metadata[["sample_id", *EVENT_KEY, "announcement_day"]], on="sample_id", validate="one_to_one")
    data = data.merge(targets.rename(columns={"forward_benchmark_excess": "target"}), on="sample_id", validate="one_to_one")
    data = data.merge(eligibility[["sample_id", "split", "supervised_model_eligible"]], on="sample_id", validate="one_to_one")
    data["supervised_model_eligible"] = parse_bool(data.supervised_model_eligible).to_numpy()
    validation = data.loc[data.supervised_model_eligible & data.split.eq("validation")].copy()
    checks["validation_rows_recomputed"] = len(validation) == len(predictions) == 19392
    checks["prediction_sample_ids_exact"] = set(validation.sample_id) == set(predictions.sample_id)
    validation = predictions[["sample_id"]].merge(validation, on="sample_id", validate="one_to_one")

    metric_checks = []
    prediction_checks = []
    for model_name in ["simple_network", "receiver_ridge", "network_ridge", "network_hgb"]:
        model = joblib.load(RUN / "models" / f"{model_name}.joblib")
        recreated = model.predict(validation[feature_sets[model_name]])
        stored = predictions.set_index("sample_id").loc[validation.sample_id, f"prediction_{model_name}"].to_numpy()
        prediction_checks.append(np.allclose(recreated, stored, rtol=1e-12, atol=1e-12))
    checks["stored_models_reproduce_validation_predictions"] = bool(all(prediction_checks))
    joined_predictions = predictions.set_index("sample_id").loc[validation.sample_id]
    for row in reported.itertuples(index=False):
        pred = joined_predictions[f"prediction_{row.model}"].to_numpy()
        recalculated = recompute_metrics(validation, pred)
        for key, value in recalculated.items():
            reported_value = getattr(row, key)
            metric_checks.append((reported_value == value) if isinstance(value, int) else
                                 np.isclose(reported_value, value, rtol=1e-11, atol=1e-13, equal_nan=True))
    checks["all_validation_metrics_recomputed"] = bool(all(metric_checks))
    non_null = reported.loc[~reported.model.eq("training_weighted_mean")].sort_values(
        ["mean_event_spearman", "event_weighted_mse"], ascending=[False, True])
    checks["champion_recomputed"] = non_null.iloc[0].model == summary["validation_champion_by_predeclared_metric"]
    checks["runner_inputs_match_recorded_hashes"] = all(
        sha256_file(ROOT / item["path"]) == item["sha256"] for item in summary["input_hashes"].values())
    checks["run_outputs_match_recorded_hashes"] = all(
        sha256_file(RUN / (f"{name}.csv" if name in {"cv_results", "validation_metrics", "validation_predictions"}
                           else f"{name}.json")) == digest
        for name, digest in summary["output_hashes"].items())
    checks["model_files_match_recorded_hashes"] = all(
        sha256_file(RUN / "models" / f"{name}.joblib") == digest for name, digest in summary["model_hashes"].items())

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = AUDIT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "baseline_run_id": RUN_ID, "passed": int(sum(checks.values())),
              "total": len(checks), "all_passed": bool(all(checks.values())), "checks": checks,
              "recomputed_validation_metrics": reported.to_dict(orient="records"),
              "input_hashes": {name: sha256_file(path) for name, path in paths.items()}}
    (out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["run_id", "baseline_run_id", "passed", "total", "all_passed"]}, indent=2))
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    main()
