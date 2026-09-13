"""Fit and compare training/validation-only baselines. Test data are not scored."""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_RUN_ID = "20260909T064151673764Z"
DATA = ROOT / "data" / "model_ready_v1" / MODEL_RUN_ID
CONFIG_PATH = ROOT / "baseline_model_v1_config.json"
OUT_ROOT = ROOT / "data" / "model_runs" / "baseline_v1"
EVENT_KEY = ["source", "announcement", "period_end"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def parse_bool(values, name: str) -> pd.Series:
    normalized = pd.Series(values).astype("string").str.lower()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name}: {invalid}")
    return normalized.eq("true")


def event_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(EVENT_KEY, dropna=False).sample_id.transform("size")
    return (1.0 / counts).to_numpy(float)


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    y = frame.target.to_numpy(float)
    weight = event_weights(frame)
    error = prediction - y
    pooled_ic = spearmanr(y, prediction).statistic if len(np.unique(prediction)) > 1 else np.nan
    event_ics = []
    for _, group in frame.assign(prediction=prediction).groupby(EVENT_KEY, dropna=False):
        if len(group) >= 3 and group.target.nunique() > 1 and group.prediction.nunique() > 1:
            value = spearmanr(group.target, group.prediction).statistic
            if np.isfinite(value):
                event_ics.append(float(value))
    return {
        "rows": int(len(frame)),
        "events": int(frame[EVENT_KEY].drop_duplicates().shape[0]),
        "event_weighted_mse": float(np.average(error ** 2, weights=weight)),
        "event_weighted_rmse": float(np.sqrt(np.average(error ** 2, weights=weight))),
        "event_weighted_mae": float(np.average(np.abs(error), weights=weight)),
        "event_weighted_direction_accuracy": float(np.average(np.sign(prediction) == np.sign(y), weights=weight)),
        "weighted_r2": float(r2_score(y, prediction, sample_weight=weight)),
        "pooled_spearman": float(pooled_ic) if np.isfinite(pooled_ic) else np.nan,
        "mean_event_spearman": float(np.mean(event_ics)) if event_ics else np.nan,
        "event_spearman_groups": int(len(event_ics)),
    }


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=alpha)),
    ])


def hgb_pipeline(params: dict, random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"], max_iter=params["max_iter"],
            min_samples_leaf=params["min_samples_leaf"], random_state=random_state)),
    ])


def fit_with_weights(model, x, y, weight):
    model.fit(x, y, **{"regressor__sample_weight": weight})
    return model


def main():
    paths = {name: DATA / f"{name}.csv" for name in
             ["model_features", "metadata", "targets", "eligibility"]}
    paths.update({"config": CONFIG_PATH, "runner_code": Path(__file__).resolve(),
                  "requirements_ml": ROOT / "requirements-ml.txt"})
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    features = pd.read_csv(paths["model_features"])
    metadata = pd.read_csv(paths["metadata"], low_memory=False)
    targets = pd.read_csv(paths["targets"], usecols=["sample_id", config["target"]])
    eligibility = pd.read_csv(paths["eligibility"], low_memory=False)
    data = features.merge(metadata[["sample_id", *EVENT_KEY, "announcement_day", "exit_session"]],
                          on="sample_id", validate="one_to_one")
    data = data.merge(targets.rename(columns={config["target"]: "target"}),
                      on="sample_id", validate="one_to_one")
    data = data.merge(eligibility[["sample_id", "split", "supervised_model_eligible"]],
                      on="sample_id", validate="one_to_one")
    data["announcement_day"] = pd.to_datetime(data.announcement_day).dt.normalize()
    data["exit_session"] = pd.to_datetime(data.exit_session).dt.normalize()
    data["supervised_model_eligible"] = parse_bool(
        data.supervised_model_eligible, "supervised_model_eligible").to_numpy()
    dev = data.loc[data.supervised_model_eligible & data.split.isin(["training", "validation"])].copy()
    if dev.target.isna().any() or set(dev.split) != {"training", "validation"}:
        raise RuntimeError("Development sample is invalid")

    all_features = [column for column in features if column != "sample_id"]
    receiver_features = [column for column in all_features if column.startswith("receiver_") or column.startswith("market_")]
    network_features = all_features
    feature_sets = {"simple_network": ["network_signal"],
                    "receiver_ridge": receiver_features,
                    "network_ridge": network_features,
                    "network_hgb": network_features}
    cv_rows = []
    folds = config["internal_training_folds"]

    def evaluate_candidate(model_name, params, columns, factory):
        fold_metrics = []
        for fold in folds:
            validation_start = pd.Timestamp(fold["validation_start"])
            train = dev.loc[(dev.split == "training") &
                            (dev.announcement_day <= pd.Timestamp(fold["train_end"])) &
                            (dev.exit_session < validation_start)].copy()
            valid = dev.loc[(dev.split == "training") & dev.announcement_day.between(
                fold["validation_start"], fold["validation_end"], inclusive="both")].copy()
            model = factory(params)
            fit_with_weights(model, train[columns], train.target, event_weights(train))
            result = metrics(valid, model.predict(valid[columns]))
            result.update({"model": model_name, "params": json.dumps(params, sort_keys=True),
                           "fold": fold["name"], "train_rows": len(train), "train_events": train[EVENT_KEY].drop_duplicates().shape[0]})
            cv_rows.append(result)
            fold_metrics.append(result["event_weighted_mse"])
        return float(np.mean(fold_metrics))

    print("training-only rolling CV", flush=True)
    simple_factory = lambda _: Pipeline([("imputer", SimpleImputer(strategy="median")),
                                         ("regressor", LinearRegression())])
    simple_score = evaluate_candidate("simple_network", {}, feature_sets["simple_network"], simple_factory)
    candidate_scores = {"simple_network": [({}, simple_score)]}
    for model_name in ["receiver_ridge", "network_ridge"]:
        candidate_scores[model_name] = []
        for alpha in config["ridge_alphas"]:
            params = {"alpha": alpha}
            score = evaluate_candidate(model_name, params, feature_sets[model_name],
                                       lambda p: ridge_pipeline(p["alpha"]))
            candidate_scores[model_name].append((params, score))
    grid = config["hist_gradient_boosting_grid"]
    candidate_scores["network_hgb"] = []
    for learning_rate, leaves, l2 in itertools.product(
            grid["learning_rate"], grid["max_leaf_nodes"], grid["l2_regularization"]):
        params = {"learning_rate": learning_rate, "max_leaf_nodes": leaves,
                  "l2_regularization": l2, "max_iter": grid["max_iter"],
                  "min_samples_leaf": grid["min_samples_leaf"]}
        score = evaluate_candidate("network_hgb", params, feature_sets["network_hgb"],
                                   lambda p: hgb_pipeline(p, config["random_state"]))
        candidate_scores["network_hgb"].append((params, score))

    selected_params = {name: min(values, key=lambda item: item[1])[0]
                       for name, values in candidate_scores.items()}
    train = dev.loc[dev.split.eq("training")].copy()
    validation = dev.loc[dev.split.eq("validation")].copy()
    print(f"fitting selected models on {len(train)} training rows; validation rows {len(validation)}", flush=True)
    fitted = {}
    fitted["simple_network"] = fit_with_weights(
        simple_factory({}), train[feature_sets["simple_network"]], train.target, event_weights(train))
    fitted["receiver_ridge"] = fit_with_weights(
        ridge_pipeline(selected_params["receiver_ridge"]["alpha"]),
        train[feature_sets["receiver_ridge"]], train.target, event_weights(train))
    fitted["network_ridge"] = fit_with_weights(
        ridge_pipeline(selected_params["network_ridge"]["alpha"]),
        train[feature_sets["network_ridge"]], train.target, event_weights(train))
    fitted["network_hgb"] = fit_with_weights(
        hgb_pipeline(selected_params["network_hgb"], config["random_state"]),
        train[feature_sets["network_hgb"]], train.target, event_weights(train))

    validation_rows = []
    predictions = validation[["sample_id", *EVENT_KEY, "announcement_day", "target"]].copy()
    train_mean = float(np.average(train.target, weights=event_weights(train)))
    null_prediction = np.full(len(validation), train_mean)
    null_metrics = metrics(validation, null_prediction)
    null_metrics.update({"model": "training_weighted_mean", "params": "{}"})
    validation_rows.append(null_metrics)
    predictions["prediction_training_weighted_mean"] = null_prediction
    for name, model in fitted.items():
        prediction = model.predict(validation[feature_sets[name]])
        result = metrics(validation, prediction)
        result.update({"model": name, "params": json.dumps(selected_params[name], sort_keys=True)})
        validation_rows.append(result)
        predictions[f"prediction_{name}"] = prediction
    validation_metrics = pd.DataFrame(validation_rows)
    eligible_champions = validation_metrics.loc[~validation_metrics.model.eq("training_weighted_mean")].dropna(
        subset=["mean_event_spearman"])
    champion = eligible_champions.sort_values(
        ["mean_event_spearman", "event_weighted_mse"], ascending=[False, True]).iloc[0].model

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "cv_results": out / "cv_results.csv",
        "validation_metrics": out / "validation_metrics.csv",
        "validation_predictions": out / "validation_predictions.csv",
        "selected_hyperparameters": out / "selected_hyperparameters.json",
        "feature_sets": out / "feature_sets.json",
    }
    pd.DataFrame(cv_rows).to_csv(outputs["cv_results"], index=False)
    validation_metrics.to_csv(outputs["validation_metrics"], index=False)
    predictions.to_csv(outputs["validation_predictions"], index=False)
    outputs["selected_hyperparameters"].write_text(
        json.dumps(selected_params, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["feature_sets"].write_text(
        json.dumps(feature_sets, indent=2, ensure_ascii=False), encoding="utf-8")
    model_paths = {}
    for name, model in fitted.items():
        path = models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        model_paths[name] = path

    cv = pd.DataFrame(cv_rows)
    chronological_folds = all(pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"])
                              for fold in folds)
    training_weight_sums = pd.Series(event_weights(train), index=train.index).groupby(
        train[EVENT_KEY].astype(str).agg("|".join, axis=1)).sum()
    checks = {
        "development_has_no_test_rows": not dev.split.eq("test").any(),
        "validation_predictions_have_no_test_rows": bool(predictions.announcement_day.between(
            "2021-01-01", "2022-12-31", inclusive="both").all()),
        "training_rows_match_frozen_eligibility": len(train) == 54241,
        "validation_rows_match_frozen_eligibility": len(validation) == 19392,
        "internal_folds_chronological": chronological_folds,
        "cv_rows_only_expected_folds": set(cv.fold) == {fold["name"] for fold in folds},
        "event_weights_sum_to_one": bool(np.allclose(training_weight_sums, 1.0)),
        "all_validation_predictions_finite": bool(np.isfinite(
            predictions.filter(like="prediction_").to_numpy(float)).all()),
        "all_models_have_validation_metrics": set(validation_metrics.model) == {
            "training_weighted_mean", "simple_network", "receiver_ridge", "network_ridge", "network_hgb"},
        "champion_is_non_null_model": champion in fitted,
    }
    after_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if after_hashes != {name: item["sha256"] for name, item in input_hashes.items()}:
        raise RuntimeError("A model input changed during fitting")
    summary = {
        "run_id": run_id,
        "model_ready_run_id": MODEL_RUN_ID,
        "scope": "training and validation only; test metrics and predictions were not produced",
        "target": config["target"],
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_events": int(train[EVENT_KEY].drop_duplicates().shape[0]),
        "validation_events": int(validation[EVENT_KEY].drop_duplicates().shape[0]),
        "selected_hyperparameters": selected_params,
        "validation_champion_by_predeclared_metric": champion,
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "config": config,
        "feature_sets": feature_sets,
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "input_hashes": input_hashes,
        "output_hashes": {name: sha256_file(path) for name, path in outputs.items()},
        "model_hashes": {name: sha256_file(path) for name, path in model_paths.items()},
        "checks": checks,
        "limitations": [
            "Validation comparison is model development evidence, not an out-of-sample fund result.",
            "Event rows are dependent; reported prediction metrics are descriptive and do not imply IID significance.",
            "Portfolio aggregation, exposure constraints, turnover and transaction costs have not been applied.",
            "Test-period predictions and metrics remain intentionally absent.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "champion": champion,
                      "selected_hyperparameters": selected_params,
                      "validation_metrics": summary["validation_metrics"], "checks": checks},
                     indent=2, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError("Baseline checks failed: " + ", ".join(k for k, value in checks.items() if not value))


if __name__ == "__main__":
    main()
