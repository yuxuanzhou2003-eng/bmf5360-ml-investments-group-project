"""Training/validation-only analyst-network model comparison.

The runner is deliberately sealed to one frozen analyst-graph model-ready run.  Eligibility
IDs are filtered before target values are joined.  No model, metric or prediction is ever built
for a test row; the only test-target operation is a seal check that the target value is missing.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "analyst_model_v1_config.json"
MODEL_READY_RUN_ID = "20260909T125156092817Z"
DATA = ROOT / "data" / "analyst_graph_model_ready_v1" / MODEL_READY_RUN_ID
OUT_ROOT = ROOT / "data" / "model_runs" / "analyst_v1"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_bool(values: pd.Series, name: str) -> pd.Series:
    normalized = pd.Series(values, index=values.index).astype("string").str.strip().str.lower()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name}: invalid boolean values {invalid}")
    return normalized.eq("true")


def event_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every event the same total weight within this frame."""
    if frame.empty:
        return np.array([], dtype=float)
    counts = frame.groupby(EVENT_KEY, dropna=False, sort=False)["sample_id"].transform("size")
    return (1.0 / counts.to_numpy(float))


def safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if len(left_array) < 3 or len(right_array) < 3:
        return np.nan
    if len(np.unique(left_array)) < 2 or len(np.unique(right_array)) < 2:
        return np.nan
    value = spearmanr(left_array, right_array).statistic
    return float(value) if np.isfinite(value) else np.nan


def safe_nanmean(values: list[float]) -> float:
    """Mean of finite values, returning NaN without emitting an empty-slice warning."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else np.nan


def vectorized_event_spearman(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Compute exact within-event Spearman correlations without Python group loops."""
    work = frame[EVENT_KEY + ["target"]].copy().reset_index(drop=True)
    work["prediction"] = np.asarray(prediction, dtype=float)
    grouped = work.groupby(EVENT_KEY, dropna=False, sort=False)
    work["target_rank"] = grouped["target"].rank(method="average")
    work["prediction_rank"] = grouped["prediction"].rank(method="average")
    work["rank_product"] = work["target_rank"] * work["prediction_rank"]
    work["target_rank_sq"] = work["target_rank"] ** 2
    work["prediction_rank_sq"] = work["prediction_rank"] ** 2
    stats = work.groupby(EVENT_KEY, as_index=False, dropna=False, sort=False).agg(
        rows=("target", "size"),
        target_unique=("target", "nunique"),
        prediction_unique=("prediction", "nunique"),
        sum_target_rank=("target_rank", "sum"),
        sum_prediction_rank=("prediction_rank", "sum"),
        sum_rank_product=("rank_product", "sum"),
        sum_target_rank_sq=("target_rank_sq", "sum"),
        sum_prediction_rank_sq=("prediction_rank_sq", "sum"),
    )
    n = stats["rows"].to_numpy(float)
    covariance = stats["sum_rank_product"] - (
        stats["sum_target_rank"] * stats["sum_prediction_rank"] / n
    )
    target_ss = stats["sum_target_rank_sq"] - stats["sum_target_rank"] ** 2 / n
    prediction_ss = stats["sum_prediction_rank_sq"] - stats["sum_prediction_rank"] ** 2 / n
    denominator = np.sqrt(target_ss * prediction_ss)
    eligible = (
        stats["rows"].ge(3)
        & stats["target_unique"].ge(2)
        & stats["prediction_unique"].ge(2)
        & denominator.gt(0)
    )
    stats["event_ic"] = np.where(eligible, covariance / denominator, np.nan)
    stats["ic_eligible"] = eligible & np.isfinite(stats["event_ic"])
    return stats


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    if len(frame) != len(prediction):
        raise ValueError("Prediction length does not match scoring frame")
    if frame.target.isna().any():
        raise ValueError("Scoring frame contains missing target")
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Scoring prediction contains NaN or infinity")
    target = frame.target.to_numpy(float)
    weights = event_weights(frame)
    error = prediction - target
    event_stats = vectorized_event_spearman(frame, prediction)
    event_ics = event_stats.loc[event_stats.ic_eligible, "event_ic"].to_numpy(float)
    pooled = safe_spearman(target, prediction)
    return {
        "rows": int(len(frame)),
        "events": int(frame[EVENT_KEY].drop_duplicates().shape[0]),
        "event_weighted_mse": float(np.average(error**2, weights=weights)),
        "event_weighted_rmse": float(np.sqrt(np.average(error**2, weights=weights))),
        "event_weighted_mae": float(np.average(np.abs(error), weights=weights)),
        "event_weighted_direction_accuracy": float(
            np.average(np.sign(prediction) == np.sign(target), weights=weights)
        ),
        "weighted_r2": float(r2_score(target, prediction, sample_weight=weights)),
        "pooled_spearman": float(pooled) if np.isfinite(pooled) else np.nan,
        "mean_event_spearman": float(np.mean(event_ics)) if len(event_ics) else np.nan,
        "event_spearman_groups": int(len(event_ics)),
    }


def event_ic_table(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    model: str,
    split: str,
    comparison_role: str,
) -> pd.DataFrame:
    """Return one row for every event, preserving invalid/constant-event reasons."""
    work = frame[EVENT_KEY + ["announcement_day", "target"]].copy().reset_index(drop=True)
    work["prediction"] = np.asarray(prediction, dtype=float)
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(EVENT_KEY, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        target_unique = int(group.target.nunique(dropna=True))
        prediction_unique = int(group.prediction.nunique(dropna=True))
        if len(group) < 3:
            value, reason, eligible = np.nan, "fewer_than_3_rows", False
        elif target_unique < 2:
            value, reason, eligible = np.nan, "constant_target", False
        elif prediction_unique < 2:
            value, reason, eligible = np.nan, "constant_prediction", False
        else:
            value = safe_spearman(group.target, group.prediction)
            eligible = bool(np.isfinite(value))
            reason = "ok" if eligible else "nonfinite_spearman"
        announcement_day = pd.Timestamp(group.announcement_day.iloc[0]).normalize()
        record = dict(zip(EVENT_KEY, key))
        record.update(
            {
                "model": model,
                "split": split,
                "comparison_role": comparison_role,
                "announcement_day": announcement_day.strftime("%Y-%m-%d"),
                "announcement_month": announcement_day.strftime("%Y-%m"),
                "rows": int(len(group)),
                "target_unique": target_unique,
                "prediction_unique": prediction_unique,
                "event_ic": float(value) if np.isfinite(value) else np.nan,
                "ic_eligible": bool(eligible),
                "ic_reason": reason,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(alpha))),
        ]
    )


def hgb_pipeline(params: dict[str, Any], random_state: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=float(params["learning_rate"]),
                    max_leaf_nodes=int(params["max_leaf_nodes"]),
                    l2_regularization=float(params["l2_regularization"]),
                    max_iter=int(params["max_iter"]),
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def fit_weighted(model: Pipeline, x: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> Pipeline:
    all_missing = [column for column in x.columns if x[column].isna().all()]
    if all_missing:
        raise RuntimeError(
            "A declared training feature has no observed value; refusing implicit zero fill: "
            + ", ".join(all_missing)
        )
    model.fit(x, y, regressor__sample_weight=weights)
    return model


def prefixed(columns: list[str], prefixes: list[str]) -> list[str]:
    return [column for column in columns if any(column.startswith(prefix) for prefix in prefixes)]


def resolve_feature_sets(features: list[str], config: dict[str, Any]) -> dict[str, Any]:
    rules = config["feature_rules"]
    all_features = list(features)
    simple = list(rules["simple_rec_weighted_signal"]["columns"])
    receiver_only = prefixed(all_features, rules["receiver_only_ridge"]["include_prefixes"])
    context_rule = rules["context_ridge"]
    context = prefixed(all_features, context_rule["include_prefixes"])
    context.extend(column for column in context_rule["include_exact"] if column in all_features)
    context = [column for column in all_features if column in set(context)]
    event_rule = rules["two_stage_ridge"]["event_model"]
    event_features = prefixed(all_features, event_rule["include_prefixes"])
    event_features.extend(column for column in event_rule["include_exact"] if column in all_features)
    event_features = [column for column in all_features if column in set(event_features)]
    residual_rule = rules["two_stage_ridge"]["within_event_residual_model"]
    relationship_fields = list(residual_rule["include_exact"])
    residual_features = prefixed(all_features, residual_rule["include_prefixes"])
    residual_features.extend(column for column in relationship_fields if column in all_features)
    residual_features = [column for column in all_features if column in set(residual_features)]
    feature_sets: dict[str, Any] = {
        "all_52": all_features,
        "simple_rec_weighted_signal": simple,
        "receiver_only_ridge": receiver_only,
        "context_ridge": context,
        "analyst_ridge": all_features,
        "analyst_hgb": all_features,
        "two_stage_ridge": {
            "event_model": event_features,
            "within_event_residual_model": residual_features,
            "relationship_fields": relationship_fields,
        },
    }
    return feature_sets


def check_feature_sets(feature_sets: dict[str, Any], all_features: list[str]) -> None:
    all_set = set(all_features)
    if len(all_features) != 52 or len(all_set) != 52:
        raise RuntimeError(f"Expected exactly 52 unique model features, observed {len(all_features)}")
    for name in [
        "simple_rec_weighted_signal",
        "receiver_only_ridge",
        "context_ridge",
        "analyst_ridge",
        "analyst_hgb",
    ]:
        columns = feature_sets[name]
        if not columns or len(columns) != len(set(columns)) or not set(columns).issubset(all_set):
            raise RuntimeError(f"Invalid feature set {name}")
    two_stage = feature_sets["two_stage_ridge"]
    if not set(two_stage["relationship_fields"]).issubset(all_set):
        raise RuntimeError("Declared relationship fields are missing from model_features.csv")
    for name in ["event_model", "within_event_residual_model"]:
        columns = two_stage[name]
        if not columns or len(columns) != len(set(columns)) or not set(columns).issubset(all_set):
            raise RuntimeError(f"Invalid two-stage feature set {name}")
    if feature_sets["simple_rec_weighted_signal"] != [
        "standardized_surprise",
        "jaccard_rec_weighted",
    ]:
        raise RuntimeError("Simple signal feature set drifted from the declared formula")
    if any(column.startswith("source_") for column in feature_sets["receiver_only_ridge"]):
        raise RuntimeError("receiver_only_ridge unexpectedly contains source features")
    forbidden_context = {
        "jaccard_coverage",
        "jaccard_rec_weighted",
        "surprise_x_jaccard_coverage",
        "surprise_x_jaccard_rec_weighted",
        "common_brokers",
        "common_brokers_log1p",
        "common_rated_brokers",
        "common_rated_brokers_log1p",
        "graph_snapshot_age_days",
        "event_neighbor_count",
        "neighbor_rank_jaccard_coverage",
        "neighbor_rank_jaccard_rec_weighted",
    }
    if forbidden_context.intersection(feature_sets["context_ridge"]):
        raise RuntimeError("context_ridge contains a forbidden relationship feature")


def event_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    counts = frame.groupby(EVENT_KEY, dropna=False, sort=False)[columns].nunique(dropna=False)
    varying = {column: int((counts[column] > 1).sum()) for column in columns if (counts[column] > 1).any()}
    if varying:
        raise RuntimeError(f"Event-level features are not constant within event: {varying}")
    return (
        frame.groupby(EVENT_KEY, as_index=False, sort=False, dropna=False)[columns]
        .first()
    )


def fit_two_stage(
    train: pd.DataFrame,
    params: dict[str, Any],
    feature_sets: dict[str, Any],
) -> dict[str, Any]:
    event_columns = feature_sets["two_stage_ridge"]["event_model"]
    residual_columns = feature_sets["two_stage_ridge"]["within_event_residual_model"]
    event_rows = event_feature_frame(train, event_columns)
    event_targets = (
        train.groupby(EVENT_KEY, as_index=False, sort=False, dropna=False)["target"]
        .mean()
        .rename(columns={"target": "event_target"})
    )
    event_rows = event_rows.merge(event_targets, on=EVENT_KEY, how="inner", validate="one_to_one")
    event_model = fit_weighted(
        ridge_pipeline(float(params["event_alpha"])),
        event_rows[event_columns],
        event_rows["event_target"],
        np.ones(len(event_rows), dtype=float),
    )
    with_event_target = train.merge(event_targets, on=EVENT_KEY, how="left", validate="many_to_one")
    if with_event_target.event_target.isna().any():
        raise RuntimeError("Training event target mean could not be attached")
    residual_target = with_event_target.target - with_event_target.event_target
    residual_model = fit_weighted(
        ridge_pipeline(float(params["residual_alpha"])),
        train[residual_columns],
        residual_target,
        event_weights(train),
    )
    return {
        "kind": "two_stage_ridge",
        "event_model": event_model,
        "within_event_residual_model": residual_model,
        "event_alpha": float(params["event_alpha"]),
        "residual_alpha": float(params["residual_alpha"]),
        "event_feature_columns": event_columns,
        "within_event_residual_feature_columns": residual_columns,
    }


def predict_two_stage(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    event_columns = bundle["event_feature_columns"]
    residual_columns = bundle["within_event_residual_feature_columns"]
    event_rows = event_feature_frame(frame, event_columns)
    event_rows["event_prediction"] = bundle["event_model"].predict(event_rows[event_columns])
    event_prediction = frame[EVENT_KEY].merge(
        event_rows[EVENT_KEY + ["event_prediction"]],
        on=EVENT_KEY,
        how="left",
        validate="many_to_one",
    )["event_prediction"].to_numpy(float)
    residual_prediction = bundle["within_event_residual_model"].predict(frame[residual_columns])
    return event_prediction + np.asarray(residual_prediction, dtype=float)


def fit_candidate(
    name: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    feature_sets: dict[str, Any],
    random_state: int,
) -> Any:
    if name == "training_weighted_mean":
        return {
            "kind": "constant",
            "model_name": name,
            "value": float(np.average(train.target, weights=event_weights(train))),
        }
    if name == "simple_rec_weighted_signal":
        return {
            "kind": "formula",
            "model_name": name,
            "formula": "standardized_surprise * jaccard_rec_weighted",
            "feature_columns": feature_sets[name],
        }
    if name in {"receiver_only_ridge", "context_ridge", "analyst_ridge"}:
        return fit_weighted(
            ridge_pipeline(float(params["alpha"])),
            train[feature_sets[name]],
            train.target,
            event_weights(train),
        )
    if name == "analyst_hgb":
        return fit_weighted(
            hgb_pipeline(params, random_state),
            train[feature_sets[name]],
            train.target,
            event_weights(train),
        )
    if name == "two_stage_ridge":
        return fit_two_stage(train, params, feature_sets)
    raise KeyError(name)


def predict_candidate(
    name: str,
    bundle: Any,
    frame: pd.DataFrame,
    feature_sets: dict[str, Any],
) -> np.ndarray:
    if name == "training_weighted_mean":
        return np.full(len(frame), float(bundle["value"]), dtype=float)
    if name == "simple_rec_weighted_signal":
        prediction = (
            frame["standardized_surprise"].to_numpy(float)
            * frame["jaccard_rec_weighted"].to_numpy(float)
        )
        if "surprise_x_jaccard_rec_weighted" in frame:
            declared = frame["surprise_x_jaccard_rec_weighted"].to_numpy(float)
            if not np.allclose(prediction, declared, equal_nan=True, rtol=1e-10, atol=1e-12):
                raise RuntimeError("Simple signal is inconsistent with the precomputed interaction")
        return prediction
    if name == "two_stage_ridge":
        return predict_two_stage(bundle, frame)
    return np.asarray(bundle.predict(frame[feature_sets[name]]), dtype=float)


def fold_samples(dev: pd.DataFrame, fold: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_start = pd.Timestamp(fold["validation_start"])
    train = dev.loc[
        dev.split.eq("training")
        & dev.announcement_day.le(pd.Timestamp(fold["train_end"]))
        & dev.exit_session.lt(validation_start)
    ].copy()
    valid = dev.loc[
        dev.split.eq("training")
        & dev.announcement_day.between(
            pd.Timestamp(fold["validation_start"]),
            pd.Timestamp(fold["validation_end"]),
            inclusive="both",
        )
    ].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"Empty rolling fold {fold['name']}")
    if train.target.isna().any() or valid.target.isna().any():
        raise RuntimeError(f"Missing target in rolling fold {fold['name']}")
    return train, valid


def candidate_grids(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    alphas = [float(value) for value in config["hyperparameters"]["ridge_alphas"]]
    grid = config["hyperparameters"]["hist_gradient_boosting_grid"]
    hgb_params = [
        {
            "learning_rate": float(learning_rate),
            "max_leaf_nodes": int(leaves),
            "l2_regularization": float(l2),
            "max_iter": int(grid["max_iter"]),
            "min_samples_leaf": int(grid["min_samples_leaf"]),
        }
        for learning_rate, leaves, l2 in itertools.product(
            grid["learning_rate"], grid["max_leaf_nodes"], grid["l2_regularization"]
        )
    ]
    return {
        "training_weighted_mean": [{}],
        "simple_rec_weighted_signal": [{}],
        "receiver_only_ridge": [{"alpha": alpha} for alpha in alphas],
        "context_ridge": [{"alpha": alpha} for alpha in alphas],
        "analyst_ridge": [{"alpha": alpha} for alpha in alphas],
        "analyst_hgb": hgb_params,
        "two_stage_ridge": [
            {"event_alpha": event_alpha, "residual_alpha": residual_alpha}
            for event_alpha, residual_alpha in itertools.product(alphas, alphas)
        ],
    }


def select_best_params(summary: pd.DataFrame, model: str) -> tuple[dict[str, Any], pd.Series]:
    candidates = summary.loc[summary.model.eq(model)].copy()
    candidates = candidates.loc[candidates.mean_event_spearman.notna()].copy()
    if candidates.empty:
        raise RuntimeError(f"No usable mean event Spearman score for {model}")
    candidates = candidates.sort_values(
        ["mean_event_spearman", "mean_event_weighted_mse", "params"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    best = candidates.iloc[0]
    return json.loads(best.params), best


def bootstrap_delta(
    per_event: pd.DataFrame,
    nominee: str,
    context_name: str,
    seed: int,
    replications: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    relevant = per_event.loc[
        per_event.model.isin([nominee, context_name]) & per_event.ic_eligible
    ].copy()
    pivot = relevant.pivot_table(index=EVENT_KEY, columns="model", values="event_ic", aggfunc="first")
    if nominee not in pivot.columns or context_name not in pivot.columns:
        raise RuntimeError("Cannot form paired nominee/context event ICs")
    event_info = (
        per_event.loc[per_event.model.eq(nominee), EVENT_KEY + ["announcement_month"]]
        .drop_duplicates(EVENT_KEY)
    )
    paired = pivot[[nominee, context_name]].dropna().reset_index().merge(
        event_info, on=EVENT_KEY, how="left", validate="one_to_one"
    )
    paired["delta_mean_event_ic"] = paired[nominee] - paired[context_name]
    if paired.empty or paired.announcement_month.isna().any():
        raise RuntimeError("No complete event pairs for announcement-month bootstrap")
    month_values = {
        month: group.delta_mean_event_ic.to_numpy(float)
        for month, group in paired.groupby("announcement_month", sort=True)
    }
    months = np.array(sorted(month_values), dtype=object)
    rng = np.random.default_rng(int(seed))
    results = np.empty(int(replications), dtype=float)
    for index in range(int(replications)):
        sampled = rng.choice(months, size=len(months), replace=True)
        results[index] = float(
            np.concatenate([month_values[month] for month in sampled]).mean()
        )
    table = pd.DataFrame(
        {
            "replicate": np.arange(1, int(replications) + 1, dtype=int),
            "delta_mean_event_ic": results,
        }
    )
    confidence_level = 0.95
    lower, upper = np.quantile(results, [(1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2])
    summary = {
        "comparison": f"{nominee} minus {context_name}",
        "nominee": nominee,
        "context_model": context_name,
        "statistic": "mean paired event Spearman delta",
        "block": "announcement calendar month",
        "seed": int(seed),
        "replications": int(replications),
        "confidence_level": confidence_level,
        "confidence_interval_percentile": [2.5, 97.5],
        "observed_delta_mean_event_ic": float(paired.delta_mean_event_ic.mean()),
        "bootstrap_ci_lower": float(lower),
        "bootstrap_ci_upper": float(upper),
        "paired_event_count": int(len(paired)),
        "announcement_month_block_count": int(len(months)),
        "paired_event_counts_by_month": {
            str(month): int(len(values)) for month, values in month_values.items()
        },
        "events_with_both_eligible_ic": int(len(paired)),
        "events_with_missing_or_ineligible_ic": int(
            per_event[EVENT_KEY].drop_duplicates().shape[0] - len(paired)
        ),
    }
    return table, summary


def input_paths() -> dict[str, Path]:
    return {
        "model_features": DATA / "model_features.csv",
        "metadata": DATA / "metadata.csv",
        "targets": DATA / "targets.csv",
        "eligibility": DATA / "eligibility.csv",
        "feature_missingness": DATA / "feature_missingness.csv",
        "split_counts": DATA / "split_counts.csv",
        "execution_inputs": DATA / "execution_inputs.csv",
        "model_ready_summary": DATA / "summary.json",
        "config": CONFIG_PATH,
        "runner_code": Path(__file__).resolve(),
        "requirements_ml": ROOT / "requirements-ml.txt",
    }


def load_development_data(
    paths: dict[str, Path],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Filter eligibility IDs first, then read and attach only development targets."""
    target_name = config["target"]
    eligibility = pd.read_csv(paths["eligibility"], low_memory=False)
    if eligibility.sample_id.duplicated().any():
        raise RuntimeError("Eligibility sample_id is not unique")
    eligibility["supervised_model_eligible_bool"] = parse_bool(
        eligibility.supervised_model_eligible, "supervised_model_eligible"
    )
    dev_eligibility = eligibility.loc[
        eligibility.supervised_model_eligible_bool
        & eligibility.split.isin(["training", "validation"]),
        ["sample_id", "split"],
    ].copy()
    dev_ids = set(dev_eligibility.sample_id)
    test_ids = set(eligibility.loc[eligibility.split.eq("test"), "sample_id"])
    if dev_ids.intersection(test_ids):
        raise RuntimeError("Eligibility split overlap between development and test")

    # This is the first target read.  Test values are inspected only to enforce the frozen seal;
    # no test row is included in dev_targets or any subsequent merge.
    targets = pd.read_csv(
        paths["targets"],
        usecols=["sample_id", "split", target_name],
        low_memory=False,
    )
    if targets.sample_id.duplicated().any():
        raise RuntimeError("Target sample_id is not unique")
    # Include both the eligibility-derived test IDs and rows explicitly marked test in the
    # target file, so a split mismatch cannot hide a non-empty test value.
    test_target_rows = targets.loc[
        targets.sample_id.isin(test_ids) | targets.split.astype("string").eq("test")
    ]
    test_target_nonnull = int(test_target_rows[target_name].notna().sum())
    if test_target_nonnull:
        raise RuntimeError("Frozen model-ready run contains a non-empty test target")
    if set(test_target_rows.sample_id) != test_ids:
        raise RuntimeError("Target table does not cover every sealed test sample ID")
    dev_targets = targets.loc[
        targets.sample_id.isin(dev_ids), ["sample_id", target_name]
    ].rename(columns={target_name: "target"})
    if len(dev_targets) != len(dev_ids) or dev_targets.target.isna().any():
        raise RuntimeError("Eligible development IDs do not have complete targets")

    features = pd.read_csv(paths["model_features"], low_memory=False)
    metadata = pd.read_csv(
        paths["metadata"],
        usecols=["sample_id", *EVENT_KEY, "announcement_day", "exit_session"],
        low_memory=False,
    )
    for name, table in [("features", features), ("metadata", metadata)]:
        if table.sample_id.duplicated().any():
            raise RuntimeError(f"{name} sample_id is not unique")
        if set(table.sample_id) != set(eligibility.sample_id):
            raise RuntimeError(f"{name} sample_id set does not match eligibility")
    if set(targets.sample_id) != set(eligibility.sample_id):
        raise RuntimeError("Target sample_id set does not match eligibility")
    data = (
        dev_eligibility.merge(features, on="sample_id", how="left", validate="one_to_one")
        .merge(metadata, on="sample_id", how="left", validate="one_to_one")
        .merge(dev_targets, on="sample_id", how="left", validate="one_to_one")
    )
    if len(data) != len(dev_ids) or data.target.isna().any():
        raise RuntimeError("Development ID joins are incomplete")
    data["announcement_day"] = pd.to_datetime(data.announcement_day, errors="coerce").dt.normalize()
    data["exit_session"] = pd.to_datetime(data.exit_session, errors="coerce").dt.normalize()
    if data["announcement_day"].isna().any() or data["exit_session"].isna().any():
        raise RuntimeError("Development rows contain missing event dates")
    if set(data.split) != {"training", "validation"}:
        raise RuntimeError("Development data contains an unexpected split")
    info = {
        "test_ids": test_ids,
        "test_target_nonnull": test_target_nonnull,
        "target_test_rows": int(len(test_target_rows)),
        "features_rows": int(len(features)),
        "metadata_rows": int(len(metadata)),
        "target_rows": int(len(targets)),
        "eligibility_rows": int(len(eligibility)),
        "dev_id_count": int(len(dev_ids)),
    }
    return data, eligibility, info


def append_data_processing_log(summary: dict[str, Any]) -> None:
    """Record an executed run after its run-specific summary has been written."""
    log_path = ROOT / "DATA_PROCESSING_LOG.md"
    processing = summary["processing_record"]
    text = (
        f"\n\n## {summary['run_id']} — analyst_model_v1\n\n"
        f"- **运行时间**：{processing['run_time_utc']}\n"
        f"- **输入版本与路径**：固定 model-ready run `{MODEL_READY_RUN_ID}`，路径 `{rel(DATA)}`；"
        f"脚本 SHA-256 `{summary['input_hashes']['runner_code']['sha256']}`，"
        f"配置 SHA-256 `{summary['input_hashes']['config']['sha256']}`。\n"
        f"- **规则与目的**：先按 `eligibility.csv` 的 `supervised_model_eligible=true` 和 "
        "`training/validation` split 筛 ID，再合并目标；采用 2018/2019/2020 rolling training-only CV；"
        "事件等权；Ridge 在每个训练折内拟合 median imputer 与 StandardScaler，HGB 仅拟合 median imputer；"
        "无 winsorization、填零、前向填充、插值或目标填补。\n"
        f"- **行与标的计数**：{json.dumps(processing['row_counts'], ensure_ascii=False, sort_keys=True)}\n"
        f"- **受影响行与排除理由**：{json.dumps(processing['affected_rows'], ensure_ascii=False, sort_keys=True)}；"
        "未物理删除非 eligible 行，完整 eligibility 与 reason code 保存在本 run 的 `eligibility_audit.csv`。\n"
        "- **quarantine**：无新增 quarantine；缺失特征保留并只在训练拟合的 median imputer 中处理。\n"
        f"- **检查**：{json.dumps(processing['checks'], ensure_ascii=False, sort_keys=True)}\n"
        f"- **限制**：{json.dumps(processing['limitations'], ensure_ascii=False)}\n"
        f"- **执行状态**：{processing['status']}。\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    started_at = datetime.now(timezone.utc)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["model_ready_run_id"] != MODEL_READY_RUN_ID:
        raise RuntimeError("Config model-ready run ID is not the fixed requested run")
    paths = input_paths()
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    input_hashes = {
        name: {"path": rel(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }

    dev, eligibility, load_info = load_development_data(paths, config)
    all_features = [column for column in pd.read_csv(paths["model_features"], nrows=0).columns if column != "sample_id"]
    if config["target"] in all_features:
        raise RuntimeError("Target column is present in model features")
    forbidden_tokens = ["residual", "trbc", "entry_", "forward_return", "label_complete", "exit_session"]
    if any(any(token in column.lower() for token in forbidden_tokens) for column in all_features):
        raise RuntimeError("Forbidden feature token found in model_features.csv")
    feature_sets = resolve_feature_sets(all_features, config)
    check_feature_sets(feature_sets, all_features)
    for column in all_features:
        raw = dev[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.isna().sum() > raw.isna().sum():
            raise RuntimeError(f"Non-numeric values found in feature {column}")
        dev[column] = numeric
    if np.isinf(dev[all_features].to_numpy(float)).any():
        raise RuntimeError("Infinity found in model features")
    if not bool(dev[all_features].replace([np.inf, -np.inf], np.nan).notna().any(axis=None)):
        raise RuntimeError("All model features are missing")
    event_split_counts = dev.groupby(EVENT_KEY, dropna=False, sort=False)["split"].nunique(dropna=False)
    if (event_split_counts > 1).any():
        raise RuntimeError("An event appears in more than one split")

    eligibility_audit = eligibility.copy()
    eligibility_audit["runner_included"] = eligibility_audit.sample_id.isin(set(dev.sample_id))
    eligibility_audit["runner_exclusion_reason"] = np.where(
        eligibility_audit.runner_included,
        "included_in_training_or_validation",
        eligibility_audit.supervised_ineligibility_reason.fillna("not_training_or_validation_or_not_supervised"),
    )
    excluded_reason_counts = (
        eligibility_audit.loc[~eligibility_audit.runner_included, "runner_exclusion_reason"]
        .value_counts(dropna=False)
        .astype(int)
        .to_dict()
    )

    folds = config["internal_training_folds"]
    grids = candidate_grids(config)
    cv_rows: list[dict[str, Any]] = []
    candidate_summary_rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        for params in grids[name]:
            fold_rows: list[dict[str, Any]] = []
            for fold in folds:
                fold_train, fold_valid = fold_samples(dev, fold)
                bundle = fit_candidate(name, params, fold_train, feature_sets, config["random_state"])
                prediction = predict_candidate(name, bundle, fold_valid, feature_sets)
                result = metrics(fold_valid, prediction)
                row = {
                    **result,
                    "stage": "training_cv",
                    "model": name,
                    "fold": fold["name"],
                    "params": json_text(params),
                    "train_rows": int(len(fold_train)),
                    "train_events": int(fold_train[EVENT_KEY].drop_duplicates().shape[0]),
                    "validation_rows": int(len(fold_valid)),
                    "validation_events": int(fold_valid[EVENT_KEY].drop_duplicates().shape[0]),
                }
                cv_rows.append(row)
                fold_rows.append(row)
            candidate_summary_rows.append(
                {
                    "model": name,
                    "params": json_text(params),
                    "folds": int(len(fold_rows)),
                    "mean_event_spearman": safe_nanmean(
                        [row["mean_event_spearman"] for row in fold_rows]
                    ),
                    "mean_event_weighted_mse": safe_nanmean(
                        [row["event_weighted_mse"] for row in fold_rows]
                    ),
                    "mean_event_weighted_rmse": safe_nanmean(
                        [row["event_weighted_rmse"] for row in fold_rows]
                    ),
                    "non_null_event_spearman_folds": int(
                        sum(np.isfinite(row["mean_event_spearman"]) for row in fold_rows)
                    ),
                }
            )
    cv_summary = pd.DataFrame(candidate_summary_rows)
    selected_params: dict[str, dict[str, Any]] = {}
    selected_cv_rows: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        if name == "training_weighted_mean":
            baseline_row = cv_summary.loc[cv_summary.model.eq(name)].iloc[0]
            selected_params[name] = {}
            selected_cv_rows[name] = baseline_row.to_dict()
            continue
        params, best = select_best_params(cv_summary, name)
        selected_params[name] = params
        selected_cv_rows[name] = best.to_dict()
    family_summary = cv_summary.loc[cv_summary.model.isin(NETWORK_FAMILIES)].copy()
    family_summary = family_summary.loc[
        family_summary.apply(lambda row: json_text(selected_params[row.model]) == row.params, axis=1)
    ].sort_values(
        ["mean_event_spearman", "mean_event_weighted_mse", "model"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if family_summary.empty:
        raise RuntimeError("No network family could be nominated from training CV")
    nominee = str(family_summary.iloc[0].model)

    train = dev.loc[dev.split.eq("training")].copy()
    validation = dev.loc[dev.split.eq("validation")].copy()
    fitted = {
        name: fit_candidate(name, selected_params[name], train, feature_sets, config["random_state"])
        for name in MODEL_NAMES
    }
    comparison_roles = {
        name: ("confirmatory" if name in {nominee, "context_ridge"} else "exploratory")
        for name in MODEL_NAMES
    }
    validation_metrics_rows: list[dict[str, Any]] = []
    predictions = validation[
        ["sample_id", *EVENT_KEY, "announcement_day", "target"]
    ].copy()
    per_event_rows: list[pd.DataFrame] = []
    for name in MODEL_NAMES:
        prediction = predict_candidate(name, fitted[name], validation, feature_sets)
        result = metrics(validation, prediction)
        result.update(
            {
                "model": name,
                "params": json_text(selected_params[name]),
                "comparison_role": comparison_roles[name],
                "cv_nominated_network_family": bool(name == nominee),
            }
        )
        validation_metrics_rows.append(result)
        predictions[f"prediction_{name}"] = prediction
        per_event_rows.append(
            event_ic_table(
                validation,
                prediction,
                model=name,
                split="validation",
                comparison_role=comparison_roles[name],
            )
        )
    validation_metrics = pd.DataFrame(validation_metrics_rows)
    per_event = pd.concat(per_event_rows, ignore_index=True)
    bootstrap_table, bootstrap_summary = bootstrap_delta(
        per_event,
        nominee=nominee,
        context_name="context_ridge",
        seed=config["bootstrap"]["seed"],
        replications=config["bootstrap"]["replications"],
    )

    run_id = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    models_dir = out / "models"
    out.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir(parents=False, exist_ok=False)
    outputs: dict[str, Path] = {
        "cv_results": out / "cv_results.csv",
        "cv_candidate_summary": out / "cv_candidate_summary.csv",
        "validation_metrics": out / "validation_metrics.csv",
        "validation_predictions": out / "validation_predictions.csv",
        "per_event_ic": out / "per_event_ic.csv",
        "bootstrap_delta_mean_event_ic": out / "bootstrap_delta_mean_event_ic.csv",
        "bootstrap_summary": out / "bootstrap_summary.json",
        "selected_hyperparameters": out / "selected_hyperparameters.json",
        "feature_sets": out / "feature_sets.json",
        "eligibility_audit": out / "eligibility_audit.csv",
    }
    pd.DataFrame(cv_rows).to_csv(outputs["cv_results"], index=False)
    cv_summary["selected_for_model"] = cv_summary.apply(
        lambda row: bool(json_text(selected_params[row.model]) == row.params), axis=1
    )
    cv_summary.to_csv(outputs["cv_candidate_summary"], index=False)
    validation_metrics.to_csv(outputs["validation_metrics"], index=False)
    predictions.to_csv(outputs["validation_predictions"], index=False)
    per_event.to_csv(outputs["per_event_ic"], index=False)
    bootstrap_table.to_csv(outputs["bootstrap_delta_mean_event_ic"], index=False)
    outputs["bootstrap_summary"].write_text(
        json.dumps(bootstrap_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    outputs["selected_hyperparameters"].write_text(
        json.dumps(
            {
                "per_model": selected_params,
                "network_family_nominee": nominee,
                "selection_order": config["selection"]["hyperparameter_order"],
                "selected_cv_rows": selected_cv_rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    outputs["feature_sets"].write_text(
        json.dumps(feature_sets, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    eligibility_audit.to_csv(outputs["eligibility_audit"], index=False)

    model_paths: dict[str, Path] = {}
    for name in MODEL_NAMES:
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(fitted[name], model_path)
        model_paths[name] = model_path

    train_event_weights = event_weights(train)
    train_weight_sums = pd.Series(train_event_weights, index=train.index).groupby(
        train[EVENT_KEY].astype("string").agg("|".join, axis=1)
    ).sum()
    expected_rows = config["expected_supervised_model_eligible_rows"]
    checks: dict[str, bool] = {
        "fixed_model_ready_run": config["model_ready_run_id"] == MODEL_READY_RUN_ID,
        "eligibility_filtered_before_target_merge": True,
        "development_has_no_test_rows": not set(dev.sample_id).intersection(load_info["test_ids"]),
        "test_target_values_all_nan": load_info["test_target_nonnull"] == 0,
        "test_target_nonempty_rows_not_used": load_info["test_target_nonnull"] == 0,
        "training_rows_match_frozen_eligibility": len(train) == int(expected_rows["training"]),
        "validation_rows_match_frozen_eligibility": len(validation) == int(expected_rows["validation"]),
        "test_supervised_eligible_count_is_zero": int(
            eligibility.loc[eligibility.split.eq("test"), "supervised_model_eligible_bool"].sum()
        )
        == int(expected_rows["test"]),
        "internal_folds_are_chronological": all(
            pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["validation_start"]) for fold in folds
        ),
        "cv_rows_only_expected_folds": set(pd.DataFrame(cv_rows).fold)
        == {fold["name"] for fold in folds},
        "cv_uses_training_split_only": all(
            row["stage"] == "training_cv" for row in cv_rows
        ),
        "event_weights_sum_to_one": bool(np.allclose(train_weight_sums.to_numpy(float), 1.0)),
        "all_validation_predictions_finite": bool(
            np.isfinite(predictions.filter(like="prediction_").to_numpy(float)).all()
        ),
        "validation_predictions_only_eligible_validation_ids": set(predictions.sample_id)
        == set(validation.sample_id),
        "validation_metrics_cover_declared_models": set(validation_metrics.model) == set(MODEL_NAMES),
        "per_event_ic_cover_declared_models": set(per_event.model) == set(MODEL_NAMES),
        "nominated_family_is_declared": nominee in NETWORK_FAMILIES,
        "confirmatory_comparison_is_nominee_vs_context": comparison_roles[nominee] == "confirmatory"
        and comparison_roles["context_ridge"] == "confirmatory",
        "bootstrap_has_fixed_replication_count": len(bootstrap_table) == int(config["bootstrap"]["replications"]),
        "bootstrap_values_finite": bool(np.isfinite(bootstrap_table.delta_mean_event_ic).all()),
    }
    after_hashes = {
        name: sha256_file(path) for name, path in paths.items()
    }
    checks["input_hashes_unchanged_during_run"] = after_hashes == {
        name: item["sha256"] for name, item in input_hashes.items()
    }
    output_hashes = {name: sha256_file(path) for name, path in outputs.items()}
    model_hashes = {name: sha256_file(path) for name, path in model_paths.items()}
    processing_record = {
        "run_time_utc": started_at.isoformat(),
        "input_versions_and_paths": {
            "model_ready_run_id": MODEL_READY_RUN_ID,
            "model_ready_root": rel(DATA),
            "config": rel(CONFIG_PATH),
            "runner": rel(Path(__file__).resolve()),
        },
        "row_counts": {
            "model_features_input_rows": int(load_info["features_rows"]),
            "metadata_input_rows": int(load_info["metadata_rows"]),
            "targets_input_rows": int(load_info["target_rows"]),
            "eligibility_input_rows": int(load_info["eligibility_rows"]),
            "eligible_training_rows_after_filter": int(len(train)),
            "eligible_validation_rows_after_filter": int(len(validation)),
            "sealed_test_rows": int(len(load_info["test_ids"])),
            "target_test_rows_checked": int(load_info["target_test_rows"]),
            "test_target_nonnull_rows": int(load_info["test_target_nonnull"]),
        },
        "instrument_counts": {
            "input_unique_sample_ids": int(eligibility.sample_id.nunique()),
            "eligible_training_sources": int(train.source.nunique()),
            "eligible_validation_sources": int(validation.source.nunique()),
            "eligible_training_events": int(train[EVENT_KEY].drop_duplicates().shape[0]),
            "eligible_validation_events": int(validation[EVENT_KEY].drop_duplicates().shape[0]),
        },
        "affected_rows": {
            "eligible_development_rows": int(len(dev)),
            "excluded_from_model_due_to_eligibility_or_split": int(len(eligibility) - len(dev)),
            "excluded_rows_retained_in_eligibility_audit": int(len(eligibility) - len(dev)),
            "excluded_by_reason_code": {str(key): int(value) for key, value in excluded_reason_counts.items()},
            "test_rows_predicted": 0,
            "test_rows_scored": 0,
        },
        "missing_value_handling": {
            "target": "no target imputation; eligible development target must be present",
            "features": "median imputation fitted separately within each training fold/model; no physical feature-row deletion",
            "test_target": "seal check only; expected all NaN and not used",
        },
        "deduplication": "no physical deduplication; duplicate sample IDs are a hard failure and event-level aggregation is only for two-stage event targets",
        "outlier_handling": "none",
        "units_and_dates": "returns and existing features are used as stored; announcement_day and exit_session are normalized to calendar dates without timezone conversion",
        "membership_and_snapshot": "membership, entry eligibility and graph snapshot selection are inherited from the fixed model-ready run; no new security exclusion is applied",
        "quarantine_location": "none; excluded rows and reason codes retained in eligibility_audit.csv",
        "checks": checks,
        "limitations": [
            "This runner validates the broad-S&P event-sleeve mechanism and is not the final AI-universe model; a later point-in-time AI-universe validation is required.",
            "The 2021-2022 validation comparison is confirmatory only for the training-CV nominee versus context_ridge; other validation rows are exploratory.",
            "Two-stage event targets use actual training event means by design; validation event means are never used.",
            "Event rows are dependent and announcement-month bootstrap is a block-level uncertainty aid, not an IID significance test.",
            "No test prediction, test metric, target aggregate or model selection uses test-period values.",
            "No portfolio construction, transaction costs, turnover or exposure constraints are applied.",
        ],
        "status": "completed" if all(checks.values()) else "failed_checks",
    }
    summary = {
        "run_id": run_id,
        "model_ready_run_id": MODEL_READY_RUN_ID,
        "scope": "training and validation only; test predictions and metrics were not produced",
        "target": config["target"],
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_events": int(train[EVENT_KEY].drop_duplicates().shape[0]),
        "validation_events": int(validation[EVENT_KEY].drop_duplicates().shape[0]),
        "selected_hyperparameters": selected_params,
        "network_family_nominee": nominee,
        "confirmatory_validation_models": [nominee, "context_ridge"],
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "bootstrap_summary": bootstrap_summary,
        "config": config,
        "feature_sets": feature_sets,
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "model_hashes": model_hashes,
        "checks": checks,
        "processing_record": processing_record,
        "scope_note": "Broad-S&P event-sleeve mechanism validation only; not the final AI-universe model. Later point-in-time AI-universe validation is required.",
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    append_data_processing_log(summary)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "network_family_nominee": nominee,
                "confirmatory_validation_models": [nominee, "context_ridge"],
                "bootstrap_summary": bootstrap_summary,
                "checks": checks,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not all(checks.values()):
        raise RuntimeError(
            "Analyst model checks failed: "
            + ", ".join(name for name, value in checks.items() if not value)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
