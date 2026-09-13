"""Validation-only diagnostics for the primary enhanced AI-pool daily model.

This script deliberately reads only the validation predictions from one model run and
the corresponding development target table.  It does not load fitted model artifacts
and it does not reference the sealed test-target file.  The focal model is
``technical_plus_ai_state/logistic``; technical-only Logistic and Random Forest are
paired comparators.

Outputs are written to a run-scoped directory under
``data/analysis/ai_daily_primary_diagnostics_v1/<model_run_id>/``:

* year_metrics.csv: fixed validation metrics for 2021, 2022, and all dates;
* quarterly_stability.csv: fixed metrics by calendar quarter;
* calibration_bins.csv and calibration_summary.csv: ten fixed-width probability bins
  and expected calibration error (ECE);
* auc_difference_bootstrap_replicates.csv and auc_difference_bootstrap_summary.csv:
  paired date-cluster bootstrap distributions and percentile confidence intervals;
* cross_sectional_by_date.csv and cross_sectional_coverage.csv: date-level rank IC,
  top-minus-bottom spread, and coverage tiers;
* summary.json and summary.md: provenance, checks, methods, results, and limitations;
* diagnose_ai_daily_primary_v1.py: a copy of this exact analysis code.

No threshold is tuned and no model is selected by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_RUN_ID = "20260910T070000000000Z"
MODEL_RUN_FAMILY = "ai_pool_daily_state_models_v1_1"
DEFAULT_MODEL_READY_ID = "20260910T052100000000Z"
OUT_FAMILY = "ai_daily_primary_diagnostics_v1"
FOCAL_MODEL = "technical_plus_ai_state/logistic"
COMPARATOR_MODELS = [
    "technical_only/logistic",
    "technical_only/random_forest",
]
MODEL_COLUMNS = {
    "technical_only/logistic": "p_up_technical_only_logistic",
    "technical_only/random_forest": "p_up_technical_only_random_forest",
    "technical_plus_ai_state/logistic": "p_up_technical_plus_ai_state_logistic",
}
ANALYSIS_VERSION = "1.0.0"
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 5360
SPREAD_FRACTION = 0.20
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 11)
TARGET_USECOLS = [
    "sample_id",
    "Instrument",
    "security_id",
    "formation_session",
    "split",
    "label_complete",
    "forward_excess_return",
    "y",
]
PREDICTION_KEY_COLUMNS = [
    "Instrument",
    "security_id",
    "formation_session",
    "split",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values and NaN to JSON-safe native values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes, list, tuple, dict)) else False:
        return None
    return value


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def safe_spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if valid.sum() < 3:
        return None
    left_valid = left_array[valid]
    right_valid = right_array[valid]
    if np.unique(left_valid).size < 2 or np.unique(right_valid).size < 2:
        return None
    result = spearmanr(left_valid, right_valid).statistic
    return safe_float(result)


def is_true_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().eq("true")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")


def load_validation_frame(model_run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge validation predictions with development targets and reconcile keys."""
    model_run = ROOT / "data" / "model_runs" / MODEL_RUN_FAMILY / model_run_id
    prediction_path = model_run / "validation_predictions.csv"
    run_summary_path = model_run / "summary.json"
    model_ready_id = DEFAULT_MODEL_READY_ID
    target_path = ROOT / "data" / "model_ready_ai_pool_daily_v1" / model_ready_id / "targets_dev.csv"

    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not run_summary_path.is_file():
        raise FileNotFoundError(run_summary_path)
    if not target_path.is_file():
        raise FileNotFoundError(target_path)

    # The explicitly named input is the development target file.  Do not discover or
    # open any other target path here.
    predictions = pd.read_csv(prediction_path, low_memory=False)
    target_dev = pd.read_csv(target_path, usecols=TARGET_USECOLS, low_memory=False)
    run_metadata = json.loads(run_summary_path.read_text(encoding="utf-8"))

    require_columns(predictions, ["sample_id", "split", *MODEL_COLUMNS.values()], "validation_predictions")
    require_columns(target_dev, TARGET_USECOLS, "targets_dev")
    if str(run_metadata.get("run_id")) != model_run_id:
        raise RuntimeError("model run summary run_id does not match requested run")

    pred_validation = predictions.loc[predictions["split"].astype(str).eq("validation")].copy()
    if len(pred_validation) != len(predictions):
        raise RuntimeError("validation_predictions.csv contains a non-validation row")
    if pred_validation["sample_id"].duplicated().any():
        raise RuntimeError("validation prediction sample_id values are not unique")

    target_validation = target_dev.loc[target_dev["split"].astype(str).eq("validation")].copy()
    target_validation = target_validation.loc[is_true_series(target_validation["label_complete"])].copy()
    if target_validation["sample_id"].duplicated().any():
        raise RuntimeError("targets_dev validation sample_id values are not unique")
    target_validation = target_validation.rename(
        columns={column: f"target_{column}" for column in TARGET_USECOLS if column != "sample_id"}
    )

    merged = pred_validation.merge(
        target_validation,
        on="sample_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = int(merged["_merge"].ne("both").sum())
        raise RuntimeError(f"{missing} validation predictions lack a complete development target")

    mismatch_checks: dict[str, bool] = {}
    for column in PREDICTION_KEY_COLUMNS:
        if column == "formation_session":
            left = pd.to_datetime(merged[column], errors="raise")
            right = pd.to_datetime(merged[f"target_{column}"], errors="raise")
            mismatch_checks[column] = bool(left.eq(right).all())
        else:
            mismatch_checks[column] = bool(
                merged[column].astype(str).eq(merged[f"target_{column}"].astype(str)).all()
            )
    mismatch_checks["y"] = bool(
        np.array_equal(
            pd.to_numeric(merged["y"], errors="raise").to_numpy(),
            pd.to_numeric(merged["target_y"], errors="raise").to_numpy(),
        )
    )
    mismatch_checks["forward_excess_return"] = bool(
        np.allclose(
            pd.to_numeric(merged["forward_excess_return"], errors="raise").to_numpy(dtype=float),
            pd.to_numeric(merged["target_forward_excess_return"], errors="raise").to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=False,
        )
    )
    if not all(mismatch_checks.values()):
        raise RuntimeError(f"prediction/target reconciliation failed: {mismatch_checks}")

    canonical = pd.DataFrame({
        "sample_id": merged["sample_id"].astype(str),
        "Instrument": merged["target_Instrument"].astype(str),
        "security_id": merged["target_security_id"].astype(str),
        "formation_session": pd.to_datetime(merged["target_formation_session"], errors="raise"),
        "split": merged["target_split"].astype(str),
        "y": pd.to_numeric(merged["target_y"], errors="raise").astype(int),
        "forward_excess_return": pd.to_numeric(
            merged["target_forward_excess_return"], errors="raise"
        ).astype(float),
    })
    for model_column in MODEL_COLUMNS.values():
        canonical[model_column] = pd.to_numeric(merged[model_column], errors="raise").astype(float)

    if canonical["split"].ne("validation").any():
        raise RuntimeError("canonical frame contains a non-validation split")
    if not canonical["y"].isin([0, 1]).all():
        raise RuntimeError("development validation labels are not binary")
    if not np.isfinite(canonical["forward_excess_return"]).all():
        raise RuntimeError("development validation forward excess returns contain non-finite values")
    for model_column in MODEL_COLUMNS.values():
        values = canonical[model_column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise RuntimeError(f"probability column is not finite and bounded: {model_column}")

    reconciliation = {
        "prediction_rows": int(len(predictions)),
        "validation_rows": int(len(canonical)),
        "target_dev_rows_loaded": int(len(target_dev)),
        "target_dev_validation_complete_rows": int(len(target_validation)),
        "prediction_target_key_checks": mismatch_checks,
        "prediction_target_keys_one_to_one": True,
        "prediction_target_values_reconciled": True,
        "predictions_all_validation": True,
        "targets_source": rel(target_path),
        "predictions_source": rel(prediction_path),
        "model_run_summary_source": rel(run_summary_path),
    }
    return canonical.sort_values(["formation_session", "Instrument"], kind="stable").reset_index(drop=True), {
        "model_run": model_run,
        "prediction_path": prediction_path,
        "target_path": target_path,
        "run_summary_path": run_summary_path,
        "run_metadata": run_metadata,
        "reconciliation": reconciliation,
    }


def auc_or_none(y: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None) -> float | None:
    if sample_weight is None:
        valid = np.isfinite(probability) & np.isfinite(y)
        y_use = y[valid]
        p_use = probability[valid]
    else:
        valid = (sample_weight > 0) & np.isfinite(probability) & np.isfinite(y)
        y_use = y[valid]
        p_use = probability[valid]
        sample_weight = sample_weight[valid]
    if np.unique(y_use).size < 2:
        return None
    value = roc_auc_score(y_use, p_use, sample_weight=sample_weight)
    return safe_float(value)


def date_cross_section(group: pd.DataFrame, model_column: str) -> dict[str, Any]:
    """Return date-level rank IC and fixed 20% top-minus-bottom spread."""
    ordered = group.sort_values([model_column, "Instrument"], ascending=[False, True], kind="stable")
    n = int(len(ordered))
    side_n = max(1, int(math.floor(n * SPREAD_FRACTION)))
    rank_ic = safe_spearman(ordered[model_column], ordered["forward_excess_return"])
    spread_runner = safe_float(
        ordered.head(side_n)["forward_excess_return"].mean()
        - ordered.tail(side_n)["forward_excess_return"].mean()
    )
    return {
        "row_count": n,
        "instrument_count": int(ordered["Instrument"].nunique()),
        "side_n_20pct": side_n,
        "rank_ic": rank_ic,
        "rank_ic_usable": bool(rank_ic is not None),
        # The runner's formula gives a number even for n=1.  Keep it for exact
        # comparability, while separate coverage fields identify meaningful tiers.
        "spread_runner": spread_runner,
        "spread_n_ge_2": spread_runner if n >= 2 else None,
        "spread_n_ge_3": spread_runner if n >= 3 else None,
        "spread_n_ge_10_per_side": spread_runner if side_n >= 10 else None,
    }


def cross_sectional_table(frame: pd.DataFrame, model_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for formation_session, group in frame.groupby("formation_session", sort=True):
        base = {
            "formation_session": pd.Timestamp(formation_session).strftime("%Y-%m-%d"),
            "year": int(pd.Timestamp(formation_session).year),
            "quarter": str(pd.Timestamp(formation_session).to_period("Q")),
        }
        for model_key in model_keys:
            stats = date_cross_section(group, MODEL_COLUMNS[model_key])
            suffix = model_key.replace("/", "__")
            for key, value in stats.items():
                base[f"{suffix}__{key}"] = value
        rows.append(base)
    table = pd.DataFrame(rows)

    coverage_rows: list[dict[str, Any]] = []
    total_dates = int(len(table))
    total_rows = int(len(frame))
    for model_key in model_keys:
        suffix = model_key.replace("/", "__")
        rank_usable = table[f"{suffix}__rank_ic_usable"].astype(bool)
        spread_runner = table[f"{suffix}__spread_runner"].notna()
        spread_ge2 = table[f"{suffix}__spread_n_ge_2"].notna()
        spread_ge3 = table[f"{suffix}__spread_n_ge_3"].notna()
        spread_ge10 = table[f"{suffix}__spread_n_ge_10_per_side"].notna()
        rank_ic_values = pd.to_numeric(table.loc[rank_usable, f"{suffix}__rank_ic"], errors="coerce")
        spread_ge2_values = pd.to_numeric(table.loc[spread_ge2, f"{suffix}__spread_n_ge_2"], errors="coerce")
        spread_ge3_values = pd.to_numeric(table.loc[spread_ge3, f"{suffix}__spread_n_ge_3"], errors="coerce")
        spread_runner_values = pd.to_numeric(table.loc[spread_runner, f"{suffix}__spread_runner"], errors="coerce")
        coverage_rows.append({
            "model_key": model_key,
            "total_validation_rows": total_rows,
            "total_formation_dates": total_dates,
            "rank_ic_usable_dates": int(rank_usable.sum()),
            "rank_ic_date_coverage": float(rank_usable.mean()) if total_dates else None,
            "rank_ic_usable_rows": int(table.loc[rank_usable, f"{suffix}__row_count"].sum()),
            "spread_runner_dates": int(spread_runner.sum()),
            "spread_n_ge_2_dates": int(spread_ge2.sum()),
            "spread_n_ge_2_date_coverage": float(spread_ge2.mean()) if total_dates else None,
            "spread_n_ge_2_rows": int(table.loc[spread_ge2, f"{suffix}__row_count"].sum()),
            "spread_n_ge_3_dates": int(spread_ge3.sum()),
            "spread_n_ge_3_date_coverage": float(spread_ge3.mean()) if total_dates else None,
            "spread_n_ge_10_per_side_dates": int(spread_ge10.sum()),
            "spread_n_ge_10_per_side_date_coverage": float(spread_ge10.mean()) if total_dates else None,
            "mean_rank_ic_usable_dates": safe_float(rank_ic_values.mean()) if not rank_ic_values.empty else None,
            "mean_spread_runner_all_dates": safe_float(spread_runner_values.mean()) if not spread_runner_values.empty else None,
            "mean_spread_n_ge_2_dates": safe_float(spread_ge2_values.mean()) if not spread_ge2_values.empty else None,
            "mean_spread_n_ge_3_dates": safe_float(spread_ge3_values.mean()) if not spread_ge3_values.empty else None,
        })
    coverage = pd.DataFrame(coverage_rows)
    return table, coverage


def metric_record(frame: pd.DataFrame, model_key: str, period: str) -> dict[str, Any]:
    model_column = MODEL_COLUMNS[model_key]
    y = frame["y"].to_numpy(dtype=int)
    probability = frame[model_column].to_numpy(dtype=float)
    cs_table, _ = cross_sectional_table(frame, [model_key])
    suffix = model_key.replace("/", "__")
    rank_values = pd.to_numeric(cs_table.loc[cs_table[f"{suffix}__rank_ic_usable"], f"{suffix}__rank_ic"], errors="coerce")
    spread_values = pd.to_numeric(cs_table[f"{suffix}__spread_n_ge_2"], errors="coerce")
    return {
        "period": period,
        "model_key": model_key,
        "rows": int(len(frame)),
        "instruments": int(frame["Instrument"].nunique()),
        "formation_dates": int(frame["formation_session"].nunique()),
        "positive_label_rate": safe_float(y.mean()),
        "roc_auc": auc_or_none(y, probability),
        "pr_auc": safe_float(average_precision_score(y, probability)) if np.unique(y).size > 1 else None,
        "brier": safe_float(brier_score_loss(y, probability)),
        "fixed_directional_accuracy_at_0_5": safe_float(np.mean((probability >= 0.5).astype(int) == y)),
        "pooled_rank_ic_forward_excess_return": safe_spearman(probability, frame["forward_excess_return"]),
        "mean_daily_rank_ic": safe_float(rank_values.mean()) if not rank_values.empty else None,
        "rank_ic_usable_dates": int(rank_values.notna().sum()),
        "mean_daily_spread_n_ge_2": safe_float(spread_values.mean()) if spread_values.notna().any() else None,
        "spread_n_ge_2_dates": int(spread_values.notna().sum()),
    }


def build_metrics(frame: pd.DataFrame, model_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    for year in [2021, 2022]:
        periods.append((str(year), frame.loc[frame["formation_session"].dt.year.eq(year)].copy()))
    year_rows = [metric_record(subset, model_key, period) for period, subset in periods for model_key in model_keys]

    quarter_rows: list[dict[str, Any]] = []
    quarters = frame["formation_session"].dt.to_period("Q").astype(str)
    for quarter, subset in frame.groupby(quarters, sort=True):
        for model_key in model_keys:
            row = metric_record(subset, model_key, str(quarter))
            row["quarter"] = str(quarter)
            quarter_rows.append(row)
    return pd.DataFrame(year_rows), pd.DataFrame(quarter_rows)


def calibration_for_scope(frame: pd.DataFrame, model_key: str, scope: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_column = MODEL_COLUMNS[model_key]
    probabilities = frame[model_column].to_numpy(dtype=float)
    labels = frame["y"].to_numpy(dtype=int)
    bin_index = np.searchsorted(CALIBRATION_EDGES, probabilities, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(CALIBRATION_EDGES) - 2)
    rows: list[dict[str, Any]] = []
    contributions: list[float] = []
    for index in range(len(CALIBRATION_EDGES) - 1):
        mask = bin_index == index
        count = int(mask.sum())
        mean_probability = safe_float(probabilities[mask].mean()) if count else None
        observed_rate = safe_float(labels[mask].mean()) if count else None
        absolute_gap = abs(mean_probability - observed_rate) if count else None
        contribution = (count / len(frame)) * absolute_gap if count and absolute_gap is not None else 0.0
        contributions.append(float(contribution))
        rows.append({
            "scope": scope,
            "model_key": model_key,
            "bin_index": index,
            "bin_lower_inclusive": float(CALIBRATION_EDGES[index]),
            "bin_upper_exclusive": float(CALIBRATION_EDGES[index + 1]),
            "n": count,
            "mean_predicted_probability": mean_probability,
            "observed_positive_rate": observed_rate,
            "absolute_gap": absolute_gap,
            "ece_contribution": float(contribution),
        })
    nonempty = [row for row in rows if row["n"] > 0]
    ece = float(sum(contributions))
    mce = max((float(row["absolute_gap"]) for row in nonempty if row["absolute_gap"] is not None), default=None)
    summary = {
        "scope": scope,
        "model_key": model_key,
        "rows": int(len(frame)),
        "nonempty_bins": int(len(nonempty)),
        "ece": ece,
        "mce": mce,
        "brier": safe_float(brier_score_loss(labels, probabilities)),
        "binning": "10 fixed-width bins on [0,1]; [0.9,1.0] includes p=1",
    }
    return rows, summary


def build_calibration(frame: pd.DataFrame, model_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    for year in [2021, 2022]:
        scopes.append((str(year), frame.loc[frame["formation_session"].dt.year.eq(year)].copy()))
    bins: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scope, subset in scopes:
        for model_key in model_keys:
            bin_rows, summary = calibration_for_scope(subset, model_key, scope)
            bins.extend(bin_rows)
            summaries.append(summary)
    return pd.DataFrame(bins), pd.DataFrame(summaries)


def date_cluster_bootstrap(
    frame: pd.DataFrame,
    comparator_key: str,
    scope: str,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Paired bootstrap of dates, preserving every row within a sampled date."""
    focal_column = MODEL_COLUMNS[FOCAL_MODEL]
    comparator_column = MODEL_COLUMNS[comparator_key]
    ordered = frame.sort_values(["formation_session", "Instrument"], kind="stable").reset_index(drop=True)
    cluster_codes, unique_dates = pd.factorize(ordered["formation_session"], sort=True)
    cluster_codes = np.asarray(cluster_codes, dtype=int)
    n_clusters = int(len(unique_dates))
    if n_clusters < 2:
        raise RuntimeError(f"date-cluster bootstrap requires at least 2 dates for {scope}")
    y = ordered["y"].to_numpy(dtype=int)
    focal_probability = ordered[focal_column].to_numpy(dtype=float)
    comparator_probability = ordered[comparator_column].to_numpy(dtype=float)
    cluster_sizes = np.bincount(cluster_codes, minlength=n_clusters)
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, Any]] = []
    valid_differences: list[float] = []
    for replicate in range(reps):
        sampled_clusters = rng.integers(0, n_clusters, size=n_clusters)
        multiplicity = np.bincount(sampled_clusters, minlength=n_clusters).astype(float)
        row_weights = multiplicity[cluster_codes]
        focal_auc = auc_or_none(y, focal_probability, row_weights)
        comparator_auc = auc_or_none(y, comparator_probability, row_weights)
        difference = (
            focal_auc - comparator_auc
            if focal_auc is not None and comparator_auc is not None
            else None
        )
        if difference is not None:
            valid_differences.append(float(difference))
        replicate_rows.append({
            "scope": scope,
            "comparator_model_key": comparator_key,
            "replicate": replicate,
            "auc_focal": focal_auc,
            "auc_comparator": comparator_auc,
            "auc_difference": difference,
        })

    point_focal = auc_or_none(y, focal_probability)
    point_comparator = auc_or_none(y, comparator_probability)
    point_difference = point_focal - point_comparator if point_focal is not None and point_comparator is not None else None
    values = np.asarray(valid_differences, dtype=float)
    if values.size:
        ci_low, ci_high = np.quantile(values, [0.025, 0.975])
        bootstrap_mean = values.mean()
        positive_probability = np.mean(values > 0)
    else:
        ci_low = ci_high = bootstrap_mean = positive_probability = None
    summary = {
        "scope": scope,
        "focal_model_key": FOCAL_MODEL,
        "comparator_model_key": comparator_key,
        "rows": int(len(ordered)),
        "date_clusters": n_clusters,
        "point_auc_focal": point_focal,
        "point_auc_comparator": point_comparator,
        "point_auc_difference": point_difference,
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_valid": int(values.size),
        "ci_level": 0.95,
        "ci_method": "percentile",
        "ci_low": safe_float(ci_low),
        "ci_high": safe_float(ci_high),
        "bootstrap_mean_difference": safe_float(bootstrap_mean),
        "probability_difference_gt_zero": safe_float(positive_probability),
        "cluster_sampling": "sample formation_session clusters with replacement; retain all rows within each sampled date",
        "seed": int(seed),
    }
    return pd.DataFrame(replicate_rows), summary


def build_bootstrap(frame: pd.DataFrame, model_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", frame)]
    for year in [2021, 2022]:
        scopes.append((str(year), frame.loc[frame["formation_session"].dt.year.eq(year)].copy()))
    for scope_index, (scope, subset) in enumerate(scopes):
        for comparator_index, comparator_key in enumerate(COMPARATOR_MODELS):
            seed = BOOTSTRAP_SEED + scope_index * 100 + comparator_index
            reps, summary = date_cluster_bootstrap(subset, comparator_key, scope, seed)
            replicate_tables.append(reps)
            summaries.append(summary)
    return pd.concat(replicate_tables, ignore_index=True), pd.DataFrame(summaries)


def compact_dict_from_row(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    rows = frame.loc[mask]
    if rows.empty:
        return {}
    return {key: json_safe(value) for key, value in rows.iloc[0].to_dict().items()}


def markdown_summary(
    summary: dict[str, Any],
    year_metrics: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    focal_year = year_metrics.loc[
        year_metrics["model_key"].eq(FOCAL_MODEL) & year_metrics["period"].isin(["2021", "2022"])
    ].copy()
    focal_year = focal_year.set_index("period")
    lines = [
        f"# AI daily primary diagnostics — {summary['model_run_id']}",
        "",
        "This is a validation-only diagnostic of the pre-specified `technical_plus_ai_state/logistic` model. "
        "It uses the run's validation predictions reconciled to `targets_dev.csv`; no threshold was tuned, "
        "no model was selected, and no sealed test target was accessed.",
        "",
        f"Analysis run: `{summary['analysis_run_id']}`; analysis version: `{ANALYSIS_VERSION}`.",
        f"Validation sample: {summary['counts']['validation_rows']} rows across "
        f"{summary['counts']['validation_dates']} formation dates and "
        f"{summary['counts']['validation_instruments']} instruments; years covered: 2021–2022.",
        "",
        "## Year metrics for the focal model",
        "",
        "| Period | Rows | Dates | ROC AUC | PR AUC | Brier | Mean daily rank IC | Rank-IC dates | Mean spread (n≥2) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period in ["2021", "2022"]:
        row = focal_year.loc[period]
        values = [
            period,
            int(row["rows"]),
            int(row["formation_dates"]),
            f"{row['roc_auc']:.4f}",
            f"{row['pr_auc']:.4f}",
            f"{row['brier']:.4f}",
            f"{row['mean_daily_rank_ic']:.4f}" if pd.notna(row["mean_daily_rank_ic"]) else "NA",
            int(row["rank_ic_usable_dates"]),
            f"{row['mean_daily_spread_n_ge_2']:.4f}" if pd.notna(row["mean_daily_spread_n_ge_2"]) else "NA",
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")

    lines.extend([
        "",
        "## Date-cluster AUC difference",
        "",
        "The interval is a paired percentile bootstrap over formation dates. Each sampled date contributes all of its rows, so rows within a date are not treated as independent clusters.",
        "",
        "| Scope | Comparator | Point difference | 95% CI | P(diff > 0) | Date clusters |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in bootstrap_summary.iterrows():
        lines.append(
            f"| {row['scope']} | {row['comparator_model_key']} | {row['point_auc_difference']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['probability_difference_gt_zero']:.3f} | "
            f"{int(row['date_clusters'])} |"
        )

    focal_coverage = coverage.loc[coverage["model_key"].eq(FOCAL_MODEL)].iloc[0]
    lines.extend([
        "",
        "## Cross-sectional coverage",
        "",
        f"There are {int(focal_coverage['rank_ic_usable_dates'])} dates with usable within-date rank IC out of "
        f"{int(focal_coverage['total_formation_dates'])} ({focal_coverage['rank_ic_date_coverage']:.1%}). "
        f"The runner's fixed 20% spread formula returns a value on {int(focal_coverage['spread_runner_dates'])} dates, "
        f"but only {int(focal_coverage['spread_n_ge_2_dates'])} dates have at least two securities and "
        f"{int(focal_coverage['spread_n_ge_10_per_side_dates'])} meet a ten-per-side gate. "
        "The 23 rank-usable dates are the only dates that support a genuine within-date rank comparison; "
        "single-security dates are retained in audit tables and are not treated as independent cross-sectional evidence.",
        "",
        "## Limitations",
        "",
        "* The validation period has only 23 dates with usable within-date ranks; quarterly and annual stability estimates therefore reflect a small number of independent time clusters.",
        "* Most validation dates contain one security because the primary H21 non-overlap anchor sample is sparse; the fixed top/bottom 20% spread is descriptive and does not meet a ten-per-side economic portfolio gate.",
        "* These are development comparisons using validation labels. They do not establish out-of-sample test performance or justify a new threshold/model choice.",
        "* The upstream model-ready run uses an exploratory static candidate registry with provisional static-source membership for many instruments; the PIT limitation from the model run remains applicable.",
        "",
        "See the CSV tables in this directory for the complete period, quarter, calibration, bootstrap, and date-level records.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-id", default=DEFAULT_MODEL_RUN_ID)
    args = parser.parse_args()
    started_at = utc_now()
    frame, sources = load_validation_frame(args.model_run_id)
    model_keys = [FOCAL_MODEL, *COMPARATOR_MODELS]
    year_metrics, quarterly = build_metrics(frame, model_keys)
    calibration_bins, calibration_summary = build_calibration(frame, model_keys)
    cross_date, cross_coverage = cross_sectional_table(frame, model_keys)
    bootstrap_reps, bootstrap_summary = build_bootstrap(frame, model_keys)

    analysis_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "data" / "analysis" / OUT_FAMILY / args.model_run_id
    out.mkdir(parents=True, exist_ok=True)
    year_metrics.to_csv(out / "year_metrics.csv", index=False)
    quarterly.to_csv(out / "quarterly_stability.csv", index=False)
    calibration_bins.to_csv(out / "calibration_bins.csv", index=False)
    calibration_summary.to_csv(out / "calibration_summary.csv", index=False)
    cross_date.to_csv(out / "cross_sectional_by_date.csv", index=False)
    cross_coverage.to_csv(out / "cross_sectional_coverage.csv", index=False)
    bootstrap_reps.to_csv(out / "auc_difference_bootstrap_replicates.csv", index=False)
    bootstrap_summary.to_csv(out / "auc_difference_bootstrap_summary.csv", index=False)
    shutil.copy2(Path(__file__), out / Path(__file__).name)

    validation_date_counts = frame.groupby("formation_session", sort=True).size()
    focal_cov = cross_coverage.loc[cross_coverage["model_key"].eq(FOCAL_MODEL)].iloc[0]
    focal_all = year_metrics.loc[year_metrics["model_key"].eq(FOCAL_MODEL) & year_metrics["period"].eq("all")].iloc[0]
    summary: dict[str, Any] = {
        "schema_version": "ai_daily_primary_diagnostics_v1",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_run_id": analysis_run_id,
        "generated_at_utc": utc_now(),
        "status": "complete_validation_diagnostics_only",
        "model_run_id": args.model_run_id,
        "model_run_family": MODEL_RUN_FAMILY,
        "focal_model": FOCAL_MODEL,
        "comparators": COMPARATOR_MODELS,
        "inputs": {
            "predictions": {
                "path": rel(sources["prediction_path"]),
                "sha256": sha256_file(sources["prediction_path"]),
            },
            "targets_dev": {
                "path": rel(sources["target_path"]),
                "sha256": sha256_file(sources["target_path"]),
            },
            "model_run_summary": {
                "path": rel(sources["run_summary_path"]),
                "sha256": sha256_file(sources["run_summary_path"]),
            },
            "sealed_test_targets": {
                "accessed": False,
                "purpose": "This diagnostic has no sealed-test input path and performs no test-period scoring.",
            },
        },
        "counts": {
            "target_dev_rows_loaded": sources["reconciliation"]["target_dev_rows_loaded"],
            "validation_rows": int(len(frame)),
            "validation_dates": int(frame["formation_session"].nunique()),
            "validation_instruments": int(frame["Instrument"].nunique()),
            "validation_years": sorted(frame["formation_session"].dt.year.unique().astype(int).tolist()),
            "date_row_count_distribution": {str(int(key)): int(value) for key, value in validation_date_counts.value_counts().sort_index().items()},
            "rank_usable_dates_focal": int(focal_cov["rank_ic_usable_dates"]),
            "rank_usable_date_coverage_focal": float(focal_cov["rank_ic_date_coverage"]),
            "spread_n_ge_2_dates_focal": int(focal_cov["spread_n_ge_2_dates"]),
            "spread_n_ge_10_per_side_dates_focal": int(focal_cov["spread_n_ge_10_per_side_dates"]),
            "bootstrap_reps_requested_per_scope_comparator": BOOTSTRAP_REPS,
        },
        "reconciliation": sources["reconciliation"],
        "methods": {
            "year_periods": ["2021", "2022", "all"],
            "quarter_periods": sorted(quarterly["quarter"].unique().tolist()),
            "metrics": [
                "ROC AUC",
                "average precision PR AUC",
                "Brier score",
                "fixed directional accuracy at p=0.5 (descriptive only)",
                "pooled Spearman rank IC against forward_excess_return",
                "mean date-level Spearman rank IC",
                "fixed 20% top-minus-bottom forward excess return spread",
            ],
            "calibration": {
                "bins": 10,
                "edges": CALIBRATION_EDGES.tolist(),
                "ece": "sum_bin(n_bin / N * abs(mean_probability - observed_rate))",
                "mce": "maximum absolute bin gap among non-empty bins",
            },
            "bootstrap": {
                "reps": BOOTSTRAP_REPS,
                "seed_base": BOOTSTRAP_SEED,
                "unit": "formation_session date cluster",
                "paired_comparison": True,
                "ci": "95% percentile interval of focal AUC minus comparator AUC",
                "no_threshold_tuning": True,
                "no_model_selection": True,
            },
            "cross_sectional_coverage": {
                "rank_ic_usable_rule": "at least 3 rows and non-constant score and forward_excess_return within date",
                "spread_rule": "sort by fixed model probability; top/bottom max(1, floor(0.20 * n)) rows",
                "spread_coverage_tiers": [
                    "runner formula (including n=1 for exact comparability)",
                    "n>=2 securities",
                    "n>=3 securities",
                    "at least 10 securities per side (side_n>=10)",
                ],
            },
        },
        "key_results": {
            "focal_all_metrics": {key: json_safe(value) for key, value in focal_all.to_dict().items()},
            "focal_year_metrics": {
                str(period): {key: json_safe(value) for key, value in year_metrics.loc[
                    year_metrics["model_key"].eq(FOCAL_MODEL) & year_metrics["period"].eq(str(period))
                ].iloc[0].to_dict().items()}
                for period in [2021, 2022]
            },
            "auc_difference_bootstrap": [
                {key: json_safe(value) for key, value in row.items()}
                for row in bootstrap_summary.to_dict(orient="records")
            ],
            "focal_cross_sectional_coverage": {key: json_safe(value) for key, value in focal_cov.to_dict().items()},
        },
        "checks": {
            "validation_predictions_only": True,
            "development_targets_only": True,
            "sealed_test_targets_not_accessed": True,
            "prediction_target_keys_one_to_one": True,
            "prediction_target_values_reconciled": True,
            "probabilities_finite_and_bounded": True,
            "binary_development_labels": True,
            "bootstrap_replicates_valid": bool((bootstrap_summary["bootstrap_reps_valid"] > 0).all()),
            "cross_sectional_date_table_written": True,
            "no_threshold_tuning": True,
            "no_model_selection": True,
            "model_artifacts_not_loaded_or_modified": True,
        },
        "limitations": [
            "Only 23 of 89 validation formation dates have at least three rows and non-constant values sufficient for a within-date rank IC; dates with one or two rows remain in audit tables.",
            "The validation anchor schedule is sparse: 60 dates have one row, six dates have two rows, and 23 dates have 40 rows. The nominal runner spread is therefore not an economic portfolio test.",
            "No validation date reaches ten securities per top/bottom side under the fixed 20% spread rule.",
            "The date-cluster bootstrap has only 45 date clusters in 2021 and 44 in 2022, so confidence intervals are indicative and sensitive to the realized date sample.",
            "All results are development validation comparisons. Sealed test targets were not opened, scored, ranked, or predicted.",
            "The upstream model-ready run uses exploratory static candidate membership; its point-in-time limitation remains applicable to interpretation.",
        ],
        "outputs": {
            "directory": rel(out),
            # summary.json and summary.md are written immediately after this
            # object is assembled, so include their names explicitly.
            "files": sorted(set(
                [path.name for path in out.iterdir() if path.is_file()]
                + ["summary.json", "summary.md"]
            )),
        },
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
    }
    (out / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "summary.md").write_text(markdown_summary(summary, year_metrics, bootstrap_summary, cross_coverage), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "analysis_run_id": analysis_run_id,
        "output_directory": rel(out),
        "validation_rows": len(frame),
        "validation_dates": int(frame["formation_session"].nunique()),
        "focal_roc_auc": focal_all["roc_auc"],
        "focal_rank_ic_usable_dates": int(focal_cov["rank_ic_usable_dates"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
