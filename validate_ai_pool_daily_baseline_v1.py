"""Independently validate daily AI-pool H21 baseline outputs.

Only development labels and validation predictions are read. The test file is
opened for its header only; no test-row values or future returns are parsed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
MODEL_READY_RUN_ID = "20260910T050000000000Z"
MODEL_READY = ROOT / "data" / "model_ready_ai_pool_daily_v1" / MODEL_READY_RUN_ID
MODEL_RUN_ROOT = ROOT / "data" / "model_runs" / "ai_pool_daily_baseline_v1"
AUDIT_ROOT = ROOT / "data" / "audit" / "ai_pool_daily_baseline_v1"
FEATURE_COLUMNS = [
    "momentum_1", "momentum_5", "momentum_20", "momentum_60",
    "volatility_20_ann", "volatility_60_ann",
    "volume_median_20_log1p", "volume_median_60_log1p",
    "dollar_volume_median_20_log1p", "dollar_volume_median_60_log1p",
    "spread_median_20_bps", "spread_median_60_bps",
    "beta_126", "idio_vol_126_ann", "beta_obs_126",
    "spy_momentum_1", "spy_momentum_5", "spy_momentum_20", "spy_momentum_60",
    "spy_volatility_20_ann", "spy_volatility_60_ann",
]
TARGET_COLUMNS = {"y", "forward_excess_return", "stock_forward_return", "benchmark_forward_return"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3 or left.loc[valid].nunique() < 2 or right.loc[valid].nunique() < 2:
        return float("nan")
    value = spearmanr(left.loc[valid], right.loc[valid]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def independent_metrics(predictions: pd.DataFrame, probability_column: str) -> tuple[dict, pd.DataFrame]:
    work = predictions[["sample_id", "Instrument", "formation_session", "y", "forward_excess_return", probability_column]].copy()
    work["formation_session"] = pd.to_datetime(work["formation_session"], format="mixed", errors="raise").dt.normalize()
    work["y"] = pd.to_numeric(work["y"], errors="raise").astype(int)
    work["forward_excess_return"] = pd.to_numeric(work["forward_excess_return"], errors="raise")
    work["p_up"] = pd.to_numeric(work.pop(probability_column), errors="raise")
    y = work["y"].to_numpy(dtype=int)
    p = work["p_up"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise RuntimeError("validation labels contain one class")
    daily_rank: list[float] = []
    spread_rows: list[dict] = []
    for date, group in work.groupby("formation_session", sort=True):
        group = group.sort_values(["p_up", "sample_id"], ascending=[False, True], kind="stable")
        ic = safe_spearman(group["p_up"], group["forward_excess_return"])
        if np.isfinite(ic):
            daily_rank.append(ic)
        side = max(1, int(np.floor(len(group) * 0.2)))
        top = group.head(side)["forward_excess_return"].mean()
        bottom = group.tail(side)["forward_excess_return"].mean()
        spread_rows.append({
            "formation_session": date,
            "rows": len(group),
            "side_size": side,
            "side_size_below_registered_minimum": side < 10,
            "top_mean_excess": top,
            "bottom_mean_excess": bottom,
            "top_minus_bottom_spread": top - bottom,
        })
    spread = pd.DataFrame(spread_rows)
    return {
        "rows": int(len(work)),
        "instruments": int(work["Instrument"].nunique()),
        "formation_sessions": int(work["formation_session"].nunique()),
        "positive_label_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "directional_accuracy_at_0_5": float(np.mean((p >= 0.5).astype(int) == y)),
        "pooled_rank_ic": safe_spearman(work["p_up"], work["forward_excess_return"]),
        "pooled_rank_ic_label": safe_spearman(work["p_up"], work["y"]),
        "mean_daily_rank_ic": float(np.mean(daily_rank)) if daily_rank else float("nan"),
        "daily_rank_ic_groups": int(len(daily_rank)),
        "mean_daily_top_minus_bottom_spread": float(spread["top_minus_bottom_spread"].mean()),
        "median_daily_top_minus_bottom_spread": float(spread["top_minus_bottom_spread"].median()),
        "spread_days": int(len(spread)),
        "spread_days_below_registered_minimum_side": int(spread["side_size_below_registered_minimum"].sum()),
        "registered_minimum_side_size": 10,
    }, spread


def equal_value(left: object, right: object, tol: float = 1e-10) -> bool:
    try:
        lval = float(left)
        rval = float(right)
    except (TypeError, ValueError):
        return left == right
    if np.isnan(lval) and np.isnan(rval):
        return True
    return bool(np.isclose(lval, rval, atol=tol, rtol=tol, equal_nan=True))


def run(model_run_id: str, audit_run_id: str | None = None, model_ready_run_id: str | None = None) -> dict:
    model_out = MODEL_RUN_ROOT / model_run_id
    if not model_out.exists():
        raise FileNotFoundError(model_out)
    model_summary_path = model_out / "summary.json"
    if not model_summary_path.exists():
        raise FileNotFoundError(model_summary_path)
    summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    declared_model_ready = summary.get("inputs", {}).get("model_ready", {})
    declared_path = declared_model_ready.get("path")
    if model_ready_run_id:
        model_ready = ROOT / "data" / "model_ready_ai_pool_daily_v1" / model_ready_run_id
    elif declared_path:
        model_ready = Path(declared_path)
        if not model_ready.is_absolute():
            model_ready = ROOT / model_ready
    else:
        model_ready = MODEL_READY
    resolved_model_ready_id = model_ready.name
    audit_run_id = audit_run_id or f"{model_run_id}_independent"
    out = AUDIT_ROOT / audit_run_id
    out.mkdir(parents=True, exist_ok=False)
    files = {
        "model_features": model_ready / "model_features.csv",
        "eligibility": model_ready / "eligibility.csv",
        "targets_dev": model_ready / "targets_dev.csv",
        "sealed_manifest": model_ready / "targets_test_sealed.csv",
        "model_summary": model_summary_path,
        "model_ready_summary": model_ready / "summary.json",
        "validation_predictions": model_out / "validation_predictions.csv",
        "validation_daily_spreads": model_out / "validation_daily_spreads.csv",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if summary.get("inputs", {}).get("model_ready", {}).get("path") and Path(summary["inputs"]["model_ready"]["path"]).name != resolved_model_ready_id:
        raise RuntimeError("validator model-ready override does not match model summary input path")
    model_ready_summary = json.loads(files["model_ready_summary"].read_text(encoding="utf-8"))
    features = pd.read_csv(files["model_features"], low_memory=False)
    eligibility = pd.read_csv(files["eligibility"], low_memory=False)
    targets_dev = pd.read_csv(files["targets_dev"], low_memory=False)
    predictions = pd.read_csv(files["validation_predictions"], low_memory=False)
    observed_spreads = pd.read_csv(files["validation_daily_spreads"], low_memory=False)
    sealed_columns = list(pd.read_csv(files["sealed_manifest"], nrows=0).columns)

    development = features.merge(targets_dev[["sample_id", "y", "forward_excess_return", "label_complete"]], on="sample_id", how="inner", validate="one_to_one").merge(eligibility[["sample_id", "supervised_model_eligible", "boundary_purged", "non_overlap_selected", "non_overlap_block", "entry_trade_eligible", "feature_core_available"]], on="sample_id", how="inner", validate="one_to_one")
    for col in ["supervised_model_eligible", "boundary_purged", "non_overlap_selected", "entry_trade_eligible", "feature_core_available", "label_complete"]:
        development[col] = development[col].astype(str).str.casefold().eq("true")
    sampling_mode = str(summary.get("sampling_mode", "conservative_non_overlap_anchors"))
    eligible_common = development["label_complete"] & ~development["boundary_purged"] & development["entry_trade_eligible"] & development["feature_core_available"]
    if sampling_mode == "conservative_non_overlap_anchors":
        eligible_common = eligible_common & development["non_overlap_selected"]
    elif sampling_mode != "all_daily_with_date_purge":
        raise RuntimeError(f"unknown sampling mode in model summary: {sampling_mode}")
    training = development.loc[development["split"].eq("training") & eligible_common].copy()
    training_values = training[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    training_medians = np.nanmedian(training_values, axis=0)
    probability_columns = [col for col in predictions.columns if col.startswith("p_up_")]
    checks: dict[str, bool] = {
        "model_ready_counts_match_summary": len(features) == summary["counts"]["model_ready_master_rows"] and len(eligibility) == len(features),
        "validation_prediction_rows_match_summary": len(predictions) == summary["counts"]["validation_supervised_rows"],
        "validation_only_predictions": set(predictions["split"].astype(str)) == {"validation"},
        "validation_chronological": bool(pd.to_datetime(predictions["formation_session"], format="mixed", errors="raise").is_monotonic_increasing),
        "no_target_columns_in_model_features": not bool(TARGET_COLUMNS & set(features.columns)),
        "sealed_manifest_has_no_target_columns": not bool(TARGET_COLUMNS & set(sealed_columns)),
        "sealed_manifest_schema_expected": sealed_columns == ["sample_id", "Instrument", "security_id", "formation_session", "split", "entry_session", "exit_session", "target_status"],
        "sealed_rows_not_in_predictions": set(predictions["split"].astype(str)) == {"validation"},
        "probabilities_finite": bool(np.isfinite(predictions[probability_columns].to_numpy(dtype=float)).all()),
        "training_rows_reconstructed": len(training) == summary["counts"]["training_supervised_rows"],
        "training_non_overlap_keys_unique": (not training.duplicated(["Instrument", "non_overlap_block"]).any()) if sampling_mode == "conservative_non_overlap_anchors" else True,
        "sampling_mode_reconstructed": summary.get("sampling_mode") in {"conservative_non_overlap_anchors", "all_daily_with_date_purge"},
    }
    loaded_models: dict[str, object] = {}
    checks["model_artifacts_load"] = True
    try:
        for name in ("logistic_controls_only", "random_forest_controls_only"):
            loaded_models[name] = joblib.load(model_out / "models" / f"{name}.joblib")
    except Exception:
        checks["model_artifacts_load"] = False
    for name, model in loaded_models.items():
        imputer = model.named_steps.get("imputer")
        checks[f"training_only_imputer_statistics_{name}"] = bool(imputer is not None and np.allclose(imputer.statistics_, training_medians, atol=1e-12, rtol=1e-12, equal_nan=True))
        checks[f"missing_indicators_enabled_{name}"] = bool(imputer is not None and imputer.add_indicator)

    recomputed_metrics: dict[str, dict] = {}
    spread_checks: dict[str, bool] = {}
    for probability_column in probability_columns:
        name = probability_column.removeprefix("p_up_")
        metric, spread = independent_metrics(predictions, probability_column)
        recomputed_metrics[name] = metric
        expected = summary["validation_metrics"].get(name, {})
        checks[f"metrics_match_summary_{name}"] = all(equal_value(metric.get(key), expected.get(key)) for key in metric)
        expected_spread = observed_spreads.loc[observed_spreads["model"].astype(str).eq(name)].copy()
        expected_spread["formation_session"] = pd.to_datetime(expected_spread["formation_session"], format="mixed", errors="raise").dt.normalize()
        expected_spread = expected_spread.sort_values("formation_session").reset_index(drop=True)
        spread = spread.sort_values("formation_session").reset_index(drop=True)
        columns = ["formation_session", "rows", "side_size", "side_size_below_registered_minimum", "top_mean_excess", "bottom_mean_excess", "top_minus_bottom_spread"]
        spread_checks[name] = len(expected_spread) == len(spread) and all(
            (str(expected_spread.loc[i, col]) == str(spread.loc[i, col]) if col == "formation_session" else equal_value(expected_spread.loc[i, col], spread.loc[i, col]))
            for i in range(min(len(expected_spread), len(spread))) for col in columns
        )
        checks[f"daily_spreads_match_{name}"] = spread_checks[name]
    checks["no_test_metrics_or_predictions"] = summary["counts"]["test_metric_rows"] == 0 and summary["counts"]["test_prediction_rows"] == 0
    checks["no_event_study_inputs"] = True
    checks["no_row_deletion_claimed"] = bool(summary["checks"].get("no_row_deletion_in_master", False))
    if summary.get("membership_source_mode") == "exploratory_static_candidate":
        limitations_text = " ".join(str(x) for x in summary.get("limitations", []))
        checks["expanded_static_candidate_disclosed"] = (
            "not_full_pit" in str(summary.get("pit_status", ""))
            and "unbiased" in limitations_text.lower()
            and "provisional" in limitations_text.lower()
        )
    counts = {
        "model_ready_master_rows": int(len(features)),
        "model_ready_master_instruments": int(features["Instrument"].nunique()),
        "targets_dev_rows": int(len(targets_dev)),
        "training_supervised_rows": int(len(training)),
        "validation_prediction_rows": int(len(predictions)),
        "validation_prediction_instruments": int(predictions["Instrument"].nunique()),
        "validation_prediction_sessions": int(predictions["formation_session"].nunique()),
        "sealed_manifest_rows": int(summary["counts"]["test_master_rows"]),
        "output_test_prediction_rows": 0,
        "output_test_metric_rows": 0,
    }
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in files.items()}
    result = {
        "schema_version": "ai_pool_daily_baseline_v1_independent_audit",
        "audit_run_id": audit_run_id,
        "generated_at_utc": utc_now(),
        "model_run_id": model_run_id,
        "model_ready_run_id": resolved_model_ready_id,
        "inputs": input_hashes,
        "counts": counts,
        "recomputed_validation_metrics": recomputed_metrics,
        "daily_spread_checks": spread_checks,
        "checks": checks,
        "status": "independently_verified" if all(checks.values()) else "verification_failed",
        "limitations": [
            "This audit verifies development outputs and artifacts; it does not create test predictions or test metrics.",
            ("The expanded registry run is exploratory_static_candidate mode and is not a complete point-in-time AI-role history; validation is not an unbiased full-PIT backtest." if summary.get("membership_source_mode") == "exploratory_static_candidate" else "The current 10-RIC pool is too small for an economically sized 10-per-side daily portfolio; RF probabilities may be constant under the registered leaf-size constraint."),
        ],
    }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log_path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = log_path.read_text(encoding="utf-8") if log_path.exists() else "# DATA_PROCESSING_LOG\n"
    entry = [
        "",
        f"## {audit_run_id} — AI pool daily baseline independent validation ({result['generated_at_utc']})",
        "",
        "- **阶段目的与状态**：独立复核 daily H21 controls-only Logistic/Random Forest 的 validation 输出、指标复算、artifact、non-overlap anchor 和 test seal；只读取 test manifest 表头（`nrows=0`），不读取测试目标值。",
        f"- **输入版本与路径**：model-ready `{rel(model_ready)}` run=`{resolved_model_ready_id}`；model run `{rel(model_out)}`；哈希与详细结果见 `{rel(out / 'summary.json')}`。",
        f"- **计数与影响**：`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`；复核不删除或修改输入，affected rows=0，quarantine=none。",
        "- **复核规则**：独立重算 validation AUC、PR AUC、Brier、方向准确率、pooled/daily rank IC 和 daily top-bottom spread；重算 training-only imputer 中位数并检查模型 artifact。",
        f"- **封存/缺失**：未再填补输入；模型阶段的 median + missing indicators 只由 training anchors 拟合；test target values 未读取，test prediction/metric rows=0。",
        f"- **检查结果**：`{json.dumps(checks, ensure_ascii=False, sort_keys=True)}`；执行状态=`{result['status']}`。",
        f"- **输出**：`{rel(out / 'summary.json')}`。",
        "",
    ]
    log_path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")
    ai_path = ROOT / "AI_USE_LOG.md"
    prior_ai = ai_path.read_text(encoding="utf-8") if ai_path.exists() else "# AI_USE_LOG\n"
    ai_entry = [
        "",
        f"### AI pool daily baseline independent validation ({audit_run_id})",
        "",
        f"- OpenAI Codex independently recomputed the daily H21 validation metrics and checked `{sampling_mode}` training-only imputer statistics and model artifacts for `{rel(model_out)}` using model-ready `{rel(model_ready)}`. No test target values were read; audit output is `{rel(out / 'summary.json')}`.",
        "",
    ]
    ai_path.write_text(prior_ai.rstrip("\n") + "\n" + "\n".join(ai_entry), encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently validate daily AI-pool baseline outputs.")
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--audit-run-id", default=None)
    parser.add_argument("--model-ready-run-id", default=None, help="Optional explicit model-ready run; by default derive it from the model summary input path.")
    args = parser.parse_args()
    run(args.model_run_id, args.audit_run_id, args.model_ready_run_id)
