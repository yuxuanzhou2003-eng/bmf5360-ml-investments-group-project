"""Independently validate the AI-pool controls-only baseline outputs.

The validator reads development labels and validation predictions. It reads only
the sealed test manifest header (nrows=0), never any test-row values or target.
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
MODEL_READY_RUN_ID = "20260910T031900000000Z"
MODEL_READY = ROOT / "data" / "model_ready_ai_pool_v1" / MODEL_READY_RUN_ID
MODEL_RUN_ROOT = ROOT / "data" / "model_runs" / "ai_pool_baseline_v1"
AUDIT_ROOT = ROOT / "data" / "audit" / "ai_pool_baseline_v1"
TARGET_COLUMNS = {"y", "forward_excess_return", "stock_forward_return", "benchmark_forward_return"}
FEATURE_COLUMNS = [
    "size_log_market_cap", "momentum_1", "momentum_5", "momentum_20", "momentum_60", "momentum_252",
    "volatility_20_ann", "volatility_60_ann", "beta_126", "idio_vol_126_ann", "beta_obs_126",
    "volume_median_20_log1p", "dollar_volume_median_20_log1p", "spread_median_20_bps",
    "spy_momentum_1", "spy_momentum_5", "spy_momentum_20", "spy_momentum_60", "spy_volatility_20_ann", "spy_volatility_60_ann",
    "book_to_market", "profitability_gross_profit_to_assets", "investment_asset_growth", "leverage_debt_to_assets",
    "operating_cash_flow_to_assets", "revenue_growth",
]


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
    monthly_rank = []
    spread_rows = []
    for date, group in work.groupby("formation_session", sort=True):
        group = group.sort_values(["p_up", "sample_id"], ascending=[False, True], kind="stable")
        ic = safe_spearman(group["p_up"], group["forward_excess_return"])
        if np.isfinite(ic):
            monthly_rank.append(ic)
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
    y = work["y"].to_numpy(dtype=int)
    p = work["p_up"].to_numpy(dtype=float)
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
        "mean_monthly_rank_ic": float(np.mean(monthly_rank)) if monthly_rank else float("nan"),
        "monthly_rank_ic_groups": int(len(monthly_rank)),
        "mean_monthly_top_minus_bottom_spread": float(spread["top_minus_bottom_spread"].mean()),
        "median_monthly_top_minus_bottom_spread": float(spread["top_minus_bottom_spread"].median()),
        "spread_months": int(len(spread)),
        "spread_months_below_registered_minimum_side": int(spread["side_size_below_registered_minimum"].sum()),
        "registered_minimum_side_size": 10,
    }, spread


def values_equal(left: object, right: object, tol: float = 1e-10) -> bool:
    try:
        l = float(left)
        r = float(right)
    except (TypeError, ValueError):
        return left == right
    if np.isnan(l) and np.isnan(r):
        return True
    return bool(np.isclose(l, r, rtol=tol, atol=tol, equal_nan=True))


def run(model_run_id: str, audit_run_id: str | None = None) -> dict:
    model_out = MODEL_RUN_ROOT / model_run_id
    if not model_out.exists():
        raise FileNotFoundError(model_out)
    audit_run_id = audit_run_id or f"{model_run_id}_independent"
    out = AUDIT_ROOT / audit_run_id
    out.mkdir(parents=True, exist_ok=False)
    files = {
        "model_features": MODEL_READY / "model_features.csv",
        "eligibility": MODEL_READY / "eligibility.csv",
        "targets_dev": MODEL_READY / "targets_dev.csv",
        "sealed_manifest": MODEL_READY / "targets_test_sealed.csv",
        "model_summary": model_out / "summary.json",
        "validation_predictions": model_out / "validation_predictions.csv",
        "validation_monthly_spreads": model_out / "validation_monthly_spreads.csv",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    summary = json.loads(files["model_summary"].read_text(encoding="utf-8"))
    features = pd.read_csv(files["model_features"], low_memory=False)
    eligibility = pd.read_csv(files["eligibility"], low_memory=False)
    targets_dev = pd.read_csv(files["targets_dev"], low_memory=False)
    predictions = pd.read_csv(files["validation_predictions"], low_memory=False)
    observed_spreads = pd.read_csv(files["validation_monthly_spreads"], low_memory=False)
    sealed_columns = list(pd.read_csv(files["sealed_manifest"], nrows=0).columns)

    development = features.merge(
        targets_dev[["sample_id", "y", "forward_excess_return", "label_complete"]],
        on="sample_id", how="inner", validate="one_to_one",
    ).merge(
        eligibility[["sample_id", "supervised_model_eligible", "boundary_purged"]],
        on="sample_id", how="inner", validate="one_to_one",
    )
    development["supervised_model_eligible"] = development["supervised_model_eligible"].astype(str).str.casefold().eq("true")
    development["boundary_purged"] = development["boundary_purged"].astype(str).str.casefold().eq("true")
    development["label_complete"] = development["label_complete"].astype(str).str.casefold().eq("true")
    training = development.loc[
        development["split"].eq("training")
        & development["supervised_model_eligible"]
        & development["label_complete"]
        & ~development["boundary_purged"]
    ].copy()
    training_values = training[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    training_medians = np.nanmedian(training_values, axis=0)

    probability_columns = [col for col in predictions.columns if col.startswith("p_up_")]
    checks: dict[str, bool] = {}
    checks["model_ready_counts_match_summary"] = (
        len(features) == summary["counts"]["model_ready_master_rows"]
        and len(eligibility) == len(features)
        and len(targets_dev) == summary["counts"]["training_master_rows"] + summary["counts"]["validation_master_rows"] - 0
    )
    checks["validation_prediction_rows_match_summary"] = len(predictions) == summary["counts"]["validation_supervised_rows"]
    checks["validation_only_predictions"] = set(predictions["split"].astype(str)) == {"validation"}
    checks["no_target_columns_in_model_features"] = not bool(TARGET_COLUMNS & set(features.columns))
    checks["sealed_manifest_has_no_target_columns"] = not bool(TARGET_COLUMNS & set(sealed_columns))
    checks["sealed_manifest_schema_expected"] = sealed_columns == ["sample_id", "Instrument", "security_id", "formation_session", "split", "entry_session", "exit_session", "target_status"]
    checks["sealed_rows_not_in_predictions"] = set(predictions["split"].astype(str)) == {"validation"}
    checks["probabilities_finite"] = bool(np.isfinite(predictions[probability_columns].to_numpy(dtype=float)).all())
    checks["model_artifacts_load"] = True
    try:
        loaded_models = {}
        for name in ("logistic_controls_only", "random_forest_controls_only"):
            loaded_models[name] = joblib.load(model_out / "models" / f"{name}.joblib")
    except Exception:
        checks["model_artifacts_load"] = False
        loaded_models = {}
    checks["training_rows_reconstructed"] = len(training) == summary["counts"]["training_supervised_rows"]
    for name, model in loaded_models.items():
        imputer = model.named_steps.get("imputer")
        checks[f"training_only_imputer_statistics_{name}"] = bool(
            imputer is not None
            and np.allclose(imputer.statistics_, training_medians, rtol=1e-12, atol=1e-12, equal_nan=True)
        )
        checks[f"missing_indicators_enabled_{name}"] = bool(imputer is not None and imputer.add_indicator)

    recomputed_metrics: dict[str, dict] = {}
    spread_checks: dict[str, bool] = {}
    for probability_column in probability_columns:
        name = probability_column.removeprefix("p_up_")
        metrics, spread = independent_metrics(predictions, probability_column)
        recomputed_metrics[name] = metrics
        expected = summary["validation_metrics"].get(name, {})
        checks[f"metrics_match_summary_{name}"] = all(values_equal(metrics.get(key), expected.get(key)) for key in metrics)
        expected_spread = observed_spreads.loc[observed_spreads["model"].astype(str).eq(name)].copy()
        expected_spread["formation_session"] = pd.to_datetime(expected_spread["formation_session"], format="mixed", errors="raise").dt.normalize()
        expected_spread = expected_spread.sort_values("formation_session").reset_index(drop=True)
        spread = spread.sort_values("formation_session").reset_index(drop=True)
        compare_columns = ["formation_session", "rows", "side_size", "side_size_below_registered_minimum", "top_mean_excess", "bottom_mean_excess", "top_minus_bottom_spread"]
        spread_checks[name] = len(expected_spread) == len(spread) and all(
            (str(expected_spread.loc[i, col]) == str(spread.loc[i, col]) if col == "formation_session" else values_equal(expected_spread.loc[i, col], spread.loc[i, col]))
            for i in range(min(len(expected_spread), len(spread))) for col in compare_columns
        )
        checks[f"monthly_spreads_match_{name}"] = spread_checks[name]

    checks["no_test_metrics_or_predictions"] = summary["counts"]["test_metric_rows"] == 0 and summary["counts"]["test_prediction_rows"] == 0
    checks["no_event_study_inputs"] = not any("event" in str(path).casefold() for path in files.values())
    checks["no_row_deletion_claimed"] = bool(summary["checks"].get("no_row_deletion_in_master", False))
    counts = {
        "model_ready_master_rows": int(len(features)),
        "model_ready_master_instruments": int(features["Instrument"].nunique()),
        "targets_dev_rows": int(len(targets_dev)),
        "validation_prediction_rows": int(len(predictions)),
        "validation_prediction_instruments": int(predictions["Instrument"].nunique()),
        "validation_prediction_sessions": int(predictions["formation_session"].nunique()),
        "sealed_manifest_rows": int(summary["counts"].get("test_master_rows", summary["counts"].get("test_target_sealed_rows", 0))),
        "output_test_prediction_rows": 0,
        "output_test_metric_rows": 0,
    }
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in files.items()}
    result = {
        "schema_version": "ai_pool_baseline_v1_independent_audit",
        "audit_run_id": audit_run_id,
        "generated_at_utc": utc_now(),
        "model_run_id": model_run_id,
        "inputs": input_hashes,
        "counts": counts,
        "recomputed_validation_metrics": recomputed_metrics,
        "monthly_spread_checks": spread_checks,
        "checks": checks,
        "status": "independently_verified" if all(checks.values()) else "verification_failed",
        "limitations": [
            "This audit verifies generated development outputs and artifact reproducibility; it does not create test predictions or test metrics.",
            "The upstream fundamental and market-cap semantics remain UNVERIFIED, and validation cross-sections are too small for the registered 10-per-side portfolio gate.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    log_path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = log_path.read_text(encoding="utf-8") if log_path.exists() else "# DATA_PROCESSING_LOG\n"
    entry = [
        "",
        f"## {audit_run_id} — AI pool baseline independent validation ({result['generated_at_utc']})",
        "",
        "- **阶段目的与状态**：独立复核 controls-only Logistic/Random Forest 的 validation 输出、指标复算、模型 artifact 可加载性和 test seal；只读取 test manifest 表头（`nrows=0`），不读取测试目标值。",
        f"- **输入版本与路径**：model-ready `{rel(MODEL_READY)}`；model run `{rel(model_out)}`；输入哈希见 `{rel(out / 'summary.json')}`。",
        f"- **样本/公司与前后计数**：`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`；复核不删除、不修改输入行，affected rows=0，quarantine=none。",
        "- **复核规则**：独立重算 validation AUC、PR AUC、Brier、方向准确率、pooled/monthly rank IC 及每月概率 top-bottom spread；比较 summary 数值、预测 split、封存清单 schema 与 artifact 加载结果。",
        f"- **缺失/封存处理**：未对输入数据再填补或重算；只确认模型阶段记录的 training-only median + missing indicators；test target values 未读取，test prediction/metric rows=0。",
        f"- **检查结果**：`{json.dumps(checks, ensure_ascii=False, sort_keys=True)}`；执行状态=`{result['status']}`。",
        f"- **输出**：`{rel(out / 'summary.json')}`；状态为独立验证记录，不替代模型 run summary。",
        "",
    ]
    log_path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")

    ai_path = ROOT / "AI_USE_LOG.md"
    prior_ai = ai_path.read_text(encoding="utf-8") if ai_path.exists() else "# AI_USE_LOG\n"
    ai_entry = [
        "",
        f"### AI pool baseline independent validation ({audit_run_id})",
        "",
        f"- OpenAI Codex independently recomputed development validation metrics and checked the model artifacts for model run `{rel(model_out)}`. No test target values were read; audit output is `{rel(out / 'summary.json')}`.",
        "",
    ]
    ai_path.write_text(prior_ai.rstrip("\n") + "\n" + "\n".join(ai_entry), encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independently validate AI pool baseline outputs.")
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--audit-run-id", default=None)
    args = parser.parse_args()
    run(args.model_run_id, args.audit_run_id)
