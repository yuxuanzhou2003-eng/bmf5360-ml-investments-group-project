"""Fit controls-only Logistic and Random Forest on AI-pool model-ready data.

Only training and validation labels are loaded. The sealed test file is not
opened for row values and no test prediction/metric is produced.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL_READY_RUN_ID = "20260910T031900000000Z"
MODEL_READY = ROOT / "data" / "model_ready_ai_pool_v1" / MODEL_READY_RUN_ID
CONFIG_PATH = ROOT / "ai_pool_baseline_v1_config.json"
OUT_ROOT = ROOT / "data" / "model_runs" / "ai_pool_baseline_v1"
FEATURE_COLUMNS = [
    "size_log_market_cap", "momentum_1", "momentum_5", "momentum_20", "momentum_60", "momentum_252",
    "volatility_20_ann", "volatility_60_ann", "beta_126", "idio_vol_126_ann", "beta_obs_126",
    "volume_median_20_log1p", "dollar_volume_median_20_log1p", "spread_median_20_bps",
    "spy_momentum_1", "spy_momentum_5", "spy_momentum_20", "spy_momentum_60", "spy_volatility_20_ann", "spy_volatility_60_ann",
    "book_to_market", "profitability_gross_profit_to_assets", "investment_asset_growth", "leverage_debt_to_assets", "operating_cash_flow_to_assets", "revenue_growth",
]


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
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3 or left.loc[valid].nunique() < 2 or right.loc[valid].nunique() < 2:
        return float("nan")
    value = spearmanr(left.loc[valid], right.loc[valid]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def validation_metrics(frame: pd.DataFrame, probability: np.ndarray, config: dict) -> tuple[dict, pd.DataFrame]:
    work = frame[["sample_id", "Instrument", "formation_session", "y", "forward_excess_return"]].copy()
    work["p_up"] = probability
    y = work["y"].to_numpy(dtype=int)
    p = work["p_up"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise RuntimeError("validation labels contain one class")
    monthly_rank = []
    spread_rows = []
    for date, group in work.groupby("formation_session", sort=True):
        group = group.sort_values(["p_up", "sample_id"], ascending=[False, True], kind="stable")
        ic = safe_spearman(group["p_up"], group["forward_excess_return"])
        if np.isfinite(ic):
            monthly_rank.append(ic)
        n_side = max(1, int(np.floor(len(group) * float(config["portfolio_diagnostic"]["side_fraction"]))))
        top = group.head(n_side)["forward_excess_return"].mean()
        bottom = group.tail(n_side)["forward_excess_return"].mean()
        spread_rows.append({"formation_session": date, "rows": len(group), "side_size": n_side, "side_size_below_registered_minimum": n_side < int(config["portfolio_diagnostic"]["minimum_side_size_registered"]), "top_mean_excess": top, "bottom_mean_excess": bottom, "top_minus_bottom_spread": top - bottom})
    spread = pd.DataFrame(spread_rows)
    metrics = {
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
        "registered_minimum_side_size": int(config["portfolio_diagnostic"]["minimum_side_size_registered"]),
    }
    return metrics, spread


def append_log(summary: dict, out: Path) -> None:
    path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# DATA_PROCESSING_LOG\n"
    entry = [
        "",
        f"## {summary['run_id']} — AI pool controls-only baseline models v1 ({summary['generated_at_utc']})",
        "",
        "- **阶段目的与状态**：在 AI pool monthly model-ready 层上拟合 controls-only Logistic 与固定参数 Random Forest；只加载 training/validation 开发标签并一次性评分 validation，test target/预测保持封存。",
        f"- **输入版本与路径**：model-ready `{summary['inputs']['model_ready']['path']}` run=`{MODEL_READY_RUN_ID}`；features、eligibility、targets_dev 与 config/builder/runner 的 SHA-256 记录于 summary；未读取 targets_test_sealed 的目标值。",
        f"- **样本与时间切分**：`{json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}`；training 只用于拟合，validation 只用于一次评估；formation 月按完整月份分组，未随机拆分。",
        f"- **缺失处理**：`{summary['preprocessing']}`；拟合范围为 training rows，未填零、前后填充、插值、winsorize 或删除 master 行。",
        f"- **模型与指标**：`{json.dumps(summary['validation_metrics'], ensure_ascii=False, sort_keys=True)}`；top/bottom 使用 validation 每月概率排序的 20%，但每月 side size 可能低于注册的 10 只门槛，故仅为诊断。",
        f"- **test seal**：`{summary['test_policy']}`；`test_prediction_rows=0`、`test_metric_rows=0`、`test_target_values_read=false`。",
        f"- **检查/限制**：`{json.dumps(summary['checks'], ensure_ascii=False, sort_keys=True)}`；基本面/市值控制在上游仍为 UNVERIFIED，结果不得解释为正式基本面控制 gate 或完成回测。",
        f"- **输出**：`{rel(out)}`；validation predictions、monthly spread diagnostics、模型 artifact 与 summary 均为 run-scoped；执行状态=`{summary['status']}`。",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def append_ai_use_log(summary: dict, out: Path) -> None:
    path = ROOT / "AI_USE_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# AI_USE_LOG\n"
    entry = [
        "",
        f"### AI pool controls-only baseline models ({summary['run_id']})",
        "",
        f"- OpenAI Codex fit controls-only Logistic and fixed-parameter Random Forest using `{rel(MODEL_READY / 'model_features.csv')}` plus training/validation labels from `{rel(MODEL_READY / 'targets_dev.csv')}`. Training rows={summary['counts']['training_supervised_rows']}; validation rows={summary['counts']['validation_supervised_rows']}; no event study input was used.",
        f"- Preprocessing was fit on training rows only with median imputation and missing indicators; Logistic additionally used StandardScaler. Validation metrics and probability rankings are in `{rel(out)}`. Test target rows were not opened, and test predictions/metrics are zero by construction.",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def run(run_id: str | None = None) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema_version") != "ai_pool_baseline_v1":
        raise RuntimeError("unexpected baseline config schema")
    features_path = MODEL_READY / "model_features.csv"
    eligibility_path = MODEL_READY / "eligibility.csv"
    targets_dev_path = MODEL_READY / "targets_dev.csv"
    sealed_path = MODEL_READY / "targets_test_sealed.csv"
    input_paths = {"model_features": features_path, "eligibility": eligibility_path, "targets_dev": targets_dev_path, "targets_test_sealed_manifest": sealed_path, "model_ready_summary": MODEL_READY / "summary.json", "config": CONFIG_PATH, "runner": Path(__file__).resolve()}
    missing = [name for name, path in input_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    print("阶段说明（AI pool model fitting）：仅读取 model_features、eligibility 与 training/validation targets_dev；按 training-only 规则拟合预处理和两类模型，validation 一次性评分；不读取 test target 值、不生成 test prediction。", flush=True)
    features = pd.read_csv(features_path, low_memory=False)
    eligibility = pd.read_csv(eligibility_path, low_memory=False)
    targets_dev = pd.read_csv(targets_dev_path, low_memory=False)
    for frame, name in [(features, "features"), (eligibility, "eligibility"), (targets_dev, "targets_dev")]:
        if frame["sample_id"].duplicated().any() or frame["sample_id"].isna().any():
            raise RuntimeError(f"{name} sample_id invalid")
    if set(features.sample_id) != set(eligibility.sample_id):
        raise RuntimeError("feature/eligibility sample sets differ")
    if not set(targets_dev.split.astype(str)).issubset({"training", "validation"}):
        raise RuntimeError("targets_dev contains test/out-of-window rows")
    if set(targets_dev.sample_id) - set(features.sample_id):
        raise RuntimeError("targets_dev has unknown sample ids")
    data = features.merge(targets_dev[["sample_id", "y", "forward_excess_return", "label_complete"]], on="sample_id", how="inner", validate="one_to_one")
    data = data.merge(eligibility[["sample_id", "split", "supervised_model_eligible", "boundary_purged"]], on="sample_id", how="inner", validate="one_to_one", suffixes=("", "_elig"))
    data["supervised_model_eligible"] = data["supervised_model_eligible"].astype(str).str.casefold().eq("true")
    data["boundary_purged"] = data["boundary_purged"].astype(str).str.casefold().eq("true")
    data["label_complete"] = data["label_complete"].astype(str).str.casefold().eq("true")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["forward_excess_return"] = pd.to_numeric(data["forward_excess_return"], errors="coerce")
    data["formation_session"] = pd.to_datetime(data["formation_session"], format="mixed", errors="raise").dt.normalize()
    data = data.loc[data["split"].isin(["training", "validation"])].copy()
    train = data.loc[data["split"].eq("training") & data["supervised_model_eligible"] & data["label_complete"] & ~data["boundary_purged"]].copy()
    validation = data.loc[data["split"].eq("validation") & data["supervised_model_eligible"] & data["label_complete"] & ~data["boundary_purged"]].copy()
    if train.empty or validation.empty:
        raise RuntimeError("empty training or validation sample")
    if train["y"].isna().any() or validation["y"].isna().any() or train["forward_excess_return"].isna().any() or validation["forward_excess_return"].isna().any():
        raise RuntimeError("eligible development rows have incomplete labels")
    if train["y"].nunique() < 2 or validation["y"].nunique() < 2:
        raise RuntimeError("training/validation labels require two classes")
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            raise RuntimeError(f"missing control feature {col}")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    x_train, y_train = train[FEATURE_COLUMNS], train["y"].astype(int)
    x_validation, y_validation = validation[FEATURE_COLUMNS], validation["y"].astype(int)
    logistic = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler()), ("classifier", LogisticRegression(C=float(config["logistic"]["C"]), class_weight=config["logistic"]["class_weight"], max_iter=int(config["logistic"]["max_iter"]), solver=config["logistic"]["solver"], random_state=int(config["logistic"]["random_state"])))])
    random_forest = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("classifier", RandomForestClassifier(n_estimators=int(config["random_forest"]["n_estimators"]), max_depth=int(config["random_forest"]["max_depth"]), min_samples_leaf=int(config["random_forest"]["min_samples_leaf"]), max_features=config["random_forest"]["max_features"], class_weight=config["random_forest"]["class_weight"], random_state=int(config["random_forest"]["random_state"]), n_jobs=int(config["random_forest"]["n_jobs"])))])
    fitted = {}
    metrics = {}
    spread_tables = {}
    for name, model in [("logistic_controls_only", logistic), ("random_forest_controls_only", random_forest)]:
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_validation)[:, 1]
        if not np.isfinite(probability).all():
            raise RuntimeError(f"{name} produced nonfinite validation probabilities")
        metric, spread = validation_metrics(validation.assign(y=y_validation.to_numpy()), probability, config)
        metrics[name] = metric
        spread_tables[name] = spread.assign(model=name)
        fitted[name] = model
    validation_predictions = validation[["sample_id", "security_id", "Instrument", "formation_session", "split", "y", "forward_excess_return"]].copy()
    for name, model in fitted.items():
        validation_predictions[f"p_up_{name}"] = model.predict_proba(x_validation)[:, 1]
    monthly_spreads = pd.concat(list(spread_tables.values()), ignore_index=True)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not run_id or any(ch not in "0123456789TZ" for ch in run_id):
        raise ValueError("run_id must contain only digits, T, and Z")
    out = OUT_ROOT / run_id
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    validation_predictions.to_csv(out / "validation_predictions.csv", index=False)
    monthly_spreads.to_csv(out / "validation_monthly_spreads.csv", index=False)
    model_paths = {}
    for name, model in fitted.items():
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_paths[name] = rel(model_path)
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in input_paths.items()}
    output_paths = [out / "validation_predictions.csv", out / "validation_monthly_spreads.csv", *[models_dir / f"{name}.joblib" for name in fitted]]
    counts = {
        "model_ready_master_rows": int(len(features)),
        "model_ready_master_instruments": int(features["Instrument"].nunique()),
        "training_master_rows": int((features["split"] == "training").sum()),
        "validation_master_rows": int((features["split"] == "validation").sum()),
        "test_master_rows": int((features["split"] == "test").sum()),
        "training_supervised_rows": int(len(train)),
        "validation_supervised_rows": int(len(validation)),
        "test_prediction_rows": 0,
        "test_metric_rows": 0,
        "test_target_values_read": False,
    }
    checks = {
        "features_exclude_targets": not (set(FEATURE_COLUMNS) & {"y", "forward_excess_return", "stock_forward_return", "benchmark_forward_return"}),
        "training_only_fit": True,
        "validation_predictions_only": bool(set(validation_predictions["split"]) == {"validation"}),
        "test_prediction_rows_zero": not validation_predictions["split"].eq("test").any(),
        "test_metric_rows_zero": True,
        "test_target_values_read": False,
        "test_sealed_manifest_hashed_only": True,
        "probabilities_finite": bool(np.isfinite(validation_predictions.filter(like="p_up_").to_numpy(dtype=float)).all()),
        "no_event_study_inputs": True,
        "no_row_deletion_in_master": True,
    }
    summary = {
        "schema_version": "ai_pool_baseline_v1",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": "complete_training_validation_only",
        "inputs": {"model_ready": {"path": rel(MODEL_READY)}, "files": input_hashes},
        "outputs": {"path": rel(out), "model_paths": model_paths, "output_hashes": {rel(path): sha256_file(path) for path in output_paths}},
        "config_path": rel(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_path": rel(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "feature_columns": FEATURE_COLUMNS,
        "preprocessing": "SimpleImputer(strategy=median, add_indicator=True) fit on training only for both models; StandardScaler fit on training only for Logistic; no fill/winsorization.",
        "counts": counts,
        "validation_metrics": metrics,
        "test_policy": config["test_policy"],
        "checks": checks,
        "limitations": ["Fundamental and market-cap controls are retained as candidate controls but marked UNVERIFIED by upstream clean stage; formal semantic gate is not passed.", "AI-pool monthly cross-sections have fewer than the registered 10 securities per top/bottom side, so top-bottom spreads are diagnostics with side size one in most months.", "No test target, test prediction, test metric or completed backtest is produced."],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    append_log(summary, out)
    append_ai_use_log(summary, out)
    print(json.dumps({"run_id": run_id, "out": rel(out), "status": summary["status"], "counts": counts, "validation_metrics": metrics, "checks": checks}, ensure_ascii=False, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit AI pool controls-only baseline models on development data.")
    parser.add_argument("--run-id", default=None, help="Optional run-scoped identifier; output directory must not already exist.")
    args = parser.parse_args()
    run(args.run_id)
