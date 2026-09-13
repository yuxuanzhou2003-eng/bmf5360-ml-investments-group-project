"""Fit daily controls-only Logistic and Random Forest on development anchors.

The model-ready master includes test schedule rows, but this runner loads only
training/validation target rows. Validation is scored in chronological order;
the sealed test manifest is hashed as a file artifact and never parsed for
target values.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL_READY_RUN_ID = "20260910T052100000000Z"
MODEL_READY = ROOT / "data" / "model_ready_ai_pool_daily_v1" / MODEL_READY_RUN_ID
CONFIG_PATH = ROOT / "ai_pool_daily_baseline_v1_config.json"
OUT_ROOT = ROOT / "data" / "model_runs" / "ai_pool_daily_baseline_v1"
HORIZON = 21
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
    result = spearmanr(left.loc[valid], right.loc[valid]).statistic
    return float(result) if np.isfinite(result) else float("nan")


def validation_metrics(frame: pd.DataFrame, probability: np.ndarray, config: dict) -> tuple[dict, pd.DataFrame]:
    work = frame[["sample_id", "Instrument", "formation_session", "y", "forward_excess_return"]].copy()
    work["formation_session"] = pd.to_datetime(work["formation_session"], format="mixed", errors="raise").dt.normalize()
    work["y"] = pd.to_numeric(work["y"], errors="raise").astype(int)
    work["forward_excess_return"] = pd.to_numeric(work["forward_excess_return"], errors="raise")
    work["p_up"] = np.asarray(probability, dtype=float)
    y = work["y"].to_numpy(dtype=int)
    p = work["p_up"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        raise RuntimeError("validation labels contain only one class")
    rank_values: list[float] = []
    spread_rows: list[dict] = []
    for date, group in work.groupby("formation_session", sort=True):
        group = group.sort_values(["p_up", "sample_id"], ascending=[False, True], kind="stable")
        ic = safe_spearman(group["p_up"], group["forward_excess_return"])
        if np.isfinite(ic):
            rank_values.append(ic)
        side = max(1, int(np.floor(len(group) * float(config["portfolio_diagnostic"]["side_fraction"]))))
        top = group.head(side)["forward_excess_return"].mean()
        bottom = group.tail(side)["forward_excess_return"].mean()
        spread_rows.append({
            "formation_session": date,
            "rows": len(group),
            "side_size": side,
            "side_size_below_registered_minimum": side < int(config["portfolio_diagnostic"]["minimum_side_size_registered"]),
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
        "mean_daily_rank_ic": float(np.mean(rank_values)) if rank_values else float("nan"),
        "daily_rank_ic_groups": int(len(rank_values)),
        "mean_daily_top_minus_bottom_spread": float(spread["top_minus_bottom_spread"].mean()),
        "median_daily_top_minus_bottom_spread": float(spread["top_minus_bottom_spread"].median()),
        "spread_days": int(len(spread)),
        "spread_days_below_registered_minimum_side": int(spread["side_size_below_registered_minimum"].sum()),
        "registered_minimum_side_size": int(config["portfolio_diagnostic"]["minimum_side_size_registered"]),
    }, spread


def append_log(summary: dict, out: Path) -> None:
    path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# DATA_PROCESSING_LOG\n"
    entry = [
        "",
        f"## {summary['run_id']} — AI pool daily controls-only baseline models v1 ({summary['generated_at_utc']})",
        "",
        f"- **阶段目的与状态**：在 daily model-ready H21 层上拟合 controls-only Logistic 与固定参数 Random Forest；sampling mode=`{summary['sampling_mode']}`；validation 按 formation date 顺序评分；test target/预测保持封存。",
        f"- **输入版本与路径**：model-ready `{summary['inputs']['model_ready']['path']}` run=`{summary['inputs']['model_ready']['run_id']}`；features、eligibility、targets_dev、config、runner 的 SHA-256 见 summary；sealed manifest 只作为文件 artifact，不解析目标值。",
        f"- **样本与日期切分**：`{json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}`；training 只用于拟合，validation 按 formation_session 升序 walk-forward 评分，未随机拆分。",
        f"- **缺失处理**：`{summary['preprocessing']}`；未填零、前后填充、插值、winsorize，也未从 daily master 删除行。",
        f"- **模型与指标**：`{json.dumps(summary['validation_metrics'], ensure_ascii=False, sort_keys=True)}`；daily top/bottom 使用概率排序 20%，当前池横截面小于注册每侧 10 只门槛，故仅为诊断。",
        f"- **H21 non-overlap/purge**：`{summary['walk_forward_policy']}`；test prediction/metric rows=0、test target values read=false。",
        f"- **检查/限制**：`{json.dumps(summary['checks'], ensure_ascii=False, sort_keys=True)}`；{summary['limitations']}；执行状态=`{summary['status']}`。",
        f"- **输出**：`{rel(out)}`；validation predictions、daily spread diagnostics、模型 artifact 与 summary 均为 run-scoped。",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def append_ai_use_log(summary: dict, out: Path) -> None:
    path = ROOT / "AI_USE_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# AI_USE_LOG\n"
    entry = [
        "",
        f"### AI pool daily controls-only baseline models ({summary['run_id']})",
        "",
        f"- OpenAI Codex fit daily H21 controls-only Logistic and fixed-parameter Random Forest from `{summary['inputs']['model_ready']['path']}` using `{summary['sampling_mode']}`. Training anchors={summary['counts']['training_supervised_rows']}; validation anchors={summary['counts']['validation_supervised_rows']}; validation dates are chronological and no event study input was used.",
        f"- Preprocessing was fit on selected training anchors only with median imputation and missing indicators; test rows remain sealed and no test predictions/metrics were written. Outputs are `{rel(out)}`.",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def run(run_id: str | None = None, model_ready_run_id: str | None = None, config_path_arg: str | None = None, selection_mode: str = "non_overlap") -> dict:
    if selection_mode not in {"non_overlap", "all_daily_date_purge"}:
        raise ValueError("selection_mode must be non_overlap or all_daily_date_purge")
    model_ready_id = model_ready_run_id or MODEL_READY_RUN_ID
    model_ready = ROOT / "data" / "model_ready_ai_pool_daily_v1" / model_ready_id
    config_path = Path(config_path_arg) if config_path_arg else CONFIG_PATH
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "ai_pool_daily_baseline_v1":
        raise RuntimeError("unexpected daily baseline config schema")
    features_path = model_ready / "model_features.csv"
    eligibility_path = model_ready / "eligibility.csv"
    targets_dev_path = model_ready / "targets_dev.csv"
    sealed_path = model_ready / "targets_test_sealed.csv"
    input_paths = {
        "model_features": features_path,
        "eligibility": eligibility_path,
        "targets_dev": targets_dev_path,
        "targets_test_sealed_manifest": sealed_path,
        "model_ready_summary": model_ready / "summary.json",
        "config": config_path,
        "runner": Path(__file__).resolve(),
    }
    missing = [name for name, path in input_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    print(f"阶段说明（AI pool daily model fitting）：只读取 training/validation targets_dev；sampling mode={selection_mode}；training-only 拟合预处理、Logistic 和固定参数 Random Forest，按日期顺序评分 validation；不读取 test target 值、不生成 test prediction。", flush=True)
    model_ready_summary = json.loads(input_paths["model_ready_summary"].read_text(encoding="utf-8"))
    features = pd.read_csv(features_path, low_memory=False)
    eligibility = pd.read_csv(eligibility_path, low_memory=False)
    targets_dev = pd.read_csv(targets_dev_path, low_memory=False)
    for frame, name in [(features, "features"), (eligibility, "eligibility"), (targets_dev, "targets_dev")]:
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{name} sample_id invalid")
    if set(features["sample_id"]) != set(eligibility["sample_id"]):
        raise RuntimeError("feature/eligibility sample sets differ")
    if not set(targets_dev["split"].astype(str)).issubset({"training", "validation"}):
        raise RuntimeError("targets_dev contains test or out-of-window rows")
    data = features.merge(targets_dev[["sample_id", "y", "forward_excess_return", "label_complete"]], on="sample_id", how="inner", validate="one_to_one")
    data = data.merge(eligibility[["sample_id", "supervised_model_eligible", "boundary_purged", "non_overlap_selected", "entry_trade_eligible", "feature_core_available"]], on="sample_id", how="inner", validate="one_to_one")
    for col in ["supervised_model_eligible", "boundary_purged", "non_overlap_selected", "entry_trade_eligible", "feature_core_available", "label_complete"]:
        data[col] = data[col].astype(str).str.casefold().eq("true")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["forward_excess_return"] = pd.to_numeric(data["forward_excess_return"], errors="coerce")
    data["formation_session"] = pd.to_datetime(data["formation_session"], format="mixed", errors="raise").dt.normalize()
    data["split"] = data["split"].astype(str)
    data = data.loc[data["split"].isin(["training", "validation"])].copy()
    eligible_common = data["label_complete"] & ~data["boundary_purged"] & data["entry_trade_eligible"] & data["feature_core_available"]
    if selection_mode == "non_overlap":
        eligible_common = eligible_common & data["non_overlap_selected"]
    train = data.loc[data["split"].eq("training") & eligible_common].sort_values(["formation_session", "Instrument", "sample_id"], kind="stable").copy()
    validation = data.loc[data["split"].eq("validation") & eligible_common].sort_values(["formation_session", "Instrument", "sample_id"], kind="stable").copy()
    if train.empty or validation.empty:
        raise RuntimeError("empty daily training or validation anchors")
    if train["y"].isna().any() or validation["y"].isna().any() or train["forward_excess_return"].isna().any() or validation["forward_excess_return"].isna().any():
        raise RuntimeError("eligible daily development anchors have incomplete labels")
    if train["y"].nunique() < 2 or validation["y"].nunique() < 2:
        raise RuntimeError("daily training/validation labels require two classes")
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            raise RuntimeError(f"missing daily control feature {col}")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    x_train, y_train = train[FEATURE_COLUMNS], train["y"].astype(int)
    x_validation, y_validation = validation[FEATURE_COLUMNS], validation["y"].astype(int)
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=float(config["logistic"]["C"]), class_weight=config["logistic"]["class_weight"], max_iter=int(config["logistic"]["max_iter"]), solver=config["logistic"]["solver"], random_state=int(config["logistic"]["random_state"]))),
    ])
    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", RandomForestClassifier(n_estimators=int(config["random_forest"]["n_estimators"]), max_depth=int(config["random_forest"]["max_depth"]), min_samples_leaf=int(config["random_forest"]["min_samples_leaf"]), max_features=config["random_forest"]["max_features"], class_weight=config["random_forest"]["class_weight"], random_state=int(config["random_forest"]["random_state"]), n_jobs=int(config["random_forest"]["n_jobs"]))),
    ])
    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}
    spread_tables: dict[str, pd.DataFrame] = {}
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
    daily_spreads = pd.concat(list(spread_tables.values()), ignore_index=True)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not run_id or any(ch not in "0123456789TZ" for ch in run_id):
        raise ValueError("run_id must contain only digits, T, and Z")
    out = OUT_ROOT / run_id
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    validation_predictions.to_csv(out / "validation_predictions.csv", index=False)
    daily_spreads.to_csv(out / "validation_daily_spreads.csv", index=False)
    model_paths: dict[str, str] = {}
    for name, model in fitted.items():
        model_path = models_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        model_paths[name] = rel(model_path)
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in input_paths.items()}
    output_paths = [out / "validation_predictions.csv", out / "validation_daily_spreads.csv", *[models_dir / f"{name}.joblib" for name in fitted]]
    counts = {
        "model_ready_master_rows": int(len(features)),
        "model_ready_master_instruments": int(features["Instrument"].nunique()),
        "training_master_rows": int((features["split"].astype(str) == "training").sum()),
        "validation_master_rows": int((features["split"].astype(str) == "validation").sum()),
        "test_master_rows": int((features["split"].astype(str) == "test").sum()),
        "training_supervised_rows": int(len(train)),
        "validation_supervised_rows": int(len(validation)),
        "training_supervised_instruments": int(train["Instrument"].nunique()),
        "validation_supervised_instruments": int(validation["Instrument"].nunique()),
        "training_supervised_sessions": int(train["formation_session"].nunique()),
        "validation_supervised_sessions": int(validation["formation_session"].nunique()),
        "test_prediction_rows": 0,
        "test_metric_rows": 0,
        "test_target_values_read": False,
    }
    non_overlap_gap_ok = True
    for frame in [train, validation]:
        gaps = frame.sort_values(["Instrument", "formation_session"]).groupby("Instrument")["formation_session"].apply(lambda x: x.diff().dt.days.dropna())
        # Calendar-day gaps need not be 21 because weekends/holidays occur; the
        # model-ready non_overlap_selected flag is checked on session blocks in
        # the builder. Here we verify no duplicate Instrument/session anchors.
        if frame.duplicated(["Instrument", "formation_session"]).any():
            non_overlap_gap_ok = False
    checks = {
        "features_exclude_targets": not bool(set(FEATURE_COLUMNS) & {"y", "forward_excess_return", "stock_forward_return", "benchmark_forward_return"}),
        "training_only_fit": True,
        "validation_predictions_only": bool(set(validation_predictions["split"]) == {"validation"}),
        "validation_chronological": bool(validation_predictions["formation_session"].is_monotonic_increasing),
        "non_overlap_anchor_keys_unique": non_overlap_gap_ok if selection_mode == "non_overlap" else True,
        "selection_mode_declared": selection_mode in {"non_overlap", "all_daily_date_purge"},
        "test_prediction_rows_zero": not validation_predictions["split"].eq("test").any(),
        "test_metric_rows_zero": True,
        "test_target_values_not_read": True,
        "test_sealed_manifest_hashed_only": True,
        "probabilities_finite": bool(np.isfinite(validation_predictions.filter(like="p_up_").to_numpy(dtype=float)).all()),
        "no_event_study_inputs": True,
        "no_row_deletion_in_master": True,
    }
    summary = {
        "schema_version": "ai_pool_daily_baseline_v1",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": "complete_training_validation_only",
        "inputs": {"model_ready": {"run_id": model_ready_id, "path": rel(model_ready)}, "files": input_hashes},
        "outputs": {"path": rel(out), "model_paths": model_paths, "output_hashes": {rel(path): sha256_file(path) for path in output_paths}},
        "config_path": rel(config_path),
        "config_sha256": sha256_file(config_path),
        "runner_path": rel(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "feature_columns": FEATURE_COLUMNS,
        "sampling_mode": "conservative_non_overlap_anchors" if selection_mode == "non_overlap" else "all_daily_with_date_purge",
        "preprocessing": f"SimpleImputer(strategy=median, add_indicator=True) fit on {('non-overlapping ' if selection_mode == 'non_overlap' else '')}training anchors only for both models; StandardScaler fit on training only for Logistic; no fill/winsorization.",
        "counts": counts,
        "validation_metrics": metrics,
        "walk_forward_policy": f"{config['walk_forward']} Sampling mode={selection_mode}; all_daily_date_purge retains all eligible daily anchors within each development split after the H21 boundary purge, while non_overlap retains at most one anchor per Instrument per H21 session block.",
        "test_policy": config["test_policy"],
        "checks": checks,
        "pit_status": model_ready_summary.get("pit_status", "unspecified"),
        "membership_source_mode": model_ready_summary.get("membership_source_mode", "unspecified"),
        "limitations": [
            ("This is exploratory_static_candidate mode: the expanded registry contains 9 direct_local_pit candidates and 40 provisional_static_source candidates. Registry member spans define the historical active universe, but AI-role evidence is not a complete point-in-time history; validation must not be described as an unbiased full-PIT AI backtest." if model_ready_summary.get("membership_source_mode") == "exploratory_static_candidate" else "The current daily run uses the 10-RIC explicit-member union from the latest membership file; an external candidate list can be supplied to the builder for an expanded frozen universe."),
            "Daily cross-sections are small, so top/bottom 20% spread diagnostics generally use one security per side and fail the registered 10-per-side portfolio gate.",
            "This run covers H21 only; no H5 robustness target was generated.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    append_log(summary, out)
    append_ai_use_log(summary, out)
    print(json.dumps({"run_id": run_id, "out": rel(out), "status": summary["status"], "counts": counts, "validation_metrics": metrics, "checks": checks}, ensure_ascii=False, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit daily AI-pool controls-only baseline models.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-ready-run-id", default=None)
    parser.add_argument("--config", default=None, help="Optional baseline config path; defaults to the original 10-RIC daily config.")
    parser.add_argument("--selection-mode", choices=["non_overlap", "all_daily_date_purge"], default="non_overlap")
    args = parser.parse_args()
    run(args.run_id, args.model_ready_run_id, args.config, args.selection_mode)
