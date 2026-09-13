"""Train four daily H21 state models using training/validation data only.

The primary sample is the model-ready ``non_overlap_selected`` anchor set.
Random-forest hyperparameters are selected with purged expanding walk-forward
folds made only from training anchors.  The sealed test-target file is never
opened by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
DEFAULT_CONFIG_PATH = ROOT / "ai_pool_daily_state_models_v1_config.json"
DEFAULT_OUT_ROOT = ROOT / "data" / "model_runs" / "ai_pool_daily_state_models_v1"
HORIZON = 21
TARGET_COLUMNS = {"y", "forward_excess_return", "stock_forward_return", "benchmark_forward_return"}
KEY_COLUMNS = ["sample_id", "security_id", "Instrument", "formation_session", "split"]


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


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3 or left.loc[valid].nunique() < 2 or right.loc[valid].nunique() < 2:
        return float("nan")
    value = spearmanr(left.loc[valid], right.loc[valid]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def unique_columns(columns: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)
    return result


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def metric_dict(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    work = frame[["formation_session", "Instrument", "y", "forward_excess_return"]].copy()
    p = np.asarray(probability, dtype=float)
    y = work["y"].astype(int).to_numpy()
    work["p_up"] = p
    rank_values: list[float] = []
    spreads: list[float] = []
    for _date, group in work.groupby("formation_session", sort=True):
        rank_values.append(safe_spearman(group["p_up"], group["forward_excess_return"]))
        ordered = group.sort_values(["p_up"], ascending=False, kind="stable")
        side = max(1, int(np.floor(len(ordered) * 0.2)))
        spreads.append(float(ordered.head(side)["forward_excess_return"].mean() - ordered.tail(side)["forward_excess_return"].mean()))
    finite_rank = [x for x in rank_values if np.isfinite(x)]
    if len(np.unique(y)) < 2:
        raise RuntimeError("a scoring fold has only one class")
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
        "mean_daily_rank_ic": float(np.mean(finite_rank)) if finite_rank else float("nan"),
        "daily_rank_ic_groups": int(len(finite_rank)),
        "mean_daily_top_minus_bottom_spread": float(np.mean(spreads)) if spreads else float("nan"),
    }


def make_logistic(config: dict[str, Any]) -> Pipeline:
    params = config["logistic"]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(params["C"]), class_weight=params["class_weight"],
            max_iter=int(params["max_iter"]), solver=params["solver"],
            random_state=int(params["random_state"]),
        )),
    ])


def make_forest(config: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    fixed = config["random_forest_fixed"]
    max_depth = params.get("max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=int(params["n_estimators"]), max_depth=max_depth,
            min_samples_leaf=int(params["min_samples_leaf"]), max_features=params["max_features"],
            class_weight=fixed["class_weight"], random_state=int(fixed["random_state"]),
            n_jobs=int(fixed["n_jobs"]),
        )),
    ])


def all_missing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if bool(frame[column].isna().all())]


def internal_folds(train: pd.DataFrame, years: list[int], horizon: int) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for year in years:
        valid = train.loc[train["formation_session"].dt.year.eq(year)].copy()
        if valid.empty:
            continue
        start = pd.Timestamp(valid["formation_session"].min())
        if "exit_session" not in train.columns:
            raise RuntimeError("training data must include exit_session for purged folds")
        candidate = train.loc[train["formation_session"] < start].copy()
        fit = candidate.loc[candidate["exit_session"] < start].copy()
        if fit.empty:
            continue
        if (fit["exit_session"] >= start).any():
            raise RuntimeError(f"purged fold includes label exit on/after validation start: {year}")
        last_fit_exit = pd.Timestamp(fit["exit_session"].max())
        purged_rows = int(len(candidate) - len(fit))
        folds.append({
            "fold_id": f"train_to_{year - 1}_validate_{year}",
            "validation_year": year,
            "validation_start": start.strftime("%Y-%m-%d"),
            "validation_end": pd.Timestamp(valid["formation_session"].max()).strftime("%Y-%m-%d"),
            "purge_rule": "fit rows require exit_session < validation_start",
            "last_fit_exit": last_fit_exit.strftime("%Y-%m-%d"),
            "candidate_rows_before_purge": int(len(candidate)),
            "purged_rows": purged_rows,
            "purge_sessions": horizon,
            "fit": fit,
            "validation": valid,
        })
    return folds


def tune_forest(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    folds = internal_folds(train, list(config["tuning"]["fold_validation_years"]), HORIZON)
    evidence: list[dict[str, Any]] = []
    grid = config["random_forest_tuning_grid"]
    for grid_index, params in enumerate(grid):
        fold_metrics: list[dict[str, Any]] = []
        for fold in folds:
            fit = fold["fit"]
            valid = fold["validation"]
            model = make_forest(config, params)
            missing = all_missing_columns(fit, feature_columns)
            model.fit(fit[feature_columns], fit["y"].astype(int))
            probability = model.predict_proba(valid[feature_columns])[:, 1]
            metrics = metric_dict(valid, probability)
            fold_metrics.append({
                "fold_id": fold["fold_id"],
                "validation_year": fold["validation_year"],
                "fit_rows": int(len(fit)),
                "validation_rows": int(len(valid)),
                "candidate_rows_before_purge": fold["candidate_rows_before_purge"],
                "purged_rows": fold["purged_rows"],
                "purge_rule": fold["purge_rule"],
                "validation_start": fold["validation_start"],
                "last_fit_exit": fold["last_fit_exit"],
                "all_missing_training_columns": missing,
                "roc_auc": metrics["roc_auc"],
                "brier": metrics["brier"],
            })
        aucs = [float(x["roc_auc"]) for x in fold_metrics if np.isfinite(x["roc_auc"])]
        briers = [float(x["brier"]) for x in fold_metrics if np.isfinite(x["brier"])]
        evidence.append({
            "grid_index": grid_index,
            "params": params,
            "folds": fold_metrics,
            "fold_count": len(fold_metrics),
            "mean_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "mean_brier": float(np.mean(briers)) if briers else float("nan"),
        })
    if not evidence or not any(item["fold_count"] for item in evidence):
        raise RuntimeError("no usable purged training-only RF tuning folds")
    selected = sorted(
        evidence,
        key=lambda item: (
            -(item["mean_roc_auc"] if np.isfinite(item["mean_roc_auc"]) else -np.inf),
            item["mean_brier"] if np.isfinite(item["mean_brier"]) else np.inf,
            item["grid_index"],
        ),
    )[0]
    fold_specs = [
        {key: value for key, value in fold.items() if key not in {"fit", "validation"}}
        for fold in folds
    ]
    return dict(selected["params"]), evidence, fold_specs


def transformed_feature_importance(model: Pipeline, feature_columns: list[str], feature_set: str, model_name: str) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    names = list(imputer.get_feature_names_out(feature_columns))
    importances = classifier.feature_importances_ if hasattr(classifier, "feature_importances_") else np.abs(classifier.coef_[0])
    rows: list[dict[str, Any]] = []
    for name, value in zip(names, importances):
        is_indicator = name.startswith("missingindicator_")
        base = name.removeprefix("missingindicator_") if is_indicator else name
        rows.append({
            "feature_set": feature_set,
            "model": model_name,
            "transformed_feature": name,
            "base_feature": base,
            "missing_indicator": is_indicator,
            "importance": float(value),
        })
    return pd.DataFrame(rows)


def append_logs(summary: dict[str, Any], out: Path) -> None:
    processing = ROOT / "DATA_PROCESSING_LOG.md"
    prior = processing.read_text(encoding="utf-8") if processing.exists() else "# DATA_PROCESSING_LOG\n"
    missingness_sentence = (
        "AIQ/BOTZ/IGV pre-listing features were excluded by the independently audited coverage-safe registry; remaining structural missingness is retained."
        if summary.get("ai_state_safe_feature_audit")
        else "AIQ/BOTZ/IGV pre-listing missingness was retained and reported."
    )
    entry = [
        "",
        f"## {summary['run_id']} — enhanced AI-pool daily state models v1",
        "",
        "- **阶段目的与执行状态**：在最终 independently verified AI state run、已验证 macro run 和 expanded daily model-ready run 上比较 technical_only、technical_plus_macro、technical_plus_ai_state、full_state 四组特征；primary 为 H21 non-overlap anchors；状态为 training/validation only。",
        f"- **输入版本与路径**：{json.dumps(summary['inputs'], ensure_ascii=False, sort_keys=True)}；`targets_test_sealed.csv` 未打开、未读取、未聚合。",
        f"- **规则与模型**：{summary['sampling']['description']}；RF 调参仅使用 training 内 purged walk-forward folds，validation 只在最终 fit 后比较；信用利差六列排除；AI state effective features 记录于 `{rel(out / 'feature_sets.json')}`，并排除技术重复列与 `spy_return_1` 数值别名。",
        f"- **样本与行影响**：{json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}；主表未物理删除，结构性上市前缺失保留为 NA。",
        f"- **缺失处理**：{summary['preprocessing']['description']}；{missingness_sentence} median imputer 与 missing indicators 在每个 fit fold 内独立拟合，原始输入未修改。",
        f"- **RF tuning evidence**：{rel(out / 'rf_tuning_evidence.json')}；四个 feature set 各自只用 training folds 选择超参数。",
        f"- **检查与限制**：{json.dumps(summary['checks'], ensure_ascii=False, sort_keys=True)}；{summary['limitations']}。",
        f"- **输出**：`{rel(out)}`；validation predictions、metrics、RF tuning evidence、feature importance、model artifacts 与 audit 均为新 run。",
        "",
    ]
    processing.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")
    ai_log = ROOT / "AI_USE_LOG.md"
    prior_ai = ai_log.read_text(encoding="utf-8") if ai_log.exists() else "# AI_USE_LOG\n"
    ai_entry = [
        "",
        f"### Enhanced AI-pool daily state models ({summary['run_id']})",
        "",
        f"- OpenAI Codex created and ran `{rel(Path(__file__).resolve())}` for four feature groups: technical_only, technical_plus_macro, technical_plus_ai_state and full_state. Inputs were the model-ready `{summary['inputs']['model_ready']['run_id']}`, macro `{summary['inputs']['macro']['run_id']}` and independently verified AI state `{summary['inputs']['ai_state']['run_id']}` runs.",
        "- The primary sample used 21-session non-overlap anchors. RF hyperparameters were selected only from purged training walk-forward folds; validation labels were used only for the final comparison. The sealed test-target file was never opened and no test predictions or metrics were produced.",
        f"- Median imputation and missing indicators were fit within each training fold; {missingness_sentence} Credit-spread six-column macro features were excluded. AI-state technical duplicates and the `spy_return_1` alias were dropped or excluded upstream, as recorded in the run summary. Outputs: `{rel(out)}`.",
        "",
    ]
    ai_log.write_text(prior_ai.rstrip("\n") + "\n" + "\n".join(ai_entry), encoding="utf-8", newline="\n")


def run(run_id: str | None = None, config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "ai_pool_daily_state_models_v1":
        raise RuntimeError("unexpected state-model config schema")
    model_ready_id = config["model_ready_run_id"]
    macro_id = config["macro_run_id"]
    ai_id = config["ai_state_run_id"]
    model_ready = ROOT / "data" / "model_ready_ai_pool_daily_v1" / model_ready_id
    macro_dir = ROOT / "data" / "clean" / "daily_macro_v1" / macro_id
    ai_dir = ROOT / "data" / "clean" / "ai_daily_macro_features_v1" / ai_id
    model_features_path = model_ready / "model_features.csv"
    eligibility_path = model_ready / "eligibility.csv"
    targets_dev_path = model_ready / "targets_dev.csv"
    macro_path = macro_dir / "macro_features_train_valid.csv"
    ai_path = ai_dir / "ai_state_features.csv"
    input_summary_paths = {
        "model_ready_summary": model_ready / "summary.json",
        "macro_summary": macro_dir / "clean_summary.json",
        "ai_state_summary": ai_dir / "summary.json",
    }
    safe_feature_audit = config.get("ai_state_safe_feature_audit")
    if safe_feature_audit:
        safe_audit_path = Path(safe_feature_audit["summary_path"])
        if not safe_audit_path.is_absolute():
            safe_audit_path = ROOT / safe_audit_path
        input_summary_paths["ai_state_safe_feature_audit"] = safe_audit_path
    required = [model_features_path, eligibility_path, targets_dev_path, macro_path, ai_path, *input_summary_paths.values()]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    print("阶段说明（增强 AI 状态模型）：读取 training/validation 的 targets_dev 与特征；按 H21 non-overlap anchor 建模；RF 只在 training purged walk-forward folds 调参；只输出 validation predictions，不打开 sealed test targets。", flush=True)

    technical = list(config["technical_features"])
    macro_features = list(config["macro_features"])
    ai_features_declared = list(config["ai_state_features"])
    alias_config = config.get("feature_alias_exclusions", {})
    dropped_duplicate_columns = unique_columns([
        column for column in ai_features_declared if column in set(technical)
    ])
    configured_duplicates = alias_config.get("technical_duplicates", dropped_duplicate_columns)
    if set(configured_duplicates) != set(dropped_duplicate_columns):
        raise RuntimeError("configured technical duplicate exclusions do not match declared feature overlap")
    declared_aliases = alias_config.get("numeric_aliases", {})
    if isinstance(declared_aliases, list):
        declared_aliases = {
            item["dropped"]: item["kept"] for item in declared_aliases
            if isinstance(item, dict) and "dropped" in item and "kept" in item
        }
    if any(kept not in technical for kept in declared_aliases.values()):
        raise RuntimeError("numeric alias target is not a technical feature")
    dropped_alias_columns = [
        column for column in ai_features_declared if column in declared_aliases
    ]
    ai_features = [
        column for column in ai_features_declared
        if column not in set(dropped_duplicate_columns) and column not in set(dropped_alias_columns)
    ]
    if "spy_return_1" not in dropped_alias_columns:
        if not alias_config.get("allow_pre_excluded_alias", False):
            raise RuntimeError("the declared spy_return_1 numeric alias must be excluded")
    safe_feature_audit_summary: dict[str, Any] | None = None
    safe_feature_audit_list: list[str] | None = None
    if safe_feature_audit:
        safe_feature_audit_summary = json.loads(input_summary_paths["ai_state_safe_feature_audit"].read_text(encoding="utf-8"))
        list_name = safe_feature_audit["feature_list_name"]
        cursor: Any = safe_feature_audit_summary
        for part in list_name.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise RuntimeError(f"safe feature list missing from audit summary: {list_name}")
            cursor = cursor[part]
        safe_feature_audit_list = list(cursor)
        if safe_feature_audit_list != ai_features_declared:
            raise RuntimeError("configured AI features do not exactly match the independently audited safe feature list")
    excluded_credit = set(config["excluded_credit_spread_features"])
    if excluded_credit & set(macro_features):
        raise RuntimeError("excluded credit-spread features leaked into macro feature list")
    feature_sets = {
        "technical_only": unique_columns(technical),
        "technical_plus_macro": unique_columns(technical + macro_features),
        "technical_plus_ai_state": unique_columns(technical + ai_features),
        "full_state": unique_columns(technical + macro_features + ai_features),
    }
    all_features = unique_columns([x for values in feature_sets.values() for x in values])
    base_usecols = unique_columns(KEY_COLUMNS + technical)
    features = pd.read_csv(model_features_path, usecols=base_usecols, low_memory=False)
    eligibility = pd.read_csv(eligibility_path, usecols=["sample_id", "boundary_purged", "entry_trade_eligible", "feature_core_available", "non_overlap_selected", "label_complete"], low_memory=False)
    targets = pd.read_csv(targets_dev_path, usecols=["sample_id", "split", "exit_session", "y", "forward_excess_return", "label_complete"], low_memory=False)
    if not set(targets["split"].astype(str)).issubset({"training", "validation"}):
        raise RuntimeError("targets_dev includes a non-development split")
    for frame, name in [(features, "model_features"), (eligibility, "eligibility"), (targets, "targets_dev")]:
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{name} sample_id invalid")
    if set(features["sample_id"]) != set(eligibility["sample_id"]):
        raise RuntimeError("model feature and eligibility sample sets differ")
    data = features.merge(targets, on="sample_id", how="inner", validate="one_to_one", suffixes=("", "_target"))
    data = data.merge(eligibility, on="sample_id", how="inner", validate="one_to_one", suffixes=("", "_eligibility"))
    if not data["split"].astype(str).eq(data["split_target"].astype(str)).all():
        raise RuntimeError("target and feature split labels differ")
    data["formation_session"] = pd.to_datetime(data["formation_session"], format="mixed", errors="raise").dt.normalize()
    data["exit_session"] = pd.to_datetime(data["exit_session"], format="mixed", errors="raise").dt.normalize()
    data["split"] = data["split"].astype(str)
    for column in ["boundary_purged", "entry_trade_eligible", "feature_core_available", "non_overlap_selected", "label_complete", "label_complete_eligibility"]:
        if column in data:
            data[column] = bool_series(data[column])
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["forward_excess_return"] = pd.to_numeric(data["forward_excess_return"], errors="coerce")

    macro_usecols = ["spy_session", *macro_features]
    macro = pd.read_csv(macro_path, usecols=macro_usecols, low_memory=False)
    macro["formation_session"] = pd.to_datetime(macro.pop("spy_session"), format="mixed", errors="raise").dt.normalize()
    if macro["formation_session"].duplicated().any():
        raise RuntimeError("macro session keys are not unique")
    ai_usecols = ["formation_session", "split", *ai_features]
    ai = pd.read_csv(ai_path, usecols=ai_usecols, low_memory=False)
    ai["formation_session"] = pd.to_datetime(ai["formation_session"], format="mixed", errors="raise").dt.normalize()
    ai = ai.loc[ai["split"].astype(str).isin({"training", "validation"})].drop(columns=["split"])
    if ai["formation_session"].duplicated().any():
        raise RuntimeError("AI-state formation session keys are not unique")
    data = data.merge(macro, on="formation_session", how="left", validate="many_to_one")
    data = data.merge(ai, on="formation_session", how="left", validate="many_to_one")
    for column in all_features:
        if column not in data.columns:
            raise RuntimeError(f"missing declared feature after merge: {column}")
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[data["split"].isin({"training", "validation"})].copy()
    eligible = (
        data["label_complete"] & ~data["boundary_purged"] & data["entry_trade_eligible"]
        & data["feature_core_available"] & data["non_overlap_selected"]
        & data["y"].notna() & data["forward_excess_return"].notna()
    )
    train = data.loc[data["split"].eq("training") & eligible].sort_values(["formation_session", "Instrument", "sample_id"], kind="stable").copy()
    validation = data.loc[data["split"].eq("validation") & eligible].sort_values(["formation_session", "Instrument", "sample_id"], kind="stable").copy()
    if train.empty or validation.empty or train["y"].nunique() < 2 or validation["y"].nunique() < 2:
        raise RuntimeError("primary training/validation sample is empty or single-class")
    if train.duplicated(["Instrument", "formation_session"]).any() or validation.duplicated(["Instrument", "formation_session"]).any():
        raise RuntimeError("primary non-overlap anchor keys are duplicated")

    # Availability diagnostics are descriptions only; no rows or values are changed.
    structural_missing: dict[str, Any] = {}
    for ticker in ["aiq", "botz", "igv"]:
        columns = [column for column in ai_features if column.startswith(f"etf_{ticker}_") and "reason_code" not in column]
        structural_missing[ticker] = {
            "features": columns,
            "by_feature": {
                column: {
                    "training_missing_rows": int(train[column].isna().sum()),
                    "validation_missing_rows": int(validation[column].isna().sum()),
                    "training_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "first_observed_training_validation": (train.loc[train[column].notna(), "formation_session"].min().strftime("%Y-%m-%d") if train[column].notna().any() else None),
                }
                for column in columns
            },
        }

    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}
    importance_tables: list[pd.DataFrame] = []
    tuning: dict[str, Any] = {}
    validation_predictions = validation[["sample_id", "security_id", "Instrument", "formation_session", "split", "y", "forward_excess_return"]].copy()
    for feature_set, columns in feature_sets.items():
        selected_params, evidence, folds = tune_forest(train, columns, config)
        tuning[feature_set] = {"selected_params": selected_params, "evidence": evidence, "folds": folds}
        models = {
            "logistic": make_logistic(config),
            "random_forest": make_forest(config, selected_params),
        }
        for model_name, model in models.items():
            model.fit(train[columns], train["y"].astype(int))
            probability = model.predict_proba(validation[columns])[:, 1]
            if not np.isfinite(probability).all():
                raise RuntimeError(f"nonfinite validation probability: {feature_set}/{model_name}")
            metrics[f"{feature_set}/{model_name}"] = metric_dict(validation, probability)
            validation_predictions[f"p_up_{feature_set}_{model_name}"] = probability
            importance_tables.append(transformed_feature_importance(model, columns, feature_set, model_name))
            fitted[f"{feature_set}_{model_name}"] = model

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not run_id or any(ch not in "0123456789TZ-_" for ch in run_id):
        raise ValueError("run_id contains unsupported characters")
    out_root = Path(config.get("output_root", DEFAULT_OUT_ROOT))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out = out_root / run_id
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    validation_predictions.to_csv(out / "validation_predictions.csv", index=False)
    metrics_frame = pd.DataFrame([
        {"model_key": key, "feature_set": key.split("/", 1)[0], "model": key.split("/", 1)[1], **value}
        for key, value in metrics.items()
    ])
    metrics_frame.to_csv(out / "validation_metrics.csv", index=False)
    importance = pd.concat(importance_tables, ignore_index=True)
    importance.to_csv(out / "feature_importance.csv", index=False)
    (out / "feature_sets.json").write_text(json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "rf_tuning_evidence.json").write_text(json.dumps(tuning, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    model_paths: dict[str, str] = {}
    for model_key, model in fitted.items():
        path = models_dir / f"{model_key}.joblib"
        joblib.dump(model, path)
        model_paths[model_key] = rel(path)

    input_paths = {
        "model_features": model_features_path,
        "eligibility": eligibility_path,
        "targets_dev": targets_dev_path,
        "model_ready_summary": input_summary_paths["model_ready_summary"],
        "macro_features_train_valid": macro_path,
        "macro_summary": input_summary_paths["macro_summary"],
        "ai_state_features": ai_path,
        "ai_state_summary": input_summary_paths["ai_state_summary"],
        "config": config_path,
        "runner": Path(__file__).resolve(),
    }
    if "ai_state_safe_feature_audit" in input_summary_paths:
        input_paths["ai_state_safe_feature_audit"] = input_summary_paths["ai_state_safe_feature_audit"]
    output_files = [out / "validation_predictions.csv", out / "validation_metrics.csv", out / "feature_importance.csv", out / "feature_sets.json", out / "rf_tuning_evidence.json", *[models_dir / f"{key}.joblib" for key in fitted]]
    checks = {
        "primary_non_overlap_anchor_sample": bool(train["non_overlap_selected"].all() and validation["non_overlap_selected"].all()),
        "primary_anchor_keys_unique": bool(not train.duplicated(["Instrument", "formation_session"]).any() and not validation.duplicated(["Instrument", "formation_session"]).any()),
        "horizon_is_21_sessions": HORIZON == int(config["horizon_sessions"]),
        "rf_tuning_training_only": all(
            item["evidence"] and all(
                fold["fit_rows"] > 0
                and fold["validation_rows"] > 0
                and pd.Timestamp(fold["last_fit_exit"]) < pd.Timestamp(fold["validation_start"])
                for candidate in item["evidence"]
                for fold in candidate["folds"]
            )
            for item in tuning.values()
        ),
        "rf_tuning_has_purged_folds": all(item["folds"] and all(int(f["purge_sessions"]) == HORIZON for f in item["folds"]) for item in tuning.values()),
        "four_feature_sets_present": list(feature_sets) == ["technical_only", "technical_plus_macro", "technical_plus_ai_state", "full_state"],
        "credit_spread_features_excluded": not bool(excluded_credit & set(all_features)),
        "no_target_columns_in_features": not bool(set(all_features) & TARGET_COLUMNS),
        "validation_predictions_only": bool(set(validation_predictions["split"].astype(str)) == {"validation"} and len(validation_predictions) == len(validation)),
        "test_predictions_zero": True,
        "test_metrics_zero": True,
        "test_targets_not_opened": True,
        "all_probabilities_finite": bool(np.isfinite(validation_predictions.filter(like="p_up_").to_numpy(dtype=float)).all()),
        "macro_key_join_complete": bool(data["formation_session"].isin(set(macro["formation_session"])).all()),
        "ai_state_key_join_complete": bool(data["formation_session"].isin(set(ai["formation_session"])).all()),
        "raw_inputs_not_modified_by_runner": True,
    }
    if safe_feature_audit:
        checks["ai_safe_feature_list_matches_audit"] = bool(safe_feature_audit_list == ai_features_declared)
    # State columns may be NA before an ETF lists; coverage is therefore checked
    # by the formation-session key, while the original values remain unchanged.
    model_ready_summary = json.loads(input_summary_paths["model_ready_summary"].read_text(encoding="utf-8"))
    macro_summary = json.loads(input_summary_paths["macro_summary"].read_text(encoding="utf-8"))
    ai_summary = json.loads(input_summary_paths["ai_state_summary"].read_text(encoding="utf-8"))
    safe_audit_descriptor = None
    if safe_feature_audit:
        safe_audit_descriptor = {
            "run_id": safe_feature_audit["run_id"],
            "path": rel(input_summary_paths["ai_state_safe_feature_audit"]),
            "feature_list_name": safe_feature_audit["feature_list_name"],
            "feature_count": len(safe_feature_audit_list or []),
        }
    summary = {
        "schema_version": "ai_pool_daily_state_models_v1",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": "complete_training_validation_only",
        "inputs": {
            "model_ready": {"run_id": model_ready_id, "path": rel(model_ready)},
            "macro": {"run_id": macro_id, "path": rel(macro_dir)},
            "ai_state": {"run_id": ai_id, "path": rel(ai_dir)},
            "ai_state_safe_feature_audit": safe_audit_descriptor,
            "files": {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in input_paths.items()},
            "sealed_test_targets": {"path": rel(model_ready / "targets_test_sealed.csv"), "opened": False, "sha256": None},
        },
        "outputs": {"path": rel(out), "model_paths": model_paths, "output_hashes": {rel(path): sha256_file(path) for path in output_files}},
        "feature_sets": feature_sets,
        "feature_alias_exclusions": {
            "technical_duplicate_columns_dropped_from_ai_state": dropped_duplicate_columns,
            "numeric_alias_columns_dropped_from_ai_state": [
                {"dropped": column, "kept": declared_aliases[column]}
                for column in dropped_alias_columns
            ],
            "declared_ai_state_feature_count": len(ai_features_declared),
            "effective_ai_state_feature_count": len(ai_features),
            "upstream_safe_list_exclusions": list(alias_config.get("upstream_safe_list_exclusions", [])),
            "stable_deduplication": "technical columns are retained first; AI-state duplicates are omitted before merge",
        },
        "ai_state_safe_feature_audit": safe_audit_descriptor,
        "excluded_credit_spread_features": sorted(excluded_credit),
        "sampling": {
            "primary": "non_overlap_selected",
            "horizon_sessions": HORIZON,
            "description": "Use model-ready non_overlap_selected anchors only; no random splitting. Each expanding training fold includes only rows with exit_session strictly before that fold's validation_start, which purges every H21 label window that could overlap validation; final validation is scored after fitting on all primary training anchors.",
        },
        "preprocessing": {
            "description": "SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True) is fit separately on each training fold and on final primary training anchors; StandardScaler is fit on training only for Logistic. All-missing training columns retain an explicit missing indicator and sklearn empty-column fallback; no source row or value is modified.",
            "structural_missingness": structural_missing,
            "winsorization": False,
            "forward_fill": False,
            "zero_fill": False,
            "interpolation": False,
        },
        "counts": {
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
            "validation_prediction_rows": int(len(validation_predictions)),
            "test_prediction_rows": 0,
            "test_metric_rows": 0,
            "rows_physically_deleted": 0,
        },
        "validation_metrics": metrics,
        "rf_tuning": {key: {"selected_params": value["selected_params"], "fold_count": len(value["folds"]), "candidate_count": len(value["evidence"])} for key, value in tuning.items()},
        "checks": checks,
        "test_policy": config["test_policy"],
        "source_run_status": {
            "model_ready": model_ready_summary.get("status"),
            "macro": macro_summary.get("status"),
            "ai_state": ai_summary.get("status"),
            "ai_state_safe_feature_audit": safe_feature_audit_summary.get("status") if safe_feature_audit_summary else None,
        },
        "limitations": [
            ("The coverage-safe AI incremental list is taken exactly from the independently verified input audit; AIQ/BOTZ/IGV structural features, duplicate SPY momentum columns, and spy_return_1 alias are excluded upstream." if safe_feature_audit else "The expanded AI candidate registry is exploratory_static_candidate; validation inherits its stated point-in-time membership limitation."),
            "Validation metrics are development comparisons; no test target, prediction, ranking, metric, portfolio or performance claim is produced.",
            "AIQ/BOTZ/IGV pre-listing structural missingness is handled in model pipelines but remains a model assumption; it does not alter input rows.",
            "Credit-spread six-column macro features are excluded because train/validation coverage is unavailable in the selected macro run.",
        ],
    }
    (out / "audit.json").write_text(json.dumps({"schema_version": "ai_pool_daily_state_models_v1_audit", "run_id": run_id, "checks": checks, "test_targets_opened": False, "test_predictions_written": False, "raw_inputs_modified": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    append_logs(summary, out)
    print(json.dumps({"run_id": run_id, "out": rel(out), "status": summary["status"], "counts": summary["counts"], "checks": checks}, ensure_ascii=False, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="JSON config path relative to the workspace or absolute")
    args = parser.parse_args()
    run(args.run_id, args.config)
