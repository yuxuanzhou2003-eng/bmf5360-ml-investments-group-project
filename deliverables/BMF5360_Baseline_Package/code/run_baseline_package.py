"""Fit the four baseline daily H21 classifiers on development data only.

The package contains the two registered feature sets (technical-only and
technical-plus-macro) and compares Logistic Regression with the same-table
Random Forest reference.  It reads ``targets_dev.csv`` only.  The sealed test
target file is deliberately absent from the input list and is never opened.
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
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "baseline_package_config.json"
HORIZON = 21
TARGET_COLUMNS = {"y", "forward_excess_return", "stock_forward_return", "benchmark_forward_return"}
KEY_COLUMNS = ["sample_id", "security_id", "Instrument", "formation_session", "split"]
DEVELOPMENT_SPLITS = {"training", "validation"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def unique_columns(columns: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)
    return result


def bool_series(series: pd.Series, name: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.casefold()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name} contains invalid boolean tokens: {invalid}")
    return normalized.eq("true")


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3 or left.loc[valid].nunique() < 2 or right.loc[valid].nunique() < 2:
        return float("nan")
    value = spearmanr(left.loc[valid], right.loc[valid]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def metric_dict(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    work = frame[["formation_session", "Instrument", "y", "forward_excess_return"]].copy()
    p = np.asarray(probability, dtype=float)
    y = work["y"].astype(int).to_numpy()
    work["p_up"] = p
    rank_values: list[float] = []
    spreads: list[float] = []
    for _date, group in work.groupby("formation_session", sort=True):
        rank_values.append(safe_spearman(group["p_up"], group["forward_excess_return"]))
        ordered = group.sort_values("p_up", ascending=False, kind="stable")
        side = max(1, int(np.floor(len(ordered) * 0.2)))
        spreads.append(float(
            ordered.head(side)["forward_excess_return"].mean()
            - ordered.tail(side)["forward_excess_return"].mean()
        ))
    finite_rank = [value for value in rank_values if np.isfinite(value)]
    if len(np.unique(y)) < 2:
        raise RuntimeError("a scoring sample has only one class")
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
            C=float(params["C"]),
            class_weight=params["class_weight"],
            max_iter=int(params["max_iter"]),
            solver=params["solver"],
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
            n_estimators=int(params["n_estimators"]),
            max_depth=max_depth,
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=fixed["class_weight"],
            random_state=int(fixed["random_state"]),
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
        candidate = train.loc[train["formation_session"] < start].copy()
        fit = candidate.loc[candidate["exit_session"] < start].copy()
        if fit.empty:
            continue
        if (fit["exit_session"] >= start).any():
            raise RuntimeError(f"purged fold includes label exit on/after validation start: {year}")
        folds.append({
            "fold_id": f"train_to_{year - 1}_validate_{year}",
            "validation_year": year,
            "validation_start": start.strftime("%Y-%m-%d"),
            "validation_end": pd.Timestamp(valid["formation_session"].max()).strftime("%Y-%m-%d"),
            "purge_rule": "fit rows require exit_session < validation_start",
            "last_fit_exit": pd.Timestamp(fit["exit_session"].max()).strftime("%Y-%m-%d"),
            "candidate_rows_before_purge": int(len(candidate)),
            "purged_rows": int(len(candidate) - len(fit)),
            "purge_sessions": int(horizon),
            "fit": fit,
            "validation": valid,
        })
    return folds


def tune_forest(
    train: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    folds = internal_folds(train, list(config["tuning"]["fold_validation_years"]), horizon)
    evidence: list[dict[str, Any]] = []
    for grid_index, params in enumerate(config["random_forest_tuning_grid"]):
        fold_metrics: list[dict[str, Any]] = []
        for fold in folds:
            fit = fold["fit"]
            valid = fold["validation"]
            model = make_forest(config, params)
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
                "all_missing_training_columns": all_missing_columns(fit, feature_columns),
                "roc_auc": metrics["roc_auc"],
                "brier": metrics["brier"],
            })
        aucs = [float(item["roc_auc"]) for item in fold_metrics if np.isfinite(item["roc_auc"])]
        briers = [float(item["brier"]) for item in fold_metrics if np.isfinite(item["brier"])]
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


def transformed_feature_importance(
    model: Pipeline,
    feature_columns: list[str],
    feature_set: str,
    model_name: str,
) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    names = list(imputer.get_feature_names_out(feature_columns))
    importances = (
        classifier.feature_importances_
        if hasattr(classifier, "feature_importances_")
        else np.abs(classifier.coef_[0])
    )
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


def resolve_config_path(value: str | Path, workspace_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), SCRIPT_DIR, workspace_root):
        path = (base / candidate).resolve()
        if path.exists():
            return path
    return (SCRIPT_DIR / candidate).resolve()


def resolve_workspace_path(workspace_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace_root / path


def append_project_logs(summary: dict[str, Any], out: Path, workspace_root: Path) -> None:
    """Append a run-scoped provenance note without claiming test execution."""
    marker = f"## {summary['run_id']} — BMF5360 baseline package"
    processing = workspace_root / "DATA_PROCESSING_LOG.md"
    prior = processing.read_text(encoding="utf-8") if processing.exists() else "# DATA_PROCESSING_LOG\n"
    if marker not in prior:
        entry = [
            "",
            marker,
            "",
            "- **阶段目的与状态**：交付 technical-only / technical+macro Logistic 与同表 Random Forest 的训练/验证代码包；本次执行记录由 runner 在实际运行时追加，测试期保持封存。",
            f"- **输入与输出**：输入版本见 `{rel(out / 'summary.json', workspace_root)}`；输出目录为 `{rel(out, workspace_root)}`。sealed test target 仅作为未打开的路径描述，不读取、不聚合、不评分。",
            "- **规则**：主样本使用 `non_overlap_selected` H21 anchors；RF 超参数只在 training 内按 `exit_session < validation_start` 的 purged folds 选择；中位数缺失处理和 Logistic 标准化只在各自训练样本拟合；不缩尾、不前向填充、不零填充。",
            f"- **计数与影响**：{json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}；源数据未修改，物理删除行=0，quarantine=none。",
            f"- **检查与限制**：{json.dumps(summary['checks'], ensure_ascii=False, sort_keys=True)}；仅有 validation predictions/metrics，不能据此作测试期性能结论。",
            "",
        ]
        processing.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")

    ai_log = workspace_root / "AI_USE_LOG.md"
    prior_ai = ai_log.read_text(encoding="utf-8") if ai_log.exists() else "# AI_USE_LOG\n"
    ai_marker = f"### BMF5360 baseline package ({summary['run_id']})"
    if ai_marker not in prior_ai:
        ai_entry = [
            "",
            ai_marker,
            "",
            f"- Codex ran the packaged baseline entry point for technical-only and technical-plus-macro Logistic/Random Forest comparisons. Output: `{rel(out, workspace_root)}`.",
            "- The runner reads development targets only, uses training-only purged RF tuning and training-fitted missing-value preprocessing, and does not open the sealed test-target file. Source inputs remain unchanged.",
            "",
        ]
        ai_log.write_text(prior_ai.rstrip("\n") + "\n" + "\n".join(ai_entry), encoding="utf-8", newline="\n")


def run(
    run_id: str | None = None,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    workspace_root: Path | str = DEFAULT_WORKSPACE_ROOT,
    output_root_override: Path | str | None = None,
) -> dict[str, Any]:
    workspace_root = Path(workspace_root).resolve()
    config_path = resolve_config_path(config_path, workspace_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "bmf5360_baseline_package_v1":
        raise RuntimeError("unexpected baseline package config schema")
    horizon = int(config["horizon_sessions"])
    if horizon != HORIZON:
        raise RuntimeError(f"this package is fixed at H21; config says H{horizon}")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not run_id or any(not (character.isascii() and (character.isalnum() or character in "-_")) for character in run_id):
        raise ValueError("run_id must contain only ASCII letters, digits, hyphens, or underscores")
    configured_out_root = output_root_override or config["paths"]["output_root"]
    out_root = resolve_workspace_path(workspace_root, configured_out_root)
    out = out_root / run_id
    if out.exists():
        raise FileExistsError(f"output run already exists: {rel(out, workspace_root)}")

    paths = config["paths"]
    model_ready_id = config["model_ready_run_id"]
    macro_id = config["macro_run_id"]
    model_ready = resolve_workspace_path(workspace_root, paths["model_ready_root"]) / model_ready_id
    macro_dir = resolve_workspace_path(workspace_root, paths["macro_root"]) / macro_id
    model_features_path = model_ready / "model_features.csv"
    eligibility_path = model_ready / "eligibility.csv"
    targets_dev_path = model_ready / "targets_dev.csv"
    macro_path = macro_dir / "macro_features_train_valid.csv"
    model_ready_summary_path = model_ready / "summary.json"
    macro_summary_path = macro_dir / "clean_summary.json"
    sealed_test_targets_path = model_ready / paths.get("sealed_test_targets_name", "targets_test_sealed.csv")
    requirements_path = resolve_config_path(config["requirements_path"], workspace_root)

    # The sealed target path is intentionally a descriptor only.  It is never
    # included in required inputs, hashed, opened, parsed, or checked for data.
    required = [
        model_features_path,
        eligibility_path,
        targets_dev_path,
        macro_path,
        model_ready_summary_path,
        macro_summary_path,
        config_path,
        SCRIPT_PATH,
        requirements_path,
    ]
    missing = [rel(path, workspace_root) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    print(
        "阶段说明：读取 model_features、eligibility、targets_dev 和训练/验证宏观特征；"
        "使用 H21 non-overlap anchors；RF 只在 training purged folds 调参；"
        "仅写 validation predictions/metrics，保持 sealed test targets 封存。",
        flush=True,
    )

    technical = list(config["technical_features"])
    macro_features = list(config["macro_features"])
    excluded_credit = set(config["excluded_credit_spread_features"])
    if excluded_credit & set(macro_features):
        raise RuntimeError("excluded credit-spread features leaked into macro feature list")
    feature_sets = {
        "technical_only": unique_columns(technical),
        "technical_plus_macro": unique_columns(technical + macro_features),
    }
    all_features = unique_columns([item for values in feature_sets.values() for item in values])
    if set(all_features) & TARGET_COLUMNS:
        raise RuntimeError("target columns leaked into feature list")

    input_paths = {
        "model_features": model_features_path,
        "eligibility": eligibility_path,
        "targets_dev": targets_dev_path,
        "model_ready_summary": model_ready_summary_path,
        "macro_features_train_valid": macro_path,
        "macro_summary": macro_summary_path,
        "config": config_path,
        "runner": SCRIPT_PATH,
        "requirements": requirements_path,
    }
    input_hashes_before = {
        name: {"path": rel(path, workspace_root), "sha256": sha256_file(path)}
        for name, path in input_paths.items()
    }

    base_usecols = unique_columns(KEY_COLUMNS + technical)
    features = pd.read_csv(model_features_path, usecols=base_usecols, low_memory=False)
    eligibility = pd.read_csv(
        eligibility_path,
        usecols=[
            "sample_id", "split", "boundary_purged", "entry_trade_eligible",
            "feature_core_available", "non_overlap_selected", "label_complete",
        ],
        low_memory=False,
    )
    targets = pd.read_csv(
        targets_dev_path,
        usecols=[
            "sample_id", "Instrument", "security_id", "formation_session", "split",
            "exit_session", "label_complete", "forward_excess_return", "y",
        ],
        low_memory=False,
    )
    if not set(targets["split"].astype(str)).issubset(DEVELOPMENT_SPLITS):
        raise RuntimeError("targets_dev includes a non-development split")
    for frame, name in [(features, "model_features"), (eligibility, "eligibility"), (targets, "targets_dev")]:
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{name} sample_id invalid")
    if set(features["sample_id"]) != set(eligibility["sample_id"]):
        raise RuntimeError("model feature and eligibility sample sets differ")

    data = features.merge(
        targets,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_target"),
    )
    data = data.merge(
        eligibility,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_eligibility"),
    )
    if not data["split"].astype(str).eq(data["split_target"].astype(str)).all():
        raise RuntimeError("target and feature split labels differ")
    if not data["split"].astype(str).eq(data["split_eligibility"].astype(str)).all():
        raise RuntimeError("eligibility and feature split labels differ")
    data["formation_session"] = pd.to_datetime(
        data["formation_session"], format="mixed", errors="raise"
    ).dt.normalize()
    data["exit_session"] = pd.to_datetime(
        data["exit_session"], format="mixed", errors="raise"
    ).dt.normalize()
    data["split"] = data["split"].astype(str)
    for column in [
        "boundary_purged", "entry_trade_eligible", "feature_core_available",
        "non_overlap_selected", "label_complete", "label_complete_eligibility",
    ]:
        if column in data:
            data[column] = bool_series(data[column], column)
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["forward_excess_return"] = pd.to_numeric(data["forward_excess_return"], errors="coerce")

    macro = pd.read_csv(macro_path, usecols=["spy_session", *macro_features], low_memory=False)
    macro["formation_session"] = pd.to_datetime(
        macro.pop("spy_session"), format="mixed", errors="raise"
    ).dt.normalize()
    if macro["formation_session"].duplicated().any():
        raise RuntimeError("macro session keys are not unique")
    data = data.merge(macro, on="formation_session", how="left", validate="many_to_one")
    for column in all_features:
        if column not in data.columns:
            raise RuntimeError(f"missing declared feature after merge: {column}")
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[data["split"].isin(DEVELOPMENT_SPLITS)].copy()
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
    if train.empty or validation.empty or train["y"].nunique() < 2 or validation["y"].nunique() < 2:
        raise RuntimeError("primary training/validation sample is empty or single-class")
    if train.duplicated(["Instrument", "formation_session"]).any() or validation.duplicated(
        ["Instrument", "formation_session"]
    ).any():
        raise RuntimeError("primary non-overlap anchor keys are duplicated")

    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}
    importance_tables: list[pd.DataFrame] = []
    tuning: dict[str, Any] = {}
    validation_predictions = validation[[
        "sample_id", "security_id", "Instrument", "formation_session", "split",
        "y", "forward_excess_return",
    ]].copy()
    expected_models = {
        f"{feature_set}_{model_name}"
        for feature_set in feature_sets
        for model_name in ("logistic", "random_forest")
    }
    for feature_set, columns in feature_sets.items():
        selected_params, evidence, folds = tune_forest(train, columns, config, horizon)
        tuning[feature_set] = {
            "selected_params": selected_params,
            "evidence": evidence,
            "folds": folds,
        }
        models = {
            "logistic": make_logistic(config),
            "random_forest": make_forest(config, selected_params),
        }
        for model_name, model in models.items():
            model.fit(train[columns], train["y"].astype(int))
            probability = model.predict_proba(validation[columns])[:, 1]
            if not np.isfinite(probability).all():
                raise RuntimeError(f"nonfinite validation probability: {feature_set}/{model_name}")
            metrics_key = f"{feature_set}/{model_name}"
            model_key = f"{feature_set}_{model_name}"
            metrics[metrics_key] = metric_dict(validation, probability)
            validation_predictions[f"p_up_{feature_set}_{model_name}"] = probability
            importance_tables.append(transformed_feature_importance(model, columns, feature_set, model_name))
            fitted[model_key] = model

    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=False)

    validation_predictions.to_csv(out / "validation_predictions.csv", index=False)
    metrics_frame = pd.DataFrame([
        {
            "model_key": key,
            "feature_set": key.split("/", 1)[0],
            "model": key.split("/", 1)[1],
            **value,
        }
        for key, value in metrics.items()
    ])
    metrics_frame.to_csv(out / "validation_metrics.csv", index=False)
    pd.concat(importance_tables, ignore_index=True).to_csv(out / "feature_importance.csv", index=False)
    (out / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "rf_tuning_evidence.json").write_text(
        json.dumps(tuning, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    model_paths: dict[str, str] = {}
    for model_key, model in fitted.items():
        path = models_dir / f"{model_key}.joblib"
        joblib.dump(model, path)
        model_paths[model_key] = rel(path, workspace_root)

    output_files = [
        out / "validation_predictions.csv",
        out / "validation_metrics.csv",
        out / "feature_importance.csv",
        out / "feature_sets.json",
        out / "rf_tuning_evidence.json",
        *[models_dir / f"{key}.joblib" for key in fitted],
    ]
    input_hashes_after = {
        name: sha256_file(path) for name, path in input_paths.items()
    }
    raw_inputs_unchanged = all(
        input_hashes_before[name]["sha256"] == input_hashes_after[name]
        for name in input_paths
    )
    checks = {
        "primary_non_overlap_anchor_sample": bool(
            train["non_overlap_selected"].all() and validation["non_overlap_selected"].all()
        ),
        "primary_anchor_keys_unique": bool(
            not train.duplicated(["Instrument", "formation_session"]).any()
            and not validation.duplicated(["Instrument", "formation_session"]).any()
        ),
        "horizon_is_21_sessions": horizon == HORIZON,
        "rf_tuning_training_only": all(
            item["evidence"]
            and all(
                fold["fit_rows"] > 0
                and fold["validation_rows"] > 0
                and pd.Timestamp(fold["last_fit_exit"]) < pd.Timestamp(fold["validation_start"])
                for candidate in item["evidence"]
                for fold in candidate["folds"]
            )
            for item in tuning.values()
        ),
        "rf_tuning_has_purged_folds": all(
            item["folds"]
            and all(int(fold["purge_sessions"]) == horizon for fold in item["folds"])
            for item in tuning.values()
        ),
        "two_feature_sets_present": list(feature_sets) == ["technical_only", "technical_plus_macro"],
        "four_baseline_models_present": set(fitted) == expected_models,
        "credit_spread_features_excluded": not bool(excluded_credit & set(all_features)),
        "no_target_columns_in_features": not bool(set(all_features) & TARGET_COLUMNS),
        "validation_predictions_only": bool(
            set(validation_predictions["split"].astype(str)) == {"validation"}
            and len(validation_predictions) == len(validation)
        ),
        "test_predictions_zero": True,
        "test_metrics_zero": True,
        "test_targets_not_opened": True,
        "all_probabilities_finite": bool(
            np.isfinite(validation_predictions.filter(like="p_up_").to_numpy(dtype=float)).all()
        ),
        "macro_key_join_complete": bool(
            data["formation_session"].isin(set(macro["formation_session"])).all()
        ),
        "raw_inputs_not_modified_by_runner": raw_inputs_unchanged,
    }
    model_ready_summary = json.loads(model_ready_summary_path.read_text(encoding="utf-8"))
    macro_summary = json.loads(macro_summary_path.read_text(encoding="utf-8"))
    summary = {
        "schema_version": "bmf5360_baseline_package_v1",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": "complete_training_validation_only",
        "scope": "technical-only and technical-plus-macro Logistic/Random Forest comparison; validation only",
        "inputs": {
            "model_ready": {"run_id": model_ready_id, "path": rel(model_ready, workspace_root)},
            "macro": {"run_id": macro_id, "path": rel(macro_dir, workspace_root)},
            "files": {
                name: {"path": item["path"], "sha256": item["sha256"]}
                for name, item in input_hashes_before.items()
            },
            "sealed_test_targets": {
                "path": rel(sealed_test_targets_path, workspace_root),
                "opened": False,
                "sha256": None,
            },
        },
        "outputs": {
            "path": rel(out, workspace_root),
            "model_paths": model_paths,
            "output_hashes": {rel(path, workspace_root): sha256_file(path) for path in output_files},
        },
        "feature_sets": feature_sets,
        "excluded_credit_spread_features": sorted(excluded_credit),
        "sampling": {
            "primary": "non_overlap_selected",
            "horizon_sessions": horizon,
            "training_years": config["training_years"],
            "validation_years": config["validation_years"],
            "description": "No random split. Primary anchors are selected upstream; each RF tuning fold requires training rows with exit_session strictly before the fold validation start.",
        },
        "preprocessing": {
            "imputer": "SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True)",
            "fit_scope": "each RF training fold and final model training anchors only",
            "logistic_scaler": "StandardScaler fit on final training anchors only",
            "winsorization": False,
            "forward_fill": False,
            "zero_fill": False,
            "interpolation": False,
            "raw_input_mutation": False,
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
        "rf_tuning": {
            key: {
                "selected_params": value["selected_params"],
                "fold_count": len(value["folds"]),
                "candidate_count": len(value["evidence"]),
            }
            for key, value in tuning.items()
        },
        "checks": checks,
        "test_policy": config["test_policy"],
        "source_run_status": {
            "model_ready": model_ready_summary.get("status"),
            "macro": macro_summary.get("status"),
        },
        "limitations": [
            "Validation metrics are development comparisons; no test target, prediction, ranking, metric, portfolio or performance claim is produced.",
            "Event rows and overlapping source observations may be dependent; reported metrics are descriptive and do not imply IID significance.",
            "The upstream candidate registry is exploratory_static_candidate for 40 of 49 instruments; that membership limitation remains applicable.",
            "Credit-spread six-column macro candidates are retained upstream but excluded because train/validation coverage is unavailable.",
        ],
    }
    (out / "audit.json").write_text(
        json.dumps(
            {
                "schema_version": "bmf5360_baseline_package_audit_v1",
                "run_id": run_id,
                "checks": checks,
                "test_targets_opened": False,
                "test_predictions_written": False,
                "raw_inputs_modified": not raw_inputs_unchanged,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    append_project_logs(summary, out, workspace_root)
    print(
        json.dumps(
            {"run_id": run_id, "out": rel(out, workspace_root), "status": summary["status"], "checks": checks},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )
    if not all(checks.values()):
        raise RuntimeError("Baseline package checks failed: " + ", ".join(
            key for key, value in checks.items() if not value
        ))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        default=str(DEFAULT_WORKSPACE_ROOT),
        help="Project root containing data/ and project logs (default: package-relative root)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Package JSON config path (default: sibling baseline_package_config.json)",
    )
    parser.add_argument("--run-id", default=None, help="Optional deterministic output directory name")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root, absolute or relative to --workspace-root",
    )
    args = parser.parse_args()
    run(
        run_id=args.run_id,
        config_path=args.config,
        workspace_root=args.workspace_root,
        output_root_override=args.output_root,
    )
