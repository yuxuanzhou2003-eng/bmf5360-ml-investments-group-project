"""Prepare read-only JSON inputs for the submission workbook and report."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z"
REGISTRY = ROOT / "data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv"
CONFIG = ROOT / "ai_pool_daily_state_models_v1_1_config.json"
MACRO_DEFS = ROOT / "data/clean/daily_macro_v1/20260910T062700000000Z/feature_definitions.json"
OUTPUT = Path(__file__).resolve().parent / "delivery_data.json"


def technical_definition(name: str) -> dict:
    if name.startswith("momentum_"):
        horizon = int(name.rsplit("_", 1)[1])
        return {"category": "个股技术", "source": "LSEG adjusted total return", "window": f"{horizon} SPY sessions", "unit": "decimal return", "operation": f"截至F-1的{horizon}日复合总收益", "logic": "刻画短期反转与中期趋势"}
    if name.startswith("volatility_"):
        horizon = int(name.split("_")[1])
        return {"category": "个股风险", "source": "LSEG adjusted total return", "window": f"{horizon} SPY sessions", "unit": "annualized decimal", "operation": f"{horizon}日收益标准差×sqrt(252)", "logic": "控制个股风险状态"}
    if name.startswith("volume_median_"):
        horizon = int(name.split("_")[2])
        return {"category": "个股流动性", "source": "LSEG ACVOL_UNS", "window": f"{horizon} SPY sessions", "unit": "log1p(volume)", "operation": f"先log1p，再取{horizon}日滚动中位数", "logic": "衡量交易活跃度并降低规模偏态"}
    if name.startswith("dollar_volume_median_"):
        horizon = int(name.split("_")[3])
        return {"category": "个股流动性", "source": "LSEG turnover or adjusted close × volume", "window": f"{horizon} SPY sessions", "unit": "log1p(USD)", "operation": f"先log1p，再取{horizon}日滚动中位数", "logic": "衡量资金容量和可交易性"}
    if name.startswith("spread_median_"):
        horizon = int(name.split("_")[2])
        return {"category": "交易成本", "source": "LSEG BID/ASK", "window": f"{horizon} SPY sessions", "unit": "basis points", "operation": f"完整报价价差的{horizon}日中位数", "logic": "控制流动性与潜在交易成本"}
    if name == "beta_126":
        return {"category": "个股风险", "source": "LSEG stock returns + SPY", "window": "126 sessions; min 100", "unit": "beta", "operation": "滚动协方差/市场方差", "logic": "控制市场敏感度"}
    if name == "idio_vol_126_ann":
        return {"category": "个股风险", "source": "LSEG stock returns + SPY", "window": "126 sessions; min 100", "unit": "annualized decimal", "operation": "市场模型残差波动率×sqrt(252)", "logic": "控制公司特有风险"}
    if name == "beta_obs_126":
        return {"category": "数据质量", "source": "LSEG stock returns + SPY", "window": "126 sessions", "unit": "count", "operation": "beta估计的有效共同观察数", "logic": "防止用过短历史估计风险"}
    if name.startswith("spy_momentum_"):
        horizon = int(name.rsplit("_", 1)[1])
        return {"category": "市场状态", "source": "LSEG SPY total return", "window": f"{horizon} SPY sessions", "unit": "decimal return", "operation": f"截至F-1的SPY {horizon}日复合收益", "logic": "控制总体市场趋势"}
    if name.startswith("spy_volatility_"):
        horizon = int(name.split("_")[2])
        return {"category": "市场状态", "source": "LSEG SPY total return", "window": f"{horizon} SPY sessions", "unit": "annualized decimal", "operation": f"SPY {horizon}日收益标准差×sqrt(252)", "logic": "控制市场风险环境"}
    raise KeyError(name)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    macro = json.loads(MACRO_DEFS.read_text(encoding="utf-8"))
    feature_sets = json.loads((RUN / "feature_sets.json").read_text(encoding="utf-8"))
    registry = pd.read_csv(REGISTRY)
    registry["member_from"] = pd.to_datetime(registry["member_from"]).dt.strftime("%Y-%m-%d")
    registry["member_to"] = pd.to_datetime(registry["member_to"]).dt.strftime("%Y-%m-%d")
    registry_rows = registry[[
        "ric", "canonical_name", "primary_group", "inclusion_reason", "member_from", "member_to",
        "delisted_ric", "pit_evidence_status", "pit_eligible_now", "local_keyword_evidence_rows",
        "source_type", "source_title", "source_url", "reason_code", "pit_limitation",
    ]].replace({np.nan: None}).to_dict("records")

    metrics = pd.read_csv(RUN / "validation_metrics.csv").replace({np.nan: None}).to_dict("records")
    raw_features = feature_sets["technical_plus_macro"]
    feature_rows = []
    for position, name in enumerate(raw_features, 1):
        if name in config["technical_features"]:
            definition = technical_definition(name)
        else:
            base = macro["definitions"][name]
            sources = base["source_series"] if isinstance(base["source_series"], list) else [base["source_series"]]
            if name.startswith("vix_"):
                category, logic = "宏观：风险情绪", "控制市场恐慌和风险偏好"
            elif name.startswith("dgs") or name.startswith("term_spread"):
                category, logic = "宏观：利率", "控制贴现率、融资环境和期限结构"
            else:
                category, logic = "宏观：美元", "控制美元与全球金融条件"
            window = "level at F-1"
            if "change_" in name:
                window = name.rsplit("_", 1)[1].replace("d", " SPY sessions")
            elif "252d" in name:
                window = "252 SPY sessions"
            definition = {"category": category, "source": "FRED " + "/".join(sources), "window": window,
                          "unit": base["units"], "operation": base["operation"], "logic": logic}
        feature_rows.append({"order": position, "feature": name, "included": True, **definition,
                             "timing": "formation session F uses information available through F-1"})

    for name in config["excluded_credit_spread_features"]:
        base = macro["definitions"][name]
        feature_rows.append({
            "order": len(feature_rows) + 1, "feature": name, "included": False,
            "category": "宏观：信用利差", "source": "FRED " + str(base["source_series"]),
            "window": "not used", "unit": base["units"], "operation": base["operation"],
            "logic": "经济含义相关，但开发期覆盖不可用，因此保留记录并排除",
            "timing": macro["credit_spread_exclusion_reason"],
        })

    model = joblib.load(RUN / "models/technical_plus_macro_logistic.joblib")
    transformed = list(model.named_steps["imputer"].get_feature_names_out(raw_features))
    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_rows = []
    for rank, (name, coefficient) in enumerate(sorted(zip(transformed, coefficients), key=lambda x: abs(x[1]), reverse=True), 1):
        base = name.replace("missingindicator_", "")
        coefficient_rows.append({
            "absolute_rank": rank, "transformed_feature": name, "base_feature": base,
            "missing_indicator": name.startswith("missingindicator_"),
            "coefficient_standardized": float(coefficient), "absolute_coefficient": float(abs(coefficient)),
            "odds_multiplier_per_1sd": float(np.exp(coefficient)),
            "direction": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
        })

    group_descriptions = {
        "gpu_accelerator": "GPU、CPU、定制加速器与加速计算平台",
        "ai_semiconductor": "晶圆、EDA、模拟/功率/连接芯片与半导体设备",
        "memory_hbm_storage": "HBM、DRAM、NAND、硬盘与企业存储",
        "server_network": "AI服务器、交换机、光互联和数据基础设施",
        "cloud_software": "云平台、数据库、企业软件和AI开发/部署工具",
        "data_center_power_cooling": "数据中心供电、制冷、机房与基础设施",
        "robotics_autonomy": "工业自动化、机器人、视觉与自动驾驶",
    }
    group_counts = registry.groupby("primary_group").agg(
        companies=("ric", "nunique"),
        direct_local_pit=("pit_evidence_status", lambda s: int((s == "direct_local_pit").sum())),
        provisional_static=("pit_evidence_status", lambda s: int((s == "provisional_static_source").sum())),
    ).reset_index()
    group_rows = []
    for row in group_counts.to_dict("records"):
        row["description"] = group_descriptions[row["primary_group"]]
        group_rows.append(row)

    selection_steps = [
        {"step": 1, "rule": "Define scope before looking at returns", "implementation": "AI supply chain only; seven economic links from compute to deployment and physical infrastructure", "rationale": "Keeps the investment universe economically coherent", "bias_control": "No return screen is used to add a company"},
        {"step": 2, "rule": "Literal RIC must exist in the 781-RIC LSEG mother universe", "implementation": "49/49 candidate RICs reconcile to the existing universe", "rationale": "Provides price, return, quote and identifier lineage", "bias_control": "No ticker/name guessing"},
        {"step": 3, "rule": "Assign exactly one primary group", "implementation": "Secondary roles remain notes; caps use the primary group only", "rationale": "Prevents double counting in sector constraints", "bias_control": "Group does not change after seeing model output"},
        {"step": 4, "rule": "Preserve membership spans", "implementation": "member_from/member_to determine date-level eligibility", "rationale": "Avoids treating pre-listing dates as genuine missing observations", "bias_control": "Five later entrants remain recorded rather than deleted"},
        {"step": 5, "rule": "Retain delisted securities", "implementation": "JNPR.N^G25 and its history remain in registry/data", "rationale": "Reduces survivorship bias", "bias_control": "Delisting alone is never an exclusion reason"},
        {"step": 6, "rule": "Separate historical PIT evidence from static role evidence", "implementation": "9 direct_local_pit; 40 provisional_static_source", "rationale": "Makes the evidence quality visible", "bias_control": "Static descriptions are not presented as historical proof"},
    ]

    processing_rows = [
        {"stage": "Universe", "input": "781-RIC corrected membership spans", "rule": "49-RIC AI taxonomy; no return screen; keep delisted", "before": "781 RIC", "after": "49 registered RIC", "missing_or_excluded": "40 roles provisional-static; 5 have no 2021-22 active span", "status": "audited"},
        {"stage": "Prices", "input": "LSEG clean v3", "rule": "Adjusted close/return/volume/quotes retained; no second split adjustment", "before": "2,084,177 rows", "after": "source preserved", "missing_or_excluded": "Missing remains NA", "status": "31/31 passed"},
        {"stage": "Technical features", "input": "Clean prices and SPY sessions", "rule": "Rolling values end on F-1", "before": "daily observations", "after": "21 included variables", "missing_or_excluded": "Windows without enough history remain NA", "status": "audited"},
        {"stage": "Macro raw", "input": "FRED 7 series", "rule": "Retain raw observations and source dates", "before": "18,345 rows", "after": "18,345 rows", "missing_or_excluded": "Provider missing tokens remain flagged", "status": "75/75 passed"},
        {"stage": "Macro alignment", "input": "FRED raw + SPY calendar", "rule": "For F, use latest publication available by F-1; no future carry", "before": "7 source series", "after": "3,228 SPY sessions", "missing_or_excluded": "Weekend/holiday carry records age/source date", "status": "158/158 passed"},
        {"stage": "Credit spreads", "input": "HY/IG OAS", "rule": "Exclude from development model; do not impute", "before": "6 candidate variables", "after": "0 included", "missing_or_excluded": "First valid source date 2023-09-11", "status": "reason-coded exclusion"},
        {"stage": "Model-ready", "input": "49-RIC daily panel", "rule": "H21 label and non-overlap anchors; preserve master rows", "before": "110,829 company-days", "after": "2,334 train + 992 validation anchors", "missing_or_excluded": "39 train / 44 validation instruments", "status": "audited"},
        {"stage": "Missing values", "input": "41 model features", "rule": "Training-only median + missing indicators inside pipeline", "before": "NA preserved in source", "after": "42 transformed columns", "missing_or_excluded": "No zero/forward fill/winsorization", "status": "model pipeline"},
        {"stage": "Validation", "input": "2021-2022 anchors", "rule": "No random split; no threshold tuning on validation", "before": "992 rows", "after": "992 predictions", "missing_or_excluded": "0 test predictions / 0 test metrics", "status": "74/74 model audit passed"},
    ]

    result = {
        "registry": registry_rows,
        "group_summary": group_rows,
        "selection_steps": selection_steps,
        "features": feature_rows,
        "coefficients": coefficient_rows,
        "metrics": metrics,
        "model": {
            "name": "technical_plus_macro/logistic",
            "target": "future 21-session stock total return minus SPY total return > 0",
            "intercept": float(model.named_steps["classifier"].intercept_[0]),
            "raw_feature_count": len(raw_features),
            "transformed_feature_count": len(transformed),
            "params": config["logistic"],
            "pipeline": "SimpleImputer(median, add_indicator, keep_empty_features) -> StandardScaler -> LogisticRegression",
        },
        "sample": {
            "master_company_days": 110829, "registry_rics": 49,
            "train_years": "2015-2020", "train_rows": 2334, "train_instruments": 39, "train_dates": 444,
            "validation_years": "2021-2022", "validation_rows": 992, "validation_instruments": 44, "validation_dates": 89,
            "test_years": "2023-01-01 to 2026-06-30", "test_status": "SEALED; 0 predictions; 0 metrics",
            "label_horizon": 21, "validation_positive_rate": 0.5070564516129032,
        },
        "processing": processing_rows,
        "sources": [
            {"artifact": "Candidate registry", "path": str(REGISTRY.relative_to(ROOT)).replace("\\", "/")},
            {"artifact": "Model configuration", "path": str(CONFIG.relative_to(ROOT)).replace("\\", "/")},
            {"artifact": "Feature sets", "path": str((RUN / "feature_sets.json").relative_to(ROOT)).replace("\\", "/")},
            {"artifact": "Validation metrics", "path": str((RUN / "validation_metrics.csv").relative_to(ROOT)).replace("\\", "/")},
            {"artifact": "Stored Logistic model", "path": str((RUN / "models/technical_plus_macro_logistic.joblib").relative_to(ROOT)).replace("\\", "/")},
            {"artifact": "Macro definitions", "path": str(MACRO_DEFS.relative_to(ROOT)).replace("\\", "/")},
            {"artifact": "Full processing log", "path": "DATA_PROCESSING_LOG.md"},
            {"artifact": "AI-use log", "path": "AI_USE_LOG.md"},
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": {k: len(result[k]) for k in ["registry", "group_summary", "selection_steps", "features", "coefficients", "metrics", "processing", "sources"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
