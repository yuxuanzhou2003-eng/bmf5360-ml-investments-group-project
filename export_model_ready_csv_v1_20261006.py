"""Export the independently verified full PIT panel to auditable CSV files."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/processed/all_sector_full_daily_pit_merge_v1_20261005/7b97d11680cb47ca"
MACRO_SOURCE = ROOT / "data/clean/daily_macro_v1/20260910T062700000000Z/macro_features.csv"
UNIVERSE_SOURCE = ROOT / "data/processed/full_v2/full_u28/m.csv"
PM_SOURCE = ROOT / "data/processed/full_v2/full_u28/pm.csv"
CA_AUDIT = ROOT / "data/audit/full_universe_corporate_actions_v1/full5214_ssp_scn_20261005_01"
OUTPUT = ROOT / "deliverables/All_Sector_Model_Ready_CSV_20261006/v1"
STAGING = OUTPUT.with_name(OUTPUT.name + ".staging")

TECHNICAL = [
    "ret_1", "ret_5", "ret_20", "ret_60", "ret_120", "vol_20", "vol_60",
    "downside_20", "downside_60", "upshare_20", "momentum_20_minus_60",
    "ret_10", "ret_40", "ret_250", "momentum_60_excluding_5", "vol_5",
    "vol_120", "vol_ratio_20_60", "skew_20", "skew_60", "max_drawdown_20",
    "max_drawdown_60", "max_drawdown_120", "current_drawdown_60",
    "current_drawdown_120", "upshare_60", "return_q05_60", "max_abs_return_20",
    "gain_loss_ratio_20",
]
STRICT_LIQUIDITY = [
    "turnover_log1p", "turnover_change_1d", "turnover_z20", "turnover_median_20", "amihud_20",
]
EXCLUDED_SHARE_VOLUME = ["volume_log1p", "volume_change_1d", "volume_z20"]
ANNUAL = [
    "annual_assets", "annual_common_equity", "annual_debt", "annual_gross_profit",
    "annual_net_income", "annual_operating_cash_flow", "annual_revenue", "debt_assets",
    "gross_margin", "net_margin", "ni_assets", "ocf_assets", "revenue_yoy", "roe",
]
VALUATION_MODEL = ["log_market_cap", "earnings_yield", "book_yield", "dividend_yield"]
VALUATION_AUDIT = ["market_cap", "pe", "pb"]
VALUATION = VALUATION_AUDIT + VALUATION_MODEL
STRICT_NUMERIC = TECHNICAL + STRICT_LIQUIDITY + ANNUAL + VALUATION_MODEL
STRICT_MARKET = TECHNICAL + STRICT_LIQUIDITY

BASE_COLUMNS = [
    "source_input_position", "Instrument", "history_ric", "date", "research_sector", "sector_code",
    "partition", "entry_date", "label_end_date", "return_1d", "price", "volume",
    "volume_vendor_default", "turnover", "accumulated_volume_adjusted_0",
]
TIMING_AND_TARGET = [
    "date_in_fixed_calendar", "label_future_complete", "label_within_partition",
    "source_feature_available_37", "history_scope_eligible", "sector_available", "label_eligible",
    "strict_market_feature_available_34", "score_eligible", "supervised_eligible",
    "pit_feature_available_m1", "m1_supervised_eligible", "annual_nonmissing_count",
    "valuation_nonmissing_count", "strict_predictor_nonmissing_count", "abnormal_return_flag", "valuation_exact_match",
    "market_cap_calcdate", "pe_calcdate", "pb_calcdate", "dividend_yield_calcdate",
    "target_return_5d", "target_up",
]
OUTPUT_COLUMNS = BASE_COLUMNS + TECHNICAL + STRICT_LIQUIDITY + ANNUAL + VALUATION + TIMING_AND_TARGET
PARTITIONS = {
    "warmup": ("model_ready_warmup_2020.csv.gz", 253, "2020-01-02", "2020-12-31"),
    "train": ("model_ready_train_2021_2023.csv.gz", 753, "2021-01-04", "2023-12-29"),
    "validation": ("model_ready_validation_2024_2025.csv.gz", 502, "2024-01-02", "2025-12-31"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def finite_all(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)


def build_feature_dictionary(source_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in OUTPUT_COLUMNS + EXCLUDED_SHARE_VOLUME:
        if column in TECHNICAL:
            group, role, included = "technical_total_return", "predictor", True
            source, timing = "TR.TotalReturn rolling history", "available at formation close; downstream execution starts t+1"
        elif column in STRICT_LIQUIDITY:
            group, role, included = "liquidity_dollar_turnover", "predictor", True
            source, timing = "TR.Turnover and TR.TotalReturn", "available at formation close"
        elif column in EXCLUDED_SHARE_VOLUME:
            group, role, included = "share_volume_excluded", "excluded_predictor", False
            source, timing = "TR.AccumulatedVolume(Adjusted=0)", "excluded until split-adjusted volume is independently verified"
        elif column in ANNUAL:
            group, role, included = "annual_pit_fundamental", "predictor", True
            source, timing = "annual LSEG fundamentals", "effective on first frozen session after announcement; expires 550 days after period end"
        elif column in VALUATION_MODEL:
            group, role, included = "daily_valuation", "predictor", True
            source, timing = "LSEG valuation", "exact effective-date match; no forward fill"
        elif column in VALUATION_AUDIT:
            group, role, included = "daily_valuation_raw", "audit", False
            source, timing = "LSEG valuation", "raw level retained for audit; not a default predictor"
        elif column == "research_sector":
            group, role, included = "classification", "categorical_predictor", True
            source, timing = "static current TRBC snapshot", "not historical PIT classification; limitation retained"
        elif column in {"target_return_5d", "target_up"}:
            group, role, included = "target", "target", False
            source, timing = "TR.TotalReturn t+2 through t+6", "boundary-purged; never a predictor"
        elif column.endswith("available_34") or column.endswith("eligible") or column.endswith("count") or column in {"date_in_fixed_calendar", "label_future_complete", "label_within_partition", "source_feature_available_37", "abnormal_return_flag", "valuation_exact_match", "sector_available"}:
            group, role, included = "audit_mask", "audit", False
            source, timing = "derived in frozen export or source panel", "not a default predictor"
        elif column in BASE_COLUMNS or column.endswith("calcdate"):
            group, role, included = "identifier_raw_or_provenance", "identifier_or_audit", False
            source, timing = "source panel", "retained for audit/execution; not a default predictor"
        else:
            group, role, included = "other", "audit", False
            source, timing = "source panel", "review before use"
        rows.append({
            "column": column,
            "present_in_partition_csv": column in OUTPUT_COLUMNS,
            "group": group,
            "role": role,
            "included_in_strict_main_predictors": included,
            "source": source,
            "timing_or_reason": timing,
            "missing_policy": "empty means missing; genuine numeric zero remains zero; no imputation in CSV export",
            "source_panel_column_present": column in source_columns,
        })
    return pd.DataFrame(rows).drop_duplicates("column")


def build_universe() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_SOURCE, dtype=str, keep_default_na=False)
    pm = pd.read_csv(PM_SOURCE, dtype=str, keep_default_na=False)
    pm_keep = [column for column in pm.columns if column not in universe.columns or column == "source_input_position"]
    universe = universe.merge(pm[pm_keep], on="source_input_position", how="left", validate="one_to_one")
    coverage = pd.read_csv(CA_AUDIT / "coverage.csv", dtype=str, keep_default_na=False)
    status = coverage.pivot(index="source_input_position", columns="request_event_type", values="coverage_status")
    status.columns = [f"corporate_action_{column.lower()}_status" for column in status.columns]
    counts = coverage.pivot(index="source_input_position", columns="request_event_type", values="event_rows_for_ric")
    counts.columns = [f"corporate_action_{column.lower()}_physical_event_rows" for column in counts.columns]
    joined = status.join(counts).reset_index()
    universe = universe.merge(joined, on="source_input_position", how="left", validate="one_to_one")
    return universe.sort_values("source_input_position", key=lambda x: pd.to_numeric(x, errors="coerce"))


def export() -> dict[str, Any]:
    if OUTPUT.exists() or STAGING.exists():
        raise RuntimeError("model-ready CSV output already exists")
    manifest = pd.read_parquet(SOURCE / "panel_manifest.parquet").sort_values("source_input_position").reset_index(drop=True)
    if len(manifest) != 5214 or manifest["source_input_position"].duplicated().any():
        raise RuntimeError("source panel manifest is not the frozen 5,214-company set")
    pm_scope = pd.read_csv(
        PM_SOURCE,
        usecols=["source_input_position", "history_scope_eligible"],
        dtype=str,
        keep_default_na=False,
    )
    pm_scope["source_input_position"] = pd.to_numeric(pm_scope["source_input_position"], errors="raise").astype(int)
    if len(pm_scope) != 5214 or pm_scope["source_input_position"].duplicated().any():
        raise RuntimeError("history-scope source is not the frozen 5,214-company set")
    history_scope_map = dict(
        zip(
            pm_scope["source_input_position"],
            pm_scope["history_scope_eligible"].str.strip().str.lower().eq("true"),
        )
    )
    first = pd.read_parquet(SOURCE / manifest.iloc[0]["output_panel_file"])
    source_columns = list(first.columns)
    required = set(OUTPUT_COLUMNS) - {"source_feature_available_37", "history_scope_eligible", "sector_available", "strict_market_feature_available_34", "score_eligible", "supervised_eligible", "pit_feature_available_m1", "m1_supervised_eligible", "annual_nonmissing_count", "valuation_nonmissing_count", "strict_predictor_nonmissing_count"}
    required.add("feature_available")
    missing = sorted(required - set(source_columns))
    if missing:
        raise RuntimeError(f"source panel missing columns: {missing}")
    STAGING.mkdir(parents=True)
    handles = {}
    counts = {name: {"rows": 0, "supervised_eligible": 0, "label_eligible": 0, "instruments": 0} for name in PARTITIONS}
    try:
        for name, (filename, _, _, _) in PARTITIONS.items():
            handles[name] = gzip.open(STAGING / filename, "wt", encoding="utf-8", newline="", compresslevel=1)
        for index, row in manifest.iterrows():
            frame = pd.read_parquet(SOURCE / row["output_panel_file"])
            if len(frame) != 1508 or frame["source_input_position"].nunique() != 1:
                raise RuntimeError(f"invalid source panel grid: {row['output_panel_file']}")
            frame = frame.rename(columns={"feature_available": "source_feature_available_37"})
            source_position = int(row["source_input_position"])
            if source_position not in history_scope_map:
                raise RuntimeError(f"missing history-scope eligibility for source position {source_position}")
            history_scope_eligible = bool(history_scope_map[source_position])
            frame["history_scope_eligible"] = history_scope_eligible
            sector_text = frame["research_sector"].fillna("").astype(str).str.strip()
            frame["sector_available"] = sector_text.ne("")
            frame["strict_market_feature_available_34"] = finite_all(frame, STRICT_MARKET)
            frame["score_eligible"] = frame["date_in_fixed_calendar"].astype(bool) & frame["history_scope_eligible"] & frame["sector_available"] & frame["strict_market_feature_available_34"]
            target_valid = np.isfinite(frame["target_return_5d"].to_numpy(float)) & frame["target_up"].isin([0.0, 1.0]).to_numpy()
            frame["supervised_eligible"] = frame["score_eligible"] & frame["label_future_complete"].astype(bool) & frame["label_within_partition"].astype(bool) & frame["label_eligible"].astype(bool) & target_valid
            frame["pit_feature_available_m1"] = finite_all(frame, ANNUAL + VALUATION_MODEL)
            frame["m1_supervised_eligible"] = frame["supervised_eligible"] & frame["pit_feature_available_m1"]
            frame["annual_nonmissing_count"] = frame[ANNUAL].notna().sum(axis=1).astype("int16")
            frame["valuation_nonmissing_count"] = frame[VALUATION].notna().sum(axis=1).astype("int8")
            frame["strict_predictor_nonmissing_count"] = frame[STRICT_NUMERIC].notna().sum(axis=1).astype("int16")
            frame = frame[OUTPUT_COLUMNS]
            if np.isinf(frame.select_dtypes(include=[np.number]).to_numpy()).any():
                raise RuntimeError(f"infinite value in source position {row['source_input_position']}")
            for name in PARTITIONS:
                part = frame.loc[frame["partition"].eq(name)]
                if len(part) != PARTITIONS[name][1]:
                    raise RuntimeError(f"partition row mismatch for source position {row['source_input_position']}: {name}")
                part.to_csv(handles[name], index=False, header=index == 0, na_rep="", quoting=csv.QUOTE_MINIMAL)
                counts[name]["rows"] += len(part)
                counts[name]["supervised_eligible"] += int(part["supervised_eligible"].sum())
                counts[name]["label_eligible"] += int(part["label_eligible"].sum())
                counts[name]["instruments"] += 1
            if (index + 1) % 250 == 0:
                print(f"exported {index + 1}/{len(manifest)} instruments", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    macro = pd.read_csv(MACRO_SOURCE)
    macro["spy_session"] = pd.to_datetime(macro["spy_session"], errors="raise")
    macro = macro.loc[macro["spy_session"].between("2020-01-02", "2025-12-31")].copy()
    macro["spy_session"] = macro["spy_session"].dt.strftime("%Y-%m-%d")
    macro.insert(3, "partition", np.select([pd.to_datetime(macro["spy_session"]).dt.year.eq(2020), pd.to_datetime(macro["spy_session"]).dt.year.le(2023)], ["warmup", "train"], default="validation"))
    macro.insert(4, "sensitivity_only", True)
    macro.insert(5, "strict_pit_eligible", False)
    macro.to_csv(STAGING / "macro_current_vintage_sensitivity_2020_2025.csv", index=False, encoding="utf-8-sig", na_rep="")

    universe = build_universe()
    universe.to_csv(STAGING / "universe_5214.csv", index=False, encoding="utf-8-sig", na_rep="", quoting=csv.QUOTE_MINIMAL)
    dictionary = build_feature_dictionary(source_columns)
    dictionary.to_csv(STAGING / "feature_dictionary.csv", index=False, encoding="utf-8-sig", na_rep="")

    expected = {name: 5214 * spec[1] for name, spec in PARTITIONS.items()}
    for name in PARTITIONS:
        if counts[name]["rows"] != expected[name] or counts[name]["instruments"] != 5214:
            raise RuntimeError(f"export count mismatch: {name}")
    if len(macro) != 1508 or macro["spy_session"].duplicated().any():
        raise RuntimeError("macro sensitivity output must have 1,508 unique sessions")
    if len(universe) != 5214 or universe["source_input_position"].duplicated().any():
        raise RuntimeError("universe output must have 5,214 unique positions")

    outputs = []
    for path in sorted(STAGING.glob("*.csv*")):
        outputs.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
    receipt = {
        "schema": "all_sector_model_ready_csv_v1",
        "status": "exported_pending_independent_reread_validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_panel": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_panel_manifest_sha256": sha(SOURCE / "panel_manifest.parquet"),
        "source_summary_sha256": sha(SOURCE / "summary.json"),
        "partition_counts": counts,
        "expected_partition_rows": expected,
        "output_columns": OUTPUT_COLUMNS,
        "m0_strict_predictor_columns": TECHNICAL + STRICT_LIQUIDITY + ["research_sector"],
        "m1_pit_increment_columns": ANNUAL + VALUATION_MODEL,
        "excluded_share_volume_features": EXCLUDED_SHARE_VOLUME,
        "macro_policy": "separate current-vintage sensitivity sidecar; never joined to strict main CSV in this stage",
        "missing_policy": "empty CSV field means missing; numeric zero retained; no imputation, forward fill, winsorization, row deletion, or company exclusion",
        "test_policy": "no 2026 input, target, prediction, model result, or test selection accessed",
        "outputs": outputs,
        "script_sha256": sha(Path(__file__)),
    }
    write_json(STAGING / "manifest.json", receipt)
    readme = """# All-sector model-ready CSV v1\n\nThis package freezes the verified 5,214-company panel before model fitting. The three large files are gzip-compressed CSVs and retain every company-date grid row. Empty fields are missing values, not zero. No imputation, winsorization, company removal, model fit, or backtest occurs here.\n\n- `model_ready_warmup_2020.csv.gz`: rolling-feature warm-up only.\n- `model_ready_train_2021_2023.csv.gz`: training partition.\n- `model_ready_validation_2024_2025.csv.gz`: validation partition.\n- `macro_current_vintage_sensitivity_2020_2025.csv`: separate sensitivity input; not strict PIT.\n- `universe_5214.csv`: security, identity, venue, history, and SSP/SCN coverage evidence.\n- `feature_dictionary.csv`: predictor, target, audit, provenance, and excluded-feature roles.\n\nThe strict main predictor list excludes `volume_log1p`, `volume_change_1d`, and `volume_z20`. Raw share volume remains for audit and later execution research. `supervised_eligible` requires an eligible label and all 34 strict market features, but it does not require complete fundamental or valuation coverage. Any model imputation must be fit inside training folds and documented separately. No 2026 data is included.\n"""
    (STAGING / "README.md").write_text(readme, encoding="utf-8")
    os.replace(STAGING, OUTPUT)
    return receipt


if __name__ == "__main__":
    print(json.dumps(export(), ensure_ascii=False))
