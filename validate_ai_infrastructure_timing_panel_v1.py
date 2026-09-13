"""Independent validation for the development-only AI Infrastructure timing panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-run", required=True)
    parser.add_argument("--config", default="ai_infrastructure_timing_v1_config.json")
    args = parser.parse_args()
    run = Path(args.panel_run)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    panel = pd.read_csv(run / "timing_panel_dev.csv", parse_dates=["Date"])
    coverage = pd.read_csv(run / "sleeve_coverage.csv", parse_dates=["Date"])
    candidates = pd.read_csv(run / "candidate_registry_core.csv")
    features = cfg["features"]
    horizon = int(cfg["horizon_sessions"])
    checks: dict[str, bool] = {}
    development_end = pd.Timestamp(cfg.get("development_end_date", cfg.get("end_date")))
    checks["no_test_period_rows"] = bool(panel["Date"].max() <= development_end)
    checks["only_core_roles"] = set(candidates["primary_group"]) == set(cfg["core_roles"])
    checks["one_coverage_row_per_date_role"] = not coverage.duplicated(["Date", "primary_group"]).any()
    checks["all_dates_have_configured_roles"] = bool(coverage.groupby("Date")["primary_group"].nunique().eq(len(cfg["core_roles"])).all())
    expected_basket = coverage.groupby("Date")["sleeve_return"].mean().reindex(panel["Date"]).to_numpy()
    expected_basket[~panel["all_sleeves_available"].to_numpy()] = np.nan
    checks["basket_equal_sleeve_weight_matches"] = bool(np.allclose(panel["ai_return_1"].to_numpy(), expected_basket, equal_nan=True, atol=1e-12))
    expected_ai = (1.0 + panel["ai_return_1"]).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    expected_spy = (1.0 + panel["spy_return_1"]).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    checks["forward_target_alignment"] = bool(np.allclose(panel["forward_ai_return"], expected_ai, equal_nan=True, atol=1e-12) and np.allclose(panel["forward_spy_return"], expected_spy, equal_nan=True, atol=1e-12))
    expected_y = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    available = panel["forward_excess_return"].notna()
    expected_y.loc[available] = (panel.loc[available, "forward_excess_return"] > 0).astype(int)
    checks["binary_target_matches_excess_return"] = panel["y"].astype("Int64").equals(expected_y)
    checks["eligible_rule_matches"] = bool((panel["model_eligible"] == (panel["y"].notna() & panel["feature_missing_count"].eq(0) & panel["all_sleeves_available"])).all())
    checks["feature_list_complete"] = set(features).issubset(panel.columns)
    checks["no_inf_model_features"] = bool(np.isfinite(panel[features].to_numpy(dtype=float)[~panel[features].isna().to_numpy()]).all())
    result = {
        "schema_version": "validate_ai_infrastructure_timing_panel_v1",
        "panel_run": str(run), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "counts": {"panel_sessions": int(len(panel)), "candidate_core_count": int(len(candidates)), "model_eligible_sessions": int(panel["model_eligible"].sum())},
        "scope": "development-only validation; no test target is read or referenced",
    }
    audit = run.parent.parent / "audit" / "ai_infrastructure_timing_panel_v1" / run.name
    audit.mkdir(parents=True, exist_ok=False)
    (audit / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
