"""Build a development-only daily panel for the AI Infrastructure Core timing study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compound_return(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0


def rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    def window_drawdown(values: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + values)
        return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))

    return series.rolling(window, min_periods=window).apply(window_drawdown, raw=True)


def build(args: argparse.Namespace) -> Path:
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(args.output_root) / args.run_id
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    (out / "quarantine").mkdir()

    registry_path = Path(cfg["candidate_registry_path"])
    returns_path = Path(cfg["returns_path"])
    macro_path = Path(cfg["macro_path"])
    registry = pd.read_csv(registry_path, parse_dates=["member_from", "member_to"])
    returns = pd.read_csv(returns_path, parse_dates=["Date"])
    macro = pd.read_csv(macro_path, parse_dates=["spy_session"])
    start = pd.Timestamp(cfg.get("development_start_date", cfg.get("start_date")))
    end = pd.Timestamp(cfg.get("development_end_date", cfg.get("end_date")))
    roles = cfg["core_roles"]

    candidates = registry.loc[registry["primary_group"].isin(roles)].copy()
    candidates = candidates[["ric", "canonical_name", "primary_group", "member_from", "member_to", "pit_evidence_status", "delisted_ric"]]
    candidates = candidates.sort_values(["primary_group", "ric"]).reset_index(drop=True)
    candidates.to_csv(out / "candidate_registry_core.csv", index=False)

    base = returns.loc[(returns["Date"] >= start) & (returns["Date"] <= end), ["Instrument", "Date", "return_decimal"]].copy()
    base["return_decimal"] = pd.to_numeric(base["return_decimal"], errors="coerce")
    spy = base.loc[base["Instrument"].eq(cfg["benchmark_ric"]), ["Date", "return_decimal"]].drop_duplicates("Date")
    spy = spy.rename(columns={"return_decimal": "spy_return_1"}).sort_values("Date")
    sessions = spy[["Date"]].copy()
    if sessions.empty:
        raise ValueError("No benchmark sessions in configured development period")

    observed = base.loc[base["Instrument"].isin(candidates["ric"]), ["Instrument", "Date", "return_decimal"]]
    candidate_days = sessions.assign(_key=1).merge(candidates.assign(_key=1), on="_key", how="inner").drop(columns="_key")
    candidate_days["active_by_registry"] = candidate_days["Date"].between(candidate_days["member_from"], candidate_days["member_to"], inclusive="both")
    candidate_days = candidate_days.merge(observed, left_on=["ric", "Date"], right_on=["Instrument", "Date"], how="left").drop(columns="Instrument")
    candidate_days["valid_return"] = candidate_days["active_by_registry"] & candidate_days["return_decimal"].notna() & np.isfinite(candidate_days["return_decimal"])
    candidate_days["reason_code"] = np.select(
        [~candidate_days["active_by_registry"], candidate_days["return_decimal"].isna(), ~np.isfinite(candidate_days["return_decimal"].fillna(0))],
        ["OUTSIDE_REGISTRY_MEMBER_SPAN", "NO_RETURN_OBSERVATION", "NONFINITE_RETURN"],
        default="VALID_RETURN",
    )
    candidate_days.to_csv(out / "candidate_day_coverage.csv", index=False)

    valid = candidate_days.loc[candidate_days["valid_return"]].copy()
    sleeve = valid.groupby(["Date", "primary_group"], as_index=False).agg(
        sleeve_return=("return_decimal", "mean"),
        valid_member_count=("ric", "nunique"),
    )
    coverage = sessions.assign(_key=1).merge(pd.DataFrame({"primary_group": roles, "_key": 1}), on="_key").drop(columns="_key")
    coverage = coverage.merge(sleeve, on=["Date", "primary_group"], how="left")
    coverage["sleeve_available"] = coverage["sleeve_return"].notna()
    coverage["reason_code"] = np.where(coverage["sleeve_available"], "VALID_SLEEVE_EQUAL_WEIGHT", "NO_VALID_MEMBER_RETURN_IN_SLEEVE")
    coverage.to_csv(out / "sleeve_coverage.csv", index=False)

    basket = coverage.groupby("Date", as_index=False).agg(
        available_sleeves=("sleeve_available", "sum"),
        ai_return_1=("sleeve_return", "mean"),
        min_sleeve_member_count=("valid_member_count", "min"),
        total_valid_member_count=("valid_member_count", "sum"),
    )
    basket["all_sleeves_available"] = basket["available_sleeves"].eq(len(roles))
    basket.loc[~basket["all_sleeves_available"], "ai_return_1"] = np.nan
    basket["basket_reason_code"] = np.where(basket["all_sleeves_available"], "VALID_SLEEVE_EQUAL_WEIGHT_BASKET", "ONE_OR_MORE_SLEEVES_UNAVAILABLE")
    panel = sessions.merge(basket, on="Date", how="left").merge(spy, on="Date", how="left")
    panel["ai_excess_return_1"] = panel["ai_return_1"] - panel["spy_return_1"]

    # Basket features use data through the formation-session close only.
    panel["ai_excess_momentum_5"] = compound_return(panel["ai_excess_return_1"], 5)
    panel["ai_excess_momentum_20"] = compound_return(panel["ai_excess_return_1"], 20)
    panel["ai_excess_momentum_60"] = compound_return(panel["ai_excess_return_1"], 60)
    panel["ai_volatility_20_ann"] = panel["ai_return_1"].rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    panel["ai_max_drawdown_60"] = rolling_drawdown(panel["ai_return_1"], 60)
    panel["spy_momentum_20"] = compound_return(panel["spy_return_1"], 20)
    panel["spy_volatility_20_ann"] = panel["spy_return_1"].rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)

    # Participation features are descriptive breadth measures, not stock rankings.
    pivot = valid.pivot(index="Date", columns="ric", values="return_decimal").reindex(sessions["Date"])
    for window in (5, 20, 60):
        per_name = (1.0 + pivot).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        panel[f"ai_breadth_positive_{window}"] = per_name.gt(0).mean(axis=1, skipna=True).to_numpy()
    panel["ai_cross_sectional_dispersion_1"] = pivot.std(axis=1, ddof=1, skipna=True).to_numpy()

    macro_columns = ["spy_session", "vix_level", "vix_change_5d", "term_spread_10y_2y", "broad_dollar_change_20d"]
    panel = panel.merge(macro[macro_columns], left_on="Date", right_on="spy_session", how="left").drop(columns="spy_session")

    horizon = int(cfg["horizon_sessions"])
    panel["forward_ai_return"] = (1.0 + panel["ai_return_1"]).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    panel["forward_spy_return"] = (1.0 + panel["spy_return_1"]).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    panel["forward_excess_return"] = panel["forward_ai_return"] - panel["forward_spy_return"]
    panel["y"] = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    labelled = panel["forward_excess_return"].notna()
    panel.loc[labelled, "y"] = (panel.loc[labelled, "forward_excess_return"] > 0).astype(int)
    panel["split"] = np.select(
        [panel["Date"] <= pd.Timestamp(cfg["train_end"]), panel["Date"].between(pd.Timestamp(cfg["validation_start"]), pd.Timestamp(cfg["validation_end"]))],
        ["training", "validation"], default="out_of_window"
    )
    feature_cols = cfg["features"]
    panel["feature_missing_count"] = panel[feature_cols].isna().sum(axis=1)
    panel["model_eligible"] = panel["y"].notna() & panel["feature_missing_count"].eq(0) & panel["all_sleeves_available"]
    panel.to_csv(out / "timing_panel_dev.csv", index=False)
    panel[["Date", "ai_return_1", "spy_return_1", "ai_excess_return_1", "available_sleeves", "total_valid_member_count", "basket_reason_code"]].to_csv(out / "basket_daily_returns.csv", index=False)
    panel.loc[~panel["model_eligible"], ["Date", "split", "feature_missing_count", "basket_reason_code", "y"]].to_csv(out / "quarantine" / "model_ineligible_sessions.csv", index=False)

    missing = panel.groupby("split")[feature_cols].agg(lambda x: int(x.isna().sum())).T.reset_index(names="feature")
    missing.to_csv(out / "feature_missingness.csv", index=False)
    counts = panel.groupby("split", dropna=False).agg(
        sessions=("Date", "size"), model_eligible_sessions=("model_eligible", "sum"), labelled_sessions=("y", "count"),
        basket_available_sessions=("all_sleeves_available", "sum"), mean_valid_members=("total_valid_member_count", "mean"),
    ).reset_index()
    counts.to_csv(out / "split_counts.csv", index=False)
    summary = {
        "schema_version": cfg["schema_version"], "run_id": args.run_id, "status": "complete_development_only",
        "config_path": str(config_path), "config_sha256": sha256(config_path), "builder_path": str(Path(__file__)),
        "builder_sha256": sha256(Path(__file__)),
        "inputs": {str(registry_path): sha256(registry_path), str(returns_path): sha256(returns_path), str(macro_path): sha256(macro_path)},
        "candidate_core_count": int(len(candidates)), "roles": roles, "feature_columns": feature_cols,
        "output_counts": counts.to_dict(orient="records"), "test_policy_verified": "Output range ends on 2022-12-30; no sealed test target path is configured or read.",
        "processing": {"row_deletion": "none", "return_imputation": "none", "winsorization": "none", "forward_fill": "none", "missing_return_handling": cfg["missing_return_policy"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ai_infrastructure_timing_v1_config.json")
    parser.add_argument("--output-root", default="data/model_ready_ai_infrastructure_timing_v1")
    parser.add_argument("--run-id", required=True)
    destination = build(parser.parse_args())
    print(destination)
