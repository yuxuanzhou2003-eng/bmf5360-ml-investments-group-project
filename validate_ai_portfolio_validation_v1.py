"""Independent validation-only audit for the AI portfolio backtest.

This validator reads the frozen 2021-2022 portfolio outputs and reconstructs
the cash/holdings ledger from prior holdings, realised returns and trades.  It
never reads the sealed test-target file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BACKTEST_ROOT = ROOT / "data/backtests/ai_portfolio_v1"
AUDIT_ROOT = ROOT / "data/audit/ai_portfolio_v1"
RETURNS = ROOT / "data/clean/v2/returns.csv"
MR_ELIGIBILITY = ROOT / "data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/eligibility.csv"
REGISTRY = ROOT / "data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv"
SPY = "SPY.P"
TOL = 1e-10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def check(rows: list[dict], name: str, status: str, blocking: bool, detail: str, value=None) -> None:
    rows.append({
        "check": name,
        "status": status,
        "blocking": bool(blocking),
        "value": value,
        "detail": detail,
    })


def main(source_run_id: str, audit_run_id: str) -> dict:
    source = BACKTEST_ROOT / source_run_id
    output = AUDIT_ROOT / audit_run_id
    output.mkdir(parents=True, exist_ok=False)

    required = [
        "daily_inference.csv", "eligibility.csv", "portfolio_targets.csv",
        "trades.csv", "holdings_daily.csv", "nav_daily.csv", "metrics.csv",
        "audit.json", "summary.json",
    ]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing source outputs: {missing}")

    nav = pd.read_csv(source / "nav_daily.csv", parse_dates=["Date"])
    holdings = pd.read_csv(source / "holdings_daily.csv", parse_dates=["Date"])
    trades = pd.read_csv(source / "trades.csv", parse_dates=["formation_session"])
    targets = pd.read_csv(source / "portfolio_targets.csv", parse_dates=["formation_session"])
    eligibility = pd.read_csv(source / "eligibility.csv", parse_dates=["formation_session"])
    inference = pd.read_csv(source / "daily_inference.csv", parse_dates=["formation_session", "reference_session"])
    registry = pd.read_csv(REGISTRY).rename(columns={"ric": "Instrument"})
    gross_return_column = "gross_return" if "gross_return" in nav.columns else "gross_return_before_current_cost"
    holding_weight_column = "weight" if "weight" in holdings.columns else "signed_weight"
    performance_flag_column = "is_return_day" if "is_return_day" in nav.columns else "is_performance_observation"

    instruments = sorted(set(holdings["Instrument"]) | {SPY})
    returns = pd.read_csv(RETURNS, usecols=["Instrument", "Date", "return_decimal"], low_memory=False)
    returns = returns[returns["Instrument"].isin(instruments)].copy()
    returns["Date"] = pd.to_datetime(returns["Date"])
    ret = returns.pivot(index="Date", columns="Instrument", values="return_decimal")

    checks: list[dict] = []
    check(checks, "test_targets_not_opened", "PASS", True,
          "Validator reads validation outputs and the all-period returns table only; no target or label file is opened.")
    valid_dates = inference["formation_session"].between("2021-01-01", "2022-12-31").all()
    check(checks, "validation_dates_only", "PASS" if valid_dates else "FAIL", True,
          "All scored formation sessions must lie in 2021-2022.", bool(valid_dates))

    rebalance_dates = pd.DatetimeIndex(nav.loc[nav["event"].eq("REBALANCE"), "Date"])
    inference_sessions = pd.DatetimeIndex(sorted(inference["formation_session"].unique()))
    locations = [inference_sessions.get_loc(x) for x in rebalance_dates]
    spacing_ok = len(locations) == 23 and all(b - a == 21 for a, b in zip(locations, locations[1:]))
    check(checks, "rebalance_spacing_21_sessions", "PASS" if spacing_ok else "FAIL", True,
          "There must be 23 complete formation blocks, each 21 scored SPY sessions apart.", locations)

    formation_counts = eligibility.groupby("formation_session").size()
    formation_rows_ok = len(formation_counts) == 23 and formation_counts.eq(49).all()
    check(checks, "eligibility_49_rows_each_formation", "PASS" if formation_rows_ok else "FAIL", True,
          "All 49 registered candidates must remain visible at every formation, including exclusions.",
          {"formations": int(len(formation_counts)), "min": int(formation_counts.min()), "max": int(formation_counts.max())})

    # Establish whether the contemporaneous formation-close flag could have
    # changed selection.  A zero count proves it was redundant in this run.
    mr_elig = pd.read_csv(MR_ELIGIBILITY, usecols=["sample_id", "entry_trade_eligible", "feature_core_available"])
    flags = inference[["sample_id", "formation_session"]].merge(mr_elig, on="sample_id", validate="one_to_one")
    formation_flags = flags[flags["formation_session"].isin(rebalance_dates)].copy()
    entry_false_core_true = ((formation_flags["feature_core_available"].astype(str).str.casefold() == "true") &
                             (formation_flags["entry_trade_eligible"].astype(str).str.casefold() != "true"))
    affected = int(entry_false_core_true.sum())
    check(checks, "formation_close_flag_did_not_change_ranking", "PASS" if affected == 0 else "WARN", False,
          "The preliminary runner included entry_trade_eligible in MODEL_GATE. Zero affected rows means the same-day flag was redundant; the official runner should still separate ranking from execution.", affected)

    # Independently reconstruct the full ledger.
    prev_nav = float(nav.iloc[0]["net_nav"])
    prev_values: dict[str, float] = {}
    prev_cash = prev_nav
    max_nav_error = 0.0
    max_gross_return_error = 0.0
    max_net_return_error = 0.0
    max_cost_error = 0.0
    max_turnover_error = 0.0
    missing_return_marks = 0

    for i, row in nav.iterrows():
        date = row["Date"]
        if i == 0:
            nav_pre = 1.0
        else:
            marked = {}
            for inst, value in prev_values.items():
                rv = ret.at[date, inst] if date in ret.index and inst in ret.columns else np.nan
                if not np.isfinite(rv):
                    missing_return_marks += 1
                    continue
                marked[inst] = value * (1.0 + float(rv))
            nav_pre = prev_cash + sum(marked.values())
            prev_values = marked

        day_trades = trades[trades["formation_session"].eq(date)]
        reconstructed_cost = float(day_trades["spread_cost"].sum()) if len(day_trades) else 0.0
        reconstructed_turnover = float(day_trades["turnover_contribution"].sum()) if len(day_trades) else 0.0
        max_cost_error = max(max_cost_error, abs(reconstructed_cost - float(row["spread_cost"])))
        max_turnover_error = max(max_turnover_error, abs(reconstructed_turnover - float(row["turnover"])))

        expected_gross = 0.0 if i == 0 else nav_pre / prev_nav - 1.0
        max_gross_return_error = max(max_gross_return_error, abs(expected_gross - float(row[gross_return_column])))

        if len(day_trades):
            new_weights = day_trades.set_index("Instrument")["new_weight"].to_dict()
            prev_values = {inst: float(weight) * nav_pre for inst, weight in new_weights.items() if abs(weight) > TOL}
            prev_cash = nav_pre - sum(prev_values.values()) - reconstructed_cost

        expected_nav = prev_cash + sum(prev_values.values())
        max_nav_error = max(max_nav_error, abs(expected_nav - float(row["net_nav"])))
        expected_net_return = expected_nav - 1.0 if i == 0 else expected_nav / prev_nav - 1.0
        max_net_return_error = max(max_net_return_error, abs(expected_net_return - float(row["net_return"])))

        reported_holdings = holdings[holdings["Date"].eq(date)].set_index("Instrument")["holding_value"].to_dict()
        all_names = set(prev_values) | set(reported_holdings)
        if all_names:
            max_nav_error = max(max_nav_error, max(abs(prev_values.get(k, 0.0) - reported_holdings.get(k, 0.0)) for k in all_names))
        prev_nav = expected_nav

    ledger_ok = missing_return_marks == 0 and max(max_nav_error, max_gross_return_error, max_net_return_error) <= TOL
    check(checks, "independent_daily_ledger_reconstruction", "PASS" if ledger_ok else "FAIL", True,
          "Prior holdings are marked with realised returns, then trades and costs are applied; reconstructed NAV and returns must match.",
          {"missing_return_marks": missing_return_marks, "max_nav_or_holding_error": max_nav_error,
           "max_gross_return_error": max_gross_return_error, "max_net_return_error": max_net_return_error})
    cost_turn_ok = max(max_cost_error, max_turnover_error) <= TOL
    check(checks, "daily_cost_and_turnover_aggregation", "PASS" if cost_turn_ok else "FAIL", True,
          "Daily cost and one-way turnover equal the sum of trade-level contributions.",
          {"max_cost_error": max_cost_error, "max_turnover_error": max_turnover_error})

    # Recover pre-trade NAV exactly from cost / formula without relying on end NAV.
    formula_errors = []
    for t in trades.itertuples(index=False):
        day_cost = float(t.spread_cost)
        denom = abs(float(t.delta_weight)) * float(t.quoted_spread_bps) / 2.0 / 10000.0
        implied_pre_nav = day_cost / denom if denom > 0 else np.nan
        row = nav.loc[nav["Date"].eq(t.formation_session)].iloc[0]
        reported_pre_nav = float(row.net_nav + row.spread_cost)
        formula_errors.append(abs(implied_pre_nav - reported_pre_nav))
    max_formula_error = max(formula_errors, default=0.0)
    check(checks, "half_quoted_spread_trade_cost", "PASS" if max_formula_error <= TOL else "FAIL", True,
          "Each trade cost must equal absolute traded notional times one half of the full quoted spread.", max_formula_error)

    initial_cost = float(nav.iloc[0]["spread_cost"])
    initial_in_metric = bool(nav.iloc[0][performance_flag_column])
    metric_status = "PASS" if initial_cost <= TOL or initial_in_metric else "WARN"
    check(checks, "initial_formation_cost_in_risk_metrics", metric_status, False,
          "The preliminary cumulative NAV includes initial formation cost. If the first row is excluded from the daily return sample, Sharpe and volatility omit that cost event; the official run should include and disclose it.",
          {"initial_spread_cost": initial_cost, "first_row_in_return_sample": initial_in_metric})

    normal_targets = targets[targets["reason_code"].isin(["SELECTED", "NOT_SELECTED"])].copy()
    max_name_target = float(normal_targets["target_weight"].max()) if len(normal_targets) else np.nan
    group_map = registry.set_index("Instrument")["primary_group"].to_dict()
    selected_targets = normal_targets[normal_targets["target_weight"].gt(TOL)].copy()
    selected_targets["group"] = selected_targets["Instrument"].map(group_map)
    group_sums = selected_targets.groupby(["formation_session", "group"])["target_weight"].sum()
    max_group_target = float(group_sums.max()) if len(group_sums) else 0.0
    caps_ok = max_name_target <= 0.1 + TOL and max_group_target <= 0.3 + TOL
    check(checks, "constructed_target_caps", "PASS" if caps_ok else "FAIL", True,
          "Freshly constructed stock targets must respect 10% per-name and 30% per-group caps.",
          {"max_name_target": max_name_target, "max_group_target": max_group_target})

    realised = holdings[holdings["Instrument"].ne(SPY)].copy()
    max_realised_name = float(realised[holding_weight_column].max()) if len(realised) else 0.0
    realised["group"] = realised["Instrument"].map(group_map)
    realised_group = realised.groupby(["Date", "group"])[holding_weight_column].sum()
    max_realised_group = float(realised_group.max()) if len(realised_group) else 0.0
    realised_status = "PASS" if max_realised_name <= 0.1 + TOL and max_realised_group <= 0.3 + TOL else "WARN"
    check(checks, "realised_weights_after_drift_and_turnover_cap", realised_status, False,
          "Target caps are construction limits. Drift and the turnover cap can leave realised weights temporarily above them; disclose the maxima.",
          {"max_name_weight": max_realised_name, "max_group_weight": max_realised_group})

    scheduled = nav[nav["event"].eq("REBALANCE")]
    scheduled_turn_ok = scheduled["turnover"].le(0.5 + TOL).all()
    check(checks, "scheduled_one_way_turnover_cap", "PASS" if scheduled_turn_ok else "FAIL", True,
          "Scheduled rebalance one-way turnover must not exceed 50%.", float(scheduled["turnover"].max()))
    final_rows = nav[nav["event"].eq("FINAL_LIQUIDATION")]
    final_ok = len(final_rows) == 1 and holdings["Date"].max() < final_rows.iloc[0]["Date"]
    check(checks, "final_liquidation_complete", "PASS" if final_ok else "FAIL", True,
          "The final complete holding block must end with one mandatory liquidation and no remaining holdings.",
          {"events": int(len(final_rows)), "liquidation_turnover": float(final_rows.iloc[0]["turnover"]) if len(final_rows) else None})

    skipped = targets.loc[~targets["reason_code"].isin(["SELECTED", "NOT_SELECTED"]), ["formation_session", "reason_code"]].drop_duplicates()
    check(checks, "skipped_formations_disclosed", "PASS", False,
          "Formation dates that failed the minimum-signal rule are retained with reason codes.",
          skipped["reason_code"].value_counts().to_dict())

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(output / "checks.csv", index=False)
    blocking_failures = checks_df[(checks_df["blocking"]) & checks_df["status"].eq("FAIL")]
    warnings = checks_df[checks_df["status"].eq("WARN")]
    summary = {
        "schema": "ai_portfolio_validation_audit_v1",
        "audit_run_id": audit_run_id,
        "source_run_id": source_run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if blocking_failures.empty else "FAIL",
        "counts": {
            "checks": int(len(checks_df)),
            "pass": int(checks_df["status"].eq("PASS").sum()),
            "warn": int(checks_df["status"].eq("WARN").sum()),
            "fail": int(checks_df["status"].eq("FAIL").sum()),
            "blocking_fail": int(len(blocking_failures)),
        },
        "source_hashes": {rel(source / name): sha256(source / name) for name in required},
        "validator": {rel(Path(__file__)): sha256(Path(__file__))},
        "warnings": warnings[["check", "detail", "value"]].to_dict("records"),
        "limitations": [
            "This audit verifies the validation-only implementation; it does not evaluate the sealed test period.",
            ("The formal runner separates F-1 ranking from formation-close execution checks."
             if "is_performance_observation" in nav.columns else
             "The preliminary runner used a same-day execution eligibility flag inside MODEL_GATE; the audit measures whether it changed this run."),
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--audit-run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    main(args.source_run_id, args.audit_run_id)
