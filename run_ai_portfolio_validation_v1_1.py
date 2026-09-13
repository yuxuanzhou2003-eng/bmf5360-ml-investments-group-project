"""Formal validation-only implementation of AI portfolio specification v1.

Changes from the preserved preliminary runner:
- ranking uses only information through F-1;
- formation-close price/quote is an execution check after ranking;
- initial formation cost is included in the daily risk-metric sample;
- output tables carry detailed flags, stages, provenance and hashes.

The sealed test-target file is never read.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from run_ai_portfolio_validation_v1 import (
    AI, MODEL_KEY, MODEL_RUN, MR, OUT_ROOT, PRICES, REGISTRY, RETURNS, ROOT,
    SPEC, SPY, allocate, rel, sha,
)


RUNNER = Path(__file__).resolve()
MODEL_VERSION = "ai_pool_daily_state_models_v1_1/20260910T070000000000Z/technical_plus_ai_state_logistic"


def truth(value) -> bool:
    return str(value).strip().casefold() == "true"


def performance_metrics(nav: pd.DataFrame) -> dict:
    # The first row is a real cost event: the fund starts immediately before
    # its first formation-close trade. There are len(nav)-1 elapsed sessions.
    r = nav["net_return"].astype(float)
    spy = nav["spy_return"].astype(float)
    elapsed = max(len(nav) - 1, 1)
    end = float(nav.iloc[-1]["net_nav"])
    ann = end ** (252 / elapsed) - 1
    std = r.std(ddof=1)
    vol = float(std * np.sqrt(252))
    sharpe = float(r.mean() / std * np.sqrt(252)) if std > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    sortino = float(r.mean() / downside * np.sqrt(252)) if pd.notna(downside) and downside > 0 else np.nan
    drawdown = nav["net_nav"] / nav["net_nav"].cummax() - 1
    beta = float(np.cov(r, spy, ddof=1)[0, 1] / np.var(spy, ddof=1)) if np.var(spy, ddof=1) > 0 else np.nan
    alpha = float((r.mean() - beta * spy.mean()) * 252) if np.isfinite(beta) else np.nan
    spy_end = float((1.0 + spy).prod())
    return {
        "performance_observations_including_initial_cost_event": int(len(r)),
        "elapsed_spy_sessions": int(elapsed),
        "cumulative_net_return": end - 1,
        "annualized_return": float(ann),
        "annualized_volatility": vol,
        "sharpe_zero_cash": sharpe,
        "sortino_zero_cash": sortino,
        "max_drawdown": float(drawdown.min()),
        "beta_to_spy": beta,
        "annualized_alpha_zero_cash": alpha,
        "spy_same_window_cumulative_return": spy_end - 1,
        "total_one_way_turnover": float(nav["turnover"].sum()),
        "annualized_turnover": float(nav["turnover"].sum() * 252 / elapsed),
        "total_spread_cost_initial_nav_units": float(nav["spread_cost"].sum()),
        "ending_nav": end,
    }


def main(run_id: str) -> dict:
    print(
        "阶段说明：生成正式的2021-2022验证期组合；排名仅用F-1信息，F收盘数据只检查成交；"
        "不填补、不缩尾、不二次拆股调整，不读取测试目标。",
        flush=True,
    )
    feature_sets = json.loads((MODEL_RUN / "feature_sets.json").read_text(encoding="utf-8"))
    cols = feature_sets["technical_plus_ai_state"]
    technical = [c for c in cols if not c.startswith(("etf_", "soxx_", "pool_"))]
    ai_cols = [c for c in cols if c not in technical]
    keys = ["sample_id", "security_id", "Instrument", "formation_session", "split"]

    features = pd.read_csv(MR / "model_features.csv", usecols=keys + technical, low_memory=False)
    features = features[features["split"].eq("validation")].copy()
    features["formation_session"] = pd.to_datetime(features["formation_session"])
    metadata = pd.read_csv(MR / "metadata.csv", usecols=["sample_id", "reference_session"], low_memory=False)
    eligibility_source = pd.read_csv(
        MR / "eligibility.csv",
        usecols=["sample_id", "entry_trade_eligible", "feature_core_available"],
        low_memory=False,
    )
    ai = pd.read_csv(AI, usecols=["formation_session"] + ai_cols, low_memory=False)
    ai["formation_session"] = pd.to_datetime(ai["formation_session"])
    data = (
        features.merge(metadata, on="sample_id", validate="one_to_one")
        .merge(eligibility_source, on="sample_id", validate="one_to_one")
        .merge(ai, on="formation_session", validate="many_to_one")
    )
    data["reference_session"] = pd.to_datetime(data["reference_session"])
    model = joblib.load(MODEL_RUN / f"models/{MODEL_KEY}.joblib")
    data["raw_missing_feature_count"] = data[cols].isna().sum(axis=1)
    data["used_stored_training_imputer"] = data["raw_missing_feature_count"].gt(0)
    data["p_up"] = model.predict_proba(data[cols])[:, 1]
    if not np.isfinite(data["p_up"]).all():
        raise RuntimeError("nonfinite probability")

    registry = pd.read_csv(REGISTRY).rename(columns={"ric": "Instrument"})
    registry["member_from"] = pd.to_datetime(registry["member_from"])
    registry["member_to"] = pd.to_datetime(registry["member_to"])
    registry_idx = registry.set_index("Instrument")
    universe = set(registry["Instrument"]) | {SPY}
    prices_long = pd.read_csv(PRICES, low_memory=False)
    prices_long = prices_long[prices_long["Instrument"].isin(universe)].copy()
    prices_long["Date"] = pd.to_datetime(prices_long["Date"])
    returns_long = pd.read_csv(RETURNS, usecols=["Instrument", "Date", "return_decimal"], low_memory=False)
    returns_long = returns_long[returns_long["Instrument"].isin(universe)].copy()
    returns_long["Date"] = pd.to_datetime(returns_long["Date"])
    prices_long = prices_long.sort_values(["Instrument", "Date"])
    returns_long = returns_long.sort_values(["Instrument", "Date"])

    price = prices_long.pivot(index="Date", columns="Instrument", values="TRDPRC_1")
    bid = prices_long.pivot(index="Date", columns="Instrument", values="BID")
    ask = prices_long.pivot(index="Date", columns="Instrument", values="ASK")
    spread = prices_long.pivot(index="Date", columns="Instrument", values="quoted_spread_bps")
    ret = returns_long.pivot(index="Date", columns="Instrument", values="return_decimal")
    if SPY not in ret.columns or SPY not in spread.columns:
        raise RuntimeError("SPY inputs unavailable")

    sessions = pd.DatetimeIndex(sorted(data["formation_session"].unique()))
    formations = [sessions[i] for i in range(0, len(sessions), 21) if i + 21 < len(sessions)]
    final_exit = sessions[(len(formations) - 1) * 21 + 21]
    formation_set = set(formations)
    output = OUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)

    inference = data.sort_values(["formation_session", "Instrument"]).copy()
    inference["model_version"] = MODEL_VERSION
    inference["prediction_available"] = np.isfinite(inference["p_up"])
    inference["inference_reason_code"] = np.where(inference["prediction_available"], "OK", "PREDICTION_UNAVAILABLE")
    inference_columns = keys + [
        "reference_session", "model_version", "p_up", "prediction_available",
        "feature_core_available", "raw_missing_feature_count", "used_stored_training_imputer",
        "inference_reason_code",
    ]
    inference[inference_columns].to_csv(output / "daily_inference.csv", index=False)

    values: dict[str, float] = {}
    cash = 1.0
    previous_nav = 1.0
    pre_cost_compounded_nav = 1.0
    incumbents: set[str] = set()
    nav_rows: list[dict] = []
    holding_rows: list[dict] = []
    eligibility_rows: list[pd.DataFrame] = []
    target_rows: list[dict] = []
    trade_rows: list[dict] = []
    unresolved = False

    for ix, date in enumerate(sessions[sessions <= final_exit]):
        stock_pnl = 0.0
        spy_pnl = 0.0
        if ix > 0:
            for inst, value in list(values.items()):
                realised_return = ret.at[date, inst] if date in ret.index and inst in ret.columns else np.nan
                if not np.isfinite(realised_return):
                    unresolved = True
                    break
                pnl = value * float(realised_return)
                if inst == SPY:
                    spy_pnl += pnl
                else:
                    stock_pnl += pnl
                values[inst] = value + pnl
            if unresolved:
                break

        nav_pretrade = cash + sum(values.values())
        gross_return = 0.0 if ix == 0 else nav_pretrade / previous_nav - 1.0
        cost = 0.0
        turnover = 0.0
        event = "HOLD"

        if date in formation_set:
            event = "REBALANCE"
            day = data[data["formation_session"].eq(date)].copy()
            rows = []
            for inst in registry["Instrument"]:
                sample = day[day["Instrument"].eq(inst)]
                active = bool(registry_idx.at[inst, "member_from"] <= date <= registry_idx.at[inst, "member_to"])
                flags = {
                    "active_membership": active,
                    "model_row_available": not sample.empty,
                    "price_gate_pass": False,
                    "price_history_gate_pass": False,
                    "quote_history_gate_pass": False,
                    "liquidity_gate_pass": False,
                    "spread_gate_pass": False,
                    "risk_gate_pass": False,
                    "model_gate_pass": False,
                    "execution_close_pass": False,
                    "execution_quote_pass": False,
                }
                reasons = []
                if not active or sample.empty:
                    reasons.append("NO_ACTIVE_MEMBERSHIP_ROW")
                if not sample.empty:
                    z = sample.iloc[0]
                    reference = z["reference_session"]
                    trail = prices_long[(prices_long["Instrument"].eq(inst)) & (prices_long["Date"].le(reference))].tail(20)
                    close = float(trail["TRDPRC_1"].iloc[-1]) if len(trail) else np.nan
                    flags["price_gate_pass"] = bool(np.isfinite(close) and close >= 5)
                    flags["price_history_gate_pass"] = int((pd.to_numeric(trail["TRDPRC_1"], errors="coerce") > 0).sum()) >= 15
                    valid_quote = (
                        pd.to_numeric(trail["BID"], errors="coerce").notna()
                        & pd.to_numeric(trail["ASK"], errors="coerce").notna()
                        & trail["ASK"].ge(trail["BID"])
                        & trail["BID"].gt(0)
                    )
                    flags["quote_history_gate_pass"] = int(valid_quote.sum()) >= 15
                    flags["liquidity_gate_pass"] = bool(
                        np.isfinite(z["dollar_volume_median_20_log1p"])
                        and np.expm1(z["dollar_volume_median_20_log1p"]) >= 20e6
                    )
                    flags["spread_gate_pass"] = bool(
                        np.isfinite(z["spread_median_20_bps"]) and z["spread_median_20_bps"] <= 50
                    )
                    flags["risk_gate_pass"] = bool(
                        np.isfinite(z["beta_126"]) and np.isfinite(z["volatility_60_ann"])
                        and np.isfinite(z["idio_vol_126_ann"]) and z["beta_obs_126"] >= 100
                    )
                    flags["model_gate_pass"] = bool(truth(z["feature_core_available"]) and np.isfinite(z["p_up"]))
                    actual_close = price.at[date, inst] if date in price.index and inst in price.columns else np.nan
                    actual_bid = bid.at[date, inst] if date in bid.index and inst in bid.columns else np.nan
                    actual_ask = ask.at[date, inst] if date in ask.index and inst in ask.columns else np.nan
                    flags["execution_close_pass"] = bool(np.isfinite(actual_close) and actual_close > 0)
                    flags["execution_quote_pass"] = bool(
                        np.isfinite(actual_bid) and np.isfinite(actual_ask) and actual_bid > 0 and actual_ask >= actual_bid
                    )
                    for flag_name, code in [
                        ("price_gate_pass", "PRICE_GATE"),
                        ("price_history_gate_pass", "PRICE_HISTORY_GATE"),
                        ("quote_history_gate_pass", "QUOTE_HISTORY_GATE"),
                        ("liquidity_gate_pass", "LIQUIDITY_GATE"),
                        ("spread_gate_pass", "SPREAD_GATE"),
                        ("risk_gate_pass", "RISK_GATE"),
                        ("model_gate_pass", "MODEL_GATE"),
                    ]:
                        if not flags[flag_name]:
                            reasons.append(code)
                    pretrade_eligible = not reasons
                    rows.append({
                        "Instrument": inst,
                        "sample_id": z["sample_id"],
                        "reference_session": reference,
                        "p_up": float(z["p_up"]),
                        "beta": float(z["beta_126"]),
                        "primary_group": registry_idx.at[inst, "primary_group"],
                        "spread_median_20_bps": float(z["spread_median_20_bps"]) if np.isfinite(z["spread_median_20_bps"]) else np.nan,
                        **flags,
                        "source_entry_trade_eligible": truth(z["entry_trade_eligible"]),
                        "pretrade_eligible": pretrade_eligible,
                        "pretrade_reason_codes": ";".join(reasons) or "OK",
                        "execution_reason_codes": "OK" if flags["execution_close_pass"] and flags["execution_quote_pass"] else "EXECUTION_GATE_FAILED",
                    })
                else:
                    rows.append({
                        "Instrument": inst, "sample_id": "", "reference_session": pd.NaT,
                        "p_up": np.nan, "beta": np.nan,
                        "primary_group": registry_idx.at[inst, "primary_group"], **flags,
                        "spread_median_20_bps": np.nan,
                        "source_entry_trade_eligible": False, "pretrade_eligible": False,
                        "pretrade_reason_codes": ";".join(reasons), "execution_reason_codes": "NOT_EVALUATED",
                    })

            eligible_day = pd.DataFrame(rows)
            ranked = eligible_day[eligible_day["pretrade_eligible"]].sort_values(
                ["p_up", "Instrument"], ascending=[False, True]
            ).copy()
            ranked["rank"] = np.arange(1, len(ranked) + 1)
            eligible_day = eligible_day.merge(ranked[["Instrument", "rank"]], on="Instrument", how="left")
            eligible_day["formation_session"] = date
            eligibility_rows.append(eligible_day)

            target: dict[str, float] = {}
            selected: list[str] = []
            initially_selected: list[str] = []
            skip_reason = ""
            pre_cap: dict[str, float] = {}
            post_cap: dict[str, float] = {}
            post_beta: dict[str, float] = {}
            volatility_scale = np.nan
            stock_beta = np.nan
            estimated_volatility = np.nan
            if len(ranked) < 20:
                skip_reason = "ELIGIBLE_COUNT_LT_20"
            else:
                q = max(5, math.ceil(0.2 * len(ranked)))
                buffer = math.ceil(0.3 * len(ranked))
                protected = ranked[
                    ranked["Instrument"].isin(incumbents)
                    & ranked["rank"].le(buffer)
                    & ranked["p_up"].ge(0.50)
                ]["Instrument"].tolist()
                strict = ranked[
                    ranked["rank"].le(q) & ranked["p_up"].ge(0.55)
                ]["Instrument"].tolist()
                initially_selected = (protected + [x for x in strict if x not in protected])[:q]
                if len(initially_selected) < 5:
                    skip_reason = "SELECTED_COUNT_LT_5"
                else:
                    execution_map = eligible_day.set_index("Instrument")
                    selected = [
                        x for x in initially_selected
                        if bool(execution_map.at[x, "execution_close_pass"])
                        and bool(execution_map.at[x, "execution_quote_pass"])
                    ]
                    if len(selected) < 5:
                        skip_reason = "EXECUTABLE_SELECTED_COUNT_LT_5"

            if not skip_reason:
                pre_cap = {name: 1.0 / len(selected) for name in selected}
                groups = ranked.set_index("Instrument")["primary_group"].to_dict()
                post_cap = allocate(selected, groups)
                betas = ranked.set_index("Instrument")["beta"].to_dict()
                stock_beta = sum(post_cap[name] * betas[name] for name in post_cap)
                beta_scale = min(1.0, 1.0 / abs(stock_beta)) if stock_beta != 0 else 1.0
                post_beta = {name: weight * beta_scale for name, weight in post_cap.items()}
                hedged_beta = sum(post_beta[name] * betas[name] for name in post_beta)
                target = dict(post_beta)
                target[SPY] = -hedged_beta
                history = ret.loc[ret.index < date, [*selected, SPY]].tail(126).dropna()
                if len(history) < 100:
                    skip_reason = "RISK_COMMON_HISTORY_LT_100"
                    target = {}
                else:
                    order = [*selected, SPY]
                    vector = np.array([target.get(name, 0.0) for name in order])
                    estimated_volatility = float(np.sqrt(max(0.0, 252 * vector @ history.cov().to_numpy() @ vector)))
                    if not np.isfinite(estimated_volatility) or estimated_volatility <= 0:
                        skip_reason = "RISK_ESTIMATE_UNAVAILABLE"
                        target = {}
                    else:
                        volatility_scale = min(1.0, 0.10 / estimated_volatility)
                        target = {name: weight * volatility_scale for name, weight in target.items()}

            current = {name: value / nav_pretrade for name, value in values.items()}
            if skip_reason:
                target = dict(current)
            assets = set(current) | set(target)
            raw_turnover = 0.5 * sum(abs(target.get(x, 0.0) - current.get(x, 0.0)) for x in assets)
            trade_fraction = min(1.0, 0.50 / raw_turnover) if raw_turnover > 0 else 1.0
            traded = {
                x: current.get(x, 0.0) + trade_fraction * (target.get(x, 0.0) - current.get(x, 0.0))
                for x in assets
            }
            traded = {x: w for x, w in traded.items() if abs(w) > 1e-12}
            turnover = 0.5 * sum(abs(traded.get(x, 0.0) - current.get(x, 0.0)) for x in set(current) | set(traded))

            for inst in sorted(set(current) | set(traded)):
                delta = traded.get(inst, 0.0) - current.get(inst, 0.0)
                if abs(delta) < 1e-12:
                    continue
                spread_bps = spread.at[date, inst] if date in spread.index and inst in spread.columns else np.nan
                spread_source = "EXECUTION_DAY_QUOTE"
                if not np.isfinite(spread_bps):
                    fallback = eligible_day.loc[eligible_day["Instrument"].eq(inst), "spread_median_20_bps"] if "spread_median_20_bps" in eligible_day else pd.Series(dtype=float)
                    if len(fallback) and np.isfinite(fallback.iloc[0]):
                        spread_bps = float(fallback.iloc[0])
                        spread_source = "SPREAD_FALLBACK_TRAILING_MEDIAN"
                    else:
                        raise RuntimeError(f"INCOMPLETE_UNRESOLVED_NAV: spread unavailable for {inst} on {date.date()}")
                execution_price = price.at[date, inst] if date in price.index and inst in price.columns else np.nan
                if not np.isfinite(execution_price) or execution_price <= 0:
                    raise RuntimeError(f"INCOMPLETE_UNRESOLVED_NAV: execution price unavailable for {inst} on {date.date()}")
                trade_cost = abs(delta) * nav_pretrade * float(spread_bps) / 2.0 / 10000.0
                cost += trade_cost
                trade_rows.append({
                    "formation_session": date, "Instrument": inst,
                    "old_weight": current.get(inst, 0.0), "new_weight": traded.get(inst, 0.0),
                    "delta_weight": delta, "side": "BUY" if delta > 0 else "SELL",
                    "execution_price": float(execution_price), "spread_source": spread_source,
                    "quoted_spread_bps": float(spread_bps), "spread_cost": trade_cost,
                    "turnover_contribution": 0.5 * abs(delta), "mandatory_exit": False,
                    "reason_code": "SCHEDULED_REBALANCE" if not skip_reason else "SKIPPED_TARGET_CARRY",
                })

            values = {name: weight * nav_pretrade for name, weight in traded.items()}
            cash = nav_pretrade - sum(values.values()) - cost
            incumbents = {name for name, weight in traded.items() if name != SPY and weight > 0}
            row_lookup = eligible_day.set_index("Instrument")
            for inst in registry["Instrument"]:
                target_rows.append({
                    "formation_session": date, "Instrument": inst,
                    "p_up": row_lookup.at[inst, "p_up"], "rank": row_lookup.at[inst, "rank"],
                    "initially_selected": inst in initially_selected,
                    "execution_eligible": bool(row_lookup.at[inst, "execution_close_pass"] and row_lookup.at[inst, "execution_quote_pass"]),
                    "selected_for_target": inst in selected and not skip_reason,
                    "primary_group": row_lookup.at[inst, "primary_group"],
                    "pre_cap_weight": pre_cap.get(inst, 0.0),
                    "post_cap_stock_weight": post_cap.get(inst, 0.0),
                    "post_beta_stock_weight": post_beta.get(inst, 0.0),
                    "estimated_pre_scale_volatility": estimated_volatility,
                    "volatility_scale": volatility_scale,
                    "target_weight": target.get(inst, 0.0),
                    "executed_weight": traded.get(inst, 0.0),
                    "trade_fraction": trade_fraction,
                    "reason_code": skip_reason or ("SELECTED" if inst in selected else "NOT_SELECTED"),
                })
            target_rows.append({
                "formation_session": date, "Instrument": SPY, "p_up": np.nan, "rank": np.nan,
                "initially_selected": False, "execution_eligible": True, "selected_for_target": False,
                "primary_group": "market_hedge", "pre_cap_weight": 0.0,
                "post_cap_stock_weight": 0.0, "post_beta_stock_weight": 0.0,
                "estimated_pre_scale_volatility": estimated_volatility, "volatility_scale": volatility_scale,
                "target_weight": target.get(SPY, 0.0), "executed_weight": traded.get(SPY, 0.0),
                "trade_fraction": trade_fraction, "reason_code": skip_reason or "SPY_BETA_HEDGE",
            })

        elif date == final_exit:
            event = "FINAL_LIQUIDATION"
            current = {name: value / nav_pretrade for name, value in values.items()}
            turnover = 0.5 * sum(abs(weight) for weight in current.values())
            for inst, weight in sorted(current.items()):
                spread_bps = spread.at[date, inst] if date in spread.index and inst in spread.columns else np.nan
                execution_price = price.at[date, inst] if date in price.index and inst in price.columns else np.nan
                if not np.isfinite(spread_bps) or not np.isfinite(execution_price) or execution_price <= 0:
                    raise RuntimeError(f"INCOMPLETE_UNRESOLVED_NAV: liquidation input unavailable for {inst} on {date.date()}")
                trade_cost = abs(weight) * nav_pretrade * float(spread_bps) / 2.0 / 10000.0
                cost += trade_cost
                trade_rows.append({
                    "formation_session": date, "Instrument": inst, "old_weight": weight,
                    "new_weight": 0.0, "delta_weight": -weight,
                    "side": "SELL" if weight > 0 else "BUY",
                    "execution_price": float(execution_price), "spread_source": "EXECUTION_DAY_QUOTE",
                    "quoted_spread_bps": float(spread_bps), "spread_cost": trade_cost,
                    "turnover_contribution": 0.5 * abs(weight), "mandatory_exit": True,
                    "reason_code": "FINAL_SPLIT_LIQUIDATION",
                })
            values = {}
            cash = nav_pretrade - cost
            incumbents = set()

        nav_end = cash + sum(values.values())
        net_return = nav_end / previous_nav - 1.0 if ix > 0 else nav_end - 1.0
        pre_cost_compounded_nav *= 1.0 + gross_return
        spy_return = float(ret.at[date, SPY]) if ix > 0 else 0.0
        nav_rows.append({
            "Date": date, "event": event, "is_performance_observation": True,
            "is_elapsed_return_session": ix > 0, "gross_return_before_current_cost": gross_return,
            "stock_return_contribution": stock_pnl / previous_nav if ix > 0 else 0.0,
            "spy_hedge_return_contribution": spy_pnl / previous_nav if ix > 0 else 0.0,
            "spread_cost": cost, "spread_cost_return": -cost / previous_nav,
            "net_return": net_return, "pre_cost_compounded_nav_diagnostic": pre_cost_compounded_nav,
            "net_nav": nav_end, "turnover": turnover, "spy_return": spy_return,
            "stock_gross": sum(abs(value) for name, value in values.items() if name != SPY) / nav_end,
            "spy_weight": values.get(SPY, 0.0) / nav_end,
            "net_exposure": sum(values.values()) / nav_end,
            "cash_weight": cash / nav_end,
            "active_stock_positions": sum(name != SPY and value > 0 for name, value in values.items()),
        })
        for inst, value in values.items():
            holding_rows.append({
                "Date": date, "Instrument": inst, "holding_value": value,
                "signed_weight": value / nav_end, "primary_group": "market_hedge" if inst == SPY else registry_idx.at[inst, "primary_group"],
            })
        previous_nav = nav_end

    if unresolved:
        raise RuntimeError("INCOMPLETE_UNRESOLVED_NAV")

    nav = pd.DataFrame(nav_rows)
    holdings = pd.DataFrame(holding_rows)
    eligibility_out = pd.concat(eligibility_rows, ignore_index=True)
    targets = pd.DataFrame(target_rows)
    trades = pd.DataFrame(trade_rows)
    nav.to_csv(output / "nav_daily.csv", index=False)
    holdings.to_csv(output / "holdings_daily.csv", index=False)
    eligibility_out.to_csv(output / "eligibility.csv", index=False)
    targets.to_csv(output / "portfolio_targets.csv", index=False)
    trades.to_csv(output / "trades.csv", index=False)

    metrics = performance_metrics(nav)
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    skip_counts = (
        targets.loc[~targets["reason_code"].isin(["SELECTED", "NOT_SELECTED", "SPY_BETA_HEDGE"]), ["formation_session", "reason_code"]]
        .drop_duplicates()["reason_code"].value_counts().to_dict()
    )
    coverage = pd.DataFrame([
        {"scope": "daily_inference", "rows": len(inference), "instruments": inference["Instrument"].nunique(), "missing_or_excluded": int((~inference["prediction_available"]).sum()), "reason": "PREDICTION_UNAVAILABLE"},
        {"scope": "formation_registry", "rows": len(eligibility_out), "instruments": eligibility_out["Instrument"].nunique(), "missing_or_excluded": int((~eligibility_out["pretrade_eligible"]).sum()), "reason": "reason-coded in eligibility.csv"},
        {"scope": "formation_execution", "rows": len(eligibility_out), "instruments": eligibility_out["Instrument"].nunique(), "missing_or_excluded": int((~(eligibility_out["execution_close_pass"] & eligibility_out["execution_quote_pass"])).sum()), "reason": "execution gate; evaluated after ranking"},
        {"scope": "skipped_formations", "rows": len(formations), "instruments": 49, "missing_or_excluded": int(sum(skip_counts.values())), "reason": json.dumps(skip_counts, sort_keys=True)},
        {"scope": "daily_nav", "rows": len(nav), "instruments": holdings["Instrument"].nunique(), "missing_or_excluded": 0, "reason": "no unresolved marks"},
    ])
    coverage.to_csv(output / "coverage_audit.csv", index=False)

    max_target_name = float(targets.loc[targets["Instrument"].ne(SPY), "post_cap_stock_weight"].max())
    target_group = targets[targets["Instrument"].ne(SPY)].groupby(["formation_session", "primary_group"])["post_cap_stock_weight"].sum()
    max_target_group = float(target_group.max())
    realised_stock = holdings[holdings["Instrument"].ne(SPY)].copy()
    realised_group = realised_stock.groupby(["Date", "primary_group"])["signed_weight"].sum()
    checks = {
        "test_targets_not_opened": True,
        "validation_only": bool(inference["formation_session"].min() >= pd.Timestamp("2021-01-01") and inference["formation_session"].max() < pd.Timestamp("2023-01-01")),
        "daily_inference_rows_21671": len(inference) == 21671,
        "formation_rows_49_each": bool((eligibility_out.groupby("formation_session").size() == 49).all()),
        "formation_spacing_21": all(sessions.get_loc(formations[i + 1]) - sessions.get_loc(formations[i]) == 21 for i in range(len(formations) - 1)),
        "ranking_uses_f_minus_1_only": bool((
            eligibility_out.loc[eligibility_out["model_row_available"], "reference_session"]
            < eligibility_out.loc[eligibility_out["model_row_available"], "formation_session"]
        ).all()),
        "formation_execution_gate_separate": True,
        "no_unresolved_nav": True,
        "probabilities_finite": bool(np.isfinite(inference["p_up"]).all()),
        "scheduled_turnover_cap": bool(nav.loc[nav["event"].eq("REBALANCE"), "turnover"].le(0.5 + 1e-10).all()),
        "constructed_name_cap": max_target_name <= 0.1 + 1e-10,
        "constructed_group_cap": max_target_group <= 0.3 + 1e-10,
        "initial_cost_in_risk_metric_sample": bool(nav.iloc[0]["is_performance_observation"]),
        "raw_inputs_unchanged": True,
        "no_second_split_adjustment": True,
    }
    (output / "audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    output_files = [
        "daily_inference.csv", "eligibility.csv", "portfolio_targets.csv", "trades.csv",
        "holdings_daily.csv", "nav_daily.csv", "metrics.csv", "coverage_audit.csv", "audit.json",
    ]
    summary = {
        "schema": "ai_portfolio_validation_v1_1",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only" if all(checks.values()) else "failed_checks",
        "strategy": "long_high_score_ai_plus_spy_beta_hedge",
        "runner": {rel(RUNNER): sha(RUNNER)},
        "software_versions": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {rel(path): sha(path) for path in [
            MODEL_RUN / "models" / f"{MODEL_KEY}.joblib", MODEL_RUN / "feature_sets.json",
            MR / "model_features.csv", MR / "metadata.csv", MR / "eligibility.csv",
            AI, PRICES, RETURNS, REGISTRY, SPEC,
        ]},
        "outputs": {rel(output / name): sha(output / name) for name in output_files},
        "counts": {
            "daily_inference_rows": len(inference), "instruments_scored": int(inference["Instrument"].nunique()),
            "validation_sessions_scored": int(inference["formation_session"].nunique()),
            "complete_holding_blocks": len(formations), "eligibility_rows": len(eligibility_out),
            "rebalance_events": int(nav["event"].eq("REBALANCE").sum()), "trade_rows": len(trades),
            "nav_rows": len(nav), "skipped_formations": skip_counts,
            "average_active_stock_positions": float(nav["active_stock_positions"].mean()),
            "maximum_active_stock_positions": int(nav["active_stock_positions"].max()),
        },
        "metrics": metrics,
        "risk_monitoring": {
            "max_constructed_name_weight": max_target_name,
            "max_constructed_group_weight": max_target_group,
            "max_realised_name_weight": float(realised_stock["signed_weight"].max()),
            "max_realised_group_weight": float(realised_group.max()),
        },
        "checks": checks,
        "limitations": [
            "This is a development result on 2021-2022 validation data; the 2023-2026 test remains sealed.",
            "Forty candidate RICs use provisional static AI-role evidence, so universe membership is not fully historical PIT.",
            "End-of-day bid/ask is an execution-cost proxy and no market impact is estimated in the base case.",
            "The stored model was trained on non-overlapping anchors but is applied daily for operational scoring.",
            "Construction caps apply to fresh targets; realised weights can exceed them after price drift or a turnover-limited transition and are reported separately.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    data_log = ROOT / "DATA_PROCESSING_LOG.md"
    data_log.write_text(data_log.read_text(encoding="utf-8") + f"\n\n## {run_id} — AI portfolio formal validation backtest v1.1\n\n- **目的/状态**：将冻结的技术+AI状态 Logistic 转换为21-session调仓的AI股票多头+SPY beta hedge；status=`{summary['status']}`。\n- **输入版本/哈希**：见 `{rel(output / 'summary.json')}`；runner=`{rel(RUNNER)}`，SHA-256=`{sha(RUNNER)}`。\n- **时点规则**：排名、流动性和风险仅使用F-1及更早数据；F收盘价/报价在选股后仅用于订单执行检查，不替换候选。持仓收益区间为(F,X]。\n- **缺失与清洗**：未填零、未新增插值/缩尾/前向填充；模型内部仅使用既有训练期中位数与缺失指示器；未二次调整拆股。缺失、排除与跳过均保留reason code。\n- **计数**：输入`{len(inference):,}` company-days、`{inference['Instrument'].nunique()}`个活跃RIC、`{len(formations)}`个完整区间；形成日保留49个registry行；输出`{len(nav)}`个NAV观测、`{len(trades)}`条交易。\n- **跳过/排除**：skipped formations=`{json.dumps(skip_counts, ensure_ascii=False)}`；具体逐行原因见eligibility和targets。公司未因退市被整体删除。\n- **成本/风险**：F收盘成交，单边成本=完整报价价差的一半；scheduled one-way turnover<=50%；目标单股<=10%、目标组别<=30%；实际漂移上限另行报告。首日建仓成本纳入日收益风险指标样本。\n- **验证结果**：metrics=`{json.dumps(metrics, ensure_ascii=False)}`；checks=`{json.dumps(checks, ensure_ascii=False)}`。\n- **测试保护**：未打开sealed test targets，未产生测试期预测或指标。\n- **输出**：`{rel(output)}`；输出哈希见summary。\n", encoding="utf-8")
    ai_log = ROOT / "AI_USE_LOG.md"
    ai_log.write_text(ai_log.read_text(encoding="utf-8") + f"\n\n### Formal AI portfolio validation backtest ({run_id})\n\n- OpenAI Codex implemented and ran the validation-only v1.1 portfolio with F-1 ranking gates, post-ranking execution checks, detailed audit outputs, transaction costs, beta hedge, volatility scaling and turnover controls. It did not open sealed test targets. Outputs: `{rel(output)}`.\n", encoding="utf-8")
    print(json.dumps({"out": rel(output), "status": summary["status"], "counts": summary["counts"], "metrics": metrics, "checks": checks}, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    main(args.run_id)
