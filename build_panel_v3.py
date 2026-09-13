"""Build the v3 event/neighbour panel on the point-in-time universe.

The builder does not fit a model or claim a trading result. It keeps features,
forward labels, execution inputs and ex-post diagnostics in separate files.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RETURNS_CLEAN = ROOT / "data" / "clean" / "v2" / "returns.csv"
CLEAN = ROOT / "data" / "clean" / "v3" / "20260909T012417705069Z"
OUT = ROOT / "data" / "panel_v3"
INTERVALS = ROOT / "data" / "audit" / "universe_rebuild" / "membership_intervals.csv"
CONFIG_PATH = ROOT / "panel_v3_config.json"
EXPECTED_INTERVAL_ROWS = 876
EXPECTED_INTERVAL_RICS = 862


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_timestamps(values, field: str):
    """Parse values as supplied; do not convert an unverified timezone."""
    parsed = pd.to_datetime(values, format="mixed", errors="raise")
    accessor = getattr(parsed, "dt", None)
    tz = accessor.tz if accessor is not None else getattr(parsed, "tz", None)
    if tz is not None:
        raise ValueError(f"{field} has timezone-aware values; its timezone is not verified")
    return parsed


def load_membership_intervals(path: Path) -> pd.DataFrame:
    intervals = pd.read_csv(path, usecols=["ric", "start", "end"])
    intervals["start"] = parse_timestamps(intervals["start"], "membership start").dt.normalize()
    intervals["end"] = parse_timestamps(intervals["end"], "membership end").dt.normalize()
    if intervals[["ric", "start", "end"]].isna().any().any() or (intervals.end < intervals.start).any():
        raise ValueError("Invalid membership interval")
    return intervals


def membership_lookup(intervals: pd.DataFrame):
    """Return exact, inclusive calendar-day membership from corrected intervals."""
    cache = {}

    def members_on(day) -> set[str]:
        day = pd.Timestamp(day).normalize()
        if day not in cache:
            cache[day] = set(intervals.loc[(intervals.start <= day) & (day <= intervals.end), "ric"])
        return cache[day]

    return members_on


def pairwise_beta(asset: pd.Series, benchmark: pd.Series, min_observations: int):
    """OLS beta using only valid asset/benchmark pairs and that pair's denominator."""
    valid = asset.notna() & benchmark.notna()
    if int(valid.sum()) < min_observations:
        return None
    x, market = asset.loc[valid], benchmark.loc[valid]
    centered_market = market - market.mean()
    denominator = float((centered_market * centered_market).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    beta = float(((x - x.mean()) * centered_market).sum() / denominator)
    return beta if np.isfinite(beta) else None


def residual_correlations(wide: pd.DataFrame, day, lookback: int, min_observations: int, benchmark: str,
                          eligible_instruments: set[str] | None = None):
    """Use at most ``lookback`` strictly past sessions and pairwise-valid beta estimates."""
    history = wide.loc[wide.index < pd.Timestamp(day).normalize()].tail(lookback)
    if len(history) < min_observations or benchmark not in history:
        return None, None
    candidates = [column for column in history.columns if column != benchmark and
                  (eligible_instruments is None or column in eligible_instruments)]
    if len(candidates) < 2:
        return None, None
    market = history[benchmark]
    block = history[candidates]
    valid = block.notna().mul(market.notna(), axis=0)
    count = valid.sum()
    x = block.where(valid)
    sum_x = x.sum()
    sum_m = valid.mul(market, axis=0).sum()
    denominator = valid.mul(market.pow(2), axis=0).sum() - sum_m.pow(2) / count
    numerator = x.mul(market, axis=0).sum() - sum_x * sum_m / count
    beta = numerator / denominator
    beta = beta.where((count >= min_observations) & np.isfinite(denominator) & (denominator > 0) & np.isfinite(beta))
    beta = beta.dropna()
    if len(beta) < 2:
        return None, None
    fitted = pd.DataFrame(np.outer(market.to_numpy(), beta.to_numpy()), index=history.index, columns=beta.index)
    residuals = block[beta.index] - fitted
    return residuals.corr(min_periods=min_observations), history.index.max()


def pick_neighbours(correlation: pd.Series, source: str, eligible_receivers: set[str], selection: dict) -> pd.Series:
    """Filter announcement-day members before applying the approved selection rule."""
    chosen = correlation.drop(labels=[source], errors="ignore").dropna()
    chosen = chosen[chosen.index.isin(eligible_receivers)]
    if selection["mode"] == "threshold":
        chosen = chosen[chosen.abs() >= selection["min_abs_correlation"]]
    elif selection["mode"] == "topk_with_threshold":
        chosen = chosen[chosen.abs() >= selection["min_abs_correlation"]]
        chosen = chosen.reindex(chosen.abs().sort_values(ascending=False).index).head(selection["max_neighbors"])
    else:  # topk
        chosen = chosen.reindex(chosen.abs().sort_values(ascending=False).index).head(selection["max_neighbors"])
    return chosen[chosen > 0] if selection.get("positive_only") else chosen


def entry_position(sessions: pd.DatetimeIndex, announcement_day) -> int:
    """Formal rule: enter at the close of the first session after the calendar announcement day."""
    day = pd.Timestamp(announcement_day).normalize()
    return int(sessions.searchsorted(day, side="right"))


def forward_label(wide: pd.DataFrame, sessions: pd.DatetimeIndex, entry_i: int, horizon: int,
                  receiver: str, benchmark: str) -> dict:
    """Return a label record even when the requested future window is unavailable."""
    record = {"entry_session": pd.NaT, "exit_session": pd.NaT, "label_complete": False,
              "receiver_forward_return": np.nan, "benchmark_forward_return": np.nan,
              "forward_benchmark_excess": np.nan}
    if entry_i >= len(sessions):
        return record
    record["entry_session"] = sessions[entry_i]
    exit_i = entry_i + horizon
    if exit_i >= len(sessions):
        return record
    record["exit_session"] = sessions[exit_i]
    future = wide.iloc[entry_i + 1:exit_i + 1]
    benchmark_complete = len(future) == horizon and future[benchmark].notna().all()
    receiver_complete = len(future) == horizon and future[receiver].notna().all()
    if benchmark_complete:
        record["benchmark_forward_return"] = float((1 + future[benchmark]).prod() - 1)
    if receiver_complete:
        record["receiver_forward_return"] = float((1 + future[receiver]).prod() - 1)
    record["label_complete"] = bool(benchmark_complete and receiver_complete)
    if record["label_complete"]:
        record["forward_benchmark_excess"] = record["receiver_forward_return"] - record["benchmark_forward_return"]
    return record


def receiver_diagnostics(all_actual_days: dict, receiver: str, announcement_day, exit_session):
    """Use all clean actuals, including events after the configured event end."""
    own = all_actual_days.get(receiver, pd.DatetimeIndex([]))
    if pd.isna(exit_session) or not len(own):
        overlap = None
    else:
        overlap = bool(((own >= pd.Timestamp(announcement_day).normalize()) & (own <= exit_session)).any())
    return {"receiver_event_coverage_present": bool(len(own)),
            "ex_post_own_announcement_overlap": overlap,
            "diagnostic_actuals_through_label_horizon": exit_session}


def event_window(actuals: pd.DataFrame, config: dict) -> pd.DataFrame:
    start = pd.Timestamp(config["event_start"]).normalize()
    end = pd.Timestamp(config["event_end"]).normalize()
    return actuals.loc[actuals.announcement_day.between(start, end, inclusive="both")].copy()


def lookup_price(price_index: pd.DataFrame, instrument: str, session) -> dict:
    """Return the observed row for an exact instrument/session key; never fill across dates."""
    empty = {"price_row_available": False, "close": np.nan, "volume": np.nan,
             "quoted_spread_bps": np.nan, "dollar_volume": np.nan,
             "dollar_volume_source": None, "has_close": False, "has_volume": False,
             "has_two_sided_quote": False, "in_sp500_that_day": None,
             "price_adjustments": None}
    if pd.isna(session) or (instrument, pd.Timestamp(session)) not in price_index.index:
        return empty
    row = price_index.loc[(instrument, pd.Timestamp(session))]
    if isinstance(row, pd.DataFrame):
        raise RuntimeError(f"Duplicate price key: {instrument} {session}")
    return {"price_row_available": True, "close": row["TRDPRC_1"], "volume": row["ACVOL_UNS"],
            "quoted_spread_bps": row["quoted_spread_bps"], "dollar_volume": row["dollar_volume"],
            "dollar_volume_source": row["dollar_volume_source"], "has_close": bool(row["has_close"]),
            "has_volume": bool(row["has_volume"]), "has_two_sided_quote": bool(row["has_two_sided_quote"]),
            "in_sp500_that_day": bool(row["in_sp500_that_day"]), "price_adjustments": row["price_adjustments"]}


def read_inputs():
    paths = {"returns_clean": RETURNS_CLEAN, "actuals_clean": CLEAN / "actuals.csv",
             "estimates_clean": CLEAN / "estimates_weekly.csv", "prices_clean": CLEAN / "prices.csv",
             "membership_intervals": INTERVALS,
             "panel_config": CONFIG_PATH, "builder_code": Path(__file__).resolve()}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))
    hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    returns = pd.read_csv(paths["returns_clean"], usecols=["Instrument", "Date", "return_decimal"])
    returns["Date"] = parse_timestamps(returns["Date"], "returns Date").dt.normalize()
    actuals = pd.read_csv(paths["actuals_clean"]).rename(columns={"eps_actual": "actual"})
    actuals["announcement"] = parse_timestamps(actuals["announcement"], "announcement")
    actuals["announcement_day"] = parse_timestamps(actuals["announcement_day"], "announcement_day").dt.normalize()
    actuals["period_end"] = parse_timestamps(actuals["period_end"], "period_end").dt.normalize()
    if not actuals.announcement.dt.normalize().equals(actuals.announcement_day):
        raise RuntimeError("announcement_day disagrees with the supplied announcement calendar day")
    if set(actuals.formal_entry_rule.dropna()) != {"first_market_session_strictly_after_announcement_day"}:
        raise RuntimeError("Actuals do not carry the approved formal entry rule")
    estimates = pd.read_csv(paths["estimates_clean"]).rename(columns={
        "Earnings Per Share - Mean": "consensus", "Earnings Per Share - Standard Deviation": "dispersion",
        "Earnings Per Share - Number of Included Estimates": "analysts", "Period End Date": "period_end"})
    estimates["snapshot"] = parse_timestamps(estimates["snapshot"], "snapshot")
    estimates["period_end"] = parse_timestamps(estimates["period_end"], "estimate period_end").dt.normalize()
    prices = pd.read_csv(paths["prices_clean"], usecols=[
        "Instrument", "Date", "TRDPRC_1", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume",
        "dollar_volume_source", "has_close", "has_volume", "has_two_sided_quote",
        "in_sp500_that_day", "price_adjustments"])
    prices["Date"] = parse_timestamps(prices["Date"], "price Date").dt.normalize()
    if prices.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("Duplicate clean price instrument/session keys")
    intervals = load_membership_intervals(paths["membership_intervals"])
    if len(intervals) != EXPECTED_INTERVAL_ROWS or intervals.ric.nunique() != EXPECTED_INTERVAL_RICS or "EVHC.N^L16" in set(intervals.ric):
        raise RuntimeError("Membership input is not the corrected 876-row / 862-RIC interval snapshot")
    return config, returns, actuals, estimates, prices, intervals, paths, hashes


def main():
    config, returns, all_actuals, estimates, prices, intervals, paths, input_hashes = read_inputs()
    benchmark = config["benchmark"]
    lookback, min_obs = config["graph_lookback_sessions"], config["graph_min_observations"]
    horizon, max_age, selection = config["forward_sessions"], config["max_snapshot_age_days"], config["neighbor_selection"]
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    if benchmark not in wide:
        raise RuntimeError(f"Missing benchmark {benchmark}")
    wide = wide.loc[wide[benchmark].notna()]
    sessions = wide.index
    price_index = prices.set_index(["Instrument", "Date"], verify_integrity=True).sort_index()
    members_on = membership_lookup(intervals)
    actuals = event_window(all_actuals, config)
    estimates = estimates.sort_values("snapshot")
    estimate_groups = {key: group for key, group in estimates.groupby(["Instrument", "period_end"], sort=False)}
    actual_days = {key: pd.DatetimeIndex(group.announcement.dt.normalize())
                   for key, group in all_actuals.groupby("Instrument", sort=False)}
    print(f"sessions {len(sessions)}  instruments {wide.shape[1]}", flush=True)

    event_rows, edges = [], []
    cache_day, cache_corr, cache_last = None, None, None
    for number, event in enumerate(actuals.sort_values("announcement").itertuples(), 1):
        if number % 2000 == 0:
            print(f"  {number}/{len(actuals)} events, {len(edges)} edges", flush=True)
        source, day = event.Instrument, event.announcement_day
        day_members = members_on(day)
        row = {"source": source, "announcement": event.announcement, "period_end": event.period_end,
               "announcement_day": day, "actual": event.actual,
               "actual_source": event.actual_source, "actual_selector": event.actual_selector,
               "formal_entry_rule": event.formal_entry_rule,
               "source_in_index_on_announcement": source in day_members}
        if not row["source_in_index_on_announcement"]:
            row["status"] = "source_not_in_index_on_announcement"; event_rows.append(row); continue
        snapshots = estimate_groups.get((source, event.period_end))
        if snapshots is None:
            row["status"] = "no_estimates_for_quarter"; event_rows.append(row); continue
        prior = snapshots[snapshots.snapshot < day]
        if prior.empty:
            row["status"] = "no_preannouncement_snapshot"; event_rows.append(row); continue
        latest = prior.iloc[-1]
        age = int((day - latest.snapshot.normalize()).days)
        row.update(snapshot=latest.snapshot, consensus=latest.consensus, dispersion=latest.dispersion,
                   analysts=latest.analysts, snapshot_age_days=age)
        if age > max_age:
            row["status"] = "stale_snapshot"; event_rows.append(row); continue
        row["status"] = "matched_snapshot"
        dispersion = pd.to_numeric(pd.Series([latest.dispersion]), errors="coerce").iloc[0]
        row["eps_difference"] = event.actual - latest.consensus
        row["standardized_surprise"] = row["eps_difference"] / dispersion if np.isfinite(dispersion) and dispersion > 0 else np.nan
        if day != cache_day:
            cache_corr, cache_last = residual_correlations(wide, day, lookback, min_obs, benchmark, day_members)
            cache_day = day
        if cache_corr is None or source not in cache_corr.columns:
            row["graph_status"] = "insufficient_graph_history"; event_rows.append(row); continue
        selected = pick_neighbours(cache_corr[source], source, day_members, selection)
        row.update(graph_status="available", graph_last_date=cache_last, neighbor_count=len(selected))
        entry_i = entry_position(sessions, day)
        if entry_i < len(sessions) and sessions[entry_i] <= day:
            row["entry_status"] = "entry_not_strictly_after_announcement_day"; event_rows.append(row); continue
        row["entry_status"] = "available" if entry_i < len(sessions) else "entry_session_unavailable"
        if entry_i < len(sessions):
            row["entry_session"] = sessions[entry_i]
            row["liquidity_reference_session"] = sessions[entry_i - config["liquidity_feature_lag_sessions"]] \
                if entry_i >= config["liquidity_feature_lag_sessions"] else pd.NaT
        event_rows.append(row)
        for receiver, correlation in selected.items():
            edge = dict(row, receiver=receiver, residual_correlation=float(correlation),
                        receiver_in_index_on_announcement=True)
            labels = forward_label(wide, sessions, entry_i, horizon, receiver, benchmark)
            edge.update(labels)
            edge["receiver_in_index_at_entry"] = (receiver in members_on(labels["entry_session"])
                                                   if pd.notna(labels["entry_session"]) else None)
            lagged = lookup_price(price_index, receiver, row.get("liquidity_reference_session", pd.NaT))
            edge.update({f"lagged_{key}": value for key, value in lagged.items()})
            entry_px = lookup_price(price_index, receiver, labels["entry_session"])
            edge.update({f"entry_{key}": value for key, value in entry_px.items()})
            edge.update(receiver_diagnostics(actual_days, receiver, day, labels["exit_session"]))
            edges.append(edge)

    events, panel = pd.DataFrame(event_rows), pd.DataFrame(edges)
    matched = events.loc[events.status.eq("matched_snapshot")].copy() if len(events) else pd.DataFrame()
    graph_ready = events.loc[events.graph_status.eq("available")] if len(events) and "graph_status" in events else pd.DataFrame()
    checks = {
        "corrected_membership_interval_shape": len(intervals) == EXPECTED_INTERVAL_ROWS and intervals.ric.nunique() == EXPECTED_INTERVAL_RICS,
        "event_window_nonempty": bool(len(events)),
        "snapshots_strictly_before_event_day": bool(len(matched) and (matched.snapshot < matched.announcement_day).all()),
        "graph_history_strictly_before_event_day": bool(len(graph_ready) and (graph_ready.graph_last_date < graph_ready.announcement_day).all()),
        "entry_strictly_after_announcement_day": bool(len(graph_ready) and (
            graph_ready.loc[graph_ready.entry_status.eq("available"), "entry_session"] >
            graph_ready.loc[graph_ready.entry_status.eq("available"), "announcement_day"]).all()),
        "receiver_membership_is_exact_announcement_day": bool(len(panel) == 0 or panel.receiver_in_index_on_announcement.all()),
        "no_self_edges": bool(len(panel) == 0 or (panel.source != panel.receiver).all()),
        "price_keys_unique": not prices.duplicated(["Instrument", "Date"]).any(),
        "lagged_liquidity_precedes_entry": bool(len(panel) == 0 or (
            panel.loc[panel.entry_session.notna(), "liquidity_reference_session"] <
            panel.loc[panel.entry_session.notna(), "entry_session"]).all()),
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT / run_id
    out.mkdir(parents=True, exist_ok=False)
    events.to_csv(out / "events.csv", index=False)
    output_files = {"events": out / "events.csv"}
    if len(panel):
        panel["sample_id"] = [hashlib.sha256(f"{s}|{a}|{p}|{r}".encode()).hexdigest()[:24]
                              for s, a, p, r in zip(panel.source, panel.announcement, panel.period_end, panel.receiver)]
        if panel.sample_id.duplicated().any():
            raise RuntimeError("Duplicate sample IDs")
        feature_cols = ["sample_id", "source", "receiver", "announcement", "announcement_day", "period_end",
                        "actual_source", "actual_selector", "formal_entry_rule",
                        "snapshot", "snapshot_age_days", "actual", "consensus", "dispersion", "analysts",
                        "eps_difference", "standardized_surprise", "graph_last_date", "residual_correlation",
                        "liquidity_reference_session", "lagged_price_row_available", "lagged_close", "lagged_volume",
                        "lagged_quoted_spread_bps", "lagged_dollar_volume", "lagged_dollar_volume_source",
                        "lagged_has_close", "lagged_has_volume", "lagged_has_two_sided_quote",
                        "lagged_in_sp500_that_day", "lagged_price_adjustments"]
        label_cols = ["sample_id", "entry_session", "exit_session", "label_complete", "receiver_forward_return",
                      "benchmark_forward_return", "forward_benchmark_excess"]
        diagnostic_cols = ["sample_id", "receiver_in_index_on_announcement", "receiver_in_index_at_entry",
                           "receiver_event_coverage_present", "ex_post_own_announcement_overlap",
                           "diagnostic_actuals_through_label_horizon"]
        execution_cols = ["sample_id", "entry_session", "entry_price_row_available", "entry_close", "entry_volume",
                          "entry_quoted_spread_bps", "entry_dollar_volume", "entry_dollar_volume_source",
                          "entry_has_close", "entry_has_volume", "entry_has_two_sided_quote",
                          "entry_in_sp500_that_day", "entry_price_adjustments"]
        for name, columns in (("features", feature_cols), ("labels", label_cols),
                              ("execution_inputs", execution_cols), ("diagnostics_ex_post", diagnostic_cols)):
            path = out / f"{name}.csv"; panel[columns].to_csv(path, index=False); output_files[name] = path
    after_hashes = {name: sha256_file(path) for name, path in paths.items()}
    before_hashes = {name: item["sha256"] for name, item in input_hashes.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("An input changed while building the panel")
    summary = {
        "run_id": run_id, "output_dir": rel(out), "engineering_only": True,
        "announcement_timezone": "vendor clock remains naive; calendar day was validated against AAPL UTC reference samples",
        "entry_rule_status": "formal conservative rule: first benchmark session strictly after announcement calendar day",
        "config": config, "input_hashes": input_hashes,
        "membership_provenance": {"path": rel(INTERVALS), "sha256": input_hashes["membership_intervals"]["sha256"],
                                  "rows": int(len(intervals)), "unique_rics": int(intervals.ric.nunique()),
                                  "rule": "inclusive start <= announcement day <= end for source and receiver"},
        "event_status_counts": events.status.value_counts().to_dict() if len(events) else {},
        "matched_snapshot_events": int(len(matched)), "graph_ready_events": int(len(graph_ready)),
        "selected_edge_rows": int(len(panel)), "complete_label_rows": int(panel.label_complete.sum()) if len(panel) else 0,
        "price_input_rows": int(len(prices)),
        "lagged_price_row_missing": int((~panel.lagged_price_row_available).sum()) if len(panel) else 0,
        "entry_price_row_missing": int((~panel.entry_price_row_available).sum()) if len(panel) else 0,
        "entry_close_missing": int(panel.entry_close.isna().sum()) if len(panel) else 0,
        "entry_volume_missing": int(panel.entry_volume.isna().sum()) if len(panel) else 0,
        "entry_quote_missing": int(panel.entry_quoted_spread_bps.isna().sum()) if len(panel) else 0,
        "entry_dollar_volume_missing": int(panel.entry_dollar_volume.isna().sum()) if len(panel) else 0,
        "checks": checks,
        "feature_schema_status": "Only t0-1 or earlier liquidity fields are in features; t0 close/volume/spread are isolated in execution_inputs.csv",
        "missing_data_policy": "No forward fill, backward fill, zero fill, interpolation, or silent row deletion",
        "limitations": ["Residual correlation is statistical association, not a causal link.",
                        "Future labels and all-actuals overlap are not used for graph or edge selection.",
                        "Weekly consensus is the verified core frequency; daily consensus remains a pilot only.",
                        "Forward total returns are labels, while entry close and liquidity are execution diagnostics; costs are not yet applied.",
                        "No model is fitted and no portfolio is backtested."],
        "output_hashes": {name: sha256_file(path) for name, path in output_files.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"config", "input_hashes", "limitations"}}, ensure_ascii=False, indent=2, default=str))
    if not all(checks.values()):
        raise RuntimeError("Panel checks failed: " + ", ".join(k for k, value in checks.items() if not value))


if __name__ == "__main__":
    main()
