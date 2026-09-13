"""Independent validation of the next-session-open diagnostic run."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "data" / "analysis" / "next_open_target_diagnostics_v1"
MODEL_READY = ROOT / "data" / "model_ready_v1" / "20260909T064151673764Z"
PRICES = ROOT / "data" / "clean" / "v3" / "20260909T012417705069Z" / "prices.csv"
AUDIT_ROOT = ROOT / "data" / "audit" / "next_open_targets_v1"
EVENT_KEY = ["source", "announcement", "period_end"]
HORIZONS = [0, 1, 2, 3, 5, 10]
REFIT_HORIZONS = [1, 5]
SPLIT_END = {"training": pd.Timestamp("2021-01-01"), "validation": pd.Timestamp("2023-01-01")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(values) -> pd.Series:
    values = pd.Series(values).astype("string").str.lower()
    if set(values.dropna()) - {"true", "false"} or values.isna().any():
        raise ValueError("Invalid boolean")
    return values.eq("true")


def event_weights(frame) -> np.ndarray:
    return (1 / frame.groupby(EVENT_KEY).sample_id.transform("size")).to_numpy(float)


def recompute_metrics(frame, prediction) -> dict:
    y = frame.target.to_numpy(float)
    weight = event_weights(frame)
    error = prediction - y
    event_ics = []
    for _, group in frame.assign(prediction=prediction).groupby(EVENT_KEY):
        if len(group) >= 3 and group.target.nunique() > 1 and group.prediction.nunique() > 1:
            value = spearmanr(group.target, group.prediction).statistic
            if np.isfinite(value):
                event_ics.append(float(value))
    pooled = spearmanr(y, prediction).statistic if len(np.unique(prediction)) > 1 else np.nan
    mean_y = np.average(y, weights=weight)
    return {
        "rows": int(len(frame)),
        "events": int(frame[EVENT_KEY].drop_duplicates().shape[0]),
        "event_weighted_mse": float(np.average(error ** 2, weights=weight)),
        "event_weighted_rmse": float(np.sqrt(np.average(error ** 2, weights=weight))),
        "event_weighted_mae": float(np.average(np.abs(error), weights=weight)),
        "event_weighted_direction_accuracy": float(np.average(np.sign(prediction) == np.sign(y), weights=weight)),
        "weighted_r2": float(1 - np.average(error ** 2, weights=weight) /
                             np.average((y - mean_y) ** 2, weights=weight)),
        "pooled_spearman": float(pooled) if np.isfinite(pooled) else np.nan,
        "mean_event_spearman": float(np.mean(event_ics)) if event_ics else np.nan,
        "event_spearman_groups": int(len(event_ics)),
    }


def raw_rank_metrics(frame, signal, target) -> dict:
    usable = frame.loc[frame[signal].notna() & frame[target].notna()]
    event_ics = []
    for _, group in usable.groupby(EVENT_KEY):
        if len(group) >= 3 and group[signal].nunique() > 1 and group[target].nunique() > 1:
            value = spearmanr(group[signal], group[target]).statistic
            if np.isfinite(value):
                event_ics.append(value)
    return {
        "rows": int(len(usable)),
        "events": int(usable[EVENT_KEY].drop_duplicates().shape[0]),
        "pooled_spearman": float(spearmanr(usable[signal], usable[target]).statistic),
        "mean_event_spearman": float(np.mean(event_ics)) if event_ics else np.nan,
        "event_ic_groups": int(len(event_ics)),
    }


def ridge(alpha):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("regressor", Ridge(alpha=alpha))])


def compare(reported_row, recomputed: dict, rtol: float, atol: float) -> bool:
    if len(reported_row) != 1:
        return False
    row = reported_row.iloc[0]
    for key, value in recomputed.items():
        if isinstance(value, int):
            if int(row[key]) != value:
                return False
        elif not np.isclose(float(row[key]), value, rtol=rtol, atol=atol, equal_nan=True):
            return False
    return True


def latest_run() -> Path:
    runs = sorted(path for path in RUN_ROOT.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"no diagnostic run under {RUN_ROOT}")
    return runs[-1]


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    stored = pd.read_csv(run / "open_targets_and_availability.csv", low_memory=False)
    reported_metrics = pd.read_csv(run / "metrics.csv")
    reported_audit = pd.read_csv(run / "target_audit.csv")
    reported_predictions = pd.read_csv(run / "validation_predictions.csv")
    checks = {}

    checks["run_outputs_match_recorded_hashes"] = all(
        sha256_file(run / name) == digest for name, digest in summary["output_hashes"].items())
    checks["input_files_match_recorded_hashes"] = all(
        sha256_file(path) == summary["input_hashes"][name] for name, path in [
            ("features", MODEL_READY / "model_features.csv"),
            ("metadata", MODEL_READY / "metadata.csv"),
            ("eligibility", MODEL_READY / "eligibility.csv"),
            ("prices", PRICES)])

    features = pd.read_csv(MODEL_READY / "model_features.csv")
    eligibility = pd.read_csv(MODEL_READY / "eligibility.csv", low_memory=False)
    for column in ["split_retained", "entry_trade_eligible", "feature_core_available"]:
        eligibility[column] = parse_bool(eligibility[column]).to_numpy()
    expected_ids = eligibility.loc[
        eligibility.split.isin(["training", "validation"]) & eligibility.split_retained &
        eligibility.entry_trade_eligible & eligibility.feature_core_available, "sample_id"]
    checks["row_scope_matches_independent_eligibility_filter"] = bool(
        set(stored.sample_id) == set(expected_ids) and len(stored) == len(expected_ids))
    checks["sample_ids_unique"] = not stored.sample_id.duplicated().any()
    checks["no_test_rows"] = not stored.split.eq("test").any()
    checks["splits_are_training_and_validation_only"] = set(stored.split) == {"training", "validation"}

    stored["announcement_day"] = pd.to_datetime(stored.announcement_day).dt.normalize()
    stored["entry_session"] = pd.to_datetime(stored.entry_session).dt.normalize()
    checks["entry_strictly_after_announcement_day"] = bool(
        stored.entry_session.gt(stored.announcement_day).all())
    checks["announcement_days_inside_declared_split_windows"] = bool(
        stored.loc[stored.split.eq("training"), "announcement_day"].between(
            "2015-01-01", "2020-12-31").all() and
        stored.loc[stored.split.eq("validation"), "announcement_day"].between(
            "2021-01-01", "2022-12-31").all())

    prices = pd.read_csv(PRICES, usecols=["Instrument", "Date", "OPEN_PRC", "TRDPRC_1"], low_memory=False)
    prices["Date"] = pd.to_datetime(prices.Date).dt.normalize()
    checks["clean_price_keys_unique"] = not prices.duplicated(["Instrument", "Date"]).any()
    prices = prices.rename(columns={"OPEN_PRC": "open_px", "TRDPRC_1": "close_px"})
    spy = prices.loc[prices.Instrument.eq("SPY.P")].sort_values("Date")
    sessions = pd.DatetimeIndex(np.sort(spy.loc[spy.close_px.notna(), "Date"].unique()))
    session_values = sessions.to_numpy()
    entry_values = stored.entry_session.to_numpy()
    position = np.searchsorted(session_values, entry_values)
    checks["entry_session_is_an_spy_session"] = bool(
        (position < len(sessions)).all() and
        (session_values[np.clip(position, 0, len(sessions) - 1)] == entry_values).all())

    receiver_open = stored[["receiver", "entry_session"]].merge(
        prices[["Instrument", "Date", "open_px"]], left_on=["receiver", "entry_session"],
        right_on=["Instrument", "Date"], how="left", validate="many_to_one").open_px.to_numpy()
    spy_open = spy.set_index("Date").open_px
    spy_close = spy.set_index("Date").close_px
    benchmark_open = spy_open.reindex(stored.entry_session).to_numpy()
    checks["receiver_entry_open_recomputed"] = bool(np.array_equal(
        receiver_open, stored.receiver_entry_open.to_numpy(float), equal_nan=True))
    checks["benchmark_entry_open_recomputed"] = bool(np.array_equal(
        benchmark_open, stored.benchmark_entry_open.to_numpy(float), equal_nan=True))
    entry_open_available = (np.nan_to_num(receiver_open, nan=-1.0) > 0) & (
        np.nan_to_num(benchmark_open, nan=-1.0) > 0)
    checks["entry_open_available_recomputed"] = bool(
        (entry_open_available == parse_bool(stored.entry_open_available).to_numpy()).all())

    limit = stored.split.map(SPLIT_END)
    audit_recomputed, exit_checks, purge_checks, avail_checks, target_checks = [], [], [], [], []
    for horizon in HORIZONS:
        exit_position = position + horizon
        valid = exit_position < len(sessions)
        exits = np.full(len(stored), np.datetime64("NaT", "ns"))
        exits[valid] = session_values[exit_position[valid]]
        exits = pd.Series(exits, index=stored.index)
        stored_exit = pd.to_datetime(stored[f"exit_session_{horizon}"]).dt.normalize()
        exit_checks.append(bool(exits.eq(stored_exit).eq(exits.notna()).all() and
                                exits.isna().eq(stored_exit.isna()).all()))
        receiver_close = stored[["receiver"]].assign(exit_session=exits).merge(
            prices[["Instrument", "Date", "close_px"]], left_on=["receiver", "exit_session"],
            right_on=["Instrument", "Date"], how="left", validate="many_to_one").close_px.to_numpy()
        benchmark_close = spy_close.reindex(exits).to_numpy()
        boundary = (exits.notna() & exits.ge(limit)).to_numpy()
        purge_checks.append(bool((boundary == parse_bool(stored[f"boundary_purged_{horizon}"]).to_numpy()).all()))
        exit_price_ok = (np.nan_to_num(receiver_close, nan=-1.0) > 0) & (
            np.nan_to_num(benchmark_close, nan=-1.0) > 0)
        available = valid & ~boundary & entry_open_available & exit_price_ok
        avail_checks.append(bool(
            (available == parse_bool(stored[f"target_available_{horizon}"]).to_numpy()).all()))
        target = np.full(len(stored), np.nan)
        target[available] = ((receiver_close[available] / receiver_open[available] - 1) -
                             (benchmark_close[available] / benchmark_open[available] - 1))
        stored_target = stored[f"target_open_to_close_{horizon}"].to_numpy(float)
        target_checks.append(bool(
            (np.isnan(target) == np.isnan(stored_target)).all() and
            np.allclose(target[available], stored_target[available], rtol=1e-12, atol=1e-15)))
        audit_recomputed.append({
            "horizon": horizon,
            "rows": int(len(stored)),
            "entry_open_missing": int((~entry_open_available).sum()),
            "exit_close_missing_or_nonpositive": int((~exit_price_ok).sum()),
            "boundary_purged": int(boundary.sum()),
            "target_available": int(available.sum()),
        })
    checks["exit_sessions_recomputed_all_horizons"] = all(exit_checks)
    checks["boundary_purge_recomputed_all_horizons"] = all(purge_checks)
    checks["target_availability_recomputed_all_horizons"] = all(avail_checks)
    checks["targets_recomputed_all_horizons"] = all(target_checks)
    checks["target_audit_recomputed"] = bool(
        reported_audit.sort_values("horizon").reset_index(drop=True).astype(int).equals(
            pd.DataFrame(audit_recomputed).sort_values("horizon").reset_index(drop=True).astype(int)))
    checks["target_available_flag_equals_target_notna"] = all(
        bool((parse_bool(stored[f"target_available_{horizon}"]).to_numpy() ==
              stored[f"target_open_to_close_{horizon}"].notna().to_numpy()).all())
        for horizon in HORIZONS)
    checks["available_targets_all_finite"] = all(
        bool(np.isfinite(stored.loc[parse_bool(stored[f"target_available_{horizon}"]).to_numpy(),
                                    f"target_open_to_close_{horizon}"].to_numpy(float)).all())
        for horizon in HORIZONS)
    checks["used_exits_never_cross_split_boundary"] = all(
        bool(pd.to_datetime(stored.loc[mask, f"exit_session_{horizon}"]).lt(
            stored.loc[mask, "split"].map(SPLIT_END)).all())
        for horizon in HORIZONS
        for mask in [parse_bool(stored[f"target_available_{horizon}"]).to_numpy()])

    panel = stored.merge(features, on="sample_id", validate="one_to_one")
    all_features = [column for column in features if column != "sample_id"]
    receiver_features = [column for column in all_features
                         if column.startswith("receiver_") or column.startswith("market_")]
    feature_sets = {"simple_network": ["network_signal"], "receiver_ridge": receiver_features,
                    "network_ridge": all_features}
    factories = {"simple_network": lambda: Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                     ("regressor", LinearRegression())]),
                 "receiver_ridge": lambda: ridge(0.01), "network_ridge": lambda: ridge(100.0)}

    raw_checks, refit_prediction_checks, refit_metric_checks = [], [], []
    for horizon in HORIZONS:
        column = f"target_open_to_close_{horizon}"
        for split_name in ["training", "validation"]:
            frame = panel.loc[panel.split.eq(split_name) & panel[column].notna()].rename(
                columns={column: "target"})
            recomputed = raw_rank_metrics(frame, "network_signal", "target")
            raw_checks.append(compare(
                reported_metrics.loc[reported_metrics.split.eq(split_name) &
                                     reported_metrics.horizon.eq(horizon) &
                                     reported_metrics.model.eq("raw_network_signal")],
                recomputed, rtol=1e-11, atol=1e-13))
    checks["raw_network_signal_metrics_recomputed"] = all(raw_checks)

    for horizon in REFIT_HORIZONS:
        column = f"target_open_to_close_{horizon}"
        train = panel.loc[panel.split.eq("training") & panel[column].notna()].rename(columns={column: "target"})
        valid = panel.loc[panel.split.eq("validation") & panel[column].notna()].rename(columns={column: "target"})
        for model_name, factory in factories.items():
            model = factory()
            columns = feature_sets[model_name]
            model.fit(train[columns], train.target, regressor__sample_weight=event_weights(train))
            prediction = model.predict(valid[columns])
            saved = reported_predictions.loc[reported_predictions.horizon.eq(horizon) &
                                             reported_predictions.model.eq(model_name)]
            aligned = saved.set_index("sample_id").reindex(valid.sample_id)
            refit_prediction_checks.append(bool(
                len(saved) == len(valid) and aligned.prediction.notna().all() and
                np.allclose(aligned.prediction.to_numpy(), prediction, rtol=1e-9, atol=1e-12) and
                np.allclose(aligned.target.to_numpy(), valid.target.to_numpy(), rtol=1e-12, atol=1e-15)))
            refit_metric_checks.append(compare(
                reported_metrics.loc[reported_metrics.split.eq("validation") &
                                     reported_metrics.horizon.eq(horizon) &
                                     reported_metrics.model.eq(model_name)],
                recompute_metrics(valid, prediction), rtol=1e-9, atol=1e-12))
    checks["independent_refit_reproduces_validation_predictions"] = all(refit_prediction_checks)
    checks["independent_refit_reproduces_validation_metrics"] = all(refit_metric_checks)

    validation_ids = set(stored.loc[stored.split.eq("validation"), "sample_id"])
    checks["stored_predictions_are_validation_rows_only"] = set(
        reported_predictions.sample_id.unique()).issubset(validation_ids)
    checks["reported_metrics_cover_all_horizons_and_models"] = bool(
        set(reported_metrics.horizon) == set(HORIZONS) and
        set(reported_metrics.model) == {"raw_network_signal", *factories} and
        set(reported_metrics.split) == {"training", "validation"})

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = AUDIT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "run_id": run_id,
        "diagnostic_run": run.name,
        "validator": "validate_next_open_targets_v1.py",
        "scope": "independent recomputation of next-open entry prices, targets, availability, "
                 "boundary purges, raw-signal metrics and refitted validation metrics",
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "all_passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "recomputed_target_audit": audit_recomputed,
        "refit_horizons": REFIT_HORIZONS,
        "validator_hash": sha256_file(Path(__file__).resolve()),
        "test_policy": "No test rows, targets, predictions or metrics were computed.",
    }
    (out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ["run_id", "diagnostic_run", "passed", "total", "all_passed"]}, indent=2))
    print(json.dumps(report["checks"], indent=2))
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    main()
