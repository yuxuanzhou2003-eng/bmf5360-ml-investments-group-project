"""Build auditable pre-formation AI industry state features.

This builder is deliberately scoped to the AI industry state portion of the
frozen AI_DAILY_MACRO_FEATURE_SPEC.md v1.0. It reads only the 49-RIC daily
model-feature membership grid, clean historical SPY/pool prices, and the five
raw ETF price files. For a formation session t, every value is looked up at
the strictly previous SPY session. No target, label, future return, or model
run is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "ai_daily_macro_feature_v1_config.json"
SCHEMA_VERSION = "ai_daily_macro_feature_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def parse_dates(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, format="mixed", errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{name} has {int(parsed.isna().sum())} invalid dates")
    return parsed


def finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) and not isinstance(value, (str, bytes)):
        return None
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False, date_format="%Y-%m-%d")
    temp.replace(path)


def read_model_features(path: Path, expected_instruments: int) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"Instrument", "formation_session"}
    missing = sorted(required - set(header))
    if missing:
        raise RuntimeError(f"model_features schema missing: {missing}")
    # Read only the membership/date grid. Existing derived feature values are
    # intentionally not used, which keeps this stage independent of model use.
    frame = pd.read_csv(path, usecols=["Instrument", "formation_session"], low_memory=False)
    frame["Instrument"] = frame["Instrument"].astype("string").str.strip()
    if frame["Instrument"].isna().any() or frame["Instrument"].eq("").any():
        raise RuntimeError("model_features has blank Instrument values")
    frame["formation_session"] = parse_dates(frame["formation_session"], "model_features.formation_session")
    if frame.duplicated(["Instrument", "formation_session"]).any():
        raise RuntimeError("model_features has duplicate Instrument/formation_session keys")
    n_instruments = int(frame["Instrument"].nunique())
    if n_instruments != expected_instruments:
        raise RuntimeError(f"expected {expected_instruments} pool instruments, found {n_instruments}")
    return frame.sort_values(["formation_session", "Instrument"], kind="stable").reset_index(drop=True)


def read_price_subset(path: Path, instruments: set[str], benchmark: str) -> tuple[pd.DataFrame, int]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"Instrument", "Date", "TRDPRC_1"}
    missing = sorted(required - set(header))
    if missing:
        raise RuntimeError(f"prices schema missing: {missing}")
    usecols = ["Instrument", "Date", "TRDPRC_1"]
    parts: list[pd.DataFrame] = []
    source_rows = 0
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=500_000):
        source_rows += len(chunk)
        selected = chunk.loc[chunk["Instrument"].astype(str).isin(instruments | {benchmark})].copy()
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise RuntimeError("filtered clean prices are empty")
    frame = pd.concat(parts, ignore_index=True, sort=False)
    frame["Instrument"] = frame["Instrument"].astype("string").str.strip()
    frame["Date"] = parse_dates(frame["Date"], "prices.Date")
    frame["close"] = finite_numeric(frame.pop("TRDPRC_1"))
    if frame.duplicated(["Instrument", "Date"]).any():
        dup = frame.loc[frame.duplicated(["Instrument", "Date"], keep=False), ["Instrument", "Date"]]
        raise RuntimeError(f"clean prices have duplicate Instrument/Date keys: {dup.head(10).to_dict('records')}")
    frame["row_present"] = True
    return frame, source_rows


def read_raw_etf(path: Path, instrument: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"Date", "TRDPRC_1"}
    missing = sorted(required - set(header))
    if missing:
        raise RuntimeError(f"{path.name} schema missing: {missing}")
    frame = pd.read_csv(path, usecols=["Date", "TRDPRC_1"], low_memory=False)
    input_rows = int(len(frame))
    frame["Date"] = parse_dates(frame["Date"], f"{path.name}.Date")
    if frame.duplicated("Date").any():
        dup = frame.loc[frame.duplicated("Date", keep=False), ["Date"]]
        raise RuntimeError(f"{path.name} has duplicate Date keys: {dup.head(10).to_dict('records')}")
    frame["close"] = finite_numeric(frame.pop("TRDPRC_1"))
    frame["row_present"] = True
    valid = frame["close"].notna() & frame["close"].gt(0)
    details = {
        "instrument": instrument,
        "path": rel(path),
        "input_rows": input_rows,
        "date_min": frame["Date"].min(),
        "date_max": frame["Date"].max(),
        "rows_with_close": int(valid.sum()),
        "missing_or_nonpositive_close": int((~valid).sum()),
        "duplicate_date_rows": 0,
    }
    return frame, details


def max_drawdown(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all() or (array <= 0).any():
        return np.nan
    running_peak = np.maximum.accumulate(array)
    return float(np.min(array / running_peak - 1.0))


def make_series_features(close: pd.Series, windows: list[int], vol_windows: list[int], drawdown_window: int, annualization: int) -> dict[str, pd.Series]:
    close = close.astype(float)
    returns = close.pct_change(fill_method=None)
    log_returns = np.log1p(returns.where(returns > -1))
    features: dict[str, pd.Series] = {}
    for window in windows:
        features[f"momentum_{window}"] = np.expm1(log_returns.rolling(window, min_periods=window).sum())
    for window in vol_windows:
        features[f"volatility_{window}_ann"] = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(float(annualization))
    features[f"max_drawdown_{drawdown_window}"] = close.rolling(drawdown_window, min_periods=drawdown_window).apply(max_drawdown, raw=True)
    features["return_1"] = returns
    return features


def lookup(series: pd.Series, dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(dates)).to_numpy(dtype=float)


def code_join(codes: list[str]) -> str:
    ordered = list(dict.fromkeys(code for code in codes if code))
    return ";".join(ordered) if ordered else "OK"


def main(config_path: Path, run_id: str | None) -> Path:
    started = utc_now()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unexpected config schema: {config.get('schema_version')}")
    inputs = config["inputs"]
    benchmark = str(inputs["benchmark"])
    expected_pool = int(inputs["expected_pool_instruments"])
    model_features_path = ROOT / inputs["model_features"]
    prices_path = ROOT / inputs["prices"]
    raw_root = ROOT / inputs["raw_etf_root"]
    etf_files: dict[str, str] = dict(inputs["raw_etf_files"])
    paths: dict[str, Path] = {
        "model_features": model_features_path,
        "prices": prices_path,
        "config": config_path,
        "builder": Path(__file__).resolve(),
    }
    for instrument, filename in etf_files.items():
        paths[f"raw_etf_{instrument}"] = raw_root / filename
        if bool(inputs.get("raw_etf_sidecars", True)):
            for suffix in [".meta.json", ".request.json"]:
                sidecar = raw_root / f"{filename[:-4]}{suffix}"
                if sidecar.exists():
                    paths[f"raw_etf_{instrument}{suffix}"] = sidecar
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing configured inputs: {missing}")
    forbidden_path_tokens = ["targets", "target", "future_return", "future_returns", "labels", "label", "model_runs", "model_outputs"]
    bad_paths = [name for name, path in paths.items() if any(token in str(path).casefold() for token in forbidden_path_tokens)]
    if bad_paths:
        raise RuntimeError(f"forbidden input path token detected: {bad_paths}")

    print("阶段说明（AI industry state features）：读取 49-RIC expanded model_features 的成员/日期网格、clean SPY/池收盘价和五个 raw ETF close；按形成日严格取上一 SPY session，构建 ETF 动量/波动/最大回撤、SOXX 相对 SPY、池 breadth/dispersion/等权相对收益及覆盖 reason code。保留输入 raw，不读目标、标签、未来收益或模型运行。", flush=True)
    model_grid = read_model_features(model_features_path, expected_pool)
    pool_instruments = set(model_grid["Instrument"].astype(str))
    clean_prices, clean_source_rows = read_price_subset(prices_path, pool_instruments, benchmark)
    raw_etf_frames: dict[str, pd.DataFrame] = {}
    raw_etf_stats: dict[str, dict[str, Any]] = {}
    for instrument, filename in etf_files.items():
        raw_etf_frames[instrument], raw_etf_stats[instrument] = read_raw_etf(raw_root / filename, instrument)

    clean_valid = clean_prices.loc[clean_prices["close"].notna() & clean_prices["close"].gt(0)].copy()
    clean_dates = pd.DatetimeIndex(sorted(clean_prices["Date"].drop_duplicates()))
    spy_valid = clean_valid.loc[clean_valid["Instrument"].eq(benchmark), ["Date", "close"]].sort_values("Date")
    spy_sessions = pd.DatetimeIndex(spy_valid["Date"].drop_duplicates())
    if len(spy_sessions) == 0:
        raise RuntimeError("no valid SPY sessions in clean prices")
    formation_sessions = pd.DatetimeIndex(sorted(model_grid["formation_session"].drop_duplicates()))
    formation_not_spy = formation_sessions.difference(spy_sessions)
    if len(formation_not_spy):
        raise RuntimeError(f"formation sessions absent from SPY calendar: {formation_not_spy[:10].tolist()}")
    ref_positions = np.searchsorted(spy_sessions.values, formation_sessions.values, side="left") - 1
    reference_sessions = pd.DatetimeIndex([spy_sessions[pos] if pos >= 0 else pd.NaT for pos in ref_positions])
    reference_valid = ~pd.isna(reference_sessions)
    if reference_valid.any() and not bool((reference_sessions[reference_valid] < formation_sessions[reference_valid]).all()):
        raise RuntimeError("reference session is not strictly before formation session")

    instruments = sorted(pool_instruments | {benchmark})
    close_wide = clean_valid.pivot(index="Date", columns="Instrument", values="close").reindex(index=spy_sessions, columns=instruments)
    presence_wide = clean_prices.pivot(index="Date", columns="Instrument", values="row_present").reindex(index=spy_sessions, columns=instruments).eq(True).fillna(False)
    if close_wide[benchmark].isna().any():
        raise RuntimeError("SPY session calendar contains a missing SPY close")
    annualization = int(config["etf"]["volatility_annualization_factor"])
    mom_windows = [int(v) for v in config["etf"]["momentum_windows_sessions"]]
    vol_windows = [int(v) for v in config["etf"]["volatility_windows_returns"]]
    drawdown_window = int(config["etf"]["drawdown_window_prices"])
    ticker_labels = {str(k): str(v) for k, v in config["etf"]["ticker_labels"].items()}
    etf_close_wide: dict[str, pd.Series] = {}
    etf_row_presence: dict[str, set[pd.Timestamp]] = {}
    etf_features: dict[str, dict[str, pd.Series]] = {}
    for instrument, frame in raw_etf_frames.items():
        series = frame.loc[frame["close"].notna() & frame["close"].gt(0)].set_index("Date")["close"].reindex(spy_sessions)
        etf_close_wide[instrument] = series
        etf_row_presence[instrument] = set(frame["Date"].tolist())
        etf_features[instrument] = make_series_features(series, mom_windows, vol_windows, drawdown_window, annualization)

    spy_features = make_series_features(close_wide[benchmark], mom_windows, vol_windows, drawdown_window, annualization)
    pool_returns = close_wide[sorted(pool_instruments)].pct_change(fill_method=None)
    pool_ma = {window: close_wide[sorted(pool_instruments)].rolling(window, min_periods=window).mean() for window in [20, 60]}
    spy_returns = spy_features["return_1"]

    active_by_date = {date: group["Instrument"].astype(str).tolist() for date, group in model_grid.groupby("formation_session", sort=True)}
    split_by_date: dict[pd.Timestamp, str] = {}
    if "split" in pd.read_csv(model_features_path, nrows=0).columns:
        split_frame = pd.read_csv(model_features_path, usecols=["formation_session", "split"], low_memory=False)
        split_frame["formation_session"] = parse_dates(split_frame["formation_session"], "model_features.formation_session")
        split_values = split_frame.groupby("formation_session")["split"].agg(lambda s: sorted(set(s.dropna().astype(str))))
        invalid_split = split_values[split_values.map(len) > 1]
        if not invalid_split.empty:
            raise RuntimeError(f"formation dates have multiple split values: {invalid_split.head().to_dict()}")
        split_by_date = {date: values[0] for date, values in split_values.items() if values}

    state_rows: list[dict[str, Any]] = []
    etf_coverage_rows: list[dict[str, Any]] = []
    pool_coverage_rows: list[dict[str, Any]] = []
    state_reason_counter: Counter[str] = Counter()
    etf_reason_counter: Counter[str] = Counter()
    pool_reason_counter: Counter[str] = Counter()

    print(f"输入检查完成：model_features={len(model_grid):,} active security-day rows、{len(pool_instruments)} pool instruments、{len(formation_sessions):,} formation sessions；clean prices filtered rows={len(clean_prices):,}（source total rows={clean_source_rows:,}）；ETF raw rows={sum(s['input_rows'] for s in raw_etf_stats.values()):,}。开始计算 lagged features。", flush=True)
    for formation, ref_pos, reference in zip(formation_sessions, ref_positions, reference_sessions):
        row: dict[str, Any] = {
            "formation_session": formation,
            "reference_session": reference,
            "reference_lag_sessions": 1,
            "split": split_by_date.get(formation, ""),
            "pool_active_member_count": int(len(active_by_date.get(formation, []))),
        }
        row_reasons: list[str] = []
        pool_reasons: list[str] = []
        if ref_pos < 0 or pd.isna(reference):
            row_reasons.append("NO_PRIOR_SPY_SESSION")
            ref_idx = None
        else:
            ref_idx = int(ref_pos)

        for instrument in etf_files:
            slug = ticker_labels[instrument]
            close_series = etf_close_wide[instrument]
            features = etf_features[instrument]
            if ref_idx is None:
                current_close = np.nan
                close_row_present = False
                feature_values = {name: np.nan for name in features}
                obs_counts = {window: 0 for window in [5, 20, 60]}
            else:
                current_close = float(close_series.iloc[ref_idx]) if pd.notna(close_series.iloc[ref_idx]) else np.nan
                close_row_present = bool(reference in etf_row_presence[instrument])
                feature_values = {name: (float(series.iloc[ref_idx]) if pd.notna(series.iloc[ref_idx]) else np.nan) for name, series in features.items()}
                obs_counts = {window: int(close_series.iloc[max(0, ref_idx - window + 1): ref_idx + 1].notna().sum()) for window in [5, 20, 60]}
            feature_reason: list[str] = []
            if not np.isfinite(current_close):
                feature_reason.append("ETF_NO_OBSERVATION")
            if not close_row_present and ref_idx is not None:
                feature_reason.append("ETF_NO_RAW_ROW_AT_REFERENCE")
            for window in mom_windows:
                key = f"momentum_{window}"
                row[f"etf_{slug}_momentum_{window}"] = feature_values[key]
                if not np.isfinite(feature_values[key]):
                    feature_reason.append(f"ETF_MOMENTUM_{window}_UNAVAILABLE")
            for window in vol_windows:
                key = f"volatility_{window}_ann"
                row[f"etf_{slug}_volatility_{window}_ann"] = feature_values[key]
                if not np.isfinite(feature_values[key]):
                    feature_reason.append(f"ETF_VOLATILITY_{window}_UNAVAILABLE")
            dd_key = f"max_drawdown_{drawdown_window}"
            row[f"etf_{slug}_max_drawdown_{drawdown_window}"] = feature_values[dd_key]
            if not np.isfinite(feature_values[dd_key]):
                feature_reason.append(f"ETF_MAX_DRAWDOWN_{drawdown_window}_UNAVAILABLE")
            for window, count in obs_counts.items():
                row[f"etf_{slug}_close_obs_count_last_{window}"] = count
            row[f"etf_{slug}_reason_code"] = code_join(feature_reason)
            for reason in feature_reason or ["OK"]:
                etf_reason_counter[reason] += 1
            etf_coverage_rows.append({
                "formation_session": formation,
                "reference_session": reference,
                "Instrument": instrument,
                "close_row_present_at_reference": close_row_present,
                "close_valid_at_reference": bool(np.isfinite(current_close)),
                "close_obs_count_last_5": obs_counts[5],
                "close_obs_count_last_20": obs_counts[20],
                "close_obs_count_last_60": obs_counts[60],
                "momentum_5_valid": bool(np.isfinite(feature_values.get("momentum_5", np.nan))),
                "momentum_20_valid": bool(np.isfinite(feature_values.get("momentum_20", np.nan))),
                "momentum_60_valid": bool(np.isfinite(feature_values.get("momentum_60", np.nan))),
                "volatility_20_valid": bool(np.isfinite(feature_values.get("volatility_20_ann", np.nan))),
                "volatility_60_valid": bool(np.isfinite(feature_values.get("volatility_60_ann", np.nan))),
                "max_drawdown_60_valid": bool(np.isfinite(feature_values.get("max_drawdown_60", np.nan))),
                "reason_code": code_join(feature_reason),
            })
            if feature_reason:
                row_reasons.extend(f"{instrument}:{reason}" for reason in feature_reason)

        for window in mom_windows:
            soxx_mom = row.get(f"etf_soxx_momentum_{window}", np.nan)
            spy_mom = float(spy_features[f"momentum_{window}"].iloc[ref_idx]) if ref_idx is not None and pd.notna(spy_features[f"momentum_{window}"].iloc[ref_idx]) else np.nan
            row[f"spy_momentum_{window}"] = spy_mom
            row[f"soxx_relative_spy_return_{window}"] = float(soxx_mom - spy_mom) if np.isfinite(soxx_mom) and np.isfinite(spy_mom) else np.nan
            if not np.isfinite(row[f"soxx_relative_spy_return_{window}"]):
                row_reasons.append(f"SOXX_RELATIVE_SPY_{window}_UNAVAILABLE")

        active = active_by_date.get(formation, [])
        if ref_idx is None or not active:
            pool_reasons.append("POOL_NO_PRIOR_SESSION_OR_ACTIVE_MEMBER")
            row_reasons.append("POOL_NO_PRIOR_SESSION_OR_ACTIVE_MEMBER")
            valid_returns = pd.Series(dtype=float)
            valid_ma20 = pd.Series(dtype=float)
            valid_ma60 = pd.Series(dtype=float)
            one_returns = pd.Series(index=active, dtype=float)
        else:
            one_returns = pool_returns.iloc[ref_idx].reindex(active)
            current = close_wide.iloc[ref_idx].reindex(active)
            ma20_row = pool_ma[20].iloc[ref_idx].reindex(active)
            ma60_row = pool_ma[60].iloc[ref_idx].reindex(active)
            valid_returns = one_returns[np.isfinite(one_returns)]
            valid_ma20 = current[(np.isfinite(current)) & (np.isfinite(ma20_row))]
            valid_ma60 = current[(np.isfinite(current)) & (np.isfinite(ma60_row))]
            prev_presence = presence_wide.iloc[ref_idx - 1].reindex(active) if ref_idx > 0 else pd.Series(False, index=active)
            previous = close_wide.iloc[ref_idx - 1].reindex(active) if ref_idx > 0 else pd.Series(np.nan, index=active)
            current_presence = presence_wide.iloc[ref_idx].reindex(active)
            for instrument in active:
                ret = one_returns.get(instrument, np.nan)
                current_price = current.get(instrument, np.nan)
                m20 = ma20_row.get(instrument, np.nan)
                m60 = ma60_row.get(instrument, np.nan)
                reasons: list[str] = []
                if not bool(current_presence.get(instrument, False)):
                    reasons.append("POOL_NO_PRICE_ROW_AT_REFERENCE")
                elif not np.isfinite(current_price):
                    reasons.append("POOL_CLOSE_MISSING_AT_REFERENCE")
                if ref_idx <= 0 or not bool(prev_presence.get(instrument, False)):
                    reasons.append("POOL_NO_PREVIOUS_PRICE_ROW")
                elif not np.isfinite(previous.get(instrument, np.nan)):
                    reasons.append("POOL_PREVIOUS_CLOSE_MISSING")
                if not np.isfinite(ret):
                    reasons.append("POOL_RETURN_UNAVAILABLE")
                if not np.isfinite(m20):
                    reasons.append("POOL_MA20_UNAVAILABLE")
                if not np.isfinite(m60):
                    reasons.append("POOL_MA60_UNAVAILABLE")
                code = code_join(reasons)
                for reason in reasons or ["OK"]:
                    pool_reason_counter[reason] += 1
                pool_coverage_rows.append({
                    "formation_session": formation,
                    "reference_session": reference,
                    "Instrument": instrument,
                    "active_member": True,
                    "current_price_row_present": bool(current_presence.get(instrument, False)),
                    "previous_price_row_present": bool(prev_presence.get(instrument, False)),
                    "close_at_reference": current_price,
                    "one_session_return": ret,
                    "moving_average_20": m20,
                    "moving_average_60": m60,
                    "return_valid": bool(np.isfinite(ret)),
                    "above_ma20_valid": bool(np.isfinite(current_price) and np.isfinite(m20)),
                    "above_ma20": bool(np.isfinite(current_price) and np.isfinite(m20) and current_price > m20),
                    "above_ma60_valid": bool(np.isfinite(current_price) and np.isfinite(m60)),
                    "above_ma60": bool(np.isfinite(current_price) and np.isfinite(m60) and current_price > m60),
                    "reason_code": code,
                })

        spy_ret = float(spy_returns.iloc[ref_idx]) if ref_idx is not None and pd.notna(spy_returns.iloc[ref_idx]) else np.nan
        row["spy_return_1"] = spy_ret
        row["pool_valid_return_count"] = int(len(valid_returns))
        row["pool_valid_ma20_count"] = int(len(valid_ma20))
        row["pool_valid_ma60_count"] = int(len(valid_ma60))
        row["pool_breadth_up_1"] = float((valid_returns > 0).mean()) if len(valid_returns) else np.nan
        row["pool_breadth_above_ma20"] = float((valid_ma20 > pool_ma[20].iloc[ref_idx].reindex(active)[valid_ma20.index]).mean()) if len(valid_ma20) and ref_idx is not None else np.nan
        row["pool_breadth_above_ma60"] = float((valid_ma60 > pool_ma[60].iloc[ref_idx].reindex(active)[valid_ma60.index]).mean()) if len(valid_ma60) and ref_idx is not None else np.nan
        row["pool_cross_sectional_dispersion_1"] = float(valid_returns.std(ddof=1)) if len(valid_returns) >= 2 else np.nan
        row["pool_equal_weight_return_1"] = float(valid_returns.mean()) if len(valid_returns) else np.nan
        row["pool_equal_weight_relative_spy_return_1"] = float(row["pool_equal_weight_return_1"] - spy_ret) if np.isfinite(row["pool_equal_weight_return_1"]) and np.isfinite(spy_ret) else np.nan
        row["pool_low_coverage_return"] = bool(row["pool_valid_return_count"] < int(config["pool"]["low_coverage_threshold"]))
        row["pool_low_coverage_ma20"] = bool(row["pool_valid_ma20_count"] < int(config["pool"]["low_coverage_threshold"]))
        row["pool_low_coverage_ma60"] = bool(row["pool_valid_ma60_count"] < int(config["pool"]["low_coverage_threshold"]))
        row["pool_low_coverage"] = bool(row["pool_low_coverage_return"] or row["pool_low_coverage_ma20"] or row["pool_low_coverage_ma60"])
        if row["pool_low_coverage_return"]:
            pool_reasons.append("POOL_LOW_COVERAGE_RETURN")
            row_reasons.append("POOL_LOW_COVERAGE_RETURN")
        if row["pool_low_coverage_ma20"]:
            pool_reasons.append("POOL_LOW_COVERAGE_MA20")
            row_reasons.append("POOL_LOW_COVERAGE_MA20")
        if row["pool_low_coverage_ma60"]:
            pool_reasons.append("POOL_LOW_COVERAGE_MA60")
            row_reasons.append("POOL_LOW_COVERAGE_MA60")
        for feature_name in ["pool_cross_sectional_dispersion_1", "pool_equal_weight_return_1", "pool_equal_weight_relative_spy_return_1"]:
            if not np.isfinite(row[feature_name]):
                pool_reasons.append(f"{feature_name.upper()}_UNAVAILABLE")
                row_reasons.append(f"{feature_name.upper()}_UNAVAILABLE")
        row["pool_reason_code"] = code_join(pool_reasons)
        row["etf_coverage_reason_codes"] = "|".join(f"{instrument}={row[f'etf_{ticker_labels[instrument]}_reason_code']}" for instrument in etf_files)
        row["feature_reason_code"] = code_join(row_reasons)
        for reason in row_reasons or ["OK"]:
            state_reason_counter[reason] += 1
        state_rows.append(row)

    state = pd.DataFrame(state_rows)
    etf_coverage = pd.DataFrame(etf_coverage_rows)
    pool_coverage = pd.DataFrame(pool_coverage_rows)
    if len(state) != len(formation_sessions):
        raise RuntimeError("state row count does not match formation sessions")
    if len(pool_coverage) != len(model_grid):
        raise RuntimeError(f"pool coverage row count {len(pool_coverage)} does not match model grid {len(model_grid)}")
    if len(etf_coverage) != len(formation_sessions) * len(etf_files):
        raise RuntimeError("ETF coverage row count does not match session x ETF grid")

    numeric_state = [column for column in state.columns if column.startswith("etf_") and any(token in column for token in ["momentum", "volatility", "max_drawdown"]) or column.startswith("soxx_relative") or column.startswith("spy_momentum") or column in {"spy_return_1", "pool_breadth_up_1", "pool_breadth_above_ma20", "pool_breadth_above_ma60", "pool_cross_sectional_dispersion_1", "pool_equal_weight_return_1", "pool_equal_weight_relative_spy_return_1"}]
    finite_values = np.array([], dtype=float)
    if numeric_state:
        matrix = state[numeric_state].to_numpy(dtype=float)
        finite_values = matrix[~np.isnan(matrix)]
        if finite_values.size and not np.isfinite(finite_values).all():
            raise RuntimeError("non-finite numeric state values found")
    forbidden_columns = {"y", "label", "label_complete", "forward_excess_return", "stock_forward_return", "benchmark_forward_return", "target", "target_reason"}
    if forbidden_columns & set(state.columns):
        raise RuntimeError(f"forbidden target columns in state output: {sorted(forbidden_columns & set(state.columns))}")

    reason_rows = []
    for scope, counter in [("state", state_reason_counter), ("etf", etf_reason_counter), ("pool_member", pool_reason_counter)]:
        reason_rows.extend({"scope": scope, "reason_code": code, "count": int(count)} for code, count in sorted(counter.items()))
    reason_codes = pd.DataFrame(reason_rows, columns=["scope", "reason_code", "count"])
    quarantine = pd.DataFrame(columns=["source", "Instrument", "Date", "reason_code", "raw_path", "raw_row"])

    run_id = run_id or run_id_now()
    if not run_id or any(ch not in "0123456789TZ" for ch in run_id):
        raise ValueError("run_id must contain only digits, T, and Z")
    clean_out = ROOT / config["outputs"]["clean_root"] / run_id
    audit_out = ROOT / config["outputs"]["audit_root"] / run_id
    quarantine_out = audit_out / "quarantine"
    clean_out.mkdir(parents=True, exist_ok=False)
    quarantine_out.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "feature_table": clean_out / config["outputs"]["feature_table"],
        "pool_coverage": clean_out / config["outputs"]["pool_coverage"],
        "etf_coverage": clean_out / config["outputs"]["etf_coverage"],
        "summary": clean_out / config["outputs"]["summary"],
        "hashes": clean_out / config["outputs"]["hashes"],
        "reason_code_counts": audit_out / config["outputs"]["reason_code_counts"],
        "quarantine": quarantine_out / Path(config["outputs"]["quarantine"]).name,
    }
    write_csv_atomic(state, output_paths["feature_table"])
    write_csv_atomic(pool_coverage, output_paths["pool_coverage"])
    write_csv_atomic(etf_coverage, output_paths["etf_coverage"])
    write_csv_atomic(reason_codes, output_paths["reason_code_counts"])
    write_csv_atomic(quarantine, output_paths["quarantine"])

    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    data_output_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in output_paths.items() if name not in {"summary", "hashes"}}
    feature_columns = [column for column in state.columns if column not in {"formation_session", "reference_session", "split", "reference_lag_sessions", "pool_active_member_count", "pool_valid_return_count", "pool_valid_ma20_count", "pool_valid_ma60_count", "pool_low_coverage_return", "pool_low_coverage_ma20", "pool_low_coverage_ma60", "pool_low_coverage", "etf_coverage_reason_codes", "pool_reason_code", "feature_reason_code"} and not column.endswith("_reason_code") and "obs_count" not in column]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "started_at_utc": started,
        "status": "complete_clean_ai_industry_state",
        "scope": "ai_industry_state_only",
        "spec_path": "AI_DAILY_MACRO_FEATURE_SPEC.md",
        "spec_version": "v1.0",
        "config": {"path": rel(config_path), "sha256": sha256_file(config_path)},
        "builder": {"path": rel(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": {
            "model_features": {"path": rel(model_features_path), "rows": int(len(model_grid)), "instruments": int(len(pool_instruments)), "formation_sessions": int(len(formation_sessions))},
            "clean_prices": {"path": rel(prices_path), "source_rows_total": int(clean_source_rows), "filtered_rows": int(len(clean_prices)), "filtered_instruments": int(clean_prices["Instrument"].nunique()), "valid_close_rows": int(len(clean_valid)), "invalid_or_missing_close_rows": int(len(clean_prices) - len(clean_valid))},
            "raw_etf": raw_etf_stats,
        },
        "input_hashes": input_hashes,
        "output_hashes": data_output_hashes,
        "counts": {
            "formation_sessions": int(len(formation_sessions)),
            "state_rows": int(len(state)),
            "pool_coverage_rows": int(len(pool_coverage)),
            "etf_coverage_rows": int(len(etf_coverage)),
            "pool_instruments": int(len(pool_instruments)),
            "etf_instruments": int(len(etf_files)),
            "quarantine_rows": int(len(quarantine)),
            "state_rows_with_any_reason": int((state["feature_reason_code"] != "OK").sum()),
            "state_rows_low_coverage": int(state["pool_low_coverage"].sum()),
            "pool_member_rows_with_any_reason": int((pool_coverage["reason_code"] != "OK").sum()),
            "etf_session_rows_with_any_reason": int((etf_coverage["reason_code"] != "OK").sum()),
        },
        "feature_columns": feature_columns,
        "rules": config["formation"] | {"etf": config["etf"], "pool": config["pool"], "missing_value_policy": config["missing_value_policy"]},
        "reason_code_counts": {"state": dict(sorted(state_reason_counter.items())), "etf": dict(sorted(etf_reason_counter.items())), "pool_member": dict(sorted(pool_reason_counter.items()))},
        "checks": {
            "reference_strictly_before_formation": bool(reference_valid.any() and (reference_sessions[reference_valid] < formation_sessions[reference_valid]).all()),
            "formation_sessions_match_model_grid": int(len(state)) == int(len(formation_sessions)),
            "pool_coverage_matches_model_grid": int(len(pool_coverage)) == int(len(model_grid)),
            "etf_coverage_matches_session_etf_grid": int(len(etf_coverage)) == int(len(formation_sessions) * len(etf_files)),
            "pool_instrument_count_is_49": len(pool_instruments) == 49,
            "state_keys_unique": not state.duplicated(["formation_session"]).any(),
            "pool_coverage_keys_unique": not pool_coverage.duplicated(["formation_session", "Instrument"]).any(),
            "etf_coverage_keys_unique": not etf_coverage.duplicated(["formation_session", "Instrument"]).any(),
            "no_future_or_target_columns": not bool(forbidden_columns & set(state.columns)),
            "no_infinite_numeric_values": bool(not finite_values.size or np.isfinite(finite_values).all()),
            "no_imputation_forward_fill_or_winsorization": True,
            "no_silent_deduplication": True,
            "raw_inputs_preserved": True,
            "physical_source_row_deletion": False,
            "test_targets_or_model_results_read": False,
        },
        "affected_rows": {
            "pool_member_rows_excluded_from_return_denominator": int((~pool_coverage["return_valid"]).sum()),
            "pool_member_rows_excluded_from_ma20_denominator": int((~pool_coverage["above_ma20_valid"]).sum()),
            "pool_member_rows_excluded_from_ma60_denominator": int((~pool_coverage["above_ma60_valid"]).sum()),
            "etf_rows_with_missing_or_insufficient_history": int((etf_coverage["reason_code"] != "OK").sum()),
            "rows_quarantined": 0,
        },
        "limitations": [
            "This run covers the AI industry state subsection only; FRED macro risk series are not included because no FRED input was supplied to this stage.",
            "The expanded registry is exploratory_static_candidate with provisional static-source membership for 40 of 49 RICs; the state features inherit that membership limitation.",
            "ETF and pool windows require complete consecutive SPY-session observations; missing closes remain NA and are reported in coverage tables.",
            "FRED-style vintage revision risk is outside this stage; no macro series were read.",
        ],
        "quarantine": {"path": rel(output_paths["quarantine"]), "rows": 0, "reason_code_policy": "No rows were physically removed; missing observations remain in coverage tables with reason codes."},
        "hash_manifest_path": rel(output_paths["hashes"]),
    }
    write_json_atomic(output_paths["summary"], summary)
    summary_hash = sha256_file(output_paths["summary"])
    hashes = {
        "schema_version": f"{SCHEMA_VERSION}_hash_manifest",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "input_hashes": input_hashes,
        "output_hashes": {**data_output_hashes, "summary": {"path": rel(output_paths["summary"]), "sha256": summary_hash}},
    }
    write_json_atomic(output_paths["hashes"], hashes)
    print(f"特征构建完成：state={len(state):,} rows，pool coverage={len(pool_coverage):,} rows，ETF coverage={len(etf_coverage):,} rows，low-coverage sessions={int(state['pool_low_coverage'].sum()):,}，quarantine=0。输出={rel(clean_out)}；hash manifest={rel(output_paths['hashes'])}。", flush=True)
    return clean_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build lagged AI industry state features.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to frozen feature config JSON")
    parser.add_argument("--run-id", default=None, help="Optional run-scoped output id")
    args = parser.parse_args()
    main(Path(args.config).resolve(), args.run_id)
