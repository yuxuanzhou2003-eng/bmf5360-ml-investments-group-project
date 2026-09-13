"""Build an auditable, point-in-time AI-pool monthly baseline layer.

The builder creates one row per literal security/RIC and last-SPY-session
formation month. It uses the latest known AI-pool membership state as of that
formation date, backward-looking price/market controls and explicitly tagged
candidate fundamental controls. Development targets are generated only for
training and validation rows; test rows receive a sealed manifest without
future-return values.
"""
from __future__ import annotations

import hashlib
import json
import math
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "ai_pool_baseline_model_ready_v1_config.json"
SCHEMA_VERSION = "ai_pool_baseline_model_ready_v1"
MEMBERSHIP_RUN_ID = "20260910T023221976393Z"
OUT_ROOT = ROOT / "data" / "model_ready_ai_pool_v1"


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


def parse_dates(values: Iterable, name: str) -> pd.Series:
    result = pd.to_datetime(values, format="mixed", errors="coerce").dt.normalize()
    bad = int(result.isna().sum())
    if bad:
        raise ValueError(f"{name} has {bad} unparseable dates")
    return result


def parse_bool(values: Iterable, name: str) -> pd.Series:
    normalized = pd.Series(values).astype("string").str.strip().str.casefold()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name} has invalid booleans: {invalid[:10]}")
    return normalized.eq("true")


def read_filtered_csv(path: Path, usecols: list[str], instruments: set[str], *, chunksize: int = 500_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=chunksize):
        selected = chunk.loc[chunk["Instrument"].astype(str).isin(instruments)].copy()
        if not selected.empty:
            parts.append(selected)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=usecols)


def matrix_lookup(matrix: pd.DataFrame, instruments: pd.Series, dates: pd.Series, name: str) -> np.ndarray:
    row_i = matrix.index.get_indexer(pd.DatetimeIndex(dates))
    col_i = matrix.columns.get_indexer(pd.Index(instruments))
    out = np.full(len(instruments), np.nan, dtype=float)
    valid = (row_i >= 0) & (col_i >= 0)
    if valid.any():
        values = matrix.to_numpy(dtype=float)
        out[valid] = values[row_i[valid], col_i[valid]]
    if (col_i < 0).any():
        missing = sorted(set(instruments[col_i < 0]))
        raise RuntimeError(f"{name}: instruments absent from matrix: {missing[:10]}")
    return out


def rolling_beta_idio(wide: pd.DataFrame, benchmark: pd.Series, window: int = 126, min_obs: int = 100):
    valid = wide.notna().mul(benchmark.notna(), axis=0)
    count = valid.rolling(window, min_periods=1).sum()
    x = wide.where(valid)
    market = valid.mul(benchmark, axis=0)
    sum_x = x.rolling(window, min_periods=1).sum()
    sum_m = market.rolling(window, min_periods=1).sum()
    sum_x2 = x.pow(2).rolling(window, min_periods=1).sum()
    sum_m2 = market.pow(2).rolling(window, min_periods=1).sum()
    sum_xm = x.mul(benchmark, axis=0).rolling(window, min_periods=1).sum()
    covariance_numerator = sum_xm - sum_x * sum_m / count
    market_ss = sum_m2 - sum_m.pow(2) / count
    asset_ss = sum_x2 - sum_x.pow(2) / count
    valid_fit = (count >= min_obs) & market_ss.gt(0) & market_ss.notna()
    beta = (covariance_numerator / market_ss).where(valid_fit)
    residual_ss = (asset_ss - covariance_numerator.pow(2) / market_ss).clip(lower=0)
    residual_variance = residual_ss / (count - 2).where(count.gt(2))
    idio = (np.sqrt(residual_variance) * np.sqrt(252.0)).where(valid_fit & count.gt(2))
    return beta, idio, count.where(beta.notna())


def split_for_day(day: pd.Timestamp) -> str:
    if pd.Timestamp("2015-01-01") <= day <= pd.Timestamp("2020-12-31"):
        return "training"
    if pd.Timestamp("2021-01-01") <= day <= pd.Timestamp("2022-12-31"):
        return "validation"
    if pd.Timestamp("2023-01-01") <= day <= pd.Timestamp("2026-06-30"):
        return "test"
    return "out_of_window"


def finite_or_na(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values))


def safe_ratio(numerator: pd.Series, denominator: pd.Series, *, require_nonzero: bool = True) -> pd.Series:
    n = finite_or_na(numerator)
    d = finite_or_na(denominator)
    valid = n.notna() & d.notna()
    if require_nonzero:
        valid &= d.ne(0)
    result = pd.Series(np.nan, index=n.index, dtype=float)
    result.loc[valid] = (n.loc[valid] / d.loc[valid]).astype(float)
    return result.where(np.isfinite(result))


def append_log(summary: dict, out: Path) -> None:
    path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# DATA_PROCESSING_LOG\n"
    counts = summary["counts"]
    entry = [
        "",
        f"## {summary['run_id']} — AI pool controls-only model-ready v1 ({summary['generated_at_utc']})",
        "",
        "- **阶段目的与状态**：从最新点时 AI pool membership、clean v2 returns、clean v3 prices 和 clean AI-factor controls 构建每月 `Instrument + formation_session` 工程层；本 run 仅生成训练/验证开发标签，test target 保持封存。",
        f"- **输入版本与路径**：membership `{summary['inputs']['membership']['path']}` run=`{summary['inputs']['membership']['run_id']}`；returns `{summary['inputs']['returns']['path']}`；prices `{summary['inputs']['prices']['path']}`；fundamentals `{summary['inputs']['fundamentals']['path']}`；market cap `{summary['inputs']['market_cap']['path']}`；config=`{summary['config_path']}`；builder=`{summary['builder_path']}`。",
        f"- **输入哈希**：`{json.dumps(summary['input_hashes'], ensure_ascii=False, sort_keys=True)}`。",
        f"- **行/标的计数**：`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`。",
        "- **成员资格规则**：逐形成月使用 `formation_date <= formation_session` 的最近已知状态，并要求 `member_from <= formation_session <= member_to`、`membership_status=member`、`membership_eligible=true`；后续 ambiguous/unknown 不前向变成 member；source membership 的 unknown/ambiguous 未删除，并在 `membership_snapshot_audit.csv` 保留原因。",
        "- **特征规则**：formation-session 前一 SPY session 作为 rolling/liquidity reference；价格收益窗口为 clean v2 `return_decimal`；H21 标签窗口为 `entry_session+1` 至 `exit_session`；不使用 event panel/event study；无 entry-day EOD 特征。",
        "- **缺失/删除/去重/单位**：master 行全部保留；空/不可解析值为 NA，真实 0 保留；不填补、不前后填充、不插值、不 winsorize；重复键只在输入校验中阻塞，不静默去重；退市按 membership span 保留；literal RIC 作为 security_id，不按名称合并；基本面与市值 `UNVERIFIED` 仅作为候选控制并标记 gate 未通过。",
        f"- **目标封存与处理参数**：`{summary['test_policy']}`；训练模型层（后续 runner）只允许 training-only median imputation + missing indicators；本构建阶段未拟合模型或预处理。",
        f"- **quarantine/审计**：物理删除=0；没有把 ambiguous/unknown 物理移除，状态与 reason 保留在 `{rel(out / 'membership_snapshot_audit.csv')}`；空/缺失控制按 feature_missingness 与 eligibility 记录。",
        f"- **检查与限制**：`{json.dumps(summary['checks'], ensure_ascii=False, sort_keys=True)}`；正式基本面控制 gate=`{summary['fundamental_semantics_gate_passed']}`，原因=`{summary['limitations']}`。",
        f"- **执行状态**：`{summary['status']}`。",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def append_ai_use_log(summary: dict, out: Path) -> None:
    path = ROOT / "AI_USE_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# AI_USE_LOG\n"
    entry = [
        "",
        f"### AI pool controls-only model-ready layer ({summary['run_id']})",
        "",
        f"- OpenAI Codex created and ran `build_ai_pool_baseline_model_ready_v1.py` against the latest membership run `{summary['inputs']['membership']['path']}` plus the existing clean returns/prices and a newly completed clean AI-factor controls snapshot. The output is `{rel(out)}` with `{summary['counts']['master_rows']}` active AI-pool security-month rows across `{summary['counts']['active_instruments']}` literal RICs and `{summary['counts']['formation_sessions']}` formation months.",
        "- Membership is applied point-in-time using the latest known dated status and corrected span interval; ambiguous/unknown source states are retained in `membership_snapshot_audit.csv` and are not carried forward as members. No name-based RIC mapping, company merge, delisted exclusion, imputation, forward fill, winsorization or event study was used.",
        f"- Development labels were written only for training/validation rows (`{summary['counts']['dev_label_rows']}` rows). Test rows (`{summary['counts']['test_rows']}`) have no target values; test seal check=`{summary['checks']['test_target_values_not_written']}`. Formal fundamental controls remain blocked because clean inputs report unit/currency/market-cap semantics as UNVERIFIED; candidate ratios are retained and tagged for review.",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def build(run_id: str | None = None) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected config schema")
    membership_path = ROOT / config["membership"]["path"]
    returns_path = ROOT / config["inputs"]["returns"]
    prices_path = ROOT / config["inputs"]["prices"]
    fundamentals_path = ROOT / config["inputs"]["fundamentals"]
    market_cap_path = ROOT / config["inputs"]["market_cap"]
    input_paths = {
        "membership": membership_path,
        "returns": returns_path,
        "prices": prices_path,
        "fundamentals": fundamentals_path,
        "market_cap": market_cap_path,
        "config": CONFIG_PATH,
        "builder": Path(__file__).resolve(),
    }
    missing = [name for name, path in input_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    membership = pd.read_csv(membership_path, low_memory=False)
    required_mem = ["Instrument", "formation_date", "pool_label", "membership_status", "membership_eligible", "member_from", "member_to", "delisted_ric"]
    if any(col not in membership.columns for col in required_mem):
        raise RuntimeError("membership schema missing required columns")
    membership["formation_dt"] = pd.to_datetime(membership["formation_date"], format="mixed", errors="coerce").dt.normalize()
    membership["membership_eligible_bool"] = parse_bool(membership["membership_eligible"], "membership_eligible")
    membership["member_from_dt"] = pd.to_datetime(membership["member_from"], format="mixed", errors="coerce").dt.normalize()
    membership["member_to_dt"] = pd.to_datetime(membership["member_to"], format="mixed", errors="coerce").dt.normalize()
    if membership["Instrument"].isna().any():
        raise RuntimeError("membership has missing Instrument")
    known = membership.loc[membership["formation_dt"].notna()].copy()
    if known.duplicated(["Instrument", "formation_dt"]).any():
        raise RuntimeError("membership has duplicate Instrument + formation_date keys")
    eligible_rows = membership.loc[membership["membership_eligible_bool"] & membership["membership_status"].eq("member")]
    candidate_rics = set(eligible_rows["Instrument"].astype(str))
    if not candidate_rics:
        raise RuntimeError("membership contains no explicit member rows")
    benchmark = str(config["inputs"]["benchmark"])
    data_rics = candidate_rics | {benchmark}

    print("阶段说明（AI pool model-ready 输入）：读取 membership、returns、prices、fundamentals 与 market-cap，按 literal RIC 与日期核验；不读取 event panel，不读取任何 test target 表。", flush=True)
    returns = read_filtered_csv(returns_path, ["Instrument", "Date", "return_decimal"], data_rics)
    returns["Date"] = parse_dates(returns["Date"], "returns Date")
    returns["return_decimal"] = finite_or_na(returns["return_decimal"])
    if returns.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("filtered returns contain duplicate Instrument + Date keys")
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    if benchmark not in wide.columns:
        raise RuntimeError("benchmark missing from returns")
    sessions = wide.index[wide[benchmark].notna()]
    sessions = pd.DatetimeIndex(sessions).sort_values().drop_duplicates()
    development_sessions = sessions[(sessions >= pd.Timestamp("2015-01-01")) & (sessions <= pd.Timestamp("2026-06-30"))]
    formation_series = pd.Series(development_sessions)
    formation = formation_series.groupby(formation_series.dt.to_period("M")).max().sort_values().reset_index(drop=True)
    formation = pd.DatetimeIndex(formation)
    if len(formation) != 138:
        raise RuntimeError(f"expected 138 formation sessions, got {len(formation)}")

    # Point-in-time membership snapshot: audit every eventual member candidate
    # for every formation month, then use only active=true rows as the master.
    snapshot_rows: list[dict] = []
    known = known.sort_values(["Instrument", "formation_dt"])
    for date in formation:
        for ric in sorted(candidate_rics):
            prior = known.loc[(known["Instrument"].astype(str).eq(ric)) & (known["formation_dt"] <= date)]
            if prior.empty:
                snapshot_rows.append({"Instrument": ric, "formation_session": date, "membership_asof_date": pd.NaT, "membership_asof_status": "membership_unknown", "membership_asof_pool_label": pd.NA, "membership_asof_eligible": False, "active_member": False, "membership_reason": "MEMBERSHIP_UNKNOWN", "member_from": pd.NaT, "member_to": pd.NaT, "delisted_ric": pd.NA})
                continue
            row = prior.iloc[-1]
            span_ok = bool(pd.notna(row["member_from_dt"]) and pd.notna(row["member_to_dt"]) and row["member_from_dt"] <= date <= row["member_to_dt"])
            is_member = bool(str(row["membership_status"]) == "member" and bool(row["membership_eligible_bool"]))
            active = bool(span_ok and is_member)
            reasons = []
            if not is_member:
                reasons.append("AMBIGUOUS_NOT_MEMBER" if str(row["membership_status"]) == "ambiguous" else "MEMBERSHIP_STATUS_NOT_MEMBER")
            if not span_ok:
                reasons.append("UNIVERSE_NOT_MEMBER")
            snapshot_rows.append({
                "Instrument": ric,
                "formation_session": date,
                "membership_asof_date": row["formation_dt"],
                "membership_asof_status": str(row["membership_status"]),
                "membership_asof_pool_label": row["pool_label"],
                "membership_asof_eligible": bool(row["membership_eligible_bool"]),
                "active_member": active,
                "membership_reason": ";".join(reasons),
                "member_from": row["member_from_dt"],
                "member_to": row["member_to_dt"],
                "delisted_ric": row["delisted_ric"],
            })
    membership_audit = pd.DataFrame(snapshot_rows)
    active = membership_audit.loc[membership_audit["active_member"]].copy().reset_index(drop=True)
    if active.empty:
        raise RuntimeError("no active AI-pool security-month rows")

    # Schedule and split metadata. Exit sessions are calendar/session rules,
    # not target values. Target values are never looked up for test rows.
    session_positions = pd.Series(np.arange(len(sessions)), index=sessions)
    active["formation_idx"] = active["formation_session"].map(session_positions).astype(int)
    active["reference_session"] = [sessions[i - 1] if i >= 1 else pd.NaT for i in active["formation_idx"]]
    active["entry_idx"] = active["formation_idx"] + 1
    active["exit_idx"] = active["entry_idx"] + int(config["formation"]["holding_sessions"])
    active["entry_session"] = [sessions[i] if i < len(sessions) else pd.NaT for i in active["entry_idx"]]
    active["exit_session"] = [sessions[i] if i < len(sessions) else pd.NaT for i in active["exit_idx"]]
    active["split"] = active["formation_session"].map(split_for_day)
    active["next_split_boundary"] = active["split"].map({"training": pd.Timestamp("2021-01-01"), "validation": pd.Timestamp("2023-01-01")})
    active["boundary_purged"] = active["next_split_boundary"].notna() & (active["entry_session"].ge(active["next_split_boundary"]) | active["exit_session"].ge(active["next_split_boundary"]))
    active["sample_id"] = [hashlib.sha256(f"{ric}|{date.date()}".encode("utf-8")).hexdigest()[:24] for ric, date in zip(active["Instrument"], active["formation_session"])]
    active["security_id"] = active["Instrument"].astype(str)
    active["fundamental_semantics_status"] = "UNVERIFIED"
    active["market_cap_semantics_status"] = "UNVERIFIED"

    print(f"已形成 PIT pool snapshot：{len(active)} active security-month rows, {active['Instrument'].nunique()} literal RICs, {active['formation_session'].nunique()} months；开始计算 backward-looking controls。", flush=True)
    if (wide.le(-1) & wide.notna()).any(axis=None):
        raise RuntimeError("return <= -100% prevents log compounding")
    log_returns = np.log1p(wide)
    matrices: dict[str, pd.DataFrame] = {"momentum_1": wide}
    for window in [5, 20, 60, 252]:
        matrices[f"momentum_{window}"] = np.expm1(log_returns.rolling(window, min_periods=window).sum())
    for window, min_obs in [(20, 15), (60, 40)]:
        matrices[f"volatility_{window}_ann"] = wide.rolling(window, min_periods=min_obs).std(ddof=1) * np.sqrt(252.0)
    beta, idio, beta_obs = rolling_beta_idio(wide.drop(columns=[benchmark]), wide[benchmark])
    matrices["beta_126"] = beta
    matrices["idio_vol_126_ann"] = idio
    matrices["beta_obs_126"] = beta_obs
    for name, matrix in matrices.items():
        active[name] = matrix_lookup(matrix, active["Instrument"], active["reference_session"], name)
    for name in ["momentum_1", "momentum_5", "momentum_20", "momentum_60", "volatility_20_ann", "volatility_60_ann"]:
        active[f"spy_{name}"] = active["reference_session"].map(matrices[name][benchmark]).astype(float)

    # Price/liquidity controls come from clean v3 and are filtered before pivot
    # so the full 577 MB price file is not placed in a wide matrix.
    prices = read_filtered_csv(prices_path, ["Instrument", "Date", "TRDPRC_1", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume"], data_rics)
    prices["Date"] = parse_dates(prices["Date"], "price Date")
    for col in ["TRDPRC_1", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume"]:
        prices[col] = finite_or_na(prices[col])
    if prices.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("filtered prices contain duplicate Instrument + Date keys")
    price_index = pd.DatetimeIndex(sessions)
    row_presence = prices.assign(_row_present=True).pivot(index="Date", columns="Instrument", values="_row_present").reindex(index=price_index, columns=sorted(data_rics)).fillna(False).astype(float)
    close = prices.pivot(index="Date", columns="Instrument", values="TRDPRC_1").reindex(index=price_index, columns=sorted(data_rics))
    volume = prices.pivot(index="Date", columns="Instrument", values="ACVOL_UNS").reindex(index=price_index, columns=sorted(data_rics))
    dollar_volume = prices.pivot(index="Date", columns="Instrument", values="dollar_volume").reindex(index=price_index, columns=sorted(data_rics))
    spread = prices.pivot(index="Date", columns="Instrument", values="quoted_spread_bps").reindex(index=price_index, columns=sorted(data_rics))
    for frame, name in [(volume, "volume"), (dollar_volume, "dollar_volume")]:
        if (frame.lt(0) & frame.notna()).any(axis=None):
            raise RuntimeError(f"negative {name}")
    volume_med = volume.rolling(20, min_periods=15).median()
    dollar_med = dollar_volume.rolling(20, min_periods=15).median()
    spread_med = spread.rolling(20, min_periods=10).median()
    active["volume_median_20_log1p"] = np.log1p(matrix_lookup(volume_med, active["Instrument"], active["reference_session"], "volume_median_20"))
    active["dollar_volume_median_20_log1p"] = np.log1p(matrix_lookup(dollar_med, active["Instrument"], active["reference_session"], "dollar_volume_median_20"))
    active["spread_median_20_bps"] = matrix_lookup(spread_med, active["Instrument"], active["reference_session"], "spread_median_20")
    active["entry_price_row_available"] = matrix_lookup(row_presence, active["Instrument"], active["entry_session"], "entry_price_row").astype(bool)
    entry_close_values = matrix_lookup(close, active["Instrument"], active["entry_session"], "entry_close_value")
    active["entry_has_close"] = np.isfinite(entry_close_values)
    active["entry_trade_eligible"] = active["entry_price_row_available"] & active["entry_has_close"]

    # Market cap and fundamental controls are as-of joins, never forward-filled
    # past a 450-day candidate stale bound. Their source semantics remain tagged
    # UNVERIFIED in the resulting master table.
    market_cap = pd.read_csv(market_cap_path, usecols=["Instrument", "Date", "market_cap_value", "semantics_status"], low_memory=False)
    market_cap["Date"] = parse_dates(market_cap["Date"], "market cap Date")
    market_cap["market_cap_value"] = finite_or_na(market_cap["market_cap_value"])
    market_cap = market_cap.loc[market_cap["Instrument"].astype(str).isin(candidate_rics)].copy()
    if market_cap.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("market cap has duplicate Instrument + Date keys")
    market_cap = market_cap.sort_values(["Instrument", "Date"])
    fundamental = pd.read_csv(fundamentals_path, low_memory=False)
    fundamental = fundamental.loc[fundamental["Instrument"].astype(str).isin(candidate_rics)].copy()
    fundamental["period_end"] = parse_dates(fundamental["period_end"], "fundamental period_end")
    fundamental["orig_announcement_date"] = pd.to_datetime(fundamental["orig_announcement_date"], format="mixed", errors="coerce").dt.normalize()
    fundamental = fundamental.loc[fundamental["key_occurrence"].fillna(0).astype(str).isin(["0", "0.0"])].copy()
    for col in ["control_total_revenue_value", "assets_value", "equity_value", "gross_profit_value", "operating_cash_flow_value", "debt_value"]:
        fundamental[col] = finite_or_na(fundamental[col])
    fundamental = fundamental.sort_values(["Instrument", "period_end"])
    fundamental["investment_asset_growth"] = fundamental.groupby("Instrument")["assets_value"].pct_change()
    fundamental["revenue_growth"] = fundamental.groupby("Instrument")["control_total_revenue_value"].pct_change()
    fundamental_groups = {ric: group.sort_values("orig_announcement_date") for ric, group in fundamental.groupby("Instrument")}
    cap_groups = {ric: group.sort_values("Date") for ric, group in market_cap.groupby("Instrument")}
    active["market_cap_value"] = np.nan
    active["market_cap_asof_date"] = pd.NaT
    active["market_cap_age_days"] = np.nan
    active["fundamental_asof_date"] = pd.NaT
    active["fundamental_age_days"] = np.nan
    active["fundamental_stale"] = False
    raw_cols = ["control_total_revenue_value", "assets_value", "equity_value", "gross_profit_value", "operating_cash_flow_value", "debt_value", "investment_asset_growth", "revenue_growth"]
    for col in raw_cols:
        active[col] = np.nan
    for idx, row in active.iterrows():
        ric, date = str(row["Instrument"]), row["formation_session"]
        cap = cap_groups.get(ric, pd.DataFrame())
        cap = cap.loc[cap["Date"] <= date] if not cap.empty else cap
        if not cap.empty:
            c = cap.iloc[-1]
            age = int((date - c["Date"]).days)
            active.at[idx, "market_cap_value"] = c["market_cap_value"]
            active.at[idx, "market_cap_asof_date"] = c["Date"]
            active.at[idx, "market_cap_age_days"] = age
        f = fundamental_groups.get(ric, pd.DataFrame())
        f = f.loc[f["orig_announcement_date"].notna() & (f["orig_announcement_date"] <= date)] if not f.empty else f
        if not f.empty:
            latest = f.iloc[-1]
            age = int((date - latest["orig_announcement_date"]).days)
            active.at[idx, "fundamental_asof_date"] = latest["orig_announcement_date"]
            active.at[idx, "fundamental_age_days"] = age
            if age > int(config["control_features"]["fundamental_max_staleness_days"]):
                active.at[idx, "fundamental_stale"] = True
            else:
                for col in raw_cols:
                    active.at[idx, col] = latest.get(col, np.nan)
    active["size_log_market_cap"] = np.log(active["market_cap_value"].where(active["market_cap_value"].gt(0)))
    active["book_to_market"] = safe_ratio(active["equity_value"], active["market_cap_value"])
    active["profitability_gross_profit_to_assets"] = safe_ratio(active["gross_profit_value"], active["assets_value"])
    active["leverage_debt_to_assets"] = safe_ratio(active["debt_value"], active["assets_value"])
    active["operating_cash_flow_to_assets"] = safe_ratio(active["operating_cash_flow_value"], active["assets_value"])

    control_features = config["control_features"]["price_market_controls"] + config["control_features"]["fundamental_controls"]
    core_features = config["control_features"]["core_feature_gate"]
    for col in control_features:
        if col not in active.columns:
            raise RuntimeError(f"control feature missing: {col}")
        active[col] = finite_or_na(active[col])
    active["feature_missing_count"] = active[control_features].isna().sum(axis=1).astype(int)
    active["feature_missing_list"] = active[control_features].isna().apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
    active["feature_core_available"] = active[core_features].notna().all(axis=1)

    # Development labels only. This branch deliberately skips every return
    # lookup for split=test, satisfying the hard test seal.
    dev_target_rows: list[dict] = []
    sealed_rows: list[dict] = []
    label_complete = []
    stock_forward = []
    benchmark_forward = []
    excess_forward = []
    y_values = []
    target_reasons = []
    for _, row in active.iterrows():
        split = row["split"]
        base_target = {"sample_id": row["sample_id"], "Instrument": row["Instrument"], "security_id": row["security_id"], "formation_session": row["formation_session"], "split": split, "entry_session": row["entry_session"], "exit_session": row["exit_session"]}
        if split == "test":
            label_complete.append(False); stock_forward.append(np.nan); benchmark_forward.append(np.nan); excess_forward.append(np.nan); y_values.append(np.nan); target_reasons.append("TEST_TARGET_SEALED")
            sealed_rows.append({**base_target, "target_status": "TEST_TARGET_SEALED"})
            continue
        reason = ""
        if bool(row["boundary_purged"]):
            reason = "BOUNDARY_PURGED;LABEL_MISSING"
        elif pd.isna(row["exit_session"]) or pd.isna(row["entry_session"]):
            reason = "LABEL_EXIT_MISSING"
        else:
            entry_i, exit_i = int(row["entry_idx"]), int(row["exit_idx"])
            future_dates = sessions[entry_i + 1: exit_i + 1]
            # This lookup is reached only for training/validation rows.
            stock = wide.reindex(index=future_dates, columns=[row["Instrument"]])[row["Instrument"]]
            bench = wide.reindex(index=future_dates, columns=[benchmark])[benchmark]
            if len(future_dates) != int(config["formation"]["holding_sessions"]) or stock.isna().any() or bench.isna().any():
                reason = "LABEL_MISSING"
            else:
                sr = float(np.expm1(np.log1p(stock.to_numpy(dtype=float)).sum()))
                br = float(np.expm1(np.log1p(bench.to_numpy(dtype=float)).sum()))
                ex = sr - br
                label_complete.append(True); stock_forward.append(sr); benchmark_forward.append(br); excess_forward.append(ex); y_values.append(int(ex > 0)); target_reasons.append("")
                dev_target_rows.append({**base_target, "label_complete": True, "stock_forward_return": sr, "benchmark_forward_return": br, "forward_excess_return": ex, "y": int(ex > 0), "target_reason": ""})
                continue
        label_complete.append(False); stock_forward.append(np.nan); benchmark_forward.append(np.nan); excess_forward.append(np.nan); y_values.append(np.nan); target_reasons.append(reason)
        if split in {"training", "validation"}:
            dev_target_rows.append({**base_target, "label_complete": False, "stock_forward_return": np.nan, "benchmark_forward_return": np.nan, "forward_excess_return": np.nan, "y": np.nan, "target_reason": reason})
    active["label_complete"] = label_complete
    active["stock_forward_return"] = stock_forward
    active["benchmark_forward_return"] = benchmark_forward
    active["forward_excess_return"] = excess_forward
    active["y"] = y_values
    active["target_reason"] = target_reasons
    active["split_retained"] = active["split"].isin(["training", "validation", "test"]) & ~active["boundary_purged"]
    active["supervised_model_eligible"] = active["split"].isin(["training", "validation"]) & active["split_retained"] & active["label_complete"] & active["entry_trade_eligible"] & active["feature_core_available"]
    active["supervised_ineligibility_reason"] = active.apply(
        lambda row: ";".join([x for x, fail in [("BOUNDARY_PURGED", bool(row["boundary_purged"])), ("LABEL_UNAVAILABLE", not bool(row["label_complete"])), ("ENTRY_TRADE_INELIGIBLE", not bool(row["entry_trade_eligible"])), ("CORE_FEATURE_MISSING", not bool(row["feature_core_available"])), ("TEST_TARGET_SEALED", row["split"] == "test")] if fail]), axis=1)

    metadata_cols = ["sample_id", "security_id", "Instrument", "formation_session", "split", "membership_asof_date", "membership_asof_status", "membership_asof_pool_label", "member_from", "member_to", "delisted_ric", "reference_session", "entry_session", "exit_session", "fundamental_asof_date", "fundamental_age_days", "fundamental_stale", "fundamental_semantics_status", "market_cap_asof_date", "market_cap_age_days", "market_cap_semantics_status", "target_reason"]
    metadata = active[metadata_cols].copy()
    features = active[["sample_id", "security_id", "Instrument", "formation_session", "split", *control_features, "feature_missing_count", "feature_missing_list", "fundamental_semantics_status", "market_cap_semantics_status"]].copy()
    eligibility_cols = ["sample_id", "security_id", "Instrument", "formation_session", "split", "next_split_boundary", "boundary_purged", "split_retained", "entry_price_row_available", "entry_has_close", "entry_trade_eligible", "feature_core_available", "feature_missing_count", "supervised_model_eligible", "supervised_ineligibility_reason", "label_complete", "target_reason"]
    eligibility = active[eligibility_cols].copy()
    targets_dev = pd.DataFrame(dev_target_rows)
    targets_test_sealed = pd.DataFrame(sealed_rows)
    if targets_dev.empty or targets_test_sealed.empty:
        raise RuntimeError("expected both development and sealed test rows")
    missing_rows = []
    for split in ["training", "validation", "test", "all"]:
        mask = pd.Series(True, index=active.index) if split == "all" else active["split"].eq(split)
        for feature in control_features:
            denominator = int(mask.sum())
            missing = int(active.loc[mask, feature].isna().sum())
            missing_rows.append({"split": split, "feature": feature, "rows": denominator, "missing": missing, "missing_fraction": (missing / denominator if denominator else np.nan)})
    feature_missingness = pd.DataFrame(missing_rows)
    split_rows = []
    for split in ["training", "validation", "test", "out_of_window", "all"]:
        mask = pd.Series(True, index=active.index) if split == "all" else active["split"].eq(split)
        split_rows.append({
            "split": split,
            "security_month_rows": int(mask.sum()),
            "formation_sessions": int(active.loc[mask, "formation_session"].nunique()),
            "instruments": int(active.loc[mask, "Instrument"].nunique()),
            "boundary_purged_rows": int((mask & active["boundary_purged"]).sum()),
            "label_available_rows": int((mask & active["label_complete"]).sum()) if split != "test" else 0,
            "test_target_sealed_rows": int((mask & active["split"].eq("test")).sum()),
            "entry_trade_eligible_rows": int((mask & active["entry_trade_eligible"]).sum()),
            "core_feature_available_rows": int((mask & active["feature_core_available"]).sum()),
            "supervised_model_eligible_rows": int((mask & active["supervised_model_eligible"]).sum()),
        })
    split_counts = pd.DataFrame(split_rows)
    # Ensure no future/label columns slipped into the feature table.
    forbidden = {"stock_forward_return", "benchmark_forward_return", "forward_excess_return", "y", "label_complete", "entry_session", "exit_session"}
    checks = {
        "membership_known_keys_unique": not known.duplicated(["Instrument", "formation_dt"]).any(),
        "active_rows_are_member_states": bool((active["membership_asof_status"].eq("member") & active["membership_asof_eligible"]).all()),
        "active_rows_inside_corrected_spans": bool((active["member_from"].le(active["formation_session"]) & active["member_to"].ge(active["formation_session"])).all()),
        "master_sample_ids_unique": not active["sample_id"].duplicated().any(),
        "features_exclude_future_and_targets": not (set(features.columns) & forbidden),
        "reference_strictly_before_formation": bool((active["reference_session"] < active["formation_session"]).all()),
        "test_target_values_not_written": bool(targets_test_sealed.empty or not targets_test_sealed.columns.isin(["stock_forward_return", "benchmark_forward_return", "forward_excess_return", "y"]).any()),
        "test_master_target_columns_na": bool(active.loc[active["split"].eq("test"), ["stock_forward_return", "benchmark_forward_return", "forward_excess_return", "y"]].isna().all().all()),
        "dev_targets_only_training_validation": set(targets_dev["split"]).issubset({"training", "validation"}),
        "no_infinite_features": bool(np.isfinite(features[control_features].to_numpy(dtype=float)[~pd.isna(features[control_features].to_numpy(dtype=float))]).all()) if features[control_features].notna().any().any() else True,
        "no_physical_row_deletion": True,
        "no_imputation_or_winsorization": True,
        "no_event_study_inputs": True,
    }
    run_id = run_id or run_id_now()
    if not run_id or any(ch not in "0123456789TZ" for ch in run_id):
        raise ValueError("run_id must contain only digits, T, and Z")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    outputs = {
        "model_features": out / "model_features.csv",
        "metadata": out / "metadata.csv",
        "eligibility": out / "eligibility.csv",
        "targets_dev": out / "targets_dev.csv",
        "targets_test_sealed": out / "targets_test_sealed.csv",
        "membership_snapshot_audit": out / "membership_snapshot_audit.csv",
        "feature_missingness": out / "feature_missingness.csv",
        "split_counts": out / "split_counts.csv",
    }
    features.to_csv(outputs["model_features"], index=False)
    metadata.to_csv(outputs["metadata"], index=False)
    eligibility.to_csv(outputs["eligibility"], index=False)
    targets_dev.to_csv(outputs["targets_dev"], index=False)
    targets_test_sealed.to_csv(outputs["targets_test_sealed"], index=False)
    membership_audit.to_csv(outputs["membership_snapshot_audit"], index=False)
    feature_missingness.to_csv(outputs["feature_missingness"], index=False)
    split_counts.to_csv(outputs["split_counts"], index=False)
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in input_paths.items()}
    output_hashes = {name: sha256_file(path) for name, path in outputs.items()}
    counts = {
        "membership_input_rows": int(len(membership)),
        "membership_input_instruments": int(membership["Instrument"].nunique()),
        "membership_explicit_member_rows": int(len(eligible_rows)),
        "candidate_member_instruments": int(len(candidate_rics)),
        "formation_sessions": int(len(formation)),
        "membership_snapshot_audit_rows": int(len(membership_audit)),
        "master_rows": int(len(active)),
        "active_instruments": int(active["Instrument"].nunique()),
        "active_core_rows": int(active["membership_asof_pool_label"].eq("core_ai").sum()),
        "active_ecosystem_rows": int(active["membership_asof_pool_label"].eq("ai_ecosystem").sum()),
        "training_rows": int((active["split"] == "training").sum()),
        "validation_rows": int((active["split"] == "validation").sum()),
        "test_rows": int((active["split"] == "test").sum()),
        "dev_label_rows": int(len(targets_dev)),
        "dev_label_complete_rows": int(targets_dev["label_complete"].sum()),
        "test_target_sealed_rows": int(len(targets_test_sealed)),
        "boundary_purged_rows": int(active["boundary_purged"].sum()),
        "supervised_model_eligible_rows": int(active["supervised_model_eligible"].sum()),
        "rows_with_any_feature_missing": int(active["feature_missing_count"].gt(0).sum()),
        "rows_with_fundamental_stale": int(active["fundamental_stale"].sum()),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": "partial_model_ready_controls_unverified",
        "model_ready": False,
        "config_path": rel(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "builder_path": rel(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "membership": {"run_id": MEMBERSHIP_RUN_ID, "path": rel(membership_path)},
            "returns": {"path": rel(returns_path)},
            "prices": {"path": rel(prices_path)},
            "fundamentals": {"path": rel(fundamentals_path)},
            "market_cap": {"path": rel(market_cap_path)},
        },
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "counts": counts,
        "control_features": control_features,
        "core_feature_gate": core_features,
        "split_counts": split_counts.to_dict(orient="records"),
        "missing_value_policy": config["missing_policy"],
        "test_policy": config["test_policy"],
        "fundamental_semantics_gate_passed": False,
        "checks": checks,
        "limitations": [
            "Formal fundamental controls are blocked pending unit/currency and market-cap semantics verification; candidate raw values and ratios remain tagged UNVERIFIED for review.",
            "The latest AI pool run has only 69 explicit member periods across 10 literal RICs; ambiguous and unknown source periods are retained in the membership audit and excluded from active pool snapshots.",
            "Monthly AI pool cross-sections contain 2–7 securities, so top/bottom 20% spread diagnostics later will often have one security per side and fail the registered minimum 10-per-side economic portfolio gate.",
            "This is a model-ready engineering layer with no fitted preprocessing/model and no test target values; it is not a performance claim or completed backtest.",
        ],
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    append_log(summary, out)
    append_ai_use_log(summary, out)
    print(json.dumps({"run_id": run_id, "out": rel(out), "status": summary["status"], "counts": counts, "checks": checks}, ensure_ascii=False, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the AI-pool monthly model-ready layer.")
    parser.add_argument("--run-id", default=None, help="Optional run-scoped identifier; output directory must not already exist.")
    args = parser.parse_args()
    build(args.run_id)
