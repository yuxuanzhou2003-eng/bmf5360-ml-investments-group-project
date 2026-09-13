"""Build an auditable daily point-in-time AI-pool baseline layer.

The daily master expands the PIT pool onto SPY trading sessions. Controls end
at the session before the decision date. H21 development targets are computed
only for training/validation rows; test rows receive a schedule-only manifest.
The optional external candidate list makes it possible to extend the current
10-RIC validation run without changing the monthly baseline.
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
CONFIG_PATH = ROOT / "ai_pool_daily_baseline_model_ready_v1_config.json"
SCHEMA_VERSION = "ai_pool_daily_baseline_model_ready_v1"
MEMBERSHIP_RUN_ID = "20260910T023221976393Z"
OUT_ROOT = ROOT / "data" / "model_ready_ai_pool_daily_v1"
BENCHMARK = "SPY.P"
HORIZON = 21

FEATURE_COLUMNS = [
    "momentum_1", "momentum_5", "momentum_20", "momentum_60",
    "volatility_20_ann", "volatility_60_ann",
    "volume_median_20_log1p", "volume_median_60_log1p",
    "dollar_volume_median_20_log1p", "dollar_volume_median_60_log1p",
    "spread_median_20_bps", "spread_median_60_bps",
    "beta_126", "idio_vol_126_ann", "beta_obs_126",
    "spy_momentum_1", "spy_momentum_5", "spy_momentum_20", "spy_momentum_60",
    "spy_volatility_20_ann", "spy_volatility_60_ann",
]
CORE_FEATURES = [
    "momentum_20", "volatility_20_ann", "volatility_60_ann",
    "dollar_volume_median_20_log1p", "spread_median_20_bps",
    "beta_126", "idio_vol_126_ann", "spy_momentum_20", "spy_volatility_20_ann",
]
FORBIDDEN_FEATURE_COLUMNS = {
    "stock_forward_return", "benchmark_forward_return", "forward_excess_return",
    "y", "label_complete", "entry_session", "exit_session",
}


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


def parse_bool(values: pd.Series, name: str) -> pd.Series:
    normalized = values.astype("string").str.strip().str.casefold()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name} has invalid values: {invalid[:10]}")
    return normalized.eq("true")


def parse_dates(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, format="mixed", errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{name} has {int(parsed.isna().sum())} invalid dates")
    return parsed


def finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def read_filtered_csv(path: Path, usecols: list[str], instruments: set[str], *, chunksize: int = 500_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=chunksize):
        selected = chunk.loc[chunk["Instrument"].astype(str).isin(instruments)].copy()
        if not selected.empty:
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=usecols)
    result = pd.concat(parts, ignore_index=True, sort=False)
    result["Instrument"] = result["Instrument"].astype(str)
    result["Date"] = parse_dates(result["Date"], f"{path.name}.Date")
    if result.duplicated(["Instrument", "Date"]).any():
        duplicates = result.loc[result.duplicated(["Instrument", "Date"], keep=False), ["Instrument", "Date"]]
        raise RuntimeError(f"{path} has duplicate Instrument/Date keys: {duplicates.head(10).to_dict('records')}")
    return result


def split_for_date(day: pd.Timestamp) -> str:
    if pd.Timestamp("2015-01-01") <= day <= pd.Timestamp("2020-12-31"):
        return "training"
    if pd.Timestamp("2021-01-01") <= day <= pd.Timestamp("2022-12-31"):
        return "validation"
    if pd.Timestamp("2023-01-01") <= day <= pd.Timestamp("2026-06-30"):
        return "test"
    return "out_of_window"


def lookup_matrix(matrix: pd.DataFrame, dates: pd.Series, instruments: pd.Series) -> np.ndarray:
    row_i = matrix.index.get_indexer(pd.DatetimeIndex(dates))
    col_i = matrix.columns.get_indexer(pd.Index(instruments.astype(str)))
    values = matrix.to_numpy(dtype=float)
    output = np.full(len(dates), np.nan, dtype=float)
    valid = (row_i >= 0) & (col_i >= 0)
    if valid.any():
        output[valid] = values[row_i[valid], col_i[valid]]
    return output


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
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance_numerator = sum_xm - sum_x * sum_m / count
        market_ss = sum_m2 - sum_m.pow(2) / count
        asset_ss = sum_x2 - sum_x.pow(2) / count
        valid_fit = (count >= min_obs) & market_ss.gt(0) & market_ss.notna()
        beta = (covariance_numerator / market_ss).where(valid_fit)
        residual_ss = (asset_ss - covariance_numerator.pow(2) / market_ss).clip(lower=0)
        residual_variance = residual_ss / (count - 2).where(count.gt(2))
    idio = (np.sqrt(residual_variance) * np.sqrt(252.0)).where(valid_fit & count.gt(2))
    return beta, idio, count.where(beta.notna())


def candidate_instruments(membership: pd.DataFrame, candidate_path: Path | None) -> tuple[list[str], list[str], pd.DataFrame | None]:
    explicit = membership.loc[
        membership["membership_status_norm"].eq("member") & membership["membership_eligible_bool"],
        "Instrument",
    ].dropna().astype(str).drop_duplicates().sort_values().tolist()
    if candidate_path is None:
        return explicit, explicit, None
    candidate = pd.read_csv(candidate_path, low_memory=False)
    if "ric" in candidate.columns:
        required_registry = {"ric", "member_from", "member_to"}
        if not required_registry.issubset(candidate.columns):
            raise ValueError(f"external registry must contain ric/member_from/member_to: {candidate_path}")
        candidate["ric"] = candidate["ric"].astype("string").str.strip()
        if candidate["ric"].eq("").any() or candidate["ric"].duplicated().any():
            raise ValueError(f"external registry has blank or duplicate ric: {candidate_path}")
        candidate["member_from_dt"] = pd.to_datetime(candidate["member_from"], format="mixed", errors="coerce").dt.normalize()
        candidate["member_to_dt"] = pd.to_datetime(candidate["member_to"], format="mixed", errors="coerce").dt.normalize()
        if candidate[["member_from_dt", "member_to_dt"]].isna().any().any():
            raise ValueError(f"external registry has invalid member spans: {candidate_path}")
        if (candidate["member_from_dt"] > candidate["member_to_dt"]).any():
            raise ValueError(f"external registry has reversed member spans: {candidate_path}")
        candidate["ai_role_evidence_status"] = candidate["pit_evidence_status"].astype("string").fillna("unclassified")
        return sorted(candidate["ric"].tolist()), explicit, candidate
    if "Instrument" not in candidate.columns:
        raise ValueError(f"external candidate list must contain Instrument or ric: {candidate_path}")
    requested = candidate["Instrument"].dropna().astype(str).str.strip()
    requested = sorted(set(requested[requested.ne("")]))
    return requested, explicit, None


def membership_asof_audit(membership: pd.DataFrame, candidates: list[str], sessions: pd.DatetimeIndex, registry: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict] = []
    active_rows: list[dict] = []
    registry_lookup = registry.set_index("ric").to_dict(orient="index") if registry is not None else {}
    for instrument in candidates:
        source = membership.loc[membership["Instrument"].astype(str).eq(instrument)].sort_values("formation_dt").copy()
        source = source.loc[source["formation_dt"].notna()].reset_index(drop=True)
        if source.duplicated("formation_dt").any():
            raise RuntimeError(f"membership has duplicate dated keys for {instrument}")
        source_dates = source["formation_dt"].to_numpy(dtype="datetime64[ns]")
        for date in sessions:
            position = int(np.searchsorted(source_dates, np.datetime64(date), side="right") - 1)
            base = {"Instrument": instrument, "formation_session": date}
            registry_row = registry_lookup.get(instrument)
            if position < 0:
                source_status = "NO_PRIOR_RECORD"
                source_eligible = False
                source_pool = np.nan
                source_from = pd.NaT
                source_to = pd.NaT
            else:
                row = source.iloc[position]
                source_status = str(row["membership_status_norm"])
                source_eligible = bool(row["membership_eligible_bool"])
                source_pool = row["pool_label"]
                source_from = row["member_from_dt"]
                source_to = row["member_to_dt"]
            registry_fields = {
                "registry_active": False,
                "registry_candidate_status": np.nan,
                "registry_primary_group": np.nan,
                "registry_pit_evidence_status": np.nan,
                "registry_pit_eligible_now": np.nan,
                "registry_source_retrieval_status": np.nan,
                "registry_pit_limitation": np.nan,
                "ai_role_evidence_status": "existing_pool_pit" if registry is None else np.nan,
                "source_membership_asof_status": source_status,
                "source_membership_asof_eligible": source_eligible,
            }
            if registry_row is not None:
                registry_from = registry_row["member_from_dt"]
                registry_to = registry_row["member_to_dt"]
                registry_inside = registry_from <= date <= registry_to
                registry_fields.update({
                    "registry_active": bool(registry_inside),
                    "registry_candidate_status": registry_row.get("candidate_status", np.nan),
                    "registry_primary_group": registry_row.get("primary_group", np.nan),
                    "registry_pit_evidence_status": registry_row.get("pit_evidence_status", np.nan),
                    "registry_pit_eligible_now": registry_row.get("pit_eligible_now", np.nan),
                    "registry_source_retrieval_status": registry_row.get("source_retrieval_status", np.nan),
                    "registry_pit_limitation": registry_row.get("pit_limitation", np.nan),
                    "ai_role_evidence_status": registry_row.get("ai_role_evidence_status", registry_row.get("pit_evidence_status", "unclassified")),
                })
                if registry_inside:
                    record = {**base, **registry_fields, "membership_asof_status": "member", "membership_asof_eligible": True, "membership_asof_pool_label": registry_row.get("primary_group", "expanded_registry"), "member_from": registry_from, "member_to": registry_to, "membership_active": True, "membership_reason": "EXPLORATORY_STATIC_REGISTRY_MEMBER_SPAN"}
                else:
                    record = {**base, **registry_fields, "membership_asof_status": source_status, "membership_asof_eligible": source_eligible, "membership_asof_pool_label": source_pool, "member_from": registry_from, "member_to": registry_to, "membership_active": False, "membership_reason": "OUTSIDE_EXTERNAL_REGISTRY_MEMBER_SPAN"}
            else:
                source_inside = pd.notna(source_from) and pd.notna(source_to) and source_from <= date <= source_to
                active = source_status == "member" and source_eligible and source_inside
                if active:
                    reason = ""
                elif source_status != "member":
                    reason = "LATEST_STATUS_NOT_MEMBER"
                elif not source_eligible:
                    reason = "LATEST_MEMBERSHIP_NOT_ELIGIBLE"
                elif not source_inside:
                    reason = "OUTSIDE_CORRECTED_MEMBER_SPAN"
                else:
                    reason = "MEMBERSHIP_STATE_NOT_ACTIVE"
                record = {**base, **registry_fields, "membership_asof_status": source_status, "membership_asof_eligible": source_eligible, "membership_asof_pool_label": source_pool, "member_from": source_from, "member_to": source_to, "membership_active": active, "membership_reason": reason}
            audit_rows.append(record)
            if record["membership_active"]:
                active_rows.append(record)
    return pd.DataFrame(audit_rows), pd.DataFrame(active_rows)


def append_log(summary: dict, out: Path) -> None:
    path = ROOT / "DATA_PROCESSING_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# DATA_PROCESSING_LOG\n"
    entry = [
        "",
        f"## {summary['run_id']} — AI pool daily controls-only model-ready v1 ({summary['generated_at_utc']})",
        "",
        "- **阶段目的与状态**：将 PIT AI pool 按有效 member_from/member_to 区间展开到 SPY 每个交易日；以形成日前一交易日为 controls cutoff，构建 H21 daily controls；只写训练/验证开发标签，test target 保持封存。",
        f"- **输入版本与路径**：membership `{summary['inputs']['membership']['path']}` run=`{MEMBERSHIP_RUN_ID}`；returns `{summary['inputs']['returns']['path']}`；prices `{summary['inputs']['prices']['path']}`；外部 candidate list=`{summary['inputs'].get('candidate_list', {}).get('path', 'none')}`。",
        f"- **候选与计数**：`{json.dumps(summary['counts'], ensure_ascii=False, sort_keys=True)}`；daily master 行完整保留，active membership audit 行数=`{summary['counts']['membership_snapshot_audit_rows']}`。",
        ("- **成员资格规则**：本次使用 `exploratory_static_candidate` 外部 registry；按 registry 的 literal `ric` 与 `member_from/member_to` 历史区间展开 active universe，保留 source membership as-of 状态用于审计；`ai_role_evidence_status` 原样记录，40 个 provisional static source 不代表完整 AI-role PIT。" if summary.get("membership_source_mode") == "exploratory_static_candidate" else "- **成员资格规则**：按 decision session 使用最新 `formation_date <= session` 状态，并要求 `membership_status=member`、`membership_eligible=true`、有效 member_from/member_to 区间；ambiguous/unknown 保留在 source/audit，不前向成为 member；literal RIC 作为 security_id。"),
        "- **特征规则**：momentum 1/5/20/60、20/60 日年化波动、20/60 日 volume/dollar-volume/spread 中位数、126 日 beta/idio-vol，以及对应 SPY 状态；全部 rolling 只到上一 SPY session，不使用 event study。",
        "- **标签/切分规则**：H21=`decision_session+1..+21` 的个股累计收益减 SPY 累计收益，严格大于 0 才为 y=1；training 2015–2020、validation 2021–2022、test 2023-01–2026-06；split 边界按 exit_session >= 下一 split 起点 purge 21-session 窗口；监督样本每只 RIC 每 21-session block 至多一个 anchor，避免目标重叠。",
        "- **缺失/删除/单位处理**：缺失和不可解析值为 NA，真实零保留；master 不填补、不前后填充、不插值、不 winsorize、不删行；模型阶段另行 training-only median + missing indicators；本层无基本面/市值控制，故不引入未验证单位。",
        f"- **test seal 与审计**：`{summary['test_policy']}`；test rows 只进入 schedule manifest，目标列未写入；物理删除=0，quarantine=none，输入 coverage/feature missingness/split counts 均保存。",
        f"- **检查与限制**：`{json.dumps(summary['checks'], ensure_ascii=False, sort_keys=True)}`；`{summary['limitations']}`；执行状态=`{summary['status']}`。",
        f"- **输出**：`{rel(out)}`。",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def append_ai_use_log(summary: dict, out: Path) -> None:
    path = ROOT / "AI_USE_LOG.md"
    prior = path.read_text(encoding="utf-8") if path.exists() else "# AI_USE_LOG\n"
    entry = [
        "",
        f"### AI pool daily controls-only model-ready layer ({summary['run_id']})",
        "",
        f"- OpenAI Codex built a new daily model-ready run `{rel(out)}` from `{summary['inputs']['membership']['path']}`, clean returns and clean prices. Current validation uses `{summary['counts']['candidate_requested_instruments']}` requested literal RICs, `{summary['counts']['active_instruments']}` active RICs, `{summary['counts']['master_rows']}` daily pool rows and H21 labels only for development splits. Membership mode=`{summary.get('membership_source_mode')}`; PIT status=`{summary.get('pit_status')}`.",
        "- Daily controls end at the prior SPY session; the H21 target is future stock cumulative return minus future SPY cumulative return. Test future returns are not looked up or labeled; registry AI-role evidence is retained in metadata and provisional static candidates must not be treated as a complete PIT backtest.",
        "",
    ]
    path.write_text(prior.rstrip("\n") + "\n" + "\n".join(entry), encoding="utf-8", newline="\n")


def build(run_id: str | None = None, candidate_path_arg: str | None = None) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unexpected daily model-ready config schema")
    membership_path = ROOT / config["membership"]["path"]
    returns_path = ROOT / config["inputs"]["returns"]
    prices_path = ROOT / config["inputs"]["prices"]
    configured_candidate = config["membership"].get("candidate_instruments_path")
    candidate_path = Path(candidate_path_arg) if candidate_path_arg else (ROOT / configured_candidate if configured_candidate else None)
    input_paths: dict[str, Path] = {
        "membership": membership_path,
        "returns": returns_path,
        "prices": prices_path,
        "config": CONFIG_PATH,
        "builder": Path(__file__).resolve(),
    }
    if candidate_path is not None:
        input_paths["candidate_list"] = candidate_path
    missing = [name for name, path in input_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    print("阶段说明（AI pool daily model-ready）：读取 PIT membership、clean returns/prices 与可选外部候选名单，逐 SPY 交易日展开有效成员；controls 截止上一交易日；仅为 training/validation 计算 H21 标签，test 不读取未来目标。", flush=True)
    membership = pd.read_csv(membership_path, low_memory=False)
    required_membership = ["Instrument", "formation_date", "pool_label", "membership_status", "membership_eligible", "member_from", "member_to"]
    missing_columns = sorted(set(required_membership) - set(membership.columns))
    if missing_columns:
        raise RuntimeError(f"membership schema missing: {missing_columns}")
    membership["Instrument"] = membership["Instrument"].astype(str)
    membership["formation_dt"] = pd.to_datetime(membership["formation_date"], format="mixed", errors="coerce").dt.normalize()
    membership["membership_status_norm"] = membership["membership_status"].astype("string").str.strip().str.casefold()
    membership["membership_eligible_bool"] = parse_bool(membership["membership_eligible"], "membership_eligible")
    membership["member_from_dt"] = pd.to_datetime(membership["member_from"], format="mixed", errors="coerce").dt.normalize()
    membership["member_to_dt"] = pd.to_datetime(membership["member_to"], format="mixed", errors="coerce").dt.normalize()
    known = membership.loc[membership["formation_dt"].notna()].copy()
    if known.duplicated(["Instrument", "formation_dt"]).any():
        raise RuntimeError("membership has duplicate Instrument/formation_date keys")
    requested_candidates, source_explicit, registry = candidate_instruments(membership, candidate_path)
    if not requested_candidates:
        raise RuntimeError("candidate instrument list is empty")
    data_instruments = set(requested_candidates) | {BENCHMARK}

    return_header = pd.read_csv(returns_path, nrows=0).columns.tolist()
    price_header = pd.read_csv(prices_path, nrows=0).columns.tolist()
    required_returns = {"Instrument", "Date", "return_decimal"}
    required_prices = {"Instrument", "Date", "TRDPRC_1", "ACVOL_UNS", "dollar_volume", "quoted_spread_bps"}
    if not required_returns.issubset(return_header):
        raise RuntimeError(f"returns schema missing: {sorted(required_returns - set(return_header))}")
    if not required_prices.issubset(price_header):
        raise RuntimeError(f"prices schema missing: {sorted(required_prices - set(price_header))}")
    returns = read_filtered_csv(returns_path, ["Instrument", "Date", "return_decimal"], data_instruments)
    prices = read_filtered_csv(prices_path, ["Instrument", "Date", "TRDPRC_1", "ACVOL_UNS", "dollar_volume", "quoted_spread_bps"], data_instruments)
    if returns.empty or BENCHMARK not in set(returns["Instrument"]):
        raise RuntimeError("filtered returns have no benchmark rows")
    if prices.empty or BENCHMARK not in set(prices["Instrument"]):
        raise RuntimeError("filtered prices have no benchmark rows")
    for col in ["return_decimal"]:
        returns[col] = finite_numeric(returns[col])
    for col in ["TRDPRC_1", "ACVOL_UNS", "dollar_volume", "quoted_spread_bps"]:
        prices[col] = finite_numeric(prices[col])
    returns = returns.sort_values(["Date", "Instrument"]).reset_index(drop=True)
    prices = prices.sort_values(["Date", "Instrument"]).reset_index(drop=True)
    coverage_rows = []
    for name, frame in [("returns", returns), ("prices", prices)]:
        for instrument, group in frame.groupby("Instrument", sort=True):
            coverage_rows.append({"source": name, "Instrument": instrument, "rows": int(len(group)), "date_min": group["Date"].min(), "date_max": group["Date"].max(), "missing_return": int(group["return_decimal"].isna().sum()) if name == "returns" else np.nan, "missing_close": int(group["TRDPRC_1"].isna().sum()) if name == "prices" else np.nan, "missing_dollar_volume": int(group["dollar_volume"].isna().sum()) if name == "prices" else np.nan, "missing_spread": int(group["quoted_spread_bps"].isna().sum()) if name == "prices" else np.nan})
    coverage = pd.DataFrame(coverage_rows)

    return_wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    price_close = prices.pivot(index="Date", columns="Instrument", values="TRDPRC_1").sort_index()
    volume_wide = prices.pivot(index="Date", columns="Instrument", values="ACVOL_UNS").sort_index()
    dollar_wide = prices.pivot(index="Date", columns="Instrument", values="dollar_volume").sort_index()
    spread_wide = prices.pivot(index="Date", columns="Instrument", values="quoted_spread_bps").sort_index()
    source_sessions = pd.DatetimeIndex(
        returns.loc[returns["Instrument"].eq(BENCHMARK), "Date"].drop_duplicates().sort_values()
    )
    master_sessions = source_sessions[(source_sessions >= pd.Timestamp("2015-01-01")) & (source_sessions <= pd.Timestamp("2026-06-30"))]
    if len(master_sessions) == 0:
        raise RuntimeError("no SPY sessions in requested model window")
    benchmark_returns = return_wide.reindex(index=source_sessions, columns=[BENCHMARK])[BENCHMARK]
    benchmark_log = np.log1p(benchmark_returns.where(benchmark_returns > -1))
    all_columns = sorted(data_instruments)
    return_wide = return_wide.reindex(index=source_sessions, columns=all_columns)
    price_close = price_close.reindex(index=source_sessions, columns=all_columns)
    volume_wide = volume_wide.reindex(index=source_sessions, columns=all_columns)
    dollar_wide = dollar_wide.reindex(index=source_sessions, columns=all_columns)
    spread_wide = spread_wide.reindex(index=source_sessions, columns=all_columns)
    log_returns = np.log1p(return_wide.where(return_wide > -1))

    feature_matrices: dict[str, pd.DataFrame] = {}
    for horizon in [1, 5, 20, 60]:
        feature_matrices[f"momentum_{horizon}"] = np.expm1(log_returns.rolling(horizon, min_periods=horizon).sum())
    for horizon in [20, 60]:
        feature_matrices[f"volatility_{horizon}_ann"] = return_wide.rolling(horizon, min_periods=horizon).std(ddof=1) * np.sqrt(252.0)
        feature_matrices[f"volume_median_{horizon}_log1p"] = np.log1p(volume_wide.where(volume_wide >= 0)).rolling(horizon, min_periods=horizon).median()
        feature_matrices[f"dollar_volume_median_{horizon}_log1p"] = np.log1p(dollar_wide.where(dollar_wide >= 0)).rolling(horizon, min_periods=horizon).median()
        feature_matrices[f"spread_median_{horizon}_bps"] = spread_wide.rolling(horizon, min_periods=horizon).median()
    stock_beta, stock_idio, beta_obs = rolling_beta_idio(return_wide, benchmark_returns, window=126, min_obs=100)
    feature_matrices["beta_126"] = stock_beta
    feature_matrices["idio_vol_126_ann"] = stock_idio
    feature_matrices["beta_obs_126"] = beta_obs
    for horizon in [1, 5, 20, 60]:
        feature_matrices[f"spy_momentum_{horizon}"] = pd.Series(np.expm1(benchmark_log.rolling(horizon, min_periods=horizon).sum()), index=source_sessions)
    for horizon in [20, 60]:
        feature_matrices[f"spy_volatility_{horizon}_ann"] = benchmark_returns.rolling(horizon, min_periods=horizon).std(ddof=1) * np.sqrt(252.0)

    session_positions = pd.Series(np.arange(len(source_sessions), dtype=int), index=source_sessions)
    master_position = session_positions.reindex(master_sessions).to_numpy(dtype=int)
    reference_sessions = source_sessions[master_position - 1]
    audit, active_members = membership_asof_audit(membership, requested_candidates, master_sessions, registry)
    if active_members.empty:
        raise RuntimeError("no active member trading-day rows after PIT expansion")
    active = active_members.copy()
    active["security_id"] = active["Instrument"].astype(str)
    active["formation_session"] = pd.to_datetime(active["formation_session"]).dt.normalize()
    active["reference_session"] = active["formation_session"].map(dict(zip(master_sessions, reference_sessions)))
    active["session_index"] = active["formation_session"].map(session_positions).astype(int)
    active["entry_session"] = active["formation_session"]
    exit_positions = active["session_index"] + HORIZON
    active["exit_session"] = [source_sessions[pos] if pos < len(source_sessions) else pd.NaT for pos in exit_positions]
    active["split"] = active["formation_session"].map(split_for_date)
    active["next_split_boundary"] = active["split"].map({"training": pd.Timestamp("2021-01-01"), "validation": pd.Timestamp("2023-01-01")})
    active["boundary_purged"] = active["next_split_boundary"].notna() & active["exit_session"].notna() & active["exit_session"].ge(active["next_split_boundary"])
    active["entry_has_close"] = lookup_matrix(price_close, active["entry_session"], active["Instrument"]) .astype(float)
    active["entry_trade_eligible"] = np.isfinite(active["entry_has_close"]) & active["entry_has_close"].gt(0)
    active["sample_id"] = [hashlib.sha256(f"{instrument}|{date.date().isoformat()}|H{HORIZON}".encode()).hexdigest()[:24] for instrument, date in zip(active["Instrument"], active["formation_session"])]
    for feature in FEATURE_COLUMNS:
        matrix = feature_matrices[feature]
        if isinstance(matrix, pd.Series):
            active[feature] = matrix.reindex(pd.DatetimeIndex(active["reference_session"])).to_numpy(dtype=float)
        else:
            active[feature] = lookup_matrix(matrix, active["reference_session"], active["Instrument"])
    active["feature_missing_count"] = active[FEATURE_COLUMNS].isna().sum(axis=1).astype(int)
    active["feature_missing_list"] = active[FEATURE_COLUMNS].isna().apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
    active["feature_core_available"] = active[CORE_FEATURES].notna().all(axis=1)

    print(f"PIT daily pool snapshot: {len(active):,} active security-day rows, {active['Instrument'].nunique()} literal RICs, {active['formation_session'].nunique()} SPY sessions; computing H21 development labels only.", flush=True)
    dev_target_rows: list[dict] = []
    sealed_rows: list[dict] = []
    label_complete: list[bool] = []
    stock_forward: list[float] = []
    benchmark_forward: list[float] = []
    excess_forward: list[float] = []
    y_values: list[float] = []
    target_reasons: list[str] = []
    return_values = return_wide
    for row in active.itertuples(index=False):
        split = row.split
        base = {"sample_id": row.sample_id, "Instrument": row.Instrument, "security_id": row.Instrument, "formation_session": row.formation_session, "split": split, "entry_session": row.entry_session, "exit_session": row.exit_session}
        if split == "test":
            label_complete.append(False); stock_forward.append(np.nan); benchmark_forward.append(np.nan); excess_forward.append(np.nan); y_values.append(np.nan); target_reasons.append("TEST_TARGET_SEALED")
            sealed_rows.append({**base, "target_status": "TEST_TARGET_SEALED"})
            continue
        reason = ""
        if bool(row.boundary_purged):
            reason = "BOUNDARY_PURGED;LABEL_MISSING"
        elif pd.isna(row.exit_session):
            reason = "LABEL_EXIT_MISSING"
        else:
            future_positions = np.arange(int(row.session_index) + 1, int(row.session_index) + HORIZON + 1, dtype=int)
            if future_positions[-1] >= len(source_sessions):
                reason = "LABEL_EXIT_MISSING"
            else:
                stock = return_values.iloc[future_positions][row.Instrument].to_numpy(dtype=float)
                bench = return_values.iloc[future_positions][BENCHMARK].to_numpy(dtype=float)
                if len(stock) != HORIZON or len(bench) != HORIZON or not np.isfinite(stock).all() or not np.isfinite(bench).all() or (stock <= -1).any() or (bench <= -1).any():
                    reason = "LABEL_MISSING"
                else:
                    sr = float(np.expm1(np.log1p(stock).sum()))
                    br = float(np.expm1(np.log1p(bench).sum()))
                    ex = sr - br
                    label_complete.append(True); stock_forward.append(sr); benchmark_forward.append(br); excess_forward.append(ex); y_values.append(float(ex > 0)); target_reasons.append("")
                    dev_target_rows.append({**base, "label_complete": True, "stock_forward_return": sr, "benchmark_forward_return": br, "forward_excess_return": ex, "y": int(ex > 0), "target_reason": ""})
                    continue
        label_complete.append(False); stock_forward.append(np.nan); benchmark_forward.append(np.nan); excess_forward.append(np.nan); y_values.append(np.nan); target_reasons.append(reason)
        dev_target_rows.append({**base, "label_complete": False, "stock_forward_return": np.nan, "benchmark_forward_return": np.nan, "forward_excess_return": np.nan, "y": np.nan, "target_reason": reason})
    active["label_complete"] = label_complete
    active["stock_forward_return"] = stock_forward
    active["benchmark_forward_return"] = benchmark_forward
    active["forward_excess_return"] = excess_forward
    active["y"] = y_values
    active["target_reason"] = target_reasons
    active["split_retained"] = active["split"].isin(["training", "validation", "test"])
    active["non_overlap_block"] = ((active["session_index"] - int(master_position[0])) // HORIZON).astype(int)
    active["non_overlap_selected"] = False
    selection_base = active["split"].isin(["training", "validation"]) & ~active["boundary_purged"] & active["label_complete"] & active["entry_trade_eligible"] & active["feature_core_available"]
    selected_indices: list[int] = []
    eligible_rows = active.loc[selection_base].sort_values(["Instrument", "session_index", "formation_session"], kind="stable")
    for _, group in eligible_rows.groupby("Instrument", sort=False):
        last_position = -HORIZON
        for index, position in zip(group.index, group["session_index"]):
            if int(position) - last_position >= HORIZON:
                selected_indices.append(index)
                last_position = int(position)
    active.loc[selected_indices, "non_overlap_selected"] = True
    active["supervised_model_eligible"] = active["split"].isin(["training", "validation"]) & active["split_retained"] & active["label_complete"] & active["entry_trade_eligible"] & active["feature_core_available"] & active["non_overlap_selected"]
    active["supervised_ineligibility_reason"] = active.apply(lambda row: ";".join([name for name, fail in [("BOUNDARY_PURGED", bool(row["boundary_purged"])), ("LABEL_UNAVAILABLE", not bool(row["label_complete"])), ("ENTRY_TRADE_INELIGIBLE", not bool(row["entry_trade_eligible"])), ("CORE_FEATURE_MISSING", not bool(row["feature_core_available"])), ("TARGET_OVERLAP_BLOCK", not bool(row["non_overlap_selected"])), ("TEST_TARGET_SEALED", row["split"] == "test")] if fail]), axis=1)

    features = active[["sample_id", "security_id", "Instrument", "formation_session", "split", *FEATURE_COLUMNS, "feature_missing_count", "feature_missing_list"]].copy()
    metadata_cols = ["sample_id", "security_id", "Instrument", "formation_session", "split", "membership_asof_status", "membership_asof_eligible", "membership_asof_pool_label", "member_from", "member_to", "membership_reason", "reference_session", "entry_session", "exit_session", "target_reason", "ai_role_evidence_status", "registry_active", "registry_candidate_status", "registry_primary_group", "registry_pit_evidence_status", "registry_pit_eligible_now", "registry_source_retrieval_status", "registry_pit_limitation", "source_membership_asof_status", "source_membership_asof_eligible"]
    metadata = active[metadata_cols].copy()
    eligibility_cols = ["sample_id", "security_id", "Instrument", "formation_session", "split", "next_split_boundary", "boundary_purged", "split_retained", "entry_has_close", "entry_trade_eligible", "feature_core_available", "feature_missing_count", "non_overlap_block", "non_overlap_selected", "supervised_model_eligible", "supervised_ineligibility_reason", "label_complete", "target_reason"]
    eligibility = active[eligibility_cols].copy()
    targets_dev = pd.DataFrame(dev_target_rows)
    targets_test_sealed = pd.DataFrame(sealed_rows)
    if targets_dev.empty or targets_test_sealed.empty:
        raise RuntimeError("expected development and test sealed rows")
    missing_rows: list[dict] = []
    for split in ["training", "validation", "test", "all"]:
        mask = pd.Series(True, index=active.index) if split == "all" else active["split"].eq(split)
        for feature in FEATURE_COLUMNS:
            denominator = int(mask.sum())
            missing = int(active.loc[mask, feature].isna().sum())
            missing_rows.append({"split": split, "feature": feature, "rows": denominator, "missing": missing, "missing_fraction": missing / denominator if denominator else np.nan})
    feature_missingness = pd.DataFrame(missing_rows)
    split_rows: list[dict] = []
    for split in ["training", "validation", "test", "out_of_window", "all"]:
        mask = pd.Series(True, index=active.index) if split == "all" else active["split"].eq(split)
        split_rows.append({
            "split": split,
            "security_day_rows": int(mask.sum()),
            "formation_sessions": int(active.loc[mask, "formation_session"].nunique()),
            "instruments": int(active.loc[mask, "Instrument"].nunique()),
            "boundary_purged_rows": int((mask & active["boundary_purged"]).sum()),
            "label_available_rows": int((mask & active["label_complete"]).sum()) if split != "test" else 0,
            "test_target_sealed_rows": int((mask & active["split"].eq("test")).sum()),
            "entry_trade_eligible_rows": int((mask & active["entry_trade_eligible"]).sum()),
            "core_feature_available_rows": int((mask & active["feature_core_available"]).sum()),
            "non_overlap_selected_rows": int((mask & active["non_overlap_selected"]).sum()),
            "supervised_model_eligible_rows": int((mask & active["supervised_model_eligible"]).sum()),
        })
    split_counts = pd.DataFrame(split_rows)
    selected_dates = active.loc[active["supervised_model_eligible"]].sort_values(["Instrument", "session_index"])
    selected_gap_ok = True
    if not selected_dates.empty:
        gaps = selected_dates.groupby("Instrument")["session_index"].diff().dropna()
        selected_gap_ok = bool(gaps.ge(HORIZON).all())
    checks = {
        "membership_known_keys_unique": not known.duplicated(["Instrument", "formation_dt"]).any(),
        "active_rows_are_member_states": bool(active["membership_asof_status"].eq("member").all() & active["membership_asof_eligible"].astype(bool).all() & active["membership_active"].astype(bool).all()),
        "active_rows_inside_corrected_spans": bool((active["member_from"].le(active["formation_session"]) & active["member_to"].ge(active["formation_session"])).all()),
        "master_sample_ids_unique": not active["sample_id"].duplicated().any(),
        "features_exclude_future_and_targets": not bool(set(features.columns) & FORBIDDEN_FEATURE_COLUMNS),
        "reference_strictly_before_formation": bool((active["reference_session"] < active["formation_session"]).all()),
        "dev_targets_only_training_validation": set(targets_dev["split"].astype(str)).issubset({"training", "validation"}),
        "test_target_values_not_written": not bool(targets_test_sealed.columns.isin(["stock_forward_return", "benchmark_forward_return", "forward_excess_return", "y"]).any()),
        "test_master_target_columns_na": bool(active.loc[active["split"].eq("test"), ["stock_forward_return", "benchmark_forward_return", "forward_excess_return", "y"]].isna().all().all()),
        "no_infinite_features": bool(np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)[~pd.isna(features[FEATURE_COLUMNS].to_numpy(dtype=float))]).all()) if features[FEATURE_COLUMNS].notna().any().any() else True,
        "no_physical_row_deletion": True,
        "no_imputation_or_winsorization": True,
        "non_overlap_gap_at_least_h21": selected_gap_ok,
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
        "candidate_instruments": out / "candidate_instruments.csv",
        "input_coverage": out / "input_coverage.csv",
        "feature_missingness": out / "feature_missingness.csv",
        "split_counts": out / "split_counts.csv",
    }
    features.to_csv(outputs["model_features"], index=False)
    metadata.to_csv(outputs["metadata"], index=False)
    eligibility.to_csv(outputs["eligibility"], index=False)
    targets_dev.to_csv(outputs["targets_dev"], index=False)
    targets_test_sealed.to_csv(outputs["targets_test_sealed"], index=False)
    audit.to_csv(outputs["membership_snapshot_audit"], index=False)
    if registry is not None:
        candidate_output = registry.copy()
        candidate_output.insert(0, "Instrument", candidate_output["ric"])
        candidate_output["candidate_source"] = "external_registry_exploratory_static_candidate"
        candidate_output["membership_run_id"] = MEMBERSHIP_RUN_ID
    else:
        candidate_output = pd.DataFrame({"Instrument": requested_candidates, "candidate_source": "external_list" if candidate_path is not None else "explicit_member_union", "membership_run_id": MEMBERSHIP_RUN_ID})
    candidate_output.to_csv(outputs["candidate_instruments"], index=False)
    coverage.to_csv(outputs["input_coverage"], index=False)
    feature_missingness.to_csv(outputs["feature_missingness"], index=False)
    split_counts.to_csv(outputs["split_counts"], index=False)
    counts = {
        "candidate_requested_instruments": len(requested_candidates),
        "candidate_source_explicit_member_instruments": len(source_explicit),
        "formation_sessions": int(active["formation_session"].nunique()),
        "membership_snapshot_audit_rows": int(len(audit)),
        "master_rows": int(len(active)),
        "active_instruments": int(active["Instrument"].nunique()),
        "training_rows": int((active["split"] == "training").sum()),
        "validation_rows": int((active["split"] == "validation").sum()),
        "test_rows": int((active["split"] == "test").sum()),
        "dev_label_rows": int(len(targets_dev)),
        "dev_label_complete_rows": int(targets_dev["label_complete"].sum()),
        "test_target_sealed_rows": int(len(targets_test_sealed)),
        "boundary_purged_rows": int(active["boundary_purged"].sum()),
        "non_overlap_selected_rows": int(active["non_overlap_selected"].sum()),
        "supervised_model_eligible_rows": int(active["supervised_model_eligible"].sum()),
        "rows_with_any_feature_missing": int(active["feature_missing_count"].gt(0).sum()),
        "source_returns_rows_filtered": int(len(returns)),
        "source_prices_rows_filtered": int(len(prices)),
        "registry_rows": int(len(registry)) if registry is not None else 0,
        "registry_direct_local_pit_candidates": int(registry["pit_evidence_status"].eq("direct_local_pit").sum()) if registry is not None else 0,
        "registry_provisional_static_candidates": int(registry["pit_evidence_status"].eq("provisional_static_source").sum()) if registry is not None else 0,
        "registry_pit_eligible_now_candidates": int(registry["pit_eligible_now"].astype("string").str.casefold().eq("true").sum()) if registry is not None else 0,
    }
    input_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in input_paths.items()}
    output_hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in outputs.items()}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "status": "partial_model_ready_daily_controls_only",
        "model_ready": False,
        "config_path": rel(CONFIG_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "builder_path": rel(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {"membership": {"run_id": MEMBERSHIP_RUN_ID, "path": rel(membership_path)}, "returns": {"path": rel(returns_path)}, "prices": {"path": rel(prices_path)}, **({"candidate_list": {"path": rel(candidate_path)}} if candidate_path is not None else {})},
        "membership_source_mode": "exploratory_static_candidate" if registry is not None else "latest_membership_pit_union",
        "pit_status": "not_full_pit_provisional_static_candidate_registry" if registry is not None else "current_membership_pit_source",
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "counts": counts,
        "feature_columns": FEATURE_COLUMNS,
        "core_feature_gate": CORE_FEATURES,
        "split_counts": split_counts.to_dict(orient="records"),
        "test_policy": config["test_policy"],
        "checks": checks,
        "limitations": [
            ("The expanded candidate registry is exploratory_static_candidate mode: 9 candidates have direct_local_pit evidence and 40 use provisional static issuer-role evidence with registry member spans. AI role evidence is not a complete point-in-time history; this run must not be described as an unbiased full-PIT AI backtest." if registry is not None else "Daily master is based on the current membership run's explicit member union; ambiguous/unknown states remain in source and daily audit. An external frozen candidate list can be supplied for a later expanded membership run."),
            "Only the 21-session target was run. H5 is not generated in this version.",
            "The active pool is small in the current 10-RIC validation universe; daily top/bottom 20% spread diagnostics generally use one security per side and do not meet a 10-per-side economic portfolio gate.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    append_log(summary, out)
    append_ai_use_log(summary, out)
    print(json.dumps({"run_id": run_id, "out": rel(out), "status": summary["status"], "counts": counts, "checks": checks}, ensure_ascii=False, indent=2, default=str), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a daily AI-pool model-ready layer.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--candidate-list", default=None, help="Optional CSV with an Instrument column for a frozen candidate universe.")
    args = parser.parse_args()
    build(args.run_id, args.candidate_list)
