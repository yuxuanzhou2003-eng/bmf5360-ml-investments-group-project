"""Build a leakage-controlled model-ready layer from common-analyst graph v2.

The builder deliberately starts from the frozen graph-v2 ``reason == ok`` edges and
the event-level panel-v3 input.  It never reads panel-v3 labels, execution inputs, or
diagnostics: entry/exit timing, membership, prices, execution evidence, and labels are
rebuilt here from the fixed returns, prices, and corrected membership inputs.

The test split is hard sealed before target lookup.  Test rows retain timing and
backward-looking features, but their target value columns are always NaN and their
target reason is ``test_target_sealed``.
"""
from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "analyst_graph_model_ready_v1_config.json"
OUT_ROOT = ROOT / "data" / "analyst_graph_model_ready_v1"
EVENT_KEY = ["source", "announcement", "period_end"]
BENCHMARK = "SPY.P"
TARGET_COLUMNS = [
    "receiver_forward_return",
    "benchmark_forward_return",
    "forward_benchmark_excess",
]
EVENT_COLUMNS = [
    "source",
    "announcement",
    "period_end",
    "announcement_day",
    "actual",
    "actual_source",
    "actual_selector",
    "formal_entry_rule",
    "consensus",
    "dispersion",
    "analysts",
    "snapshot",
    "snapshot_age_days",
    "status",
    "eps_difference",
    "standardized_surprise",
]
GRAPH_COLUMNS = [
    "source",
    "announcement",
    "period_end",
    "receiver",
    "snapshot",
    "snapshot_age_days",
    "common_brokers",
    "jaccard_coverage",
    "jaccard_rec_weighted",
    "jaccard_named_only",
    "common_rated_brokers",
    "reason",
]
PRICE_COLUMNS = [
    "Instrument",
    "Date",
    "TRDPRC_1",
    "ACVOL_UNS",
    "quoted_spread_bps",
    "dollar_volume",
    "dollar_volume_source",
    "has_close",
    "has_volume",
    "has_two_sided_quote",
    "in_sp500_that_day",
    "price_adjustments",
]


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
        return str(path.resolve())


def parse_date(values, name: str, allow_missing: bool = False) -> pd.Series:
    result = pd.to_datetime(values, format="mixed", errors="coerce").dt.normalize()
    if not allow_missing and result.isna().any():
        raise ValueError(f"{name} contains missing or invalid dates")
    return result


def parse_bool(values, name: str) -> pd.Series:
    normalized = pd.Series(values).astype("string").str.strip().str.lower()
    invalid = sorted(set(normalized.dropna()) - {"true", "false"})
    if invalid or normalized.isna().any():
        raise ValueError(f"{name} contains invalid booleans: {invalid[:10]}")
    return normalized.eq("true")


def matrix_lookup(matrix: pd.DataFrame, instruments: pd.Series, dates: pd.Series) -> np.ndarray:
    """Exact instrument/session lookup; missing keys remain NaN and never fill."""
    row_i = matrix.index.get_indexer(pd.DatetimeIndex(dates))
    col_i = matrix.columns.get_indexer(pd.Index(instruments))
    output = np.full(len(instruments), np.nan, dtype=float)
    valid = (row_i >= 0) & (col_i >= 0)
    if valid.any():
        output[valid] = matrix.to_numpy()[row_i[valid], col_i[valid]]
    return output


def rolling_beta_idio(wide: pd.DataFrame, benchmark: pd.Series, window: int, min_obs: int):
    """Pairwise rolling OLS beta and annualized idiosyncratic volatility."""
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
        valid_fit = (count >= min_obs) & np.isfinite(market_ss) & market_ss.gt(0)
        beta = (covariance_numerator / market_ss).where(valid_fit)
        residual_ss = (asset_ss - covariance_numerator.pow(2) / market_ss).clip(lower=0)
        residual_variance = residual_ss / (count - 2).where(count.gt(2))
    idio_vol = (np.sqrt(residual_variance) * np.sqrt(252.0)).where(valid_fit & count.gt(2))
    return beta, idio_vol, count.where(beta.notna())


def split_for_day(day: pd.Timestamp) -> str:
    if pd.Timestamp("2015-01-01") <= day <= pd.Timestamp("2020-12-31"):
        return "training"
    if pd.Timestamp("2021-01-01") <= day <= pd.Timestamp("2022-12-31"):
        return "validation"
    if pd.Timestamp("2023-01-01") <= day <= pd.Timestamp("2026-06-30"):
        return "test"
    return "out_of_window"


def entry_position(sessions: pd.DatetimeIndex, announcement_day: pd.Timestamp) -> int:
    """First SPY session strictly after the announcement calendar day."""
    return int(sessions.searchsorted(pd.Timestamp(announcement_day).normalize(), side="right"))


def forward_label(
    wide: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    entry_i: int,
    horizon: int,
    receiver: str,
    benchmark: str,
) -> dict:
    """Compute a five-session receiver-minus-benchmark label for train/validation only."""
    record = {
        "entry_session": pd.NaT,
        "exit_session": pd.NaT,
        "label_available": False,
        "reason": "entry_session_unavailable",
        "receiver_forward_return": np.nan,
        "benchmark_forward_return": np.nan,
        "forward_benchmark_excess": np.nan,
    }
    if entry_i >= len(sessions):
        return record
    record["entry_session"] = sessions[entry_i]
    exit_i = entry_i + horizon
    if exit_i >= len(sessions):
        record["reason"] = "label_horizon_incomplete"
        return record
    record["exit_session"] = sessions[exit_i]
    future = wide.iloc[entry_i + 1 : exit_i + 1]
    benchmark_complete = benchmark in future and len(future) == horizon and future[benchmark].notna().all()
    receiver_complete = receiver in future and len(future) == horizon and future[receiver].notna().all()
    if benchmark_complete:
        record["benchmark_forward_return"] = float((1.0 + future[benchmark]).prod() - 1.0)
    if receiver_complete:
        record["receiver_forward_return"] = float((1.0 + future[receiver]).prod() - 1.0)
    if benchmark_complete and receiver_complete:
        record["label_available"] = True
        record["reason"] = "ok"
        record["forward_benchmark_excess"] = (
            record["receiver_forward_return"] - record["benchmark_forward_return"]
        )
    else:
        missing = []
        if not receiver_complete:
            missing.append("receiver_return_incomplete")
        if not benchmark_complete:
            missing.append("benchmark_return_incomplete")
        record["reason"] = ";".join(missing)
    return record


def sealed_target_record(sessions: pd.DatetimeIndex, entry_i: int, horizon: int) -> dict:
    """Return timing only; deliberately has no return matrix or target lookup argument."""
    entry = sessions[entry_i] if entry_i < len(sessions) else pd.NaT
    exit_i = entry_i + horizon
    exit_session = sessions[exit_i] if exit_i < len(sessions) else pd.NaT
    return {
        "entry_session": entry,
        "exit_session": exit_session,
        "label_available": False,
        "reason": "test_target_sealed",
        "receiver_forward_return": np.nan,
        "benchmark_forward_return": np.nan,
        "forward_benchmark_excess": np.nan,
    }


def out_of_window_target_record(sessions: pd.DatetimeIndex, entry_i: int, horizon: int) -> dict:
    entry = sessions[entry_i] if entry_i < len(sessions) else pd.NaT
    exit_i = entry_i + horizon
    exit_session = sessions[exit_i] if exit_i < len(sessions) else pd.NaT
    return {
        "entry_session": entry,
        "exit_session": exit_session,
        "label_available": False,
        "reason": "out_of_window",
        "receiver_forward_return": np.nan,
        "benchmark_forward_return": np.nan,
        "forward_benchmark_excess": np.nan,
    }


def boundary_purged_target_record(sessions: pd.DatetimeIndex, entry_i: int, horizon: int) -> dict:
    """Return timing only so a label never reads across a split boundary."""
    entry = sessions[entry_i] if entry_i < len(sessions) else pd.NaT
    exit_i = entry_i + horizon
    exit_session = sessions[exit_i] if exit_i < len(sessions) else pd.NaT
    return {
        "entry_session": entry,
        "exit_session": exit_session,
        "label_available": False,
        "reason": "boundary_purged",
        "receiver_forward_return": np.nan,
        "benchmark_forward_return": np.nan,
        "forward_benchmark_excess": np.nan,
    }


def lookup_price(price_index: pd.DataFrame, instrument: str, session) -> dict:
    """Exact price row lookup; no temporal fill and no replacement of missing fields."""
    empty = {
        "price_row_available": False,
        "close": np.nan,
        "volume": np.nan,
        "quoted_spread_bps": np.nan,
        "dollar_volume": np.nan,
        "dollar_volume_source": None,
        "has_close": False,
        "has_volume": False,
        "has_two_sided_quote": False,
        "in_sp500_that_day": None,
        "price_adjustments": None,
    }


def lookup_price_frame(
    price_index: pd.DataFrame,
    instruments: pd.Series,
    dates: pd.Series,
) -> pd.DataFrame:
    """Vectorized exact key lookup with the same no-fill semantics as lookup_price."""
    keys = pd.MultiIndex.from_arrays(
        [instruments.astype(str).to_numpy(), pd.to_datetime(dates).to_numpy()],
        names=["Instrument", "Date"],
    )
    source = price_index.copy()
    source["_price_row_available"] = True
    result = source.reindex(keys).reset_index(drop=True)
    result["price_row_available"] = result.pop("_price_row_available").fillna(False).astype(bool)
    for column in ["has_close", "has_volume", "has_two_sided_quote"]:
        result[column] = result[column].fillna(False).astype(bool)
    return result
    key = (instrument, pd.Timestamp(session))
    if pd.isna(session) or key not in price_index.index:
        return empty
    # Use a one-key frame so pandas cannot interpret the MultiIndex tuple as a
    # row/column selector when a scalar Series is requested.
    selected = price_index.loc[[key]]
    if len(selected) != 1:
        raise RuntimeError(f"Duplicate price key: {instrument} {session}")
    row = selected.iloc[0]
    return {
        "price_row_available": True,
        "close": row["TRDPRC_1"],
        "volume": row["ACVOL_UNS"],
        "quoted_spread_bps": row["quoted_spread_bps"],
        "dollar_volume": row["dollar_volume"],
        "dollar_volume_source": row["dollar_volume_source"],
        "has_close": bool(row["has_close"]),
        "has_volume": bool(row["has_volume"]),
        "has_two_sided_quote": bool(row["has_two_sided_quote"]),
        "in_sp500_that_day": bool(row["in_sp500_that_day"]),
        "price_adjustments": row["price_adjustments"],
    }


def semicolon_reasons(index: pd.Index, rules: list[tuple[str, pd.Series]]) -> pd.Series:
    output = pd.Series("", index=index, dtype="string")
    for reason, failed in rules:
        failed = failed.fillna(True).astype(bool)
        output.loc[failed] = output.loc[failed].map(
            lambda value: reason if not value else f"{value};{reason}"
        )
    return output


def build_targets(
    base: pd.DataFrame,
    wide: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """Build labels with the test hard-seal branch before any target lookup."""
    records = []
    for row in base.itertuples(index=False):
        if row.boundary_purged:
            # This branch comes before the train/validation label path so no
            # validation or test return is read across a declared boundary.
            record = boundary_purged_target_record(sessions, row.entry_index, horizon)
        elif row.split == "test":
            # This branch intentionally does not call forward_label and does not inspect returns.
            record = sealed_target_record(sessions, row.entry_index, horizon)
        elif row.split in {"training", "validation"}:
            record = forward_label(wide, sessions, row.entry_index, horizon, row.receiver, BENCHMARK)
        else:
            record = out_of_window_target_record(sessions, row.entry_index, horizon)
        record.update({"sample_id": row.sample_id, "split": row.split})
        records.append(record)
    targets = pd.DataFrame(records)
    return targets[
        [
            "sample_id",
            "split",
            "entry_session",
            "exit_session",
            "label_available",
            "reason",
            *TARGET_COLUMNS,
        ]
    ]


def append_processing_log(summary: dict) -> None:
    """Append a run-scoped audit record without replacing earlier processing history."""
    path = ROOT / "DATA_PROCESSING_LOG.md"
    split_lines = "\n".join(
        f"  - {row['split']}: edges={row['edge_rows']}, source_events={row['source_events']}, "
        f"label_available={row['label_available_rows']}, entry_trade_eligible={row['entry_trade_eligible_rows']}, "
        f"core_feature_available={row['core_feature_available_rows']}, "
        f"supervised_model_eligible={row['supervised_model_eligible_rows']}"
        for row in summary["split_counts"]
    )
    input_lines = "\n".join(
        f"  - {name}: {item['path']} sha256={item['sha256']}"
        for name, item in summary["input_hashes"].items()
    )
    output_lines = "\n".join(
        f"  - {name}: {item['path']} sha256={item['sha256']}"
        for name, item in summary["output_hashes"].items()
    )
    limitations = "\n".join(f"  - {value}" for value in summary["limitations"])
    entry = f"""

## {summary['run_timestamp_utc']} — analyst_graph_model_ready_v1 ({summary['run_id']})

- Status: {summary['status']}; this is a run-scoped retrospective record written after execution.
- Purpose: rebuild model-ready common-analyst graph features, execution evidence, eligibility and split-aware labels from the fixed graph-v2 edges and independent event/market inputs.
- Inputs and hashes:
{input_lines}
- Rule and rationale: retain every graph-v2 `reason==ok` edge; merge events with `many_to_one`; regenerate `sample_id`; use the first SPY session strictly after announcement day as entry and the immediately preceding SPY session as the rolling-feature cutoff; compute five-session receiver-minus-SPY labels only for training and validation; hard-seal test targets before lookup.
- Before/after rows and instruments: graph source rows={summary['graph_rows_total']}, selected reason-ok rows={summary['selected_edge_rows']}, output rows={summary['output_rows']}; graph source instruments={summary['graph_instruments']}, output source instruments={summary['output_source_instruments']}, output receiver instruments={summary['output_receiver_instruments']}; rows physically deleted={summary['rows_physically_deleted']}.
- Graph reason counts: {json.dumps(summary['graph_reason_counts'], ensure_ascii=False)}
- Target reason counts (no test target values inspected): {json.dumps(summary['target_reason_counts'], ensure_ascii=False)}
- Entry and supervised reason counts: entry={json.dumps(summary['entry_ineligibility_reason_counts'], ensure_ascii=False)}; supervised={json.dumps(summary['supervised_ineligibility_reason_counts'], ensure_ascii=False)}
- Split counts:
{split_lines}
- Affected rows: all {summary['selected_edge_rows']} selected edges received a regenerated sample id, entry/exit timing and feature/eligibility records; test rows={summary['test_rows']} remain target-sealed.
- Missing-value handling: missing values remain NA; no fill, interpolation, zero fill, winsorization, clipping, or physical row deletion. Missing fields are represented by feature missingness and eligibility reasons.
- Exclusion/quarantine: no quarantine file was created and no source file was modified. The {summary['graph_non_ok_rows']} graph rows with non-ok reasons remain in the frozen source graph and were excluded by the declared reason-ok input filter.
- Checks: {json.dumps(summary['checks'], ensure_ascii=False)}
- Output hashes:
{output_lines}
- Limitations:
{limitations}
"""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(entry)


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    paths = {
        "graph_edges": ROOT / config["inputs"]["graph_edges"],
        "panel_events": ROOT / config["inputs"]["panel_events"],
        "returns": ROOT / config["inputs"]["returns"],
        "prices": ROOT / config["inputs"]["prices"],
        "corrected_membership_spans": ROOT / config["inputs"]["corrected_membership_spans"],
        "config": CONFIG_PATH,
        "builder_code": Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))
    input_hashes = {
        name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items()
    }

    graph_all = pd.read_csv(paths["graph_edges"], usecols=GRAPH_COLUMNS, low_memory=False)
    graph_reason_counts = {str(key): int(value) for key, value in graph_all.reason.value_counts(dropna=False).items()}
    selected = graph_all.loc[graph_all.reason.eq(config["graph_filter"]["reason_required"])].copy()
    expected_edges = int(config["graph_filter"]["expected_selected_edge_rows"])
    if len(selected) != expected_edges:
        raise RuntimeError(f"Expected {expected_edges} reason-ok edges, observed {len(selected)}")
    if selected.duplicated(EVENT_KEY + ["receiver"]).any():
        raise RuntimeError("Selected graph edges contain duplicate event/receiver keys")

    events = pd.read_csv(paths["panel_events"], usecols=EVENT_COLUMNS, low_memory=False)
    events = events.rename(columns={
        "snapshot": "estimate_snapshot",
        "snapshot_age_days": "estimate_snapshot_age_days",
        "status": "estimate_status",
    })
    if events.duplicated(EVENT_KEY).any():
        raise RuntimeError("Panel events are not unique on the event key")
    base = selected.merge(events, on=EVENT_KEY, how="left", validate="many_to_one", indicator=True)
    unmatched = int(base["_merge"].ne("both").sum())
    if unmatched:
        raise RuntimeError(f"Selected graph edges without a panel event match: {unmatched}")
    base = base.drop(columns="_merge")
    if len(base) != expected_edges:
        raise RuntimeError("many_to_one merge changed selected edge row count")

    base["announcement_day"] = parse_date(base["announcement_day"], "announcement_day")
    base["snapshot_dt"] = parse_date(base["snapshot"], "graph snapshot")
    base["period_end_dt"] = parse_date(base["period_end"], "period_end")
    base["graph_snapshot_age_days"] = (
        base["announcement_day"] - base["snapshot_dt"]
    ).dt.days.astype(float)
    observed_age = pd.to_numeric(base["snapshot_age_days"], errors="coerce")
    age_match = np.isclose(
        observed_age.to_numpy(float),
        base["graph_snapshot_age_days"].to_numpy(float),
        equal_nan=True,
    )
    if not bool(age_match.all()):
        raise RuntimeError("Graph snapshot age does not recompute from snapshot and announcement day")
    if not bool(base["snapshot_dt"].lt(base["announcement_day"]).all()):
        raise RuntimeError("Graph snapshot is not strictly before announcement day")

    spans = pd.read_csv(paths["corrected_membership_spans"], low_memory=False)
    required_span_columns = {"ric", "member_from", "member_to"}
    if not required_span_columns.issubset(spans.columns):
        raise RuntimeError("Corrected membership spans are missing required columns")
    spans["member_from"] = parse_date(spans["member_from"], "membership member_from")
    spans["member_to"] = parse_date(spans["member_to"], "membership member_to")
    if (spans["member_from"] > spans["member_to"]).any():
        raise RuntimeError("Membership spans have member_from after member_to")
    if spans["ric"].isna().any():
        raise RuntimeError("Membership spans contain missing RICs")

    # Cache corrected membership sets only for dates used by this edge set.
    unique_days = pd.DatetimeIndex(
        pd.concat([base["announcement_day"], base["announcement_day"]], ignore_index=True).dropna().unique()
    )
    membership_cache = {
        day: set(spans.loc[
            spans["member_from"].le(day) & spans["member_to"].ge(day), "ric"
        ].astype(str))
        for day in unique_days
    }

    returns = pd.read_csv(
        paths["returns"],
        usecols=["Instrument", "Date", "return_decimal", "in_sp500_that_day"],
        low_memory=False,
    )
    returns["Date"] = parse_date(returns["Date"], "returns Date")
    returns["return_decimal"] = pd.to_numeric(returns["return_decimal"], errors="coerce")
    returns["in_sp500_that_day"] = parse_bool(
        returns["in_sp500_that_day"], "returns in_sp500_that_day"
    ).to_numpy()
    if returns.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("Duplicate returns instrument/session keys")
    if ((returns["return_decimal"] <= -1) & returns["return_decimal"].notna()).any():
        raise RuntimeError("Total return <= -100% prevents log compounding")
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    if BENCHMARK not in wide:
        raise RuntimeError("SPY.P is missing from returns input")
    wide = wide.loc[wide[BENCHMARK].notna()]
    sessions = pd.DatetimeIndex(wide.index)
    if not len(sessions):
        raise RuntimeError("No SPY sessions available")
    membership = returns.loc[~returns["Instrument"].eq(BENCHMARK)].pivot(
        index="Date", columns="Instrument", values="in_sp500_that_day"
    )
    membership = (
        membership.reindex(index=sessions, columns=wide.columns)
        .fillna(False)
        .astype(bool)
    )

    # Rebuild timing and event-level split before labels.  No panel-v3 timing fields are used.
    base["entry_index"] = [entry_position(sessions, day) for day in base["announcement_day"]]
    base["entry_session"] = [
        sessions[index] if index < len(sessions) else pd.NaT for index in base["entry_index"]
    ]
    # Announcement dates were cached before entry sessions were known. Extend the
    # same corrected-span cache so entry membership is evaluated on the actual
    # entry date rather than falling through to an empty set.
    for day in pd.DatetimeIndex(base["entry_session"].dropna().unique()):
        if day not in membership_cache:
            membership_cache[day] = set(spans.loc[
                spans["member_from"].le(day) & spans["member_to"].ge(day), "ric"
            ].astype(str))
    base["liquidity_reference_session"] = [
        sessions[index - 1] if 0 < index < len(sessions) else pd.NaT
        for index in base["entry_index"]
    ]
    horizon = int(config["forward_label"]["horizon_sessions"])
    base["exit_session"] = [
        sessions[index + horizon] if index + horizon < len(sessions) else pd.NaT
        for index in base["entry_index"]
    ]
    base["split"] = base["announcement_day"].map(split_for_day)
    next_boundary = base["split"].map({
        "training": pd.Timestamp("2021-01-01"),
        "validation": pd.Timestamp("2023-01-01"),
    })
    base["next_split_boundary"] = next_boundary
    base["boundary_purged"] = next_boundary.notna() & (
        base["entry_session"].ge(next_boundary) | base["exit_session"].ge(next_boundary)
    )

    if not bool(
        base.loc[base["entry_session"].notna(), "entry_session"].gt(
            base.loc[base["entry_session"].notna(), "announcement_day"]
        ).all()
    ):
        raise RuntimeError("Entry session is not strictly after announcement day")
    if not bool(
        base.loc[base["entry_session"].notna(), "liquidity_reference_session"].lt(
            base.loc[base["entry_session"].notna(), "entry_session"]
        ).all()
    ):
        raise RuntimeError("Liquidity reference session is not strictly before entry")

    # The event group must have one split and one purge decision across all receivers.
    event_group = base.groupby(EVENT_KEY, sort=False, dropna=False)
    if event_group["split"].nunique().gt(1).any() or event_group["boundary_purged"].nunique().gt(1).any():
        raise RuntimeError("An event crosses split or purge assignments")

    base["source_in_index_on_announcement"] = [
        str(source) in membership_cache[day]
        for source, day in zip(base["source"], base["announcement_day"])
    ]
    base["receiver_in_index_on_announcement"] = [
        str(receiver) in membership_cache[day]
        for receiver, day in zip(base["receiver"], base["announcement_day"])
    ]
    base["receiver_in_index_at_entry"] = [
        bool(pd.notna(entry) and str(receiver) in membership_cache.get(entry, set()))
        for receiver, entry in zip(base["receiver"], base["entry_session"])
    ]

    # Regenerate sample IDs from the selected edge keys, using the original key strings.
    base["sample_id"] = [
        hashlib.sha256(f"{source}|{announcement}|{period_end}|{receiver}".encode("utf-8")).hexdigest()[:24]
        for source, announcement, period_end, receiver in zip(
            base["source"], base["announcement"], base["period_end"], base["receiver"]
        )
    ]
    if base["sample_id"].duplicated().any() or base["sample_id"].isna().any():
        raise RuntimeError("Regenerated sample IDs are not unique and nonmissing")

    # Hard seal is executed here, before any target lookup.  Test rows never enter forward_label.
    print(f"selected edges {len(base)}; rebuilding train/validation labels with test target seal", flush=True)
    targets = build_targets(base, wide, sessions, horizon)
    if len(targets) != len(base) or set(targets["sample_id"]) != set(base["sample_id"]):
        raise RuntimeError("Target rows or sample IDs do not match selected graph edges")
    test_mask = targets["split"].eq("test")
    if not bool(targets.loc[test_mask, TARGET_COLUMNS].isna().all().all()):
        raise RuntimeError("Test target values are not all NaN after hard seal")
    if not bool(targets.loc[test_mask, "reason"].eq("test_target_sealed").all()):
        raise RuntimeError("Test target rows do not all carry test_target_sealed")

    # Build graph and event features first.  jaccard_named_only stays metadata-only.
    feature = pd.DataFrame({"sample_id": base["sample_id"]})
    surprise = pd.to_numeric(base["standardized_surprise"], errors="coerce")
    feature["standardized_surprise"] = surprise
    feature["surprise_abs"] = surprise.abs()
    feature["surprise_sign"] = np.sign(surprise)
    feature["jaccard_coverage"] = pd.to_numeric(base["jaccard_coverage"], errors="coerce")
    feature["jaccard_rec_weighted"] = pd.to_numeric(base["jaccard_rec_weighted"], errors="coerce")
    feature["surprise_x_jaccard_coverage"] = surprise * feature["jaccard_coverage"]
    feature["surprise_x_jaccard_rec_weighted"] = surprise * feature["jaccard_rec_weighted"]
    common = pd.to_numeric(base["common_brokers"], errors="coerce")
    rated_common = pd.to_numeric(base["common_rated_brokers"], errors="coerce")
    feature["common_brokers"] = common
    feature["common_brokers_log1p"] = np.log1p(common.where(common.ge(0)))
    feature["common_rated_brokers"] = rated_common
    feature["common_rated_brokers_log1p"] = np.log1p(rated_common.where(rated_common.ge(0)))
    feature["graph_snapshot_age_days"] = base["graph_snapshot_age_days"]
    event_count = event_group["sample_id"].transform("size").astype(float)
    feature["event_neighbor_count"] = event_count

    # Deterministic ranks: stronger coverage first, receiver name resolves ties.
    rank_order = base.reset_index(names="_base_index").sort_values(
        EVENT_KEY + ["jaccard_coverage", "receiver"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    rank_order["neighbor_rank_jaccard_coverage"] = rank_order.groupby(
        EVENT_KEY, sort=False
    ).cumcount() + 1
    rank_order = rank_order.set_index("_base_index")
    feature["neighbor_rank_jaccard_coverage"] = rank_order.loc[
        base.index, "neighbor_rank_jaccard_coverage"
    ].to_numpy(float)
    rank_order_rec = base.reset_index(names="_base_index").sort_values(
        EVENT_KEY + ["jaccard_rec_weighted", "receiver"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    rank_order_rec["neighbor_rank_jaccard_rec_weighted"] = rank_order_rec.groupby(
        EVENT_KEY, sort=False
    ).cumcount() + 1
    rank_order_rec = rank_order_rec.set_index("_base_index")
    feature["neighbor_rank_jaccard_rec_weighted"] = rank_order_rec.loc[
        base.index, "neighbor_rank_jaccard_rec_weighted"
    ].to_numpy(float)
    analysts = pd.to_numeric(base["analysts"], errors="coerce")
    feature["analysts_log1p"] = np.log1p(analysts.where(analysts.ge(0)))

    # Return features use the old builder's windows and exact point-in-time matrix lookup.
    return_cfg = config["return_features"]
    log_returns = np.log1p(wide)
    return_matrices = {"ret_1": wide}
    for window in return_cfg["momentum_windows"]:
        return_matrices[f"mom_{window}"] = np.expm1(
            log_returns.rolling(int(window), min_periods=int(window)).sum()
        )
    for window, min_obs in return_cfg["volatility_windows_and_min_observations"]:
        return_matrices[f"vol_{window}_ann"] = (
            wide.rolling(int(window), min_periods=int(min_obs)).std(ddof=1)
            * np.sqrt(float(return_cfg["annualization_sessions"]))
        )
    beta, idio, beta_count = rolling_beta_idio(
        wide,
        wide[BENCHMARK],
        int(return_cfg["beta_idio_window"]),
        int(return_cfg["beta_idio_min_observations"]),
    )
    return_matrices["beta_126"] = beta
    return_matrices["idio_vol_126_ann"] = idio
    return_matrices["beta_obs_126"] = beta_count
    reference = base["liquidity_reference_session"]
    for prefix, instruments in [("source", base["source"]), ("receiver", base["receiver"])]:
        for name, matrix in return_matrices.items():
            feature[f"{prefix}_{name}"] = matrix_lookup(matrix, instruments, reference)
    receiver_mom_rank = return_matrices["mom_20"].where(membership).rank(axis=1, pct=True)
    receiver_vol_rank = return_matrices["vol_20_ann"].where(membership).rank(axis=1, pct=True)
    feature["receiver_mom_20_pct_rank"] = matrix_lookup(
        receiver_mom_rank, base["receiver"], reference
    )
    feature["receiver_vol_20_pct_rank"] = matrix_lookup(
        receiver_vol_rank, base["receiver"], reference
    )
    ref_i = sessions.get_indexer(pd.DatetimeIndex(reference))
    market_feature_map = {
        "market_mom_20": return_matrices["mom_20"][BENCHMARK],
        "market_mom_60": return_matrices["mom_60"][BENCHMARK],
        "market_vol_20_ann": return_matrices["vol_20_ann"][BENCHMARK],
        "market_vol_60_ann": return_matrices["vol_60_ann"][BENCHMARK],
    }
    for name, series in market_feature_map.items():
        values = np.full(len(base), np.nan)
        valid = ref_i >= 0
        values[valid] = series.to_numpy()[ref_i[valid]]
        feature[name] = values
    del log_returns, beta, idio, beta_count, receiver_mom_rank, receiver_vol_rank
    gc.collect()

    # Prices are read independently of panel-v3 execution.  Exact rows support both
    # point-in-time rolling liquidity and rebuilt execution evidence.
    needed = sorted(set(base["source"].astype(str)) | set(base["receiver"].astype(str)))
    prices = pd.read_csv(paths["prices"], usecols=PRICE_COLUMNS, low_memory=False)
    prices["Instrument"] = prices["Instrument"].astype(str)
    prices["Date"] = parse_date(prices["Date"], "price Date")
    for column in ["TRDPRC_1", "ACVOL_UNS", "quoted_spread_bps", "dollar_volume"]:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    for column in ["has_close", "has_volume", "has_two_sided_quote", "in_sp500_that_day"]:
        prices[column] = parse_bool(prices[column], f"prices {column}").to_numpy()
    if prices.duplicated(["Instrument", "Date"]).any():
        raise RuntimeError("Duplicate clean price instrument/session keys")
    if ((prices["ACVOL_UNS"] < 0) & prices["ACVOL_UNS"].notna()).any() or (
        (prices["dollar_volume"] < 0) & prices["dollar_volume"].notna()
    ).any():
        raise RuntimeError("Negative volume or dollar volume")
    prices = prices.loc[prices["Instrument"].isin(needed)].copy()
    price_index = prices.set_index(["Instrument", "Date"], verify_integrity=True).sort_index()
    price_membership = prices.pivot(
        index="Date", columns="Instrument", values="in_sp500_that_day"
    ).reindex(index=sessions, columns=needed).fillna(False).astype(bool)
    volume = prices.pivot(index="Date", columns="Instrument", values="ACVOL_UNS").reindex(
        index=sessions, columns=needed
    )
    dollar_volume = prices.pivot(
        index="Date", columns="Instrument", values="dollar_volume"
    ).reindex(index=sessions, columns=needed)
    spread = prices.pivot(
        index="Date", columns="Instrument", values="quoted_spread_bps"
    ).reindex(index=sessions, columns=needed)
    liq_cfg = config["liquidity_features"]
    liq_window = int(liq_cfg["window"])
    volume_min = int(liq_cfg["volume_and_dollar_volume_min_observations"])
    spread_min = int(liq_cfg["spread_min_observations"])
    volume_med = volume.rolling(liq_window, min_periods=volume_min).median()
    dollar_volume_med = dollar_volume.rolling(liq_window, min_periods=volume_min).median()
    spread_med = spread.rolling(liq_window, min_periods=spread_min).median()
    liquidity_matrices = {
        "volume_median_20_log1p": np.log1p(volume_med),
        "dollar_volume_median_20_log1p": np.log1p(dollar_volume_med),
        "spread_median_20_bps": spread_med,
    }
    for prefix, instruments in [("source", base["source"]), ("receiver", base["receiver"])]:
        for name, matrix in liquidity_matrices.items():
            feature[f"{prefix}_{name}"] = matrix_lookup(matrix, instruments, reference)
    receiver_dvol_rank = dollar_volume_med.where(price_membership).rank(axis=1, pct=True)
    receiver_spread_rank = spread_med.where(price_membership).rank(axis=1, pct=True)
    feature["receiver_dollar_volume_20_pct_rank"] = matrix_lookup(
        receiver_dvol_rank, base["receiver"], reference
    )
    feature["receiver_spread_20_pct_rank"] = matrix_lookup(
        receiver_spread_rank, base["receiver"], reference
    )

    source_reference = lookup_price_frame(
        price_index, base["source"], base["liquidity_reference_session"]
    )
    receiver_reference = lookup_price_frame(
        price_index, base["receiver"], base["liquidity_reference_session"]
    )
    receiver_entry = lookup_price_frame(price_index, base["receiver"], base["entry_session"])
    execution = pd.DataFrame({
        "sample_id": base["sample_id"],
        "source": base["source"],
        "receiver": base["receiver"],
        "entry_session": base["entry_session"],
        "liquidity_reference_session": base["liquidity_reference_session"],
        "source_in_index_on_announcement": base["source_in_index_on_announcement"],
        "receiver_in_index_on_announcement": base["receiver_in_index_on_announcement"],
        "receiver_in_index_at_entry": base["receiver_in_index_at_entry"],
        "reference_source_price_row_available": source_reference["price_row_available"],
        "reference_source_close": source_reference["TRDPRC_1"],
        "reference_source_volume": source_reference["ACVOL_UNS"],
        "reference_source_dollar_volume": source_reference["dollar_volume"],
        "reference_source_quoted_spread_bps": source_reference["quoted_spread_bps"],
        "reference_receiver_price_row_available": receiver_reference["price_row_available"],
        "reference_receiver_close": receiver_reference["TRDPRC_1"],
        "reference_receiver_volume": receiver_reference["ACVOL_UNS"],
        "reference_receiver_dollar_volume": receiver_reference["dollar_volume"],
        "reference_receiver_quoted_spread_bps": receiver_reference["quoted_spread_bps"],
        "entry_price_row_available": receiver_entry["price_row_available"],
        "entry_close": receiver_entry["TRDPRC_1"],
        "entry_volume": receiver_entry["ACVOL_UNS"],
        "entry_dollar_volume": receiver_entry["dollar_volume"],
        "entry_quoted_spread_bps": receiver_entry["quoted_spread_bps"],
        "entry_dollar_volume_source": receiver_entry["dollar_volume_source"],
        "entry_has_close": receiver_entry["has_close"],
        "entry_has_volume": receiver_entry["has_volume"],
        "entry_has_two_sided_quote": receiver_entry["has_two_sided_quote"],
        "entry_in_sp500_that_day_from_price": receiver_entry["in_sp500_that_day"],
        "entry_price_adjustments": receiver_entry["price_adjustments"],
    })
    entry_price_row_available = receiver_entry["price_row_available"].set_axis(base.index)
    entry_has_close = receiver_entry["has_close"].set_axis(base.index)
    entry_has_volume = receiver_entry["has_volume"].set_axis(base.index)
    entry_has_quote = receiver_entry["has_two_sided_quote"].set_axis(base.index)
    entry_has_dollar_volume = receiver_entry["dollar_volume"].notna().set_axis(base.index)
    entry_trade_eligible = (
        base["receiver_in_index_at_entry"]
        & entry_price_row_available
        & entry_has_close
    )

    # Direct reference rows are included as point-in-time liquidity evidence and are
    # also used for explicit lag-one features; no panel-v3 lagged fields are imported.
    for prefix, rows in [("source", source_reference), ("receiver", receiver_reference)]:
        direct_dollar = rows["dollar_volume"].set_axis(base.index)
        direct_spread = rows["quoted_spread_bps"].set_axis(base.index)
        feature[f"{prefix}_dollar_volume_lag1_log1p"] = np.log1p(
            direct_dollar.where(direct_dollar.ge(0))
        )
        feature[f"{prefix}_spread_lag1_bps"] = direct_spread
    del prices, price_membership, volume, dollar_volume, spread
    del volume_med, dollar_volume_med, spread_med, receiver_dvol_rank, receiver_spread_rank
    gc.collect()

    feature_columns = [column for column in feature.columns if column != "sample_id"]
    forbidden_tokens = [str(token).lower() for token in config["forbidden_model_feature_tokens"]]
    forbidden_feature_columns = [
        column for column in feature_columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden_feature_columns:
        raise RuntimeError(f"Forbidden model feature columns: {forbidden_feature_columns}")
    core_features = list(config["core_features"])
    missing_core_config = sorted(set(core_features) - set(feature_columns))
    if missing_core_config:
        raise RuntimeError(f"Configured core features are absent: {missing_core_config}")
    feature_core_available = feature[core_features].notna().all(axis=1)
    feature_missing_count = feature[feature_columns].isna().sum(axis=1).astype(int)
    core_missing_list = feature[core_features].isna().apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1
    )

    label_available = targets["label_available"].astype(bool)
    split_retained = base["split"].isin(["training", "validation", "test"]) & ~base["boundary_purged"]
    supervised_model_eligible = (
        base["split"].isin(["training", "validation"])
        & split_retained
        & label_available
        & entry_trade_eligible
        & feature_core_available
    )
    entry_reason = semicolon_reasons(base.index, [
        ("source_not_index_member_on_announcement", ~base["source_in_index_on_announcement"]),
        ("receiver_not_index_member_on_announcement", ~base["receiver_in_index_on_announcement"]),
        ("receiver_not_index_member_at_entry", ~base["receiver_in_index_at_entry"]),
        ("entry_price_row_missing", ~entry_price_row_available),
        ("entry_close_missing", ~entry_has_close),
    ])
    supervised_reason = semicolon_reasons(base.index, [
        ("out_of_window", ~base["split"].isin(["training", "validation", "test"])),
        ("test_target_sealed", base["split"].eq("test")),
        ("boundary_purged", base["boundary_purged"]),
        ("label_unavailable", ~label_available),
        ("entry_trade_ineligible", ~entry_trade_eligible),
        ("core_feature_missing", ~feature_core_available),
    ])

    metadata = base[[
        "sample_id", "source", "receiver", "announcement", "announcement_day", "period_end",
        "actual", "actual_source", "actual_selector", "formal_entry_rule", "consensus",
        "dispersion", "analysts", "estimate_snapshot", "estimate_snapshot_age_days",
        "estimate_status", "eps_difference", "standardized_surprise",
        "graph_snapshot_age_days", "jaccard_coverage", "jaccard_rec_weighted", "jaccard_named_only",
        "common_brokers", "common_rated_brokers", "source_in_index_on_announcement",
        "receiver_in_index_on_announcement", "receiver_in_index_at_entry", "entry_session",
        "liquidity_reference_session", "exit_session", "split", "boundary_purged",
    ]].copy()
    metadata["graph_snapshot"] = base["snapshot"]
    metadata["neighbor_rank_jaccard_coverage"] = feature["neighbor_rank_jaccard_coverage"]
    metadata["neighbor_rank_jaccard_rec_weighted"] = feature["neighbor_rank_jaccard_rec_weighted"]
    metadata["event_neighbor_count"] = feature["event_neighbor_count"]

    eligibility = pd.DataFrame({
        "sample_id": base["sample_id"],
        "split": base["split"],
        "next_split_boundary": next_boundary,
        "boundary_purged": base["boundary_purged"],
        "split_retained": split_retained,
        "source_in_index_on_announcement": base["source_in_index_on_announcement"],
        "receiver_in_index_on_announcement": base["receiver_in_index_on_announcement"],
        "receiver_in_index_at_entry": base["receiver_in_index_at_entry"],
        "entry_price_row_available": entry_price_row_available,
        "entry_has_close": entry_has_close,
        "entry_has_volume": entry_has_volume,
        "entry_has_two_sided_quote": entry_has_quote,
        "entry_has_dollar_volume": entry_has_dollar_volume,
        "entry_trade_eligible": entry_trade_eligible,
        "entry_ineligibility_reason": entry_reason,
        "label_available": label_available,
        "label_reason": targets["reason"],
        "core_feature_available": feature_core_available,
        "feature_missing_count": feature_missing_count,
        "core_feature_missing_list": core_missing_list,
        "supervised_model_eligible": supervised_model_eligible,
        "supervised_ineligibility_reason": supervised_reason,
    })

    missing_rows = []
    for split_name in ["training", "validation", "test", "out_of_window", "all"]:
        mask = pd.Series(True, index=feature.index) if split_name == "all" else base["split"].eq(split_name)
        denominator = int(mask.sum())
        for column in feature_columns:
            count = int(feature.loc[mask, column].isna().sum())
            missing_rows.append({
                "split": split_name,
                "feature": column,
                "rows": denominator,
                "missing": count,
                "missing_fraction": count / denominator if denominator else np.nan,
            })
    feature_missingness = pd.DataFrame(missing_rows)

    split_rows = []
    for split_name in ["training", "validation", "test", "out_of_window", "all"]:
        mask = pd.Series(True, index=base.index) if split_name == "all" else base["split"].eq(split_name)
        split_rows.append({
            "split": split_name,
            "edge_rows": int(mask.sum()),
            "source_events": int(base.loc[mask, EVENT_KEY].drop_duplicates().shape[0]),
            "boundary_purged_rows": int((mask & base["boundary_purged"]).sum()),
            "label_available_rows": int((mask & label_available).sum()),
            "entry_trade_eligible_rows": int((mask & entry_trade_eligible).sum()),
            "core_feature_available_rows": int((mask & feature_core_available).sum()),
            "supervised_model_eligible_rows": int((mask & supervised_model_eligible).sum()),
        })
    split_counts = pd.DataFrame(split_rows)

    checks = {
        "selected_reason_ok_edge_count": len(selected) == expected_edges,
        "all_selected_edges_retained": len(feature) == expected_edges,
        "edge_event_many_to_one_join": unmatched == 0,
        "sample_ids_unique": not base["sample_id"].duplicated().any(),
        "sample_id_sets_equal": set(feature["sample_id"]) == set(targets["sample_id"]) == set(eligibility["sample_id"]),
        "graph_snapshot_strictly_before_announcement": bool(base["snapshot_dt"].lt(base["announcement_day"]).all()),
        "entry_strictly_after_announcement": bool(
            base.loc[base["entry_session"].notna(), "entry_session"].gt(
                base.loc[base["entry_session"].notna(), "announcement_day"]
            ).all()
        ),
        "rolling_reference_is_previous_spy_session": bool(
            all(
                (index == 0 and pd.isna(reference_day))
                or (index > 0 and reference_day == sessions[index - 1])
                for index, reference_day in zip(base["entry_index"], base["liquidity_reference_session"])
            )
        ),
        "no_self_edges": bool((base["source"] != base["receiver"]).all()),
        "returns_keys_unique": not returns.duplicated(["Instrument", "Date"]).any(),
        "price_keys_unique": not price_index.index.duplicated().any(),
        "test_target_values_all_nan": bool(targets.loc[test_mask, TARGET_COLUMNS].isna().all().all()),
        "test_reason_sealed": bool(targets.loc[test_mask, "reason"].eq("test_target_sealed").all()),
        "boundary_target_values_all_nan": bool(
            targets.loc[base["boundary_purged"].to_numpy(), TARGET_COLUMNS].isna().all().all()
        ),
        "boundary_reason_purged": bool(
            targets.loc[base["boundary_purged"].to_numpy(), "reason"].eq("boundary_purged").all()
        ),
        "test_never_supervised": not bool(
            eligibility.loc[eligibility["split"].eq("test"), "supervised_model_eligible"].any()
        ),
        "named_only_not_model_feature": "jaccard_named_only" not in feature_columns,
        "forbidden_feature_tokens_absent": not forbidden_feature_columns,
        "features_have_no_infinity": bool(
            (np.isfinite(feature[feature_columns].to_numpy(float)) | np.isnan(feature[feature_columns].to_numpy(float))).all()
        ),
        "eligibility_does_not_delete_rows": len(eligibility) == expected_edges,
        "event_split_single_valued": bool(event_group["split"].nunique().le(1).all()),
        "event_purge_single_valued": bool(event_group["boundary_purged"].nunique().le(1).all()),
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    outputs = {
        "metadata": out / "metadata.csv",
        "model_features": out / "model_features.csv",
        "targets": out / "targets.csv",
        "eligibility": out / "eligibility.csv",
        "feature_missingness": out / "feature_missingness.csv",
        "split_counts": out / "split_counts.csv",
        "execution_inputs": out / "execution_inputs.csv",
    }
    metadata.to_csv(outputs["metadata"], index=False)
    feature.to_csv(outputs["model_features"], index=False)
    targets.to_csv(outputs["targets"], index=False)
    eligibility.to_csv(outputs["eligibility"], index=False)
    feature_missingness.to_csv(outputs["feature_missingness"], index=False)
    split_counts.to_csv(outputs["split_counts"], index=False)
    execution.to_csv(outputs["execution_inputs"], index=False)

    target_reason_counts = {
        str(key): int(value) for key, value in targets["reason"].value_counts(dropna=False).items()
    }
    entry_reason_counts = {
        str(key): int(value)
        for key, value in entry_reason.loc[entry_reason.ne("")].value_counts(dropna=False).items()
    }
    supervised_reason_counts = {
        str(key): int(value)
        for key, value in supervised_reason.loc[supervised_reason.ne("")].value_counts(dropna=False).items()
    }
    summary = {
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "output_dir": rel(out),
        "graph_rows_total": int(len(graph_all)),
        "graph_non_ok_rows": int(len(graph_all) - len(selected)),
        "selected_edge_rows": int(len(selected)),
        "output_rows": int(len(feature)),
        "graph_instruments": int(pd.unique(pd.concat([graph_all["source"], graph_all["receiver"]])).size),
        "output_source_instruments": int(base["source"].nunique()),
        "output_receiver_instruments": int(base["receiver"].nunique()),
        "rows_physically_deleted": 0,
        "test_rows": int(test_mask.sum()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "core_features": core_features,
        "diagnostic_metadata_only": ["jaccard_named_only"],
        "graph_reason_counts": graph_reason_counts,
        "target_reason_counts": target_reason_counts,
        "entry_ineligibility_reason_counts": entry_reason_counts,
        "supervised_ineligibility_reason_counts": supervised_reason_counts,
        "split_counts": split_counts.to_dict(orient="records"),
        "entry_trade_ineligible_rows": int((~entry_trade_eligible).sum()),
        "label_unavailable_rows": int((~label_available).sum()),
        "rows_with_any_feature_missing": int(feature_missing_count.gt(0).sum()),
        "preprocessing_fitted": False,
        "missing_value_policy": config["missing_policy"],
        "config": config,
        "input_hashes": input_hashes,
        "code_sha256": input_hashes["builder_code"]["sha256"],
        "config_sha256": input_hashes["config"]["sha256"],
        "checks": checks,
        "limitations": [
            "Test target values are hard-sealed before lookup; no test target, prediction, metric or target aggregate is produced.",
            "Training and validation labels are five-session cumulative total-return excess labels; transaction costs are not applied.",
            "Eligibility flags retain missing observations for downstream training-only preprocessing decisions.",
            "Daily quoted spread is an end-of-day spread proxy rather than a guaranteed executable quote.",
            "The graph's jaccard_named_only is diagnostic metadata and is intentionally excluded from model_features and core eligibility.",
            "No model is fitted and no portfolio or performance claim is produced.",
        ],
        "output_hashes": {},
    }
    summary["output_hashes"] = {
        name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in outputs.items()
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    append_processing_log(summary)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "formal_run_path": rel(out),
                "rows": len(feature),
                "feature_count": len(feature_columns),
                "test_rows": int(test_mask.sum()),
                "target_reason_counts": target_reason_counts,
                "split_counts": summary["split_counts"],
                "checks": checks,
                "output_hashes": summary["output_hashes"],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    if not all(checks.values()):
        raise RuntimeError(
            "Analyst graph model-ready checks failed: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
