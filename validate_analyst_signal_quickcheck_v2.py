"""Independent metric and target validation for analyst signal quickcheck v2."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "data/analysis/analyst_signal_quickcheck_v2/20260909T114601795253Z"
GRAPH = ROOT / "data/analyst_graph_v2/20260909T113109814872Z"
PANEL = ROOT / "data/panel_v3/20260909T035245922341Z"
RETURNS = ROOT / "data/clean/v2/returns.csv"
TRBC = ROOT / "data/raw/trbc_diagnostic_v1/trbc_current.csv"
OUT_ROOT = ROOT / "data/audit/analyst_signal_quickcheck_v2"
KEY = ["source", "announcement", "period_end"]
SPLITS = {"training": (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01")),
          "validation": (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"), pd.Timestamp("2023-01-01"))}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_metrics(frame, signal):
    usable = frame.loc[frame[signal].notna() & frame.target.notna()]
    pooled = spearmanr(usable[signal], usable.target).statistic
    values = []
    for _, group in usable.groupby(KEY):
        if len(group) >= 3 and group[signal].nunique() > 1 and group.target.nunique() > 1:
            value = spearmanr(group[signal], group.target).statistic
            if np.isfinite(value):
                values.append(value)
    return len(usable), usable[KEY].drop_duplicates().shape[0], pooled, np.mean(values), len(values)


def main():
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    observed = pd.read_csv(RUN / "quickcheck_metrics.csv")
    events = pd.read_csv(PANEL / "events.csv", low_memory=False)
    events["announcement_day"] = pd.to_datetime(events.announcement_day).dt.normalize()
    events["entry_session"] = pd.to_datetime(events.entry_session, errors="coerce").dt.normalize()
    development = events.loc[events.announcement_day.lt(pd.Timestamp("2023-01-01")),
                             KEY + ["announcement_day", "entry_session", "standardized_surprise"]]
    returns = pd.read_csv(RETURNS, usecols=["Instrument", "Date", "return_decimal"])
    returns["Date"] = pd.to_datetime(returns.Date).dt.normalize()
    returns = returns.loc[returns.Date.lt(pd.Timestamp("2023-01-01"))]
    wide = returns.pivot(index="Date", columns="Instrument", values="return_decimal").sort_index()
    sessions, columns = wide.index, wide.columns
    values = wide.to_numpy(float)
    forward = np.ones_like(values)
    complete = np.ones_like(values, dtype=bool)
    for step in range(1, 6):
        shifted = np.full_like(values, np.nan)
        shifted[:-step] = values[step:]
        forward *= 1 + np.nan_to_num(shifted, nan=0.0)
        complete &= np.isfinite(shifted)
    forward = np.where(complete, forward - 1, np.nan)
    spy_i = columns.get_loc("SPY.P")

    def attach(edges, strengths):
        frame = edges.merge(development, on=KEY, how="inner", validate="many_to_one")
        frame = frame.loc[frame.entry_session.notna()].copy()
        entry_i = sessions.get_indexer(frame.entry_session)
        frame = frame.loc[entry_i >= 0].copy()
        entry_i = entry_i[entry_i >= 0]
        exit_i = entry_i + 5
        valid_exit = exit_i < len(sessions)
        frame["exit_session"] = pd.NaT
        frame.loc[valid_exit, "exit_session"] = sessions[exit_i[valid_exit]]
        frame["split"] = np.select(
            [frame.announcement_day.between(SPLITS["training"][0], SPLITS["training"][1]),
             frame.announcement_day.between(SPLITS["validation"][0], SPLITS["validation"][1])],
            ["training", "validation"], default="excluded")
        limit = frame.split.map({name: bounds[2] for name, bounds in SPLITS.items()})
        frame["boundary_purged"] = (~valid_exit) | frame.exit_session.ge(limit)
        receiver_i = columns.get_indexer(frame.receiver)
        safe = (receiver_i >= 0) & ~frame.boundary_purged.to_numpy()
        target = np.full(len(frame), np.nan)
        target[safe] = forward[entry_i[safe], receiver_i[safe]] - forward[entry_i[safe], spy_i]
        frame["target"] = target
        for strength in strengths:
            frame[f"signal_{strength}"] = frame.standardized_surprise * frame[strength]
        return frame

    new_edges = pd.read_csv(GRAPH / "edges.csv")
    new_edges = new_edges.loc[new_edges.reason.eq("ok"), KEY + ["receiver", "jaccard_coverage",
        "jaccard_rec_weighted", "jaccard_named_only", "common_brokers"]]
    new_strengths = ["jaccard_coverage", "jaccard_rec_weighted", "jaccard_named_only"]
    new = attach(new_edges, new_strengths)
    old_edges = pd.read_csv(PANEL / "features.csv", low_memory=False,
                            usecols=KEY + ["receiver", "residual_correlation"])
    old = attach(old_edges, ["residual_correlation"])
    sector_data = pd.read_csv(TRBC)
    sector_col = next(column for column in sector_data if "Economic Sector" in column)
    sector = sector_data.set_index("Instrument")[sector_col]
    for frame in [new, old]:
        frame["same_sector"] = frame.source.map(sector).eq(frame.receiver.map(sector))

    rows = []
    specs = [("analyst_graph", new, [f"signal_{x}" for x in new_strengths]),
             ("rejected_residual_graph", old, ["signal_residual_correlation"])]
    for graph_name, frame, signals in specs:
        for stratum, subset in [("all", frame), ("cross_sector", frame.loc[~frame.same_sector]),
                                ("same_sector", frame.loc[frame.same_sector])]:
            for split in ["training", "validation"]:
                part = subset.loc[subset.split.eq(split)]
                for signal in signals:
                    n, n_events, pooled, event_mean, groups = rank_metrics(part, signal)
                    rows.append({"graph": graph_name, "stratum": stratum, "split": split,
                                 "signal": signal, "rows": n, "events": n_events,
                                 "pooled_spearman": pooled, "mean_event_spearman": event_mean,
                                 "event_groups": groups})
    expected = pd.DataFrame(rows)
    key_cols = ["graph", "stratum", "split", "signal"]
    comparison = observed.merge(expected, on=key_cols, suffixes=("_observed", "_expected"),
                                validate="one_to_one", how="outer", indicator=True)
    integer_match = all((comparison[f"{column}_observed"] == comparison[f"{column}_expected"]).all()
                        for column in ["rows", "events", "event_groups"])
    numeric_match = all(np.allclose(comparison[f"{column}_observed"],
                                    comparison[f"{column}_expected"], rtol=0, atol=1e-12,
                                    equal_nan=True)
                        for column in ["pooled_spearman", "mean_event_spearman"])
    checks = {
        "output_hash": summary["output_hash"] == sha256(RUN / "quickcheck_metrics.csv"),
        "graph_input_hash": summary["input_hashes"]["graph_edges"] == sha256(GRAPH / "edges.csv"),
        "panel_input_hash": summary["input_hashes"]["panel_events"] == sha256(PANEL / "events.csv"),
        "returns_input_hash": summary["input_hashes"]["returns"] == sha256(RETURNS),
        "trbc_input_hash": summary["input_hashes"]["trbc_diagnostic"] == sha256(TRBC),
        "code_input_hash": summary["input_hashes"]["code"] == sha256(ROOT / "quickcheck_analyst_signal_v2.py"),
        "no_test_events_joined": bool(new.announcement_day.max() < pd.Timestamp("2023-01-01") and
                                      old.announcement_day.max() < pd.Timestamp("2023-01-01")),
        "return_matrix_ends_before_test": bool(returns.Date.max() < pd.Timestamp("2023-01-01")),
        "boundary_targets_are_missing": bool(new.loc[new.boundary_purged, "target"].isna().all() and
                                             old.loc[old.boundary_purged, "target"].isna().all()),
        "metric_key_set_recomputed": bool(comparison._merge.eq("both").all()),
        "integer_metrics_recomputed": bool(integer_match),
        "rank_metrics_recomputed": bool(numeric_match),
        "summary_new_edge_count": summary["edges_new_graph"] == len(new),
        "summary_old_edge_count": summary["edges_rejected_graph"] == len(old),
        "summary_boundary_counts": (summary["boundary_purged_new"] == int(new.boundary_purged.sum()) and
                                    summary["boundary_purged_old"] == int(old.boundary_purged.sum())),
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "quickcheck_run": RUN.name, "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_passed": all(checks.values()),
              "rows_recomputed": {"analyst_graph": len(new), "residual_graph": len(old)},
              "test_policy": "Returns and events were truncated before 2023 prior to target construction."}
    (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise RuntimeError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    main()
