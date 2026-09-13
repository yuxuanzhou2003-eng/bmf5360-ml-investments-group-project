"""Audit pre-event earnings/consensus readiness for universe v2.

This is deliberately separate from panel construction: it does not build a
graph, inspect forward returns, fit a model, or select observations by labels.
Importing the module is side-effect free; ``main`` writes one new audit run.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean" / "v2"
AUDIT = ROOT / "data" / "audit"
INTERVALS = ROOT / "data" / "audit" / "universe_rebuild" / "membership_intervals.csv"
SPANS_PATH = ROOT / "data" / "audit" / "universe_rebuild" / "universe_spans_2015_2026.csv"
CONFIG_PATH = ROOT / "panel_v2_config.json"
VALIDATION_PATH = ROOT / "data" / "audit" / "v2" / "validation.json"
EXPECTED_ROWS = {"actuals": 32415, "estimates": 424101, "returns": 2081816}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_dates(values):
    parsed = pd.to_datetime(values, format="mixed", errors="coerce")
    if getattr(parsed, "dt", None) is not None:
        try:
            if parsed.dt.tz is not None:
                parsed = parsed.dt.tz_localize(None)
        except (AttributeError, TypeError):
            parsed = pd.to_datetime(values, format="mixed", errors="coerce", utc=True).dt.tz_localize(None)
    return parsed


def read_inputs():
    paths = {
        "actuals": CLEAN / "actuals.csv",
        "estimates": CLEAN / "estimates.csv",
        "returns": CLEAN / "returns.csv",
        "membership_intervals": INTERVALS,
        "universe_spans": SPANS_PATH,
        "config": CONFIG_PATH,
        "clean_validation": VALIDATION_PATH,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))
    hashes = {name: {"path": rel(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    actuals = pd.read_csv(paths["actuals"]).rename(columns={
        "Period End Date": "period_end", "Earnings Per Share - Actual": "actual"
    })
    estimates = pd.read_csv(paths["estimates"]).rename(columns={
        "Period End Date": "period_end",
        "Earnings Per Share - Mean": "consensus",
        "Earnings Per Share - Standard Deviation": "dispersion",
        "Earnings Per Share - Number of Included Estimates": "analysts",
    })
    returns = pd.read_csv(paths["returns"], usecols=["Instrument", "Date"])
    intervals = pd.read_csv(paths["membership_intervals"])
    universe_spans = pd.read_csv(paths["universe_spans"])
    actuals["announcement"] = parse_dates(actuals["announcement"])
    actuals["announcement_day"] = actuals["announcement"].dt.normalize()
    actuals["period_end"] = parse_dates(actuals["period_end"]).dt.normalize()
    estimates["snapshot"] = parse_dates(estimates["snapshot"])
    estimates["period_end"] = parse_dates(estimates["period_end"]).dt.normalize()
    returns["Date"] = parse_dates(returns["Date"])
    intervals["start"] = parse_dates(intervals["start"]).dt.normalize()
    intervals["end"] = parse_dates(intervals["end"]).dt.normalize()
    observed = {"actuals": len(actuals), "estimates": len(estimates), "returns": len(returns)}
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    checks = {
        "expected_clean_row_counts": observed == EXPECTED_ROWS,
        "clean_validation_223_of_223_passed": validation.get("passed") == validation.get("total") == 223 and not validation.get("failed"),
        "requested_universe_spans_ric_count_is_782": int(universe_spans.ric.nunique()) == 782,
    }
    if not all(checks.values()):
        raise RuntimeError("Input validation failed: " + ", ".join(k for k, v in checks.items() if not v))
    return actuals, estimates, returns, intervals, universe_spans, config, hashes, checks


def analyze(actuals, estimates, returns, intervals, universe_spans, config):
    start = pd.Timestamp(config["event_start"]).normalize()
    end = pd.Timestamp(config["event_end"]).normalize()
    age_limit = int(config["max_snapshot_age_days"])
    events = actuals[actuals.announcement_day.between(start, end)].copy()
    spans = {ric: list(zip(g.start, g.end)) for ric, g in intervals.groupby("ric")}
    events["source_kind"] = np.where(events.Instrument.astype(str).str.contains("^", regex=False), "caret_ric", "live")
    events["source_in_index_on_announcement"] = [
        any(s <= d <= e for s, e in spans.get(i, []))
        for i, d in zip(events.Instrument, events.announcement_day)
    ]
    events["status"] = np.where(events.source_in_index_on_announcement, "eligible", "source_not_in_index_on_announcement")
    events["selected_snapshot"] = pd.NaT
    events["selected_snapshot_day"] = pd.NaT
    events["consensus"] = np.nan
    events["snapshot_age_days"] = np.nan
    events["dispersion"] = np.nan
    events["analysts"] = np.nan
    events["standardized_surprise"] = np.nan
    events["dispersion_status"] = pd.NA
    events["valid_standardized_surprise"] = False
    groups = {key: g.sort_values("snapshot") for key, g in estimates.groupby(["Instrument", "period_end"], sort=False)}
    for idx, event in events[events.source_in_index_on_announcement].iterrows():
        group = groups.get((event.Instrument, event.period_end))
        if group is None:
            events.at[idx, "status"] = "missing_estimates_for_quarter"
            continue
        prior = group[group.snapshot < event.announcement_day]
        if prior.empty:
            events.at[idx, "status"] = "no_preannouncement_snapshot"
            continue
        latest = prior.iloc[-1]
        age = int((event.announcement_day - latest.snapshot.normalize()).days)
        events.at[idx, "selected_snapshot"] = latest.snapshot
        events.at[idx, "selected_snapshot_day"] = latest.snapshot.normalize()
        events.at[idx, "snapshot_age_days"] = age
        events.at[idx, "consensus"] = latest.consensus
        events.at[idx, "dispersion"] = latest.dispersion
        events.at[idx, "analysts"] = latest.analysts
        if age > age_limit:
            events.at[idx, "status"] = "stale_snapshot"
        else:
            events.at[idx, "status"] = "matched_snapshot"
    eligible = events[events.source_in_index_on_announcement]
    matched = eligible[eligible.status.eq("matched_snapshot")]
    statuses = ["missing_estimates_for_quarter", "no_preannouncement_snapshot", "stale_snapshot", "matched_snapshot"]
    funnel = {
        "event_window_total": int(len(events)),
        "source_not_in_index_on_announcement": int((~events.source_in_index_on_announcement).sum()),
        "eligible_source_events": int(len(eligible)),
        **{status: int((eligible.status == status).sum()) for status in statuses},
        "denominator_definition": "eligible_source_events = actuals with announcement_day in configured window and source in index on that exact day",
    }
    disp = pd.to_numeric(matched.dispersion, errors="coerce")
    actual = pd.to_numeric(matched.actual, errors="coerce")
    consensus = pd.to_numeric(matched.consensus, errors="coerce")
    finite_disp = np.isfinite(disp)
    positive_disp = finite_disp & (disp > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        surprise = (actual - consensus) / disp
    valid_surprise = positive_disp & np.isfinite(actual) & np.isfinite(consensus) & np.isfinite(surprise)
    status = np.select(
        [~finite_disp, finite_disp & (disp == 0), finite_disp & (disp < 0), valid_surprise],
        ["nonfinite_dispersion", "zero_dispersion", "negative_dispersion", "valid_standardized_surprise"],
        default="positive_dispersion_nonfinite_surprise",
    )
    events.loc[matched.index, "dispersion_status"] = status
    events.loc[matched.index, "standardized_surprise"] = surprise.where(valid_surprise, np.nan)
    events.loc[matched.index, "valid_standardized_surprise"] = valid_surprise
    dispersion_counts = {key: int((status == key).sum()) for key in (
        "valid_standardized_surprise", "zero_dispersion", "nonfinite_dispersion",
        "negative_dispersion", "positive_dispersion_nonfinite_surprise",
    )}
    dispersion_summary = {
        "denominator_eligible_source_events": int(len(eligible)),
        "denominator_matched_snapshot_events": int(len(matched)),
        "counts": dispersion_counts,
        "share_of_matched_snapshot_events": {key: (value / len(matched) if len(matched) else None) for key, value in dispersion_counts.items()},
    }
    analyst_values = pd.to_numeric(matched.analysts, errors="coerce")
    analyst_summary = {
        "denominator_eligible_source_events": int(len(eligible)),
        "denominator_matched_snapshot_events": int(len(matched)),
        "nonmissing": int(analyst_values.notna().sum()),
        "missing": int(analyst_values.isna().sum()),
        "zero": int((analyst_values == 0).sum()),
        "positive": int((analyst_values > 0).sum()),
        "min": float(analyst_values.min()) if analyst_values.notna().any() else None,
        "p25": float(analyst_values.quantile(.25)) if analyst_values.notna().any() else None,
        "median": float(analyst_values.median()) if analyst_values.notna().any() else None,
        "p75": float(analyst_values.quantile(.75)) if analyst_values.notna().any() else None,
        "max": float(analyst_values.max()) if analyst_values.notna().any() else None,
    }
    source_comparison = {}
    for kind in ("live", "caret_ric"):
        all_kind = events[events.source_kind.eq(kind)]
        eligible_kind = all_kind[all_kind.source_in_index_on_announcement]
        matched_kind = eligible_kind[eligible_kind.status.eq("matched_snapshot")]
        source_comparison[kind] = {
            "event_window": int(len(all_kind)),
            "eligible_source_events": int(len(eligible_kind)),
            "membership_eligibility_rate": len(eligible_kind) / len(all_kind) if len(all_kind) else None,
            "matched_snapshot_events": int(len(matched_kind)),
            "snapshot_match_rate_among_eligible": len(matched_kind) / len(eligible_kind) if len(eligible_kind) else None,
            "snapshot_match_rate_of_window": len(matched_kind) / len(all_kind) if len(all_kind) else None,
        }
    by_year = {}
    for year in range(2015, 2027):
        y = events[events.announcement_day.dt.year.eq(year)]
        ey = y[y.source_in_index_on_announcement]
        by_year[str(year)] = {
            "event_window": int(len(y)),
            "eligible_source_events": int(len(ey)),
            "source_not_in_index": int((~y.source_in_index_on_announcement).sum()),
            **{status: int((ey.status == status).sum()) for status in statuses},
        }
    failure = events[events.status.ne("matched_snapshot")].copy()
    missing = failure.groupby(["Instrument", "source_kind"], dropna=False).agg(
        event_count=("status", "size"),
        source_not_in_index=("status", lambda s: int((s == "source_not_in_index_on_announcement").sum())),
        missing_quarter=("status", lambda s: int((s == "missing_estimates_for_quarter").sum())),
        no_prior_snapshot=("status", lambda s: int((s == "no_preannouncement_snapshot").sum())),
        stale_snapshot=("status", lambda s: int((s == "stale_snapshot").sum())),
    ).reset_index().sort_values(["source_kind", "Instrument"])
    span_rics = set(universe_spans.ric.astype(str))
    actual_rics = set(actuals.Instrument.astype(str))
    estimate_rics = set(estimates.Instrument.astype(str))
    ric_coverage = pd.DataFrame({"Instrument": sorted(span_rics)})
    ric_coverage["has_clean_actuals"] = ric_coverage.Instrument.isin(actual_rics)
    ric_coverage["has_clean_estimates"] = ric_coverage.Instrument.isin(estimate_rics)
    ric_summary = {
        "membership_span_ric_count": int(len(span_rics)),
        "clean_actuals_ric_count": int(len(actual_rics)),
        "clean_estimates_ric_count": int(len(estimate_rics)),
        "missing_actuals_ric_vs_spans": sorted(span_rics - actual_rics),
        "missing_estimates_ric_vs_spans": sorted(span_rics - estimate_rics),
        "actuals_ric_not_in_spans": sorted(actual_rics - span_rics),
        "estimates_ric_not_in_spans": sorted(estimate_rics - span_rics),
    }
    selected = events[events.status.isin(["stale_snapshot", "matched_snapshot"])]
    summary = {
        "analysis_run_scope": {"event_start": str(start.date()), "event_end_inclusive": str(end.date()), "years": list(range(2015, 2027))},
        "funnel": funnel,
        "by_event_year": by_year,
        "surprise_dispersion": dispersion_summary,
        "analyst_count_distribution": analyst_summary,
        "source_comparison": source_comparison,
        "source_kind_definition": "caret_ric means the Instrument text contains a literal ^; it is a coverage stratum, not an independently verified delisting date.",
        "ric_coverage_against_782_spans": ric_summary,
        "missing_companies": missing,
        "event_statuses": events,
        "ric_coverage": ric_coverage,
        "checks": {
            "funnel_partitions_eligible_events": sum(funnel[s] for s in statuses) == len(eligible),
            "selected_snapshots_strictly_before_announcement_day": bool((selected.selected_snapshot_day < selected.announcement_day).all()),
            "matched_snapshots_at_most_14_days_old": bool(matched.snapshot_age_days.between(0, age_limit).all()),
            "valid_standardized_surprises_are_finite": bool(np.isfinite(events.loc[events.valid_standardized_surprise, "standardized_surprise"]).all()),
        },
        "input_rows_and_instruments": {
            "actuals": {"rows": int(len(actuals)), "instruments": int(actuals.Instrument.nunique())},
            "estimates": {"rows": int(len(estimates)), "instruments": int(estimates.Instrument.nunique())},
            "returns": {"rows_read": int(len(returns)), "instruments": int(returns.Instrument.nunique()),
                        "min_date": str(returns.Date.min().date()), "max_date": str(returns.Date.max().date())},
            "membership_intervals": {"rows": int(len(intervals)), "instruments": int(intervals.ric.nunique())},
            "requested_universe_spans": {"rows": int(len(universe_spans)), "instruments": int(universe_spans.ric.nunique())},
        },
    }
    return summary


def markdown(summary, hashes, run_id):
    f = summary["funnel"]
    lines = ["# Universe v2 事件就绪审计", "", f"运行 ID：`{run_id}`", "",
             "公告窗口按 announcement day 计，包含 2015-01-01 至 2026-06-30。所有快照漏斗比例的共同分母是公告日属于指数的 eligible source events；未使用图、未来收益、模型或标签筛选。", "",
             "## 漏斗", "", "|阶段|行数|", "|---|---:|"]
    for key, value in f.items():
        if isinstance(value, int):
            lines.append(f"|{key}|{value}|")
    lines += ["", "## live 与含 ^ 的 RIC", "", "|类型|窗口事件|eligible|成员资格率|快照匹配|eligible 内匹配率|窗口内匹配率|", "|---|---:|---:|---:|---:|---:|---:|"]
    for kind, value in summary["source_comparison"].items():
        fmt = lambda x: "NA" if x is None else f"{x:.2%}"
        lines.append(f"|{kind}|{value['event_window']}|{value['eligible_source_events']}|{fmt(value['membership_eligibility_rate'])}|{value['matched_snapshot_events']}|{fmt(value['snapshot_match_rate_among_eligible'])}|{fmt(value['snapshot_match_rate_of_window'])}|")
    d = summary["surprise_dispersion"]
    lines += ["", "## 离散度与标准化 surprise", "", f"匹配快照事件为 {d['denominator_matched_snapshot_events']}；其中有效标准化 surprise 为 {d['counts']['valid_standardized_surprise']}，零离散度为 {d['counts']['zero_dispersion']}，非有限离散度为 {d['counts']['nonfinite_dispersion']}。", "", "## 请求 universe spans 的 RIC 覆盖", ""]
    r = summary["ric_coverage_against_782_spans"]
    interval_count = summary["input_rows_and_instruments"]["membership_intervals"]["instruments"]
    lines += [f"研究请求清单 universe_spans 含 {r['membership_span_ric_count']} 个 RIC；逐日资格仍按历史 membership_intervals 的 {interval_count} 个 RIC 核对。缺 clean actuals 的 RIC（{len(r['missing_actuals_ric_vs_spans'])}）：`{', '.join(r['missing_actuals_ric_vs_spans'])}`。", f"缺 clean estimates 的 RIC（{len(r['missing_estimates_ric_vs_spans'])}）：`{', '.join(r['missing_estimates_ric_vs_spans'])}`。"]
    lines += ["", "## 输入 SHA-256", "", "|输入|路径|SHA-256|", "|---|---|---|"]
    lines += [f"|{name}|{item['path']}|`{item['sha256']}`|" for name, item in hashes.items()]
    lines += ["", "`event_statuses.csv` 保留了窗口内每个 actual 事件及所选快照、consensus、离散度、分析师数、年龄和标准化 surprise 状态；`ric_coverage.csv` 列出请求 universe_spans 的全部 782 个 RIC。未填补缺失值，也不因 ^ 标记或退市身份删除证券。本审计未使用图、未来收益、标签、模型或 LSEG 请求。", ""]
    return "\n".join(lines)


def main():
    actuals, estimates, returns, intervals, universe_spans, config, hashes, input_checks = read_inputs()
    code_path = Path(__file__).resolve()
    hashes["analysis_code"] = {"path": rel(code_path), "sha256": sha256_file(code_path)}
    summary = analyze(actuals, estimates, returns, intervals, universe_spans, config)
    after = {name: sha256_file(ROOT / item["path"]) for name, item in hashes.items() if name != "analysis_code"}
    before = {name: item["sha256"] for name, item in hashes.items() if name != "analysis_code"}
    if before != after:
        raise RuntimeError("Input changed during event-readiness analysis")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = AUDIT / "event_readiness" / run_id
    out.mkdir(parents=True, exist_ok=False)
    missing = summary.pop("missing_companies")
    event_statuses = summary.pop("event_statuses")
    ric_coverage = summary.pop("ric_coverage")
    missing.to_csv(out / "missing_companies.csv", index=False)
    event_statuses.to_csv(out / "event_statuses.csv", index=False)
    ric_coverage.to_csv(out / "ric_coverage.csv", index=False)
    summary.update({"run_id": run_id, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "input_hashes": hashes, "config": config,
                    "input_checks": input_checks,
                    "expected_input_rows": EXPECTED_ROWS,
                    "missing_companies_csv": rel(out / "missing_companies.csv"),
                    "event_statuses_csv": rel(out / "event_statuses.csv"),
                    "ric_coverage_csv": rel(out / "ric_coverage.csv"),
                    "method": "Exact announcement-day membership; latest same Instrument/period snapshot with snapshot < normalized announcement day; max age from snapshot day.",
                    "limitations": ["Clean-table required fields are assumed valid; no imputation or outlier removal is applied.", "Membership eligibility depends on the supplied reconstructed intervals.", "No graph, forward return, model, or label-based selection is performed."]})
    (out / "event_readiness.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "event_readiness.md").write_text(markdown(summary, hashes, run_id), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "output_dir": rel(out), "funnel": summary["funnel"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
