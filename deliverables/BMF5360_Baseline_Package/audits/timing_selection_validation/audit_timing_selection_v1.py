"""Validation-only decomposition of pooled timing and within-date selection signal.

Reads only existing development-validation predictions/features.  It never reads
sealed test targets, retrains a model, changes the company pool, or selects a model.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RUN_ID = "20260910T070000000000Z"
PRED = ROOT / "data/model_runs/ai_pool_daily_state_models_v1_1" / RUN_ID / "validation_predictions.csv"
FEATURES = ROOT / "data/model_ready_ai_pool_daily_v1/20260910T052100000000Z/model_features.csv"
MACRO = ROOT / "data/clean/daily_macro_v1/20260910T062700000000Z/macro_features_train_valid.csv"
CONFIG = ROOT / "ai_pool_daily_state_models_v1_1_config.json"
OUT = ROOT / "data/analysis/ai_timing_selection_audit_v1/20260911T001000000000Z"
MODELS = {
    "technical_only_logistic": "p_up_technical_only_logistic",
    "technical_plus_macro_logistic": "p_up_technical_plus_macro_logistic",
}
REPS, SEED, BLOCK_SESSIONS = 1000, 5360, 21

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(y) and len(np.unique(y)) == 2 else float("nan")

def cross_section(frame: pd.DataFrame, column: str) -> dict:
    date_rows, weighted_correct, weighted_pairs, ics = [], 0.0, 0, []
    for d, g in frame.groupby("formation_session", sort=True):
        y, p, r = g.y.to_numpy(int), g[column].to_numpy(float), g.forward_excess_return.to_numpy(float)
        usable_auc = len(g) >= 2 and len(np.unique(y)) == 2
        auc = safe_auc(y, p) if usable_auc else float("nan")
        pairs = int((y == 1).sum() * (y == 0).sum()) if usable_auc else 0
        if pairs:
            weighted_correct += auc * pairs
            weighted_pairs += pairs
        if len(g) >= 3 and np.nanstd(p) > 0 and np.nanstd(r) > 0:
            ic = float(spearmanr(p, r).statistic)
            if np.isfinite(ic): ics.append(ic)
        date_rows.append({"formation_session": str(d.date()), "rows": len(g), "positive": int(y.sum()), "pairs": pairs, "auc": auc})
    return {
        "pair_weighted_within_date_auc": weighted_correct / weighted_pairs if weighted_pairs else float("nan"),
        "equal_date_within_date_auc": float(np.nanmean([x["auc"] for x in date_rows])),
        "within_date_auc_groups": int(sum(np.isfinite(x["auc"]) for x in date_rows)),
        "within_date_pairs": int(weighted_pairs),
        "mean_date_rank_ic": float(np.mean(ics)) if ics else float("nan"),
        "rank_ic_groups": len(ics),
        "date_rows": date_rows,
    }

def blocks(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    position = {date: i for i, date in enumerate(sessions)}
    first = min(position[x] for x in frame.formation_session.unique())
    out = frame.copy()
    out["calendar_block"] = out.formation_session.map(lambda x: (position[x] - first) // BLOCK_SESSIONS)
    return out

def bootstrap(frame: pd.DataFrame, metric, seed: int) -> tuple[float, float, float, int]:
    groups = [g for _, g in frame.groupby("calendar_block", sort=True)]
    rng, values = np.random.default_rng(seed), []
    for _ in range(REPS):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        value = metric(sample)
        if np.isfinite(value): values.append(value)
    a = np.asarray(values, float)
    return float(np.quantile(a, .025)), float(np.quantile(a, .975)), float(np.std(a, ddof=1)), len(a)

def main() -> None:
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    OUT.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    technical, macro = config["technical_features"], config["macro_features"]
    usecols = ["sample_id", "formation_session", *technical]
    pred = pd.read_csv(PRED, parse_dates=["formation_session"])
    feat = pd.read_csv(FEATURES, usecols=usecols, parse_dates=["formation_session"])
    macro_values = pd.read_csv(MACRO, usecols=["spy_session", *macro], parse_dates=["spy_session"])
    macro_values = macro_values.rename(columns={"spy_session": "formation_session"})
    frame = pred.merge(feat, on=["sample_id", "formation_session"], validate="one_to_one")
    frame = frame.merge(macro_values, on="formation_session", how="left", validate="many_to_one")
    if not frame.split.eq("validation").all() or len(frame) != 992:
        raise RuntimeError("expected the frozen 992-row validation prediction set")
    frame["y"] = frame.y.astype(int)
    macro_calendar = pd.read_csv(MACRO, usecols=["spy_session"], parse_dates=["spy_session"])["spy_session"].drop_duplicates().sort_values()
    frame = blocks(frame, pd.DatetimeIndex(macro_calendar))

    constant_rows = []
    for feature in technical + macro:
        nuniques = frame.groupby("formation_session")[feature].nunique(dropna=False)
        constant_rows.append({
            "feature": feature,
            "feature_family": "technical" if feature in technical else "macro",
            "constant_on_all_validation_dates": bool((nuniques <= 1).all()),
            "mean_unique_values_per_date": float(nuniques.mean()),
            "min_unique_values_per_date": int(nuniques.min()),
            "max_unique_values_per_date": int(nuniques.max()),
        })
    pd.DataFrame(constant_rows).to_csv(OUT / "within_date_feature_variation.csv", index=False)

    results, by_date = {}, []
    for name, col in MODELS.items():
        raw_auc = safe_auc(frame.y.to_numpy(), frame[col].to_numpy())
        # This centering keeps within-date order but is intentionally labelled as
        # a diagnostic only: pooled AUC still compares observations across dates.
        centered = frame[col] - frame.groupby("formation_session")[col].transform("mean")
        centered_auc = safe_auc(frame.y.to_numpy(), centered.to_numpy())
        xs = cross_section(frame, col)
        for row in xs.pop("date_rows"):
            row["model"] = name
            by_date.append(row)
        results[name] = {"pooled_auc": raw_auc, "date_demeaned_pooled_auc_diagnostic": centered_auc, **xs}
    pd.DataFrame(by_date).to_csv(OUT / "within_date_metrics.csv", index=False)

    tech, macro_col = MODELS.values()
    metrics = {
        "technical_only_pooled_auc": lambda x: safe_auc(x.y.to_numpy(), x[tech].to_numpy()),
        "technical_plus_macro_pooled_auc": lambda x: safe_auc(x.y.to_numpy(), x[macro_col].to_numpy()),
        "pooled_auc_delta_macro_minus_technical": lambda x: safe_auc(x.y.to_numpy(), x[macro_col].to_numpy()) - safe_auc(x.y.to_numpy(), x[tech].to_numpy()),
        "within_date_auc_delta_macro_minus_technical": lambda x: cross_section(x, macro_col)["pair_weighted_within_date_auc"] - cross_section(x, tech)["pair_weighted_within_date_auc"],
    }
    boot = []
    for i, (name, fn) in enumerate(metrics.items()):
        point = float(fn(frame))
        low, high, se, n = bootstrap(frame, fn, SEED + i)
        boot.append({"metric": name, "point": point, "ci_2_5": low, "ci_97_5": high, "bootstrap_sd": se, "valid_replicates": n, "method": "21-SPY-session moving-calendar-block bootstrap"})
    pd.DataFrame(boot).to_csv(OUT / "block_bootstrap_summary.csv", index=False)

    missing = frame[["sample_id", "formation_session", "Instrument", "spread_median_60_bps"]].copy()
    missing["spread_median_60_bps_missing"] = missing.spread_median_60_bps.isna()
    missing_by_date = missing.groupby("formation_session").agg(rows=("sample_id", "size"), missing=("spread_median_60_bps_missing", "sum")).reset_index()
    missing_by_date["missing_rate"] = missing_by_date.missing / missing_by_date.rows
    missing.to_csv(OUT / "spread_missingness_by_row.csv", index=False)
    missing_by_date.to_csv(OUT / "spread_missingness_by_date.csv", index=False)

    summary = {
        "purpose": "validation-only timing-versus-selection decomposition; not model selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in [PRED, FEATURES, MACRO, CONFIG]},
        "test_target_opened": False,
        "rows": len(frame), "formation_dates": int(frame.formation_session.nunique()),
        "calendar_blocks": int(frame.calendar_block.nunique()), "block_sessions": BLOCK_SESSIONS,
        "raw_feature_counts": {"technical": len(technical), "macro": len(macro), "total": len(technical) + len(macro)},
        "constant_feature_counts": pd.DataFrame(constant_rows).groupby("feature_family")["constant_on_all_validation_dates"].sum().astype(int).to_dict(),
        "model_results": results,
        "spread_median_60_missing": {"rows": len(missing), "missing_rows": int(missing.spread_median_60_bps_missing.sum()), "missing_rate": float(missing.spread_median_60_bps_missing.mean()), "dates_with_any_missing": int((missing_by_date.missing > 0).sum()), "dates_all_missing": int((missing_by_date.missing == missing_by_date.rows).sum())},
        "limitations": ["Pooled AUC remains a mixed time-series/cross-sectional measure.", "Block bootstrap is the primary uncertainty estimate; IID DeLong assumptions do not hold for these clustered, overlapping labels.", "Only dates with both classes identify within-date AUC; only nonconstant dates identify rank IC.", "No costs, portfolio construction, test data, refitting, or feature selection are included."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "rows": len(frame), "blocks": int(frame.calendar_block.nunique())}, ensure_ascii=False))

if __name__ == "__main__":
    main()
