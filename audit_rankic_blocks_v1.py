"""Validation-only 21-session block inference for within-date rank IC."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z/validation_predictions.csv"
MACRO = ROOT / "data/clean/daily_macro_v1/20260910T062700000000Z/macro_features_train_valid.csv"
OUT = ROOT / "data/analysis/ai_timing_selection_audit_v1/20260911T001000000000Z"
COLS = {"technical_only_logistic": "p_up_technical_only_logistic", "technical_plus_macro_logistic": "p_up_technical_plus_macro_logistic"}

pred = pd.read_csv(PRED, parse_dates=["formation_session"])
calendar = pd.read_csv(MACRO, usecols=["spy_session"], parse_dates=["spy_session"]).spy_session.drop_duplicates().sort_values()
position = {d: i for i, d in enumerate(calendar)}
first = min(position[d] for d in pred.formation_session.unique())
pred["calendar_block"] = pred.formation_session.map(lambda d: (position[d] - first) // 21)
rows = []
for name, col in COLS.items():
    for d, g in pred.groupby("formation_session"):
        if len(g) >= 3 and g[col].nunique() > 1 and g.forward_excess_return.nunique() > 1:
            ic = spearmanr(g[col], g.forward_excess_return).statistic
            if np.isfinite(ic): rows.append({"model": name, "formation_session": d, "calendar_block": int(g.calendar_block.iloc[0]), "rank_ic": ic})
daily = pd.DataFrame(rows)
block = daily.groupby(["model", "calendar_block"], as_index=False).agg(mean_rank_ic=("rank_ic", "mean"), dates=("formation_session", "nunique"))
rng = np.random.default_rng(5360)
summary = []
for name, g in block.groupby("model"):
    values = g.mean_rank_ic.to_numpy()
    means = np.mean(values[rng.integers(0, len(values), size=(5000, len(values)))], axis=1)
    se = values.std(ddof=1) / np.sqrt(len(values))
    summary.append({"model": name, "daily_ic_groups": int((daily.model == name).sum()), "nonoverlapping_21_session_blocks": len(values), "mean_block_rank_ic": values.mean(), "block_mean_se_iid": se, "block_mean_t_vs_zero": values.mean()/se, "block_bootstrap_ci_2_5": np.quantile(means, .025), "block_bootstrap_ci_97_5": np.quantile(means, .975)})
wide = block.pivot(index="calendar_block", columns="model", values="mean_rank_ic").dropna()
delta = wide["technical_plus_macro_logistic"] - wide["technical_only_logistic"]
delta_boot = np.mean(delta.to_numpy()[rng.integers(0, len(delta), size=(5000, len(delta)))], axis=1)
delta_se = delta.std(ddof=1) / np.sqrt(len(delta))
summary.append({"model": "delta_macro_minus_technical", "daily_ic_groups": int(len(delta)), "nonoverlapping_21_session_blocks": len(delta), "mean_block_rank_ic": delta.mean(), "block_mean_se_iid": delta_se, "block_mean_t_vs_zero": delta.mean()/delta_se, "block_bootstrap_ci_2_5": np.quantile(delta_boot, .025), "block_bootstrap_ci_97_5": np.quantile(delta_boot, .975)})
daily.to_csv(OUT / "within_date_rank_ic.csv", index=False)
block.to_csv(OUT / "rank_ic_21session_blocks.csv", index=False)
pd.DataFrame(summary).to_csv(OUT / "rank_ic_block_summary.csv", index=False)
print(pd.DataFrame(summary).to_json(orient="records"))
