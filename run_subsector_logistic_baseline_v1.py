"""Run a cross-sectional subsector rotation Logistic baseline."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'data/model_ready_subsector_targets_v2/20260912T081316939312Z/subsector_target_panel.csv'
FEATURES = ['subsector_return_1', 'momentum_5', 'momentum_20', 'momentum_60',
            'volatility_20_ann', 'breadth_positive_20', 'dispersion_1']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xsection_zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        mean = out.groupby('Date')[col].transform('mean')
        std = out.groupby('Date')[col].transform('std')
        out[col + '_csz'] = (out[col] - mean) / std.replace(0, np.nan)
    return out


def date_resample_diagnostic_ci(values: np.ndarray, seed: int = 5360, reps: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return (float(np.quantile(draws, .025)), float(np.quantile(draws, .975)))


def main() -> None:
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'data/model_runs/subsector_logistic_baseline_v1' / run
    out.mkdir(parents=True, exist_ok=False)
    panel = pd.read_csv(TARGET, parse_dates=['Date', 'target_end_session'])
    panel = panel.loc[panel.model_eligible & panel.target_available].copy()
    panel['y'] = (panel['subsector_forward_excess'] > 0).astype(int)
    panel = xsection_zscore(panel, FEATURES)
    zcols = [c + '_csz' for c in FEATURES]
    before = len(panel)
    panel = panel.dropna(subset=zcols + ['y', 'subsector_forward_return', 'ai_benchmark_forward_return'])
    dropped = before - len(panel)
    train = panel.loc[panel.split.eq('training')].copy()
    valid = panel.loc[panel.split.eq('validation')].copy()
    model = LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, solver='lbfgs')
    model.fit(train[zcols], train.y)
    valid['p_up'] = model.predict_proba(valid[zcols])[:, 1]
    valid['predicted_up'] = (valid.p_up >= .5).astype(int)

    auc = float(roc_auc_score(valid.y, valid.p_up))
    brier = float(brier_score_loss(valid.y, valid.p_up))
    acc = float(accuracy_score(valid.y, valid.predicted_up))
    daily = []
    for d, g in valid.groupby('Date', sort=True):
        g = g.sort_values('p_up', ascending=False)
        k = min(2, len(g))
        model_ret = float(g.head(k).subsector_forward_return.mean())
        equal_ret = float(g.subsector_forward_return.mean())
        model_excess = float(g.head(k).subsector_forward_excess.mean())
        mom = g.sort_values('momentum_20', ascending=False)
        mom_excess = float(mom.head(k).subsector_forward_excess.mean())
        ic = g['p_up'].rank().corr(g['subsector_forward_excess'].rank())
        daily.append({'Date': d, 'selected_count': k, 'model_top2_return': model_ret,
                      'equal_subsector_return': equal_ret, 'model_top2_excess': model_excess,
                      'momentum_top2_excess': mom_excess, 'rank_ic': ic})
    daily = pd.DataFrame(daily)
    top2_ci = date_resample_diagnostic_ci(daily.model_top2_excess.to_numpy())
    coefs = pd.DataFrame({'feature': FEATURES, 'coefficient': model.coef_[0]})
    coefs['abs_coefficient'] = coefs.coefficient.abs()
    coefs = coefs.sort_values('abs_coefficient', ascending=False)
    valid.to_csv(out / 'validation_predictions.csv', index=False)
    daily.to_csv(out / 'daily_strategy_diagnostics.csv', index=False)
    coefs.to_csv(out / 'coefficients.csv', index=False)
    summary = {
        'schema_version': 'subsector_logistic_baseline_v1', 'run_id': run,
        'status': 'complete_development_only_no_2026_test', 'target_input': str(TARGET),
        'target_sha256': sha(TARGET), 'features': FEATURES,
        'rows': {'eligible_before_complete_case': before, 'complete_case': len(panel),
                 'dropped_missing_feature_or_target': dropped, 'training': len(train),
                 'validation': len(valid), 'validation_dates': int(valid.Date.nunique())},
        'model': {'type': 'LogisticRegression', 'C': .5, 'class_weight': 'balanced',
                  'cross_section_standardization': 'date-wise z-score; no imputation'},
        'metrics': {'validation_auc_pooled': auc, 'validation_brier': brier,
                    'validation_directional_accuracy': acc,
                    'validation_mean_daily_rank_ic': float(daily.rank_ic.mean()),
                    'validation_mean_model_top2_excess': float(daily.model_top2_excess.mean()),
                    'model_top2_excess_date_resample_diagnostic_95ci': top2_ci,
                    'validation_mean_momentum_top2_excess': float(daily.momentum_top2_excess.mean()),
                    'validation_mean_equal_subsector_return': float(daily.equal_subsector_return.mean())},
        'checks': {'no_test_dates_read': bool(panel.Date.max() < pd.Timestamp('2026-01-01')),
                   'split_order': bool(train.Date.max() < valid.Date.min()),
                   'predictions_only_after_fit': True,
                   'datewise_rank_diagnostics': True},
        'limitations': ['Only 7 pre-defined static subsector sleeves; labels overlap across dates.',
                        'The reported date-resampling interval is descriptive and does not adjust for five-session overlap or date clustering.',
                        'The current model is a baseline ranking model, not a transaction-cost or full fund backtest.',
                        'The 2026 H1 test labels remain unopened.'],
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'output': str(out), 'metrics': summary['metrics'], 'rows': summary['rows'], 'checks': summary['checks']}, indent=2, default=str))


if __name__ == '__main__':
    main()
