"""Independent aggregation, cash arithmetic and saved prediction checks."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss

ap = argparse.ArgumentParser()
ap.add_argument('--panel-dir', type=Path, required=True)
ap.add_argument('--run-dir', type=Path)
ap.add_argument('--audit-dir', type=Path, required=True)
a = ap.parse_args()
a.audit_dir.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parent
src = root / 'data/model_ready_rotation_macro_v3/20260913021922Z'
s = pd.read_csv(src / 'features.csv', parse_dates=['Date']).sort_values(['Date', 'primary_group'])
f = pd.read_csv(a.panel_dir / 'features.csv', parse_dates=['Date'])
t = pd.read_csv(a.panel_dir / 'targets.csv', parse_dates=['Date', 'entry_session', 'target_end_session'])
tech = ['subsector_return_1', 'momentum_5', 'momentum_20', 'momentum_60', 'volatility_20_ann', 'breadth_positive_20', 'dispersion_1']
macro = ['vix_level', 'vix_change_5d', 'dgs2_level', 'dgs2_change_20d', 'dgs10_change_20d', 'term_spread_10y_2y', 'broad_dollar_change_20d', 'dgs3mo_level']
checks = {'unique_dates': f.Date.nunique() == len(f) == 1255, 'matching_target_dates': f.Date.equals(t.Date),
    'development_only': bool(t.Date.between('2021-01-01', '2025-12-31').all())}
for c in tech:
    expected = np.mean(s[c].to_numpy().reshape(-1, 7), axis=1)
    checks['raw_mean_' + c] = bool(np.allclose(f[c], expected, equal_nan=True))
for c in macro:
    checks['macro_' + c] = bool(np.allclose(f[c], s.groupby('Date')[c].first().to_numpy(), equal_nan=True))
expected_cash = f.dgs3mo_level.to_numpy() * np.array([(x - y).days if pd.notna(x) and pd.notna(y) else np.nan for x, y in zip(t.target_end_session, t.entry_session)]) / 36500
checks['cash_calendar_units'] = bool(np.allclose(t.cash_forward_return, expected_cash, equal_nan=True))
known = t.target_available
checks['available_labels_finite'] = bool(np.isfinite(t.loc[known, ['y', 'ai_excess_cash', 'ai_benchmark_forward_return', 'cash_forward_return']]).all().all())
checks['label_direction'] = bool((t.loc[known, 'y'] == (t.loc[known, 'ai_benchmark_forward_return'] > t.loc[known, 'cash_forward_return'])).all())
checks['excess_cash_difference'] = bool(np.allclose(t.loc[known, 'ai_excess_cash'], t.loc[known, 'ai_benchmark_forward_return'] - expected_cash[known]))
checks['boundary_purge'] = bool((t.boundary_purged == (t.Date.lt('2024-01-01') & t.target_end_session.ge('2024-01-01'))).all())
checks['eligibility_ex_ante'] = bool((f.feature_eligible.to_numpy() == s.groupby('Date').feature_eligible.all().to_numpy()).all())
if a.run_dir:
    p = pd.read_csv(a.run_dir / 'validation_predictions.csv', parse_dates=['Date'])
    mask = f.feature_eligible & f.Date.ge('2024-01-01')
    checks['prediction_dates_ex_ante'] = p.Date.reset_index(drop=True).equals(f.loc[mask, 'Date'].reset_index(drop=True))
    d = pd.read_csv(a.run_dir / 'validation_diagnostics.csv', parse_dates=['Date'])
    m = pd.read_csv(a.run_dir / 'metrics.csv')
    for name in ['logistic_F0', 'logistic_F1', 'random_forest_F0', 'random_forest_F1']:
        obj = joblib.load(a.run_dir / 'models' / (name + '.joblib'))
        rebuilt = obj['pipeline'].predict_proba(f.loc[mask, obj['columns']])[:, 1]
        checks['prediction_reproduced_' + name] = bool(np.allclose(p[name], rebuilt, atol=1e-12))
        ee = d.loc[d.target_available]
        row = m.loc[m.model.eq(name) & m.period.eq('all')].iloc[0]
        checks['auc_' + name] = bool(np.isclose(roc_auc_score(ee.y, ee[name]), row.auc))
        checks['brier_' + name] = bool(np.isclose(brier_score_loss(ee.y, ee[name]), row.brier))
        checks['economic_' + name] = bool(np.isclose(((ee[name] >= .5) * ee.ai_excess_cash).mean(), row.mean_excess_cash))
    cal = pd.DatetimeIndex(sorted(f.loc[f.Date.ge('2024-01-01'), 'Date'].unique()))
    z = d.set_index('Date').reindex(cal)
    delta = z.random_forest_F1_excess_cash - z.random_forest_F0_excess_cash
    rng = np.random.default_rng(5380)
    draws = []
    for _ in range(1000):
        starts = rng.integers(0, len(cal) - 19, size=int(np.ceil(len(cal) / 20)))
        ix = np.concatenate([np.arange(q, q + 20) for q in starts])[:len(cal)]
        draws.append(float(delta.iloc[ix].mean()))
    ci = np.quantile(draws, [.025, .975])
    b = pd.read_csv(a.run_dir / 'paired_intervals.csv')
    ref = b.loc[b.model.eq('random_forest_F1') & b.comparator.eq('random_forest_F0') & b.metric.eq('mean_excess_cash') & b.block_length.eq(20)].iloc[0]
    checks['independent_paired_ci'] = bool(np.allclose(ci, [ref.ci_low, ref.ci_high], atol=1e-12))
    anchor = pd.read_csv(a.run_dir / 'nonoverlap_fixed_anchors.csv', parse_dates=['Date'])
    checks['fixed_anchors_no_future_reselection'] = anchor.Date.tolist() == cal[::5].tolist()
result = {'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
          'scope': 'Reconstructs aggregation/cash/label/purge and optional saved-model predictions, metrics and one paired CI; not full tradability or vintage certification.'}
(a.audit_dir / ('model_verification.json' if a.run_dir else 'panel_verification.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
assert all(checks.values())
