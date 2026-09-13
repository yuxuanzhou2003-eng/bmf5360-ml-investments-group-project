"""Daily AI timing trial with a disclosed formation-known cash proxy."""
import argparse, hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/model_ready_rotation_macro_v3/20260913021922Z'
TECH = ['subsector_return_1', 'momentum_5', 'momentum_20', 'momentum_60', 'volatility_20_ann', 'breadth_positive_20', 'dispersion_1']
MACRO = ['vix_level', 'vix_change_5d', 'dgs2_level', 'dgs2_change_20d', 'dgs10_change_20d', 'term_spread_10y_2y', 'broad_dollar_change_20d', 'dgs3mo_level']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_summary(out, data):
    (out / 'summary.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, default=int), encoding='utf-8')
    print(json.dumps(data, indent=2, ensure_ascii=False, default=int), flush=True)


def build(out):
    source = pd.read_csv(SOURCE / 'features.csv', parse_dates=['Date']).sort_values(['Date', 'primary_group'])
    assert source.Date.between('2021-01-01', '2025-12-31').all()
    assert not source.duplicated(['Date', 'primary_group']).any()
    assert source.groupby('Date').size().eq(7).all()
    assert source.groupby('Date')[MACRO].nunique(dropna=False).le(1).all().all()
    f = source.groupby('Date')[TECH].agg(lambda x: x.mean() if x.notna().all() else np.nan)
    f = f.join(source.groupby('Date')[MACRO].first()).reset_index()
    f['feature_eligible'] = np.isfinite(f[TECH + MACRO]).all(axis=1) & f.Date.map(source.groupby('Date').feature_eligible.all())
    f.to_csv(out / 'features.csv', index=False)
    src = pd.read_csv(SOURCE / 'targets.csv', parse_dates=['Date', 'entry_session', 'target_end_session'])
    assert src.Date.between('2021-01-01', '2025-12-31').all()
    cols = ['entry_session', 'target_end_session', 'ai_benchmark_forward_return', 'target_available', 'boundary_purged', 'split']
    assert src.groupby('Date')[cols].nunique(dropna=False).le(1).all().all()
    t = src[['Date'] + cols].drop_duplicates().sort_values('Date').reset_index(drop=True)
    assert f.Date.equals(t.Date)
    t['calendar_days'] = (t.target_end_session - t.entry_session).dt.days
    t['cash_proxy_annual_rate_percent'] = f.dgs3mo_level
    t['cash_forward_return'] = t.cash_proxy_annual_rate_percent / 100 * t.calendar_days / 365
    t['target_available'] &= np.isfinite(t.cash_forward_return) & t.calendar_days.gt(0)
    t['ai_excess_cash'] = (t.ai_benchmark_forward_return - t.cash_forward_return).where(t.target_available)
    t['y'] = (t.ai_excess_cash > 0).astype(float).where(t.target_available)
    t.to_csv(out / 'targets.csv', index=False)
    q = f[['Date', 'feature_eligible']].merge(t, on='Date', validate='one_to_one')
    q['model_eligible'] = q.feature_eligible & q.target_available & ~q.boundary_purged
    q.loc[~q.model_eligible].to_csv(out / 'exclusions.csv', index=False)
    counts = q.groupby('split').agg(rows=('Date', 'size'), feature_available=('feature_eligible', 'sum'), target_available=('target_available', 'sum'), purged=('boundary_purged', 'sum'), model_eligible=('model_eligible', 'sum')).reset_index()
    checks = {'one_row_per_date': len(f) == 1255 and not f.Date.duplicated().any(),
        'no_test_labels': bool(t.target_end_session.dropna().lt('2026-01-01').all()),
        'cash_rate_known_at_formation': bool(np.allclose(t.cash_proxy_annual_rate_percent, f.dgs3mo_level, equal_nan=True)),
        'purged_train_exits': bool(q.loc[q.model_eligible & q.split.eq('training'), 'target_end_session'].lt('2024-01-01').all())}
    save_summary(out, {'status': 'timing_panel_complete', 'counts': counts.to_dict('records'), 'checks': checks,
        'inputs': {str(SOURCE / name): sha(SOURCE / name) for name in ['features.csv', 'targets.csv']}, 'script_sha256': sha(__file__),
        'processing': {'source_rows_changed': 0, 'imputation': 'none', 'aggregation': 'equal7 raw technical means; not cross-sectional z scores', 'cash': 'formation-known 3mo yield percent /100 * calendar days /365, simple interest proxy'}, 'completed_utc': datetime.now(timezone.utc).isoformat()})
    assert all(checks.values())


def run(panel, out):
    f = pd.read_csv(panel / 'features.csv', parse_dates=['Date'])
    t = pd.read_csv(panel / 'targets.csv', parse_dates=['Date', 'entry_session', 'target_end_session'])
    assert f.Date.equals(t.Date) and f.Date.between('2021-01-01', '2025-12-31').all()
    train = f.feature_eligible & f.Date.lt('2024-01-01') & t.target_available & ~t.boundary_purged
    valid = f.feature_eligible & f.Date.ge('2024-01-01')
    assert t.loc[train, 'target_end_session'].lt('2024-01-01').all()
    ytrain = t.loc[train, 'y'].astype(int)
    pred = f.loc[valid, ['Date', 'momentum_20']].copy()
    configs, model_names, hashes = {}, [], {}
    (out / 'models').mkdir()
    for kind in ['logistic', 'random_forest']:
        for group, columns in [('F0', TECH), ('F1', TECH + MACRO)]:
            name = kind + '_' + group
            model_names.append(name)
            est = LogisticRegression(C=.5, class_weight=None, max_iter=2000, random_state=5360) if kind == 'logistic' else RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=20, max_features=.7, random_state=5360, n_jobs=2)
            pipe = make_pipeline(StandardScaler(), est)
            pipe.fit(f.loc[train, columns], ytrain)
            pred[name] = pipe.predict_proba(f.loc[valid, columns])[:, 1]
            path = out / 'models' / (name + '.joblib')
            joblib.dump({'pipeline': pipe, 'columns': columns}, path)
            hashes[name], configs[name] = sha(path), est.get_params()
            vals = est.coef_[0] if kind == 'logistic' else est.feature_importances_
            pd.DataFrame({'feature': columns, 'value': vals}).to_csv(out / (name + '_feature_weights.csv'), index=False)
            print('Fitted ' + name, flush=True)
    pred.to_csv(out / 'validation_predictions.csv', index=False)
    scored = pred.merge(t, on='Date', validate='one_to_one')
    names = model_names + ['always_ai', 'always_cash', 'trend20']
    for name in names:
        gate = (scored[name] >= .5).astype(float) if name in model_names else np.ones(len(scored)) if name == 'always_ai' else np.zeros(len(scored)) if name == 'always_cash' else scored.momentum_20.gt(0).astype(float)
        scored[name + '_ai_weight'] = gate
        scored[name + '_excess_cash'] = (gate * scored.ai_excess_cash).where(scored.target_available)
    scored.to_csv(out / 'validation_diagnostics.csv', index=False)
    scored.loc[~scored.target_available].to_csv(out / 'predictions_without_target.csv', index=False)
    e = scored.loc[scored.target_available].copy()
    metrics = []
    for period in ['all', '2024', '2025']:
        a = e if period == 'all' else e.loc[e.Date.dt.year.eq(int(period))]
        for name in names + ['constant_train_rate']:
            rec = {'model': name, 'period': period, 'dates': len(a), 'positive_label_rate': float(a.y.mean())}
            if name in model_names or name == 'constant_train_rate':
                prob = a[name].to_numpy() if name in model_names else np.full(len(a), ytrain.mean())
                rec.update(auc=float(roc_auc_score(a.y, prob)), brier=float(brier_score_loss(a.y, prob)), accuracy=float(np.mean((prob >= .5) == a.y)))
            if name in names:
                rec.update(ai_fraction=float(a[name + '_ai_weight'].mean()), mean_excess_cash=float(a[name + '_excess_cash'].mean()), mean_minus_always_ai=float((a[name + '_excess_cash'] - a.ai_excess_cash).mean()))
            metrics.append(rec)
    m = pd.DataFrame(metrics)
    m.to_csv(out / 'metrics.csv', index=False)
    calendar = pd.DatetimeIndex(sorted(f.loc[f.Date.ge('2024-01-01'), 'Date'].unique()), name='Date')
    z = scored.set_index('Date').reindex(calendar)
    good = z.target_available.eq(True)
    z.loc[~good, ['y'] + model_names + [n + '_excess_cash' for n in names]] = np.nan
    z.to_csv(out / 'validation_calendar.csv')
    anchors = z.iloc[::5].copy()
    anchors.to_csv(out / 'nonoverlap_fixed_anchors.csv')
    nonoverlap = [{'model': name, 'scheduled_windows': len(anchors), 'evaluable_windows': int(anchors[name + '_excess_cash'].notna().sum()), 'mean_excess_cash': float(anchors[name + '_excess_cash'].mean()), 'mean_minus_always_ai': float((anchors[name + '_excess_cash'] - anchors.always_ai_excess_cash).mean())} for name in names]
    pd.DataFrame(nonoverlap).to_csv(out / 'nonoverlap_summary.csv', index=False)
    intervals = bootstrap(z, m, model_names, names)
    intervals.to_csv(out / 'paired_intervals.csv', index=False)
    save_summary(out, {'status': 'completed_timing_diagnostic_not_full_fund', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'counts': {'training_dates': int(train.sum()), 'prediction_dates': len(pred), 'evaluation_dates': len(e), 'calendar_dates': len(calendar), 'withheld_prediction_dates': int((~scored.target_available).sum()), 'fixed_nonoverlap_evaluable': int(anchors.always_ai_excess_cash.notna().sum()), 'fixed_nonoverlap_scheduled': len(anchors)},
        'training_positive_rate': float(ytrain.mean()), 'metrics': m.replace({np.nan: None}).to_dict('records'), 'nonoverlap': nonoverlap,
        'hashes': {'features': sha(panel / 'features.csv'), 'targets': sha(panel / 'targets.csv'), 'script': sha(__file__), 'protocol': sha(ROOT / 'AI_TIMING_MACRO_TRIAL_V1_PROTOCOL_20260913.md')},
        'model_hashes': hashes, 'model_configs': configs, 'runtime': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'checks': {'train_exits_before_validation': bool(t.loc[train, 'target_end_session'].lt('2024-01-01').all()), 'predictions_from_features_only': pred.Date.reset_index(drop=True).equals(f.loc[valid, 'Date'].reset_index(drop=True)), 'finite_predictions': bool(np.isfinite(pred[model_names]).all().all()), 'no_test_dates': bool(t.Date.lt('2026-01-01').all())},
        'limitations': ['Cash is a formation-rate simple-interest proxy, not realized tradable cash/ETF return.', 'Daily H5 diagnostics overlap; fixed anchors with gaps are not compounded and no NAV/Sharpe/cost result is claimed.', 'Static universe, current macro vintage, informative missingness and reused development validation remain.', 'Binary .5 rule ignores return magnitude and risk; intervals unadjusted for total research multiple comparisons.']})


def bootstrap(z, metrics, model_names, names):
    intervals = []
    point = metrics.loc[metrics.period.eq('all')].set_index('model')
    for block in [20, 5, 10, 40]:
        rng = np.random.default_rng(5360 + block)
        draws = {name: {'auc': [], 'brier': [], 'mean_excess_cash': []} for name in names}
        for _ in range(1000):
            starts = rng.integers(0, len(z) - block + 1, size=int(np.ceil(len(z) / block)))
            ix = (starts[:, None] + np.arange(block)).ravel()[:len(z)]
            b = z.iloc[ix]
            for name in names:
                draws[name]['mean_excess_cash'].append(float(b[name + '_excess_cash'].mean()))
                if name in model_names:
                    ok = b.y.notna() & b[name].notna()
                    yy, pp = b.loc[ok, 'y'], b.loc[ok, name]
                    draws[name]['auc'].append(float(roc_auc_score(yy, pp)) if yy.nunique() == 2 else np.nan)
                    draws[name]['brier'].append(float(np.mean((pp - yy) ** 2)))
        pairs = [('logistic_F1', 'logistic_F0'), ('random_forest_F1', 'random_forest_F0')] + [(name, 'always_ai') for name in model_names] + [(name, 'trend20') for name in model_names]
        for name, base in pairs:
            for metric in ['auc', 'brier', 'mean_excess_cash']:
                if not draws[name][metric] or not draws[base][metric]:
                    continue
                diff = np.asarray(draws[name][metric]) - np.asarray(draws[base][metric])
                lo, hi = np.nanquantile(diff, [.025, .975])
                intervals.append({'model': name, 'comparator': base, 'metric': metric, 'block_length': block,
                    'estimate': float(point.loc[name, metric] - point.loc[base, metric]), 'ci_low': float(lo), 'ci_high': float(hi)})
        print('Paired bootstrap block=' + str(block), flush=True)
    return pd.DataFrame(intervals)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['panel', 'models'], required=True)
    p.add_argument('--panel-dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    build(args.output_dir) if args.stage == 'panel' else run(args.panel_dir, args.output_dir)
