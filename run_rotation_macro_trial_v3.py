"""Six fixed models; ex-ante predictions and paired time-block diagnostics."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TECH = ['subsector_return_1', 'momentum_5', 'momentum_20', 'momentum_60', 'volatility_20_ann', 'breadth_positive_20', 'dispersion_1']
MACRO = ['vix_level', 'vix_change_5d', 'dgs2_level', 'dgs2_change_20d', 'dgs10_change_20d', 'term_spread_10y_2y', 'broad_dollar_change_20d', 'dgs3mo_level']
INTER = [(0, 4), (1, 1), (4, 3), (6, 2)]
SEED = 5360


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def concordance(p, r):
    i, j = np.triu_indices(len(p), 1)
    actual = np.sign(r[i] - r[j])
    pred = np.sign(p[i] - p[j])
    distinct = actual != 0
    return float(np.mean(np.where(pred[distinct] == 0, .5, pred[distinct] == actual[distinct]))) if distinct.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    fp, tp = args.panel_dir / 'features.csv', args.panel_dir / 'targets.csv'
    f = pd.read_csv(fp, parse_dates=['Date']).sort_values(['Date', 'primary_group']).reset_index(drop=True)
    t = pd.read_csv(tp, parse_dates=['Date', 'entry_session', 'target_end_session']).sort_values(['Date', 'primary_group']).reset_index(drop=True)
    assert f[['Date', 'primary_group']].equals(t[['Date', 'primary_group']])
    assert f.Date.between('2021-01-01', '2025-12-31').all()
    assert t.Date.between('2021-01-01', '2025-12-31').all()
    fit_mask = f.feature_eligible & f.Date.lt('2024-01-01') & t.target_available & ~t.boundary_purged
    pred_mask = f.feature_eligible & f.Date.ge('2024-01-01')
    assert t.loc[fit_mask, 'target_end_session'].lt('2024-01-01').all()
    assert f.loc[fit_mask].groupby('Date').size().eq(7).all()
    assert f.loc[pred_mask].groupby('Date').size().eq(7).all()
    ytrain = t.loc[fit_mask, 'y'].astype(int).to_numpy()
    assert len(np.unique(ytrain)) == 2
    # These transformations never depend on validation targets.
    macro_fit_mask = f.feature_eligible & f.Date.lt('2024-01-01')
    macro_scaler = StandardScaler().fit(f.loc[macro_fit_mask, MACRO])
    macro_all = macro_scaler.transform(f[MACRO])
    tech_all = f[[x + '_csz' for x in TECH]].to_numpy()
    inter_all = np.column_stack([macro_all[:, a] * tech_all[:, b] for a, b in INTER])
    mats = {'F0': tech_all, 'F1': np.column_stack([tech_all, macro_all]),
            'F2': np.column_stack([tech_all, macro_all, inter_all])}
    names = {'F0': [x + '_csz' for x in TECH]}
    names['F1'] = names['F0'] + MACRO
    names['F2'] = names['F1'] + [MACRO[a] + '_x_' + TECH[b] + '_csz' for a, b in INTER]
    pred = f.loc[pred_mask, ['Date', 'primary_group', 'momentum_20']].copy()
    model_names, parameters, artifact_hashes, fit_key_hashes = [], {}, {}, {}
    weights = []
    for kind in ['logistic', 'random_forest']:
        for group in ['F0', 'F1', 'F2']:
            key = kind + '_' + group
            model_names.append(key)
            base = (LogisticRegression(C=.5, class_weight=None, max_iter=2000, random_state=SEED)
                    if kind == 'logistic' else RandomForestClassifier(n_estimators=300, max_depth=4,
                    min_samples_leaf=60, max_features=.7, random_state=SEED, n_jobs=2))
            pipe = make_pipeline(StandardScaler(), base)
            pipe.fit(mats[group][fit_mask], ytrain)
            fit_key_hashes[key] = hashlib.sha256(f.loc[fit_mask, ['Date', 'primary_group']].to_csv(index=False).encode()).hexdigest()
            pred[key] = pipe.predict_proba(mats[group][pred_mask])[:, 1]
            artifact = out / 'models' / (key + '.joblib')
            joblib.dump({'pipeline': pipe, 'macro_scaler': macro_scaler, 'features': names[group],
                         'tech_columns': TECH, 'macro_columns': MACRO, 'interactions': INTER}, artifact)
            artifact_hashes[artifact.name] = sha(artifact)
            parameters[key] = base.get_params()
            importance = base.coef_[0] if kind == 'logistic' else base.feature_importances_
            weights.extend({'model': key, 'feature': name, 'value': float(val),
                            'kind': 'coefficient' if kind == 'logistic' else 'impurity_importance'}
                           for name, val in zip(names[group], importance))
            print('Fitted ' + key, flush=True)
    # Save the unlabelled predictions before the first validation-label join.
    pred.to_csv(out / 'validation_predictions.csv', index=False)
    pd.DataFrame(weights).to_csv(out / 'feature_weights.csv', index=False)
    scored = pred.merge(t, on=['Date', 'primary_group'], how='left', validate='one_to_one')
    scored.to_csv(out / 'validation_scored.csv', index=False)
    scored.loc[~scored.target_available].to_csv(out / 'predictions_without_evaluable_target.csv', index=False)
    excludes = f[['Date', 'primary_group']].copy()
    excludes['feature_incomplete_date'] = ~f.feature_eligible
    excludes['target_unavailable_date'] = ~t.target_available
    excludes['boundary_purged'] = t.boundary_purged
    excludes.loc[excludes.iloc[:, 2:].any(axis=1)].to_csv(out / 'model_exclusions.csv', index=False)
    evaluation = scored.loc[scored.target_available].copy()
    assert not evaluation.empty and evaluation.groupby('Date').size().eq(7).all()
    calendar = pd.DatetimeIndex(sorted(f.loc[f.Date.ge('2024-01-01'), 'Date'].unique()))
    cal_audit = f.loc[f.Date.ge('2024-01-01'), ['Date', 'feature_eligible']].drop_duplicates().merge(
        t.loc[t.Date.ge('2024-01-01'), ['Date', 'target_available']].drop_duplicates(), on='Date', validate='one_to_one')
    cal_audit.to_csv(out / 'validation_calendar.csv', index=False)
    groups = sorted(f.primary_group.unique())
    assert len(groups) == 7
    n, k = len(calendar), len(model_names)
    # Full original validation calendar includes missing-feature/target days as NA.
    daily_arrays = np.full((n, k + 1, 3), np.nan)
    probabilities = np.full((n, 7, k), np.nan)
    labels = np.full((n, 7), np.nan)
    di = {d: i for i, d in enumerate(calendar)}
    daily_rows = []
    for d, g in evaluation.groupby('Date', sort=True):
        g = g.set_index('primary_group').reindex(groups)
        r = g.subsector_forward_excess.to_numpy()
        labels[di[d]] = g.y
        for j, key in enumerate(model_names + ['momentum20']):
            p = g[key].to_numpy() if key != 'momentum20' else g.momentum_20.to_numpy()
            # groups are lexical; stable sorting breaks tied scores deterministically.
            chosen = np.argsort(-p, kind='stable')[:3]
            ic = float(np.corrcoef(rankdata(p), rankdata(r))[0, 1]) if np.std(p) and np.std(r) else np.nan
            vals = [ic, concordance(p, r), float(r[chosen].mean())]
            daily_arrays[di[d], j] = vals
            if j < k:
                probabilities[di[d], :, j] = p
            daily_rows.append({'Date': d, 'model': key, 'selected_count': 3, 'rank_ic': vals[0], 'pairwise_concordance': vals[1],
                'top3_excess_5d': vals[2], 'top3_return_5d': float(g.subsector_forward_return.iloc[chosen].mean()),
                'ai_benchmark_return_5d': float(g.ai_benchmark_forward_return.iloc[0]),
                'selected_groups': '|'.join(np.asarray(groups)[chosen])})
        daily_rows.append({'Date': d, 'model': 'equal7', 'selected_count': 7, 'rank_ic': np.nan, 'pairwise_concordance': .5,
            'top3_excess_5d': 0.0, 'top3_return_5d': float(g.ai_benchmark_forward_return.iloc[0]),
            'ai_benchmark_return_5d': float(g.ai_benchmark_forward_return.iloc[0]), 'selected_groups': '|'.join(groups)})
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(out / 'daily_diagnostics.csv', index=False)
    records = []
    for period in ['all', '2024', '2025']:
        e = evaluation if period == 'all' else evaluation.loc[evaluation.Date.dt.year.eq(int(period))]
        dd = daily if period == 'all' else daily.loc[daily.Date.dt.year.eq(int(period))]
        for key in model_names + ['momentum20', 'equal7', 'constant_half', 'constant_train_rate']:
            row = {'model': key, 'period': period, 'rows': len(e), 'dates': e.Date.nunique()}
            dsub = dd.loc[dd.model.eq(key)]
            row.update({c: float(dsub[c].mean()) for c in ['rank_ic', 'pairwise_concordance', 'top3_excess_5d', 'top3_return_5d']})
            if key not in ['momentum20', 'equal7']:
                p = e[key].to_numpy() if key in model_names else np.full(len(e), .5 if key == 'constant_half' else ytrain.mean())
                row.update(auc=float(roc_auc_score(e.y, p)), brier=float(brier_score_loss(e.y, p)),
                           accuracy=float(np.mean((p >= .5) == e.y.to_numpy())))
            records.append(row)
    metrics = pd.DataFrame(records)
    metrics.to_csv(out / 'metrics.csv', index=False)
    metric_names = ['rank_ic', 'pairwise_concordance', 'top3_excess_5d']
    point = np.nanmean(daily_arrays, axis=0)
    contrasts = [(kind + '_' + f, kind + '_F0') for kind in ['logistic', 'random_forest'] for f in ['F1', 'F2']]
    all_names = model_names + ['momentum20']
    ci_rows, auc_rows = [], []
    for block in [20, 5, 10, 40]:
        rng = np.random.default_rng(SEED + block)
        draws = np.empty((1000, k + 1, 3))
        auc_draws = np.empty((1000, k)) if block == 20 else None
        for rep in range(1000):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            ix = (starts[:, None] + np.arange(block)).ravel()[:n]
            draws[rep] = np.nanmean(daily_arrays[ix], axis=0)
            if block == 20:
                yy = labels[ix].ravel()
                pp = probabilities[ix].reshape(-1, k)
                good = np.isfinite(yy) & np.isfinite(pp).all(axis=1)
                yy, pp = yy[good], pp[good]
                pos = yy == 1
                np1, nn = pos.sum(), (~pos).sum()
                auc_draws[rep] = ((rankdata(pp, axis=0)[pos].sum(axis=0) - np1 * (np1 + 1) / 2) / (np1 * nn)) if np1 and nn else np.nan
        for a, key in enumerate(all_names):
            for z, metric in enumerate(metric_names):
                lo, hi = np.nanquantile(draws[:, a, z], [.025, .975])
                ci_rows.append(dict(model=key, comparator='', metric=metric, block_length=block, estimate=point[a, z], ci_low=lo, ci_high=hi))
        for key, baseline in contrasts + [(m, 'momentum20') for m in model_names]:
            a, b = all_names.index(key), all_names.index(baseline)
            for z, metric in enumerate(metric_names):
                lo, hi = np.nanquantile(draws[:, a, z] - draws[:, b, z], [.025, .975])
                ci_rows.append(dict(model=key, comparator=baseline, metric=metric, block_length=block,
                                    estimate=point[a, z] - point[b, z], ci_low=lo, ci_high=hi))
        if block == 20:
            for key, baseline in contrasts:
                a, b = model_names.index(key), model_names.index(baseline)
                lo, hi = np.nanquantile(auc_draws[:, a] - auc_draws[:, b], [.025, .975])
                mm = metrics.loc[metrics.period.eq('all')].set_index('model')
                auc_rows.append(dict(model=key, comparator=baseline, metric='pooled_auc', block_length=20,
                    estimate=mm.loc[key, 'auc'] - mm.loc[baseline, 'auc'], ci_low=lo, ci_high=hi))
        print('Completed paired bootstrap block=' + str(block), flush=True)
    ci = pd.DataFrame(ci_rows + auc_rows)
    ci.to_csv(out / 'bootstrap_intervals.csv', index=False)
    ci.loc[ci.comparator.ne('')].to_csv(out / 'paired_comparisons.csv', index=False)
    summary = {'status': 'completed_exploratory_rotation_diagnostic_not_fund_backtest',
       'completed_utc': datetime.now(timezone.utc).isoformat(),
       'panel_dir': str(args.panel_dir.resolve()), 'hashes': {'features': sha(fp), 'targets': sha(tp), 'runner': sha(__file__),
           'protocol': sha(Path(__file__).resolve().parent / 'ROTATION_MACRO_TRIAL_V3_PROTOCOL_20260913.md')},
       'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
       'counts': {'training_rows': int(fit_mask.sum()), 'training_dates': int(f.loc[fit_mask, 'Date'].nunique()),
                  'validation_prediction_rows': len(pred), 'validation_prediction_dates': pred.Date.nunique(),
                  'evaluation_rows': len(evaluation), 'evaluation_dates': evaluation.Date.nunique(),
                  'validation_calendar_dates': n, 'target_missing_prediction_rows': int((~scored.target_available).sum())},
       'training_positive_rate': float(ytrain.mean()), 'model_parameters': parameters, 'artifact_hashes': artifact_hashes,
       'fit_key_hashes': fit_key_hashes,
       'feature_groups': names, 'metrics': metrics.loc[metrics.period.eq('all')].replace({np.nan: None}).to_dict('records'),
       'metrics_by_period': metrics.replace({np.nan: None}).to_dict('records'),
       'tie_policy': 'portfolio ties lexical by group; rankIC average ranks; pairwise predicted tie half-credit excluding realized ties',
       'return_column_note': 'top3_return/excess columns represent selected strategy returns; equal7 row holds all7 with zero self-excess, selected_count=7',
       'macro_scaler_fit': 'all ex-ante feature-eligible training dates; final model pipeline fit on common supervised purged training sample',
       'bootstrap': {'type': 'paired_moving_blocks_on_full_validation_calendar', 'primary_length': 20, 'sensitivity_lengths': [5, 10, 40], 'reps': 1000, 'seed_base': SEED},
       'checks': {'no_2026_dates': bool(f.Date.lt('2026-01-01').all()), 'training_exit_before_validation': bool(t.loc[fit_mask, 'target_end_session'].lt('2024-01-01').all()),
                  'common_training_keys': len(set(fit_key_hashes.values())) == 1, 'seven_groups_every_evaluation_date': bool(evaluation.groupby('Date').size().eq(7).all()),
                  'prediction_keys_from_features_only': pred[['Date', 'primary_group']].reset_index(drop=True).equals(f.loc[pred_mask, ['Date', 'primary_group']].reset_index(drop=True)),
                  'probabilities_finite': bool(np.isfinite(pred[model_names]).all().all())},
       'limitations': ['Static retrospective candidate membership, source corporate-action and historical data-vintage limitations remain.',
                      'Common complete-case dates omit missing features/targets; informative missingness may bias evaluation.',
                      '2024-2025 already used for development; all six configurations are exploratory.',
                      'Five-day top3 diagnostics overlap and are gross, not executable fund NAV or net returns.',
                      'Pointwise paired bootstrap intervals do not correct the whole model-search family.']}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int), encoding='utf-8')
    print(json.dumps({'output': str(out), 'counts': summary['counts'], 'metrics': summary['metrics']}, ensure_ascii=False, indent=2, default=int), flush=True)


if __name__ == '__main__':
    main()
