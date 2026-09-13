"""Round-trip model predictions and independent paired block check."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
P = ROOT / 'data/model_ready_rotation_macro_v3/20260913021922Z'
R = ROOT / 'data/model_runs/rotation_macro_v3/20260913021922Z'
A = ROOT / 'data/audit/rotation_macro_v3/20260913021922Z'
f = pd.read_csv(P / 'features.csv', parse_dates=['Date']).sort_values(['Date', 'primary_group']).reset_index(drop=True)
t = pd.read_csv(P / 'targets.csv', parse_dates=['Date']).sort_values(['Date', 'primary_group']).reset_index(drop=True)
p = pd.read_csv(R / 'validation_predictions.csv', parse_dates=['Date'])
d = pd.read_csv(R / 'daily_diagnostics.csv', parse_dates=['Date'])
s = json.loads((R / 'summary.json').read_text(encoding='utf-8'))
predmask = f.feature_eligible & f.Date.ge('2024-01-01')
checks = {'prediction_keys_ex_ante': f.loc[predmask, ['Date', 'primary_group']].reset_index(drop=True).equals(p[['Date', 'primary_group']]),
          'input_features_hash_matches': hashlib.sha256((P / 'features.csv').read_bytes()).hexdigest() == s['hashes']['features'],
          'input_targets_hash_matches': hashlib.sha256((P / 'targets.csv').read_bytes()).hexdigest() == s['hashes']['targets']}
for key, expected_hash in s['artifact_hashes'].items():
    checks['artifact_hash_' + key] = hashlib.sha256((R / 'models' / key).read_bytes()).hexdigest() == expected_hash
    obj = joblib.load(R / 'models' / key)
    tech = f.loc[predmask, [c + '_csz' for c in obj['tech_columns']]].to_numpy()
    macro = obj['macro_scaler'].transform(f.loc[predmask, obj['macro_columns']])
    inter = np.column_stack([macro[:, a] * tech[:, b] for a, b in obj['interactions']])
    name = key.removesuffix('.joblib')
    x = tech if name.endswith('F0') else np.column_stack([tech, macro]) if name.endswith('F1') else np.column_stack([tech, macro, inter])
    reproduced = obj['pipeline'].predict_proba(x)[:, 1]
    checks['reproduced_' + name] = bool(np.allclose(reproduced, p[name], atol=1e-12))
    checks['fit_count_' + name] = int(obj['pipeline'][0].n_samples_seen_) == 4760
scored = p.merge(t, on=['Date', 'primary_group'], validate='one_to_one')
scored = scored.loc[scored.target_available]
metrics = pd.read_csv(R / 'metrics.csv')
for name in s['model_parameters']:
    row = metrics.loc[metrics.model.eq(name) & metrics.period.eq('all')].iloc[0]
    checks['auc_' + name] = bool(np.isclose(roc_auc_score(scored.y, scored[name]), row.auc))
    ic = scored.groupby('Date').apply(lambda g: g[name].corr(g.subsector_forward_excess, method='spearman'), include_groups=False).mean()
    checks['ic_' + name] = bool(np.isclose(ic, row.rank_ic))
calendar = pd.DatetimeIndex(sorted(f.loc[f.Date.ge('2024-01-01'), 'Date'].unique()))
pivot = d.pivot(index='Date', columns='model', values='top3_excess_5d').reindex(calendar)
delta = (pivot.random_forest_F1 - pivot.random_forest_F0).to_numpy()
rng = np.random.default_rng(5380)
means = []
for _ in range(1000):
    starts = rng.integers(0, len(calendar) - 20 + 1, size=int(np.ceil(len(calendar) / 20)))
    indices = np.concatenate([np.arange(x, x + 20) for x in starts])[:len(calendar)]
    means.append(np.nanmean(delta[indices]))
ci = np.quantile(means, [.025, .975])
boot = pd.read_csv(R / 'paired_comparisons.csv')
ref = boot.loc[boot.model.eq('random_forest_F1') & boot.comparator.eq('random_forest_F0') & boot.metric.eq('top3_excess_5d') & boot.block_length.eq(20)].iloc[0]
checks['independent_paired_block_ci'] = bool(np.allclose(ci, [ref.ci_low, ref.ci_high], atol=1e-12))
checks['label_missing_predictions_retained'] = len(pd.read_csv(R / 'predictions_without_evaluable_target.csv')) == 84
result = {'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
          'scope': 'Saved-model probability reproduction, source/artifact hashes, AUC and daily IC reproduction, one independently recomputed paired block CI; no upstream economic-data certification.',
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(A / 'model_artifact_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
assert all(checks.values())
