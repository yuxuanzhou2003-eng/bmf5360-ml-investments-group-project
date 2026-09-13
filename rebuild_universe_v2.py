"""Local-only backward reconstruction of the S&P 500 membership timeline. Reads universe_probe, writes universe_rebuild. No network, no edits to existing tables."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'data' / 'raw' / 'universe_probe'
OUT = ROOT / 'data' / 'audit' / 'universe_rebuild'
OUT.mkdir(parents=True, exist_ok=True)

anchor_meta = json.loads((SRC / 'spx_current_anchor.meta.json').read_text(encoding='utf-8'))
anchor_date = pd.Timestamp(anchor_meta['retrieved_at_utc']).tz_convert(None).normalize()
anchor = pd.read_csv(SRC / 'spx_current_anchor.csv')
anchor_set = set(anchor['Constituent RIC'].dropna())

frames = []
for path in sorted(SRC.glob('spx_jl_*.csv')):
    df = pd.read_csv(path)
    df['source_file'] = path.name
    frames.append(df)
changes = pd.concat(frames, ignore_index=True)
changes = changes.rename(columns={'Date': 'change_date', 'Constituent RIC': 'ric', 'Constituent Name': 'name', 'Change': 'direction'})
changes['change_date'] = pd.to_datetime(changes['change_date'], format='mixed', errors='coerce').dt.tz_localize(None).dt.normalize()
before = len(changes)
changes = changes.dropna(subset=['change_date', 'ric', 'direction'])
changes = changes.drop_duplicates(['change_date', 'ric', 'direction']).sort_values('change_date')
report = {'anchor_date': str(anchor_date.date()), 'anchor_size': len(anchor_set),
          'change_rows_raw': before, 'change_rows_used': len(changes),
          'change_date_min': str(changes.change_date.min().date()), 'change_date_max': str(changes.change_date.max().date()),
          'direction_counts': changes.direction.value_counts().to_dict()}

current = set(anchor_set)
anomalies = []
same_day_roundtrips = []
segments = [{'valid_from': changes.change_date.max(), 'valid_to': anchor_date, 'size': len(current), 'members': sorted(current)}]
for day, group in sorted(changes.groupby('change_date'), key=lambda kv: kv[0], reverse=True):
    # Treat each day's changes atomically. Join+leave for the same RIC is
    # net zero at daily resolution; sequential mutations depend on row order.
    joins = set(group.loc[group.direction == 'Joiner', 'ric'])
    leaves = set(group.loc[group.direction == 'Leaver', 'ric'])
    for ric in sorted(joins & leaves):
        same_day_roundtrips.append({'date': str(day.date()), 'ric': ric,
                                   'in_after_day': ric in current,
                                   'rule': 'same_day_join_and_leave_net_zero'})
    for ric in sorted(joins - leaves):
        if ric not in current:
            anomalies.append({'date': str(day.date()), 'ric': ric, 'direction': 'Joiner', 'issue': 'joiner_not_present_before_rollback'})
    for ric in sorted(leaves - joins):
        if ric in current:
            anomalies.append({'date': str(day.date()), 'ric': ric, 'direction': 'Leaver', 'issue': 'leaver_already_present_before_rollback'})
    current = (current - (joins - leaves)) | (leaves - joins)
    for row in group.loc[~group.direction.isin(['Joiner', 'Leaver'])].itertuples():
        anomalies.append({'date': str(day.date()), 'ric': row.ric, 'direction': str(row.direction), 'issue': 'unknown_direction'})
    prior = sorted(changes.loc[changes.change_date < day, 'change_date'].unique())
    segments.append({'valid_from': prior[-1] if prior else pd.Timestamp('1900-01-01'), 'valid_to': day - pd.Timedelta(days=1), 'size': len(current), 'members': sorted(current)})

segments = sorted(segments, key=lambda s: s['valid_from'])
timeline = pd.DataFrame([{k: v for k, v in s.items() if k != 'members'} for s in segments])
timeline.to_csv(OUT / 'membership_segments.csv', index=False)

def members_on(date):
    date = pd.Timestamp(date)
    for seg in segments:
        if seg['valid_from'] <= date <= seg['valid_to']:
            return seg['members']
    return None

probe_points = [f'{y}-06-30' for y in range(2010, 2027)]
sizes = []
for point in probe_points:
    m = members_on(point)
    sizes.append({'date': point, 'size': len(m) if m is not None else None,
                  'delisted_suffix': sum('^' in x for x in m) if m else None})
sizes_df = pd.DataFrame(sizes)
sizes_df.to_csv(OUT / 'size_by_year.csv', index=False)

universe_union = set()
for seg in segments: universe_union |= set(seg['members'])
suffixed = {x for x in universe_union if '^' in x}
roots = {}
for ric in universe_union:
    root = ric.split('^')[0].split('.')[0]
    roots.setdefault(root, set()).add(ric)
multi = {k: sorted(v) for k, v in roots.items() if len(v) > 1}

report.update({'segments': len(segments), 'anomaly_count': len(anomalies),
               'same_day_roundtrip_count': len(same_day_roundtrips),
               'union_distinct_rics': len(universe_union), 'delisted_suffix_rics': len(suffixed),
               'same_root_multi_ric_groups': len(multi)})
pd.DataFrame(anomalies, columns=['date','ric','direction','issue']).to_csv(OUT / 'rollback_anomalies.csv', index=False)
pd.DataFrame(same_day_roundtrips, columns=['date','ric','in_after_day','rule']).to_csv(OUT / 'same_day_roundtrips.csv', index=False)
(OUT / 'same_root_groups.json').write_text(json.dumps(multi, indent=2), encoding='utf-8')
pd.DataFrame(sorted(universe_union), columns=['ric']).to_csv(OUT / 'union_universe.csv', index=False)
(OUT / 'rebuild_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
print()
print(sizes_df.to_string(index=False))

# Per-RIC membership intervals: a company can leave and rejoin, so one RIC may have several rows.
runs = {}
intervals = []
ordered = sorted(segments, key=lambda s: s['valid_from'])
prev_members = set()
for seg in ordered:
    members = set(seg['members'])
    for ric in members - prev_members:
        runs[ric] = seg['valid_from']
    for ric in prev_members - members:
        intervals.append({'ric': ric, 'start': runs.pop(ric), 'end': seg['valid_from'] - pd.Timedelta(days=1)})
    prev_members = members
for ric, start in runs.items():
    intervals.append({'ric': ric, 'start': start, 'end': ordered[-1]['valid_to']})
iv = pd.DataFrame(intervals).sort_values(['ric', 'start'])
iv.to_csv(OUT / 'membership_intervals.csv', index=False)
multi_spell = iv.ric.value_counts()
print()
print('membership intervals:', len(iv), '| RICs with >1 spell:', int((multi_spell > 1).sum()))
print(multi_spell[multi_spell > 1].head(10).to_string())
