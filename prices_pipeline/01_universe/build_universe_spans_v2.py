"""Local-only: derive per-RIC S&P 500 membership spans for the 2015-2026 study window."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT/'data'/'raw'/'universe_probe'
OUT = ROOT/'data'/'audit'/'universe_rebuild'
STUDY_START, STUDY_END = pd.Timestamp('2015-01-01'), pd.Timestamp('2026-09-07')
WARMUP_DAYS = 400   # return history needed before a company's first in-window event

anchor_meta = json.loads((SRC/'spx_current_anchor.meta.json').read_text(encoding='utf-8'))
anchor_date = pd.Timestamp(anchor_meta['retrieved_at_utc']).tz_convert(None).normalize()
anchor_set = set(pd.read_csv(SRC/'spx_current_anchor.csv')['Constituent RIC'].dropna())

frames = [pd.read_csv(p) for p in sorted(SRC.glob('spx_jl_*.csv'))]
ch = pd.concat(frames, ignore_index=True).rename(columns={'Date':'d','Constituent RIC':'ric','Constituent Name':'name','Change':'dir'})
ch['d'] = pd.to_datetime(ch.d, format='mixed', errors='coerce').dt.tz_localize(None).dt.normalize()
ch = ch.dropna(subset=['d','ric','dir']).drop_duplicates(['d','ric','dir']).sort_values('d')

# Rebuild segments backwards from the anchor, same algorithm as rebuild_universe_v2.
current = set(anchor_set)
segments = [(ch.d.max(), anchor_date, set(current))]
for day, grp in sorted(ch.groupby('d'), key=lambda kv: kv[0], reverse=True):
    joins = set(grp.loc[grp.dir == 'Joiner', 'ric'])
    leaves = set(grp.loc[grp.dir == 'Leaver', 'ric'])
    current = (current - (joins - leaves)) | (leaves - joins)
    prior = ch.loc[ch.d < day, 'd']
    segments.append((prior.max() if len(prior) else pd.Timestamp('1900-01-01'), day - pd.Timedelta(days=1), set(current)))
segments = sorted(segments, key=lambda s: s[0])

# Per-RIC membership span, clipped to the study window.
spans = {}
for lo, hi, members in segments:
    lo_c, hi_c = max(lo, STUDY_START), min(hi, STUDY_END)
    if lo_c > hi_c: continue
    for ric in members:
        if ric in spans:
            spans[ric][0] = min(spans[ric][0], lo_c); spans[ric][1] = max(spans[ric][1], hi_c)
        else:
            spans[ric] = [lo_c, hi_c]

names = ch.drop_duplicates('ric').set_index('ric')['name'].to_dict()
rows = []
for ric, (lo, hi) in sorted(spans.items()):
    fetch_start = (lo - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
    fetch_end = min(hi + pd.Timedelta(days=10), STUDY_END).strftime('%Y-%m-%d')
    rows.append({'ric': ric, 'name': names.get(ric, ''), 'delisted_ric': '^' in ric,
                 'member_from': lo.date(), 'member_to': hi.date(),
                 'member_days': (hi - lo).days, 'fetch_start': fetch_start, 'fetch_end': fetch_end})
df = pd.DataFrame(rows)
# The original spans file is the immutable collection manifest: changing its
# ordering would change cached batch names. Emit revised eligibility separately.
df.to_csv(OUT/'eligible_spans_2015_2026_corrected.csv', index=False)

est_sessions = (df.member_days + WARMUP_DAYS).clip(upper=(STUDY_END-pd.Timestamp('2014-01-01')).days) * 252/365
summary = {'study_window': f'{STUDY_START.date()} to {STUDY_END.date()}', 'warmup_days': WARMUP_DAYS,
           'distinct_rics': len(df), 'delisted_rics': int(df.delisted_ric.sum()), 'live_rics': int((~df.delisted_ric).sum()),
           'median_member_days': int(df.member_days.median()), 'full_window_members': int((df.member_days > 4000).sum()),
           'est_return_rows': int(est_sessions.sum()),
           'est_quarterly_events': int((est_sessions/252*4).sum()),
           'est_weekly_estimate_rows': int((est_sessions/252*52*2).sum())}
(OUT/'eligible_spans_summary_corrected.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
print(json.dumps(summary, indent=2, default=str))
print()
print('span 分布:'); print(df.member_days.describe().round(0).to_string())
print()
print('样例（退市股）:'); print(df[df.delisted_ric].head(6).to_string(index=False))
