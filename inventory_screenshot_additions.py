"""Run-scoped local coverage extraction; no filling or model changes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import pandas as pd

root=Path(__file__).resolve().parent
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=root/'data/audit/screenshot_additions'/stamp
out.mkdir(parents=True)
queue=root/'data/audit/screenshot_taxonomy_v1/20260913025822Z/identifier_review_queue.csv'
spans=root/'data/audit/universe_rebuild/eligible_spans_2015_2026_corrected.csv'
q=pd.read_csv(queue).fillna(''); s=pd.read_csv(spans)
mapping={r.split('.')[0]:r for r in s.ric if r in ['AES.N','AME.N','BWA.N','CAT.N','CMI.N','EMR.N','GEV.N','GNRC.N','SMCI.OQ']}
added=q[q.ticker_raw.isin(mapping)].copy()
added['matched_local_ric']=added.ticker_raw.map(mapping)
added['match_status']='local_registry_evidence_pending_vendor_refresh'
added.to_csv(out/'additional_candidates.csv',index=False)
q[~q.ticker_raw.isin(mapping)].to_csv(out/'remaining_identifier_queue.csv',index=False)
coverage=[]; sources=[]
for kind,rel,field in [('returns','data/clean/v2/returns.csv','return_decimal'),('prices','data/clean/v3/20260909T012417705069Z/prices.csv','TRDPRC_1')]:
    path=root/rel; chunks=[]; scanned=0
    for c in pd.read_csv(path,chunksize=100000):
        scanned+=len(c)
        chunks.append(c[c.Instrument.isin(mapping.values()) & c.Date.between('2020-01-01','2025-12-31')].copy())
    frame=pd.concat(chunks,ignore_index=True)
    frame.to_csv(out/f'{kind}_local_2020_2025.csv',index=False)
    for ric in mapping.values():
        f=frame[frame.Instrument.eq(ric)]
        coverage.append(dict(kind=kind,ric=ric,rows=len(f),first_date=f.Date.min(),last_date=f.Date.max(),missing_values=int(f[field].isna().sum()),duplicate_instrument_dates=int(f.duplicated(['Instrument','Date']).sum())))
    sources.append(dict(path=rel,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest(),source_rows=scanned,retained_rows=len(frame)))
pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
summary=dict(run_utc=stamp,queue_before=len(q),added_candidates=len(added),queue_remaining=len(q)-len(added),sources=sources,rules='Exact local RIC evidence; 2020 warmup through 2025 only. No imputation, deduplication, deletion of originals or model execution. Membership intervals are not listing dates. Raw source provenance retained in extracted rows.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
entry=f'\n\n## Screenshot candidate local coverage — {stamp}\nExecuted inventory_screenshot_additions.py. Output: {out.relative_to(root)}. Exact-match supplementation: {len(q)} pending entities -> {len(added)} locally evidenced candidates + {len(q)-len(added)} pending. See summary.json for source/script hashes and scanned/retained counts; coverage.csv for per-instrument dates, missing values and duplicates. Extracted only 2020–2025 dates; other source rows remain unchanged in source files, not quarantined or deleted. No filling, winsorization, deduplication, membership-date reinterpretation, or test-label access. Local coverage is not a passed model-readiness audit; vendor identity refresh and corporate-action checks remain.\n'
with (root/'DATA_PROCESSING_LOG.md').open('a',encoding='utf-8') as f:f.write(entry)
with (root/'AI_USE_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n{stamp}: Codex implemented and executed screenshot candidate local coverage supplement; Luna performed read-only inventory, root verified against broader local registry. No new model results. Output {out.relative_to(root)}.\n')
print(out)
print(pd.DataFrame(coverage).to_string(index=False))
