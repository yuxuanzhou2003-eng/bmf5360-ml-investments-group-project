import csv, json
from pathlib import Path
from collections import defaultdict

run = Path('data/panel_v2/20260908T053814601256Z')
out = Path('data/audit/panel_v2/20260908T111211716224Z')
out.mkdir(parents=True, exist_ok=True)
def rows(name):
    with (run/name).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
ev, ft, lb, dx = rows('events.csv'), rows('features.csv'), rows('labels.csv'), rows('diagnostics_ex_post.csv')
def truth(x): return str(x).strip().lower() == 'true'
matched = sum(r.get('status') == 'matched_snapshot' for r in ev)
graphready = sum(r.get('graph_status') == 'available' for r in ev)
zero = sum(r.get('graph_status') == 'available' and float(r.get('neighbor_count') or 0) == 0 for r in ev)
eligible = sum(r.get('graph_status') == 'available' and float(r.get('neighbor_count') or 0) >= 1 for r in ev)
neg = sum(float(r['residual_correlation']) < 0 for r in ft if r.get('residual_correlation') not in ('', None))
pos = sum(float(r['residual_correlation']) >= 0 for r in ft if r.get('residual_correlation') not in ('', None))
fmap = {r['sample_id']: r for r in ft}
incomp = [r for r in lb if not truth(r.get('label_complete'))]
receivers = sorted({fmap[r['sample_id']].get('receiver','') for r in incomp if r['sample_id'] in fmap})
overlap = defaultdict(int)
for r in dx:
    v = r.get('ex_post_own_announcement_overlap','').strip().lower()
    overlap['true' if v == 'true' else 'false' if v == 'false' else 'unknown'] += 1
result = {'run':'20260908T053814601256Z','events':{'rows':len(ev),'matched_snapshot':matched,'graph_ready':graphready,'insufficient_history':matched-graphready,'graph_ready_zero_neighbor':zero,'eligible_ge_1_edge':eligible},'features':{'rows':len(ft),'negative_correlation_edges':neg,'nonnegative_correlation_edges':pos},'labels':{'rows':len(lb),'incomplete_rows':len(incomp),'incomplete_unique_receiver_ric':receivers,'incomplete_receiver_ric_count':len(receivers),'incomplete_receiver_ric_has_caret':sum('^' in x for x in receivers)},'ex_post_overlap':dict(overlap),'notes':['Counts are read-only and retrospective.','Unknown overlap means blank or value other than True/False.']}
(out/'output_profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
md = ['# Panel v2 输出计数审计（回顾性）','',f"输入 run：`{result['run']}`",'', '| 项目 | 数量 |','|---|---:|']
for k,v in [('events 总行数',len(ev)),('matched_snapshot',matched),('graph_ready',graphready),('matched 但 graph 不就绪',matched-graphready),('graph_ready 且零邻居',zero),('eligible（至少1条边）',eligible),('features 边行数',len(ft)),('负相关边',neg),('非负相关边',pos),('labels 不完整行',len(incomp)),('不完整标签唯一 receiver RIC',len(receivers)),('其中含 `^`',result['labels']['incomplete_receiver_ric_has_caret'])]: md.append(f'| {k} | {v} |')
md += ['', '## 不完整标签 receiver RIC', '', '、'.join(receivers) if receivers else '无', '', '## ex-post own-announcement overlap', '', 'True：{0}；False：{1}；Unknown：{2}。Unknown 为缺失或非 True/False 值。'.format(overlap['true'],overlap['false'],overlap['unknown']), '', '未删除、填补、去重或改写任何源数据。']
(out/'output_profile.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
