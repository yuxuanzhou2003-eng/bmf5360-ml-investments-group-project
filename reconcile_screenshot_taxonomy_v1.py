"""Preserve source role rows and conservatively reconcile existing symbols."""
import argparse, csv, hashlib, json, re, shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
TRANSCRIPTION = ROOT / 'screenshot_ai_structure_20260913.json'
REGISTRY = ROOT / 'data/audit/ai_pool_expansion_v1/20260910T045000Z/candidate_registry.csv'


def save_csv(path, rows):
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-dir', type=Path, required=True)
    out = ap.parse_args().audit_dir
    assert out.exists() and not (out / 'summary.json').exists()
    doc = json.loads(TRANSCRIPTION.read_text(encoding='utf-8'))
    with REGISTRY.open(encoding='utf-8-sig') as f:
        registry = list(csv.DictReader(f))
    by_symbol = defaultdict(list)
    for r in registry:
        by_symbol[r['ric'].split('.')[0]].append(r)
    rows, entities = [], {}
    for g in doc['groups']:
        for sub in g['subgroups']:
            for c in sub['companies']:
                raw = c['ticker_raw'].strip()
                private = raw.lower() == 'private'
                key = 'private_unverified:' + c['name'].lower() if private else 'source_symbol:' + raw.upper()
                matches = by_symbol.get(raw, []) if re.fullmatch(r'[A-Z]{1,6}', raw) and c['confidence'] != 'low' else []
                ric = matches[0]['ric'] if len(matches) == 1 else ''
                status = 'local_registry_symbol_match_only' if ric else 'image_private_claim_unverified' if private else 'identifier_and_listing_verification_pending'
                row = dict(source_role_id=len(rows) + 1, source_group=g['name'], source_subgroup=sub['name'],
                    company_name_raw=c['name'], ticker_raw=raw, confidence=c['confidence'], notes=c['notes'],
                    entity_key=key, matched_local_ric=ric, match_status=status)
                rows.append(row)
                e = entities.setdefault(key, dict(entity_key=key, name_raw=c['name'], ticker_raw=raw,
                    matched_local_ric=ric, match_status=status, roles=[], source_occurrences=0))
                e['roles'].append(g['name'] + ' > ' + sub['name'])
                e['source_occurrences'] += 1
    entity_rows = [{**e, 'roles': ' | '.join(e['roles'])} for e in entities.values()]
    save_csv(out / 'role_memberships.csv', rows)
    save_csv(out / 'candidate_entities.csv', entity_rows)
    save_csv(out / 'identifier_review_queue.csv', [e for e in entity_rows if not e['matched_local_ric']])
    matched = {r['matched_local_ric'] for r in rows if r['matched_local_ric']}
    old = [{'ric': r['ric'], 'canonical_name': r['canonical_name'], 'old_primary_group': r['primary_group'],
            'screenshot_status': 'visible_symbol_match' if r['ric'] in matched else 'not_visible_in_current_crop',
            'action': 'preserve_existing_data; no membership or return changes'} for r in registry]
    save_csv(out / 'old_registry_reconciliation.csv', old)
    shutil.copy2(TRANSCRIPTION, out / 'transcription_reviewed.json')
    report = {'status': 'source_structure_adopted_identifier_audit_only', 'primary_groups': len(doc['groups']),
        'subgroups_visible': sum(len(g['subgroups']) for g in doc['groups']), 'source_role_rows': len(rows),
        'provisional_entity_keys': len(entity_rows), 'existing_registry_rows': len(registry), 'matched_existing_entities': len(matched),
        'old_not_visible_in_crop': len(registry) - len(matched), 'pending_entity_keys': sum(not e['matched_local_ric'] for e in entity_rows),
        'private_label_keys_unverified': sum(e['match_status'] == 'image_private_claim_unverified' for e in entity_rows),
        'low_confidence_role_rows': sum(r['confidence'] == 'low' for r in rows), 'multi_role_entity_keys': sum(e['source_occurrences'] > 1 for e in entity_rows),
        'hashes': {'image': sha(out / 'source_image.jpg'), 'transcription_reviewed': sha(TRANSCRIPTION), 'registry': sha(REGISTRY), 'script': sha(__file__)},
        'checks': {'seven_columns': len(doc['groups']) == 7, 'all_source_roles_retained': len(rows) == sum(len(sub['companies']) for g in doc['groups'] for sub in g['subgroups']),
                   'all_old_registry_retained': len(old) == len(registry), 'unique_entity_keys': len(entities) == len(entity_rows)},
        'processing': {'financial_data_rows_modified': 0, 'companies_deleted': 0, 'new_ric_inventions': 0, 'new_price_fetches': 0, 'portfolio_weights_changed': 0},
        'limitations': ['Entity keys are provisional source-symbol/name groupings, not verified issuer/legal-entity IDs.', 'Private status is image text only.', 'Current crop is incomplete; absence never means ineligible.', 'Exact local symbol match does not certify current listing or historical coverage.']}
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
