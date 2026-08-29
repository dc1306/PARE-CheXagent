"""Run safe text merge + evaluation on existing PARE reports."""
import json, sys, re, numpy as np
from collections import Counter
sys.path.insert(0, '/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/canonical_baseline')

from safe_merger import safe_merge
from f1chexbert import F1CheXbert

scorer = F1CheXbert(device='cuda:0')
print("Scorer loaded", flush=True)

BL_PATH = '/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/canonical_baseline/chexagent_8b_reports.json'
PARE_PATH = '/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/pare_standardized/pare_test_reports.json'
OUT_DIR = '/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/pare_standardized'

bl = json.load(open(BL_PATH))
pare = json.load(open(PARE_PATH))
bl_by = {r['study_id']: r for r in bl}
pare_by = {r['study_id']: r for r in pare}

clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip()

# ── Step 1: Build merged TEXT reports ──
merged_reports = []
all_actions = Counter()

for i, p in enumerate(pare):
    sid = p['study_id']
    b = bl_by[sid]
    bl_text = clean(b.get('candidate_findings', ''))

    if p.get('steered') and p.get('targets'):
        st_text = clean(p.get('candidate_findings', ''))
        merged_text, actions = safe_merge(bl_text, st_text, p['targets'], scorer)
        for act in actions.values():
            all_actions[act] += 1
    else:
        merged_text = bl_text
        actions = {}

    merged_reports.append({
        'study_id': sid,
        'candidate_findings': merged_text,
        'steered': p.get('steered', False),
        'actions': actions,
        'section_findings': p.get('section_findings', '')
    })

    if (i+1) % 200 == 0:
        print(f"  Merged {i+1}/{len(pare)}", flush=True)

print(f"\nMerger actions: {json.dumps(dict(all_actions), indent=2)}", flush=True)

json.dump(merged_reports, open(f'{OUT_DIR}/pare_merged_reports.json', 'w'), indent=1)
print(f"Saved {len(merged_reports)} merged reports", flush=True)

# ── Step 2: Evaluate ──
ref_studies = [r for r in merged_reports if r.get('section_findings')]
print(f"\nEvaluating {len(ref_studies)} studies with references...", flush=True)

refs = [clean(r['section_findings']).lower() for r in ref_studies]
bl_cands = [clean(bl_by[r['study_id']]['candidate_findings']).lower() for r in ref_studies]

pare_cands = []
for r in ref_studies:
    pr = pare_by[r['study_id']]
    if pr.get('steered'):
        pare_cands.append(clean(pr['candidate_findings']).lower())
    else:
        pare_cands.append(clean(bl_by[r['study_id']]['candidate_findings']).lower())

mg_cands = [clean(r['candidate_findings']).lower() for r in ref_studies]

def to_json(obj):
    if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return float(obj)
    return obj

print("  Scoring baseline...", flush=True)
bl_s = to_json(scorer(refs, bl_cands))
print("  Scoring PARE...", flush=True)
pa_s = to_json(scorer(refs, pare_cands))
print("  Scoring merged-text...", flush=True)
mg_s = to_json(scorer(refs, mg_cands))

for name, s in [("Baseline", bl_s), ("PARE", pa_s), ("Merged-Text", mg_s)]:
    if isinstance(s, list) and len(s) >= 2 and isinstance(s[1], dict):
        d = s[1]
        mi = d.get('micro avg',{}).get('f1-score', 0)
        ma = d.get('macro avg',{}).get('f1-score', 0)
        print(f"  {name}: Micro-F1={mi:.4f}, Macro-F1={ma:.4f}")

json.dump({'baseline': bl_s, 'pare': pa_s, 'merged_text': mg_s, 'actions': dict(all_actions)},
          open(f'{OUT_DIR}/pare_merged_eval.json','w'), indent=2)
print(f"\nDone. Saved to {OUT_DIR}/pare_merged_eval.json", flush=True)
