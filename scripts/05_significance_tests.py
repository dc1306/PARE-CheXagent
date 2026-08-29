"""Statistical significance tests + full per-pathology metrics for PARE."""
import json, numpy as np, re, sys
sys.stdout = open('/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/canonical_baseline/significance.out', 'w')
sys.stderr = sys.stdout

from f1chexbert import F1CheXbert
from scipy import stats

scorer = F1CheXbert(device='cuda:0')
print("Scorer loaded", flush=True)

BL = json.load(open('/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/canonical_baseline/chexagent_8b_reports.json'))
PARE = json.load(open('/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/pare_standardized/pare_test_reports.json'))
MERGED = json.load(open('/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/pare_standardized/pare_merged_reports.json'))

bl_by = {r['study_id']:r for r in BL}
pare_by = {r['study_id']:r for r in PARE}
mg_by = {r['study_id']:r for r in MERGED}

clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**","")).strip().lower()

ref_studies = [m for m in MERGED if m.get('section_findings')]
print(f"Studies with ref: {len(ref_studies)}", flush=True)

LABELS = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
          "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
          "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
TARG5 = ['Atelectasis','Cardiomegaly','Lung Opacity','Pleural Effusion','Edema']

def label_all(texts, name):
    out = []
    for i, t in enumerate(texts):
        if not t.strip():
            out.append([0]*14); continue
        raw = scorer.get_label(t, mode='rrg')
        out.append([int(raw[j]) if j < len(raw) and raw[j] != '' else 0 for j in range(14)])
        if (i+1) % 500 == 0: print(f"  {name}: {i+1}/{len(texts)}", flush=True)
    return np.array(out)

print("Labeling refs...", flush=True)
ref_labels = label_all([clean(s['section_findings']) for s in ref_studies], "ref")
print("Labeling baseline...", flush=True)
bl_labels = label_all([clean(bl_by[s['study_id']]['candidate_findings']) for s in ref_studies], "bl")
print("Labeling PARE...", flush=True)
pa_texts = []
for s in ref_studies:
    pr = pare_by[s['study_id']]
    if pr.get('steered'):
        pa_texts.append(clean(pr['candidate_findings']))
    else:
        pa_texts.append(clean(bl_by[s['study_id']]['candidate_findings']))
pa_labels = label_all(pa_texts, "pare")
print("Labeling merged...", flush=True)
mg_labels = label_all([clean(mg_by[s['study_id']]['candidate_findings']) for s in ref_studies], "mg")

N = len(ref_studies)

# ── Full metrics ──
def full_metrics(pred, ref, name):
    print(f"\n{'='*120}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  {'Path':<30} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Sens':>8} {'Spec':>8} {'Acc':>8} {'F1':>8}", flush=True)
    print(f"  {'-'*110}", flush=True)
    r = {}
    for j, l in enumerate(LABELS):
        tp=int(((ref[:,j]==1)&(pred[:,j]==1)).sum())
        fp=int(((ref[:,j]==0)&(pred[:,j]==1)).sum())
        fn=int(((ref[:,j]==1)&(pred[:,j]==0)).sum())
        tn=int(((ref[:,j]==0)&(pred[:,j]==0)).sum())
        se=tp/(tp+fn) if(tp+fn)>0 else 0; sp=tn/(tn+fp) if(tn+fp)>0 else 0
        ac=(tp+tn)/(tp+fp+fn+tn) if(tp+fp+fn+tn)>0 else 0
        f1=2*tp/(2*tp+fp+fn) if(2*tp+fp+fn)>0 else 0
        m=' *' if l in TARG5 else ''
        print(f"  {l:<30} {tp:>5} {fp:>5} {fn:>5} {tn:>5} {se:>8.4f} {sp:>8.4f} {ac:>8.4f} {f1:>8.4f}{m}", flush=True)
        r[l]={'TP':tp,'FP':fp,'FN':fn,'TN':tn,'sens':se,'spec':sp,'acc':ac,'f1':f1}
    t5f=np.mean([r[l]['f1'] for l in TARG5])
    t5se=np.mean([r[l]['sens'] for l in TARG5])
    t5sp=np.mean([r[l]['spec'] for l in TARG5])
    mtp=sum(r[l]['TP'] for l in LABELS); mfp=sum(r[l]['FP'] for l in LABELS); mfn=sum(r[l]['FN'] for l in LABELS)
    mi=2*mtp/(2*mtp+mfp+mfn) if(2*mtp+mfp+mfn)>0 else 0
    ma=np.mean([r[l]['f1'] for l in LABELS])
    print(f"\n  Micro-14={mi:.4f}  Macro-14={ma:.4f}  Target-5={t5f:.4f}  T5-Sens={t5se:.4f}  T5-Spec={t5sp:.4f}", flush=True)
    return r

bl_r = full_metrics(bl_labels, ref_labels, "BASELINE")
pa_r = full_metrics(pa_labels, ref_labels, "PARE")
mg_r = full_metrics(mg_labels, ref_labels, "MERGED TEXT")

# ── Bootstrap CI ──
print(f"\n{'='*120}", flush=True)
print("BOOTSTRAP CONFIDENCE INTERVALS (B=2000)", flush=True)
np.random.seed(42)
B = 2000
TIDX = [LABELS.index(p) for p in TARG5]

def macro_f1(pred, ref, idxs):
    f1s = []
    for j in idxs:
        tp=((ref[:,j]==1)&(pred[:,j]==1)).sum()
        fp=((ref[:,j]==0)&(pred[:,j]==1)).sum()
        fn=((ref[:,j]==1)&(pred[:,j]==0)).sum()
        f1=2*tp/(2*tp+fp+fn) if(2*tp+fp+fn)>0 else 0
        f1s.append(f1)
    return np.mean(f1s)

bl_boot=[]; pa_boot=[]; mg_boot=[]
for b in range(B):
    idx = np.random.choice(N, N, replace=True)
    bl_boot.append(macro_f1(bl_labels[idx], ref_labels[idx], TIDX))
    pa_boot.append(macro_f1(pa_labels[idx], ref_labels[idx], TIDX))
    mg_boot.append(macro_f1(mg_labels[idx], ref_labels[idx], TIDX))

bl_boot=np.array(bl_boot); pa_boot=np.array(pa_boot); mg_boot=np.array(mg_boot)
ci = lambda a: (np.percentile(a,2.5), np.percentile(a,97.5))

delta_pa = pa_boot - bl_boot
delta_mg = mg_boot - bl_boot

print(f"\n  {'Condition':<20} {'Mean':>7} {'95% CI':>22} {'p(Δ>0)':>8}")
print(f"  {'-'*65}")
print(f"  {'Baseline':<20} {bl_boot.mean():>7.4f} [{ci(bl_boot)[0]:.4f}, {ci(bl_boot)[1]:.4f}]")
print(f"  {'PARE':<20} {pa_boot.mean():>7.4f} [{ci(pa_boot)[0]:.4f}, {ci(pa_boot)[1]:.4f}]")
print(f"  {'Merged Text':<20} {mg_boot.mean():>7.4f} [{ci(mg_boot)[0]:.4f}, {ci(mg_boot)[1]:.4f}]")
print(f"  {'Δ(PARE-BL)':<20} {delta_pa.mean():>+7.4f} [{ci(delta_pa)[0]:+.4f}, {ci(delta_pa)[1]:+.4f}] {(delta_pa>0).mean():>8.4f}")
print(f"  {'Δ(Merged-BL)':<20} {delta_mg.mean():>+7.4f} [{ci(delta_mg)[0]:+.4f}, {ci(delta_mg)[1]:+.4f}] {(delta_mg>0).mean():>8.4f}")

# ── McNemar's test ──
print(f"\n{'='*120}", flush=True)
print("McNEMAR'S TEST (BL vs Merged Text, per Target-5 pathology)", flush=True)
print(f"  {'Pathology':<22} {'b(BL✓,MG✗)':>12} {'c(BL✗,MG✓)':>12} {'χ²':>8} {'p-value':>10} {'sig':>5}")
print(f"  {'-'*75}")
for p in TARG5:
    j = LABELS.index(p)
    bl_correct = (bl_labels[:,j] == ref_labels[:,j]).astype(int)
    mg_correct = (mg_labels[:,j] == ref_labels[:,j]).astype(int)
    b_count = int(((bl_correct==1)&(mg_correct==0)).sum())
    c_count = int(((bl_correct==0)&(mg_correct==1)).sum())
    if b_count + c_count > 0:
        chi2 = (abs(b_count-c_count)-1)**2 / (b_count+c_count)
        pval = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2 = 0; pval = 1.0
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"  {p:<22} {b_count:>12} {c_count:>12} {chi2:>8.2f} {pval:>10.6f} {sig:>5}")

# ── Delta table ──
print(f"\n{'='*120}")
print(f"  FULL DELTA TABLE (Target-5)")
print(f"  {'Path':<22} {'BL_F1':>7} {'PA_F1':>7} {'MG_F1':>7} {'dPA':>7} {'dMG':>7} | {'BL_Se':>7} {'MG_Se':>7} | {'BL_Sp':>7} {'MG_Sp':>7} | {'BL_Ac':>7} {'MG_Ac':>7}")
print(f"  {'-'*115}")
for p in TARG5:
    b=bl_r[p]; s=pa_r[p]; m=mg_r[p]
    print(f"  {p:<22} {b['f1']:>7.4f} {s['f1']:>7.4f} {m['f1']:>7.4f} {s['f1']-b['f1']:>+7.4f} {m['f1']-b['f1']:>+7.4f} | {b['sens']:>7.4f} {m['sens']:>7.4f} | {b['spec']:>7.4f} {m['spec']:>7.4f} | {b['acc']:>7.4f} {m['acc']:>7.4f}")

print("\nDONE.", flush=True)
