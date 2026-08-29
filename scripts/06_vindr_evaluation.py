#!/usr/bin/env python3
"""
VinDr-CXR Cross-Dataset Generalization — PARE Evaluation
Uses MIMIC-CXR-trained probes + OT maps on VinDr-CXR test set.
Reproduces the exact pare_reverify_part2.py pipeline on a new dataset.
"""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch, json, time, sys, gc, numpy as np, warnings, re
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.linalg import sqrtm, inv
warnings.filterwarnings('ignore')

DEVICE = "cuda:0"; DTYPE = torch.float16
MODEL_NAME = "StanfordAIMI/CheXagent-8b"
PROMPT = "Describe the findings in this chest X-ray."
FORMATTED_PROMPT = f" USER: <s>{PROMPT} ASSISTANT: <s>"
N_VIS = 128; STEER_LAYER = 3; PCA_DIMS = 128; BEST_LAMBDA = 1.0

PATHOLOGIES = ['Atelectasis','Cardiomegaly','LungOpacity','PleuralEffusion','PulmonaryEdema']
LABEL_KEYS = {'Atelectasis':'y_uo_Atelectasis','Cardiomegaly':'y_uo_Cardiomegaly',
    'LungOpacity':'y_uo_Lung Opacity','PleuralEffusion':'y_uo_Pleural Effusion',
    'PulmonaryEdema':'y_uo_Pulmonary Edema'}
CHEXBERT_IDX = {'Cardiomegaly':1,'LungOpacity':2,'PulmonaryEdema':4,'Atelectasis':7,'PleuralEffusion':9}

MIMIC_EXP = Path('/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/pare_reverify')
FEAT_DIR = Path('/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/official_cls_v2')
OUTDIR = Path('/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/vindr_evaluation')
OUTDIR.mkdir(exist_ok=True)

LOG = OUTDIR / 'evaluation.log'
def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"; print(line, flush=True)
    with open(LOG,'a') as f: f.write(line+'\n')

log("="*70)
log("  VinDr-CXR Cross-Dataset Generalization")
log("="*70)

# ═══════════════════════════════════════════════════
# STEP 0: Load VinDr-CXR test labels + convert DICOMs
# ═══════════════════════════════════════════════════
import pandas as pd
labels_df = pd.read_csv('experiments/vindr_cxr/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv')
vindr_gt = {}
for _, row in labels_df.iterrows():
    vindr_gt[row['image_id']] = {
        'Atelectasis': int(row.get('Atelectasis', 0)),
        'Cardiomegaly': int(row.get('Cardiomegaly', 0)),
        'LungOpacity': int(row.get('Lung Opacity', 0)),
        'PleuralEffusion': int(row.get('Pleural effusion', 0)),
        'PulmonaryEdema': int(row.get('Edema', 0)),
    }
img_ids = sorted(vindr_gt.keys())
prevalence = {p: sum(v[p] for v in vindr_gt.values()) for p in PATHOLOGIES}
log(f"VinDr-CXR test: {len(img_ids)} images")
log(f"Prevalence: {prevalence}")

# Convert DICOMs
import pydicom
dicom_dir = '/mnt/raid/obed/arjun/smooth_ae_vinbigdata/smooth_ae_vinbig/vinbigdata/test'
jpg_dir = OUTDIR / 'test_jpgs'
jpg_dir.mkdir(exist_ok=True)

converted = 0
for img_id in img_ids:
    jpg_path = jpg_dir / f'{img_id}.jpg'
    if jpg_path.exists(): converted += 1; continue
    dicom_path = f'{dicom_dir}/{img_id}.dicom'
    if not os.path.exists(dicom_path): continue
    try:
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array.astype(float)
        if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = arr.max() - arr
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(arr).convert('RGB').save(str(jpg_path), quality=95)
        converted += 1
    except Exception as ex:
        if converted < 5: log(f"  DICOM ERR: {ex}")
    if converted % 500 == 0: log(f"  Converted {converted}...")
log(f"  {converted} JPGs ready")

# ═══════════════════════════════════════════════════
# STEP 1: Load MIMIC-trained probes + Monge maps
# ═══════════════════════════════════════════════════
log("\n[Step1] Loading MIMIC-trained probes + Monge maps...")
train_manifest = json.load(open(FEAT_DIR/'train_manifest.json'))[:60000]
val_manifest = json.load(open(FEAT_DIR/'validate_manifest.json'))
train_idx = np.load(MIMIC_EXP/'train_idx.npy')
train_sub = [train_manifest[i] for i in train_idx]

train_l3 = np.load(MIMIC_EXP/'train_l3_tokens.npy')  # (15000, 128, 4096)
train_l31 = np.load(MIMIC_EXP/'train_l31_mean.npy')
val_l31 = np.load(MIMIC_EXP/'val_l31_mean.npy')
train_cb = json.load(open(MIMIC_EXP/'train_chexbert_labels.json'))

gt_train = {}; gt_val = {}
for p in PATHOLOGIES:
    k = LABEL_KEYS[p]
    gt_train[p] = np.array([int(train_sub[i].get(k,0)) for i in range(len(train_sub))])
    gt_val[p] = np.array([int(e.get(k,0)) for e in val_manifest])

# Train probes on MIMIC L31
sc31 = StandardScaler()
d31_tr = sc31.fit_transform(train_l31)
d31_va = sc31.transform(val_l31)

probes = {}; thresholds = {}
for p in PATHOLOGIES:
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.01,
                              class_weight='balanced', max_iter=5000, random_state=42)
    clf.fit(d31_tr, gt_train[p])
    val_scores = clf.predict_proba(d31_va)[:,1]
    fpr,tpr,th = roc_curve(gt_val[p], val_scores)
    thresholds[p] = float(th[np.argmax(tpr - fpr)])
    probes[p] = clf
    log(f"  {p:20s}: thresh={thresholds[p]:.4f}")

# Build Monge maps from MIMIC training data (same as part2)
log("\n[Step2] Building Monge maps from MIMIC training data...")
monge_maps = {}
for p in PATHOLOGIES:
    bl_labels = np.array([train_cb.get(train_sub[i]['dicom_id'],{}).get(p,0) for i in range(len(train_sub))])
    gt = gt_train[p]
    tp_mask = (gt == 1) & (bl_labels == 0)  # FN cases: GT positive but baseline missed
    fn_feats = train_l3[tp_mask]  # shape: (n_fn, 128, 4096)
    
    # Also get TP cases for source distribution
    tp_pos_mask = (gt == 1) & (bl_labels == 1)
    tp_feats = train_l3[tp_pos_mask] if tp_pos_mask.sum() > 0 else fn_feats
    
    log(f"  {p}: FN={tp_mask.sum()}, TP={tp_pos_mask.sum()}")
    
    if fn_feats.shape[0] < 10:
        log(f"    WARNING: too few FN samples, skipping"); continue
    
    # PCA on visual tokens
    all_tokens = np.vstack([fn_feats.reshape(-1, 4096), tp_feats.reshape(-1, 4096)])
    sc = StandardScaler(); all_sc = sc.fit_transform(all_tokens)
    pca = PCA(n_components=PCA_DIMS, random_state=42); pca.fit(all_sc)
    
    # Source (FN) and Target (TP) distributions in PCA space
    fn_sc = sc.transform(fn_feats.reshape(-1, 4096))
    fn_pca = pca.transform(fn_sc).reshape(fn_feats.shape[0], N_VIS, PCA_DIMS)
    fn_mean = fn_pca.reshape(-1, PCA_DIMS)
    
    tp_sc = sc.transform(tp_feats.reshape(-1, 4096))
    tp_pca = pca.transform(tp_sc).reshape(tp_feats.shape[0], N_VIS, PCA_DIMS)
    tp_mean = tp_pca.reshape(-1, PCA_DIMS)
    
    mu_s = fn_mean.mean(axis=0); mu_t = tp_mean.mean(axis=0)
    cov_s = LedoitWolf().fit(fn_mean).covariance_
    cov_t = LedoitWolf().fit(tp_mean).covariance_
    
    # Monge map: T(x) = mu_t + A(x - mu_s)
    S_half = sqrtm(cov_s).real
    S_half_inv = inv(S_half)
    M = sqrtm(S_half @ cov_t @ S_half).real
    A = S_half_inv @ M @ S_half_inv
    
    monge_maps[p] = {
        'A': torch.tensor(A, dtype=torch.float32),
        'mu_s': torch.tensor(mu_s, dtype=torch.float32),
        'mu_t': torch.tensor(mu_t, dtype=torch.float32),
        'sc_mean': torch.tensor(sc.mean_, dtype=torch.float32),
        'sc_scale': torch.tensor(sc.scale_, dtype=torch.float32),
        'pca_comp': torch.tensor(pca.components_, dtype=torch.float32),
        'pca_mean': torch.tensor(pca.mean_, dtype=torch.float32),
    }
    log(f"    Monge map built, var_explained={pca.explained_variance_ratio_.sum():.4f}")

# ═══════════════════════════════════════════════════
# STEP 3: Run CheXagent on VinDr-CXR (baseline + features + steering)
# ═══════════════════════════════════════════════════
log("\n[Step3] Running CheXagent on VinDr-CXR...")
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

bl_cache = OUTDIR / 'baseline_reports.json'
feat_cache = OUTDIR / 'features_l31_mean.npy'
l3_cache = OUTDIR / 'features_l3_tokens.npy'
st_cache = OUTDIR / 'steered_reports.json'

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
model.eval()
proc = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
gen_config = GenerationConfig.from_pretrained(MODEL_NAME)
tok = proc.tokenizer
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
dec_layers = model.language_model.model.layers

# Phase A: Baseline + feature extraction
if bl_cache.exists() and feat_cache.exists() and l3_cache.exists():
    log("  Loading cached baseline + features...")
    bl_reports = json.load(open(bl_cache))
    vindr_l31 = np.load(feat_cache)
    vindr_l3 = np.load(l3_cache)
else:
    bl_reports = {}
    l31_list = []; l3_list = []
    hook_data = {}
    
    def l3_hook(module, inp, output):
        hook_data['l3'] = output[0].detach().cpu()
    def l31_hook(module, inp, output):
        hook_data['l31'] = output[0].detach().cpu()
    
    h3 = dec_layers[STEER_LAYER].register_forward_hook(l3_hook)
    h31 = dec_layers[31].register_forward_hook(l31_hook)
    
    t0 = time.time(); errs = 0
    for idx, img_id in enumerate(img_ids):
        jpg_path = jpg_dir / f'{img_id}.jpg'
        if not jpg_path.exists():
            bl_reports[img_id] = ''
            l31_list.append(np.zeros(4096)); l3_list.append(np.zeros((N_VIS, 4096)))
            continue
        try:
            img = Image.open(str(jpg_path)).convert('RGB')
            inputs = proc(images=[img], text=FORMATTED_PROMPT, return_tensors='pt')
            for k,v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE) if v.dtype in (torch.long,torch.int) else v.to(DEVICE, dtype=DTYPE)
            
            with torch.no_grad():
                ids = model.generate(**inputs, generation_config=gen_config, max_new_tokens=300, pad_token_id=tok.pad_token_id)
            
            report = tok.decode(ids[0], skip_special_tokens=True).strip()
            bl_reports[img_id] = report
            
            # Extract features
            l31_feat = hook_data.get('l31', torch.zeros(1,1,4096))
            l31_list.append(l31_feat.mean(dim=1).squeeze().numpy()[:4096])
            
            l3_feat = hook_data.get('l3', torch.zeros(1,N_VIS,4096))
            l3_tokens = l3_feat[0,:N_VIS,:].numpy() if l3_feat.shape[1] >= N_VIS else np.zeros((N_VIS, 4096))
            l3_list.append(l3_tokens)
            
        except Exception as ex:
            bl_reports[img_id] = ''; errs += 1
            l31_list.append(np.zeros(4096)); l3_list.append(np.zeros((N_VIS, 4096)))
            if errs <= 3: log(f"  ERR: {ex}")
        
        if (idx+1) % 100 == 0:
            rate = (idx+1)/(time.time()-t0)
            log(f"  BL {idx+1}/{len(img_ids)} ({rate:.1f}/s, err:{errs})")
    
    h3.remove(); h31.remove()
    vindr_l31 = np.array(l31_list); vindr_l3 = np.array(l3_list)
    json.dump(bl_reports, open(bl_cache, 'w'))
    np.save(feat_cache, vindr_l31); np.save(l3_cache, vindr_l3)
    log(f"  Baseline done: {len(bl_reports)}, l31={vindr_l31.shape}, l3={vindr_l3.shape}")

# Phase B: Probe scores on VinDr features
log("\n[Step4] Computing probe scores on VinDr features...")
d31_vindr = sc31.transform(vindr_l31)
probe_scores_vindr = {}
for p in PATHOLOGIES:
    probe_scores_vindr[p] = probes[p].predict_proba(d31_vindr)[:,1]
    fires = sum(probe_scores_vindr[p] >= thresholds[p])
    log(f"  {p}: fires={fires}/{len(img_ids)} ({fires/len(img_ids)*100:.1f}%)")

# Phase C: Steered reports
if st_cache.exists():
    log("\n  Loading cached steered reports...")
    st_reports = json.load(open(st_cache))
else:
    log("\n[Step5] Generating steered reports...")
    # Precompute deltas
    test_deltas = {}
    for idx, img_id in enumerate(img_ids):
        tokens = torch.tensor(vindr_l3[idx], dtype=torch.float32)
        per_path = {}
        for p in PATHOLOGIES:
            if p not in monge_maps: continue
            mm = monge_maps[p]
            x_sc = (tokens - mm['sc_mean']) / (mm['sc_scale'] + 1e-8)
            x_pca = (x_sc - mm['pca_mean']) @ mm['pca_comp'].T
            transported = mm['mu_t'] + (x_pca - mm['mu_s']) @ mm['A'].T
            delta = (transported - x_pca) @ mm['pca_comp'] * (mm['sc_scale'] + 1e-8)
            per_path[p] = (delta * BEST_LAMBDA).to(torch.bfloat16)
        test_deltas[img_id] = per_path
    log(f"  Deltas precomputed for {len(test_deltas)} images")
    
    st_reports = {}
    t0 = time.time(); errs = 0
    for idx, img_id in enumerate(img_ids):
        targets = [p for p in PATHOLOGIES if probe_scores_vindr[p][idx] >= thresholds[p] and p in monge_maps]
        if not targets:
            st_reports[img_id] = bl_reports.get(img_id, ''); continue
        
        try:
            jpg_path = jpg_dir / f'{img_id}.jpg'
            img = Image.open(str(jpg_path)).convert('RGB')
            inputs = proc(images=[img], text=FORMATTED_PROMPT, return_tensors='pt')
            for k,v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE) if v.dtype in (torch.long,torch.int) else v.to(DEVICE, dtype=DTYPE)
            
            fired = [False]
            deltas = test_deltas[img_id]
            def make_hook(tgts, dlt, flag):
                def hook_fn(module, inp, output):
                    if flag[0]: return output
                    h = output[0] if isinstance(output, tuple) else output
                    if h.shape[1] > 1:
                        flag[0] = True
                        hm = h.clone()
                        total = torch.zeros(N_VIS, h.shape[2], dtype=torch.float32)
                        for p in tgts: total += dlt[p].float()
                        hm[0,:N_VIS,:] += total.to(device=h.device, dtype=h.dtype)
                        return (hm,) + output[1:] if isinstance(output, tuple) else hm
                    return output
                return hook_fn
            
            hook = dec_layers[STEER_LAYER].register_forward_hook(make_hook(targets, deltas, fired))
            with torch.no_grad():
                ids = model.generate(**inputs, generation_config=gen_config, max_new_tokens=300, pad_token_id=tok.pad_token_id)
            hook.remove()
            st_reports[img_id] = tok.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as ex:
            st_reports[img_id] = bl_reports.get(img_id, ''); errs += 1
            if errs <= 3: log(f"  STEER ERR: {ex}")
        
        if (idx+1) % 100 == 0:
            rate = (idx+1)/(time.time()-t0)
            log(f"  Steered {idx+1}/{len(img_ids)} ({rate:.1f}/s, err:{errs})")
    
    json.dump(st_reports, open(st_cache, 'w'))
    log(f"  Done: {len(st_reports)} steered reports ({errs} errors)")

del model, proc; torch.cuda.empty_cache(); gc.collect()

# ═══════════════════════════════════════════════════
# STEP 6: CheXbert evaluation
# ═══════════════════════════════════════════════════
log("\n[Step6] CheXbert evaluation...")
from f1chexbert import F1CheXbert
scorer = F1CheXbert(device=DEVICE)

def get_labels(reports):
    labels = {}
    for img_id in img_ids:
        r = reports.get(img_id, '')
        if not r.strip(): labels[img_id]={p:0 for p in PATHOLOGIES}; continue
        raw = scorer.get_label(r, mode='rrg')
        labels[img_id] = {p: int(raw[CHEXBERT_IDX[p]]) if CHEXBERT_IDX[p]<len(raw) else 0 for p in PATHOLOGIES}
    return labels

def eval_metrics(labels, name):
    ttp=tfp=tfn=ttn=0; pp={}
    for p in PATHOLOGIES:
        tp=fp=fn=tn=0
        for img_id in img_ids:
            g=vindr_gt[img_id][p]; pred=labels.get(img_id,{}).get(p,0)
            if g==1 and pred==1: tp+=1
            elif g==0 and pred==1: fp+=1
            elif g==1 and pred==0: fn+=1
            else: tn+=1
        ttp+=tp; tfp+=fp; tfn+=fn; ttn+=tn
        se=tp/(tp+fn) if tp+fn else 0; sp=tn/(tn+fp) if tn+fp else 0
        pp[p]=(se,sp,tp,fp,fn)
    sens=ttp/(ttp+tfn) if ttp+tfn else 0; spec=ttn/(ttn+tfp) if ttn+tfp else 0
    prec=ttp/(ttp+tfp) if ttp+tfp else 0; f1=2*prec*sens/(prec+sens) if prec+sens else 0
    log(f"\n  {name}")
    log(f"  {'─'*65}")
    log(f"  {'Pathology':<20s} {'Sens':>6s} {'Spec':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    for p in PATHOLOGIES:
        se,sp,tp,fp,fn = pp[p]
        log(f"  {p:<20s} {se:.4f} {sp:.4f} {tp:5d} {fp:5d} {fn:5d}")
    log(f"  {'─'*65}")
    log(f"  OVERALL: F1={f1:.4f} Sens={sens:.4f} Spec={spec:.4f} Prec={prec:.4f}")
    return f1

cb_bl = get_labels(bl_reports); cb_st = get_labels(st_reports)
log("\n" + "="*70)
log("VinDr-CXR CROSS-DATASET RESULTS (MIMIC-trained → VinDr-CXR)")
log("="*70)
f1_bl = eval_metrics(cb_bl, "BASELINE")
f1_st = eval_metrics(cb_st, "PARE STEERED")
log(f"\n  Δ(PARE vs BL): {f1_st-f1_bl:+.4f}")
log(f"\n  MIMIC-CXR ref: BL=0.399 → PARE=0.522, Δ=+0.123")

n_healthy = sum(1 for v in vindr_gt.values() if all(v[p]==0 for p in PATHOLOGIES))
h_st = sum(1 for idx,img_id in enumerate(img_ids) if all(vindr_gt[img_id][p]==0 for p in PATHOLOGIES) and any(probe_scores_vindr[p][idx]>=thresholds[p] for p in PATHOLOGIES))
log(f"\n  Healthy: {n_healthy}/{len(img_ids)}, steered: {h_st}/{n_healthy} ({h_st/max(n_healthy,1)*100:.1f}%)")
log("\n✅ VINDR-CXR EVALUATION COMPLETE")
