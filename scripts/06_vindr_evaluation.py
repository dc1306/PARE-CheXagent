#!/usr/bin/env python3
"""
06_vindr_pare.py — VinDr-CXR Adapted PARE Pipeline
====================================================
Complete end-to-end: baseline → feature extraction → probe training →
Monge OT maps → gated steering → CheXbert evaluation.

Uses VinDr-CXR train for PARE training, VinDr-CXR test for evaluation.
Does NOT use MIMIC-trained components (that's the transfer experiment).

VinDr specifics:
  - PA-view only (single image per sample)
  - Radiologist-annotated labels (not report-derived)
  - No reference reports → evaluate CheXbert labels vs radiologist GT
  - Edema excluded (0 test positives)
"""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_DEVICE_ID", "0")
import torch, json, time, sys, gc, re, pickle, warnings
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.linalg import sqrtm, inv
from f1chexbert import F1CheXbert
warnings.filterwarnings('ignore')

# ── Configuration ──
DEVICE = "cuda:0"; DTYPE = torch.float16
MODEL_NAME = "StanfordAIMI/CheXagent-8b"
PROMPT = "Write the findings section for this chest X-ray."
MAX_NEW_TOKENS = 512
N_VIS = 128; STEER_LAYER = 3; PROBE_LAYER = 31
PCA_DIMS = 128; BEST_LAMBDA = 1.0; COV_REG = 1e-5

VINDR_ROOT = os.environ.get("VINDR_ROOT",
    "/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/vindr_cxr/physionet.org/files/vindr-cxr/1.0.0")
OUT_DIR = os.environ.get("OUTPUT_DIR",
    "/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/vindr_standardized")
os.makedirs(OUT_DIR, exist_ok=True)

# VinDr Target-4 (Edema excluded: 0 test positives)
PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Lung Opacity', 'Pleural Effusion']
VINDR_COLS = {'Atelectasis': 'Atelectasis', 'Cardiomegaly': 'Cardiomegaly',
              'Lung Opacity': 'Lung Opacity', 'Pleural Effusion': 'Pleural effusion'}
CHEXBERT_IDX = {'Cardiomegaly': 1, 'Lung Opacity': 2, 'Atelectasis': 7, 'Pleural Effusion': 9}

LOG = open(f"{OUT_DIR}/vindr_pare.log", "w")
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    LOG.write(line + "\n"); LOG.flush()
    print(line, flush=True)

# ══════════════════════════════════════════════════════════════════════
# Step 0: Load VinDr data
# ══════════════════════════════════════════════════════════════════════
log("Step 0: Loading VinDr-CXR data")

train_labels = pd.read_csv(f"{VINDR_ROOT}/annotations/image_labels_train.csv")
test_labels = pd.read_csv(f"{VINDR_ROOT}/annotations/image_labels_test.csv")

# Train: majority vote across annotators
train_gt = train_labels.groupby('image_id')[list(VINDR_COLS.values())].max().reset_index()
test_gt = test_labels  # Test has single consensus annotation

log(f"  Train: {len(train_gt)} images")
log(f"  Test: {len(test_gt)} images")
for p in PATHOLOGIES:
    vc = VINDR_COLS[p]
    tr_pos = (train_gt[vc] > 0).sum()
    te_pos = (test_gt[vc] > 0).sum()
    log(f"  {p:22s}: train={tr_pos}, test={te_pos}")

# Find image paths
def find_images(image_ids, split):
    paths = {}
    img_dir = f"{VINDR_ROOT}/{split}"
    for iid in image_ids:
        for ext in ['.png', '.jpg', '.jpeg', '.dicom']:
            p = f"{img_dir}/{iid}{ext}"
            if os.path.exists(p):
                paths[iid] = p; break
    return paths

log("  Finding train images...")
train_paths = find_images(train_gt['image_id'].values, 'train')
log(f"  Found {len(train_paths)}/{len(train_gt)} train images")

# For test, check pre-converted JPGs first
test_jpg_dir = f"{os.path.dirname(OUT_DIR)}/vindr_evaluation/test_jpgs"
if os.path.isdir(test_jpg_dir) and len(os.listdir(test_jpg_dir)) > 0:
    log(f"  Using pre-converted test JPGs from {test_jpg_dir}")
    test_paths = {}
    for f in os.listdir(test_jpg_dir):
        if f.endswith('.jpg') or f.endswith('.png'):
            iid = os.path.splitext(f)[0]
            test_paths[iid] = os.path.join(test_jpg_dir, f)
else:
    test_paths = find_images(test_gt['image_id'].values, 'test')
log(f"  Found {len(test_paths)}/{len(test_gt)} test images")

# ══════════════════════════════════════════════════════════════════════
# Step 1: Load model
# ══════════════════════════════════════════════════════════════════════
log("\nStep 1: Loading CheXagent-8B")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE).eval()

# Build chat template
conv = [{"from": "human", "value": f"<image>\n{PROMPT}"}]
input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
TEXT_IDS = input_ids  # reused for all images

def load_image(path):
    img = Image.open(path).convert("RGB")
    pixel_values = model.process_images([img]).to(DEVICE, dtype=DTYPE)
    return pixel_values

# ══════════════════════════════════════════════════════════════════════
# Step 2: Generate baseline reports + extract features (train)
# ══════════════════════════════════════════════════════════════════════
TRAIN_CACHE = f"{OUT_DIR}/train_cache.pkl"
if os.path.exists(TRAIN_CACHE):
    log("\nStep 2: Loading cached train features")
    with open(TRAIN_CACHE, 'rb') as f:
        cache = pickle.load(f)
    train_l3 = cache['l3']; train_l31 = cache['l31']
    train_reports = cache['reports']; train_ids = cache['ids']
    train_labels_arr = cache['labels']
else:
    log("\nStep 2: Generating train baseline + extracting features")
    train_ids = []; train_l3 = []; train_l31 = []; train_reports = []
    train_labels_arr = {p: [] for p in PATHOLOGIES}

    # Sort by image_id for determinism
    valid_train = train_gt[train_gt['image_id'].isin(train_paths)].reset_index(drop=True)
    log(f"  Processing {len(valid_train)} train images")

    hooks = []
    l3_out = [None]; l31_out = [None]

    def hook_l3(m, inp, out):
        if hasattr(out, 'last_hidden_state'):
            l3_out[0] = out.last_hidden_state.detach()
        elif isinstance(out, tuple):
            l3_out[0] = out[0].detach()

    def hook_l31(m, inp, out):
        if hasattr(out, 'last_hidden_state'):
            l31_out[0] = out.last_hidden_state.detach()
        elif isinstance(out, tuple):
            l31_out[0] = out[0].detach()

    hooks.append(model.language_model.model.layers[STEER_LAYER].register_forward_hook(hook_l3))
    hooks.append(model.language_model.model.layers[PROBE_LAYER].register_forward_hook(hook_l31))

    errors = 0
    for i, row in valid_train.iterrows():
        iid = row['image_id']
        try:
            pv = load_image(train_paths[iid])
            with torch.no_grad():
                out = model.generate(input_ids=TEXT_IDS, pixel_values=pv,
                                     max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                report = tokenizer.decode(out[0], skip_special_tokens=True).strip()

            # Extract features
            vis_l3 = l3_out[0][0, :N_VIS, :].cpu().numpy()
            vis_l31 = l31_out[0][0, :N_VIS, :].mean(dim=0).cpu().numpy()

            train_ids.append(iid)
            train_l3.append(vis_l3)
            train_l31.append(vis_l31)
            train_reports.append(report)
            for p in PATHOLOGIES:
                train_labels_arr[p].append(1 if row[VINDR_COLS[p]] > 0 else 0)

        except Exception as e:
            errors += 1
            if errors <= 5: log(f"  Error on {iid}: {e}")

        if (i+1) % 200 == 0:
            log(f"  {i+1}/{len(valid_train)} (err={errors})")

    for h in hooks: h.remove()

    train_l3 = np.array(train_l3)
    train_l31 = np.array(train_l31)
    train_labels_arr = {p: np.array(v) for p, v in train_labels_arr.items()}

    log(f"  Train done: {len(train_ids)} images, {errors} errors")
    log(f"  L3 shape: {train_l3.shape}, L31 shape: {train_l31.shape}")

    with open(TRAIN_CACHE, 'wb') as f:
        pickle.dump({'l3': train_l3, 'l31': train_l31, 'reports': train_reports,
                      'ids': train_ids, 'labels': train_labels_arr}, f)
    log(f"  Cached to {TRAIN_CACHE}")

# ══════════════════════════════════════════════════════════════════════
# Step 3: Train probes + Monge maps
# ══════════════════════════════════════════════════════════════════════
log("\nStep 3: Training probes + Monge maps")

scaler31 = StandardScaler().fit(train_l31)
X31 = scaler31.transform(train_l31)

probes = {}; thresholds = {}; monge_maps = {}; pca_models = {}; scalers_l3 = {}

for p in PATHOLOGIES:
    y = train_labels_arr[p]
    n_pos = y.sum(); n_neg = len(y) - n_pos
    log(f"\n  {p}: pos={n_pos}, neg={n_neg}")

    if n_pos < 10:
        log(f"  SKIP: insufficient positives")
        continue

    # Train L31 probe
    probe = LogisticRegression(C=0.01, max_iter=3000, solver='lbfgs')
    probe.fit(X31, y)
    probs = probe.predict_proba(X31)[:, 1]

    # Youden threshold
    fpr, tpr, thr = roc_curve(y, probs)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    threshold = thr[best_idx]
    auc = roc_auc_score(y, probs)
    log(f"  Probe: AUROC={auc:.4f}, threshold={threshold:.4f}")

    probes[p] = probe
    thresholds[p] = threshold

    # Build Monge map from L3 tokens
    # FN = positive but probe says negative, TP = positive and probe says positive
    pos_mask = y == 1
    fn_mask = pos_mask & (probs < threshold)
    tp_mask = pos_mask & (probs >= threshold)

    fn_count = fn_mask.sum(); tp_count = tp_mask.sum()
    log(f"  FN={fn_count}, TP={tp_count}")

    if fn_count < 15 or tp_count < 15:
        log(f"  SKIP Monge: insufficient FN/TP")
        continue

    # PCA on L3 visual tokens (flatten: N_VIS * hidden_dim → PCA_DIMS)
    fn_l3 = train_l3[fn_mask].reshape(fn_count, -1)
    tp_l3 = train_l3[tp_mask].reshape(tp_count, -1)

    pca = PCA(n_components=PCA_DIMS, random_state=42)
    all_l3 = np.vstack([fn_l3, tp_l3])
    pca.fit(all_l3)
    var_exp = pca.explained_variance_ratio_.sum()
    log(f"  PCA: {PCA_DIMS}d, var_explained={var_exp:.4f}")

    fn_pca = pca.transform(fn_l3)
    tp_pca = pca.transform(tp_l3)

    # Monge OT map: T(x) = A(x - mu_fn) + mu_tp
    mu_fn = fn_pca.mean(axis=0)
    mu_tp = tp_pca.mean(axis=0)

    cov_fn = LedoitWolf().fit(fn_pca).covariance_ + COV_REG * np.eye(PCA_DIMS)
    cov_tp = LedoitWolf().fit(tp_pca).covariance_ + COV_REG * np.eye(PCA_DIMS)

    sqrt_fn = sqrtm(cov_fn)
    sqrt_fn_inv = inv(sqrt_fn)
    inner = sqrtm(sqrt_fn @ cov_tp @ sqrt_fn)
    A = sqrt_fn_inv @ inner @ sqrt_fn_inv

    monge_maps[p] = {'A': A.real, 'mu_fn': mu_fn, 'mu_tp': mu_tp}
    pca_models[p] = pca
    log(f"  Monge map built")

# Save components
COMP_PATH = f"{OUT_DIR}/vindr_pare_components.pkl"
with open(COMP_PATH, 'wb') as f:
    pickle.dump({
        'probes': probes, 'thresholds': thresholds, 'monge_maps': monge_maps,
        'pca_models': pca_models, 'scaler31': scaler31,
        'pathologies': PATHOLOGIES, 'config': {
            'probe_layer': PROBE_LAYER, 'steer_layer': STEER_LAYER,
            'pca_dims': PCA_DIMS, 'lambda': BEST_LAMBDA, 'n_vis': N_VIS
        }
    }, f)
log(f"\nSaved components to {COMP_PATH}")

# ══════════════════════════════════════════════════════════════════════
# Step 4: Test — baseline + steered reports
# ══════════════════════════════════════════════════════════════════════
log("\nStep 4: Generating test baseline + steered reports")

# Compute deltas for each pathology
deltas = {}
for p in PATHOLOGIES:
    if p not in monge_maps:
        continue
    mm = monge_maps[p]
    pca = pca_models[p]

    # Precompute delta for mean FN → TP shift
    mu_fn_full = pca.inverse_transform(mm['mu_fn'].reshape(1, -1)).reshape(N_VIS, -1)
    shifted = pca.inverse_transform(
        (mm['A'] @ (mm['mu_fn'] - mm['mu_fn']).reshape(-1, 1) + mm['mu_tp'].reshape(-1, 1)).T
    ).reshape(N_VIS, -1)
    # Actually compute per-sample at test time
    deltas[p] = {'A': mm['A'], 'mu_fn': mm['mu_fn'], 'mu_tp': mm['mu_tp'], 'pca': pca}

valid_test = test_gt[test_gt['image_id'].isin(test_paths)].reset_index(drop=True)
log(f"  Processing {len(valid_test)} test images")

hooks = []
l3_out = [None]; l31_out = [None]
def hook_l3(m, inp, out):
    if hasattr(out, 'last_hidden_state'): l3_out[0] = out.last_hidden_state.detach()
    elif isinstance(out, tuple): l3_out[0] = out[0].detach()
def hook_l31(m, inp, out):
    if hasattr(out, 'last_hidden_state'): l31_out[0] = out.last_hidden_state.detach()
    elif isinstance(out, tuple): l31_out[0] = out[0].detach()
hooks.append(model.language_model.model.layers[STEER_LAYER].register_forward_hook(hook_l3))
hooks.append(model.language_model.model.layers[PROBE_LAYER].register_forward_hook(hook_l31))

results = []
errors = 0

for i, row in valid_test.iterrows():
    iid = row['image_id']
    try:
        pv = load_image(test_paths[iid])

        # Baseline
        with torch.no_grad():
            out = model.generate(input_ids=TEXT_IDS, pixel_values=pv,
                                 max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            bl_report = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        bl_l3 = l3_out[0][0, :N_VIS, :].clone()
        bl_l31 = l31_out[0][0, :N_VIS, :].mean(dim=0).cpu().numpy()

        # Probe scores
        x31 = scaler31.transform(bl_l31.reshape(1, -1))
        probe_scores = {}
        fired = []
        for p in PATHOLOGIES:
            if p in probes:
                prob = probes[p].predict_proba(x31)[:, 1][0]
                probe_scores[p] = float(prob)
                if prob >= thresholds[p]:
                    fired.append(p)

        # Steered generation (if any pathology fires)
        if fired and any(p in monge_maps for p in fired):
            # Compute combined delta
            delta = torch.zeros_like(bl_l3)
            for p in fired:
                if p not in monge_maps: continue
                dd = deltas[p]
                l3_np = bl_l3.cpu().float().numpy().reshape(N_VIS, -1)
                l3_flat = l3_np.reshape(1, -1)
                l3_pca = dd['pca'].transform(l3_flat)
                shifted_pca = (dd['A'] @ (l3_pca.flatten() - dd['mu_fn']).reshape(-1, 1) + dd['mu_tp'].reshape(-1, 1)).T
                shifted_full = dd['pca'].inverse_transform(shifted_pca).reshape(N_VIS, -1)
                d = shifted_full - l3_np
                delta += torch.tensor(d, device=DEVICE, dtype=DTYPE)

            delta = delta * BEST_LAMBDA / max(len(fired), 1)

            # Hook to inject delta
            def make_steer_hook(delta_tensor):
                def hook(m, inp, out):
                    if hasattr(out, 'last_hidden_state'):
                        h = out.last_hidden_state
                    elif isinstance(out, tuple):
                        h = out[0]
                    else:
                        return out
                    if h.shape[1] > N_VIS:
                        h[:, :N_VIS, :] += delta_tensor
                    return out
                return hook

            steer_hook = model.language_model.model.layers[STEER_LAYER].register_forward_hook(
                make_steer_hook(delta)
            )
            with torch.no_grad():
                out = model.generate(input_ids=TEXT_IDS, pixel_values=pv,
                                     max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                st_report = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            steer_hook.remove()
        else:
            st_report = bl_report

        # GT labels
        gt = {}
        for p in PATHOLOGIES:
            gt[p] = 1 if row[VINDR_COLS[p]] > 0 else 0

        results.append({
            'image_id': iid,
            'baseline_report': bl_report,
            'steered_report': st_report,
            'steered': len(fired) > 0,
            'targets': fired,
            'probe_scores': probe_scores,
            'gt': gt,
        })

    except Exception as e:
        errors += 1
        if errors <= 5: log(f"  Error on {iid}: {e}")

    if (i+1) % 100 == 0:
        steered_so_far = sum(1 for r in results if r['steered'])
        log(f"  {i+1}/{len(valid_test)} (steered={steered_so_far}, err={errors})")

for h in hooks: h.remove()

json.dump(results, open(f"{OUT_DIR}/vindr_test_results.json", 'w'), indent=1)
log(f"\nTest done: {len(results)} images, {errors} errors")
log(f"Steered: {sum(1 for r in results if r['steered'])}/{len(results)}")

# ══════════════════════════════════════════════════════════════════════
# Step 5: Evaluate — CheXbert labels vs radiologist GT
# ══════════════════════════════════════════════════════════════════════
log("\nStep 5: CheXbert evaluation")
scorer = F1CheXbert(device=DEVICE)

clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip()

for condition, key in [("BASELINE", "baseline_report"), ("PARE", "steered_report")]:
    log(f"\n  {condition}")
    log(f"  {'Pathology':<22} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Sens':>8} {'Spec':>8} {'F1':>8}")
    log(f"  {'-'*80}")

    all_f1s = []
    for p in PATHOLOGIES:
        tp = fp = fn = tn = 0
        for r in results:
            # CheXbert label the report
            text = clean(r[key])
            if not text:
                pred = 0
            else:
                raw = scorer.get_label(text, mode='rrg')
                idx = CHEXBERT_IDX[p]
                pred = int(raw[idx]) if idx < len(raw) and raw[idx] != '' else 0

            gt = r['gt'][p]
            if pred == 1 and gt == 1: tp += 1
            elif pred == 1 and gt == 0: fp += 1
            elif pred == 0 and gt == 1: fn += 1
            else: tn += 1

        se = tp/(tp+fn) if (tp+fn) > 0 else 0
        sp = tn/(tn+fp) if (tn+fp) > 0 else 0
        f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
        all_f1s.append(f1)
        log(f"  {p:<22} {tp:>5} {fp:>5} {fn:>5} {tn:>5} {se:>8.4f} {sp:>8.4f} {f1:>8.4f}")

    macro = np.mean(all_f1s)
    log(f"  {'MACRO-F1':<22} {'':>5} {'':>5} {'':>5} {'':>5} {'':>8} {'':>8} {macro:>8.4f}")

log("\n✅ VINDR-CXR PARE EVALUATION COMPLETE")
