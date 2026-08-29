#!/usr/bin/env python3
"""
PARE Standardized Test — Part 2
Applies PARE steering on the SAME 3,269 test studies as the frozen baseline.
Uses components built by pare_standardized_train.py.

Architecture:
  TEST CXR → CheXagent-8B forward pass
    ├─ L31 mean → probe → gate (score ≥ threshold?)
    └─ L3 visual tokens → Monge delta
  If gated: generate with L3 steering hook
  If not: use stored baseline report (exact preservation)

Then: F1CheXbert eval + Baseline∨Steered merged eval
"""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch, json, sys, time, re, gc, warnings
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
warnings.filterwarnings('ignore')

BASELINE_DIR = Path("/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/canonical_baseline")
EXP_DIR = Path("/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/pare_standardized")
DEVICE="cuda:0"; DTYPE=torch.float16; CHECKPOINT="StanfordAIMI/CheXagent-8b"
N_VIS=128; STEER_LAYER=3; PROBE_LAYER=31; BEST_LAMBDA=1.0
LOG_FILE = EXP_DIR / "test.log"

def log(msg):
    ts=time.strftime("%Y-%m-%d %H:%M:%S"); line=f"[{ts}] {msg}"
    print(line,flush=True)
    with open(LOG_FILE,"a") as f: f.write(line+"\n")


def compute_delta(l3_tokens, targets, monge_maps, lam):
    """Compute PARE steering delta from L3 visual tokens."""
    total = torch.zeros(N_VIS, 4096, dtype=torch.float32)
    for p in targets:
        mm = monge_maps[p]
        tokens = l3_tokens[:N_VIS].float()
        x_sc = (tokens - torch.tensor(mm['sc_mean'])) / (torch.tensor(mm['sc_scale']) + 1e-8)
        x_pca = (x_sc - torch.tensor(mm['pca_mean'])) @ torch.tensor(mm['pca_comp']).T
        transported = torch.tensor(mm['mu_t']) + (x_pca - torch.tensor(mm['mu_s'])) @ torch.tensor(mm['A']).T
        delta = (transported - x_pca) @ torch.tensor(mm['pca_comp']) * (torch.tensor(mm['sc_scale']) + 1e-8)
        total += delta * lam
    return total


def generate_steered_reports():
    log(f"\n{'='*60}")
    log(f"PARE Standardized Test")
    log(f"{'='*60}")

    # Load baseline manifest + reports
    manifest = json.load(open(BASELINE_DIR / "study_test_manifest.json"))
    baseline_reports = json.load(open(BASELINE_DIR / "chexagent_8b_reports.json"))
    bl_by_study = {r['study_id']: r for r in baseline_reports}
    log(f"Loaded: {len(manifest)} test studies, {len(baseline_reports)} baseline reports")

    # Assert baseline/test alignment
    manifest_ids = {s['study_id'] for s in manifest}
    baseline_ids = {r['study_id'] for r in baseline_reports}
    assert len(manifest) == len(baseline_reports), f"Count mismatch: {len(manifest)} vs {len(baseline_reports)}"
    assert manifest_ids == baseline_ids, f"Study ID mismatch: {len(manifest_ids - baseline_ids)} missing"
    log(f"  Baseline/manifest alignment verified: {len(manifest_ids)} studies")

    # Load PARE components + provenance check
    comp = pickle.load(open(EXP_DIR / 'pare_components.pkl', 'rb'))
    assert comp.get('checkpoint') == CHECKPOINT, f"Component checkpoint mismatch"
    assert comp.get('steer_layer') == STEER_LAYER, f"Steer layer mismatch"
    assert comp.get('probe_layer') == PROBE_LAYER, f"Probe layer mismatch"
    assert comp.get('n_vis') == N_VIS, f"N_VIS mismatch"
    probes = comp['probes']; thresholds = comp['thresholds']
    monge_maps = comp['monge_maps']; sc31 = comp['sc31']
    PATHOLOGIES = comp['pathologies']
    # Assert all pathologies have components
    for p in PATHOLOGIES:
        assert p in probes, f"Missing probe for {p}"
        assert p in monge_maps, f"Missing Monge map for {p}"
        assert p in thresholds, f"Missing threshold for {p}"
    log(f"PARE components: {len(probes)} probes, {len(monge_maps)} Monge maps")
    log(f"Thresholds: {thresholds}")

    # Check for resume
    results_path = EXP_DIR / "pare_test_reports.json"
    checkpoint_path = EXP_DIR / "pare_test_checkpoint.json"
    if results_path.exists():
        log(f"Loading completed results")
        return json.load(open(results_path)), PATHOLOGIES

    results = []; start_idx = 0
    if checkpoint_path.exists():
        ckpt = json.load(open(checkpoint_path))
        results = ckpt['results']; start_idx = len(results)
        log(f"Resuming from {start_idx}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
    model.eval()
    proc = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    tok = proc.tokenizer
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    dec_layers = model.language_model.model.layers

    log(f"  SAME: do_sample=False, num_beams=1, max_new_tokens=512")
    log(f"  PARE: layer={STEER_LAYER}, probe_layer={PROBE_LAYER}, lambda={BEST_LAMBDA}")

    errors=0; steered_count=0; unsteered_count=0; t0=time.time()

    for i in range(start_idx, len(manifest)):
        sample = manifest[i]
        try:
            images = [Image.open(p).convert("RGB") for p in sample['image_paths']]
            indication = sample['section_indication'] or "None provided"
            raw_prompt = f'Given the indication: "{indication}", write a structured findings section for the CXR.'
            prompt = f' USER: <s>{raw_prompt} ASSISTANT: <s>'

            inputs = proc(images=images, text=prompt, return_tensors='pt')
            for k,v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE) if v.dtype in (torch.long,torch.int) else v.to(DEVICE, dtype=DTYPE)

            # ── Pass 1: Forward to capture L3 + L31 ──
            captured = {'l3': None, 'l31': None}
            def make_capture(name, vis_only):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.shape[1] > 1:
                        if vis_only:
                            captured[name] = h[0, :N_VIS, :].detach().cpu().float()
                        else:
                            captured[name] = h[0].detach().cpu().float().mean(dim=0, keepdim=True)
                    return output
                return hook_fn

            h3 = dec_layers[STEER_LAYER].register_forward_hook(make_capture('l3', True))
            h31 = dec_layers[PROBE_LAYER].register_forward_hook(make_capture('l31', False))
            try:
                with torch.no_grad():
                    _ = model(**{k:v for k,v in inputs.items()})
            finally:
                h3.remove(); h31.remove()

            assert captured['l3'] is not None, "L3 hook failed"
            assert captured['l31'] is not None, "L31 hook failed"

            # ── Probe gating (L31 → probe → threshold) ──
            l31_scaled = sc31.transform(captured['l31'].numpy())
            targets = []
            probe_scores = {}
            for p in PATHOLOGIES:
                if p not in probes or p not in monge_maps: continue
                score = float(probes[p].predict_proba(l31_scaled)[0, 1])
                probe_scores[p] = score
                if score >= thresholds[p]:
                    targets.append(p)

            # ── Decision: steer or use baseline ──
            if targets:
                steered_count += 1
                delta = compute_delta(captured['l3'], targets, monge_maps, BEST_LAMBDA)
                delta_norm = float(delta.norm())

                fired = [False]
                def make_steer_hook(d, flag):
                    def hook_fn(module, inp, output):
                        if flag[0]: return output
                        h = output[0] if isinstance(output, tuple) else output
                        if h.shape[1] > 1:
                            flag[0] = True
                            hm = h.clone()
                            hm[0, :N_VIS, :] += d.to(device=h.device, dtype=h.dtype)
                            return (hm,) + output[1:] if isinstance(output, tuple) else hm
                        return output
                    return hook_fn

                hook = dec_layers[STEER_LAYER].register_forward_hook(make_steer_hook(delta, fired))
                try:
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, do_sample=False, num_beams=1,
                            max_new_tokens=512, temperature=1.0, top_p=1.0, use_cache=True,
                            pad_token_id=tok.pad_token_id)
                finally:
                    hook.remove()

                # Assert hook actually fired
                assert fired[0], f"Steering hook did NOT fire for study {sample['study_id']}"

                text = tok.decode(output_ids[0], skip_special_tokens=True).strip()
            else:
                unsteered_count += 1
                # Use stored baseline report (exact preservation)
                bl = bl_by_study.get(sample['study_id'], {})
                text = bl.get('candidate_findings', '')
                delta_norm = 0.0

            results.append({
                'study_id': sample['study_id'],
                'subject_id': sample['subject_id'],
                'section_findings': sample['section_findings'],
                'candidate_findings': text,
                'steered': bool(targets),
                'targets': targets,
                'probe_scores': probe_scores,
                'delta_norm': delta_norm,
            })

        except Exception as e:
            errors += 1
            log(f"  ERROR study {sample['study_id']}: {e}")
            bl = bl_by_study.get(sample['study_id'], {})
            results.append({
                'study_id': sample['study_id'], 'subject_id': sample['subject_id'],
                'section_findings': sample['section_findings'],
                'candidate_findings': bl.get('candidate_findings', ''),
                'steered': False, 'targets': [], '_error': str(e),
                'probe_scores': {}, 'delta_norm': 0.0,
            })

        done = i + 1
        if done % 50 == 0:
            json.dump({'results': results}, open(checkpoint_path, 'w'))
        if done % 100 == 0:
            elapsed = time.time() - t0; rate = (done-start_idx)/elapsed if elapsed>0 else 0
            eta = (len(manifest)-done)/rate if rate>0 else 0
            log(f"  {done}/{len(manifest)} ({rate:.1f}/s, ETA:{eta/60:.0f}m, "
                f"steered:{steered_count}, unsteered:{unsteered_count}, err:{errors})")

    log(f"\n  Generation complete: {len(results)}, steered:{steered_count}, "
        f"unsteered:{unsteered_count}, errors:{errors}")
    assert errors == 0, f"FATAL: {errors} errors"

    json.dump(results, open(results_path, 'w'), indent=2, ensure_ascii=False)
    if checkpoint_path.exists(): checkpoint_path.unlink()
    del model, proc; torch.cuda.empty_cache(); gc.collect()
    return results, PATHOLOGIES


def evaluate(results, PATHOLOGIES):
    """F1CheXbert eval + merged (Baseline ∨ Steered) evaluation."""
    log(f"\n{'='*60}")
    log(f"Evaluation")
    log(f"{'='*60}")

    clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip().lower()

    # Load baseline reports for merged eval
    baseline_reports = json.load(open(BASELINE_DIR / "chexagent_8b_reports.json"))
    bl_by_study = {r['study_id']: r for r in baseline_reports}
    pare_by_study = {r['study_id']: r for r in results}

    # ── PARE-only F1CheXbert ──
    candidates = [clean(s['candidate_findings']) for s in results]
    references = [clean(s['section_findings']) for s in results]
    pairs = [(c,r) for c,r in zip(candidates, references) if r]
    cands = [p[0] for p in pairs]; refs = [p[1] for p in pairs]
    log(f"  Total: {len(results)}, with ref: {len(pairs)}, skipped: {len(results)-len(pairs)}")

    from f1chexbert import F1CheXbert
    scorer = F1CheXbert(device=DEVICE)

    def to_json(obj):
        if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_json(v) for v in obj]
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        return obj

    log(f"  Running F1CheXbert (PARE)...")
    pare_scores = to_json(scorer(refs, cands))
    # F1CheXbert may return list of classification reports
    if isinstance(pare_scores, list) and len(pare_scores) >= 2:
        score_dict = pare_scores[1] if isinstance(pare_scores[1], dict) else {}
    elif isinstance(pare_scores, dict):
        score_dict = pare_scores
    else:
        score_dict = {}
    avgs = {k:v for k,v in score_dict.items() if 'avg' in k} if score_dict else {"raw": str(pare_scores)[:500]}
    log(f"  PARE F1CheXbert: {json.dumps(avgs, indent=2, default=str)}")

    # ── Merged (Baseline ∨ Steered): per-pathology ──
    log(f"\n  Building merged predictions (BL ∨ ST)...")
    LABELS = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
        "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis",
        "Pneumothorax","Pleural Effusion","Pleural Other","Fracture",
        "Support Devices","No Finding"]

    # Label both baseline and PARE reports
    def label_all(report_list, name):
        labels = []
        for i, s in enumerate(report_list):
            text = clean(s.get('candidate_findings', ''))
            if not text:
                labels.append([0]*14); continue
            raw = scorer.get_label(text, mode='rrg')
            labels.append([int(raw[j]) if j < len(raw) else 0 for j in range(14)])
            if (i+1)%500==0: log(f"    {name}: {i+1}/{len(report_list)}")
        return np.array(labels)

    # For studies with reference only
    ref_studies = [s for s in results if s['section_findings']]
    bl_for_eval = [bl_by_study[s['study_id']] for s in ref_studies if s['study_id'] in bl_by_study]
    st_for_eval = [pare_by_study[s['study_id']] for s in ref_studies if s['study_id'] in pare_by_study]

    log(f"  Labeling baseline reports...")
    bl_labels = label_all(bl_for_eval, "baseline")
    log(f"  Labeling PARE reports...")
    st_labels = label_all(st_for_eval, "PARE")
    log(f"  Labeling reference reports...")
    ref_for_label = [{'candidate_findings': clean(s['section_findings'])} for s in ref_studies]
    ref_labels = label_all(ref_for_label, "reference")

    # Merged = BL ∨ ST (if either says positive, merged says positive)
    merged_labels = np.maximum(bl_labels, st_labels)

    # Compute per-pathology F1 for baseline, PARE, merged
    def compute_f1_table(pred, ref_arr, name):
        log(f"\n  {'='*50}")
        log(f"  {name}")
        log(f"  {'Pathology':<35} {'TP':>5} {'FP':>5} {'FN':>5} {'F1':>7}")
        log(f"  {'-'*60}")
        per_f1 = {}
        for j, label in enumerate(LABELS):
            tp = int(((ref_arr[:,j]==1) & (pred[:,j]==1)).sum())
            fp = int(((ref_arr[:,j]==0) & (pred[:,j]==1)).sum())
            fn = int(((ref_arr[:,j]==1) & (pred[:,j]==0)).sum())
            f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
            per_f1[label] = f1
            log(f"  {label:<35} {tp:>5} {fp:>5} {fn:>5} {f1:>7.4f}")
        TARG5 = ["Atelectasis","Cardiomegaly","Lung Opacity","Pleural Effusion","Edema"]
        target_f1 = float(np.mean([per_f1[l] for l in TARG5]))
        log(f"\n  {name} Target-5 Macro-F1: {target_f1:.4f}")
        return per_f1, target_f1

    bl_pf, bl_t5 = compute_f1_table(bl_labels, ref_labels, "BASELINE (diagnostic)")
    st_pf, st_t5 = compute_f1_table(st_labels, ref_labels, "PARE (diagnostic)")
    mg_pf, mg_t5 = compute_f1_table(merged_labels, ref_labels, "MERGED BL∨ST (diagnostic)")

    # Load baseline official scores
    baseline_eval = json.load(open(BASELINE_DIR / "chexagent_8b_eval.json"))

    # ── Summary ──
    log(f"\n{'='*60}")
    log(f"  FINAL COMPARISON")
    log(f"{'='*60}")
    log(f"  Baseline Target-5:  {bl_t5:.4f}")
    log(f"  PARE Target-5:      {st_t5:.4f}  (Δ={st_t5-bl_t5:+.4f})")
    log(f"  Merged Target-5:    {mg_t5:.4f}  (Δ={mg_t5-bl_t5:+.4f})")
    log(f"  Baseline Official Micro-F1-14: {baseline_eval['official_f1chexbert'][1]['micro avg']['f1-score']:.4f}")

    n_steered = sum(1 for s in results if s.get('steered'))
    json.dump({
        'model': CHECKPOINT,
        'protocol': 'PARE standardized: same baseline + L3 steering',
        'pare_config': {'steer_layer': STEER_LAYER, 'probe_layer': PROBE_LAYER,
            'lambda': BEST_LAMBDA, 'n_vis': N_VIS},
        'n_studies': len(results), 'n_evaluated': len(pairs),
        'n_steered': n_steered, 'n_unsteered': len(results)-n_steered,
        'pare_official_f1chexbert': pare_scores,
        'diagnostic': {
            'baseline': {'per_pathology': bl_pf, 'target5': bl_t5},
            'pare': {'per_pathology': st_pf, 'target5': st_t5},
            'merged': {'per_pathology': mg_pf, 'target5': mg_t5},
        },
        'delta_target5_pare': st_t5 - bl_t5,
        'delta_target5_merged': mg_t5 - bl_t5,
    }, open(EXP_DIR / "pare_eval.json", "w"), indent=2)

    del scorer; torch.cuda.empty_cache(); gc.collect()
    log(f"\n✅ DONE — saved to pare_eval.json")


if __name__ == "__main__":
    results, PATHOLOGIES = generate_steered_reports()
    evaluate(results, PATHOLOGIES)
