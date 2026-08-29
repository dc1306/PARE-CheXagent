#!/usr/bin/env python3
"""
PARE Standardized Training — Part 1
Build all PARE components under the SAME protocol as the frozen baseline.
Extracts L3+L31 features, generates train baseline reports, builds Monge maps + probes.

ETA: ~15 hours for 15K train studies + 1.8K val studies
"""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch, json, sys, time, re, gc, glob, warnings
import pandas as pd, numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.linalg import sqrtm, inv
warnings.filterwarnings('ignore')

# Paths (SAME as baseline)
MIMIC_SPLIT = "/mnt/raid/obed/jamir/DATA1/MIMIC-CXR/mimic-cxr-2.0.0-split.csv"
MIMIC_META  = "/mnt/raid/obed/jamir/DATA1/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv"
CHEXPERT_CSV= "/mnt/raid/obed/jamir/DATA1/MIMIC-CXR/mimic-cxr-2.0.0-chexpert.csv"
REPORTS_DIR = "/mnt/raid/obed/avadhut/MIMIC/mimic-cxr-reports"
IMAGE_DIRS  = ["/mnt/raid/obed/Medical_MoE_Project/data/images/files",
               "/mnt/raid/obed/jamir/DATA1/MIMIC-CXR/files"]
BASELINE_DIR= Path("/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/canonical_baseline")
EXP_DIR     = Path("/mnt/raid/obed/Medical_MoE_Project/Medical-VLM-Enrichment/experiments/pare_standardized")
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE="cuda:0"; DTYPE=torch.float16; CHECKPOINT="StanfordAIMI/CheXagent-8b"
MAX_IMAGES=2; N_VIS=128; STEER_LAYER=3; PROBE_LAYER=31; PCA_DIMS=128
N_TRAIN=15000; BEST_LAMBDA=1.0  # Subsample from ~222K train studies
PATHOLOGIES=['Atelectasis','Cardiomegaly','Lung Opacity','Pleural Effusion','Edema']
# CheXpert CSV column names for GT labels
CHEXPERT_COLS={'Atelectasis':'Atelectasis','Cardiomegaly':'Cardiomegaly',
    'Lung Opacity':'Lung Opacity','Pleural Effusion':'Pleural Effusion','Edema':'Edema'}
LOG_FILE = EXP_DIR / "train.log"

def log(msg):
    ts=time.strftime("%Y-%m-%d %H:%M:%S"); line=f"[{ts}] {msg}"
    print(line,flush=True)
    with open(LOG_FILE,"a") as f: f.write(line+"\n")

def parse_report(rpath):
    with open(rpath) as f: text=f.read()
    sections={}
    for pat,name in [(r'(?:INDICATION|HISTORY|CLINICAL INFORMATION|CLINICAL HISTORY|REASON FOR EXAM(?:INATION)?)\s*:?\s*','indication'),
                     (r'FINDINGS?\s*:?\s*','findings'),(r'IMPRESSION\s*:?\s*','impression')]:
        for m in re.finditer(pat,text,re.IGNORECASE):
            sections.setdefault(name,[]).append((m.start(),m.end()))
    boundaries=[]
    for name,spans in sections.items():
        for s,e in spans: boundaries.append((s,e,name))
    boundaries.sort(key=lambda x:x[0]); result={}
    for i,(start,cstart,name) in enumerate(boundaries):
        end=boundaries[i+1][0] if i+1<len(boundaries) else len(text)
        c=re.sub(r'\s+',' ',text[cstart:end]).strip()
        if c and name not in result: result[name]=c
    return result

# ═══ STEP 0: Build train/val study manifests ═══
def build_manifest(split_name):
    log(f"Building {split_name} manifest...")
    split_df=pd.read_csv(MIMIC_SPLIT)
    df=split_df[split_df['split']==split_name]
    log(f"  {split_name} DICOMs: {len(df)}")
    all_jpgs={}
    for d in IMAGE_DIRS:
        if not os.path.exists(d): continue
        for p in glob.glob(f'{d}/**/*.jpg',recursive=True):
            all_jpgs[os.path.basename(p).replace('.jpg','')]=p
    meta_df=pd.read_csv(MIMIC_META)
    view_map=dict(zip(meta_df['dicom_id'],meta_df['ViewPosition'].fillna('')))
    # CheXpert GT labels — U-Ones policy: uncertain (-1) → positive (1)
    # Consistent with the baseline evaluation methodology
    def chexpert_uones(x):
        if pd.isna(x): return 0
        x = int(x)
        return 1 if x in (1, -1) else 0

    cp=pd.read_csv(CHEXPERT_CSV)
    cp_key={}
    for _,row in cp.iterrows():
        k=(int(row['subject_id']),int(row['study_id']))
        cp_key[k]={p:chexpert_uones(row.get(c,0)) for p,c in CHEXPERT_COLS.items()}
    studies=defaultdict(list)
    for _,row in df.iterrows():
        d=row['dicom_id']
        if d in all_jpgs:
            studies[(row['subject_id'],row['study_id'])].append({
                'dicom_id':d,'image_path':all_jpgs[d],'view':view_map.get(d,'')})
    manifest=[]
    for (sid,stid),images in studies.items():
        vp={'PA':0,'AP':1,'LATERAL':2,'LL':3}
        images.sort(key=lambda x:vp.get(x['view'].upper(),4))
        selected=images[:MAX_IMAGES]
        p_dir=f"p{str(sid)[:2]}"
        rpath=os.path.join(REPORTS_DIR,p_dir,f"p{sid}",f"s{stid}.txt")
        sections=parse_report(rpath) if os.path.exists(rpath) else {}
        gt=cp_key.get((int(sid),int(stid)),{p:0 for p in PATHOLOGIES})
        manifest.append({
            'subject_id':int(sid),'study_id':int(stid),
            'image_paths':[img['image_path'] for img in selected],
            'views':[img['view'] for img in selected],
            'n_images':len(selected),
            'section_indication':sections.get('indication',''),
            'section_findings':sections.get('findings',''),
            'gt_labels':gt,
        })
    log(f"  {split_name}: {len(manifest)} studies")
    return manifest

# ═══ STEP 1: Extract L3+L31 + generate baseline reports ═══
def extract_and_generate(manifest, split_name, generate=True):
    """Extract L3 visual tokens, L31 mean, and optionally generate baseline reports."""
    l3_path=EXP_DIR/f'{split_name}_l3.npy'
    l31_path=EXP_DIR/f'{split_name}_l31.npy'
    reports_path=EXP_DIR/f'{split_name}_reports.json'
    checkpoint=EXP_DIR/f'{split_name}_checkpoint.json'

    if l3_path.exists() and l31_path.exists() and (not generate or reports_path.exists()):
        log(f"Loading cached {split_name} features")
        return np.load(l3_path), np.load(l31_path), json.load(open(reports_path)) if generate else None

    from transformers import AutoModelForCausalLM, AutoProcessor
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,torch_dtype=DTYPE,trust_remote_code=True).to(DEVICE)
    model.eval()
    proc=AutoProcessor.from_pretrained(CHECKPOINT,trust_remote_code=True)
    tok=proc.tokenizer
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    dec_layers=model.language_model.model.layers

    # Resume
    start_idx=0; reports=[]
    l3_list=[]; l31_list=[]
    if checkpoint.exists():
        ckpt=json.load(open(checkpoint))
        start_idx=ckpt['idx']; reports=ckpt.get('reports',[])
        if os.path.exists(EXP_DIR/f'{split_name}_l3_partial.npy'):
            l3_list=list(np.load(EXP_DIR/f'{split_name}_l3_partial.npy'))
            l31_list=list(np.load(EXP_DIR/f'{split_name}_l31_partial.npy'))
        log(f"Resuming {split_name} from {start_idx}")

    log(f"\n{'='*60}")
    log(f"Extracting L3+L31 {'+ generating reports ' if generate else ''}({split_name})")
    log(f"  {len(manifest)} studies, starting from {start_idx}")
    log(f"{'='*60}")

    errors=0; t0=time.time()
    for i in range(start_idx, len(manifest)):
        sample=manifest[i]
        try:
            images=[Image.open(p).convert("RGB") for p in sample['image_paths']]
            indication=sample['section_indication'] or "None provided"
            raw_prompt=f'Given the indication: "{indication}", write a structured findings section for the CXR.'
            prompt=f' USER: <s>{raw_prompt} ASSISTANT: <s>'

            inputs=proc(images=images,text=prompt,return_tensors='pt')
            for k,v in inputs.items():
                if isinstance(v,torch.Tensor):
                    inputs[k]=v.to(DEVICE) if v.dtype in (torch.long,torch.int) else v.to(DEVICE,dtype=DTYPE)

            # Hooks to capture L3 visual tokens and L31 mean
            captured={'l3':None,'l31':None}
            def make_capture(layer_name,vis_only):
                def hook_fn(module,inp,output):
                    h=output[0] if isinstance(output,tuple) else output
                    if h.shape[1]>1:  # prefill only
                        if vis_only:
                            captured[layer_name]=h[0,:N_VIS,:].detach().cpu().half()
                        else:
                            captured[layer_name]=h[0].detach().cpu().half().mean(dim=0)
                    return output
                return hook_fn

            h3=dec_layers[STEER_LAYER].register_forward_hook(make_capture('l3',True))
            h31=dec_layers[PROBE_LAYER].register_forward_hook(make_capture('l31',False))
            try:
                if generate:
                    with torch.no_grad():
                        output_ids=model.generate(**inputs,do_sample=False,num_beams=1,
                            max_new_tokens=512,temperature=1.0,top_p=1.0,use_cache=True,
                            pad_token_id=tok.pad_token_id)
                    text=tok.decode(output_ids[0],skip_special_tokens=True).strip()
                    reports.append({'study_id':sample['study_id'],'subject_id':sample['subject_id'],
                        'candidate_findings':text})
                else:
                    with torch.no_grad():
                        _=model(**{k:v for k,v in inputs.items()})
            finally:
                h3.remove(); h31.remove()

            assert captured['l3'] is not None, "L3 hook did not fire"
            assert captured['l31'] is not None, "L31 hook did not fire"
            l3_list.append(captured['l3'].numpy())
            l31_list.append(captured['l31'].numpy())

        except Exception as e:
            errors+=1; log(f"  ERROR {split_name}[{i}]: {e}")
            l3_list.append(np.zeros((N_VIS,4096),dtype=np.float16))
            l31_list.append(np.zeros(4096,dtype=np.float16))
            if generate: reports.append({'study_id':sample['study_id'],'subject_id':sample['subject_id'],
                'candidate_findings':f'__ERROR__:{e}','_error':True})

        done=i+1
        if done%50==0:
            np.save(EXP_DIR/f'{split_name}_l3_partial.npy',np.array(l3_list))
            np.save(EXP_DIR/f'{split_name}_l31_partial.npy',np.array(l31_list))
            json.dump({'idx':done,'reports':reports},open(checkpoint,'w'))
        if done%100==0:
            elapsed=time.time()-t0; rate=(done-start_idx)/elapsed if elapsed>0 else 0
            eta=(len(manifest)-done)/rate if rate>0 else 0
            log(f"  {done}/{len(manifest)} ({rate:.1f}/s, ETA:{eta/60:.0f}m, err:{errors})")

    log(f"  {split_name} complete: {len(l3_list)} features, {errors} errors")
    assert errors == 0, f"FATAL: {errors} extraction errors in {split_name}"
    l3_arr=np.array(l3_list); l31_arr=np.array(l31_list)
    np.save(l3_path,l3_arr); np.save(l31_path,l31_arr)
    if generate: json.dump(reports,open(reports_path,'w'),indent=2,ensure_ascii=False)
    # Cleanup
    for f in [EXP_DIR/f'{split_name}_l3_partial.npy',EXP_DIR/f'{split_name}_l31_partial.npy',checkpoint]:
        if f.exists(): f.unlink()
    del model,proc; torch.cuda.empty_cache(); gc.collect()
    return l3_arr, l31_arr, reports if generate else None

# ═══ STEP 2: CheXbert label train reports → TP/FN ═══
def label_and_identify_tp_fn(manifest, reports):
    log(f"\nCheXbert-labeling train reports → TP/FN...")
    labels_path=EXP_DIR/'train_chexbert_labels.json'
    if labels_path.exists():
        log("  Loading cached labels")
        return json.load(open(labels_path))

    from f1chexbert import F1CheXbert
    scorer=F1CheXbert(device=DEVICE)
    CHEXBERT_IDX={'Cardiomegaly':1,'Lung Opacity':2,'Edema':4,'Atelectasis':7,'Pleural Effusion':9}
    labels={}
    for i,r in enumerate(reports):
        text=r.get('candidate_findings','')
        if not text or text.startswith('__ERROR__'):
            labels[i]={p:0 for p in PATHOLOGIES}; continue
        raw=scorer.get_label(text,mode='rrg')
        labels[i]={p:int(raw[CHEXBERT_IDX[p]]) if CHEXBERT_IDX[p]<len(raw) else 0 for p in PATHOLOGIES}
        if (i+1)%500==0: log(f"    Labeled {i+1}/{len(reports)}")
    json.dump(labels,open(labels_path,'w'),indent=2)
    del scorer; torch.cuda.empty_cache(); gc.collect()
    log(f"  Labeled {len(labels)} reports")
    return labels

# ═══ STEP 3: Train probes + thresholds + Monge maps ═══
def build_pare_components(train_manifest, train_l3, train_l31, train_labels,
                          val_manifest, val_l31):
    log(f"\n{'='*60}")
    log(f"Building PARE components")
    log(f"{'='*60}")

    gt_train={p:np.array([s['gt_labels'].get(p,0) for s in train_manifest]) for p in PATHOLOGIES}
    gt_val={p:np.array([s['gt_labels'].get(p,0) for s in val_manifest]) for p in PATHOLOGIES}

    # TP/FN from CheXbert baseline predictions vs GT
    tp_idx={p:[] for p in PATHOLOGIES}; fn_idx={p:[] for p in PATHOLOGIES}
    for i in range(len(train_manifest)):
        cb=train_labels.get(str(i),train_labels.get(i,{}))
        for p in PATHOLOGIES:
            gt=gt_train[p][i]; pred=cb.get(p,0)
            if gt==1 and pred==1: tp_idx[p].append(i)
            elif gt==1 and pred==0: fn_idx[p].append(i)
    for p in PATHOLOGIES:
        log(f"  {p:20s}: TP={len(tp_idx[p])}, FN={len(fn_idx[p])}")

    # ── Probes on L31 ──
    log("\n  Training L31 probes (train), thresholds (val)...")
    sc31=StandardScaler()
    d31_tr=sc31.fit_transform(train_l31.astype(np.float32))
    d31_va=sc31.transform(val_l31.astype(np.float32))

    probes={}; thresholds={}
    for p in PATHOLOGIES:
        clf=LogisticRegression(penalty='l1',solver='liblinear',C=0.01,
            class_weight='balanced',max_iter=5000,random_state=42)
        clf.fit(d31_tr,gt_train[p])
        val_scores=clf.predict_proba(d31_va)[:,1]
        fpr,tpr,th=roc_curve(gt_val[p],val_scores)
        best=np.argmax(tpr-fpr)
        thresholds[p]=float(th[best])
        probes[p]=clf
        log(f"    {p:20s}: threshold={thresholds[p]:.4f}")

    # ── Monge maps on L3 ──
    log("\n  Building Monge maps (FN→TP in L3 space)...")
    def compute_monge(X_s,X_t):
        mu_s=X_s.mean(0); mu_t=X_t.mean(0); d=X_s.shape[1]
        Sig_s=LedoitWolf().fit(X_s).covariance_ if X_s.shape[0]>d else np.cov(X_s.T)+1e-3*np.eye(d)
        Sig_t=LedoitWolf().fit(X_t).covariance_ if X_t.shape[0]>d else np.cov(X_t.T)+1e-3*np.eye(d)
        S_sqrt=np.real(sqrtm(Sig_s)); S_inv_sqrt=np.real(inv(S_sqrt))
        M=S_sqrt@Sig_t@S_sqrt; A=S_inv_sqrt@np.real(sqrtm(M))@S_inv_sqrt
        return mu_s.astype(np.float32),mu_t.astype(np.float32),((A+A.T)/2).astype(np.float32)

    monge_maps={}
    for p in PATHOLOGIES:
        tp_i=np.array(tp_idx[p]); fn_i=np.array(fn_idx[p])
        if len(tp_i)<10 or len(fn_i)<10:
            log(f"    {p}: SKIP (TP={len(tp_i)}, FN={len(fn_i)} too small)")
            continue
        tp_tok=train_l3[tp_i].reshape(-1,4096).astype(np.float32)
        fn_tok=train_l3[fn_i].reshape(-1,4096).astype(np.float32)
        all_tok=np.vstack([tp_tok,fn_tok])
        sc=StandardScaler().fit(all_tok)
        tp_sc=sc.transform(tp_tok); fn_sc=sc.transform(fn_tok)
        rng=np.random.RandomState(42)
        n_fit=min(100000,len(tp_sc)+len(fn_sc))
        n_tp_s=int(n_fit*len(tp_sc)/(len(tp_sc)+len(fn_sc)))
        tp_sub=tp_sc[rng.choice(len(tp_sc),min(n_tp_s,len(tp_sc)),replace=False)]
        fn_sub=fn_sc[rng.choice(len(fn_sc),min(n_fit-n_tp_s,len(fn_sc)),replace=False)]
        pca=PCA(n_components=PCA_DIMS,random_state=42).fit(np.vstack([tp_sub,fn_sub]))
        mu_s,mu_t,A=compute_monge(pca.transform(fn_sc),pca.transform(tp_sc))
        monge_maps[p]={'A':A,'mu_s':mu_s,'mu_t':mu_t,
            'sc_mean':sc.mean_.astype(np.float32),'sc_scale':sc.scale_.astype(np.float32),
            'pca_mean':pca.mean_.astype(np.float32),'pca_comp':pca.components_.astype(np.float32)}
        log(f"    {p}: TP={len(tp_i)}, FN={len(fn_i)}, var={pca.explained_variance_ratio_.sum():.4f}")
        del tp_tok,fn_tok; gc.collect()

    # Save components with provenance metadata
    import pickle, hashlib
    manifest_hash = hashlib.md5(json.dumps([s['study_id'] for s in train_manifest]).encode()).hexdigest()
    pickle.dump({
        'probes':probes,'thresholds':thresholds,'monge_maps':monge_maps,
        'sc31':sc31,'pathologies':PATHOLOGIES,
        # Provenance — verified by pare_standardized_test.py
        'checkpoint':CHECKPOINT,'steer_layer':STEER_LAYER,'probe_layer':PROBE_LAYER,
        'n_vis':N_VIS,'pca_dims':PCA_DIMS,'lambda':BEST_LAMBDA,
        'n_train':len(train_manifest),'train_manifest_hash':manifest_hash,
    },open(EXP_DIR/'pare_components.pkl','wb'))
    log(f"\n  Saved PARE components (hash={manifest_hash[:8]}) to {EXP_DIR/'pare_components.pkl'}")
    return probes,thresholds,monge_maps,sc31


# ═══ MAIN ═══
if __name__=="__main__":
    log(f"\n{'='*60}")
    log(f"PARE Standardized Training Pipeline")
    log(f"{'='*60}")

    # Step 0: Build manifests
    train_full=build_manifest('train')
    val_manifest=build_manifest('validate')
    json.dump(val_manifest,open(EXP_DIR/'val_manifest.json','w'),indent=2)

    # Subsample train
    rng=np.random.RandomState(42)
    train_idx=np.sort(rng.choice(len(train_full),min(N_TRAIN,len(train_full)),replace=False))
    train_manifest=[train_full[i] for i in train_idx]
    json.dump(train_manifest,open(EXP_DIR/'train_manifest.json','w'),indent=2)
    np.save(EXP_DIR/'train_idx.npy',train_idx)
    log(f"Subsampled {len(train_manifest)} train studies from {len(train_full)}")

    # Step 1a: Train features + baseline reports
    train_l3,train_l31,train_reports=extract_and_generate(train_manifest,'train',generate=True)

    # Step 1b: Val features (no generation needed)
    _,val_l31,_=extract_and_generate(val_manifest,'val',generate=False)

    # Step 2: CheXbert label train → TP/FN
    train_labels=label_and_identify_tp_fn(train_manifest,train_reports)

    # Step 3: Build probes + Monge maps
    probes,thresholds,monge_maps,sc31=build_pare_components(
        train_manifest,train_l3,train_l31,train_labels,val_manifest,val_l31)

    log(f"\n✅ Part 1 COMPLETE — PARE components ready")
    log(f"Run pare_standardized_test.py next")
