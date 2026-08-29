# PARE: Probing and Representation Engineering for CheXagent-8B

**Inference-time pathology sensitivity recovery for frozen medical VLMs.**

PARE improves CheXagent-8B's diagnostic sensitivity without finetuning by:
1. **Probing** (L31) hidden representations to detect under-reported pathologies
2. **Steering** (L3) visual token representations via Monge OT transport maps
3. **Merging** steered findings into baseline reports (append-only, baseline-immutable)

## Results (MIMIC-CXR Official Test Split, N=3,269)

| Metric | Baseline | PARE | Merged Text | Δ |
|--------|:--------:|:----:|:-----------:|:-:|
| Target-5 Macro-F1 | 0.5254 | 0.5452 | **0.5727** | **+0.0473** |
| Micro-F1-14 | 0.4592 | 0.4816 | **0.4945** | **+0.0353** |
| Macro-F1-14 | 0.3307 | 0.3366 | **0.3494** | **+0.0187** |

**Statistical significance:** Bootstrap 95% CI for Δ(Merged−BL) = [+0.039, +0.056], p=1.000

**L31 Probe AUROC:** Mean 0.7929 across Target-5 pathologies

## Pipeline

```
01_baseline_generation.py   → Frozen CheXagent-8B baseline reports
02_pare_train.py            → Train L31 probes + L3 Monge OT maps (15K studies)
03_pare_test.py             → Probe-gated steering on test set (3,269 studies)
04_merge_eval.py            → Safe append-only text merger + F1CheXbert eval
05_significance_tests.py    → Bootstrap CI + McNemar's tests
06_vindr_evaluation.py      → Cross-dataset generalization (VinDr-CXR)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- `transformers`, `f1chexbert` (requires `scikit-learn==1.5.2`)
- `scipy`, `scikit-learn`
- Access to MIMIC-CXR dataset (PhysioNet credentialed)
- CheXagent-8B model weights (`StanfordAIMI/CheXagent-8b`)

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | CheXagent-8B |
| Probe layer | L31 |
| Steering layer | L3 |
| λ (OT strength) | 1.0 |
| PCA dimensions | 128 |
| Visual tokens steered | 128 |
| GT labeling | U-Ones (uncertain → positive) |
| Decoding | Greedy (beam=1, deterministic) |

## Protocol

1. **Baseline**: Generate reports with frozen CheXagent-8B
2. **Feature extraction**: Extract L3 (128 visual tokens) and L31 (mean-pooled) representations
3. **Probe training**: Logistic regression on L31 features per pathology
4. **OT map construction**: Monge maps from FN→TP distributions in PCA-128 space
5. **Gated steering**: Apply L3 transport only when L31 probe fires above threshold
6. **Report merging**: Append genuinely new PARE findings to immutable baseline text
7. **Evaluation**: F1CheXbert against radiologist reference reports

## Target Pathologies

Atelectasis, Cardiomegaly, Lung Opacity, Pleural Effusion, Edema

## License

Research use only. Requires MIMIC-CXR data use agreement.
