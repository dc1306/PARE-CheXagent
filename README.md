# PARE: Probing and Representation Engineering for CheXagent-8B

**Inference-time pathology sensitivity recovery for frozen medical VLMs.**

PARE improves CheXagent-8B's diagnostic sensitivity without finetuning by:
1. **Probing** (L31) hidden representations to detect under-reported pathologies
2. **Steering** (L3) visual token representations via Monge OT transport maps
3. **Merging** steered findings into baseline reports (append-only, baseline-immutable)

## Results

### Primary Evaluation: MIMIC-CXR (N=3,269 test studies)

| Metric | Baseline | PARE | Merged Text | Δ |
|--------|:--------:|:----:|:-----------:|:-:|
| Target-5 Macro-F1 | 0.5254 | 0.5452 | **0.5727** | **+0.0473** |
| Micro-F1-14 | 0.4592 | 0.4816 | **0.4945** | **+0.0353** |
| Macro-F1-14 | 0.3307 | 0.3366 | **0.3494** | **+0.0187** |

**Statistical significance:** Bootstrap 95% CI for Δ(Merged−BL) = [+0.039, +0.056], entirely above zero

**L31 Probe AUROC:** Mean 0.7929 across Target-5 pathologies

### External Validation: VinDr-CXR (N=3,000 test images)

Cross-dataset generalization using MIMIC-trained probes + OT maps evaluated on independently radiologist-annotated Vietnamese CXRs. See `configs/pare_vindr.yaml`.

## Pipeline

```
scripts/
├── 00_prepare_mimic.py         # MIMIC-CXR manifest + U-Ones GT labels
├── 01_baseline_generation.py   # Frozen CheXagent-8B baseline reports
├── 02_pare_train.py            # Train L31 probes + L3 Monge OT maps
├── 03_pare_test.py             # Probe-gated steering on test set
├── 04_merge_eval.py            # Safe append-only text merger + F1CheXbert
├── 05_significance_tests.py    # Bootstrap CI + McNemar's tests
└── 06_vindr_evaluation.py      # VinDr cross-dataset generalization

configs/
├── pare_mimic.yaml             # MIMIC-CXR experiment settings
└── pare_vindr.yaml             # VinDr-CXR evaluation settings

utils/
└── safe_merger.py              # Append-only report merger (baseline immutable)
```

## Setup

```bash
pip install -r requirements.txt
```

### Data Access

- **MIMIC-CXR**: Requires PhysioNet credentialed access
- **VinDr-CXR**: Requires PhysioNet credentialed access + CITI training
- **CheXagent-8B**: HuggingFace model `StanfordAIMI/CheXagent-8b`

### Reproduction

```bash
# 1. Prepare data manifest
export MIMIC_ROOT=/path/to/MIMIC-CXR
python scripts/00_prepare_mimic.py

# 2. Generate baseline reports
python scripts/01_baseline_generation.py

# 3. Train PARE (probes + OT maps)
python scripts/02_pare_train.py

# 4. Run PARE on test set
python scripts/03_pare_test.py

# 5. Merge reports + evaluate
python scripts/04_merge_eval.py

# 6. Statistical significance
python scripts/05_significance_tests.py
```

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

## Expected Outputs

Running the full pipeline produces (not committed):

```
outputs/
├── test_manifest.json          # From 00_prepare_mimic.py
├── chexagent_8b_reports.json   # From 01_baseline_generation.py
├── pare_components.pkl         # From 02_pare_train.py (probes, maps, scalers)
├── pare_test_reports.json      # From 03_pare_test.py
├── pare_merged_reports.json    # From 04_merge_eval.py
├── pare_merged_eval.json       # From 04_merge_eval.py
├── pare_all_reports.csv        # From 04_merge_eval.py (master CSV)
└── significance.out            # From 05_significance_tests.py
```

## Target Pathologies

Atelectasis, Cardiomegaly, Lung Opacity, Pleural Effusion, Edema

## License

Research use only. Requires MIMIC-CXR and VinDr-CXR data use agreements.
