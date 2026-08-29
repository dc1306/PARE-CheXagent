"""
00_prepare_mimic.py — MIMIC-CXR Data Preparation
=================================================
Prepares the official MIMIC-CXR test manifest from raw PhysioNet files.

Inputs:
  - mimic-cxr-2.0.0-chexpert.csv   (CheXpert labels)
  - mimic-cxr-2.0.0-split.csv      (official train/val/test split)
  - mimic-cxr-reports/              (sectioned radiology reports)
  - files/                          (DICOM-derived JPGs)

Output:
  - test_manifest.json: one entry per test study with:
      study_id, subject_id, image_paths (≤2, frontal priority),
      section_findings (reference text), U-Ones GT labels

Labelling Policy (U-Ones):
  CheXpert uncertain (-1) → positive (1)
  CheXpert positive  (1)  → positive (1)
  CheXpert negative  (0)  → negative (0)
  CheXpert NaN/blank      → negative (0)
"""

import os, json, re, csv, sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Configuration ──
MIMIC_ROOT = os.environ.get("MIMIC_ROOT", "/path/to/MIMIC-CXR")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
CHEXPERT_CSV = os.path.join(MIMIC_ROOT, "mimic-cxr-2.0.0-chexpert.csv")
SPLIT_CSV = os.path.join(MIMIC_ROOT, "mimic-cxr-2.0.0-split.csv")
REPORTS_DIR = os.path.join(MIMIC_ROOT, "mimic-cxr-reports")
IMAGE_DIR = os.path.join(MIMIC_ROOT, "files")

TARGET_PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Lung Opacity",
    "Pleural Effusion", "Edema"
]

# CheXpert column → GT key mapping
GT_COLUMNS = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Lung Opacity": "Lung Opacity",
    "Pleural Effusion": "Pleural Effusion",
    "Edema": "Edema",
}

# CheXbert 14-label indices for evaluation
CHEXBERT_INDICES = {
    "Cardiomegaly": 1, "Lung Opacity": 2, "Edema": 4,
    "Atelectasis": 7, "Pleural Effusion": 9,
}


def uones(val):
    """U-Ones: uncertain (-1) → positive (1)."""
    if pd.isna(val):
        return 0
    v = int(float(val))
    return 1 if v in (1, -1) else 0


def find_images(subject_id, study_id, image_dir, max_images=2):
    """Find ≤2 images for a study, frontal-priority."""
    sid = str(subject_id)
    prefix = f"p{sid[:2]}"
    study_dir = os.path.join(image_dir, prefix, f"p{sid}", f"s{study_id}")

    if not os.path.isdir(study_dir):
        return []

    jpgs = sorted([f for f in os.listdir(study_dir) if f.endswith(".jpg")])
    if not jpgs:
        return []

    # Frontal priority: prefer PA/AP views
    frontal = [f for f in jpgs if any(v in f.lower() for v in ["frontal", "pa", "ap"])]
    lateral = [f for f in jpgs if f not in frontal]

    selected = frontal[:max_images]
    remaining = max_images - len(selected)
    if remaining > 0:
        selected += lateral[:remaining]

    return [os.path.join(study_dir, f) for f in selected[:max_images]]


def load_findings(subject_id, study_id, reports_dir):
    """Load findings section from sectioned reports."""
    sid = str(subject_id)
    prefix = f"p{sid[:2]}"
    report_path = os.path.join(reports_dir, prefix, f"p{sid}", f"s{study_id}.txt")

    if not os.path.isfile(report_path):
        return ""

    text = open(report_path).read()
    # Extract FINDINGS section
    match = re.search(
        r"FINDINGS?[:\s]*\n(.*?)(?=\n\s*(?:IMPRESSION|$))",
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    return ""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading CheXpert labels from {CHEXPERT_CSV}...")
    chexpert = pd.read_csv(CHEXPERT_CSV)

    print(f"Loading splits from {SPLIT_CSV}...")
    splits = pd.read_csv(SPLIT_CSV)

    # Filter to test split
    test_dicoms = splits[splits["split"] == "test"]["dicom_id"].values
    test_studies = splits[splits["split"] == "test"][["subject_id", "study_id"]].drop_duplicates()

    print(f"Test studies: {len(test_studies)}")

    # Build manifest
    manifest = []
    for _, row in test_studies.iterrows():
        subject_id = int(row["subject_id"])
        study_id = int(row["study_id"])

        # Find images
        images = find_images(subject_id, study_id, IMAGE_DIR)
        if not images:
            continue

        # Get CheXpert labels (U-Ones)
        study_labels = chexpert[chexpert["study_id"] == study_id]
        gt = {}
        for path_name, col_name in GT_COLUMNS.items():
            if col_name in study_labels.columns and len(study_labels) > 0:
                gt[f"y_uo_{path_name}"] = uones(study_labels[col_name].iloc[0])
            else:
                gt[f"y_uo_{path_name}"] = 0

        # Load reference findings
        findings = load_findings(subject_id, study_id, REPORTS_DIR)

        entry = {
            "study_id": study_id,
            "subject_id": subject_id,
            "image_paths": images,
            "section_findings": findings,
            **gt,
        }
        manifest.append(entry)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "test_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=1)

    # Stats
    with_ref = sum(1 for m in manifest if m["section_findings"])
    prevalence = {
        p: sum(m.get(f"y_uo_{p}", 0) for m in manifest)
        for p in TARGET_PATHOLOGIES
    }

    print(f"\nManifest: {len(manifest)} studies")
    print(f"With reference findings: {with_ref}")
    print(f"Prevalence (U-Ones):")
    for p, count in prevalence.items():
        print(f"  {p}: {count}/{len(manifest)} ({100*count/len(manifest):.1f}%)")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
