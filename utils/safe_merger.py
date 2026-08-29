"""
Safe Append-Only Report Merger
==============================
Baseline text is IMMUTABLE. The only permitted operation is:

    MergedReport = BaselineReport + new_PARE_findings

Design:
  1. CheXbert labels the FULL baseline report  → BL pathology set
  2. CheXbert labels the FULL PARE report      → ST pathology set
  3. Genuinely new = (ST positive) AND (BL NOT positive) AND (in fired targets)
  4. For each genuinely new pathology, extract the responsible PARE sentence
  5. Append those sentences to the baseline report verbatim

Guarantees:
  - Baseline text is NEVER modified, deleted, or replaced
  - Can only add information, never remove it

NOT guaranteed:
  - F1(merged) >= F1(baseline) — appended sentences can introduce FP
  - Contradiction-free — baseline may negate what PARE appends
  - Pathology-isolated — appended sentences may carry collateral findings

On CheXbert failure: merge is ABORTED for that pathology (safe fallback).
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# CheXbert 14-label schema
LABELS_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]
LABEL_TO_IDX = {p: i for i, p in enumerate(LABELS_14)}


def _chexbert_label_report(text: str, scorer: Any) -> Optional[Dict[str, int]]:
    """CheXbert-label a FULL report. Returns None on failure (merge aborts)."""
    try:
        raw = scorer.get_label(text, mode="rrg")
        return {
            p: int(raw[i]) if i < len(raw) and raw[i] != "" else 0
            for p, i in LABEL_TO_IDX.items()
        }
    except Exception as e:
        logger.warning(f"CheXbert failed on report: {e}")
        return None  # Caller must abort merge


def _split_sentences(text: str) -> List[str]:
    """Split report into sentences."""
    if not text or not text.strip():
        return []
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]


def _label_sentence(sentence: str, scorer: Any) -> Optional[Dict[str, int]]:
    """CheXbert-label a single sentence. Returns None on failure."""
    try:
        raw = scorer.get_label(sentence, mode="rrg")
        return {
            p: int(raw[i]) if i < len(raw) and raw[i] != "" else 0
            for p, i in LABEL_TO_IDX.items()
        }
    except Exception:
        return None  # Sentence skipped in evidence extraction


def safe_merge(
    baseline_text: str,
    steered_text: str,
    fired_pathologies: List[str],
    scorer: Any,
) -> Tuple[str, Dict[str, str]]:
    """Append-only merge: baseline is immutable.

    Returns (merged_text, actions_dict).

    Actions per pathology:
      - 'already_in_baseline': BL already has this finding → no change
      - 'appended': genuinely new finding from PARE → appended
      - 'no_new_evidence': PARE didn't produce this finding either
      - 'not_fired': pathology wasn't in the fired set
    """
    actions: Dict[str, str] = {}

    if not baseline_text.strip() or not steered_text.strip():
        for p in fired_pathologies:
            actions[p] = "no_parseable_text"
        return baseline_text, actions

    # Step 1: CheXbert-label FULL reports (not sentences)
    bl_labels = _chexbert_label_report(baseline_text, scorer)
    st_labels = _chexbert_label_report(steered_text, scorer)

    # CheXbert failure → abort entire merge (return baseline unchanged)
    if bl_labels is None or st_labels is None:
        for p in fired_pathologies:
            actions[p] = "chexbert_failure_aborted"
        return baseline_text, actions

    # Step 2: Find genuinely new findings
    #   new = fired AND (PARE positive) AND (baseline NOT positive)
    new_pathologies = []
    for p in fired_pathologies:
        if p not in LABEL_TO_IDX:
            actions[p] = "unknown_pathology"
            continue
        if bl_labels.get(p, 0) == 1:
            actions[p] = "already_in_baseline"
            continue
        if st_labels.get(p, 0) == 1:
            new_pathologies.append(p)
        else:
            actions[p] = "no_new_evidence"

    # Step 3: If nothing new, return baseline unchanged
    if not new_pathologies:
        return baseline_text, actions

    # Step 4: Extract evidence sentences from PARE report
    st_sents = _split_sentences(steered_text)
    if not st_sents:
        for p in new_pathologies:
            actions[p] = "no_parseable_sentences"
        return baseline_text, actions

    # Label each PARE sentence to find which carries the evidence
    st_sent_labels = [_label_sentence(s, scorer) for s in st_sents]

    append_sents: List[str] = []
    used_sent_idx: set = set()  # avoid duplicating sentences

    for p in new_pathologies:
        # Find PARE sentences positive for this pathology
        # Skip sentences where CheXbert labeling failed (None)
        candidates = [
            j for j in range(len(st_sents))
            if st_sent_labels[j] is not None
            and st_sent_labels[j].get(p, 0) == 1
            and j not in used_sent_idx
        ]
        if candidates:
            # Take the first positive sentence (most relevant by position)
            best_j = candidates[0]
            append_sents.append(st_sents[best_j])
            used_sent_idx.add(best_j)
            actions[p] = "appended"
        else:
            # Full-report label was positive but no individual sentence is
            # (context-dependent detection). Append nothing — safe fallback.
            actions[p] = "no_sentence_evidence"

    # Step 5: Append to baseline (baseline text is NEVER modified)
    merged = baseline_text.rstrip()
    if append_sents:
        if not merged.endswith("."):
            merged += "."
        merged += " " + " ".join(append_sents)
        if not merged.endswith("."):
            merged += "."

    return merged, actions
