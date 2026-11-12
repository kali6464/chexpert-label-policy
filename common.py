"""Shared helpers with no torch dependency, so the split, resize and
comparison scripts can run on a login node."""
import math
import numpy as np

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
COMPETITION_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
]
COMPETITION_IDX = [ALL_LABELS.index(n) for n in COMPETITION_LABELS]

_PREFIXES = ("CheXpert-v1.0-small/", "CheXpert-v1.0/")


def strip_prefix(path):
    """'CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg'
    -> 'train/patient00001/study1/view1_frontal.jpg'"""
    p = str(path).lstrip("./")
    for pref in _PREFIXES:
        if p.startswith(pref):
            return p[len(pref):]
    return p


def patient_id(path):
    return strip_prefix(path).split("/")[-3]


def study_id(path):
    """Unique per study: 'patient00001/study1'."""
    return "/".join(strip_prefix(path).split("/")[-3:-1])


def positive_labels_for_policy(policy):
    """Which labels have their uncertain (-1) entries turned into positives.

    'zeros'               -> none (U-Zeros)
    'ignore'              -> none; train.py also drops uncertain entries from the loss (U-Ignore)
    'ones'                -> all 14 (U-Ones)
    'Atelectasis'         -> just that label (the earlier 'mixed' policy)
    'Atelectasis+Edema'   -> any '+'-joined list of label names
    """
    policy = str(policy).strip()
    if policy.lower() in ("zeros", "ignore"):
        return []
    if policy.lower() == "ones":
        return list(ALL_LABELS)
    names = [n.strip() for n in policy.split("+") if n.strip()]
    unknown = [n for n in names if n not in ALL_LABELS]
    if unknown:
        raise ValueError(f"Unknown label(s) in uncertainty_policy: {unknown}")
    return names


def resolve_labels(raw, policy="zeros", keep_uncertain=False):
    """raw: [N, 14] array with 1 / 0 / -1 (uncertain) / NaN (not mentioned).

    Returns (targets, uncertain_mask).
    NaN always becomes 0. Uncertain entries become 1 or 0 according to
    `policy`, unless keep_uncertain=True, in which case they stay -1 so the
    evaluation code can exclude them.
    """
    raw = np.asarray(raw, dtype=np.float32)
    uncertain = raw == -1
    out = np.nan_to_num(raw, nan=0.0)
    if not keep_uncertain:
        fill = np.zeros(len(ALL_LABELS), dtype=np.float32)
        for name in positive_labels_for_policy(policy):
            fill[ALL_LABELS.index(name)] = 1.0
        out = np.where(uncertain, fill[None, :], out)
    return out.astype(np.float32), uncertain


def lr_at(step, total_steps, warmup_steps, base_lr, min_ratio=0.01):
    """Per-step linear warmup, then cosine decay to base_lr * min_ratio."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    t = min(max(t, 0.0), 1.0)
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))
