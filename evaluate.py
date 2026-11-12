import numpy as np
from sklearn.metrics import roc_auc_score

from common import ALL_LABELS, COMPETITION_IDX, COMPETITION_LABELS


def predict(model, loader, device, use_tta=False, use_amp=True):
    """Returns (y_true [N,14], y_prob [N,14]) in loader order."""
    import torch

    model.eval()
    ys, ps = [], []
    amp_on = use_amp and device.type == "cuda"
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_on):
                logits = model(images)
                if use_tta:
                    logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
            ys.append(labels.numpy())
            ps.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _auc(y_true, y_prob):
    """AUC over entries that are not uncertain (-1); None if only one class."""
    keep = y_true != -1
    yt, yp = y_true[keep], y_prob[keep]
    if np.unique(yt).size < 2:
        return None
    return float(roc_auc_score(yt, yp))


def summarize(y_true, y_prob):
    per_class = {name: _auc(y_true[:, i], y_prob[:, i]) for i, name in enumerate(ALL_LABELS)}
    comp = [per_class[n] for n in COMPETITION_LABELS if per_class[n] is not None]
    defined = [v for v in per_class.values() if v is not None]
    return {
        "mean_competition": float(np.mean(comp)) if comp else float("nan"),
        "mean_all_defined": float(np.mean(defined)) if defined else float("nan"),
        "n_labels_defined": len(defined),
        "per_class": per_class,
    }


def aggregate_by_study(y_true, y_prob, study_ids):
    """CheXpert protocol: a study's score is the max over its views."""
    uniq, inv = np.unique(np.asarray(study_ids), return_inverse=True)
    t = np.full((len(uniq), y_true.shape[1]), -np.inf)
    p = np.full((len(uniq), y_prob.shape[1]), -np.inf)
    np.maximum.at(t, inv, y_true)
    np.maximum.at(p, inv, y_prob)
    return t.astype(np.float32), p.astype(np.float32), uniq


def bootstrap_samples(y_true, y_probs, n_boot=2000, seed=0):
    """Resample rows (studies) with replacement.

    y_probs: list of [N,14] arrays from different models on the SAME rows,
    so the same resample is applied to each (a paired bootstrap).
    Returns array [n_boot, n_models, 6]: five competition AUCs then their mean.
    Resamples where a label has a single class are NaN.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = np.full((n_boot, len(y_probs), len(COMPETITION_IDX) + 1), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        for m, yp in enumerate(y_probs):
            aucs = [_auc(yt[:, j], yp[idx, j]) for j in COMPETITION_IDX]
            if any(a is None for a in aucs):
                continue
            out[b, m, :-1] = aucs
            out[b, m, -1] = np.mean(aucs)
    return out


def bootstrap_ci(y_true, y_prob, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI for each competition label and for their mean."""
    s = bootstrap_samples(y_true, [y_prob], n_boot, seed)[:, 0, :]
    lo = np.nanpercentile(s, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(s, 100 * (1 - alpha / 2), axis=0)
    names = COMPETITION_LABELS + ["mean_competition"]
    return {n: [float(lo[i]), float(hi[i])] for i, n in enumerate(names)}


def format_report(stats, ci=None, prefix=""):
    def fmt(name, value):
        if value is None:
            return f"{prefix}  {name:22s} n/a"
        line = f"{prefix}  {name:22s} {value:.4f}"
        if ci and name in ci:
            line += f"  [{ci[name][0]:.4f}, {ci[name][1]:.4f}]"
        return line

    lines = [fmt("mean_competition", stats["mean_competition"])]
    lines += [fmt(n, stats["per_class"][n]) for n in COMPETITION_LABELS]
    return "\n".join(lines)
