"""Score every finished CheXpert run on an external, radiologist-labeled data set.

  python score_external.py --name vindr --csv /path/external/vindr_test.csv --img_dir /path/external
  python score_external.py --name vindr --csv ... --img_dir ... --summary_only   # rebuild tables only

Run from the CheXpert_pipeline folder (it imports data_loader, evaluate, model, score_test).
No training and no model selection: each run's already-selected checkpoint is scored.

Analysis plan, fixed before scoring:
  * Evaluation is per image. A pathology is used if it has at least --min_pos positives and
    --min_pos negatives; the summary metric is the mean AUC over the pathologies used.
  * Primary comparison: Atelectasis + Edema against U-Zeros, for each backbone, on three-seed
    ensembles (averaged probabilities), paired bootstrap, 5,000 resamples. With --cluster nih
    the bootstrap resamples patients (all images of a patient together), which is required when
    a patient has several images; otherwise it resamples images.
  * Secondary: Atelectasis against U-Zeros, Atelectasis + Edema against Atelectasis,
    U-Ignore against U-Zeros (if U-Ignore runs exist), and each pair of backbones under
    Atelectasis + Edema. Everything else is exploratory.

Per run it writes external_<name>_predictions.npz into the run folder, and the tables go to
<runs_root>/external_<name>_summary.txt.
"""
import argparse
import os
import re

import numpy as np

from common import COMPETITION_LABELS, COMPETITION_IDX
from score_test import find_runs, split_name

POLICIES = {"zeros": "U-Zeros", "Atelectasis": "Atelectasis", "Atelectasis+Edema": "Atel.+Edema",
            "ignore": "U-Ignore", "ones": "U-Ones"}
BACKBONES = ["densenet121", "deit_base_distilled_patch16_384", "convnext_base"]


def score_run(run_dir, csv, img_dir, out_path, device):
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from data_loader import CheXpertDataset, get_transforms
    from evaluate import predict
    from model import MultiLabelNet

    with open(os.path.join(run_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)
    ds = CheXpertDataset(csv, img_dir, get_transforms(int(cfg["img_size"]), False), keep_uncertain=True, views="all")
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]) * 2, shuffle=False,
                        num_workers=int(cfg.get("num_workers", 8)), pin_memory=True)
    model = MultiLabelNet(cfg["backbone"], dropout=float(cfg.get("dropout", 0.0))).to(device)
    state = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state["model"])
    y_true, y_prob = predict(model, loader, device, bool(cfg.get("use_tta", False)), bool(cfg.get("use_amp", True)))
    np.savez(out_path, paths=np.array(ds.paths), y_true=y_true, y_prob=y_prob)
    del model


def auc(yt, yp):
    from sklearn.metrics import roc_auc_score
    keep = yt != -1
    yt, yp = yt[keep], yp[keep]
    return None if np.unique(yt).size < 2 else float(roc_auc_score(yt, yp))


def usable_labels(y_true, min_pos):
    out = []
    for name, j in zip(COMPETITION_LABELS, COMPETITION_IDX):
        col = y_true[:, j]
        if (col == 1).sum() >= min_pos and (col == 0).sum() >= min_pos:
            out.append((name, j))
    return out


def mean_auc(y_true, y_prob, labels):
    vals = [auc(y_true[:, j], y_prob[:, j]) for _, j in labels]
    return None if any(v is None for v in vals) else float(np.mean(vals)), vals


def patient_groups(paths, scheme):
    """List of index arrays, one per patient. scheme 'nih': patient = digits before '_' in the file name."""
    if scheme is None:
        return None
    if scheme != "nih":
        raise SystemExit(f"unknown --cluster scheme {scheme!r}")
    keys = [re.match(r"(\d+)_", os.path.basename(str(p))).group(1) for p in paths]
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    return [np.array(v) for v in groups.values()]


def paired(y_true, pa, pb, labels, n_boot=5000, seed=0, groups=None):
    rng = np.random.default_rng(seed)
    n, d = len(y_true), []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        ma, _ = mean_auc(y_true[idx], pa[idx], labels)
        mb, _ = mean_auc(y_true[idx], pb[idx], labels)
        if ma is not None and mb is not None:
            d.append(ma - mb)
    d = np.array(d)
    full = mean_auc(y_true, pa, labels)[0] - mean_auc(y_true, pb, labels)[0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return full, lo, hi, float((d > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short name, e.g. vindr or reflacx")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--runs_root", default="./runs")
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--cluster", default=None, help="resample patients instead of images; 'nih' for ChestX-ray14")
    a = ap.parse_args()

    fname = f"external_{a.name}_predictions.npz"
    runs = [d for d in find_runs(a.runs_root) if split_name(os.path.basename(d))[1] in POLICIES]
    print(f"{len(runs)} finished runs found")
    if not a.summary_only:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, d in enumerate(runs, 1):
            out = os.path.join(d, fname)
            if os.path.exists(out):
                print(f"[{i}/{len(runs)}] {os.path.basename(d)}: already scored", flush=True)
                continue
            print(f"[{i}/{len(runs)}] {os.path.basename(d)}: scoring", flush=True)
            score_run(d, a.csv, a.img_dir, out, device)

    preds = {}
    for d in runs:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            z = np.load(p)
            preds[os.path.basename(d)] = (z["paths"], z["y_true"], z["y_prob"])
    if not preds:
        raise SystemExit("No external predictions found")
    ref_paths, y_true, _ = next(iter(preds.values()))
    for k, (p, yt, _) in preds.items():
        if not (np.array_equal(p, ref_paths) and np.array_equal(yt, y_true)):
            raise SystemExit(f"{k} was scored on different images")
    labels = usable_labels(y_true, a.min_pos)
    pgroups = patient_groups(ref_paths, a.cluster)
    L = [f"External test: {a.name}, {len(y_true)} images"
         + (f", {len(pgroups)} patients (bootstrap resamples patients)" if pgroups is not None else " (bootstrap resamples images)"),
         "Pathologies used (positives / negatives): " + ", ".join(
             f"{n} ({int((y_true[:, j] == 1).sum())}/{int((y_true[:, j] == 0).sum())})" for n, j in labels),
         "Not used: " + (", ".join(n for n in COMPETITION_LABELS if n not in dict(labels)) or "none"), ""]

    L.append(f"{'run':52s} {'mean':>6s} " + " ".join(f"{n[:6]:>6s}" for n, _ in labels))
    groups = {}
    for k in sorted(preds):
        m, vals = mean_auc(y_true, preds[k][2], labels)
        L.append(f"{k:52s} {m:.4f} " + " ".join(f"{v:.4f}" for v in vals))
        b, pol, _ = split_name(k)
        groups.setdefault((b, pol), []).append(m)
    L += ["", f"{'backbone':32s} {'policy':14s} {'n':>2s} {'mean':>6s} {'sd':>6s}"]
    for (b, pol), v in sorted(groups.items()):
        sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
        L.append(f"{b:32s} {POLICIES[pol]:14s} {len(v):2d} {np.mean(v):.4f} {sd:.4f}")

    def ens(b, pol):
        ks = [k for k in preds if split_name(k)[0] == b and split_name(k)[1] == pol and split_name(k)[2] in (42, 43, 44)]
        return (np.mean([preds[k][2] for k in ks], axis=0), len(ks)) if ks else (None, 0)

    L += ["", "Paired bootstrap on seed ensembles (difference in mean AUC, first minus second)"]
    comps = [("Atelectasis+Edema", "zeros", "PRIMARY"), ("Atelectasis", "zeros", ""),
             ("Atelectasis+Edema", "Atelectasis", ""), ("ignore", "zeros", "")]
    for pa, pb, tag in comps:
        for b in BACKBONES:
            ea, na = ens(b, pa)
            eb, nb = ens(b, pb)
            if ea is None or eb is None:
                continue
            diff, lo, hi, p = paired(y_true, ea, eb, labels, a.n_boot, groups=pgroups)
            star = " *" if lo > 0 or hi < 0 else ""
            L.append(f"  {POLICIES[pa]:12s} vs {POLICIES[pb]:12s} {b:32s} ({na} vs {nb} seeds) "
                     f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}] P>0 {p:.3f}{star} {tag}")
    for i in range(3):
        for j in range(i + 1, 3):
            ea, _ = ens(BACKBONES[j], "Atelectasis+Edema")
            eb, _ = ens(BACKBONES[i], "Atelectasis+Edema")
            if ea is None or eb is None:
                continue
            diff, lo, hi, p = paired(y_true, ea, eb, labels, a.n_boot, groups=pgroups)
            star = " *" if lo > 0 or hi < 0 else ""
            L.append(f"  {BACKBONES[j]} - {BACKBONES[i]} (Atel.+Edema): {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] P>0 {p:.3f}{star}")
    L.append("\n* interval excludes 0.")
    text = "\n".join(L)
    print(text)
    out = os.path.join(a.runs_root, f"external_{a.name}{'_patients' if pgroups is not None else ''}_summary.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
