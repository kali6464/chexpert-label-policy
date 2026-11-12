"""Is run A better than run B on the official validation set, or is the gap noise?

  python compare.py runs/convnext_base__Atelectasis__s42 runs/convnext_base__zeros__s42

Uses a paired bootstrap over studies: both models are scored on the same
resampled studies, and the difference A - B is collected each time.
If the 95% interval for a difference excludes 0, the data support a real gap.
"""
import argparse
import os

import numpy as np

from common import COMPETITION_LABELS
from evaluate import bootstrap_samples


def _load(run_dir):
    d = np.load(os.path.join(run_dir, "official_val_predictions.npz"))
    order = np.argsort(d["study_ids"])
    return d["study_ids"][order], d["y_true_study"][order], d["y_prob_study"][order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids_a, yt_a, yp_a = _load(args.run_a)
    ids_b, yt_b, yp_b = _load(args.run_b)
    if not (np.array_equal(ids_a, ids_b) and np.array_equal(yt_a, yt_b)):
        raise SystemExit("The two runs were not scored on the same studies (check the 'views' setting).")

    s = bootstrap_samples(yt_a, [yp_a, yp_b], args.n_boot, args.seed)
    diff = s[:, 0, :] - s[:, 1, :]
    names = COMPETITION_LABELS + ["mean_competition"]
    print(f"A = {args.run_a}\nB = {args.run_b}\n{len(ids_a)} studies, {args.n_boot} resamples\n")
    print(f"{'':22s} {'A':>7s} {'B':>7s} {'A-B':>8s}   95% CI of A-B        P(A>B)")
    for i, name in enumerate(names):
        d = diff[:, i][~np.isnan(diff[:, i])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        flag = "  *" if (lo > 0 or hi < 0) else ""
        print(f"{name:22s} {np.nanmean(s[:, 0, i]):7.4f} {np.nanmean(s[:, 1, i]):7.4f} "
              f"{d.mean():+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {np.mean(d > 0):.3f}{flag}")
    print("\n* interval excludes 0. A and B columns are bootstrap means; "
          "the point estimates are in each run's results.json.")


if __name__ == "__main__":
    main()
