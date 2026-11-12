"""Paired bootstrap comparison on the TEST set (run score_test.py first).

  python compare_test.py runs/A runs/B
  python compare_test.py runs/A_s42,runs/A_s43,runs/A_s44 runs/B_s42,runs/B_s43,runs/B_s44

Give several comma-separated runs on a side to compare seed ensembles: their
predicted probabilities are averaged before scoring.
"""
import argparse
import os

import numpy as np

from common import COMPETITION_LABELS
from evaluate import bootstrap_samples


def load_side(spec, fname):
    ids0 = yt0 = None
    probs = []
    for run_dir in [s for s in spec.split(",") if s]:
        d = np.load(os.path.join(run_dir, fname))
        order = np.argsort(d["study_ids"])
        ids, yt, yp = d["study_ids"][order], d["y_true_study"][order], d["y_prob_study"][order]
        if ids0 is None:
            ids0, yt0 = ids, yt
        elif not (np.array_equal(ids, ids0) and np.array_equal(yt, yt0)):
            raise SystemExit(f"{run_dir} was not scored on the same studies as the others")
        probs.append(yp)
    return ids0, yt0, np.mean(probs, axis=0), len(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("side_a")
    ap.add_argument("side_b")
    ap.add_argument("--file", default="test_predictions.npz")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ids_a, yt_a, yp_a, na = load_side(args.side_a, args.file)
    ids_b, yt_b, yp_b, nb = load_side(args.side_b, args.file)
    if not (np.array_equal(ids_a, ids_b) and np.array_equal(yt_a, yt_b)):
        raise SystemExit("The two sides were not scored on the same studies.")

    s = bootstrap_samples(yt_a, [yp_a, yp_b], args.n_boot, args.seed)
    diff = s[:, 0, :] - s[:, 1, :]
    print(f"A = {args.side_a}  ({na} run{'s' if na > 1 else ''})\nB = {args.side_b}  ({nb} run{'s' if nb > 1 else ''})")
    print(f"{len(ids_a)} studies, {args.n_boot} resamples, file {args.file}\n")
    print(f"{'':22s} {'A':>7s} {'B':>7s} {'A-B':>8s}   95% CI of A-B        P(A>B)")
    for i, name in enumerate(COMPETITION_LABELS + ["mean_competition"]):
        d = diff[:, i][~np.isnan(diff[:, i])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        flag = "  *" if (lo > 0 or hi < 0) else ""
        print(f"{name:22s} {np.nanmean(s[:, 0, i]):7.4f} {np.nanmean(s[:, 1, i]):7.4f} "
              f"{d.mean():+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {np.mean(d > 0):.3f}{flag}")
    print("\n* interval excludes 0. A and B columns are bootstrap means.")


if __name__ == "__main__":
    main()
