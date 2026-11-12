"""Score finished runs on the sealed CheXpert test set. RUN THIS ONCE, after every
configuration is frozen. No training, no model selection: each run's already-selected
checkpoint is loaded and scored.

  python score_test.py                       # every finished run in ./runs
  python score_test.py --summary_only        # just rebuild the tables from saved results

Per run it writes test_predictions.npz and test_results.json into the run folder,
then prints a per-run table and a per-configuration table (mean and SD over seeds).
Runs that already have test_results.json are not scored again.
"""
import argparse
import json
import os
import re

import numpy as np
import yaml

from common import COMPETITION_LABELS

SKIP_PREFIXES = ("smoke",)


def find_runs(runs_root):
    """Finished runs = folders holding a checkpoint and results.json. Sorted by name."""
    out = []
    for name in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, name)
        if not os.path.isdir(d) or name.startswith(SKIP_PREFIXES):
            continue
        if all(os.path.exists(os.path.join(d, f)) for f in ("best_model.pth", "results.json", "config_used.yaml")):
            out.append(d)
    return out


def split_name(run_name):
    """'densenet121__Atelectasis+Edema__s43' -> ('densenet121', 'Atelectasis+Edema', 43)"""
    m = re.match(r"^(.*)__(.*)__s(\d+)$", run_name)
    if not m:
        return run_name, "?", -1
    return m.group(1), m.group(2), int(m.group(3))


def score_run(run_dir, test_csv, n_boot, device):
    import torch
    from torch.utils.data import DataLoader

    from data_loader import CheXpertDataset, get_transforms
    from evaluate import aggregate_by_study, bootstrap_ci, predict, summarize
    from model import MultiLabelNet

    with open(os.path.join(run_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)
    ds = CheXpertDataset(test_csv, cfg["img_dir"], get_transforms(int(cfg["img_size"]), False),
                         keep_uncertain=True, views="all")
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]) * 2, shuffle=False,
                        num_workers=int(cfg.get("num_workers", 8)), pin_memory=True)
    model = MultiLabelNet(cfg["backbone"], dropout=float(cfg.get("dropout", 0.0))).to(device)
    state = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state["model"])  # strict: fails loudly if the checkpoint does not match

    y_true, y_prob = predict(model, loader, device, bool(cfg.get("use_tta", False)), bool(cfg.get("use_amp", True)))
    yt_s, yp_s, uniq = aggregate_by_study(y_true, y_prob, ds.study_ids)
    study, image = summarize(yt_s, yp_s), summarize(y_true, y_prob)
    ci = bootstrap_ci(yt_s, yp_s, n_boot=n_boot, seed=0)

    np.savez(os.path.join(run_dir, "test_predictions.npz"),
             study_ids=uniq, y_true_study=yt_s, y_prob_study=yp_s,
             image_study_ids=ds.study_ids, y_true_image=y_true, y_prob_image=y_prob)
    result = {"run_name": os.path.basename(run_dir), "selected_step": int(state["step"]),
              "n_images": int(len(y_true)), "n_studies": int(len(uniq)),
              "test_study": study, "test_study_ci95": ci, "test_image": image}
    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    del model
    return result


def load_results(run_dirs):
    out = []
    for d in run_dirs:
        p = os.path.join(d, "test_results.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def build_tables(results):
    """Returns (per_run_lines, per_config_lines) as lists of strings."""
    short = ["Atel", "Card", "Cons", "Edema", "PlEff"]
    head = f"{'run':58s} {'mean':>6s}  {'95% CI':15s} " + " ".join(f"{s:>6s}" for s in short)
    per_run, groups = [head], {}
    for r in sorted(results, key=lambda r: r["run_name"]):
        st, ci = r["test_study"], r["test_study_ci95"]["mean_competition"]
        vals = [st["per_class"][n] for n in COMPETITION_LABELS]
        per_run.append(f"{r['run_name']:58s} {st['mean_competition']:.4f}  [{ci[0]:.3f}, {ci[1]:.3f}] "
                       + " ".join("   n/a" if v is None else f"{v:.4f}" for v in vals))
        backbone, policy, _ = split_name(r["run_name"])
        groups.setdefault((backbone, policy), []).append([st["mean_competition"]] + vals)
    head2 = f"{'backbone':34s} {'policy':20s} {'n':>2s} {'mean':>6s} {'sd':>6s} " + " ".join(f"{s:>6s}" for s in short)
    per_cfg = [head2]
    for (backbone, policy), rows in sorted(groups.items()):
        a = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
        m = np.nanmean(a, axis=0)
        sd = a[:, 0].std(ddof=1) if len(a) > 1 else float("nan")
        per_cfg.append(f"{backbone:34s} {policy:20s} {len(a):2d} {m[0]:.4f} {sd:6.4f} "
                       + " ".join(f"{v:.4f}" for v in m[1:]))
    return per_run, per_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="./runs")
    ap.add_argument("--test_csv", default=None, help="default: test_labels.csv next to the run's img_dir")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()

    run_dirs = find_runs(args.runs_root)
    print(f"{len(run_dirs)} finished runs found in {args.runs_root}")
    if not args.summary_only:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, d in enumerate(run_dirs, 1):
            name = os.path.basename(d)
            if os.path.exists(os.path.join(d, "test_results.json")):
                print(f"[{i}/{len(run_dirs)}] {name}: already scored, skipping", flush=True)
                continue
            with open(os.path.join(d, "config_used.yaml")) as f:
                img_dir = yaml.safe_load(f)["img_dir"]
            test_csv = args.test_csv or os.path.join(img_dir, "test_labels.csv")
            r = score_run(d, test_csv, args.n_boot, device)
            print(f"[{i}/{len(run_dirs)}] {name}: {r['n_studies']} studies, "
                  f"test mean AUC {r['test_study']['mean_competition']:.4f}", flush=True)

    results = load_results(run_dirs)
    per_run, per_cfg = build_tables(results)
    text = ("TEST SET, per study, five competition labels\n\n" + "\n".join(per_run)
            + "\n\nBy configuration (average over seeds; sd = SD of the mean AUC across seeds)\n\n" + "\n".join(per_cfg) + "\n")
    print("\n" + text)
    with open(os.path.join(args.runs_root, "test_summary.txt"), "w") as f:
        f.write(text)
    print(f"Saved to {os.path.join(args.runs_root, 'test_summary.txt')}")


if __name__ == "__main__":
    main()
