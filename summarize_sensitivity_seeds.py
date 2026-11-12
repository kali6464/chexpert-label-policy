"""Three-seed table for the sensitivity check. No GPU needed.

  python summarize_sensitivity_seeds.py

For every backbone and setting: test mean AUC of seeds 42, 43 and 44, their mean and SD, and
the range of the backbone means within each setting. Baselines (main-grid settings) come from
./runs, the three variants from ./runs_sens. Every run is reported.
"""
import json
import os

import numpy as np

BACKBONES = [("densenet121", "DenseNet-121"), ("deit_base_distilled_patch16_384", "DeiT-Base"),
             ("convnext_base", "ConvNeXt-Base")]
SETTINGS = [("baseline", "1x LR, 3 epochs"), ("lr0.5x", "0.5x LR, 3 epochs"),
            ("lr2x", "2x LR, 3 epochs"), ("ep6", "1x LR, 6 epochs")]
SEEDS = (42, 43, 44)


def run_dir(short, setting, seed):
    if setting == "baseline":
        return os.path.join("runs", f"{short}__Atelectasis+Edema__s{seed}")
    return os.path.join("runs_sens", f"{short}__AE-{setting}__s{seed}")


def test_auc(d):
    try:
        with open(os.path.join(d, "test_results.json")) as f:
            return json.load(f)["test_study"]["mean_competition"]
    except FileNotFoundError:
        return None


def main():
    out = ["Test mean AUC, Atelectasis+Edema policy, seeds 42/43/44 (single runs)", ""]
    missing, means = [], {}
    for setting, text in SETTINGS:
        out.append(text)
        for short, label in BACKBONES:
            vals = []
            for seed in SEEDS:
                v = test_auc(run_dir(short, setting, seed))
                if v is None:
                    missing.append(run_dir(short, setting, seed))
                vals.append(v)
            have = [v for v in vals if v is not None]
            cells = " ".join("   n/a" if v is None else f"{v:.4f}" for v in vals)
            if len(have) == len(SEEDS):
                m, sd = float(np.mean(have)), float(np.std(have, ddof=1))
                means[(setting, short)] = m
                out.append(f"  {label:14s} {cells}   mean {m:.4f}  SD {sd:.4f}")
            else:
                out.append(f"  {label:14s} {cells}")
        ms = [means.get((setting, s)) for s, _ in BACKBONES]
        if all(m is not None for m in ms):
            out.append(f"  range of backbone means: {max(ms) - min(ms):.4f}")
        out.append("")
    out.append("Change from the main-grid setting (mean over seeds):")
    for short, label in BACKBONES:
        base = means.get(("baseline", short))
        diffs = [f"{t}: {means[(s, short)] - base:+.4f}" for s, t in SETTINGS[1:]
                 if base is not None and (s, short) in means]
        out.append(f"  {label:14s} " + "   ".join(diffs))
    if missing:
        out += ["", "Not finished or not scored yet:"] + ["  " + m for m in missing]
    text = "\n".join(out)
    print(text)
    os.makedirs("runs_sens", exist_ok=True)
    with open(os.path.join("runs_sens", "sensitivity_summary_3seeds.txt"), "w") as f:
        f.write(text + "\n")
    print("\nSaved to runs_sens/sensitivity_summary_3seeds.txt")


if __name__ == "__main__":
    main()
