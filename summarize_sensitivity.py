"""Table for the backbone sensitivity check. No GPU needed.

  python summarize_sensitivity.py

Reads the seed-42 Atelectasis+Edema baseline of each backbone from ./runs and the nine
variants from ./runs_sens, and prints test mean AUC (five competition pathologies, per
study) for every backbone under every setting, plus the spread between backbones.
Every run is reported; no setting is picked by its test score.
"""
import json
import os

BACKBONES = [("densenet121", "DenseNet-121"), ("deit_base_distilled_patch16_384", "DeiT-Base"),
             ("convnext_base", "ConvNeXt-Base")]
SETTINGS = [("baseline", "base LR, 3 epochs"), ("lr0.5x", "0.5x LR, 3 epochs"),
            ("lr2x", "2x LR, 3 epochs"), ("ep6", "base LR, 6 epochs")]


def run_dir(short, setting):
    if setting == "baseline":
        return os.path.join("runs", f"{short}__Atelectasis+Edema__s42")
    return os.path.join("runs_sens", f"{short}__AE-{setting}__s42")


def load(short, setting):
    d = run_dir(short, setting)
    try:
        with open(os.path.join(d, "test_results.json")) as f:
            test = json.load(f)
        with open(os.path.join(d, "results.json")) as f:
            res = json.load(f)
    except FileNotFoundError:
        return None
    return {"test": test["test_study"]["mean_competition"], "ci": test["test_study_ci95"]["mean_competition"],
            "select": res["select_auc"], "val": res["official_val_study"]["mean_competition"],
            "step": test["selected_step"]}


def main():
    lines = [f"{'setting':20s} " + " ".join(f"{label:>15s}" for _, label in BACKBONES) + f" {'max - min':>10s}"]
    detail, missing = [], []
    for setting, text in SETTINGS:
        vals = []
        for short, label in BACKBONES:
            r = load(short, setting)
            if r is None:
                missing.append(run_dir(short, setting))
                vals.append(None)
                continue
            vals.append(r["test"])
            detail.append(f"{label:14s} {text:20s} test {r['test']:.4f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
                          f"validation {r['val']:.4f}  selection {r['select']:.4f}  selected step {r['step']}")
        have = [v for v in vals if v is not None]
        spread = f"{max(have) - min(have):10.4f}" if len(have) == len(BACKBONES) else f"{'n/a':>10s}"
        lines.append(f"{text:20s} " + " ".join(f"{'missing':>15s}" if v is None else f"{v:15.4f}" for v in vals) + f" {spread}")

    out = ["Test mean AUC, Atelectasis+Edema policy, seed 42 (single runs)", ""] + lines + ["", "Per run:"] + detail
    if missing:
        out += ["", "Not finished or not scored yet:"] + ["  " + m for m in missing]
    text = "\n".join(out)
    print(text)
    os.makedirs("runs_sens", exist_ok=True)
    with open(os.path.join("runs_sens", "sensitivity_summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\nSaved to runs_sens/sensitivity_summary.txt")


if __name__ == "__main__":
    main()
