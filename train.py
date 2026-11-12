"""Unified CheXpert training.

  python train.py --config config.yaml
  python train.py --config config.yaml backbone=densenet121 lr=1e-4 uncertainty_policy=zeros seed=1
  python train.py --config config.yaml max_steps=100 eval_every=50 log_every=10 run_name=smoke   # 5-minute smoke test

What differs from the earlier scripts:
  * warmup and cosine decay are applied per step, not per epoch
  * checkpoints are chosen on held-out TRAINING patients (select_split.csv)
  * the official validation set is scored once, at the end, with bootstrap CIs
  * predictions are saved so two runs can be compared with compare.py
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml

from common import lr_at
from data_loader import get_dataloaders
from evaluate import aggregate_by_study, bootstrap_ci, format_report, predict, summarize
from model import MultiLabelNet

_FLOAT_KEYS = ("lr", "weight_decay", "grad_clip", "dropout", "min_lr_ratio")
_INT_KEYS = ("batch_size", "img_size", "num_epochs", "warmup_steps", "eval_every",
             "log_every", "num_workers", "seed", "patience_evals", "n_boot", "max_steps")


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value pairs that override the config")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        cfg[key] = yaml.safe_load(value)
    for k in _FLOAT_KEYS:
        if cfg.get(k) is not None:
            cfg[k] = float(cfg[k])  # yaml reads '1e-4' as a string
    for k in _INT_KEYS:
        if cfg.get(k) is not None:
            cfg[k] = int(cfg[k])
    cfg["uncertainty_policy"] = str(cfg["uncertainty_policy"])
    if not cfg.get("run_name"):
        short = cfg["backbone"].split(".")[0]
        cfg["run_name"] = f"{short}__{cfg['uncertainty_policy']}__s{cfg['seed']}"
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.get("runs_root", "./runs"), cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    log_path = os.path.join(run_dir, "log.jsonl")
    ckpt_path = os.path.join(run_dir, "best_model.pth")

    train_loader, select_loader, val_loader = get_dataloaders(cfg)
    total_steps = len(train_loader) * cfg["num_epochs"]
    if cfg.get("max_steps"):  # smoke test: stop early but still run selection and final scoring
        total_steps = min(total_steps, cfg["max_steps"])
    print(f"run: {cfg['run_name']}  device: {device}")
    print(f"backbone={cfg['backbone']} policy={cfg['uncertainty_policy']} lr={cfg['lr']} "
          f"wd={cfg['weight_decay']} seed={cfg['seed']}")
    print(f"train batches/epoch: {len(train_loader)}  total steps: {total_steps}  "
          f"select images: {len(select_loader.dataset)}  official val images: {len(val_loader.dataset)}",
          flush=True)

    model = MultiLabelNet(cfg["backbone"], dropout=cfg.get("dropout", 0.0)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = torch.nn.BCEWithLogitsLoss()
    # U-Ignore: uncertain entries contribute nothing to the loss; the loss is averaged over the
    # remaining (certain) entries. Every other policy uses the unchanged criterion above.
    ignore_uncertain = cfg["uncertainty_policy"].lower() == "ignore"
    elementwise = torch.nn.BCEWithLogitsLoss(reduction="none")
    use_tta = bool(cfg.get("use_tta", False))
    patience = cfg.get("patience_evals") or 0

    best, stale, step, running, t0 = -1.0, 0, 0, 0.0, time.time()
    stop = False
    model.train()
    for epoch in range(cfg["num_epochs"]):
        for images, labels, uncertain in train_loader:
            lr_now = lr_at(step, total_steps, cfg["warmup_steps"], cfg["lr"], cfg.get("min_lr_ratio", 0.01))
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                if ignore_uncertain:
                    keep = (~uncertain.to(device, non_blocking=True)).float()
                    loss = (elementwise(logits.float(), labels) * keep).sum() / keep.sum().clamp_min(1.0)
                else:
                    loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            if cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            step += 1

            if step % cfg["log_every"] == 0:
                print(f"step {step}/{total_steps} epoch {epoch + 1} "
                      f"loss={running / cfg['log_every']:.4f} lr={lr_now:.2e} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
                running = 0.0

            if step % cfg["eval_every"] == 0 or step == total_steps:
                y_true, y_prob = predict(model, select_loader, device, use_tta, use_amp)
                stats = summarize(y_true, y_prob)
                model.train()
                score = stats["mean_competition"]
                improved = score > best
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "epoch": epoch + 1, "lr": lr_now,
                                        "select_mean_competition": score,
                                        "select_per_class": stats["per_class"],
                                        "elapsed_s": time.time() - t0}) + "\n")
                print(f"[select] step {step}: mean competition AUC {score:.4f}"
                      f"{'  (best, saved)' if improved else ''}", flush=True)
                if improved:
                    best, stale = score, 0
                    torch.save({"model": model.state_dict(), "step": step, "select_auc": best}, ckpt_path)
                else:
                    stale += 1
                    if patience and stale >= patience:
                        print(f"early stop: no improvement in {patience} evaluations")
                        stop = True
                        break
            if step >= total_steps:
                stop = True
                break
        if stop:
            break

    # ---- final scoring: official validation set, once, with the selected checkpoint ----
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    y_true, y_prob = predict(model, val_loader, device, use_tta, use_amp)
    image_stats = summarize(y_true, y_prob)
    study_ids = val_loader.dataset.study_ids
    yt_s, yp_s, uniq = aggregate_by_study(y_true, y_prob, study_ids)
    study_stats = summarize(yt_s, yp_s)
    ci = bootstrap_ci(yt_s, yp_s, n_boot=cfg.get("n_boot", 2000), seed=0)

    print(f"\nSelected checkpoint: step {state['step']}, held-out AUC {state['select_auc']:.4f}")
    print(f"Official validation, per study ({len(uniq)} studies), 95% bootstrap CI:")
    print(format_report(study_stats, ci))
    print(f"Official validation, per image ({len(y_true)} images):")
    print(format_report(image_stats))

    np.savez(os.path.join(run_dir, "official_val_predictions.npz"),
             study_ids=uniq, y_true_study=yt_s, y_prob_study=yp_s,
             image_study_ids=study_ids, y_true_image=y_true, y_prob_image=y_prob)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"run_name": cfg["run_name"], "selected_step": state["step"],
                   "select_auc": state["select_auc"], "official_val_study": study_stats,
                   "official_val_study_ci95": ci, "official_val_image": image_stats}, f, indent=2)
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    train()
