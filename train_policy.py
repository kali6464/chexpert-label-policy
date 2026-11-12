"""Train with a learned (or fixed soft) per-label policy for uncertain labels.

  python train_policy.py --config config.yaml backbone=densenet121 lr=1e-4 weight_decay=1e-5 seed=42

Same data, schedule, checkpoint selection and reporting as train.py. Differences:
  * uncertain entries get target alpha_c and weight beta_c (see learned_policy.py)
  * alpha, beta are learned from the official validation set, used here as the GUIDE set
  * so the validation score printed at the end is NOT an unbiased estimate for this
    method; the sealed test set is the fair comparison

Extra settings (all optional, pass as key=value):
  policy_lr=1e-3          learning rate of the policy; 0 = keep the policy fixed
  policy_start=1000       first step at which the policy is updated
  guide_every=25          steps between refreshes of the guide-set gradient
  policy_init_alpha=0.5   starting target for uncertain entries
  policy_init_beta=0.5    starting weight for uncertain entries
  e.g. U-Ignore baseline: policy_lr=0 policy_init_beta=0
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from common import ALL_LABELS, COMPETITION_LABELS, lr_at
from data_loader import CheXpertDataset, get_transforms
from evaluate import aggregate_by_study, bootstrap_ci, format_report, predict, summarize
from learned_policy import UncertaintyPolicy, backbone_features, guide_head_gradient
from model import MultiLabelNet
from train import load_config, set_seed


def policy_dict(policy):
    a = policy.alpha().detach().cpu().tolist()
    b = policy.beta().detach().cpu().tolist()
    return {name: [round(a[i], 4), round(b[i], 4)] for i, name in enumerate(ALL_LABELS)}


def policy_line(pd):
    return "  ".join(f"{n[:5]} a={pd[n][0]:.2f} b={pd[n][1]:.2f}" for n in COMPETITION_LABELS)


def load_guide(dataset, device, workers):
    xs, ys = [], []
    for x, y, _ in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def train():
    cfg = load_config()
    policy_lr = float(cfg.get("policy_lr", 1e-3))
    policy_start = int(cfg.get("policy_start", 1000))
    guide_every = int(cfg.get("guide_every", 25))
    init_alpha = float(cfg.get("policy_init_alpha", 0.5))
    init_beta = float(cfg.get("policy_init_beta", 0.5))
    learn = policy_lr > 0
    if not any(arg.startswith("run_name=") for arg in sys.argv[1:]):
        tag = "learned" if learn else f"fixed-a{init_alpha:g}-b{init_beta:g}"
        cfg["run_name"] = f"{cfg['backbone'].split('.')[0]}__{tag}__s{cfg['seed']}"

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.get("runs_root", "./runs"), cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    log_path = os.path.join(run_dir, "log.jsonl")
    ckpt_path = os.path.join(run_dir, "best_model.pth")

    size, workers, bs = cfg["img_size"], cfg["num_workers"], cfg["batch_size"]
    views, hflip = cfg.get("views", "all"), bool(cfg.get("hflip", False))
    train_ds = CheXpertDataset(cfg["train_csv"], cfg["img_dir"], get_transforms(size, True, hflip),
                               keep_uncertain=True, views=views)
    select_ds = CheXpertDataset(cfg["select_csv"], cfg["img_dir"], get_transforms(size, False),
                                keep_uncertain=True, views=views)
    guide_ds = CheXpertDataset(cfg["official_val_csv"], cfg["img_dir"], get_transforms(size, False),
                               keep_uncertain=True, views=views)
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **common)
    select_loader = DataLoader(select_ds, batch_size=bs * 2, shuffle=False, **common)
    val_loader = DataLoader(guide_ds, batch_size=bs * 2, shuffle=False, **common)

    total_steps = len(train_loader) * cfg["num_epochs"]
    if cfg.get("max_steps"):
        total_steps = min(total_steps, cfg["max_steps"])
    print(f"run: {cfg['run_name']}  device: {device}")
    print(f"backbone={cfg['backbone']} lr={cfg['lr']} wd={cfg['weight_decay']} seed={cfg['seed']}")
    print(f"policy: {'LEARNED' if learn else 'FIXED'}  policy_lr={policy_lr} start={policy_start} "
          f"guide_every={guide_every} init_alpha={init_alpha} init_beta={init_beta}")
    print(f"train batches/epoch: {len(train_loader)}  total steps: {total_steps}  "
          f"select images: {len(select_ds)}  guide images: {len(guide_ds)}", flush=True)

    model = MultiLabelNet(cfg["backbone"], dropout=cfg.get("dropout", 0.0)).to(device)
    policy = UncertaintyPolicy(len(ALL_LABELS), init_alpha, init_beta, learn=learn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    policy_opt = torch.optim.Adam(policy.parameters(), lr=policy_lr) if learn else None
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_tta = bool(cfg.get("use_tta", False))

    guide_x = guide_y = G_W = None
    guide_loss = float("nan")
    if learn:
        guide_x, guide_y = load_guide(guide_ds, device, workers)
        print(f"guide set on device: {tuple(guide_x.shape)}", flush=True)

    best, step, running, t0 = -1.0, 0, 0.0, time.time()
    done = False
    model.train()
    for epoch in range(cfg["num_epochs"]):
        for images, labels, unc in train_loader:
            lr_now = lr_at(step, total_steps, cfg["warmup_steps"], cfg["lr"], cfg.get("min_lr_ratio", 0.01))
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            unc = unc.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                feats = backbone_features(model, images)
                logits = model.head(model.drop(feats))
            logits = logits.float()
            targets, weights = policy.targets_and_weights(labels, unc)
            loss = (weights * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            if cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            # ---- policy update from the guide set (closed-form head-only lookahead) ----
            if learn and step >= policy_start:
                if G_W is None or step % guide_every == 0:
                    G_W, guide_loss = guide_head_gradient(model, guide_x, guide_y, use_amp)
                S = feats.detach().float() @ G_W.t()
                meta = policy.meta_objective(torch.sigmoid(logits.detach()), unc, S)
                policy_opt.zero_grad(set_to_none=True)
                meta.backward()
                policy_opt.step()
                policy.clamp_()

            running += loss.item()
            step += 1

            if step % cfg["log_every"] == 0:
                print(f"step {step}/{total_steps} epoch {epoch + 1} loss={running / cfg['log_every']:.4f} "
                      f"lr={lr_now:.2e} guide_loss={guide_loss:.4f} elapsed={time.time() - t0:.0f}s", flush=True)
                running = 0.0

            if step % cfg["eval_every"] == 0 or step == total_steps:
                y_true, y_prob = predict(model, select_loader, device, use_tta, use_amp)
                stats = summarize(y_true, y_prob)
                model.train()
                score = stats["mean_competition"]
                improved = score > best
                pd = policy_dict(policy)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "epoch": epoch + 1, "lr": lr_now,
                                        "select_mean_competition": score, "guide_loss": guide_loss,
                                        "policy_alpha_beta": pd,
                                        "select_per_class": stats["per_class"],
                                        "elapsed_s": time.time() - t0}) + "\n")
                print(f"[select] step {step}: mean competition AUC {score:.4f}"
                      f"{'  (best, saved)' if improved else ''}")
                print(f"[policy] {policy_line(pd)}", flush=True)
                if improved:
                    best = score
                    torch.save({"model": model.state_dict(), "step": step, "select_auc": best,
                                "alpha": policy.alpha().detach().cpu(),
                                "beta": policy.beta().detach().cpu()}, ckpt_path)
            if step >= total_steps:
                done = True
                break
        if done:
            break

    final_policy = policy_dict(policy)
    print("\nFinal policy (alpha = target, beta = weight for uncertain entries):")
    for name in ALL_LABELS:
        print(f"  {name:28s} alpha={final_policy[name][0]:.3f}  beta={final_policy[name][1]:.3f}")

    # ---- scoring with the selected checkpoint ----
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    y_true, y_prob = predict(model, val_loader, device, use_tta, use_amp)
    image_stats = summarize(y_true, y_prob)
    study_ids = guide_ds.study_ids
    yt_s, yp_s, uniq = aggregate_by_study(y_true, y_prob, study_ids)
    study_stats = summarize(yt_s, yp_s)
    ci = bootstrap_ci(yt_s, yp_s, n_boot=cfg.get("n_boot", 2000), seed=0)

    note = "GUIDE SET for this method: optimistic, not a fair comparison" if learn else "not used in training"
    print(f"\nSelected checkpoint: step {state['step']}, held-out AUC {state['select_auc']:.4f}")
    print(f"Official validation ({note}), per study ({len(uniq)} studies), 95% bootstrap CI:")
    print(format_report(study_stats, ci))
    print(f"Official validation, per image ({len(y_true)} images):")
    print(format_report(image_stats))

    np.savez(os.path.join(run_dir, "official_val_predictions.npz"),
             study_ids=uniq, y_true_study=yt_s, y_prob_study=yp_s,
             image_study_ids=study_ids, y_true_image=y_true, y_prob_image=y_prob)
    selected_policy = {name: [round(float(state["alpha"][i]), 4), round(float(state["beta"][i]), 4)]
                       for i, name in enumerate(ALL_LABELS)}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"run_name": cfg["run_name"], "policy_learned": learn,
                   "selected_step": state["step"], "select_auc": state["select_auc"],
                   "policy_at_selected_step": selected_policy, "policy_final": final_policy,
                   "official_val_study": study_stats, "official_val_study_ci95": ci,
                   "official_val_image": image_stats,
                   "validation_is_guide_set": learn}, f, indent=2)
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    train()
