"""Learned per-label policy for uncertain labels.

Every uncertain (-1) entry of label c is trained toward a soft target
alpha_c in (0, 1) with loss weight beta_c in (0, 1). The classic fixed
policies are special cases:

    U-Zeros : alpha = 0, beta = 1
    U-Ones  : alpha = 1, beta = 1
    U-Ignore: beta = 0

alpha and beta are learned during training from a small radiologist-labelled
guide set. The signal is a one-step lookahead on the classifier head only:
"if the head took one gradient step on this batch under the current policy,
would the guide set be ranked better or worse?" For a linear head that has a
closed form (checked against finite differences):

    S_ic         = < dL_guide/dW_c , f_i >
    objective    = -(1/B) * sum over uncertain (i, c) of  beta_c * (p_ic - alpha_c) * S_ic

where f_i are the batch features, p_ic the current predictions, and L_guide a
pairwise ranking loss (a smooth stand-in for AUC) on the guide set.
Minimising the objective raises alpha_c when uncertain cases of label c
resemble the guide set's positives, and lowers it when they resemble negatives.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class UncertaintyPolicy(nn.Module):
    def __init__(self, num_labels=14, init_alpha=0.5, init_beta=0.5, learn=True):
        super().__init__()
        self.learn = bool(learn)
        if self.learn:
            self.a = nn.Parameter(torch.full((num_labels,), _logit(init_alpha)))
            self.b = nn.Parameter(torch.full((num_labels,), _logit(init_beta)))
        else:  # fixed policy: exact values, 0 and 1 allowed
            self.register_buffer("fixed_alpha", torch.full((num_labels,), float(init_alpha)))
            self.register_buffer("fixed_beta", torch.full((num_labels,), float(init_beta)))

    def alpha(self):
        return torch.sigmoid(self.a) if self.learn else self.fixed_alpha

    def beta(self):
        return torch.sigmoid(self.b) if self.learn else self.fixed_beta

    def targets_and_weights(self, labels, uncertain):
        """labels: [B, C] float with -1 at uncertain entries; uncertain: [B, C] bool.
        Returns the training targets and per-entry loss weights (no gradient to the policy)."""
        alpha, beta = self.alpha().detach(), self.beta().detach()
        targets = torch.where(uncertain, alpha.expand_as(labels), labels)
        weights = torch.where(uncertain, beta.expand_as(labels), torch.ones_like(labels))
        return targets, weights

    def meta_objective(self, probs, uncertain, S):
        """probs, S: [B, C] float (no grad); uncertain: [B, C] bool."""
        m = uncertain.to(probs.dtype)
        per_entry = self.beta().unsqueeze(0) * (probs - self.alpha().unsqueeze(0)) * S
        return -(per_entry * m).sum() / probs.shape[0]

    @torch.no_grad()
    def clamp_(self, limit=6.0):
        if self.learn:
            self.a.clamp_(-limit, limit)
            self.b.clamp_(-limit, limit)


def backbone_features(model, x):
    """Pooled features of a MultiLabelNet, exactly as its forward() computes them."""
    f = model.backbone(x)
    if isinstance(f, tuple):
        f = (f[0] + f[1]) / 2
    return f


def pairwise_auc_loss(z, y):
    """Smooth AUC surrogate. z, y: [N, C]. For each label with both classes:
    mean over (positive, negative) pairs of softplus(-(z_pos - z_neg)); summed over labels."""
    total = z.new_zeros(())
    for c in range(z.shape[1]):
        pos = z[y[:, c] == 1, c]
        neg = z[y[:, c] == 0, c]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        total = total + F.softplus(-(pos.unsqueeze(1) - neg.unsqueeze(0))).mean()
    return total


def guide_head_gradient(model, guide_x, guide_y, use_amp, chunk=64):
    """dL_guide/dW for the classifier head at its current weights.
    Returns (G_W [C, d], guide loss). The bias has zero gradient for a ranking loss."""
    was_training = model.training
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, guide_x.shape[0], chunk):
            with torch.autocast(device_type=guide_x.device.type, enabled=use_amp):
                feats.append(backbone_features(model, guide_x[i:i + chunk]).float())
    if was_training:
        model.train()
    feats = torch.cat(feats)
    W = model.head.weight.detach().float().clone().requires_grad_(True)
    loss = pairwise_auc_loss(feats @ W.t(), guide_y)
    (G_W,) = torch.autograd.grad(loss, W)
    return G_W.detach(), float(loss)
