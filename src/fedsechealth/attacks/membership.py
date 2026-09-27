"""Membership inference: was this patient part of the training data?

Threat model: the adversary queries a trained model (the final global model,
which every hospital receives, or one hospital's local model, which the server
sees in plain FedAvg) on a target record whose label it knows, and decides
whether the record was used for training. For a medical model, membership
alone can be sensitive (e.g. "this patient was treated in the oncology unit
of hospital A").

Implemented scores (higher = "member")
--------------------------------------
* ``loss`` - negative cross-entropy loss (Yeom et al., CSF 2018).
* ``confidence`` - probability of the true class.
* ``entropy`` - negative modified entropy (Song & Mittal, USENIX Sec 2021).
* ``lira`` - offline Likelihood Ratio Attack (Carlini et al., IEEE S&P 2022):
  the target's logit-scaled confidence is compared with its distribution
  under shadow models that did *not* train on it. Per-example calibration
  catches records that are easy (or hard) for every model, which global
  thresholds miss.

Evaluation follows Carlini et al.: ROC AUC, and above all the true-positive
rate at a low false-positive rate (1 %), i.e. how many patients the attacker
identifies *confidently*.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

MIA_ATTACKS = ["loss", "confidence", "entropy", "lira"]


@torch.no_grad()
def model_logits(
    model: nn.Module, x: np.ndarray, device: torch.device, bs: int = 1024
) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i : i + bs]).to(device)).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).double().numpy()


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def loss_scores(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _log_softmax(logits)[np.arange(len(y)), y]


def confidence_scores(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(loss_scores(logits, y))


def entropy_scores(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Negative modified entropy: -[(1 - p_y) log p_y + sum_{i != y} p_i log(1 - p_i)]."""
    p = np.exp(_log_softmax(logits))
    eps = 1e-12
    idx = np.arange(len(y))
    p_y = p[idx, y]
    term_y = (1 - p_y) * np.log(p_y + eps)
    others = p * np.log(1 - p + eps)
    others[idx, y] = 0.0
    return term_y + others.sum(1)


def scaled_logit(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """phi = log(p_y / (1 - p_y)), computed stably from logits (LiRA's statistic)."""
    idx = np.arange(len(y))
    z_y = logits[idx, y]
    others = logits.copy()
    others[idx, y] = -np.inf
    m = others.max(1)
    lse_others = m + np.log(np.exp(others - m[:, None]).sum(1))
    return z_y - lse_others


def lira_offline_scores(
    target_logits: np.ndarray, y: np.ndarray, shadow_logits: list[np.ndarray]
) -> np.ndarray:
    """Offline LiRA: z-score of the target statistic against its OUT (shadow) distribution."""
    phi = scaled_logit(target_logits, y)
    out = np.stack([scaled_logit(s, y) for s in shadow_logits])
    mu, sd = out.mean(0), out.std(0) + 1e-6
    return (phi - mu) / sd


def mia_scores(
    attack: str,
    logits: np.ndarray,
    y: np.ndarray,
    shadow_logits: list[np.ndarray] | None = None,
) -> np.ndarray:
    if attack == "loss":
        return loss_scores(logits, y)
    if attack == "confidence":
        return confidence_scores(logits, y)
    if attack == "entropy":
        return entropy_scores(logits, y)
    if attack == "lira":
        if not shadow_logits:
            raise ValueError("LiRA needs shadow model outputs")
        return lira_offline_scores(logits, y, shadow_logits)
    raise ValueError(f"Unknown membership attack {attack!r}; choose from {MIA_ATTACKS}")


def mia_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict:
    """AUC, TPR at 1 % / 5 % FPR, and the best balanced attack accuracy."""
    labels = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    scores = np.r_[member_scores, nonmember_scores]
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "tpr_at_1fpr": float(np.interp(0.01, fpr, tpr)),
        "tpr_at_5fpr": float(np.interp(0.05, fpr, tpr)),
        "attack_accuracy": float(np.max(1 - (fpr + (1 - tpr)) / 2)),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
