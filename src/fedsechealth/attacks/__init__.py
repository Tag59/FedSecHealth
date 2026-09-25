"""Attacks against federated learning."""

from .gradient_inversion import (
    Reconstruction,
    analytic_linear_attack,
    dlg_attack,
    idlg_attack,
    infer_label_from_bias_grad,
    inverting_gradients_attack,
    total_variation,
)
from .metrics import image_metrics, match_batch, psnr, ssim, tabular_metrics
from .poisoning import ATTACKS, Adversary, AdversaryConfig, alie_z, apply_trigger, backdoor_asr

ATTACK_NAMES = {
    "analytic": "Analytic (linear layer)",
    "idlg": "iDLG",
    "dlg": "DLG",
    "ig": "Inverting Gradients",
}

__all__ = [
    "ATTACKS",
    "ATTACK_NAMES",
    "Adversary",
    "AdversaryConfig",
    "alie_z",
    "apply_trigger",
    "backdoor_asr",
    "Reconstruction",
    "analytic_linear_attack",
    "dlg_attack",
    "idlg_attack",
    "image_metrics",
    "infer_label_from_bias_grad",
    "inverting_gradients_attack",
    "match_batch",
    "psnr",
    "ssim",
    "tabular_metrics",
    "total_variation",
]
