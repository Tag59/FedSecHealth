"""Attacks against federated learning."""

from .gradient_inversion import (
    AttackResult,
    analytic_linear_attack,
    dlg_attack,
    idlg_attack,
    infer_label_from_bias_grad,
    reconstruction_metrics,
)

__all__ = [
    "AttackResult",
    "analytic_linear_attack",
    "dlg_attack",
    "idlg_attack",
    "infer_label_from_bias_grad",
    "reconstruction_metrics",
]
