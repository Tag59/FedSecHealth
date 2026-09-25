"""Malicious hospitals: data and model poisoning attacks on federated training.

Threat model: ``n_malicious`` of the participating hospitals are controlled by
an adversary. They follow the protocol's message format but may train on
poisoned data and/or send manipulated updates. The server does not know which
hospitals are malicious.

Implemented attacks
-------------------
* ``label_flip`` - data poisoning: malicious hospitals train on labels
  mapped y -> C - 1 - y (Biggio et al., 2012; Tolpegin et al., 2020).
* ``sign_flip`` - model poisoning: send -scale x the honest update.
* ``gaussian`` - model poisoning: send random noise N(0, noise_std^2).
* ``alie`` - "A Little Is Enough" (Baruch et al., NeurIPS 2019): colluding
  hospitals send mean - z * std of the benign updates, coordinate-wise. The
  shift stays within the natural spread of honest updates, which lets it slip
  through median / Krum style defenses. We use the omniscient variant (the
  attacker knows the benign updates), i.e. a strong attacker.
* ``backdoor`` - targeted poisoning (Gu et al., 2017; Bagdasaryan et al.,
  2020): a fraction of the malicious hospitals' samples get a small trigger
  and the target label; the update is boosted by ``scale`` (model
  replacement). The model behaves normally on clean inputs but predicts the
  target class whenever the trigger is present.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import norm
from torch import nn

from ..data import ClientData, FederatedDataset

ATTACKS = ["none", "label_flip", "sign_flip", "gaussian", "alie", "backdoor"]


@dataclass
class AdversaryConfig:
    attack: str = "none"
    n_malicious: int = 0
    scale: float = 1.0  # sign_flip multiplier / backdoor boost factor
    noise_std: float = 1.0  # gaussian attack
    backdoor_target: int = 0
    poison_fraction: float = 0.5  # share of each malicious hospital's samples that get the trigger
    trigger_size: int = 3  # side of the square trigger (images) / number of features (tabular)
    alie_z: float | None = None  # None -> the value from Baruch et al. for (n, f)


def apply_trigger(x: np.ndarray, fds: FederatedDataset, size: int) -> np.ndarray:
    """Stamp the backdoor trigger on standardised inputs (returns a copy).

    Images: a white ``size`` x ``size`` square one pixel away from the bottom-right corner.
    Tabular: the first ``size`` features set to +4 standard deviations (an unusual value).
    """
    x = x.copy()
    if fds.modality == "image":
        white = ((1.0 - fds.feature_mean) / fds.feature_std).reshape(-1, 1, 1)
        x[:, :, -size - 1 : -1, -size - 1 : -1] = white
    else:
        x[:, :size] = 4.0
    return x


def alie_z(n: int, f: int) -> float:
    """z_max from Baruch et al.: the largest shift still hidden among the benign majority."""
    s = n // 2 + 1 - f
    return float(norm.ppf((n - s) / n)) if 0 < s < n else 1.0


class Adversary:
    """Controls the malicious hospitals: poisons their data and/or their updates."""

    def __init__(self, cfg: AdversaryConfig, n_clients: int, n_classes: int, seed: int = 0) -> None:
        if cfg.attack not in ATTACKS:
            raise ValueError(f"Unknown attack {cfg.attack!r}; choose from {ATTACKS}")
        if cfg.n_malicious >= n_clients:
            raise ValueError("At least one hospital must be honest.")
        self.cfg = cfg
        self.n_clients = n_clients
        self.n_classes = n_classes
        self.malicious = list(range(cfg.n_malicious)) if cfg.attack != "none" else []
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(seed)

    @property
    def active(self) -> bool:
        return bool(self.malicious)

    # ------------------------------------------------------------------ data poisoning
    def poison_clients(self, clients: list[ClientData], fds: FederatedDataset) -> list[ClientData]:
        out = []
        for c in clients:
            if c.client_id not in self.malicious:
                out.append(c)
            elif self.cfg.attack == "label_flip":
                out.append(
                    ClientData(c.client_id, c.x, (self.n_classes - 1 - c.y).astype(c.y.dtype))
                )
            elif self.cfg.attack == "backdoor":
                n_poison = int(self.cfg.poison_fraction * len(c))
                idx = self.rng.choice(len(c), size=n_poison, replace=False)
                x, y = c.x.copy(), c.y.copy()
                x[idx] = apply_trigger(c.x[idx], fds, self.cfg.trigger_size)
                y[idx] = self.cfg.backdoor_target
                out.append(ClientData(c.client_id, x, y))
            else:
                out.append(c)
        return out

    # ------------------------------------------------------------------ model poisoning
    def corrupt(self, updates: torch.Tensor) -> torch.Tensor:
        """Replace the malicious rows of the (n_clients, d) update matrix."""
        if not self.active:
            return updates
        a, mal = self.cfg.attack, self.malicious
        updates = updates.clone()
        if a == "sign_flip":
            updates[mal] = -self.cfg.scale * updates[mal]
        elif a == "gaussian":
            noise = torch.randn((len(mal), updates.shape[1]), generator=self.gen)
            updates[mal] = noise.to(updates) * self.cfg.noise_std
        elif a == "alie":
            benign = [i for i in range(len(updates)) if i not in mal]
            mu, sd = updates[benign].mean(0), updates[benign].std(0)
            z = self.cfg.alie_z if self.cfg.alie_z is not None else alie_z(len(updates), len(mal))
            updates[mal] = mu - z * sd
        elif a == "backdoor":
            updates[mal] = self.cfg.scale * updates[mal]
        return updates

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, model: nn.Module, fds: FederatedDataset, device: torch.device) -> dict:
        if self.cfg.attack != "backdoor":
            return {}
        asr = backdoor_asr(model, fds, device, self.cfg.backdoor_target, self.cfg.trigger_size)
        return {"backdoor_asr": asr}


@torch.no_grad()
def backdoor_asr(
    model: nn.Module, fds: FederatedDataset, device: torch.device, target: int, trigger_size: int
) -> float:
    """Share of triggered test inputs (whose true class is not ``target``) predicted as ``target``.

    Also meaningful for a clean model: it gives the baseline rate an attack must beat.
    """
    keep = fds.y_test != target
    x = apply_trigger(fds.x_test[keep], fds, trigger_size)
    model.eval()
    preds = torch.cat(
        [
            model(torch.from_numpy(x[i : i + 1024]).to(device)).argmax(1)
            for i in range(0, len(x), 1024)
        ]
    )
    return (preds == target).float().mean().item()
