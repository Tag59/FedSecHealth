# FedSecHealth

**A reproducible testbed for privacy & security attacks and defenses in federated learning on medical data.**

[![CI](https://github.com/OWNER/FedSecHealth/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/FedSecHealth/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

🇬🇧 English · [🇫🇷 Français](#-version-française)

---

Hospitals want to train models together without sharing patient records.
Federated Learning (FL) promises exactly that: only model updates leave the
hospital. **But gradients leak data.** FedSecHealth simulates hospitals
collaborating on a diagnostic model, lets an honest-but-curious server
**reconstruct patient records from their gradients**, and measures how
well defenses such as **Differential Privacy (DP-SGD)** stop the attack and
what they cost in accuracy.

## Key results (v0.1, Breast Cancer Wisconsin)

**1. Federation helps when hospitals differ.** Five hospitals with strongly
label-skewed data (Dirichlet α = 0.3). Mean ± std over 5 seeds:

<p align="center"><img src="docs/figures/train_noniid.png" width="620"></p>

| Setting | Test accuracy |
|---|---|
| Each hospital alone (mean) | **0.718** |
| Federated (FedAvg, 30 rounds) | **0.975** |
| Centralized (pooled data, not private) | **0.977** |

**2. Gradients leak patient records, exactly.** One gradient computed on a
single patient is enough for the server to recover all 30 clinical features and
the diagnosis:

RECONSTRUCTION_TABLE

**3. Differential privacy breaks the attack at a small accuracy cost.**

<p align="center"><img src="docs/figures/tradeoff_iid.png" width="620"></p>

TRADEOFF_TABLE

## What's inside

| Component | Details |
|---|---|
| Data | Breast Cancer Wisconsin (569 patients, 30 features); IID or **Dirichlet non-IID** splits; **federated feature standardisation** (hospitals only share sums, never records) |
| FL engine | Deterministic FedAvg simulator in pure PyTorch; centralized and local-only baselines |
| Attacks | **Analytic** linear-layer inversion (Phong et al. 2017), **DLG** (Zhu et al. 2019), **iDLG** (Zhao et al. 2020) |
| Defenses | **DP-SGD** per hospital via Opacus, with RDP (ε, δ) accounting |
| Engineering | `src/` package, typed YAML configs, CLI, tests, CI, ruff |

## Threat model

- **Adversary:** an *honest-but-curious* aggregation server (or anyone who
  intercepts updates). It follows the protocol, knows the architecture and the
  current weights, and observes each hospital's update.
- **Goal:** recover the private inputs (patient features) and labels (diagnosis).
- **Setting:** the update is the gradient of one local step (FedSGD). This is the
  worst case for the hospital and the standard benchmark setting for gradient inversion.
- **Success criterion:** relative L2 error ‖x̂ − x‖ / ‖x‖ < 10 % in standardised feature space.
- **DP noise in attacks** is calibrated with the *same* schedule as training
  (sampling rate, number of steps, δ = 1e-5), so each ε in the attack study is the
  budget a hospital would actually spend.

## Quick start

```bash
git clone https://github.com/OWNER/FedSecHealth && cd FedSecHealth
uv sync                                               # installs CPU PyTorch + deps
uv run fedsechealth train    -c configs/breast_cancer_noniid.yaml
uv run fedsechealth demo     -c configs/breast_cancer_iid.yaml --epsilon 5
uv run fedsechealth tradeoff -c configs/breast_cancer_iid.yaml
uv run pytest
```

Override any config value from the command line: `-s fl.rounds=10 -s fl.dp.enabled=true`.
Outputs (JSON + figures) go to `results/<experiment name>/`.

## Project layout

```
src/fedsechealth/
  data.py          datasets, IID / Dirichlet partitions, federated standardisation
  models.py        Opacus-compatible models
  fl.py            hospitals, FedAvg, baselines, DP-SGD training
  privacy.py       DP config, noise calibration, DP gradient release
  attacks/         gradient inversion (analytic, DLG, iDLG)
  experiments.py   train / attack / trade-off / demo pipelines
  cli.py           command-line interface
configs/           YAML experiment definitions
tests/             pytest suite
```

## Roadmap

Medical imaging (MedMNIST) + Inverting Gradients → malicious hospitals
(poisoning, backdoors) + robust aggregation → membership inference and secure
aggregation → Flower/Docker deployment and dashboard. See [ROADMAP.md](ROADMAP.md).

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017.
- Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016.
- Phong et al., *Privacy-Preserving Deep Learning via Additively Homomorphic Encryption*, IEEE TIFS 2017.
- Zhu, Liu, Han, *Deep Leakage from Gradients*, NeurIPS 2019.
- Zhao, Mopuri, Bilen, *iDLG: Improved Deep Leakage from Gradients*, 2020.
- Geiping et al., *Inverting Gradients: How easy is it to break privacy in federated learning?*, NeurIPS 2020.
- Hsu, Qi, Brown, *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*, 2019.

---

## 🇫🇷 Version française

**Un banc d'essai reproductible des attaques et défenses sur la vie privée et la sécurité en apprentissage fédéré appliqué aux données médicales.**

Des hôpitaux veulent entraîner un modèle commun sans partager les dossiers de
leurs patients. Le Federated Learning (FL) le permet : seules les mises à jour
du modèle quittent l'hôpital. **Mais les gradients laissent fuiter les
données.** FedSecHealth simule des hôpitaux qui collaborent sur un modèle de
diagnostic, laisse un serveur « honnête mais curieux » **reconstruire les
dossiers patients à partir de leurs gradients**, puis mesure l'efficacité des
défenses comme la **confidentialité différentielle (DP-SGD)** et leur coût en
précision.

### Résultats principaux (v0.1)

1. **La fédération est utile quand les hôpitaux sont hétérogènes.** Avec 5
   hôpitaux aux données très déséquilibrées (Dirichlet α = 0,3), un hôpital seul
   atteint **71,8 %** de précision, le modèle fédéré **97,5 %**, soit
   quasiment le niveau d'un entraînement centralisé (97,7 %) sans jamais
   mettre les données en commun.
2. **Les gradients révèlent les dossiers patients, exactement.** Un seul
   gradient suffit au serveur pour retrouver les 30 caractéristiques cliniques
   et le diagnostic (voir le tableau plus haut).
3. **La confidentialité différentielle met l'attaque en échec pour un faible coût en précision** (voir la figure et le tableau plus haut).

### Modèle de menace

Le serveur d'agrégation suit le protocole mais cherche à apprendre des
informations sur les patients. Il connaît l'architecture et les poids, et
observe la mise à jour de chaque hôpital (gradient d'une étape locale, le
pire cas). Une reconstruction est réussie si l'erreur relative est
inférieure à 10 %. Le bruit DP utilisé dans les attaques est calibré sur le
même calendrier que l'entraînement, donc chaque ε correspond au budget
réellement dépensé par un hôpital.

### Démarrage rapide

```bash
uv sync
uv run fedsechealth train    -c configs/breast_cancer_noniid.yaml
uv run fedsechealth demo     -c configs/breast_cancer_iid.yaml --epsilon 5
uv run fedsechealth tradeoff -c configs/breast_cancer_iid.yaml
```

### Feuille de route

Imagerie médicale (MedMNIST) et Inverting Gradients, puis hôpitaux
malveillants (empoisonnement, backdoors) et agrégation robuste, puis inférence
d'appartenance et agrégation sécurisée, puis déploiement Flower/Docker et
tableau de bord. Voir [ROADMAP.md](ROADMAP.md).

## License

MIT, see [LICENSE](LICENSE).
