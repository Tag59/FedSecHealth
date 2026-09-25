#!/usr/bin/env bash
# Regenerate every result and figure in the README (CPU: several hours in total).
set -euo pipefail
export PYTHONWARNINGS=ignore
run() { uv run fedsechealth "$@"; }

# Tabular (v0.1)
run train    -c configs/breast_cancer_iid.yaml
run train    -c configs/breast_cancer_noniid.yaml
run demo     -c configs/breast_cancer_iid.yaml --epsilon 5
run tradeoff -c configs/breast_cancer_iid.yaml
run tradeoff -c configs/breast_cancer_noniid.yaml

# Imaging: attacks on a fresh model, single image
run attack -c configs/pneumonia_attack.yaml
run attack -c configs/bloodmnist_attack.yaml

# Imaging: harder settings for the attacker (Inverting Gradients only)
run attack -c configs/pneumonia_attack.yaml -s name=pneumonia_attack_trained \
    -s attack.trained_rounds=10 -s "attack.methods=[ig]"
for b in 4 16; do
  run attack -c configs/pneumonia_attack.yaml -s name=pneumonia_attack_batch$b \
      -s attack.batch_size=$b -s attack.n_targets=5 -s attack.gallery_size=8 \
      -s "attack.methods=[ig]" -s "attack.noise_multipliers=[]"
done

# Imaging: federated training and DP trade-off
run train    -c configs/bloodmnist_noniid.yaml
run tradeoff -c configs/bloodmnist_noniid.yaml

# v0.3: malicious hospitals vs. robust aggregation
run robustness -c configs/bloodmnist_byzantine.yaml
run robustness -c configs/pneumonia_backdoor.yaml
run robustness -c configs/bloodmnist_byzantine.yaml -s name=bloodmnist_byzantine_sweep \
    -s "seeds=[0]" -s "robustness.attacks=[none,alie,sign_flip]" \
    -s "robustness.n_malicious=[1,2,3,4]"
