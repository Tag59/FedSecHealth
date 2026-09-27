# Roadmap

Each milestone is a self-contained, presentable release.

## v0.1: Foundations (done)
- [x] Clean package (`src/` layout, `uv`, typed configs, CLI)
- [x] Breast Cancer Wisconsin across N simulated hospitals, IID and Dirichlet non-IID splits
- [x] Federated feature standardisation (no pooled statistics)
- [x] FedAvg simulator, plus centralized and local-only baselines
- [x] DP-SGD per hospital with Opacus and (ε, δ) accounting
- [x] Gradient inversion: analytic (linear layer), DLG, iDLG
- [x] Privacy/utility trade-off sweep (ε vs. accuracy vs. attack success)
- [x] Tests + CI (ruff, pytest)

## v0.2: Medical imaging (done)
- [x] MedMNIST (PneumoniaMNIST, BloodMNIST, DermaMNIST), federated per-channel standardisation
- [x] CNN (GroupNorm, Opacus-compatible) and LeNet (DLG reference model)
- [x] Inverting Gradients (Geiping et al., 2020): cosine loss + TV prior, signed Adam, box constraints
- [x] Attacks decoupled from ground truth; PSNR / SSIM with Hungarian matching for batches
- [x] Reconstruction galleries, noise-multiplier sweep (where attacks break, and at which ε)
- [x] Attacks on trained models and on batches of 4 and 16 images
- [x] Balanced accuracy for imbalanced medical classes
- [ ] LPIPS (needs pretrained weights), label inference for batches (Wainakh et al.)
- [ ] GPU support (ROCm/CUDA) tested on real hardware, see `docs/GPU.md`

## v0.3: Malicious hospitals (integrity, done)
- [x] Byzantine clients: label flipping, sign flipping, Gaussian updates, ALIE (Baruch et al., 2019)
- [x] Backdoor (trigger) attacks with model-replacement boosting (Bagdasaryan et al., 2020)
- [x] Robust aggregation: coordinate-wise median, trimmed mean, Krum / Multi-Krum, norm clipping, FLTrust
- [x] Metrics: balanced accuracy, backdoor success rate (with clean-model baseline), influence kept by malicious clients
- [x] Aggregator x attack grid and sweeps over the number of malicious hospitals
- [ ] Adaptive attacks aware of the defense (e.g. FLTrust-aware, Fang et al. 2020)
- [ ] Combining DP with robust aggregation

## v0.4: Advanced privacy (current)
- [x] Membership inference: loss, confidence, modified entropy, offline LiRA with shadow models
- [x] Evaluation at low false-positive rates (TPR at 1 % FPR), global model vs. a hospital's local model
- [x] Secure aggregation (Bonawitz et al., 2017): X25519 key agreement, ChaCha20 masks, Shamir-shared secrets, dropout recovery
- [x] Client-level DP (DP-FedAvg) vs. sample-level DP (DP-SGD)
- [ ] Online LiRA (shadow models trained with and without each target)
- [ ] Gradient compression / pruning as a (weak) defense, for comparison
- [ ] Distributed DP (noise added by hospitals under secure aggregation)

## v0.5: Realistic deployment and presentation
- [ ] Flower deployment (`ServerApp` / `ClientApp`) reusing the same hospital logic; Docker Compose with one container per hospital
- [ ] TLS between hospitals and server, threat model document (STRIDE)
- [ ] Streamlit dashboard: run scenarios, visualise reconstructions live
- [ ] Technical report (≈8 pages, paper style) + slide deck for interviews
