# Roadmap

Each milestone is a self-contained, presentable release.

## v0.1: Foundations (current)
- [x] Clean package (`src/` layout, `uv`, typed configs, CLI)
- [x] Breast Cancer Wisconsin across N simulated hospitals, IID and Dirichlet non-IID splits
- [x] Federated feature standardisation (no pooled statistics)
- [x] FedAvg simulator, plus centralized and local-only baselines
- [x] DP-SGD per hospital with Opacus and (ε, δ) accounting
- [x] Gradient inversion: analytic (linear layer), DLG, iDLG
- [x] Privacy/utility trade-off sweep (ε vs. accuracy vs. attack success)
- [x] Tests + CI (ruff, pytest)

## v0.2: Medical imaging
- [ ] MedMNIST (PneumoniaMNIST, DermaMNIST, BloodMNIST) with a small CNN (GroupNorm, Opacus-compatible)
- [ ] Inverting Gradients (Geiping et al., 2020): cosine loss + total variation prior
- [ ] Image metrics: PSNR, SSIM, LPIPS; reconstruction galleries
- [ ] Attacks on larger batches and on trained (not only initial) models
- [ ] GPU support (ROCm/CUDA), see `docs/GPU.md`

## v0.3: Malicious hospitals (integrity)
- [ ] Byzantine clients: label flipping, sign flipping, scaled updates
- [ ] Backdoor (trigger) attacks, model replacement (Bagdasaryan et al., 2020)
- [ ] Robust aggregation: Krum / Multi-Krum, coordinate-wise median, trimmed mean, FLTrust
- [ ] Metrics: main-task accuracy and attack success rate vs. % malicious clients

## v0.4: Advanced privacy
- [ ] Membership inference (loss/threshold and shadow models; LiRA-style)
- [ ] Secure aggregation (pairwise masking, Bonawitz et al., 2017): what it does and does not protect
- [ ] Client-level vs. sample-level DP; DP-FedAvg with server-side noise
- [ ] Gradient compression / pruning as a (weak) defense, for comparison

## v0.5: Realistic deployment and presentation
- [ ] Flower deployment (`ServerApp` / `ClientApp`) reusing the same hospital logic; Docker Compose with one container per hospital
- [ ] TLS between hospitals and server, threat model document (STRIDE)
- [ ] Streamlit dashboard: run scenarios, visualise reconstructions live
- [ ] Technical report (≈8 pages, paper style) + slide deck for interviews
