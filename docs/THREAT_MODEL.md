# Threat model (STRIDE)

This document analyses the FedSecHealth deployment (`deploy/`) with STRIDE and
links every threat to the mitigation in place and, where one exists, to the
experiment in this repository that measures it. It is the security
counterpart of the research results in the README.

## System and trust boundaries

```
 Hospital A (SuperNode) ──┐                    ┌── Operator workstation
 Hospital B (SuperNode) ──┼── TLS, gRPC 9092 ──┤   (`flwr run`, HTTPS 8000,
 Hospital C (SuperNode) ──┘        │           │    127.0.0.1 only)
                                   ▼           │
                        Aggregation server (SuperLink + ServerApp)
```

| Asset | Where it lives | Why it matters |
|---|---|---|
| Patient data (images, records, labels) | Each hospital only | Health data (GDPR art. 9, medical secrecy) |
| Model updates | In transit, and at the server each round | Can be inverted into patient data (v0.1, v0.2) |
| Global model | Server, then every hospital | Leaks membership (v0.4); can carry a backdoor (v0.3) |
| CA key, server TLS key | Operator (demo: `deploy/certs/`) | Whoever holds them can impersonate the server |
| Run configuration / Flower App | Operator → SuperLink | Decides what code runs on every hospital |

Trust boundaries: (1) the network between hospitals and the server;
(2) the server itself, which is **not** trusted with individual updates;
(3) each hospital, which is **not** trusted by the others (it may be
malicious); (4) the software supply chain of the code the server ships.

## STRIDE analysis

| # | Category | Threat | Mitigation in FedSecHealth | Evidence / residual risk |
|---|---|---|---|---|
| S1 | **Spoofing** | A rogue server impersonates the SuperLink and collects hospital updates | TLS with a private CA; SuperNodes verify the server certificate (`--root-certificates`, SAN `superlink` / `localhost`) | Deployment refuses connections without the CA. Residual: CA key protection (demo keeps it on disk) |
| S2 | Spoofing | A rogue participant joins the federation as a hospital | Not mitigated yet (TLS authenticates the server only) | Flower SuperNode authentication (per-node keys, `--enable-supernode-auth`) is on the roadmap |
| T1 | **Tampering** | Updates modified in transit | TLS (integrity + confidentiality of the channel) | |
| T2 | Tampering | A malicious hospital poisons the model (sabotage, backdoor) | Robust aggregation available in the deployment (`strategy = median / trimmed_mean / krum / multi_krum`) | v0.3: one hospital of ten plants a backdoor under FedAvg (99.8 % success); no rule wins against every attack (ALIE breaks median / Krum) |
| T3 | Tampering | Submitted app pulls malicious dependencies at run time | `--disable-runtime-dependency-installation`: apps run in the pinned, locked environment (`uv.lock`) of the image | Residual: trust in the image build and in PyPI at build time |
| R1 | **Repudiation** | A hospital denies having sent a harmful update | Server logs per round; not signed | Signed updates / SuperNode auth needed for accountability |
| I1 | **Information disclosure** | Server reconstructs patient images from an update | DP-SGD inside each hospital (`dp-enabled`); secure aggregation (research code, v0.4) | v0.2: X-rays reconstructed at SSIM 0.98 without noise; broken at ε = 8. Secure aggregation hides individual updates (masked upload correlation 0.0003) but is not yet wired into the Flower deployment |
| I2 | Information disclosure | Anyone holding the model infers whether a patient was in a hospital's data | DP-SGD | v0.4: LiRA identifies 4.4 % of patients at 1 % FPR without DP; chance level at ε = 8, with a utility cost |
| I3 | Information disclosure | Eavesdropper reads updates on the network | TLS | |
| I4 | Information disclosure | Compromised container reads other hospitals' data | One container per hospital, data mounted read-only, non-root user, read-only root filesystem, no published hospital port | In this simulation every container mounts the whole dataset and selects its shard; a real hospital would mount only its own data |
| D1 | **Denial of service** | Hospitals drop out mid-round | Flower waits for `min_available_nodes`; secure aggregation tolerates dropouts above its threshold (v0.4: 30 % per round) | A single hospital failing stalls a round that requires all of them (demo setting) |
| D2 | Denial of service | Flooding the server | Control API bound to 127.0.0.1; only the Fleet API is reachable by hospitals | No rate limiting in the demo |
| E1 | **Elevation of privilege** | Code execution on hospitals via the Flower App the server ships | Hospitals only run the app installed in their image; no runtime installs | Residual: the operator who can `flwr run` controls what executes on every hospital, so that access must be restricted |
| E2 | Elevation of privilege | Container escape / privilege escalation | Non-root user (UID 1000), read-only filesystem, tmpfs for scratch space | Standard container hardening; no seccomp/AppArmor profile beyond Docker defaults |

## What is deliberately out of scope

- Physical security, hospital IT compromise beyond the FL client, insider
  access to raw data inside a hospital.
- Formal compliance work (GDPR DPIA, HDS certification in France): the
  threat model informs it but does not replace it.
- Side channels on the aggregation server (timing, memory).

## Main takeaways for a real deployment

1. **TLS is necessary but only protects the channel.** The server still sees
   every update; that exposure is exactly what gradient inversion and
   membership inference exploit.
2. **Privacy and robustness pull in opposite directions.** Secure
   aggregation would hide the updates the robust rules need to inspect; the
   deployment must pick, per threat model, which risk dominates.
3. **The operator is a single point of trust.** Whoever can submit a Flower
   App controls the code on every hospital: access to the control API
   (here bound to localhost) must be as protected as the CA key.
