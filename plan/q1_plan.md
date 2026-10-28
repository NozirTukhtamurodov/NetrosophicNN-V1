# Q1 analysis plan — PRE-REGISTERED

Written: 2026-10-02, before any of the experiments below were run. This file is not edited after it is written.
Anything decided after seeing results is labelled EXPLORATORY in the report.
Reference for all existing settings: `audit/AUDIT_REPORT.md`, 2026-10-02.
All new outputs go to `q1/`. stdout of every run goes to `q1/logs/` with its wall-clock time. No existing result is overwritten.

## Facts established before writing (no results seen)

- **KSC:** the image cube `KSC.mat` / `KSC_corrected.mat` (512×614×176) is available from HuggingFace `danaroth/kennedy_space_center`. The ground truth `KSC_gt.mat` is not:
  - absent from that repository;
  - ehu.eus returns 403;
  - the probed GitHub mirrors return 404.

  **KSC is therefore not used.** The scenes are Pavia University and Salinas only. Indian Pines is not used for the CNN.
- **No neutrosophic CNN exists in the project.** The only CNN code is `neutro_cnn/hsi_train.py`, which is evidential. **NS-U is implemented as specified in B3.** It has no independent I head, so the rung N3 = (T, F, I) is **not applicable** and is not run.

## P1. Detector

- Model: StandardScaler + LogisticRegression (L2, C = 1, lbfgs, max_iter 2000).
- Validation: stratified 5-fold out-of-fold (StratifiedKFold, shuffle = True, random_state = 0).
- AUC: one AUC over all OOF scores.
- Bootstrap: paired, B = 3000, numpy default_rng(1337), percentile 95% CI. The same resample indices are used for every rung of one sample. Resamples containing a single class are skipped.
- **Success rule:** Δ > 0.03 AND CI_low > 0.
- This is the existing `ns_logprob_detect.oof_scores` / `boot_delta`, unchanged.

## P2. Target

y = 1 if the network's predicted class ≠ true class, on test pixels only. For the MC-dropout and ensemble ladders, see P3.

## P3. Ladders

Nested unless stated. H is the natural-log entropy.

**Softmax network.** p = softmax(logits).

| Rung | Features |
|---|---|
| S1 | H(p) |
| S2 | (H, p_max) |
| S3 | (H, p_max, p_2) |

Steps: S2 − S1, S3 − S2.

**Evidential network.** α = softplus(logits) + 1, S = Σα, u = K/S, b = (α − 1)/S.

| Rung | Features |
|---|---|
| R1 | H(α/S) |
| R2 | (H, u) |
| R3+ | (H, u, b_max, b_2) |
| R3 | (b_max, b_2, u) — head-to-head, not nested |

Steps: R2 − R1, R3+ − R2, R3 − R2.

**Neutrosophic network (NS-U).**

- T_k = sigmoid(z_T,k), F_k = sigmoid(z_F,k).
- k* = argmax_k (T_k − F_k); T = T_k*, F = F_k*.
- Softmax-type features of the same network are taken from p̃ = softmax(z_T − z_F): H(p̃) and p̃_max.

| Rung | Features |
|---|---|
| N1 | T − F |
| N2 | (T, F) |
| N2base | (H(p̃), p̃_max) |
| N2+ | (H(p̃), p̃_max, T, F) |

Steps: N2 − N1, N2+ − N2base. N3 is not applicable (no I head).

Also reported: the share of test pixels with T + F > 1 and with T + F < 1.

**MC dropout** (softmax network, 20 passes, dropout active).

- Target: the errors of the **deterministic** softmax prediction, i.e. the same y as the S ladder.
- Features:
  - PE = entropy of the mean probability;
  - MI = PE − mean entropy (BALD);
  - p_max of the mean.

| Rung | Features |
|---|---|
| M1 | PE |
| M2 | (PE, MI) |
| M3 | (PE, MI, p_max) |

Steps: M2 − M1, M3 − M2.

The OA of the MC-mean prediction is reported as well.

**Deep ensemble** (5 softmax networks, split seed 0).

- Target: errors of the ensemble prediction (argmax of the mean probability).

| Rung | Features |
|---|---|
| E1 | H(mean) |
| E2 | (H, MI) |
| E3 | (H, MI, p_max) |

Steps: E2 − E1, E3 − E2.

## P4. Primary comparisons

Every step listed in P3, for:

1. every run (scene × model × split seed × training seed);
2. the pooled concatenation over all runs of one scene × model.

## P5. Variability

- Mean ± sd (sample sd, ddof = 1) over training seeds of every AUC and Δ:
  - within each split (3 training seeds);
  - over all 9 runs of a scene × model.
- Bootstrap CIs are reported per run in addition.

## A. Hyperspectral protocol

- **A1. Preprocessing.** Per-band standardisation and PCA (30 components) are fitted on the **training pixels** of each split only, then applied to all pixels. Everything else is as in the audit:
  - 16×16 blocks;
  - ≥ 15% of labelled pixels to training;
  - 6-px Chebyshev buffer;
  - 11×11 patches with reflect padding;
  - 40 epochs, AdamW lr 1e-3, wd 1e-4, cosine schedule;
  - batch 128;
  - no validation set.
- **A2. Seeds.** Split seed s ∈ {0, 1, 2} (numpy default_rng(s) for the block shuffle). Training seed t ∈ {0, 1, 2} (torch.manual_seed(t), also used for the PCA random_state). This gives 9 runs per scene per model.
- **A3. Scenes.** Pavia University and Salinas. KSC is unavailable (see above).
- **A4. Outputs.** Per-pixel test outputs are saved for every run as `.npz` in `q1/hsi/`, with:
  - y, pred, pixel coordinates;
  - raw logits (both heads for NS-U);
  - MC-dropout mean probabilities and per-pixel PE/MI where applicable;
  - the training-pixel count per class.

## B. Models

Backbone: HSINet, unchanged from the audit B1, unless stated.

| ID | Model | Specification |
|---|---|---|
| B1 | Softmax CNN | Cross-entropy loss. |
| B2 | Evidential CNN | As in the audit: softplus evidence, MSE-EDL loss, KL annealed with λ = min(1, epoch/10). |
| B3 | NS-U | Two linear heads (128 → K each) after the shared dropout layer. Loss = mean over pixels of Σ_k [BCE(T_k, y_k) + BCE(F_k, 1 − y_k)], with y one-hot. Prediction k* = argmax (T − F). |
| B4 | MC dropout | On every B1 run: 20 test-time passes with only the Dropout(0.3) layer in train mode (BatchNorm kept in eval mode). Fixed seed torch.manual_seed(1000 + t) before the passes. |
| B5 | Deep ensemble | Split seed 0, init seeds t = 0..4. Members t = 0, 1, 2 are the B1 runs of split 0 (identical configuration); only t = 3, 4 are trained additionally. One ensemble per scene. |
| B6 | HybridSN-style | Input 1 × 30 × 11 × 11. Conv3d 8 @ (7 spectral × 3 × 3) → Conv3d 16 @ (5 × 3 × 3) → Conv3d 32 @ (3 × 3 × 3), valid padding, ReLU after each. Reshape to channels × 5 × 5 → Conv2d 64 @ 3 × 3, valid, ReLU → flatten → Linear 256 + ReLU → Dropout 0.4 → Linear K. Trained in softmax (B1 loss) and evidential (B2 loss) versions, split seeds 0, 1, 2, training seed 0, with the same optimiser and schedule. |

Note on B6: the original HybridSN uses 25 × 25 patches. 11 × 11 is kept here so that split and buffer stay identical.

## C. Reporting

- **C1–C3.** As requested, from the saved `.npz` files.
- **C4. Cross-model.** Each model's AUCs refer to **its own errors**, so targets differ between models. This is stated next to the table.
- **C5. Per-class accuracy** per run, and the classes with 0 training pixels in any split.
- **C6. Old vs new preprocessing.** Old = transductive-PCA runs (`neutro_cnn/data/hsi_*_s{s}.jsonl`, init seed = split seed). New = evidential runs with split s and training seed t = s. These differ only in the PCA fitting (and in non-deterministic cuDNN kernels).

## D. Preference data (CPU, existing scored files)

- **D1.** For every stored preference ladder block (E3.4–E3.8 sets, scorer variants and SHP E4a/E4b), recompute the point Δ with fold seeds 0, 1, 2, 3, 4. Report mean ± sd (ddof = 1). Bootstrap CIs are not recomputed per fold seed: fold seed 0 reproduces the stored CIs.
- **D2.** AUC of the signed margin T − F against the corruption flag on the 2,000 DPO training pairs. No DPO training.

## G. Two short articles (CPU)

**G1. Spatial leakage.** Settings are as in `neutro_cnn/c0_leakage.py`:

- 10% training, block 16, buffer 5, seeds 0–2;
- raw spectra, standardised on training pixels;
- 1-NN (Euclidean), and LR with the original max_iter = 300.
- **Additional pre-registered row:** LR with max_iter = 5000, to address the non-convergence.

Per scene × classifier × protocol × seed, report:

- OA;
- AA (= mean per-class recall, which by definition equals balanced accuracy — both are reported);
- κ;
- n_train, n_test;
- per-class accuracy;
- training pixels per class.

**Sweep:** 1-NN on Pavia University and Salinas, block ∈ {8, 16, 32} × buffer ∈ {0, 3, 5, 10}, seeds 0–2, 10% training. Leakage = OA(random) − OA(block, buffer), using the random-split OA of the same seed.

**G2. Length confound.** Existing scored files of the four sources and SHP.

- Lengths are in characters, **as originally labelled**: a swapped pair is un-swapped.
- Report:
  - median and IQR for chosen and rejected;
  - share of pairs where chosen is longer;
  - raw length-only AUC (score = L_presented-chosen − L_presented-rejected, target = swap flag), with no fitting;
  - for filter thresholds 0.10 / 0.25 / 0.50: the retained share and the same AUC on the retained pairs.
- **UltraFeedback:** the same AUC by rating-gap bins [1, 2), [2, 3), [3, 4), ≥ 4, plus (0, 1) if present, within each UF file separately.
- **SHP:** by vote-ratio bins [1, 1.5), [1.5, 2), [2, 3), [3, 5), ≥ 5.

## E. Metadata

GPU-hours = the sum of logged wall times of the GPU runs. Every deviation from the request is listed in the report.
