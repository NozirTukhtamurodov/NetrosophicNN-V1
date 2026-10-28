# Revision analyses (post hoc): report

Manuscript: "How Many Degrees of Freedom Does a Neutrosophic Uncertainty Representation Have? Theory, and Tests on
Hyperspectral Convolutional Networks and Preference Data". Review of 26 Oct 2026.

*Release copy: file paths in this report refer to the working layout of the project (`q1/`, `revision/`, `results/revision/`); the corresponding release scripts and files are listed in `CODE_MAP.md`.*

**All analyses in this file are revision analyses (post hoc).** They were not part of the pre-registered plan
(`audit/q1_plan.md`). The original results (`q1/hsi/`, `q1/tables/`, `audit/`) were not edited. Every new file is in
`results/revision/`; the code is in `revision/`. Every run, split, seed and contrast is reported in the CSV files; nothing
was selected or dropped.

**Reproduction check.** The recomputation engine reproduces all 196 per-run contrasts of the original table
(`q1/tables/C3_deltas.csv`): the maximum absolute difference in mean ΔAUC and CI bounds is 5.0e-5. The q1 table is
rounded to 4 decimals. The 28 pooled contrasts are reproduced with a maximum difference of 4.9e-5
(`check_engine_vs_table4.txt`, `c12_blocked_vs_original_pooled.csv`).

**Table 4 = the primary family.** Table 4 of the manuscript has 11 rows, and each cell reads "mean ΔAUC (SD over runs); runs passing rule (10)". It covers 166 per-run comparisons: 162 CNN (9 runs per scene per row) and 4 ensemble (1 per scene, split 0). HybridSN (30 per-run steps) is secondary and not in Table 4. Every revised table below keeps the 11 Table 4 rows in the manuscript order; the B6 single-coordinate rows are added beneath them, marked "revision, post hoc". The recomputed Table 4 (`t4_table4_original.csv`) equals the manuscript in every cell, with one exception: NS-U N2 − N1, Pavia, SD = 0.06047, which prints as 0.060 here and as 0.061 in the manuscript. The manuscript value is a double rounding (0.0605 → 0.061). All formatted tables are collected in `t4_tables.md`, with one CSV per table (`t4_*.csv`).

**Retrained models (replicas).** The q1 runs did not save network weights. C1(a) and C3 need them, so all 70 q1
configurations were retrained with the same code, seeds and call order, this time saving the weights
(`revision/hsi_rev.py`, variant `replica`). Training is **not deterministic** on this GPU: q1 did not enable deterministic
cuDNN algorithms, and the replicas reuse that code unchanged. Fidelity of each replica to the stored q1 outputs is in
`c21_replica_fidelity.csv`:

| Group | Replicas | Bit-identical | Prediction agreement with q1 (min–max) | OA, replica − q1 (min–max) |
|---|---|---|---|---|
| HSINet softmax (incl. ensemble members), evidential, NS-U | 58 | 20 | 0.9331–1.0000 | −0.0120 to +0.0108 |
| HybridSN softmax and evidential | 12 | 0 | 0.4000–0.9787 | −0.0300 to +0.0131 |

The 20 bit-identical replicas are all 9 Pavia softmax, all 9 Pavia evidential, and Pavia NS-U s0 t0, s0 t1. **C1(a) and
C3 are therefore run on retrained replicas, not on the exact manuscript models.** For the other 50 runs these are
networks from the same protocol and seeds, differing by GPU non-determinism. The degenerate Salinas HybridSN evidential
split 0 replica agrees with q1 on only 40% of predictions.

---

## Comment 21 (and 2): pre-registration, provenance, repository contents (A1)

**What was run.** `revision/a1_a2_splits.py`, `revision/a1_manifest.py`.

| Item | Value |
|---|---|
| Plan file | `audit/q1_plan.md` |
| SHA-256 | `ea3d0f528f94c52c0b59bf30256bb3efedeed41ddcdf7d0f22beeff02e79aabf` |
| Recorded hash | `q1/logs/q1_plan.sha256` (same value), file written 2026-10-02 10:38:10 +05:00 |
| Plan creation time / last modification (NTFS) | 2026-10-02 10:37:58 / 2026-10-02 10:37:58 |
| First HSI run start (`q1/logs/runs_hsi.csv`) | 2026-10-02 10:39:24 (`pavia_u_softmax_s0_t0`) |
| Last HSI output written | 2026-10-02 14:29:59 (`salinas_hybrid_evidential_s2_t0.npz`) |
| Git history | **none** — the project directory is not a git repository |

- Split files: `results/revision/splits/<scene>_split<s>.npz`. Each holds the train and test pixel coordinates, labels and
  16×16 block IDs. They were regenerated with the original function (`split_block_buffer`, rng = default_rng(split)).
  They match all 70 saved runs exactly: n_train, per-class train counts, and test coordinates and labels
  (`c21_split_check_vs_saved_runs.csv`, 0 mismatches). Split manifest with hashes: `c21_split_manifest.csv`.
- Repository manifest (124 files with SHA-256, size and modification time): raw scenes, plan, per-pixel outputs, run log,
  result tables, code, preference inputs and scores, split files. See `c21_repository_manifest.csv`.
- Seeds: split seed s ∈ {0, 1, 2} (numpy default_rng(s)); training seed t ∈ {0, 1, 2} (torch.manual_seed(t), also the PCA
  random_state); ensemble members t = 0..4 on split 0; MC-dropout seed 1000 + t; detector fold seed 0; bootstrap seed 1337.

**Summary.** The plan file's hash equals the hash recorded 12 s after it was written, and the first run started 86 s
after the plan was written. These times are local file-system timestamps, not an external registry. There is no git
history.

**Deviations / missing.** No git repository, so there is no commit hash and no history (see *Missing inputs*). No
external timestamp.

## Comment 13: class coverage and training-absent classes (A2, B7)

**A2.** Full table (scene × split × class: labelled, train and test pixels; train and test blocks): `c13_class_coverage.csv`.
Classes with **0 training pixels**:

| Scene | Split | Class | Labelled px | Test px | Test blocks |
|---|---|---|---|---|---|
| Pavia University | 0 | 5 | 1,345 | 1,345 | 25 |
| Salinas | 1 | 14 | 1,070 | 1,004 | 16 |
| Salinas | 1 | 16 | 1,807 | 1,807 | 11 |

**B7.** The ladders were rerun with the test pixels of those classes removed. This applies only to the two splits
that have such classes, Pavia split 0 and Salinas split 1, with 3 runs each. The ensemble exists only for split 0, so
Salinas has no ensemble cell. The detector was refit on the subset. Per comparison: `c13_absent_classes_removed.csv`.

Totals over the Table 4 comparisons of these runs (runs passing rule (10)): Pavia split 0, 9/29 with all classes → 3/29
with the unseen class removed. Salinas split 1, 3/27 → 3/27.

Only the splits with training-absent classes: Pavia University split 0 (class 5) and Salinas split 1 (classes 14, 16); 3 runs each; ensemble only exists for split 0 (Pavia: 1 run; Salinas: n/a). The detector is refit on the subset.

| Model | Step | Adds | Type | Row source | Pavia University split 0: all classes (same runs) | Pavia University split 0: unseen classes removed | Salinas split 1: all classes (same runs) | Salinas split 1: unseen classes removed |
|---|---|---|---|---|---|---|---|---|
| Softmax | S2 − S1 | p_max | nested | Table 4 | +0.012 (0.013); 0/3 | +0.000 (0.000); 0/3 | +0.001 (0.003); 0/3 | +0.002 (0.002); 0/3 |
| Softmax | S3 − S2 | p_(2) | nested | Table 4 | -0.002 (0.013); 0/3 | -0.001 (0.004); 0/3 | +0.000 (0.001); 0/3 | +0.002 (0.002); 0/3 |
| MC dropout | M2 − M1 | MI | nested | Table 4 | +0.056 (0.009); 3/3 | +0.009 (0.007); 0/3 | -0.000 (0.000); 0/3 | +0.006 (0.004); 0/3 |
| MC dropout | M3 − M2 | p_max | nested | Table 4 | +0.041 (0.095); 1/3 | -0.000 (0.002); 0/3 | -0.000 (0.001); 0/3 | +0.003 (0.003); 0/3 |
| Ensemble | E2 − E1 | MI | nested | Table 4 | +0.035; 1/1 | +0.000; 0/1 | n/a | n/a |
| Ensemble | E3 − E2 | p_max | nested | Table 4 | -0.008; 0/1 | -0.001; 0/1 | n/a | n/a |
| Evidential | R2 − R1 | u | nested | Table 4 | +0.048 (0.025); 2/3 | +0.000 (0.000); 0/3 | -0.000 (0.000); 0/3 | +0.000 (0.000); 0/3 |
| Evidential | R3+ − R2 | (b_max, b_(2)) | nested, 2-D (combined increment) | Table 4 | +0.007 (0.004); 0/3 | -0.000 (0.000); 0/3 | -0.001 (0.001); 0/3 | -0.000 (0.000); 0/3 |
| Evidential | R3 − R2 | triple vs pair | head-to-head | Table 4 | -0.049 (0.031); 0/3 | +0.000 (0.001); 0/3 | -0.000 (0.001); 0/3 | -0.000 (0.000); 0/3 |
| NS-U | N2 − N1 | F given T − F | nested | Table 4 | -0.004 (0.005); 0/3 | -0.011 (0.019); 0/3 | +0.002 (0.004); 0/3 | +0.002 (0.003); 0/3 |
| NS-U | N2+ − N2base | (T, F) | nested, 2-D (combined increment) | Table 4 | +0.110 (0.122); 2/3 | +0.075 (0.023); 3/3 | +0.106 (0.052); 3/3 | +0.075 (0.015); 3/3 |
| Evidential | (H,u,b_max) − R2 | b_max | nested, 1-D | revision, post hoc | +0.008 (0.004); 0/3 | -0.000 (0.000); 0/3 | -0.001 (0.001); 0/3 | -0.000 (0.000); 0/3 |
| Evidential | R3+ − (H,u,b_max) | b_(2) given b_max | nested, 1-D | revision, post hoc | -0.001 (0.001); 0/3 | +0.000 (0.000); 0/3 | -0.000 (0.000); 0/3 | +0.000 (0.000); 0/3 |
| Evidential | (H,u,b_(2)) − R2 | b_(2) | nested, 1-D | revision, post hoc | +0.007 (0.004); 0/3 | -0.000 (0.000); 0/3 | -0.001 (0.001); 0/3 | -0.000 (0.000); 0/3 |
| Evidential | BF − R2 | binary-frame triple (b_max, 1−u−b_max, u) vs (H,u) | head-to-head | revision, post hoc | -0.050 (0.032); 0/3 | +0.000 (0.001); 0/3 | -0.000 (0.001); 0/3 | -0.000 (0.000); 0/3 |
| Evidential | BF − (b_max,u) | b_other = 1−u−b_max | nested, affine-dependent | revision, post hoc | -0.000 (0.000); 0/3 | -0.000 (0.000); 0/3 | -0.000 (0.000); 0/3 | -0.000 (0.000); 0/3 |
| NS-U | N2base+T − N2base | T | nested, 1-D | revision, post hoc | +0.112 (0.125); 2/3 | +0.088 (0.005); 3/3 | +0.103 (0.050); 3/3 | +0.073 (0.013); 3/3 |
| NS-U | N2+ − (N2base+T) | F given T | nested, 1-D | revision, post hoc | -0.002 (0.003); 0/3 | -0.013 (0.022); 0/3 | +0.002 (0.003); 0/3 | +0.001 (0.002); 0/3 |
| NS-U | N2base+F − N2base | F | nested, 1-D | revision, post hoc | +0.113 (0.127); 2/3 | +0.086 (0.006); 3/3 | +0.103 (0.049); 3/3 | +0.073 (0.013); 3/3 |

**Summary.**
- In Pavia split 0, every passing comparison of softmax, MC dropout, evidential and the ensemble stops passing once the
  1,345 pixels of the training-absent class are removed:
  - MC dropout M2 − M1: 3/3 → 0/3;
  - MC dropout M3 − M2: 1/3 → 0/3;
  - evidential R2 − R1: 2/3 → 0/3;
  - ensemble E2 − E1: 1/1 → 0/1.
- NS-U N2+ − N2base passes in 3/3 runs after removal in both scenes (2/3 and 3/3 before).
- In Salinas split 1, no comparison of the other models passes, before or after removal.

## Comment 14: accounting of comparisons, p-values and multiplicity (A3, B3)

**A3.** All 224 contrasts are listed in `c14_all_comparisons.csv`. The accounting is **confirmed**:

| Family | Per-run contrasts | Pooled contrasts |
|---|---|---|
| CNN (HSINet): softmax S 18, MC dropout M 18, evidential R 27, NS-U N 18 per scene | 162 | 18 |
| Deep ensemble (split 0, one per scene × 2 steps) | 4 | — |
| **Table 4 = primary family (CNN + ensemble)** | **166** | |
| HybridSN, secondary: softmax 6 + evidential 9 per scene | 30 (18 evidential, degenerate) | 10 (6 evidential) |
| **Total** | **196** | **28** |

Of the 166 Table 4 comparisons, 28 pass rule (10): Pavia 19, Salinas 9. No HybridSN comparison passes; of the 28 pooled,
2 pass.

**B3.** Two-sided bootstrap p = 2·min(share ΔAUC* ≤ 0, share ΔAUC* ≥ 0), computed from the same B = 3000 resamples (seed 1337).
Benjamini–Hochberg at q = 0.05 over the **166 Table 4 comparisons**. The B6 rows form a separate post hoc family of
144 comparisons. Per comparison: `t4_B3_pvalues_per_run.csv`; all 196: `c14_pvalues_bh.csv`.

| Family | n | BH rejections | BH rejection and Δ > 0.03 | Rule (10) passes | p = 0 at resolution 1/B |
|---|---|---|---|---|---|
| Table 4 (primary) | 166 | 134 | 28 | 28 | 115 |
| B6 single-coordinate rows (revision, post hoc) | 144 | 127 | 30 | 30 | 106 |

p = 2·min(share Δ* ≤ 0, share Δ* ≥ 0) from the same B = 3000 resamples. BH family for the Table 4 rows = the 166 per-run comparisons of Table 4; the B6 rows form a separate post hoc family (144 comparisons).

| Model | Step | Adds | Type | Row source | Pavia University: rule (10): mean (SD); passing | Pavia University: BH q=0.05 rejections | Pavia University: BH rejection and Δ > 0.03 | Salinas: rule (10): mean (SD); passing | Salinas: BH q=0.05 rejections | Salinas: BH rejection and Δ > 0.03 |
|---|---|---|---|---|---|---|---|---|---|---|
| Softmax | S2 − S1 | p_max | nested | Table 4 | +0.008 (0.009); 0/9 | 8/9 | 0/9 | +0.000 (0.002); 0/9 | 5/9 | 0/9 |
| Softmax | S3 − S2 | p_(2) | nested | Table 4 | -0.002 (0.007); 0/9 | 5/9 | 0/9 | -0.000 (0.003); 0/9 | 3/9 | 0/9 |
| MC dropout | M2 − M1 | MI | nested | Table 4 | +0.029 (0.027); 6/9 | 9/9 | 6/9 | -0.000 (0.001); 0/9 | 5/9 | 0/9 |
| MC dropout | M3 − M2 | p_max | nested | Table 4 | +0.011 (0.053); 1/9 | 8/9 | 1/9 | +0.002 (0.004); 0/9 | 7/9 | 0/9 |
| Ensemble | E2 − E1 | MI | nested | Table 4 | +0.035; 1/1 | 1/1 | 1/1 | +0.000; 0/1 | 0/1 | 0/1 |
| Ensemble | E3 − E2 | p_max | nested | Table 4 | -0.008; 0/1 | 1/1 | 0/1 | -0.007; 0/1 | 1/1 | 0/1 |
| Evidential | R2 − R1 | u | nested | Table 4 | +0.021 (0.025); 2/9 | 9/9 | 2/9 | +0.001 (0.002); 0/9 | 6/9 | 0/9 |
| Evidential | R3+ − R2 | (b_max, b_(2)) | nested, 2-D (combined increment) | Table 4 | +0.003 (0.005); 0/9 | 8/9 | 0/9 | +0.001 (0.003); 0/9 | 6/9 | 0/9 |
| Evidential | R3 − R2 | triple vs pair | head-to-head | Table 4 | -0.019 (0.027); 0/9 | 8/9 | 0/9 | +0.001 (0.002); 0/9 | 8/9 | 0/9 |
| NS-U | N2 − N1 | F given T − F | nested | Table 4 | +0.010 (0.060); 1/9 | 9/9 | 1/9 | +0.012 (0.017); 2/9 | 9/9 | 2/9 |
| NS-U | N2+ − N2base | (T, F) | nested, 2-D (combined increment) | Table 4 | +0.130 (0.076); 8/9 | 9/9 | 8/9 | +0.072 (0.057); 7/9 | 9/9 | 7/9 |
| Evidential | (H,u,b_max) − R2 | b_max | nested, 1-D | revision, post hoc | +0.002 (0.006); 0/9 | 8/9 | 0/9 | +0.001 (0.002); 0/9 | 8/9 | 0/9 |
| Evidential | R3+ − (H,u,b_max) | b_(2) given b_max | nested, 1-D | revision, post hoc | +0.001 (0.001); 0/9 | 7/9 | 0/9 | +0.000 (0.000); 0/9 | 7/9 | 0/9 |
| Evidential | (H,u,b_(2)) − R2 | b_(2) | nested, 1-D | revision, post hoc | +0.002 (0.005); 0/9 | 7/9 | 0/9 | +0.001 (0.002); 0/9 | 8/9 | 0/9 |
| Evidential | BF − R2 | binary-frame triple (b_max, 1−u−b_max, u) vs (H,u) | head-to-head | revision, post hoc | -0.020 (0.028); 0/9 | 8/9 | 0/9 | +0.001 (0.002); 0/9 | 8/9 | 0/9 |
| Evidential | BF − (b_max,u) | b_other = 1−u−b_max | nested, affine-dependent | revision, post hoc | -0.000 (0.000); 0/9 | 7/9 | 0/9 | +0.000 (0.000); 0/9 | 6/9 | 0/9 |
| NS-U | N2base+T − N2base | T | nested, 1-D | revision, post hoc | +0.121 (0.075); 8/9 | 9/9 | 8/9 | +0.062 (0.063); 6/9 | 8/9 | 6/9 |
| NS-U | N2+ − (N2base+T) | F given T | nested, 1-D | revision, post hoc | +0.009 (0.062); 1/9 | 9/9 | 1/9 | +0.010 (0.017); 2/9 | 9/9 | 2/9 |
| NS-U | N2base+F − N2base | F | nested, 1-D | revision, post hoc | +0.117 (0.079); 7/9 | 9/9 | 7/9 | +0.061 (0.065); 6/9 | 9/9 | 6/9 |

Mean, SD, min and max of ΔAUC over runs, per split and per training seed: `c14_delta_by_level.csv`.

**Summary.** BH rejects 134 of the 166 null hypotheses of no difference. Adding the effect-size condition (Δ > 0.03)
leaves the same 28 comparisons that pass rule (10). 115 of the 166 p-values are 0 at the bootstrap resolution (1/3000).

## Comment 15: decision threshold and alternative selective-prediction metrics (B2, B8)

**B2.** The verdict at Δ thresholds 0.01, 0.02, 0.03 and 0.05, always together with CI_lo > 0. Per comparison:
`c15_threshold_sensitivity.csv` (all 196 plus the pooled rows).

Totals over the 166 Table 4 comparisons:

| Threshold | Pavia | Salinas | Total |
|---|---|---|---|
| 0.01 | 28/83 | 11/83 | 39/166 |
| 0.02 | 22/83 | 9/83 | 31/166 |
| 0.03 (rule (10)) | 19/83 | 9/83 | 28/166 |
| 0.05 | 14/83 | 5/83 | 19/166 |

Runs passing per threshold; mean (SD) of ΔAUC is the same for all thresholds.

| Model | Step | Adds | Type | Row source | Pavia University: mean (SD) | Pavia University: pass thr 0.01 | Pavia University: pass thr 0.02 | Pavia University: pass thr 0.03 | Pavia University: pass thr 0.05 | Salinas: mean (SD) | Salinas: pass thr 0.01 | Salinas: pass thr 0.02 | Salinas: pass thr 0.03 | Salinas: pass thr 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Softmax | S2 − S1 | p_max | nested | Table 4 | +0.008 (0.009) | 3/9 | 1/9 | 0/9 | 0/9 | +0.000 (0.002) | 0/9 | 0/9 | 0/9 | 0/9 |
| Softmax | S3 − S2 | p_(2) | nested | Table 4 | -0.002 (0.007) | 0/9 | 0/9 | 0/9 | 0/9 | -0.000 (0.003) | 0/9 | 0/9 | 0/9 | 0/9 |
| MC dropout | M2 − M1 | MI | nested | Table 4 | +0.029 (0.027) | 6/9 | 6/9 | 6/9 | 2/9 | -0.000 (0.001) | 0/9 | 0/9 | 0/9 | 0/9 |
| MC dropout | M3 − M2 | p_max | nested | Table 4 | +0.011 (0.053) | 1/9 | 1/9 | 1/9 | 1/9 | +0.002 (0.004) | 0/9 | 0/9 | 0/9 | 0/9 |
| Ensemble | E2 − E1 | MI | nested | Table 4 | +0.035 | 1/1 | 1/1 | 1/1 | 0/1 | +0.000 | 0/1 | 0/1 | 0/1 | 0/1 |
| Ensemble | E3 − E2 | p_max | nested | Table 4 | -0.008 | 0/1 | 0/1 | 0/1 | 0/1 | -0.007 | 0/1 | 0/1 | 0/1 | 0/1 |
| Evidential | R2 − R1 | u | nested | Table 4 | +0.021 (0.025) | 5/9 | 3/9 | 2/9 | 2/9 | +0.001 (0.002) | 0/9 | 0/9 | 0/9 | 0/9 |
| Evidential | R3+ − R2 | (b_max, b_(2)) | nested, 2-D (combined increment) | Table 4 | +0.003 (0.005) | 1/9 | 0/9 | 0/9 | 0/9 | +0.001 (0.003) | 0/9 | 0/9 | 0/9 | 0/9 |
| Evidential | R3 − R2 | triple vs pair | head-to-head | Table 4 | -0.019 (0.027) | 0/9 | 0/9 | 0/9 | 0/9 | +0.001 (0.002) | 0/9 | 0/9 | 0/9 | 0/9 |
| NS-U | N2 − N1 | F given T − F | nested | Table 4 | +0.010 (0.060) | 2/9 | 1/9 | 1/9 | 1/9 | +0.012 (0.017) | 3/9 | 2/9 | 2/9 | 0/9 |
| NS-U | N2+ − N2base | (T, F) | nested, 2-D (combined increment) | Table 4 | +0.130 (0.076) | 9/9 | 9/9 | 8/9 | 8/9 | +0.072 (0.057) | 8/9 | 7/9 | 7/9 | 5/9 |
| Evidential | (H,u,b_max) − R2 | b_max | nested, 1-D | revision, post hoc | +0.002 (0.006) | 2/9 | 0/9 | 0/9 | 0/9 | +0.001 (0.002) | 0/9 | 0/9 | 0/9 | 0/9 |
| Evidential | R3+ − (H,u,b_max) | b_(2) given b_max | nested, 1-D | revision, post hoc | +0.001 (0.001) | 0/9 | 0/9 | 0/9 | 0/9 | +0.000 (0.000) | 0/9 | 0/9 | 0/9 | 0/9 |
| Evidential | (H,u,b_(2)) − R2 | b_(2) | nested, 1-D | revision, post hoc | +0.002 (0.005) | 1/9 | 0/9 | 0/9 | 0/9 | +0.001 (0.002) | 0/9 | 0/9 | 0/9 | 0/9 |
| Evidential | BF − R2 | binary-frame triple (b_max, 1−u−b_max, u) vs (H,u) | head-to-head | revision, post hoc | -0.020 (0.028) | 0/9 | 0/9 | 0/9 | 0/9 | +0.001 (0.002) | 0/9 | 0/9 | 0/9 | 0/9 |
| Evidential | BF − (b_max,u) | b_other = 1−u−b_max | nested, affine-dependent | revision, post hoc | -0.000 (0.000) | 0/9 | 0/9 | 0/9 | 0/9 | +0.000 (0.000) | 0/9 | 0/9 | 0/9 | 0/9 |
| NS-U | N2base+T − N2base | T | nested, 1-D | revision, post hoc | +0.121 (0.075) | 9/9 | 9/9 | 8/9 | 7/9 | +0.062 (0.063) | 6/9 | 6/9 | 6/9 | 5/9 |
| NS-U | N2+ − (N2base+T) | F given T | nested, 1-D | revision, post hoc | +0.009 (0.062) | 2/9 | 1/9 | 1/9 | 1/9 | +0.010 (0.017) | 3/9 | 2/9 | 2/9 | 0/9 |
| NS-U | N2base+F − N2base | F | nested, 1-D | revision, post hoc | +0.117 (0.079) | 9/9 | 9/9 | 7/9 | 7/9 | +0.061 (0.065) | 6/9 | 6/9 | 6/9 | 5/9 |

HybridSN (secondary): 3, 0, 0, 0 of 30 at the four thresholds.

**B8.** AUC, AUPRC (errors = positive class) and AURC (area under the risk–coverage curve; pixels accepted in order of
increasing detector score), for every model, rung and run: `c15_auprc_aurc_by_run.csv`. Mean ± SD over runs:
`c15_auprc_aurc_summary.csv`. Risk–coverage curves for the rung with the lowest mean AURC of each model and ladder,
averaged over runs: `c15_risk_coverage_pavia_u.png`, `c15_risk_coverage_salinas.png` (300 dpi) and
`c15_risk_coverage_curves.csv`. The choice of the plotted rung was made after seeing the values.

Selected means (AUC / AUPRC / AURC):

| Scene | Model | Rung | AUC | AUPRC | AURC |
|---|---|---|---|---|---|
| Pavia | softmax | S1 / S2 / S3 | 0.768 / 0.777 / 0.775 | 0.347 / 0.350 / 0.349 | 0.0542 / 0.0531 / 0.0540 |
| Pavia | MC dropout | M1 / M2 / M3 | 0.772 / 0.800 / 0.811 | 0.350 / 0.431 / 0.422 | 0.0564 / 0.0528 / 0.0470 |
| Pavia | ensemble | E1 / E2 / E3 | 0.865 / 0.900 / 0.892 | 0.374 / 0.566 / 0.549 | 0.0175 / 0.0136 / 0.0146 |
| Pavia | evidential | R1 / R2 / R3+ / R3 | 0.780 / 0.801 / 0.804 / 0.782 | 0.321 / 0.437 / 0.450 / 0.403 | 0.0495 / 0.0498 / 0.0505 / 0.0533 |
| Pavia | NS-U | N1 / N2 / N2base / N2+ | 0.850 / 0.859 / 0.731 / 0.861 | 0.493 / 0.552 / 0.412 / 0.550 | 0.0425 / 0.0440 / 0.0755 / 0.0436 |
| Salinas | softmax | S1 / S2 / S3 | 0.814 / 0.814 / 0.813 | 0.357 / 0.361 / 0.362 | 0.0334 / 0.0335 / 0.0341 |
| Salinas | MC dropout | M1 / M2 / M3 | 0.807 / 0.807 / 0.809 | 0.349 / 0.349 / 0.363 | 0.0344 / 0.0345 / 0.0341 |
| Salinas | ensemble | E1 / E2 / E3 | 0.882 / 0.882 / 0.875 | 0.266 / 0.279 / 0.291 | 0.0074 / 0.0074 / 0.0083 |
| Salinas | evidential | R1 / R2 / R3+ / R3 | 0.848 / 0.848 / 0.849 / 0.850 | 0.588 / 0.605 / 0.609 / 0.611 | 0.0372 / 0.0369 / 0.0371 / 0.0371 |
| Salinas | NS-U | N1 / N2 / N2base / N2+ | 0.837 / 0.849 / 0.778 / 0.850 | 0.432 / 0.445 / 0.417 / 0.458 | 0.0323 / 0.0274 / 0.0447 / 0.0266 |

**Summary.** The number of Table 4 comparisons that pass falls from 39 to 19 as the threshold rises from 0.01 to 0.05.
At every threshold, NS-U N2+ − N2base is the row with the most passing runs. AUPRC and AURC refer to each model's own
errors, so their levels are not directly comparable between models.

## Comment 7: size of the residual T + F − 1 in NS-U (B1)

**What was run.** `revision/b1_residuals.py`, on the saved test outputs of the 18 NS-U runs, at the predicted class
k* = argmax(T − F). Per run: `c07_residuals_by_run.csv`; all points: `c07_points_F_vs_1minusT.csv`. Figures (300 dpi):
`c07_F_vs_1minusT_by_scene.png`, `c07_F_vs_1minusT_<scene>_by_seed.png` (split × training seed),
`c07_F_vs_1minusT_<scene>_by_class.png` (by predicted class).

| Scene | mean abs res | median abs res | SD abs res | p05 | p95 | p99 | share abs res > 0.05 | share T+F > 1 |
|---|---|---|---|---|---|---|---|---|
| Pavia (range over 9 runs) | 0.0023–0.0064 | 0.00012–0.00045 | 0.0081–0.0196 | −0.0375 to −0.0028 | +0.0017 to +0.0318 | +0.0120 to +0.1117 | 0.0053–0.0403 | 0.278–0.814 |
| Salinas (range over 9 runs) | 0.0024–0.0064 | 0.00023–0.00047 | 0.0061–0.0206 | −0.0209 to −0.0015 | +0.0030 to +0.0115 | +0.0179 to +0.1346 | 0.0021–0.0428 | 0.351–0.636 |

**Summary.** Across the 18 runs, the median |T + F − 1| at the predicted class is 1.2e-4 to 4.7e-4, and its mean is
0.0022 to 0.0064. Between 0.2% and 4.3% of test pixels have |T + F − 1| > 0.05. The share with T + F > 1 ranges from
0.28 to 0.81 and depends on the run.

**Deviations.** Only test pixels: training-pixel outputs were not saved in q1. Training-pixel residuals for the same
networks are in C1(a). Precision: this table uses float64. The q1 share T + F > 1 was computed in float32, where the
sigmoid saturates to 1. Both are reported; the float32 column (`share_TplusF_gt_1_float32_as_in_q1`) equals the q1 values
exactly; the float64 shares differ by at most 0.0021.

## Comment 9: detector sensitivity: regularisation and rank (B5)

**(a)** Every ladder step was refit with C ∈ {0.01, 0.1, 1, 10, 100} and unpenalised (C = ∞), using the same folds and
resamples. Per comparison: `c09_C_sensitivity.csv`, with ΔAUC, CI, verdict and convergence warnings for every C, and a
flag when the verdict changes.

Totals over the 166 Table 4 comparisons (runs passing rule (10)):

| C | 0.01 | 0.1 | 1 (original) | 10 | 100 | unpenalised |
|---|---|---|---|---|---|---|
| Pavia | 16/83 | 19/83 | 19/83 | 19/83 | 19/83 | 19/83 |
| Salinas | 6/83 | 8/83 | 9/83 | 9/83 | 9/83 | 9/83 |

Cell: mean ΔAUC (SD over runs); runs passing rule (10). Last column per scene: runs whose verdict changes across C.

| Model | Step | Adds | Type | Row source | Pavia University: C = 0.01 | Pavia University: C = 0.1 | Pavia University: C = 1 (original) | Pavia University: C = 10 | Pavia University: C = 100 | Pavia University: unpenalised | Pavia University: runs with verdict change | Salinas: C = 0.01 | Salinas: C = 0.1 | Salinas: C = 1 (original) | Salinas: C = 10 | Salinas: C = 100 | Salinas: unpenalised | Salinas: runs with verdict change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Softmax | S2 − S1 | p_max | nested | Table 4 | +0.007 (0.010); 0/9 | +0.008 (0.008); 0/9 | +0.008 (0.009); 0/9 | +0.008 (0.009); 0/9 | +0.008 (0.009); 0/9 | +0.008 (0.009); 0/9 | 0/9 | +0.000 (0.001); 0/9 | +0.000 (0.003); 0/9 | +0.000 (0.002); 0/9 | +0.000 (0.002); 0/9 | +0.000 (0.002); 0/9 | +0.000 (0.002); 0/9 | 0/9 |
| Softmax | S3 − S2 | p_(2) | nested | Table 4 | +0.002 (0.003); 0/9 | +0.000 (0.006); 0/9 | -0.002 (0.007); 0/9 | -0.003 (0.008); 0/9 | -0.003 (0.008); 0/9 | -0.003 (0.008); 0/9 | 0/9 | +0.001 (0.002); 0/9 | +0.000 (0.003); 0/9 | -0.000 (0.003); 0/9 | -0.000 (0.004); 0/9 | -0.000 (0.004); 0/9 | -0.000 (0.004); 0/9 | 0/9 |
| MC dropout | M2 − M1 | MI | nested | Table 4 | +0.025 (0.025); 4/9 | +0.028 (0.027); 6/9 | +0.029 (0.027); 6/9 | +0.029 (0.027); 6/9 | +0.029 (0.027); 6/9 | +0.029 (0.027); 6/9 | 2/9 | -0.001 (0.001); 0/9 | -0.000 (0.001); 0/9 | -0.000 (0.001); 0/9 | -0.000 (0.001); 0/9 | -0.000 (0.001); 0/9 | -0.000 (0.001); 0/9 | 0/9 |
| MC dropout | M3 − M2 | p_max | nested | Table 4 | +0.002 (0.015); 1/9 | +0.010 (0.048); 1/9 | +0.011 (0.053); 1/9 | +0.011 (0.053); 1/9 | +0.011 (0.053); 1/9 | +0.011 (0.053); 1/9 | 0/9 | +0.002 (0.004); 0/9 | +0.002 (0.004); 0/9 | +0.002 (0.004); 0/9 | +0.002 (0.004); 0/9 | +0.002 (0.004); 0/9 | +0.002 (0.004); 0/9 | 0/9 |
| Ensemble | E2 − E1 | MI | nested | Table 4 | +0.029; 0/1 | +0.033; 1/1 | +0.035; 1/1 | +0.035; 1/1 | +0.035; 1/1 | +0.035; 1/1 | 1/1 | -0.001; 0/1 | -0.000; 0/1 | +0.000; 0/1 | +0.000; 0/1 | +0.000; 0/1 | +0.000; 0/1 | 0/1 |
| Ensemble | E3 − E2 | p_max | nested | Table 4 | -0.002; 0/1 | -0.006; 0/1 | -0.008; 0/1 | -0.008; 0/1 | -0.008; 0/1 | -0.008; 0/1 | 0/1 | -0.007; 0/1 | -0.007; 0/1 | -0.007; 0/1 | -0.007; 0/1 | -0.007; 0/1 | -0.007; 0/1 | 0/1 |
| Evidential | R2 − R1 | u | nested | Table 4 | +0.023 (0.035); 2/9 | +0.022 (0.028); 2/9 | +0.021 (0.025); 2/9 | +0.021 (0.024); 2/9 | +0.021 (0.024); 2/9 | +0.021 (0.024); 2/9 | 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | 0/9 |
| Evidential | R3+ − R2 | (b_max, b_(2)) | nested, 2-D (combined increment) | Table 4 | -0.001 (0.007); 0/9 | +0.001 (0.003); 0/9 | +0.003 (0.005); 0/9 | +0.004 (0.005); 0/9 | +0.005 (0.006); 0/9 | +0.005 (0.006); 0/9 | 0/9 | +0.001 (0.001); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | 0/9 |
| Evidential | R3 − R2 | triple vs pair | head-to-head | Table 4 | -0.019 (0.032); 0/9 | -0.019 (0.029); 0/9 | -0.019 (0.027); 0/9 | -0.019 (0.028); 0/9 | -0.019 (0.028); 0/9 | -0.019 (0.028); 0/9 | 0/9 | +0.001 (0.001); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | 0/9 |
| NS-U | N2 − N1 | F given T − F | nested | Table 4 | +0.005 (0.013); 1/9 | +0.018 (0.045); 1/9 | +0.010 (0.060); 1/9 | +0.008 (0.063); 1/9 | +0.008 (0.063); 1/9 | +0.008 (0.064); 1/9 | 0/9 | +0.001 (0.003); 0/9 | +0.008 (0.013); 1/9 | +0.012 (0.017); 2/9 | +0.011 (0.018); 2/9 | +0.011 (0.018); 2/9 | +0.011 (0.018); 2/9 | 2/9 |
| NS-U | N2+ − N2base | (T, F) | nested, 2-D (combined increment) | Table 4 | +0.127 (0.072); 8/9 | +0.139 (0.075); 8/9 | +0.130 (0.076); 8/9 | +0.128 (0.078); 8/9 | +0.128 (0.078); 8/9 | +0.128 (0.078); 8/9 | 0/9 | +0.066 (0.066); 6/9 | +0.071 (0.058); 7/9 | +0.072 (0.057); 7/9 | +0.070 (0.055); 7/9 | +0.069 (0.055); 7/9 | +0.069 (0.055); 7/9 | 1/9 |
| Evidential | (H,u,b_max) − R2 | b_max | nested, 1-D | revision, post hoc | +0.001 (0.001); 0/9 | +0.001 (0.003); 0/9 | +0.002 (0.006); 0/9 | +0.002 (0.006); 0/9 | +0.002 (0.006); 0/9 | +0.002 (0.006); 0/9 | 0/9 | +0.001 (0.001); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | 0/9 |
| Evidential | R3+ − (H,u,b_max) | b_(2) given b_max | nested, 1-D | revision, post hoc | -0.002 (0.007); 0/9 | +0.000 (0.000); 0/9 | +0.001 (0.001); 0/9 | +0.002 (0.003); 0/9 | +0.002 (0.004); 0/9 | +0.003 (0.004); 0/9 | 0/9 | +0.000 (0.001); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | 0/9 |
| Evidential | (H,u,b_(2)) − R2 | b_(2) | nested, 1-D | revision, post hoc | -0.001 (0.007); 0/9 | +0.001 (0.003); 0/9 | +0.002 (0.005); 0/9 | +0.002 (0.006); 0/9 | +0.002 (0.006); 0/9 | +0.002 (0.006); 0/9 | 0/9 | +0.001 (0.001); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | 0/9 |
| Evidential | BF − R2 | binary-frame triple (b_max, 1−u−b_max, u) vs (H,u) | head-to-head | revision, post hoc | -0.020 (0.035); 0/9 | -0.020 (0.031); 0/9 | -0.020 (0.028); 0/9 | -0.019 (0.028); 0/9 | -0.019 (0.028); 0/9 | -0.019 (0.028); 0/9 | 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | 0/9 |
| Evidential | BF − (b_max,u) | b_other = 1−u−b_max | nested, affine-dependent | revision, post hoc | -0.009 (0.024); 0/9 | -0.000 (0.001); 0/9 | -0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 | 0/9 | +0.000 (0.001); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | 0/9 |
| NS-U | N2base+T − N2base | T | nested, 1-D | revision, post hoc | +0.122 (0.075); 8/9 | +0.122 (0.075); 8/9 | +0.121 (0.075); 8/9 | +0.121 (0.075); 8/9 | +0.121 (0.075); 8/9 | +0.121 (0.075); 8/9 | 0/9 | +0.066 (0.064); 6/9 | +0.062 (0.063); 6/9 | +0.062 (0.063); 6/9 | +0.062 (0.063); 6/9 | +0.061 (0.063); 6/9 | +0.061 (0.062); 6/9 | 0/9 |
| NS-U | N2+ − (N2base+T) | F given T | nested, 1-D | revision, post hoc | +0.005 (0.013); 1/9 | +0.018 (0.046); 1/9 | +0.009 (0.062); 1/9 | +0.007 (0.066); 1/9 | +0.007 (0.067); 1/9 | +0.007 (0.067); 1/9 | 0/9 | +0.000 (0.003); 0/9 | +0.009 (0.013); 2/9 | +0.010 (0.017); 2/9 | +0.008 (0.018); 2/9 | +0.008 (0.018); 2/9 | +0.008 (0.018); 2/9 | 2/9 |
| NS-U | N2base+F − N2base | F | nested, 1-D | revision, post hoc | +0.118 (0.079); 8/9 | +0.118 (0.079); 7/9 | +0.117 (0.079); 7/9 | +0.117 (0.079); 7/9 | +0.117 (0.078); 7/9 | +0.117 (0.079); 7/9 | 1/9 | +0.065 (0.067); 6/9 | +0.062 (0.065); 6/9 | +0.061 (0.065); 6/9 | +0.061 (0.065); 6/9 | +0.061 (0.065); 6/9 | +0.061 (0.064); 6/9 | 0/9 |

**(b)** Numerical rank of every standardised design matrix: `c09_design_rank.csv`.
- All Table 4 rungs have full column rank.
- Rank < number of columns occurs only for the degenerate evidential HybridSN (R2_b2, R3, R3+), where coordinates are
  constant or duplicated.
- The binary-frame triple (b_max, b_other = 1 − u − b_max, u) is affine-dependent by construction. With the stored
  float32 logits it is numerically full rank (condition number about 1e5), because the identity holds only to float32
  rounding. Its rank-reduced version is exactly (b_max, u), and the "BF − (b_max,u)" row shows ΔAUC ≈ 0 at every C ≥ 0.1.

**Summary.** The verdict changes in 6 of the 166 Table 4 comparisons, and only for C ≤ 0.1:
- Pavia MC dropout M2 − M1 (2 runs);
- Pavia ensemble E2 − E1 (1);
- Salinas NS-U N2 − N1 (2);
- Salinas NS-U N2+ − N2base (1).

The verdicts are identical for C = 1, 10, 100 and the unpenalised fit.

## Comments 10, 11: single-coordinate rungs (B6)

Below the 11 Table 4 rows, in every revised table (B2, B3, B4, B5, B7 and `t4_table4_original.csv`). Per run: `c10_11_single_coordinate_steps.csv`; rung AUCs: `c10_evidential_binary_frame_aucs.csv`; sums b_max + b₂ + u:
`engine_evid_sums.csv`. Mean over runs (SD), and the number of runs passing the rule:

| Scene | Model | Step | Mean Δ | SD | Min | Max | Passes |
|---|---|---|---|---|---|---|---|
| Pavia | evidential | (H,u,b_max) − (H,u) | +0.0022 | 0.0055 | −0.0032 | +0.0104 | 0/9 |
| Pavia | evidential | R3+ − (H,u,b_max) | +0.0006 | 0.0011 | −0.0014 | +0.0024 | 0/9 |
| Pavia | evidential | (H,u,b₂) − (H,u) | +0.0023 | 0.0053 | −0.0029 | +0.0103 | 0/9 |
| Pavia | evidential | binary-frame triple − (H,u) | −0.0195 | 0.0280 | −0.0767 | −0.0000 | 0/9 |
| Pavia | evidential | binary-frame triple − (b_max,u) | −0.0001 | 0.0001 | −0.0003 | +0.0000 | 0/9 |
| Salinas | evidential | (H,u,b_max) − (H,u) | +0.0009 | 0.0024 | −0.0017 | +0.0065 | 0/9 |
| Salinas | evidential | R3+ − (H,u,b_max) | +0.0001 | 0.0001 | −0.0001 | +0.0004 | 0/9 |
| Salinas | evidential | (H,u,b₂) − (H,u) | +0.0010 | 0.0024 | −0.0017 | +0.0066 | 0/9 |
| Salinas | evidential | binary-frame triple − (H,u) | +0.0010 | 0.0025 | −0.0019 | +0.0066 | 0/9 |
| Salinas | evidential | binary-frame triple − (b_max,u) | +0.0000 | 0.0001 | −0.0000 | +0.0001 | 0/9 |
| Pavia | NS-U | N2base+T − N2base | +0.1209 | 0.0748 | +0.0293 | +0.2558 | 8/9 |
| Pavia | NS-U | N2+ − (N2base+T) | +0.0094 | 0.0618 | −0.0436 | +0.1655 | 1/9 |
| Pavia | NS-U | N2base+F − N2base | +0.1170 | 0.0785 | +0.0201 | +0.2587 | 7/9 |
| Salinas | NS-U | N2base+T − N2base | +0.0618 | 0.0629 | −0.0246 | +0.1389 | 6/9 |
| Salinas | NS-U | N2+ − (N2base+T) | +0.0105 | 0.0166 | −0.0062 | +0.0419 | 2/9 |
| Salinas | NS-U | N2base+F − N2base | +0.0614 | 0.0649 | −0.0201 | +0.1361 | 6/9 |

Top-two truncation, b_max + b₂ + u per run: Pavia evidential, mean of run means 0.9997, minimum over pixels and runs
0.8075; Salinas evidential 0.9999, minimum 0.8623; HybridSN evidential 1.0000, minimum 0.99994 (Pavia) and 1.0 (Salinas).
The same steps for HybridSN evidential are in the CSV (no step passes).

**Summary.**
- **Evidential.** No single-coordinate step passes in either scene. The binary-frame triple has the same AUC as
  (b_max, u) to within 0.0003 in every run.
- **NS-U.** Adding T alone or F alone to N2base gives most of the N2+ − N2base gain (Pavia +0.121 / +0.117 mean). Adding
  the second coordinate afterwards passes in 1/9 (Pavia) and 2/9 (Salinas) runs.

## Comment 12: spatial robustness of the detector and the CI (B4)

What was run:
- the detector refit with GroupKFold(5) by the original 16×16 blocks, and by 32×32 blocks (feasible in every run, ≥ 5 groups);
- CIs from a spatial block bootstrap that resamples test blocks with replacement (B = 3000, seed 1337);
- the original out-of-fold scores with the block bootstrap only.

Per comparison: `c12_blocked_vs_original.csv` (196 comparisons) and `engine_steps_b4_b6.csv` (B6 rows). Pooled rows:
`c12_blocked_vs_original_pooled.csv`. Fold diagnostic: `c12_fold_auc_diagnostic.csv`.

Totals over the 166 Table 4 comparisons (runs passing rule (10)):

| Variant | Pavia | Salinas | Total |
|---|---|---|---|
| Original (pixel CV, pixel bootstrap) | 19/83 | 9/83 | 28/166 |
| Original OOF + 16×16 block bootstrap | 18/83 | 8/83 | 26/166 |
| 16×16 block CV + block bootstrap | 17/83 | 9/83 | 26/166 |
| 32×32 block CV + block bootstrap | 16/83 | 9/83 | 25/166 |

Cell: mean ΔAUC (SD over runs); runs passing rule (10). GroupKFold(5) by blocks; bootstrap resamples test blocks with replacement (B = 3000, seed 1337).

| Model | Step | Adds | Type | Row source | Pavia University: original (pixel CV, pixel bootstrap) | Pavia University: original OOF + 16×16 block bootstrap | Pavia University: 16×16 block CV + block bootstrap | Pavia University: 32×32 block CV + block bootstrap | Salinas: original (pixel CV, pixel bootstrap) | Salinas: original OOF + 16×16 block bootstrap | Salinas: 16×16 block CV + block bootstrap | Salinas: 32×32 block CV + block bootstrap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Softmax | S2 − S1 | p_max | nested | Table 4 | +0.008 (0.009); 0/9 | +0.008 (0.009); 0/9 | +0.015 (0.015); 1/9 | +0.013 (0.018); 2/9 | +0.000 (0.002); 0/9 | +0.000 (0.002); 0/9 | -0.003 (0.005); 0/9 | -0.004 (0.012); 0/9 |
| Softmax | S3 − S2 | p_(2) | nested | Table 4 | -0.002 (0.007); 0/9 | -0.002 (0.006); 0/9 | +0.006 (0.013); 0/9 | +0.012 (0.004); 0/9 | -0.000 (0.003); 0/9 | -0.001 (0.003); 0/9 | +0.013 (0.007); 0/9 | +0.017 (0.018); 2/9 |
| MC dropout | M2 − M1 | MI | nested | Table 4 | +0.029 (0.027); 6/9 | +0.028 (0.027); 5/9 | +0.023 (0.032); 4/9 | +0.018 (0.034); 2/9 | -0.000 (0.001); 0/9 | -0.000 (0.001); 0/9 | -0.002 (0.003); 0/9 | -0.004 (0.006); 0/9 |
| MC dropout | M3 − M2 | p_max | nested | Table 4 | +0.011 (0.053); 1/9 | +0.011 (0.052); 1/9 | +0.012 (0.022); 1/9 | +0.016 (0.017); 1/9 | +0.002 (0.004); 0/9 | +0.002 (0.004); 0/9 | -0.008 (0.005); 0/9 | -0.011 (0.009); 0/9 |
| Ensemble | E2 − E1 | MI | nested | Table 4 | +0.035; 1/1 | +0.034; 1/1 | +0.024; 0/1 | +0.032; 1/1 | +0.000; 0/1 | +0.000; 0/1 | -0.001; 0/1 | -0.001; 0/1 |
| Ensemble | E3 − E2 | p_max | nested | Table 4 | -0.008; 0/1 | -0.008; 0/1 | +0.004; 0/1 | +0.005; 0/1 | -0.007; 0/1 | -0.007; 0/1 | -0.015; 0/1 | -0.010; 0/1 |
| Evidential | R2 − R1 | u | nested | Table 4 | +0.021 (0.025); 2/9 | +0.021 (0.024); 2/9 | +0.015 (0.023); 1/9 | +0.015 (0.024); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.002); 0/9 | -0.002 (0.003); 0/9 | +0.001 (0.005); 0/9 |
| Evidential | R3+ − R2 | (b_max, b_(2)) | nested, 2-D (combined increment) | Table 4 | +0.003 (0.005); 0/9 | +0.003 (0.005); 0/9 | +0.007 (0.008); 0/9 | +0.016 (0.017); 2/9 | +0.001 (0.003); 0/9 | +0.001 (0.003); 0/9 | -0.009 (0.015); 0/9 | -0.012 (0.021); 0/9 |
| Evidential | R3 − R2 | triple vs pair | head-to-head | Table 4 | -0.019 (0.027); 0/9 | -0.019 (0.028); 0/9 | -0.028 (0.023); 0/9 | -0.041 (0.016); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | -0.007 (0.010); 0/9 | -0.012 (0.019); 0/9 |
| NS-U | N2 − N1 | F given T − F | nested | Table 4 | +0.010 (0.060); 1/9 | +0.009 (0.060); 1/9 | +0.004 (0.026); 1/9 | -0.006 (0.024); 0/9 | +0.012 (0.017); 2/9 | +0.012 (0.017); 2/9 | +0.000 (0.013); 0/9 | -0.001 (0.018); 0/9 |
| NS-U | N2+ − N2base | (T, F) | nested, 2-D (combined increment) | Table 4 | +0.130 (0.076); 8/9 | +0.129 (0.076); 8/9 | +0.149 (0.085); 9/9 | +0.139 (0.097); 8/9 | +0.072 (0.057); 7/9 | +0.071 (0.056); 6/9 | +0.109 (0.055); 9/9 | +0.092 (0.064); 7/9 |
| Evidential | (H,u,b_max) − R2 | b_max | nested, 1-D | revision, post hoc | +0.002 (0.006); 0/9 | +0.002 (0.006); 0/9 | +0.006 (0.009); 0/9 | +0.016 (0.017); 2/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | -0.009 (0.015); 0/9 | -0.012 (0.021); 0/9 |
| Evidential | R3+ − (H,u,b_max) | b_(2) given b_max | nested, 1-D | revision, post hoc | +0.001 (0.001); 0/9 | +0.001 (0.001); 0/9 | +0.000 (0.002); 0/9 | +0.000 (0.002); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | -0.000 (0.001); 0/9 |
| Evidential | (H,u,b_(2)) − R2 | b_(2) | nested, 1-D | revision, post hoc | +0.002 (0.005); 0/9 | +0.002 (0.005); 0/9 | +0.007 (0.009); 0/9 | +0.016 (0.017); 2/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | -0.009 (0.015); 0/9 | -0.012 (0.021); 0/9 |
| Evidential | BF − R2 | binary-frame triple (b_max, 1−u−b_max, u) vs (H,u) | head-to-head | revision, post hoc | -0.020 (0.028); 0/9 | -0.020 (0.028); 0/9 | -0.028 (0.023); 0/9 | -0.041 (0.017); 0/9 | +0.001 (0.002); 0/9 | +0.001 (0.003); 0/9 | -0.008 (0.010); 0/9 | -0.012 (0.019); 0/9 |
| Evidential | BF − (b_max,u) | b_other = 1−u−b_max | nested, affine-dependent | revision, post hoc | -0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 | -0.000 (0.000); 0/9 |
| NS-U | N2base+T − N2base | T | nested, 1-D | revision, post hoc | +0.121 (0.075); 8/9 | +0.121 (0.074); 8/9 | +0.144 (0.094); 8/9 | +0.146 (0.105); 9/9 | +0.062 (0.063); 6/9 | +0.061 (0.062); 6/9 | +0.107 (0.054); 9/9 | +0.094 (0.054); 9/9 |
| NS-U | N2+ − (N2base+T) | F given T | nested, 1-D | revision, post hoc | +0.009 (0.062); 1/9 | +0.009 (0.062); 1/9 | +0.005 (0.030); 1/9 | -0.006 (0.028); 0/9 | +0.010 (0.017); 2/9 | +0.010 (0.016); 1/9 | +0.002 (0.017); 0/9 | -0.002 (0.019); 0/9 |
| NS-U | N2base+F − N2base | F | nested, 1-D | revision, post hoc | +0.117 (0.079); 7/9 | +0.117 (0.078); 7/9 | +0.140 (0.098); 8/9 | +0.142 (0.108); 8/9 | +0.061 (0.065); 6/9 | +0.061 (0.064); 6/9 | +0.108 (0.057); 9/9 | +0.094 (0.054); 7/9 |

Pooled rows (28; groups = run × block): the 2 pooled passes are NS-U N2+ − N2base in every variant. Pavia: +0.081
original, +0.102 block 16, +0.117 block 32. Salinas: +0.051, +0.088, +0.091. HybridSN: 0 passes in every variant.

**Summary.** With blocked folds and a block bootstrap, 25–26 of the 166 Table 4 comparisons pass, against 28
originally. NS-U N2+ − N2base passes in 8/9 or 9/9 Pavia runs in every variant, and in 6/9 to 9/9 Salinas runs.

**Note.** Under GroupKFold the pooled out-of-fold AUC of a rung can differ from the mean within-fold AUC, because scores
from differently calibrated fold models are mixed. Both are given in `c12_fold_auc_diagnostic.csv`.

## Comment 16: evidential diagnostics (B10; C2 below)

Per run: `c16_evidential_evidence_by_run.csv`; by class: `c16_evidential_evidence_by_class.csv`; per-class accuracy
against softmax of the same split and seed: `c16_perclass_accuracy_evidential_vs_softmax.csv`. Calibration: comment 17.

| Model, scene | Mean S−K (range over runs) | Median S−K | Share with evidence > 0.01 on ≤ 1 class | Share with total evidence < 0.01 | Accuracy |
|---|---|---|---|---|---|
| HSINet evidential, Pavia (9) | 30.5–34.6 | 30.7–36.7 | 0.727–0.859 | 0.0000 (all runs) | 0.821–0.928 |
| HSINet evidential, Salinas (9) | 38.1–42.9 | 40.7–45.9 | 0.766–0.889 | 0.0000–0.0814 | 0.780–0.932 |
| HybridSN evidential, Pavia (3) | 291.6–367.5 | 211.9–295.4 | 0.987–0.998 | 0.081–0.206 | 0.673–0.807 |
| HybridSN evidential, Salinas (3) | 302.6–414.7 | 0.0–342.1 | 0.9999–1.0000 | 0.178–0.626 | 0.418–0.738 |

**Summary.** In HybridSN evidential, 8–63% of test pixels carry total evidence below 0.01, and almost every pixel has
evidence on at most one class. In Salinas split 0, the median S−K is 0 and accuracy is 0.418. HSINet evidential has no
pixel with total evidence < 0.01 on Pavia, and up to 8.1% on Salinas.

## Comment 17: calibration (B9; shift and OOD: C3 below)

Per run: `c17_calibration_by_run.csv`; summary: `c17_calibration_summary.csv`. NLL (natural log, p clipped at 1e-12),
Brier (sum over classes), ECE (15 equal-width bins, top label), mean ± SD over runs:

| Scene | Model | Probabilities | Runs | Accuracy | NLL | Brier | ECE |
|---|---|---|---|---|---|---|---|
| Pavia | softmax | p | 9 | 0.870 ± 0.044 | 1.017 ± 0.301 | 0.239 ± 0.085 | 0.108 ± 0.044 |
| Pavia | MC dropout | mean p | 9 | 0.870 ± 0.044 | 0.943 ± 0.287 | 0.236 ± 0.086 | 0.104 ± 0.045 |
| Pavia | ensemble | mean p | 1 | 0.927 | 0.773 | 0.130 | 0.051 |
| Pavia | evidential | α/S | 9 | 0.869 ± 0.046 | 0.629 ± 0.145 | 0.234 ± 0.069 | 0.096 ± 0.037 |
| Pavia | NS-U | softmax(zT − zF) | 9 | 0.873 ± 0.043 | 2.240 ± 1.069 | 0.245 ± 0.088 | 0.120 ± 0.045 |
| Pavia | NS-U | T/ΣT | 9 | 0.873 ± 0.043 | 0.792 ± 0.214 | 0.229 ± 0.088 | 0.100 ± 0.048 |
| Pavia | HybridSN softmax | p | 3 | 0.845 ± 0.041 | 2.806 ± 1.211 | 0.298 ± 0.084 | 0.145 ± 0.042 |
| Pavia | HybridSN evidential | α/S | 3 | 0.736 ± 0.077 | 1.082 ± 0.240 | 0.379 ± 0.096 | 0.109 ± 0.041 |
| Salinas | softmax | p | 9 | 0.910 ± 0.054 | 0.581 ± 0.399 | 0.156 ± 0.093 | 0.065 ± 0.043 |
| Salinas | MC dropout | mean p | 9 | 0.910 ± 0.054 | 0.535 ± 0.372 | 0.153 ± 0.091 | 0.060 ± 0.041 |
| Salinas | ensemble | mean p | 1 | 0.959 | 0.189 | 0.067 | 0.019 |
| Salinas | evidential | α/S | 9 | 0.871 ± 0.053 | 0.697 ± 0.134 | 0.230 ± 0.051 | 0.197 ± 0.043 |
| Salinas | NS-U | softmax(zT − zF) | 9 | 0.904 ± 0.060 | 1.191 ± 0.690 | 0.177 ± 0.106 | 0.085 ± 0.050 |
| Salinas | NS-U | T/ΣT | 9 | 0.904 ± 0.060 | 0.453 ± 0.278 | 0.154 ± 0.090 | 0.059 ± 0.039 |
| Salinas | HybridSN softmax | p | 3 | 0.864 ± 0.051 | 1.885 ± 1.082 | 0.256 ± 0.098 | 0.124 ± 0.048 |
| Salinas | HybridSN evidential | α/S | 3 | 0.596 ± 0.155 | 1.420 ± 0.325 | 0.475 ± 0.112 | 0.058 ± 0.033 |

"Accuracy" is the accuracy of the argmax of the listed probability vector. For NS-U, argmax T/ΣT and argmax softmax(zT−zF)
can differ from the prediction argmax(T − F) used in the ladders.

**Summary.** On both scenes the ensemble has the lowest NLL, Brier score and ECE. For NS-U, the renormalised T/ΣT has
lower NLL than softmax(zT − zF) in both scenes (0.79 vs 2.24 Pavia; 0.45 vs 1.19 Salinas).

## Comment 19: exact pair-level formulas for the preference data (A4)

Scorer `Qwen/Qwen2.5-1.5B-Instruct` (fp16 weights on CUDA, logits cast to float32), the same for every rung. For every
pair, the prompt and **each** response are scored separately by teacher forcing (`ns_logprob.py`):

```python
# ns_logprob.py, lines 63–66: prompt with default special tokens (Qwen adds no BOS: tok("hello") -> [14990]),
# truncated to the first 512 tokens; response without special tokens, truncated to its first 512 tokens
p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT).input_ids
a_ids = tok(answer, return_tensors="pt", truncation=True, max_length=MAX_ANSWER, add_special_tokens=False).input_ids
if a_ids.shape[1] < 2: return None          # line 67–68: the pair is skipped (exclusion rule)
# lines 70–80: one forward pass over [prompt; response]; only RESPONSE positions are used
ids = torch.cat([p_ids, a_ids], dim=1); logits = model(ids).logits[0].float()
start = p_ids.shape[1]; tgt = ids[0, start:]; lg = logits[start - 1:-1]
logp = F_.log_softmax(lg, dim=-1)
tok_lp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)    # ln p(response token t | prefix)
ent = -(logp.exp() * logp).sum(dim=-1)                  # entropy of next-token distribution, full vocabulary, nats
# lines 83, 85: per-RESPONSE means over its tokens
"mean_lp": float(tok_lp.mean()), "mean_ent": float(ent.mean())
```

```python
# ns_logprob_detect.py, lines 80–85, 87–90
T = np.array([r["chosen"]["mean_lp"] for r in rows])      # response presented as chosen
F = np.array([r["rejected"]["mean_lp"] for r in rows])    # response presented as rejected
I = np.array([(r["chosen"]["mean_ent"] + r["rejected"]["mean_ent"]) / 2 for r in rows])
leak = np.array([r["len_chosen"] - r["len_rejected"] for r in rows], dtype=float)   # characters
feats = {"margin": (T - F), "T, F": np.c_[T, F], "T, F, I": np.c_[T, F, I]}
# lines 93–95 (length-covariate control): L = [leak, |leak|] appended to EVERY rung
```

Pair-level formulas, for presented chosen c and presented rejected r, with response tokens a₁..a_n (prompt tokens
excluded):

- T = (1/n_c) Σ_t ln p(a_t^c | prompt, a_<t^c);
- F = (1/n_r) Σ_t ln p(a_t^r | prompt, a_<t^r);
- I = ½ [ (1/n_c) Σ_t H_t^c + (1/n_r) Σ_t H_t^r ], where H_t is the entropy over the full vocabulary of the next-token
  distribution at response position t (nats).

**Details.**
- I is computed on both responses, as a per-token mean within each response; the two response means are then averaged
  with equal weight, not token-weighted.
- There is no padding: batch size is 1. No chat template is used: raw prompt text is followed directly by raw response
  text. Prompt positions are excluded.
- No sigmoid is applied. The module docstring of `ns_logprob.py` mentions sigmoid(mean logP − c), but the detector uses
  raw mean log-probabilities.
- Shared across rungs: the scorer, the tokenisation and truncation, the target (swap flag), and the length control.
  Length filter: |ℓ_c − ℓ_r| / max(ℓ_c, ℓ_r, 1) ≤ 0.25 in characters, applied identically to all rungs. Alternatively,
  the length covariates [ℓ_c − ℓ_r, |ℓ_c − ℓ_r|] are added to every rung.

**Summary.** T and F are per-token mean log-probabilities of the presented chosen and rejected response, and I is the
equal-weight mean of the two responses' per-token mean entropies. All three come from one scorer with the same masking,
and the same length control is applied to every rung.

## Comment 20: data flow for the preference experiments (A5)

Recomputed from the files in `data/` (`revision/a5_flow.py`): `c20_flow_by_source.csv`, `c20_exclusions.csv`,
`c20_shp_selection.csv`, `c20_ultrafeedback_overlap.csv`.

| Source | Drawn | Excluded | Scored | Length-filtered (≤ 0.25) | Removed by filter | Retained share |
|---|---|---|---|---|---|---|
| orca_dpo_pairs | 12,000 | 248 | 11,752 | 2,882 | 8,870 | 0.245 |
| UltraFeedback, gap ≥ 4 draw | 10,000 | 850 | 9,150 | 1,498 | 7,652 | 0.164 |
| UltraFeedback, gap ≥ 2 draw | 15,000 | 837 | 14,163 | 3,070 | 11,093 | 0.217 |
| Anthropic/hh-rlhf | 15,000 | 20 | 14,980 | 3,888 | 11,092 | 0.260 |
| SHP | 15,000 | 0 | 15,000 | 2,618 | 12,382 | 0.175 |
| **Sum of rows** | **67,000** | **1,955** | **65,045** | 13,956 | 51,089 | |

**Exclusions.** All 1,955 exclusions are explained by the < 2-token response rule; no other reason occurs:

| Source | Chosen < 2 tokens | Rejected < 2 tokens | Both |
|---|---|---|---|
| orca | 181 | 66 | 1 |
| UltraFeedback gap ≥ 4 | 403 | 363 | 84 |
| UltraFeedback gap ≥ 2 | 397 | 340 | 100 |
| Anthropic/hh-rlhf | 8 | 12 | 0 |

**UltraFeedback gap ≥ 4 vs gap ≥ 2.** The two files are **separate draws**: gap ≥ 2 was drawn from a pool that contains
the gap ≥ 4 pairs. They overlap. Pairs are matched by prompt and the unordered pair of responses.

| Item | Pairs |
|---|---|
| Pairs in both draws (drawn) | 6,514 |
| Pairs in both draws and scored in both | 5,967 |
| Unique UltraFeedback pairs drawn / scored | 18,486 / 17,346 |
| Unique pairs over all five files, scored | 65,045 − 5,967 = **59,078** |

So the gap ≥ 4 set is not a subset of the gap ≥ 2 *draw*. Its pairs belong to the gap ≥ 2 *population*, and 5,967 of
the 9,150 scored gap ≥ 4 pairs also appear in the scored gap ≥ 2 draw.

**SHP.** Scored SHP pairs: **15,000** (not 24,150). SHP has no corruption; the target in E4a is a 50% presentation swap
(7,477 swapped). E4b selection (`c20_shp_selection.csv`):

| Subset | Scored | Length-balanced |
|---|---|---|
| Near tie, ratio < 1.5 | 2,893 | 543 |
| Decisive, ratio ≥ 3 | 5,595 | 942 |
| 1.5 ≤ ratio < 3 (excluded from E4b) | 6,512 | 1,133 |

**Summary.**
- 67,000 pairs were drawn and 65,045 were scored. All 1,955 exclusions are pairs with a response of fewer than 2 tokens.
- The two UltraFeedback files share 5,967 scored pairs, so 59,078 distinct pairs were scored in total.
- There are 15,000 scored SHP pairs. The reviewer's 24,150 equals 65,045 minus the orca, UltraFeedback gap ≥ 2 and
  HH-RLHF counts; that is, it treats the UltraFeedback gap ≥ 4 file as not counted.

## Comments 5, 6: explicit head penalty (C1)

### C1(a): theorem check on frozen backbones

**What was run.** `revision/c1a_heads.py`, on the 18 retrained NS-U backbones (replicas; 2 of 18 bit-identical to q1, the others agree with q1 on 98.2–99.95%
of test predictions; `c21_replica_fidelity.csv`).
- The training-pixel features h = f(x) were taken with dropout off.
- Only the heads T (W, b) and F (V, c) were refit, by full-batch L-BFGS (strong Wolfe, float64), on eq. (6) plus
  λ/2(‖W‖² + ‖b‖² + ‖V‖² + ‖c‖²), starting from the trained heads.
- λ ∈ {0 (control), 1e-4, 1e-3, 1e-2}; iterations ran until ‖∇‖ < 1e-6.
- Per run: `c06_head_refit_by_run.csv`. It includes the final gradient norm, max and mean |V + W| and |c + b|, test
  accuracy, and B1 statistics on training and test pixels.

| λ | Runs reaching ‖∇‖ < 1e-6 | Final ‖∇‖ | max abs(V + W) | max abs(c + b) | Test mean abs(T+F−1) | Test p99 of T+F−1 | Head parameter norm | Test accuracy |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | 18/18 | 3.2e-12 – 1.7e-11 | 0.167 – 1.557 | 0.096 – 0.624 | 4.2e-4 – 1.1e-2 | 2.6e-7 – 7.5e-2 | 22.4 – 54.5 | 0.823 – 0.963 |
| 1e-4 | 18/18 | 1.2e-10 – 7.3e-10 | 1.5e-7 – 5.2e-7 | 2.0e-8 – 1.3e-7 | 8.5e-10 – 1.3e-8 | 1.1e-8 – 1.5e-7 | 7.7 – 10.9 | 0.824 – 0.963 |
| 1e-3 | 18/18 | 3.0e-10 – 8.4e-10 | 3.5e-8 – 9.6e-8 | 6.6e-9 – 3.1e-8 | 4.5e-10 – 2.9e-9 | 3.4e-9 – 3.2e-8 | 6.1 – 8.6 | 0.823 – 0.964 |
| 1e-2 | 18/18 | 3.6e-10 – 4.2e-9 | 3.3e-9 – 2.9e-8 | 5.9e-10 – 1.3e-8 | 5.9e-11 – 1.4e-9 | 3.9e-10 – 2.2e-8 | 4.6 – 6.3 | 0.822 – 0.963 |

**Summary.**
- **With an explicit penalty (λ > 0),** L-BFGS converges in all 18 runs. V = −W and c = −b hold to at most 5.2e-7, and
  |T + F − 1| on test pixels is at most 1.5e-7 at the 99th percentile.
- **Without the penalty (λ = 0),** the gradient norm also falls below 1e-6, but V + W stays away from zero (max 0.17–1.56)
  and test residuals keep the order of the original q1 runs (mean up to 1.1e-2). In this case the objective goes to 0:
  the training features are separable and the head norm grows to 22–55.

**Deviation.** At λ = 0 the small gradient reflects a vanishing loss (no finite minimiser), not convergence to a
stationary point. The training-pixel residuals are near 0 at λ = 0 as well (≤ 9e-13), because the training predictions
saturate.

### C1(b): end-to-end training with an explicit head penalty

**What was run.** `revision/hsi_rev.py --variant nsu_pen`. NS-U was retrained for all 9 split × seed combinations per
scene (18 runs per λ), with the original splits, seeds and data order.
- Loss = eq. (6) + λ/2 ‖heads‖², with biases included in the penalty.
- The head parameters use Adam with no decoupled weight decay (AdamW, weight_decay = 0). The backbone is unchanged:
  AdamW with lr 1e-3 and weight decay 1e-4.
- Schedule: cosine over **100 epochs**, chosen so that training reaches a plateau.
- λ ∈ {1e-4, 1e-3, 1e-2}; matched control λ = 0, which is identical except for the penalty.
- 72 runs in total.

Outputs:
- Per run: `c05_06_penalised_nsu_by_run.csv`. It includes the loss of the last epoch, the relative change over the last
  10 epochs, the full-batch head gradient norm at the end, and the B1 statistics.
- Ladder (prespecified steps and B6 steps): `c05_06_penalised_nsu_ladder.csv`.
- Loss curves: `c05_06_penalised_nsu_loss_curves.png`.

| λ | Scene | Accuracy (min–max; mean) | Head ‖∇‖ at end | Runs with relative loss change < 1% over the last 10 epochs | Test mean abs(T+F−1) | Test p99 of T+F−1 | Share abs res > 0.05 |
|---|---|---|---|---|---|---|---|
| 0 | Pavia | 0.824–0.927; 0.872 | 8.0e-4 – 1.0e-3 | 5/18 (both scenes) | 0.0010 – 0.0058 | 0.006 – 0.123 | 0.0016 – 0.0436 |
| 0 | Salinas | 0.841–0.960; 0.913 | 5.8e-4 – 1.7e-3 | | 0.0012 – 0.0090 | 0.011 – 0.112 | 0.0016 – 0.0748 |
| 1e-4 | Pavia | 0.824–0.927; 0.872 | 4.9e-4 – 6.5e-4 | 5/18 | 0.0010 – 0.0049 | 0.005 – 0.109 | 0.0012 – 0.0368 |
| 1e-4 | Salinas | 0.841–0.964; 0.913 | 5.9e-4 – 1.7e-3 | | 0.0013 – 0.0084 | 0.015 – 0.096 | 0.0024 – 0.0706 |
| 1e-3 | Pavia | 0.824–0.926; 0.871 | 4.3e-3 – 4.8e-3 | 10/18 | 0.0007 – 0.0043 | 0.004 – 0.100 | 0.0000 – 0.0370 |
| 1e-3 | Salinas | 0.847–0.964; 0.914 | 6.1e-3 – 7.0e-3 | | 0.0007 – 0.0047 | 0.010 – 0.067 | 0.0001 – 0.0265 |
| 1e-2 | Pavia | 0.824–0.924; 0.871 | 1.6e-2 – 1.7e-2 | 12/18 | 0.00014 – 0.00071 | 0.0010 – 0.0145 | 0.0000 |
| 1e-2 | Salinas | 0.828–0.966; 0.908 | 2.0e-2 – 2.3e-2 | | 0.00006 – 0.00031 | 0.0003 – 0.0028 | 0.0000 |

NS-U ladder under each λ (mean Δ over 9 runs, SD, range, runs passing the rule):

| λ | Scene | N2−N1 | N2+−N2base | N2base+T − N2base | N2+ − (N2base+T) |
|---|---|---|---|---|---|
| 0 | Pavia | +0.0052 (0.0351) [−0.045, +0.079], 1/9 | +0.1273 (0.0497) [+0.060, +0.206], 9/9 | +0.1214, 9/9 | +0.0059, 1/9 |
| 0 | Salinas | +0.0016 (0.0049) [−0.006, +0.011], 0/9 | +0.1140 (0.0502) [+0.048, +0.192], 9/9 | +0.1094, 8/9 | +0.0046, 0/9 |
| 1e-4 | Pavia | +0.0041 (0.0374), 2/9 | +0.1473 (0.0626), 9/9 | +0.1401, 9/9 | +0.0072, 2/9 |
| 1e-4 | Salinas | −0.0052 (0.0320), 0/9 | +0.0991 (0.0377), 9/9 | +0.1035, 8/9 | −0.0043, 0/9 |
| 1e-3 | Pavia | +0.0112 (0.0151), 1/9 | +0.1846 (0.0487), 9/9 | +0.1716, 9/9 | +0.0131, 1/9 |
| 1e-3 | Salinas | +0.0071 (0.0179), 1/9 | +0.1163 (0.0511), 8/9 | +0.1085, 8/9 | +0.0077, 1/9 |
| 1e-2 | Pavia | +0.0030 (0.0042) [−0.000, +0.011], 0/9 | +0.1504 (0.0796) [+0.036, +0.280], 9/9 | +0.1473, 8/9 | +0.0030, 0/9 |
| 1e-2 | Salinas | +0.0000 (0.0000), 0/9 | +0.0582 (0.0554) [−0.024, +0.130], 6/9 | +0.0582, 6/9 | +0.0000, 0/9 |

N2base+F − N2base is in the CSV; it equals N2base+T − N2base to within 0.0004 in mean.

**Summary.**
- **Residual.** With λ = 1e-2, the test residual |T + F − 1| has mean ≤ 7.1e-4 and p99 ≤ 0.0145, and no pixel exceeds
  0.05. With λ = 0 (control), the mean is up to 0.0090 and the p99 up to 0.123.
- **N2−N1** (adding the second coordinate) passes in at most 2 of 9 runs for any λ. At λ = 1e-2 its mean is +0.0030
  (Pavia) and +0.0000 (Salinas).
- **N2+ − N2base** still passes in 9/9 (Pavia) and 6/9 (Salinas) runs at λ = 1e-2. The same gain is obtained by adding T
  alone (N2base+T − N2base), and adding F afterwards gives +0.0030 / +0.0000.
- **Accuracy** changes by at most 0.006 in mean between λ = 0 and λ = 1e-2.

**Deviations.**
- 100 epochs instead of the original 40 (all λ, including the control), to approach a plateau.
- The prespecified plateau criterion was not met in every run: relative loss change < 1% over the last 10 epochs was
  reached in 5/18 (λ = 0), 5/18 (1e-4), 10/18 (1e-3) and 12/18 (1e-2) runs.
- The final head gradient norms are 4.9e-4 to 2.3e-2, not near 0. The cosine schedule drives the learning rate to 0 at
  the last epoch, so training stops rather than converges. Reported as is.

## Comment 16 (continued): alternative evidential objective (C2)

**What was run.** The evidential HSINet was retrained on the same 18 split × seed combinations, with the same 40-epoch
protocol, in two variants:
- `evid_log`: the Type-II maximum-likelihood ("log") loss of Sensoy et al. (2018), Σ_k y_k (log S − log α_k), with the
  same KL annealing λ_t = min(1, t/10).
- `evid_mse_anneal20`: the original MSE form with annealing λ_t = min(1, t/20).

Outputs:
- Per run: `c16_alt_evidential_by_run.csv` (accuracy, NLL, Brier, ECE, S−K, evidence shares).
- Ladder (prespecified and B6 steps): `c16_alt_evidential_ladder.csv`.
- Per-class accuracy: `c16_alt_evidential_perclass_acc.csv`.

| Variant | Scene | Accuracy | NLL | Brier | ECE | Mean S−K (range) | Share with evidence > 0.01 on ≤ 1 class | Share total evidence < 0.01 |
|---|---|---|---|---|---|---|---|---|
| original MSE, anneal 10 (B9/B10) | Pavia | 0.869 ± 0.046 | 0.629 ± 0.145 | 0.234 ± 0.069 | 0.096 ± 0.037 | 30.5–34.6 | 0.727–0.859 | 0.0000 |
| log (Type-II ML), anneal 10 | Pavia | 0.868 ± 0.044 | 0.619 ± 0.144 | 0.232 ± 0.066 | 0.096 ± 0.039 | 33.6–39.6 | 0.704–0.833 | 0.0000 |
| MSE, anneal 20 | Pavia | 0.869 ± 0.045 | 0.630 ± 0.142 | 0.234 ± 0.067 | 0.103 ± 0.052 | 30.8–35.9 | 0.730–0.858 | 0.0000 |
| original MSE, anneal 10 (B9/B10) | Salinas | 0.871 ± 0.053 | 0.697 ± 0.134 | 0.230 ± 0.051 | 0.197 ± 0.043 | 38.1–42.9 | 0.766–0.889 | 0.0000–0.0814 |
| log (Type-II ML), anneal 10 | Salinas | 0.897 ± 0.052 | 0.595 ± 0.149 | 0.195 ± 0.058 | 0.190 ± 0.039 | 43.6–45.5 | 0.708–0.868 | 0.0000–0.0133 |
| MSE, anneal 20 | Salinas | 0.886 ± 0.046 | 0.633 ± 0.126 | 0.207 ± 0.049 | 0.193 ± 0.034 | 41.0–44.9 | 0.731–0.891 | 0.0000–0.0646 |

Evidential ladder (mean Δ over 9 runs (SD), runs passing the rule):

| Variant | Scene | R2−R1 | R3+−R2 | R3−R2 | (H,u,b_max)−(H,u) | binary-frame triple − (b_max,u) |
|---|---|---|---|---|---|---|
| log | Pavia | +0.0861 (0.1468), 3/9 | +0.0013 (0.0073), 0/9 | −0.0029 (0.0045), 0/9 | +0.0002, 0/9 | −0.0000, 0/9 |
| log | Salinas | +0.0048 (0.0080), 0/9 | +0.0006 (0.0010), 0/9 | −0.0022 (0.0043), 0/9 | +0.0005, 0/9 | +0.0001, 0/9 |
| MSE, anneal 20 | Pavia | +0.0674 (0.1272), 3/9 | −0.0071 (0.0152), 0/9 | −0.0025 (0.0034), 0/9 | −0.0082, 0/9 | −0.0000, 0/9 |
| MSE, anneal 20 | Salinas | −0.0002 (0.0008), 0/9 | +0.0013 (0.0022), 0/9 | +0.0011 (0.0018), 0/9 | +0.0014, 0/9 | +0.0000, 0/9 |

**Summary.**
- The log objective changes Pavia accuracy by −0.0001 and Salinas by +0.026 in mean, compared with the original MSE form.
- In both variants no step beyond (H, u) passes the rule in either scene. R2−R1 passes in 3 of 9 Pavia runs in both
  variants.
- The binary-frame triple again matches (b_max, u) to within 0.0001 in mean.

**Deviations.** The Type-II ML form was chosen (the request allowed "log or digamma"); the digamma cross-entropy form
was not run.

## Comment 17 (continued): shift and OOD (C3)

**(i) Covariate shift.** Done on the 70 retrained replicas (see "Retrained models"), with no retraining.
- **Noise:** N(0, (σ·sd_b)²) was added to the raw spectra of all pixels before standardisation and PCA, σ ∈ {0.05, 0.1,
  0.2}. sd_b is the SD of band b over the training pixels of the split.
- **Band zeroing:** 10% or 20% of the bands were set to 0, chosen with rng 1337; the same bands are used for every run.
  The noise also uses rng 1337.
- Then each run's stored standardisation and PCA were applied, with the same 11×11 patches.
- The detector was refit and bootstrapped as in q1. "none" = the replica without shift.

Outputs: per run `c17_shift_calibration.csv` and `c17_shift_ladders.csv`; summaries `c17_shift_summary.csv` and
`c17_shift_ladder_counts.csv`.

Accuracy / NLL / ECE, mean over runs:

| Scene | Model (probabilities) | none | noise 0.05 | noise 0.1 | noise 0.2 | bands 10% | bands 20% |
|---|---|---|---|---|---|---|---|
| Pavia | softmax (p) | 0.870/1.017/0.108 | 0.871/1.009/0.107 | 0.871/0.991/0.105 | 0.867/0.940/0.101 | 0.325/8.860/0.578 | 0.323/10.911/0.604 |
| Pavia | MC dropout (mean p) | 0.870/0.943/0.104 | 0.871/0.936/0.102 | 0.870/0.919/0.100 | 0.867/0.873/0.094 | 0.324/7.596/0.542 | 0.323/9.192/0.540 |
| Pavia | ensemble (mean p) | 0.927/0.772/0.052 | 0.926/0.774/0.052 | 0.925/0.779/0.052 | 0.918/0.799/0.052 | 0.319/6.153/0.466 | 0.485/6.017/0.331 |
| Pavia | evidential (α/S) | 0.869/0.629/0.096 | 0.869/0.628/0.099 | 0.868/0.630/0.105 | 0.863/0.652/0.128 | 0.308/2.651/0.419 | 0.258/2.684/0.461 |
| Pavia | NS-U (softmax(zT−zF)) | 0.873/2.243/0.120 | 0.873/2.208/0.119 | 0.873/2.139/0.119 | 0.867/1.927/0.120 | 0.241/15.103/0.713 | 0.240/17.538/0.744 |
| Pavia | NS-U (T/ΣT) | 0.873/0.794/0.100 | 0.873/0.790/0.098 | 0.873/0.785/0.096 | 0.867/0.783/0.089 | 0.242/8.483/0.648 | 0.239/10.637/0.718 |
| Pavia | HybridSN softmax | 0.850/2.873/0.141 | 0.847/2.895/0.143 | 0.842/2.960/0.147 | 0.823/3.198/0.165 | 0.433/14.454/0.557 | 0.585/10.694/0.410 |
| Pavia | HybridSN evidential | 0.715/1.117/0.109 | 0.712/1.127/0.110 | 0.708/1.155/0.114 | 0.690/1.248/0.124 | 0.404/3.538/0.533 | 0.418/3.250/0.444 |
| Salinas | softmax (p) | 0.909/0.577/0.065 | 0.910/0.578/0.065 | 0.909/0.589/0.066 | 0.902/0.646/0.072 | 0.117/13.841/0.817 | 0.186/13.909/0.730 |
| Salinas | MC dropout (mean p) | 0.909/0.532/0.060 | 0.910/0.532/0.059 | 0.909/0.541/0.060 | 0.902/0.592/0.066 | 0.116/10.625/0.731 | 0.184/10.867/0.612 |
| Salinas | ensemble (mean p) | 0.958/0.197/0.019 | 0.958/0.196/0.019 | 0.956/0.200/0.021 | 0.947/0.224/0.030 | 0.145/10.905/0.658 | 0.270/7.671/0.298 |
| Salinas | evidential (α/S) | 0.869/0.704/0.199 | 0.869/0.704/0.198 | 0.867/0.710/0.197 | 0.853/0.752/0.188 | 0.140/3.276/0.427 | 0.180/3.486/0.453 |
| Salinas | NS-U (softmax(zT−zF)) | 0.904/1.189/0.085 | 0.904/1.186/0.084 | 0.903/1.212/0.086 | 0.889/1.411/0.099 | 0.129/19.749/0.834 | 0.104/19.594/0.834 |
| Salinas | NS-U (T/ΣT) | 0.904/0.452/0.058 | 0.904/0.452/0.058 | 0.903/0.462/0.059 | 0.889/0.532/0.071 | 0.127/11.496/0.779 | 0.106/11.733/0.759 |
| Salinas | HybridSN softmax | 0.862/1.905/0.126 | 0.863/1.893/0.126 | 0.862/1.898/0.127 | 0.855/1.971/0.132 | 0.173/21.743/0.811 | 0.161/22.156/0.826 |
| Salinas | HybridSN evidential | 0.573/1.532/0.077 | 0.572/1.535/0.077 | 0.570/1.543/0.079 | 0.565/1.562/0.080 | 0.135/3.910/0.438 | 0.117/6.478/0.826 |

Ladder verdicts under shift (contrasts passing the rule; full list per run in `c17_shift_ladders.csv`):

| Condition | CNN per-run (162) | Ensemble (4) | HybridSN (30) |
|---|---|---|---|
| none (replicas) | 27 | 1 | 0 |
| noise 0.05 | 27 | 1 | 0 |
| noise 0.1 | 26 | 1 | 0 |
| noise 0.2 | 25 | 1 | 0 |
| bands zeroed 10% | 75 | 1 | 2 |
| bands zeroed 20% | 53 | 1 | 2 |

**Summary.** Gaussian noise up to σ = 0.2 band SD lowers accuracy by at most 0.027 (HybridSN softmax, Pavia), and leaves
the number of passing CNN contrasts at 25–27. Zeroing 10% or 20% of the bands lowers accuracy to 0.10–0.59, and the
number of passing CNN contrasts rises to 75 and 53. The steps that pass under band zeroing include softmax, MC-dropout,
evidential and NS-U steps (per-step counts in `c17_shift_ladders.csv`).

**(ii) OOD = training-absent classes** (A2). Test pixels of training-absent classes were treated as OOD, the other test
pixels as in-distribution. This uses the stored q1 outputs, with no shift and no detector fitting; it is the AUROC of
each single uncertainty score (`c17_ood_auroc.csv`). Mean [min, max] over runs:

| Score | Pavia split 0 (1,345 OOD / 28,176 ID) | Salinas split 1 (2,811 OOD / 30,614 ID) |
|---|---|---|
| softmax H(p) | 0.470 [0.000, 0.720] | 0.749 [0.636, 0.868] |
| softmax 1 − p_max | 0.478 [0.000, 0.733] | 0.740 [0.635, 0.851] |
| MC dropout PE | 0.552 [0.002, 0.834] | 0.761 [0.656, 0.875] |
| MC dropout MI | 0.603 [0.008, 0.905] | 0.769 [0.674, 0.873] |
| ensemble H(mean) / MI / 1 − p_max | 0.828 / 0.925 / 0.830 (single ensemble) | not applicable (ensemble on split 0; Salinas split 0 has no absent class) |
| evidential H(α/S) | 0.720 [0.625, 0.798] | 0.777 [0.749, 0.800] |
| evidential u = K/S | 0.373 [0.154, 0.586] | 0.776 [0.755, 0.792] |
| evidential 1 − b_max | 0.743 [0.625, 0.824] | 0.773 [0.742, 0.802] |
| NS-U −(T − F) | 0.969 [0.955, 0.980] | 0.902 [0.861, 0.959] |
| NS-U 1 − T | 0.968 [0.953, 0.978] | 0.902 [0.855, 0.960] |
| NS-U F | 0.970 [0.957, 0.983] | 0.902 [0.866, 0.957] |
| NS-U H(p̃) | 0.932 [0.840, 0.983] | 0.871 [0.840, 0.925] |
| HybridSN softmax H(p) | 0.224 (1 run) | 0.829 (1 run) |
| HybridSN evidential u | 0.019 (1 run) | 0.580 (1 run) |

**Summary.** For the training-absent classes, the NS-U scores have the highest mean AUROC: 0.97 (Pavia) and 0.90
(Salinas). Softmax H(p) on Pavia split 0 ranges from 0.000 to 0.720 across the 3 training seeds. In one run the AUROC is 0.000 to four decimals: the OOD pixels
receive lower entropy than almost all in-distribution pixels. Evidential u has a mean AUROC of 0.373 on Pavia.

**Deviations.**
- (i) uses retrained replicas (see "Retrained models"), not the exact q1 networks.
- sd_b for the noise is taken over the training pixels of the split; the request did not specify the reference set.
- Shifted ladders reuse the original detector protocol.
- OOD in (ii) is defined by the A2 classes and is available only for Pavia split 0 and Salinas split 1.

## Comment 8: positive control with independent annotations (D)

**What was run.** The plan was written before any training: `D_plan.md`. Code: `revision/d_cifar10h.py`.
- **Data:** CIFAR-10H, 47–63 independent human labels per CIFAR-10 test image (sha256-verified file).
- **Target:** the disagreement target is the entropy of the annotator label distribution, normalised by ln 10. It was
  computed either from all annotators or from m randomly chosen annotators per image (rng 1337).
- **Training:** ResNet-18, pretrained on CIFAR-10 train, then fine-tuned for 30 epochs on 7,000 images per fold. Five
  folds give out-of-fold predictions for all 10,000 images.
- **Heads:** a classifier head trained with cross-entropy on the CIFAR-10 labels (no human data), and an indeterminacy
  head Î = sigmoid(z) trained with MSE to the target.
- **Detection target:** the classifier's own errors against the CIFAR-10 label.
- **Ladder:** D1 = (H(p), p_max) → D2 = (H(p), p_max, Î) → D3 = (sorted p, Î). Same detector, bootstrap
  (B = 3000, seed 1337) and rule as in the manuscript.
- **Runs:** m = all with seeds 0, 1, 2; m ∈ {20, 10, 5, 3, 2} with seed 0. 40 fine-tunes, 31.2 min GPU in total.

Per run: `c08_cifar10h_positive_control.csv`.

| Annotators per image m | Seed | Accuracy | Errors | Spearman ρ(Î, H of all annotators) | D2−D1: Δ [95% CI] | Rule | D3−D2: Δ [95% CI] | Rule |
|---|---|---|---|---|---|---|---|---|
| all | 0 | 0.9425 | 575 | 0.336 | −0.0057 [−0.0136, +0.0026] | no | −0.0181 [−0.0221, −0.0143] | no |
| all | 1 | 0.9398 | 602 | 0.350 | +0.0015 [−0.0075, +0.0108] | no | −0.0113 [−0.0146, −0.0081] | no |
| all | 2 | 0.9430 | 570 | 0.327 | −0.0032 [−0.0134, +0.0070] | no | −0.0140 [−0.0174, −0.0107] | no |
| 20 | 0 | 0.9433 | 567 | 0.321 | −0.0074 [−0.0155, +0.0004] | no | −0.0212 [−0.0258, −0.0168] | no |
| 10 | 0 | 0.9424 | 576 | 0.292 | −0.0113 [−0.0193, −0.0033] | no | −0.0208 [−0.0252, −0.0166] | no |
| 5 | 0 | 0.9416 | 584 | 0.266 | −0.0091 [−0.0162, −0.0021] | no | −0.0224 [−0.0284, −0.0171] | no |
| 3 | 0 | 0.9401 | 599 | 0.212 | −0.0104 [−0.0172, −0.0037] | no | −0.0253 [−0.0309, −0.0199] | no |
| 2 | 0 | 0.9404 | 596 | 0.127 | −0.0107 [−0.0162, −0.0057] | no | −0.0248 [−0.0310, −0.0195] | no |

AUCs of D1 / D2 for m = all, seed 0: 0.9201 / 0.9145 (all rungs per run are in the CSV).

**Summary.**
- The predicted-disagreement head does not pass the rule in any of the 8 runs, for any number of annotators per image.
  D2−D1 ranges from −0.0113 to +0.0015.
- Its rank correlation with the all-annotator entropy rises with m: 0.127 at m = 2, 0.336 at m = all for seed 0.
- The full representation D3 is below D2 in every run.
- In this setting, about 50 independent annotations per image were not sufficient for the third signal to pass. No
  amount of annotation data tested made it pass.

**Deviations.** None from `D_plan.md`. The target of the ladder is the classifier's error against the CIFAR-10 label,
as pre-specified. A ladder whose target is human disagreement itself was not run.

---

## Missing inputs

| # | What is missing | Effect | Consequence |
|---|---|---|---|
| 1 | Git repository: no history, no commit hash | A1 (comment 21), ENVIRONMENT.md | Provenance rests on SHA-256 hashes and local file timestamps (`c21_repository_manifest.csv`), with no external or version-control timestamp. |
| 2 | Manuscript text (Section 6.3, eq. (6), rule (10)) | Exact wording only | Table 4 (11 rows, 166 comparisons) was supplied by the author and matches the recomputation (see "Table 4 = the primary family"). Ladders, eq. (6) and rule (10) were taken from the pre-registered plan and the code. |
| 3 | Network weights of the q1 runs (never saved) | C1(a), C3(i) | Run on retrained replicas: 20/70 bit-identical, the others from the same protocol and seeds (`c21_replica_fidelity.csv`). |
| 4 | Training-pixel outputs of the q1 runs (never saved) | B1 on training pixels | B1 is on test pixels only. Training-pixel residuals are reported for the retrained NS-U backbones in C1(a). |
| 5 | Digamma cross-entropy form of EDL | C2 | Not run; the Type-II ML (log) form was run instead (the request allowed either). |
| 6 | Ensemble on splits 1 and 2 (only split 0 was trained in q1) | C3(ii) OOD for Salinas | No ensemble OOD value for Salinas split 1. |
