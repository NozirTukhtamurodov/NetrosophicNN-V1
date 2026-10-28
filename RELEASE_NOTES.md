# Release notes, version 1.0.0 (2026-10-27)

This release holds two packages for the article *How Many Degrees of Freedom Does a Neutrosophic Uncertainty
Representation Have?*:

- `github/`: curated code, analysis results, figure inputs, preference scores and splits. Every file is under
  50 MB.
- `zenodo/nsdof_outputs_v1.0.zip`: stored network outputs and trained weights.

`CODE_MAP.md` maps every result to its script. These notes record how the code was selected and cleaned, and
the evidence that the cleaning changed no result.

## 1. What was kept

Only code reachable from a `CODE_MAP.md` entry point was kept. Each release file came from these originals:

| Release file (`github/code/`) | Original | Notes |
|---|---|---|
| `common.py` | `q1/fastboot.py`, `revision/common.py`, `neutro_cnn/hsi_train.py` (`load_scene`, `make_patches`, `split_block_buffer`) | one copy of each helper; paths via `NSDOF_*` variables |
| `hsi_models.py` | `neutro_cnn/hsi_train.py` (HSINet, `edl_loss`), `q1/hsi_q1.py` (NSUNet, HybridSN), `revision/hsi_rev.py` (`kl_term`, `evid_log_loss`, `evid_mse_loss_a20`) | AST-identical to the originals |
| `hsi_train_q1.py`, `hsi_run_q1.py` | `q1/hsi_q1.py`, `q1/run_hsi.py` | |
| `hsi_analyze.py`, `hsi_summarize.py`, `hsi_explore_nsu.py` | `q1/analyze_hsi.py`, `q1/summarize.py`, `q1/explore_nsu.py` | |
| `hsi_train_transductive.py` | `neutro_cnn/hsi_train.py` (`main`) | Indian Pines entry removed |
| `hsi_transductive_audit.py` | `neutro_cnn/audit/audit_a.py`, `audit_b6.py` | JSON keys kept as in the original (Russian), so the outputs stay identical |
| `pref_sample_pairs.py`, `pref_sample_shp.py` | `ns_data_multi.py`, `ns_data_natural.py` | |
| `pref_score.py` | `ns_logprob.py` | |
| `pref_detect.py`, `pref_detect_shp.py` | `ns_logprob_detect.py`, `ns_natural_detect.py` | |
| `pref_fold_variability.py` | `q1/pref_d1.py` | unreported parts removed (D2, the UF gap > 0 and HH-RLHF 2,000-pair runs) |
| `pref_scorer_robustness.py` | new | counts the 48 comparisons of Section 7.2 from the stored reports |
| `rev_*.py` (15 files) | `revision/{a1_a2_splits, a3_accounting, a5_flow, b1_residuals, b_engine, b_tables, b4_pooled, b4_b6steps, b9_b10_calib_evid, c1a_heads, hsi_rev, drive_train, c_analyse, c3_shift, t4_tables}.py` | `rev_c3_shift.py summary` is new; it writes `c17_shift_summary.csv` and `c17_shift_ladder_counts.csv`, which were previously assembled by hand |
| `cifar10h_common.py`, `cifar10h_positive_control.py` | `nsme/nsunc/cifar.py`, `c10h.py`, `runner.py` (determinism flags), `revision/d_cifar10h.py` | pretraining extracted from the NS-ME runner (Open issue 7) |
| `simulation/simulation.py`, `figures/{style, fig1_simulation, fig2_fig3_hsi, fig4_preference}.py` | supplied by the authors (written 2026-10-03) | reformatted only |
| `figures/make_figure_inputs.py` | new | writes the three figure-input files that HANDOFF.md specifies |
| `scripts/download_data.py`, `tests/test_figure_inputs.py` | new | |

The other contents of `github/` are:

- `results/`: the stored tables and reports. `results/q1/transductive_a.json` and `transductive_b6.json` are the
  stored `neutro_cnn/audit/audit_a.json` and `audit_b6.json` (Section 6.5), renamed;
- `preference_scores/`: numbers-only versions of the score files, plus `preference_pair_ids.csv`;
- `splits/`;
- `plan/`.

## 2. What was not released, and why

| Material | Files | Reason |
|---|---|---|
| DPO / judge-panel pipeline | `train_dpo.py`, `train_dpo_cuda.py`, `ns_judge.py`, `ns_detect.py`, `calibrate_weights.py`, `merge_export.py`, `data/judgments*/`, `data/ns_labels*.jsonl`, `ns_dpo_dataset*.jsonl`, `dpo_out/`, `dpo_data.csv` | the paper does not report it (excluded by the authors) |
| Spatial-leakage experiment | `neutro_cnn/c0_leakage.py`, `q1/g1_leakage.py`, `q1/g1_summary.py`, `q1/g1/` | excluded by the authors |
| Length-leak direction analysis | `q1/g2_length.py`, `q1/g2/` | excluded by the authors |
| Indian Pines, KSC | Indian Pines entry in `neutro_cnn/hsi_train.py` (`SCENES`), its runs and data | excluded by the authors. No code references KSC; it is mentioned only in `q1/Q1_REPORT.md` |
| Earlier RAG / UQ phase | `main.py`, `neutro_core.py`, `ns_rag.py`, `ns_uq.py`, `ns_align.py`, `ns_collector.py`, `ns_dataset_builder.py` | not used by the paper |
| orca-487 builder | `ns_data.py` | the paper uses its 487 pairs only. They are identified in `preference_pair_ids.csv` (`sample = orca_487`: row index, swap flag, stored `i_consistency`). `download_data.py` rebuilds the file byte for byte (SHA-256 checked). The builder also needs the judge-panel verdicts of the DPO pipeline |
| Exploratory audits that are not reported | `neutro_cnn/hsi_detect.py`, `neutro_cnn/audit/{audit_a5, audit_b3, audit_b6_folds}.py`, `audit/` (2026-10-02 recomputation), `q4a/` | no manuscript item uses them; the revision analyses supersede them |
| Revision helpers | `revision/a1_manifest.py` (file list of the working tree, with absolute paths and mtimes), `revision/make_article_kit.py` (assembles manuscript text), `revision/update_report_t4.py` (one-off edit of `REVISION_REPORT.md`) | they produce no result; `MANIFEST_sha256.csv` replaces the first |
| NS-ME experiment line | `nsme/` except the parts listed in section 1 | a separate study |
| Files that are not article results | `results/revision/csv/c21_repository_manifest.csv` (file list of the old working tree, including Indian Pines and KSC), `results/q1/runs_hsi.csv` (training wall-clock log), the `.ruff_cache/` lint cache | not needed for any result. `MANIFEST_sha256.csv` lists the files of this repository |
| Documents and environment | `ARTICLE_*.md`, `DOCS/`, `FILES_KB.md`, `q1/Q1_REPORT*`, `CLAUDE.md`, `*.bat`, `training/` (the virtual environment) | not code, or internal |
| Third-party data | `.mat` scenes, CIFAR-10/10H, the preference texts | not redistributed; `scripts/download_data.py` fetches them from pinned sources |

## 3. Lines of code before and after cleaning

"Before" is the whole original files listed in section 1; "after" is the release files.

- **Code lines** exclude blank lines, comments and docstrings.
- **Statements** counts AST statements, which does not depend on formatting.

Line counts grow after cleaning because `ruff format` at line length 100 splits the dense original lines; the
statement count shows the actual reduction.

| Part | Files | Lines | Code lines | Statements | Files | Lines | Code lines | Statements |
|---|---|---|---|---|---|---|---|---|
| | **before** | | | | **after** | | | |
| hyperspectral | 9 | 1,104 | 872 | 809 | 9 | 1,624 | 1,380 | 883 |
| preference | 6 | 937 | 635 | 534 | 7 | 777 | 615 | 433 |
| revision | 16 | 1,876 | 1,579 | 1,401 | 15 | 3,358 | 3,039 | 1,309 |
| CIFAR-10H | 3 | 617 | 524 | 495 | 2 | 347 | 288 | 227 |
| simulation and figures | 5 | 356 | 287 | 256 | 6 | 511 | 418 | 313 |
| new: download script and tests | 0 | 0 | 0 | 0 | 2 | 272 | 225 | 136 |
| **Total** | 39 | 4,890 | 3,897 | 3,495 | 41 | 6,889 | 5,965 | 3,301 |

Hyperspectral statements grew because the shared helpers that were previously imported across folders now live
in `common.py` and `hsi_models.py`. Counted by `release/_work/loc.py`.

## 4. Removed for security or portability

- **Absolute paths.** `ROOT = 'D:\\AI_TRAINING_NOZIR'` in the revision scripts and `revision/d_cifar10h.py`, the
  session scratch path the audit scripts wrote to, and every `sys.path.insert` across folders were removed. All
  paths now derive from `common.py` (`NSDOF_DATA_DIR`, `NSDOF_OUTPUTS_DIR`, `NSDOF_RESULTS_DIR`,
  `NSDOF_SPLITS_DIR`) or from command-line arguments; `Q1_OUT` became `NSDOF_OUTPUTS_DIR`.
- **Cross-drive path.** `rev_a1_a2_splits.py` wrote `os.path.relpath(file, ROOT)`, which fails when the splits and
  the repository are on different drives. It now writes the file name.
- **Personal data.** The reviewer's name in `REVISION_REPORT.md` was replaced by "Review of 26 Oct 2026."; a
  note says that the paths in the report refer to the original working tree.
- **Third-party text.** The score files (`preference_scores/*.jsonl`) keep numbers only: lengths, mean
  log-probabilities and entropies, swap flag, id. Prompts and responses are rebuilt from the pinned dataset
  revisions.
- **Line endings.** `download_data.py` writes each pair file with the line ending of the original (LF for orca-487,
  CRLF for the others), so its SHA-256 check passes on any OS.
- **Messages.** Console messages were translated from Russian to English. JSON keys stay Russian, because
  changing them would change the outputs.
- **Secret scan.** Both packages were scanned for `hf_…` tokens, drive paths, `token=`, `api_key`, `password`,
  `secret`, e-mail addresses and the user name. Nothing real was found: the only hits are URLs and the
  `hf_dataset` column name. The packages contain no API keys, tokens or credentials.

## 5. Third-party code

No third-party source code was copied.

- `hsi_models.HybridSN` is an implementation, written for this study, of the architecture of Roy et al. (2020)
  with 11 × 11 patches.
- `cifar10h_common.resnet18` is torchvision's ResNet-18 with a 3 × 3 stride-1 stem and no max-pool.
- Everything else uses the libraries in `requirements.txt` (PyTorch, scikit-learn, SciPy, NumPy, transformers,
  datasets, matplotlib).

## 6. Proof that cleaning changed no result

The uncleaned original scripts were run once more, today, with only their root path changed: the "before"
baseline. The cleaned release code was then run on the same inputs (stored network outputs, stored score files,
same data): the "after" run. Comparison is byte-exact first, then cell by cell with tolerance 1e-12
(`release/_work/compare.py`, `compare_all.py`; full table in
`release/_work/comparison_after_vs_before.csv`).

| Group | Files | Identical | Explained differences |
|---|---|---|---|
| q1 tables (Tables 3–4, Sections 4, 6), after vs before | 12 | 12 | — |
| q1 tables, after vs stored originals | 12 | 9 | Only pooled Salinas rows differ (2 in C3_aucs, 4 in C3_deltas, 3 in X_nsu_exploratory; listed in Open issue 3). After = before = the fresh-venv run, so neither the cleaning nor the environment causes this. No verdict changes, and no per-run row (and therefore no Table 4 cell) differs |
| Transductive audit (Section 6.5), after vs before and vs the stored originals | 2 | 2 | — |
| Preference detector reports (Tables 5–6, Section 7.2), after vs before | 11 | 11 | — |
| Fold-seed variability (Tables 5–6) | 1 | 0 | after is an exact subset of before; the 12 missing rows are the removed, unreported runs (HH-RLHF 2,000 and UF gap > 0) |
| Revision outputs (Tables S1–S18; CSV, `t4_tables.md`, `check_engine_vs_table4.txt`), after vs stored | 59 | 53 | `c14_all_comparisons` row 215 and `c15_threshold_sensitivity` row 219: the same Salinas pooled row. `c06_head_refit_by_run`: only `wall_s` (run time). `c12_fold_auc_diagnostic`, `engine_foldauc`: same rows in a different order, because the original also iterates over a Python `set`. `c21_split_manifest`: only the `file` column (now a file name); the SHA-256 values are equal |
| Split files | 6 | 6 | — |
| Figure inputs (Figs. 2–4) | 4 | 4 | — |
| Simulation (Table 2, Fig. 1): `sim_results.json`, `s3_maps.npz` vs the authors' copy | 2 | 2 | — |
| Data download: 4 scenes, CIFAR-10H, CIFAR-10, 7 rebuilt pair files | 13 | 13 (SHA-256) | — |

`results/revision/REVISION_REPORT.md` and the two older Qwen-1.5B reports are stored inputs, not regenerated
(`CODE_MAP.md`, last section). The comparison table marks them `INPUT_COPY`.

**Training code** (exempt from rerunning): `release/_work/train_diff.py` compares the original and release
training scripts after AST normalisation, which removes docstrings, comments and formatting. The diffs are in
`release/_work/training_diffs/*.diff`.

- **Models and losses.** HSINet, NSUNet, HybridSN, `edl_loss`, `kl_term`, `evid_log_loss`, `evid_mse_loss_a20`,
  `make_patches` and `split_block_buffer` are AST-identical.
- **`load_scene`.** Only the data path changed.
- **`main` functions.** The diffs touch only:
  - output paths (environment variables);
  - translated messages;
  - removed progress and statistics prints;
  - the removed development options `--limit` (`pref_score.py`) and `--epochs` (`rev_train.py`, which keeps the
    same epoch counts: 100 for `nsu_pen`, 40 otherwise);
  - `with open(…)` blocks.
- **CIFAR-10 pretraining.** It now calls `train_softmax`. That function matches `cifar.train('softmax_hard', …)`
  line for line: K = 10, cross-entropy, SGD with momentum 0.9 and weight decay 5e-4, cosine schedule, bf16,
  crop + flip, 100 epochs, lr 0.1. It also sets the same determinism flags as the original `runner.py`.

## 7. Lint, tests and sizes

| Check | Result |
|---|---|
| `ruff check` (rules E, F, W; line length 100) on `code/`, `scripts/`, `tests/` | All checks passed (41 files) |
| `ruff check --select F401,F841` | 0 findings |
| `ruff format --check` | 41 files already formatted |
| `vulture --min-confidence 80` | 0 findings |
| `pytest tests` | 2 passed |

| Package | Files | Size | Largest file | SHA-256 |
|---|---|---|---|---|
| `github/` | 179 (177 listed in the manifest + `MANIFEST_sha256.csv` + `RELEASE_NOTES.md`) | 65.1 MB | `results/revision/csv/c07_points_F_vs_1minusT.csv` (18.0 MB); no file over 50 MB | every file listed with its SHA-256 in `MANIFEST_sha256.csv` |
| `zenodo/nsdof_outputs_v1.0.zip` | 898 entries (897 files + manifest) | 3,171,490,511 bytes = 3.17 GB (uncompressed 3.22 GB) | `hsi_outputs_revision` 1.6 GB, `shift/` 0.96 GB, `weights_revision/` 0.31 GB | `3486a59316e14a826124145f08ee8333a4b972080e56ac3c9c23612ec175cc8a` (also in `nsdof_outputs_v1.0.zip.sha256`) |

**Final check in a fresh virtual environment.** System Python 3.12.6, new venv, pinned packages from
`requirements.txt` (numpy, scipy, scikit-learn, pandas, matplotlib, pytest, ruff, vulture; Table 4 needs no
PyTorch), on a fresh copy of `github/` and the unzipped archive (`SCRATCH/final_check.sh`).

| Step | Result |
|---|---|
| `github/MANIFEST_sha256.csv` against the copy | 181 files, 0 mismatches (before 3 non-result files were removed; see section 2) |
| Unzip `nsdof_outputs_v1.0.zip`, check against its `MANIFEST_sha256.csv` | 897 files, 0 mismatches |
| `python scripts/download_data.py --checksum-only` | all 13 files verified |
| Regenerate Tables 3–4 from the archive (`hsi_analyze.py`, `hsi_summarize.py`) and compare with `results/q1` | `C3_delta_mean_sd.csv` (Table 4): identical. `C1_runs`, `C2_summary`, `B5_ensemble_metrics`, `C4_cross_model`, `C5_*`, `C6_old_vs_new_pca`, `B3_nsu_TF_shares`: identical. 2,051 rows compared; the only 6 that differ are the pooled Salinas rows of Open issue 3, which the dev-venv run reproduces exactly |
| `ruff check`, `ruff format --check`, `vulture --min-confidence 80`, `pytest` | all checks passed; 41 files formatted; 0 findings; 2 passed |
| Secret / path / e-mail scan of the copy and of the unzipped archive | 0 hits (after excluding URLs and the `hf_dataset` column names) |

The zip also contains `README_DATA.md` and `MANIFEST_sha256.csv` (path, size and SHA-256 of every file). The
upload metadata is in `zenodo/.zenodo.json`. The covariate-shift outputs (`hsi_outputs_revision/shift/`, 962 MB)
are included: without them Table S14 could only be rebuilt by GPU inference.

## 8. Open issues

1. **No version control.** The experiments ran in a working tree without git, so no commit ties the code to the
   stored outputs. The before/after comparison (section 6) is the substitute.
2. **Original data URLs.** The UPV/EHU site with the original Pavia University and Salinas files was unreachable.
   The Hugging Face mirrors `danaroth/pavia` and `danaroth/salinas` (pinned revisions) are byte-identical to the
   files used.
3. **Pooled Salinas rows differ from the stored tables.** The pooled detector fits a logistic regression on
   303,453 samples. The current code, the uncleaned originals run today and a fresh venv all agree with each
   other, but they differ from the stored q1 tables in these rows only (stored → now):
   - NS-U N2 − N1: AUC_upper 0.8589 → 0.8590, Δ +0.0004 → +0.0005, CI [+0.0003, +0.0005] → [+0.0004, +0.0006];
   - NS-U N2+ − N2base: AUC_lower 0.8083 → 0.8084;
   - softmax S2 − S1 and S3 − S2: bootstrap `p_gt0` only, 0.523 → 0.534 and 0.561 → 0.536 (Δ and CI unchanged);
   - exploratory: T_only − N2base Δ +0.0503 → +0.0501; N2 − T_only Δ +0.0002 → +0.0003;
     N2plus − N2 AUC_lower 0.8589 → 0.8590.

   The same rows carry over into `c14_all_comparisons.csv` (row 215) and `c15_threshold_sensitivity.csv`
   (row 219). No verdict changes, and no per-run row differs, so every cell of Table 4 is unchanged. Why the
   stored tables differ was not traced.
4. **Section 7.2 rests on two older reports.** The 48 comparisons count 12 (Qwen2.5-1.5B on orca-487 and on
   UF ≥ 4, 2,000 pairs) from `ns_logprob_report.json` and `ns_logprob_report_uf_gap4.json`. These were written
   by an earlier version of the detector that had no `length_covariate` block. Their common blocks are identical
   to the current code's output. The current code adds 6 covariate comparisons (54 in total), and none of them
   passes either, so "none of 48" also holds as "none of 54".
5. **Retrained networks are not bit-identical.** GPU training is non-deterministic: 20 of 70 replicas match the
   stored outputs bit for bit (Table S18). All analyses therefore read the stored outputs.
6. **npz timestamps.** `np.savez` writes zip timestamps, so regenerated `.npz` files (splits, outputs) differ in
   file hash while their arrays are identical.
7. **CIFAR-10 pretraining was not rerun.** It originally ran inside the NS-ME runner (`nsme/nsunc/runner.py`).
   The release reimplements it as `cifar10h_common.train_softmax`, equivalent by code inspection (section 6),
   but it was not rerun. The archived weights `weights_revision/cifar10_pretrain/` are the ones used.
8. **Table 4 † rows.** `t4_table4_original.csv` has 11 prespecified rows and **8** single-coordinate (B6) rows,
   but the manuscript map lists 7 † rows. Please check which B6 row is not printed, or whether the table should
   list 8.
9. **Section 6.4 sources assumed.** For "HybridSN degeneration", `CODE_MAP.md` points to
   `C1_runs.csv` / `C2_summary.csv` and the `degenerate_evidential_hybridsn` flag in `c14_all_comparisons.csv`.
   Please confirm that these are the numbers the text quotes.
10. **Sampler commands not recorded.** The exact `pref_sample_pairs.py` / `pref_sample_shp.py` command lines for
    the six large samples were not recorded. The samples are reproduced exactly from
    `preference_pair_ids.csv` instead (SHA-256 verified). The samplers write in text mode, so their line endings
    depend on the OS.
11. **Row order of two revision CSVs.** `engine_foldauc.csv` and `c12_fold_auc_diagnostic.csv` iterate over a
    Python `set`, so their row order follows hash randomisation (the contents are fixed). Set
    `PYTHONHASHSEED=0` for a fixed order.
12. **Fonts.** The figures were generated on Windows without Liberation Serif (fallback: DejaVu Sans). Their data
    is unchanged, but the typeface differs from the manuscript figures.
13. **Zenodo record.** The GitHub URL is in `.zenodo.json`; the reserved DOI 10.5281/zenodo.23726966 is in
    `README.md`. The record is a draft until the authors publish it on Zenodo.
