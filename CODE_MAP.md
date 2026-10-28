# Code map

Each table, figure and number quoted in the paper, with the script, function, inputs and output that produce it.

**How to read the paths.**
- Scripts in `code/` are run from `code/`. Scripts in `code/simulation/` and `code/figures/` are run from the
  repository root.
- `outputs/` is the unzipped Zenodo archive (`NSDOF_OUTPUTS_DIR`), `data/` holds the third-party data
  (`NSDOF_DATA_DIR`), and `results/` is `NSDOF_RESULTS_DIR`.
- "Function" names the function that writes the output; `<module>` means module level.

**Regenerating vs. retraining.** Every analysis entry point reads stored network outputs, so every number below
can be regenerated without a GPU. Training is listed separately in each row, and its results are not
bit-reproducible (see `ENVIRONMENT.md`).

## Main text: tables

| Result | Script | Function | Inputs | Output |
|---|---|---|---|---|
| Table 1 | — (static table, no code) | | | |
| Table 2: settings S1–S3, n = 20,000 | `code/simulation/simulation.py` | `main` | none (seeded) | `results/simulation/sim_results.json` |
| Table 3: OA/AA/κ, HSINet softmax, evidential, NS-U (9 runs each), HybridSN softmax and evidential (3 runs each) | training: `code/hsi_run_q1.py` → `code/hsi_train_q1.py`. Analysis: `code/hsi_analyze.py`, then `code/hsi_summarize.py` | `hsi_train_q1.main`; `hsi_analyze.main`; `hsi_summarize.main` | `data/hsi/*.mat`; `outputs/hsi_outputs_q1/*.npz` | `outputs/hsi_outputs_q1/*.npz`; `results/q1/C1_runs.csv` (per run); `results/q1/C2_summary.csv` (mean, SD) |
| Table 3: deep ensemble (split 0) | `code/hsi_analyze.py` | `main` | `outputs/hsi_outputs_q1/*_softmax_s0_t*.npz` | `results/q1/B5_ensemble_metrics.csv` |
| Table 4: 11 prespecified rows (166 per-run comparisons + pooled) | `code/hsi_analyze.py`, then `code/hsi_summarize.py` | `hsi_analyze.main`; `hsi_summarize.main` | `outputs/hsi_outputs_q1/*.npz` | `results/q1/C3_aucs.csv`, `results/q1/C3_deltas.csv` (per run and pooled); `results/q1/C3_delta_mean_sd.csv` |
| Table 4: † rows (single-coordinate steps; revision, post hoc) | `code/rev_engine.py`, then `code/rev_t4_tables.py` | `<module>` | `outputs/hsi_outputs_q1/*.npz` | `results/revision/csv/engine_steps.csv`; `results/revision/csv/t4_table4_original.csv` (11 rows + B6 rows), `t4_tables.md`; per run (via `rev_tables.py`): `c10_11_single_coordinate_steps.csv`, `c10_evidential_binary_frame_aucs.csv` |
| Table 4: check that the engine reproduces the published rows | `code/rev_tables.py` | `<module>` | `engine_steps.csv`, `results/q1/C3_deltas.csv` | `results/revision/csv/check_engine_vs_table4.txt` |
| Table 5: preference detection, orca, UF ≥ 4, UF ≥ 2, HH-RLHF; length filter and length covariate | pairs: `scripts/download_data.py` (or `code/pref_sample_pairs.py`). Scores: `code/pref_score.py`. Detector: `code/pref_detect.py` | `rebuild_pairs`; `pref_score.main`; `pref_detect.main` (`block`) | `preference_scores/preference_pair_ids.csv`; `preference_scores/ns_logprob_{orca_big, ultrafeedback_gap4big, ultrafeedback_gap2big, hh_rlhf_big}.jsonl` | `results/preference/rep_{orca_big, ultrafeedback_gap4big, ultrafeedback_gap2big, hh_rlhf_big}.json` |
| Table 5: SD over 5 fold assignments | `code/pref_fold_variability.py` | `main` | the score files above and the stored reports | `results/preference/D1_fold_seed_variability.csv` |
| Table 6: SHP near ties | pairs: `scripts/download_data.py` (or `code/pref_sample_shp.py`). Scores: `code/pref_score.py`. Detector: `code/pref_detect_shp.py` | `pref_detect_shp.main` | `preference_scores/ns_logprob_shp.jsonl`; `preference_pair_ids.csv` (vote ratio) | `results/preference/rep_shp_natural.json`; SD in `D1_fold_seed_variability.csv` |

## Main text: figures

| Result | Script | Function | Inputs | Output |
|---|---|---|---|---|
| Fig. 1: S3 image and Guo–Cheng T, I, F maps | `code/simulation/simulation.py`, then `code/figures/fig1_simulation.py` | `main` | `results/simulation/s3_maps.npz` | `figures/fig1_simulation.png` |
| Fig. 2a: NS-U mean T + F vs share of test pixels with T + F > 1 | `code/figures/make_figure_inputs.py`, then `code/figures/fig2_fig3_hsi.py` | `hsi_inputs`; `main` | `results/q1/B3_nsu_TF_shares.csv` → `results/q1/nsu_sum_by_run.csv` | `figures/fig2_nsu.png` |
| Fig. 2b: exploratory decomposition | `code/hsi_explore_nsu.py` → `make_figure_inputs.py` → `fig2_fig3_hsi.py` | `main`; `hsi_inputs`; `main` | `outputs/hsi_outputs_q1/*_nsu_*.npz` → `results/q1/X_nsu_exploratory.csv` → `results/q1/nsu_explore.csv` | `figures/fig2_nsu.png` |
| Fig. 3: per-run ΔAUC | `make_figure_inputs.py` → `fig2_fig3_hsi.py` | `hsi_inputs`; `main` | `results/q1/C3_deltas.csv` → `results/q1/ladders_runs.csv` | `figures/fig3_ladders.png` |
| Fig. 4: preference steps | `make_figure_inputs.py` → `code/figures/fig4_preference.py` | `preference_inputs`; `main` | `results/preference/rep_*.json` → `results/preference/preference_steps.csv` | `figures/fig4_preference.png` |

`tests/test_figure_inputs.py` checks `nsu_sum_by_run.csv` and `preference_steps.csv` against the manuscript values
to 4 decimals (`pytest tests`).

## Main text: numbers quoted in the text

| Result | Script | Function | Inputs | Output |
|---|---|---|---|---|
| Section 4: detector (StandardScaler + LR, C = 1, 5-fold out-of-fold) and paired bootstrap (B = 3000, seed 1337) | `code/common.py` | `oof_scores`, `boot_multi`, `boot_dist`, `summ`, `rule` | — | used by every analysis |
| Section 4: point and bootstrap-mean rules agree in all 224 comparisons | `code/hsi_analyze.py` | `main` | `outputs/hsi_outputs_q1/*.npz` | `results/q1/C3_deltas.csv`, columns `pass_point_rule` = `pass_bootmean_rule` in 224 of 224 rows (196 per run + 28 pooled) |
| Section 4: list of the 224 comparisons | `code/rev_a3_accounting.py` | `<module>` | `results/q1/C3_deltas.csv` | `results/revision/csv/c14_all_comparisons.csv` |
| Section 5: error rate 0.245 (S1), positive rate 0.350 (S2), rank 3 | `code/simulation/simulation.py` | `main` | none | `sim_results.json`: `S1.error_rate` 0.2451, `S2.positive_rate` 0.3499, `S1.rank_1_T_F_I` 3 |
| Section 6.4: best rung per model | `code/hsi_summarize.py` | `main` | `results/q1/C3_aucs.csv` | `results/q1/C4_cross_model.csv` |
| Section 6.4: exploratory NS-U decomposition | `code/hsi_explore_nsu.py` | `main` | `outputs/hsi_outputs_q1/*_nsu_*.npz` | `results/q1/X_nsu_exploratory.csv` |
| Section 6.4: degeneration of evidential HybridSN | `code/hsi_analyze.py`, `code/hsi_summarize.py` | `main` | `outputs/hsi_outputs_q1/*_hybrid_evidential_*.npz` | `results/q1/C1_runs.csv`, `C2_summary.csv`; flagged in `c14_all_comparisons.csv` (`degenerate_evidential_hybridsn`) |
| Section 6.4: classes with no training pixels | `code/hsi_summarize.py` | `main` | `results/q1/C5_perclass.csv` | `results/q1/C5_classes_no_training.csv` |
| Section 6.5: transductive preprocessing vs train-only PCA | training: `code/hsi_train_transductive.py`. Analysis: `code/hsi_transductive_audit.py`, then `code/hsi_summarize.py` | `main` | `outputs/hsi_outputs_transductive/*.jsonl`; `results/q1/C2_summary.csv`, `C3_deltas.csv` | `results/q1/transductive_a.json`, `transductive_b6.json`; `results/q1/C6_old_vs_new_pca.csv` |
| Section 7.1 / Table S16: preference data flow | `code/rev_a5_flow.py` | `<module>` | `data/preference/*.jsonl`; `preference_scores/*.jsonl` | `results/revision/csv/c20_*.csv` |
| Section 7.2: 48 comparisons, three scorers, none passes | scores: `code/pref_score.py --model …`. Detector: `code/pref_detect.py`. Count: `code/pref_scorer_robustness.py` | `main` | scores `preference_scores/ns_logprob{,_orca__qwen05,_orca__smol17,_ultrafeedback_gap4,_ultrafeedback_gap4__qwen05,_ultrafeedback_gap4__smol17}.jsonl` → reports `results/preference/ns_logprob_report.json`, `ns_logprob_report_uf_gap4.json`, `rep_orca__{qwen05,smol17}.json`, `rep_ultrafeedback_gap4__{qwen05,smol17}.json` | `results/preference/scorer_robustness.csv` (48 rows, 0 pass) |

## Supplement (revision analyses, post hoc)

The tables are formatted in `results/revision/REVISION_REPORT.md` from the CSV files below. Run
`rev_engine.py` before `rev_tables.py`, `rev_b4_pooled.py`, `rev_b4_b6steps.py` and `rev_t4_tables.py`.

| Result | Script | Function | Inputs | Output (`results/revision/csv/`) |
|---|---|---|---|---|
| Table S1: Δ thresholds | `rev_engine.py` → `rev_tables.py` → `rev_t4_tables.py` | `<module>` | `outputs/hsi_outputs_q1` | `t4_B2_thresholds.csv`, `c15_threshold_sensitivity.csv`, `c15_threshold_counts.csv` |
| Table S2: p-values, Benjamini–Hochberg | `rev_tables.py`, `rev_t4_tables.py` | `<module>` | `engine_steps.csv` | `c14_pvalues_bh.csv`, `c14_bh_counts.csv`, `c14_delta_by_level.csv` (ΔAUC by split and seed), `t4_B3_BH.csv`, `t4_B3_pvalues_per_run.csv` |
| Table S3: spatial blocking | `rev_engine.py`, `rev_b4_pooled.py`, `rev_b4_b6steps.py` → `rev_tables.py` → `rev_t4_tables.py` | `<module>` | `outputs/hsi_outputs_q1` | `c12_blocked_vs_original.csv`, `c12_blocked_vs_original_pooled.csv`, `c12_blocked_counts.csv`, `c12_fold_auc_diagnostic.csv` (from `engine_foldauc.csv`), `engine_steps_b4_b6.csv`, `t4_B4_spatial.csv` |
| Table S4: detector regularisation C | `rev_engine.py` → `rev_tables.py` → `rev_t4_tables.py` | `<module>` | `engine_steps.csv` | `c09_C_sensitivity.csv`, `c09_design_rank.csv`, `t4_B5_C_sensitivity.csv` |
| Table S5: training-absent classes removed | `rev_tables.py` → `rev_t4_tables.py` | `<module>` | `engine_steps.csv` | `c13_absent_classes_removed.csv`, `t4_B7_unseen_classes.csv` |
| Table S6: class coverage of every scene × split | `rev_a1_a2_splits.py` | `<module>` | `data/hsi/*.mat` | `c13_class_coverage.csv`; also `splits/*.npz`, `c21_split_manifest.csv`, `c21_split_check_vs_saved_runs.csv` |
| Table S7: NS-U residual T + F − 1 | `rev_b1_residuals.py` | `<module>` | `outputs/hsi_outputs_q1/*_nsu_*.npz` | `c07_residuals_by_run.csv`, `c07_points_F_vs_1minusT.csv` |
| Table S8: AUC, AUPRC, AURC per rung | `rev_engine.py` → `rev_tables.py` | `<module>` | `engine_rungs.csv` | `c15_auprc_aurc_summary.csv`, `c15_auprc_aurc_by_run.csv`, `c15_risk_coverage_curves.csv` |
| Table S9: calibration (NLL, Brier, ECE) | `rev_calibration_evidence.py` | `main` | `outputs/hsi_outputs_q1` | `c17_calibration_summary.csv`, `c17_calibration_by_run.csv` |
| Table S10: evidential diagnostics | `rev_calibration_evidence.py` | `main` | `outputs/hsi_outputs_q1/*evidential*.npz` | `c16_evidential_evidence_by_run.csv`, `c16_evidential_evidence_by_class.csv`, `c16_perclass_accuracy_evidential_vs_softmax.csv`; also `engine_evid_sums.csv` (`rev_engine.py`) |
| Table S11: head refit on frozen NS-U features | training: `rev_run_train.py` → `rev_train.py --variant replica`. Analysis: `rev_c1a_heads.py` | `<module>` | `outputs/weights_revision/replica/*.pt` | `c06_head_refit_by_run.csv` |
| Table S12: NS-U with explicit head penalty | training: `rev_train.py --variant nsu_pen --lam …`. Analysis: `rev_c_analyse.py c1b` | `<module>` | `outputs/hsi_outputs_revision/nsu_pen_lam*/` | `c05_06_penalised_nsu_by_run.csv`, `c05_06_penalised_nsu_ladder.csv` |
| Table S13: alternative evidential objectives | training: `rev_train.py --variant evid_log` and `evid_mse_anneal20`. Analysis: `rev_c_analyse.py c2` | `<module>` | `outputs/hsi_outputs_revision/evid_*/` | `c16_alt_evidential_by_run.csv`, `c16_alt_evidential_ladder.csv`, `c16_alt_evidential_perclass_acc.csv` |
| Table S14: covariate shift | `rev_c3_shift.py gpu` (network outputs under shift; stored in the archive), then `cpu`, then `summary` | `summary` | `gpu`: `outputs/weights_revision/replica/*.pt`, `data/hsi`. `cpu`/`summary`: `outputs/hsi_outputs_revision/shift/` | `c17_shift_summary.csv`, `c17_shift_ladder_counts.csv`, `c17_shift_calibration.csv`, `c17_shift_ladders.csv` |
| Table S15: OOD detection of training-absent classes | `rev_c3_shift.py ood` | `ood` | `outputs/hsi_outputs_q1` | `c17_ood_auroc.csv` |
| Table S16: preference data flow | `rev_a5_flow.py` | `<module>` | `data/preference`, `preference_scores` | `c20_flow_by_source.csv`, `c20_exclusions.csv`, `c20_ultrafeedback_overlap.csv`, `c20_shp_selection.csv` |
| Table S17: CIFAR-10H positive control | `cifar10h_positive_control.py train`, then `analyse` (plan: `plan/D_plan.md`) | `train_all`; `analyse` | `data/cifar10h`, `data/torchvision`; `outputs/weights_revision/cifar10_pretrain` | `outputs/cifar10h_outputs/*.npz`; `c08_cifar10h_positive_control.csv` |
| Table S18: fidelity of the retrained networks | `rev_c_analyse.py fidelity` | `fidelity` | `outputs/hsi_outputs_revision/replica`, `outputs/hsi_outputs_q1` | `c21_replica_fidelity.csv` |
| Fig. S1: F vs 1 − T, by scene | `rev_b1_residuals.py` | `<module>` | as Table S7 | `results/revision/figures/c07_F_vs_1minusT_by_scene.png` |
| Figs. S2a, S2b: by split and seed | `rev_b1_residuals.py` | `<module>` | as Table S7 | `figures/c07_F_vs_1minusT_{pavia_u,salinas}_by_seed.png` |
| Figs. S3a, S3b: by predicted class | `rev_b1_residuals.py` | `<module>` | as Table S7 | `figures/c07_F_vs_1minusT_{pavia_u,salinas}_by_class.png` |
| Figs. S4a, S4b: risk–coverage curves | `rev_tables.py` | `<module>` | `c15_risk_coverage_curves.csv` | `figures/c15_risk_coverage_{pavia_u,salinas}.png` |
| Fig. S5: training loss with head penalty | `rev_c_analyse.py c1b` | `<module>` | `outputs/hsi_outputs_revision/nsu_pen_lam*/` | `figures/c05_06_penalised_nsu_loss_curves.png` |

## Not regenerated by this code

| File | Origin |
|---|---|
| `results/revision/REVISION_REPORT.md` | report written during the revision from the CSV files above; formatting text, not computed |
| `results/preference/ns_logprob_report.json`, `ns_logprob_report_uf_gap4.json` | stored reports of an earlier detector version (no `length_covariate` block); used for Section 7.2 (see `RELEASE_NOTES.md`, Open issues) |
