# Revised tables in the format of Table 4 (revision analyses, post hoc)

Rows 1–11 = Table 4 (primary family, 166 per-run comparisons: 162 CNN + 4 ensemble), in the manuscript order. Rows below are the single-coordinate steps of B6, marked "revision, post hoc". HybridSN is secondary and not shown (its rows are in the per-run CSVs).

### Table 4 (original rule (10), Δ > 0.03 and CI_lo > 0) with the B6 rows appended

Cell: mean ΔAUC (SD over runs); runs passing. Recomputed by the revision engine (equals q1 within rounding).

| Model | Step | Adds | Type | Row source | Pavia University | Salinas |
|---|---|---|---|---|---|---|
| Softmax | S2 − S1 | p_max | nested | Table 4 | +0.008 (0.009); 0/9 | +0.000 (0.002); 0/9 |
| Softmax | S3 − S2 | p_(2) | nested | Table 4 | -0.002 (0.007); 0/9 | -0.000 (0.003); 0/9 |
| MC dropout | M2 − M1 | MI | nested | Table 4 | +0.029 (0.027); 6/9 | -0.000 (0.001); 0/9 |
| MC dropout | M3 − M2 | p_max | nested | Table 4 | +0.011 (0.053); 1/9 | +0.002 (0.004); 0/9 |
| Ensemble | E2 − E1 | MI | nested | Table 4 | +0.035; 1/1 | +0.000; 0/1 |
| Ensemble | E3 − E2 | p_max | nested | Table 4 | -0.008; 0/1 | -0.007; 0/1 |
| Evidential | R2 − R1 | u | nested | Table 4 | +0.021 (0.025); 2/9 | +0.001 (0.002); 0/9 |
| Evidential | R3+ − R2 | (b_max, b_(2)) | nested, 2-D (combined increment) | Table 4 | +0.003 (0.005); 0/9 | +0.001 (0.003); 0/9 |
| Evidential | R3 − R2 | triple vs pair | head-to-head | Table 4 | -0.019 (0.027); 0/9 | +0.001 (0.002); 0/9 |
| NS-U | N2 − N1 | F given T − F | nested | Table 4 | +0.010 (0.060); 1/9 | +0.012 (0.017); 2/9 |
| NS-U | N2+ − N2base | (T, F) | nested, 2-D (combined increment) | Table 4 | +0.130 (0.076); 8/9 | +0.072 (0.057); 7/9 |
| Evidential | (H,u,b_max) − R2 | b_max | nested, 1-D | revision, post hoc | +0.002 (0.006); 0/9 | +0.001 (0.002); 0/9 |
| Evidential | R3+ − (H,u,b_max) | b_(2) given b_max | nested, 1-D | revision, post hoc | +0.001 (0.001); 0/9 | +0.000 (0.000); 0/9 |
| Evidential | (H,u,b_(2)) − R2 | b_(2) | nested, 1-D | revision, post hoc | +0.002 (0.005); 0/9 | +0.001 (0.002); 0/9 |
| Evidential | BF − R2 | binary-frame triple (b_max, 1−u−b_max, u) vs (H,u) | head-to-head | revision, post hoc | -0.020 (0.028); 0/9 | +0.001 (0.002); 0/9 |
| Evidential | BF − (b_max,u) | b_other = 1−u−b_max | nested, affine-dependent | revision, post hoc | -0.000 (0.000); 0/9 | +0.000 (0.000); 0/9 |
| NS-U | N2base+T − N2base | T | nested, 1-D | revision, post hoc | +0.121 (0.075); 8/9 | +0.062 (0.063); 6/9 |
| NS-U | N2+ − (N2base+T) | F given T | nested, 1-D | revision, post hoc | +0.009 (0.062); 1/9 | +0.010 (0.017); 2/9 |
| NS-U | N2base+F − N2base | F | nested, 1-D | revision, post hoc | +0.117 (0.079); 7/9 | +0.061 (0.065); 6/9 |

### B2 (comment 15): verdict at Δ thresholds 0.01 / 0.02 / 0.03 / 0.05 (always with CI_lo > 0)

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

### B3 (comment 14): two-sided bootstrap p-values and Benjamini–Hochberg (q = 0.05)

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

### B4 (comment 12): original vs spatially blocked detector and bootstrap

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

### B5 (comment 9): detector regularisation C

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

### B7 (comment 13): test pixels of training-absent classes removed

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
