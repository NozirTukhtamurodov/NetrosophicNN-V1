# Environment

## Machine used for every result

| Component | Value |
|---|---|
| OS | Windows 11 Pro 10.0.22631 |
| Python | 3.12.6 |
| GPU | NVIDIA GeForce RTX 4060, 8 GB, driver 616.56, CUDA 13.2 |
| PyTorch | 2.13.0+cu132, torchvision 0.28.0+cu132 |
| Other packages | `requirements.txt` (pinned) |

## Setup

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
python scripts/download_data.py  # data into ./data, every file checked against its SHA-256
```

`python scripts/download_data.py --checksum-only` only verifies files that are already present.
Pass `--skip-preference` if you do not need the preference pairs. Rebuilding them loads four Hugging Face
datasets, about 3 GB of downloads.

## Directories (environment variables)

All paths are relative to the repository root unless one of these variables is set.

| Variable | Default | Contents |
|---|---|---|
| `NSDOF_DATA_DIR` | `data/` | third-party data: `hsi/*.mat`, `cifar10h/`, `torchvision/`, `preference/*.jsonl` |
| `NSDOF_OUTPUTS_DIR` | `outputs/` | the unzipped Zenodo archive (`hsi_outputs_q1/`, `hsi_outputs_revision/`, `weights_revision/`, `cifar10h_outputs/`, `hsi_outputs_transductive/`) |
| `NSDOF_RESULTS_DIR` | `results/` | tables, reports and figure inputs written by the analysis scripts |
| `NSDOF_SPLITS_DIR` | `splits/` | the six block-split files |

The scripts in `code/` import `common.py` from the same folder. Run them from `code/`,
or add `code/` to `PYTHONPATH`.

## Run times on the machine above

| Step | Time |
|---|---|
| 70 prespecified hyperspectral runs (`hsi_run_q1.py`, GPU) | 2.2 h |
| Revision training: 70 replicas, 72 penalised NS-U, 36 alternative evidential (`rev_run_train.py`, GPU) | 1.7 h |
| Preference scoring, Qwen2.5-1.5B-Instruct, one sample of 12,000 to 15,000 pairs (`pref_score.py`, GPU) | 1–2 h |
| `hsi_analyze.py` (Tables 3–4) | 7 min |
| `hsi_transductive_audit.py` (Section 6.5) | 25 min |
| `rev_engine.py` / `rev_b4_pooled.py` | 18 min / 15 min |
| One preference detector report (`pref_detect.py`) | 1–3 min |
| `simulation/simulation.py` (Table 2) | 2 min |

Every analysis script runs on the CPU. Only training and preference scoring need the GPU.

## Determinism

- **Analyses (CPU).** Analysis outputs are bit-identical between runs, with one exception: a few pooled
  Salinas rows of `C3_aucs.csv`, `C3_deltas.csv` and `X_nsu_exploratory.csv` (NS-U AUCs and deltas, softmax
  bootstrap `p_gt0`). They fit a logistic regression on 303,453 samples and differ from the stored tables in
  the 4th decimal, although today's runs agree with each other (`RELEASE_NOTES.md`, Open issue 3). Rows written by iterating over a
  Python `set` (`engine_foldauc.csv`, `c12_fold_auc_diagnostic.csv`) come out in hash-randomised order; their
  contents are identical. Set `PYTHONHASHSEED=0` for a fixed order.
- **GPU training is not bit-reproducible.** 20 of the 70 retrained replicas are bit-identical to the stored
  outputs (Table S18). All analyses therefore read the stored network outputs from the Zenodo archive and do
  not retrain.
- **npz archives.** `np.savez` stores zip timestamps, so the SHA-256 of a newly written `.npz` differs even when
  its arrays are identical. Compare the arrays, not the file hash.

## Fonts

The figures use Liberation Serif (`fonts-liberation` on Debian/Ubuntu). If the font is missing, matplotlib falls
back to DejaVu Sans; the data in the figures does not change.
