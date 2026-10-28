"""Revision analyses (post hoc), comments 21 and 13 (A1, A2): split files (pixel indices, labels,
16x16 block IDs),
their check against the stored runs, and the per-class coverage table (Table S6)."""

import hashlib
import os

import common as C
import numpy as np

os.makedirs(C.SPLITS_DIR, exist_ok=True)
man, cov, chk = [], [], []
for sc in C.SCENE_LIST:
    _, gt = C.load_scene(sc)
    K = int(gt.max())
    blk = C.block_ids(gt.shape)
    for s in range(3):
        tr, te = C.split_block_buffer(gt, C.FRAC, np.random.default_rng(s), buffer=C.BUFFER)
        ty, tx = np.where(tr)
        ey, ex = np.where(te)
        f = os.path.join(C.SPLITS_DIR, f"{sc}_split{s}.npz")
        np.savez_compressed(
            f,
            train_y=ty.astype(np.int16),
            train_x=tx.astype(np.int16),
            train_block=blk[tr].astype(np.int32),
            train_label=gt[tr].astype(np.int16),
            test_y=ey.astype(np.int16),
            test_x=ex.astype(np.int16),
            test_block=blk[te].astype(np.int32),
            test_label=gt[te].astype(np.int16),
            block_size=C.BLOCK,
            frac=C.FRAC,
            buffer=C.BUFFER,
            split_seed=s,
        )
        with open(f, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        n_blk_tr = len(np.unique(blk[tr]))
        man.append(
            [
                sc,
                s,
                os.path.basename(f),
                int(tr.sum()),
                int(te.sum()),
                n_blk_tr,
                len(np.unique(blk[te])),
                sha,
            ]
        )
        for sc2, m, s2, t, p in C.runs():
            if sc2 != sc or s2 != s:
                continue
            d = np.load(p)
            ok_n = int(d["n_train"]) == int(tr.sum())
            ok_c = np.array_equal(d["train_counts"], np.bincount(gt[tr] - 1, minlength=K))
            ok_xy = (
                np.array_equal(d["ys"], ey)
                and np.array_equal(d["xs"], ex)
                and np.array_equal(d["y"], gt[te] - 1)
            )
            chk.append([sc, m, s, t, ok_n, ok_c, ok_xy])
        for k in range(1, K + 1):
            mtr, mte = tr & (gt == k), te & (gt == k)
            cov.append(
                [
                    sc,
                    s,
                    k,
                    int((gt == k).sum()),
                    int(mtr.sum()),
                    int(mte.sum()),
                    len(np.unique(blk[mtr])),
                    len(np.unique(blk[mte])),
                    "ZERO_TRAIN" if mtr.sum() == 0 else "",
                ]
            )

C.write_csv(
    C.REV / "c21_split_manifest.csv",
    [
        "scene",
        "split_seed",
        "file",
        "n_train_px",
        "n_test_px",
        "n_train_blocks",
        "n_test_blocks",
        "sha256",
    ],
    man,
)
C.write_csv(
    C.REV / "c21_split_check_vs_saved_runs.csv",
    [
        "scene",
        "model",
        "split_seed",
        "train_seed",
        "n_train_match",
        "train_counts_match",
        "test_coords_labels_match",
    ],
    chk,
)
C.write_csv(
    C.REV / "c13_class_coverage.csv",
    [
        "scene",
        "split_seed",
        "class",
        "n_labelled_px",
        "n_train_px",
        "n_test_px",
        "n_train_blocks",
        "n_test_blocks",
        "flag",
    ],
    cov,
)
print(f"runs checked: {len(chk)}, mismatches: {sum(not all(r[4:]) for r in chk)}")
