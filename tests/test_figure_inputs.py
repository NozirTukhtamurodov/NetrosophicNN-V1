"""Figure inputs must reproduce the manuscript values (4 decimals)."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(a, b):
    return round(float(a), 4) == round(float(b), 4) or abs(float(a) - float(b)) < 5e-5


def test_nsu_sum_by_run():
    got = {
        (r["scene"], r["split_seed"], r["train_seed"]): r
        for r in read(ROOT / "results/q1/nsu_sum_by_run.csv")
    }
    exp = read(HERE / "expected_nsu_sum_by_run.csv")
    assert len(got) == len(exp)
    for e in exp:
        g = got[(e["scene"], e["split_seed"], e["train_seed"])]
        assert close(g["share_TplusF_gt_1"], e["share_TplusF_gt_1"])
        assert close(g["mean_TplusF"], e["mean_TplusF"])


def test_preference_steps():
    got = read(ROOT / "results/preference/preference_steps.csv")
    exp = read(HERE / "expected_preference_steps.csv")
    assert [(g["panel"], g["label"]) for g in got] == [(e["panel"], e["label"]) for e in exp]
    for g, e in zip(got, exp):
        for k in ("delta", "ci_lo", "ci_hi"):
            assert close(g[k], e[k]), (e["label"], k, g[k], e[k])
