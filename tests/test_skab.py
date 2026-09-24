"""
Tests for the SKAB benchmark harness (app/services/anomaly/skab.py).

    python tests/test_skab.py
    SKAB_DIR=path/to/SKAB python tests/test_skab.py     # + reproduce the published leaderboard numbers

Without the dataset, a small synthetic SKAB-format folder is generated.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.anomaly import skab  # noqa: E402


def _fake_skab(root: str, seed: int = 0) -> str:
    """3 fault families x 2 experiments, 8 sensors, a 120-row level shift in 2 sensors, plus a slow temperature drift."""
    rng = np.random.default_rng(seed)
    for gi, (g, sensors) in enumerate((("valve1", (7,)), ("valve2", (7, 2)), ("other", (0, 1)))):
        os.makedirs(os.path.join(root, g), exist_ok=True)
        for k in range(2):
            n = 800
            X = rng.normal(size=(n, 8)) * 0.1 + np.arange(8)
            X[:, 4] += np.linspace(0, 1.5, n)            # nuisance drift on "Temperature"
            y = np.zeros(n, int)
            a = 520 + 20 * k
            y[a:a + 150] = 1
            for j in sensors:
                X[a:a + 150, j] += 1.0
            with open(os.path.join(root, g, f"{k}.csv"), "w") as f:
                f.write("datetime;" + ";".join(skab.SENSORS) + ";anomaly;changepoint\n")
                for i in range(n):
                    f.write(f"2020-03-{10 + 2 * gi + k:02d} {10 + i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d};"
                            + ";".join(f"{v:.5f}" for v in X[i]) + f";{y[i]};0\n")
    os.makedirs(os.path.join(root, "anomaly-free"), exist_ok=True)
    return root


def test_metrics_match_skab_formulas():
    t = [np.array([0, 0, 1, 1, 1, 0, 0, 0])]
    p = [np.array([1, 0, 0, 1, 1, 1, 0, 1])]
    m = skab.metrics(p, t)
    TP, FP, FN, TN = 2, 3, 1, 2
    assert m["f1"] == round(TP / (TP + (FN + FP) / 2), 3)
    assert m["far"] == round(100 * FP / (FP + TN), 2) and m["mar"] == round(100 * FN / (FN + TP), 2)
    assert m["faults_caught"] == 1 and m["median_delay_s"] == 1
    assert m["false_alarm_episodes"] == 2     # the run at row 0 and the run at row 7; row 5 touches the fault run


def test_persist_and_smoothing():
    f = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool)
    assert skab.persist(f, 3).astype(int).tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    assert np.allclose(skab.causal_mean(np.array([2.0, 4, 6, 8]), 2), [2, 3, 5, 7])


def test_everything_is_causal():
    """Changing the future must not change any earlier score (drift-robust input, point and window models)."""
    root = _fake_skab(tempfile.mkdtemp())
    e = skab.load(root)[0]
    e2 = skab.Experiment(e.name, e.group, e.X.copy(), e.y)
    e2.X[650:] += 5.0
    assert np.allclose(skab.drift_robust(e.X, 120)[:650], skab.drift_robust(e2.X, 120)[:650])
    for key in ("pca", "dense_ae"):
        a = skab.score_file(skab.DETECTORS[key], e, 120).test
        b = skab.score_file(skab.DETECTORS[key], e2, 120).test
        cut = 650 - skab.TRAIN_ROWS
        assert np.allclose(a[:cut], b[:cut]), key


def test_grouped_cv_never_sees_the_held_out_labels():
    root = _fake_skab(tempfile.mkdtemp())
    exps = skab.load(root)
    views = {v: [skab.score_file(skab.PCAT2Q, e, tau) for e in exps] for v, tau in skab.VIEWS.items()}
    _, chosen = skab.tune_cv(views, exps)
    for e in exps:                       # scramble the labels of one family
        if e.group == "other":
            e.y[:] = np.random.default_rng(0).integers(0, 2, len(e.y))
    _, chosen2 = skab.tune_cv(views, exps)
    assert chosen["other"] == chosen2["other"]


def test_end_to_end_on_synthetic_skab():
    root = _fake_skab(tempfile.mkdtemp())
    res = skab.run(root, models=["fixed", "pca", "dense_ae"], with_ark=True, with_supervised=True, log=lambda *_: None)
    rows = {r["key"]: r for r in res["rows"]}
    assert {"fixed", "fixed+", "pca", "pca+", "ensemble+", "supervised", "supervised_lfo", "supervised_past",
            "hybrid", "hybrid_lfo", "hybrid_past", "ark"} <= set(rows)
    # the chronologically first experiment has no past to learn from: the supervised tier stays silent there
    assert rows["hybrid_past"]["tiers"]["urgent_rows"] < rows["hybrid"]["tiers"]["urgent_rows"]
    # the drift-robust improved PCA should raise far fewer false alarms than the plain fixed limit
    assert rows["pca+"]["false_alarm_episodes_per_h"] < rows["fixed"]["false_alarm_episodes_per_h"]
    assert rows["supervised"]["f1"] > 0.8
    md = skab.markdown(res)
    assert "Published SKAB leaderboard" in md


def test_overlapping_recordings_are_held_out_together():
    mk = lambda n, a, b: skab.Experiment(n, "other", np.random.default_rng(int(n[-1])).normal(size=(500, 8)),
                                         np.zeros(500, int), a, b)
    exps = [mk("other/1", "2020-01-01 10:00:00", "2020-01-01 10:10:00"),
            mk("other/2", "2020-01-01 10:09:00", "2020-01-01 10:20:00"),     # overlaps other/1
            mk("other/3", "2020-01-02 10:00:00", "2020-01-02 10:10:00")]
    exps[2].X[:50] = exps[0].X[:50]                                          # shares identical rows with other/1
    assert skab.overlap_partners(exps) == [{1, 2}, {0}, {0}]


def test_reproduces_published_leaderboard():
    """With the real SKAB repo: our metric code, applied to the leaderboard's own saved predictions,
    must give the published numbers (e.g. Conv-AE F1 0.78, FAR 13.55 %, MAR 28.02 %)."""
    root = os.environ.get("SKAB_DIR")
    if not root or not os.path.isdir(os.path.join(root, "results")):
        return
    import glob
    import pickle

    import pandas as pd
    truth = {}
    for f in glob.glob(os.path.join(root, "data", "*", "*.csv")):
        if "anomaly-free" in f:
            continue
        df = pd.read_csv(f, sep=";", index_col="datetime", parse_dates=True).iloc[skab.TRAIN_ROWS:]
        truth[df.index[0]] = df["anomaly"].values
    published = {"Conv_AE": (0.78, 13.55, 28.02), "T2-q": (0.76, 26.62, 24.92), "LSTM_AE": (0.74, 29.96, 25.92),
                 "Isolation_Forest": (0.29, 2.56, 82.89), "MSET": (0.78, 39.73, 14.13)}
    for name, (f1, far, mar) in published.items():
        P = pickle.load(open(os.path.join(root, "results", f"results-{name}.pkl"), "rb"))
        m = skab.metrics([np.nan_to_num(np.asarray(p.values, float)) > 0.5 for p in P],
                         [truth[pd.Timestamp(p.index[0])] for p in P])
        assert (round(m["f1"], 2), m["far"], m["mar"]) == (f1, far, mar), (name, m)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
