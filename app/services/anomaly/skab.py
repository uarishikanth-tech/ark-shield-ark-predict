"""
SKAB benchmark: fixed-limit alarms vs 8 ML / deep-learning models vs ARK Predict
on REAL labelled industrial sensor data.

SKAB (Skoltech Anomaly Benchmark, github.com/waico/SKAB) is a water-pump test rig
with 8 sensors (2 accelerometers, motor current, pressure, 2 temperatures,
voltage, flow rate), 1 row per second, 34 experiments, each with one real fault
(partly closed valves, cavitation, rotor imbalance, ...) labelled by the authors.

Protocol = the official SKAB leaderboard, so numbers compare 1:1 with the published ones:
  * each file on its own; the first 400 rows (healthy start) train the model, later rows are test
  * point-wise confusion matrix pooled over all 34 test sets
  * F1, FAR (false-alarm rate, % of normal rows flagged), MAR (missed-alarm rate, % of faulty rows missed)
Plus what an operator feels: separate false-alarm episodes per hour, faults caught, detection delay.
``tests/test_skab.py`` checks that our metric code reproduces the published leaderboard numbers.

Three things we add on top of the plain models ("improved" rows):
  1. drift-robust inputs — the two temperature channels wander slowly for reasons that have nothing
     to do with faults (room / fluid warming). We feed the models each temperature's deviation from its
     own slow moving average (time constant tau) instead of its raw level.
  2. alarm logic — a raw anomaly score becomes an alarm through a limit (quantile of healthy scores x
     factor), optional smoothing, and debouncing (N rows in a row above the limit).
  3. honest tuning — tau and the alarm logic are chosen by *grouped cross-validation* over the three
     fault families (valve1 / valve2 / other): settings for a family are picked on the other two only.

And a supervised model that learns what faults look like from other labelled experiments (gradient boosting),
checked three ways: leave-one-experiment-out (time-overlapping recordings held out together), a whole fault
family unseen in training, and the realistic one — trained only on experiments recorded *before* the one scored.

Honesty notes: the drift-robust time constants (120 s / 600 s) and the supervised model's 10-s smoothing were
picked during exploration on the full dataset (small effect: about ±0.01 F1).

    git clone https://github.com/waico/SKAB            # 4 MB, once
    python -m app.services.anomaly.skab SKAB/data                         # everything (~15 min, conv-AE is the slow one)
    python -m app.services.anomaly.skab SKAB/data --fast                  # skip the conv autoencoder (~4 min)
    python -m app.services.anomaly.skab SKAB/data --json results.json --md results.md

All models are causal: a row is scored only from rows up to it, exactly like the live stream.
(Some leaderboard entries flag a row using rows that come after it.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from .deepnp import ConvAutoencoder, LSTMAutoencoder, windows

TRAIN_ROWS = 400
SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature",
           "Thermocouple", "Voltage", "Volume Flow RateRMS"]
SLOW_CHANNELS = (4, 5)            # Temperature, Thermocouple
VIEWS = {"raw": None, "drift-robust τ=120 s": 120, "drift-robust τ=600 s": 600}

# Published SKAB leaderboard (outlier detection; F1, FAR %, MAR %) — README of github.com/waico/SKAB
LEADERBOARD = [
    ("Conv-AE", 0.78, 13.55, 28.02), ("MSET", 0.78, 39.73, 14.13), ("T-squared+Q (PCA)", 0.76, 26.62, 24.92),
    ("LSTM-AE", 0.74, 29.96, 25.92), ("T-squared", 0.66, 19.21, 42.6), ("LSTM-VAE", 0.56, 9.13, 55.03),
    ("Vanilla LSTM", 0.54, 12.54, 59.53), ("Vanilla AE", 0.39, 2.59, 75.15), ("MSCRED", 0.36, 49.94, 69.88),
    ("Isolation forest", 0.29, 2.56, 82.89),
]


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
@dataclass
class Experiment:
    name: str          # e.g. "valve1/3"
    group: str         # fault family: valve1 / valve2 / other
    X: np.ndarray      # (n, 8) raw sensor values
    y: np.ndarray      # (n,) 0/1 anomaly label
    t_start: str = ""  # first / last timestamp ("2020-03-09 10:14:33" sorts as text)
    t_end: str = ""

    @property
    def Xtr(self):
        return self.X[:TRAIN_ROWS]

    @property
    def yte(self):
        return self.y[TRAIN_ROWS:]


def load(data_dir: str) -> list[Experiment]:
    """Every labelled SKAB csv under data_dir (the anomaly-free file is skipped), in a stable order."""
    exps = []
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            if not fn.endswith(".csv") or "anomaly-free" in root or "anomaly-free" in fn:
                continue
            with open(os.path.join(root, fn), encoding="utf-8-sig") as f:
                header = f.readline().strip().split(";")
                rows = [line.strip().split(";") for line in f if line.strip()]
            col = {h: i for i, h in enumerate(header)}
            if "anomaly" not in col or not all(s in col for s in SENSORS):
                continue
            X = np.array([[float(r[col[s]]) for s in SENSORS] for r in rows])
            y = np.array([int(float(r[col["anomaly"]])) for r in rows])
            group = os.path.basename(root)
            tcol = col.get("datetime")
            t0, t1 = (rows[0][tcol], rows[-1][tcol]) if tcol is not None and rows else ("", "")
            exps.append(Experiment(f"{group}/{fn[:-4]}", group, X, y, t0, t1))
    exps.sort(key=lambda e: (e.group, int(e.name.split("/")[1]) if e.name.split("/")[1].isdigit() else 0))
    if not exps:
        raise SystemExit(f"no SKAB csv files under {data_dir!r} — git clone https://github.com/waico/SKAB")
    return exps


def drift_robust(X: np.ndarray, tau: float | None, channels=SLOW_CHANNELS) -> np.ndarray:
    """Replace each slow channel by its deviation from a causal exponential moving average (time constant tau s)."""
    if not tau:
        return X
    X = X.copy()
    a = 1.0 / tau
    for j in channels:
        m = X[:60, j].mean()
        col = X[:, j].copy()
        for i, v in enumerate(col):
            X[i, j] = v - m
            m += a * (v - m)
    return X


# ----------------------------------------------------------------------
# detectors: fit on the healthy rows, return one anomaly score per row
# ----------------------------------------------------------------------
class Scaler:
    def __init__(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9

    def __call__(self, X):
        return (X - self.mu) / self.sd


@dataclass(frozen=True)
class Settings:
    """How a raw score becomes an alarm. Defaults follow the SKAB leaderboard notebooks."""
    q: float = 0.99          # quantile of the healthy training scores ...
    factor: float = 4 / 3    # ... stretched by this factor = the alarm limit (UCL)
    smooth: int = 1          # causal moving average of the score over this many rows
    persist: int = 1         # alarm only once this many rows in a row are above the limit

    def as_dict(self):
        return {"q": self.q, "factor": round(self.factor, 3), "smooth": self.smooth, "persist": self.persist}


class Detector:
    key = label = family = ""
    window = 1               # rows of history one score needs
    zero_based = True        # score 0 = perfect (reconstruction errors); else centre on the healthy median
    defaults = Settings()
    fixed_limit: float | None = None

    def fit(self, Z: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def score(self, Z: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """One score per row of Z from row window-1 on (len = len(Z) - window + 1)."""
        raise NotImplementedError


class FixedLimit(Detector):
    """What most plants run today: alarm when any sensor leaves mean ± 3 std of its healthy values."""
    key, label, family = "fixed", "Fixed 3σ limit (today's alarms)", "baseline"
    defaults = Settings(q=1.0, factor=1.0)
    fixed_limit = 3.0

    def fit(self, Z):
        pass

    def score(self, Z):
        return np.abs(Z).max(axis=1)


class PCAT2Q(Detector):
    """Hotelling T² inside the principal subspace + Q (squared prediction error) outside it."""
    key, label, family = "pca", "PCA T²+Q", "statistical"

    def fit(self, Z, var=0.95):
        lam, V = np.linalg.eigh(np.cov(Z, rowvar=False))
        lam, V = lam[::-1], V[:, ::-1]
        k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), var) + 1)
        self.P, self.lam = V[:, :k], np.maximum(lam[:k], 1e-9)
        t2, q = self._parts(Z)
        self.t2_ref, self.q_ref = np.quantile(t2, 0.99) + 1e-9, np.quantile(q, 0.99) + 1e-9

    def _parts(self, Z):
        T = Z @ self.P
        return (T ** 2 / self.lam).sum(axis=1), ((Z - T @ self.P.T) ** 2).sum(axis=1)

    def score(self, Z):
        t2, q = self._parts(Z)
        return np.maximum(t2 / self.t2_ref, q / self.q_ref)   # alarm if either statistic is out


class SkModel(Detector):
    family = "classical ML"
    zero_based = False
    defaults = Settings(q=0.99, factor=1.5, smooth=5)

    def fit(self, Z):
        self.m = self.make()
        self.m.fit(Z)

    def score(self, Z):
        return -self.m.score_samples(Z)


class IForest(SkModel):
    key, label = "iforest", "Isolation Forest"

    @staticmethod
    def make():
        from sklearn.ensemble import IsolationForest
        return IsolationForest(n_estimators=200, random_state=0)


class LOF(SkModel):
    key, label = "lof", "Local Outlier Factor"

    @staticmethod
    def make():
        from sklearn.neighbors import LocalOutlierFactor
        return LocalOutlierFactor(n_neighbors=30, novelty=True)


class OCSVM(SkModel):
    key, label = "ocsvm", "One-Class SVM"

    @staticmethod
    def make():
        from sklearn.svm import OneClassSVM
        return OneClassSVM(nu=0.02, gamma="scale")


class Windowed(Detector):
    """Reconstruction-error models on sliding windows; a window's score belongs to its last row."""
    family = "deep learning"
    defaults = Settings(q=0.99, factor=1.5)

    def fit(self, Z):
        self.model = self.build(Z.shape[1])
        self.model.fit(windows(Z, self.window))

    def score(self, Z):
        W = windows(Z, self.window)
        return np.abs(W - self.model.reconstruct(W)).mean(axis=1).sum(axis=1)   # residual of the SKAB notebooks


class DenseAE(Windowed):
    key, label = "dense_ae", "Dense autoencoder (MLP)"
    window = 10

    def build(self, c):
        from sklearn.neural_network import MLPRegressor
        win = self.window

        class _M:
            def fit(self, W):
                import warnings
                self.m = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="tanh", max_iter=300,
                                      early_stopping=True, random_state=0)
                F = W.reshape(len(W), -1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.m.fit(F, F)

            def reconstruct(self, W):
                return self.m.predict(W.reshape(len(W), -1)).reshape(len(W), win, c)
        return _M()


class ConvAE(Windowed):
    key, label = "conv_ae", "Conv-1D autoencoder"
    window = 60
    defaults = Settings(q=0.99, factor=4 / 3)

    def build(self, c):
        return ConvAutoencoder(c, window=self.window, epochs=40, batch=32, lr=1e-3, seed=0)


class LSTMAE(Windowed):
    key, label = "lstm_ae", "LSTM autoencoder"
    window = 10

    def build(self, c):
        return LSTMAutoencoder(c, window=self.window, hidden=32, epochs=40, batch=32, lr=2e-3, seed=0)


DETECTORS: dict[str, type[Detector]] = {d.key: d for d in (FixedLimit, PCAT2Q, IForest, LOF, OCSVM, DenseAE, ConvAE, LSTMAE)}
ENSEMBLE = ("pca", "lof", "dense_ae", "lstm_ae", "conv_ae")


# ----------------------------------------------------------------------
# turning scores into alarms
# ----------------------------------------------------------------------
def causal_mean(s: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(s, float)
    c = np.cumsum(np.insert(np.asarray(s, float), 0, 0.0))
    i = np.arange(len(s))
    lo = np.maximum(0, i - k + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def persist(flags: np.ndarray, k: int) -> np.ndarray:
    """Alarm on a row only once the last k rows were all above the limit (debouncing)."""
    flags = np.asarray(flags, bool)
    if k <= 1:
        return flags
    i = np.arange(len(flags))
    last_off = np.maximum.accumulate(np.where(flags, -1, i))      # index of the latest row below the limit
    return (i - last_off) >= k                                      # = length of the current run above it


@dataclass
class FileScores:
    """Raw scores of one detector on one experiment; reusable for any Settings (so tuning is cheap)."""
    train: np.ndarray               # scores on the healthy training rows (to set the limit)
    test: np.ndarray                # one score per test row
    base: float = 0.0               # score of a perfectly normal row
    fixed_limit: float | None = None

    def ucl(self, s: Settings) -> float:
        if self.fixed_limit is not None:
            return self.fixed_limit
        tr = causal_mean(self.train, s.smooth)
        return self.base + (float(np.quantile(tr, s.q)) - self.base) * s.factor

    def normalised(self, s: Settings) -> np.ndarray:
        """1.0 = exactly on the alarm limit."""
        return (causal_mean(self.test, s.smooth) - self.base) / max(self.ucl(s) - self.base, 1e-12)

    def flags(self, s: Settings) -> np.ndarray:
        return persist(self.normalised(s) > 1.0, s.persist)


def score_file(det_cls: type[Detector], e: Experiment, tau: float | None = None) -> FileScores:
    det = det_cls()
    X = drift_robust(e.X, tau)
    sc = Scaler(X[:TRAIN_ROWS])
    Z = sc(X)
    det.fit(Z[:TRAIN_ROWS])
    w = det.window
    train = det.score(Z[:TRAIN_ROWS])
    test = det.score(Z[TRAIN_ROWS - w + 1:])          # causal windows ending on each test row
    assert len(test) == len(e.X) - TRAIN_ROWS
    base = 0.0 if det.zero_based else float(np.median(train))
    return FileScores(train, test, base, det.fixed_limit)


def ensemble_file(parts: list[FileScores], settings: list[Settings]) -> FileScores:
    """Average of the members' normalised scores (each member's own limit = 1.0)."""
    n_tr = min(len(p.train) for p in parts)
    te, tr = [], []
    for p, s in zip(parts, settings):
        span = max(p.ucl(s) - p.base, 1e-12)
        te.append((causal_mean(p.test, s.smooth) - p.base) / span)
        tr.append(((causal_mean(p.train, s.smooth) - p.base) / span)[-n_tr:])
    return FileScores(np.mean(tr, axis=0), np.mean(te, axis=0), 0.0)


# ----------------------------------------------------------------------
# supervised: learn from labelled past faults
# ----------------------------------------------------------------------
def _roll(Z, k):
    c = np.cumsum(np.vstack([np.zeros((1, Z.shape[1])), Z]), axis=0)
    i = np.arange(len(Z))
    lo = np.maximum(0, i - k + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)[:, None]


def supervised_features(e: Experiment) -> np.ndarray:
    """Causal per-row features, scaled by the experiment's own healthy start (no labels used)."""
    Z = Scaler(e.Xtr)(e.X)
    F = [Z]
    for k in (5, 20, 60):
        m = _roll(Z, k)
        F += [m, np.sqrt(np.maximum(_roll(Z ** 2, k) - m ** 2, 0))]     # rolling mean and std
    lag = np.vstack([np.repeat(Z[:1], 60, 0), Z[:-60]])
    F.append(_roll(Z, 10) - _roll(lag, 10))                              # change over the last minute
    return np.hstack(F)


SUPERVISED_SETTINGS = Settings(q=0.5, factor=1.0, smooth=10, persist=1)   # probability > 0.5, 10 s smoothing


def overlap_partners(exps: list[Experiment], min_rows: int = 10) -> list[set[int]]:
    """SKAB has a few experiments whose recordings overlap in time and share identical sensor rows
    (other/2-3-4, other/10-11, other/12-13). For leave-one-experiment-out, such a partner must be held out
    together with the experiment. Partners = overlapping time spans, or >= min_rows identical rows."""
    seen: dict[bytes, set[int]] = {}
    for i, e in enumerate(exps):
        for row in np.ascontiguousarray(e.X):
            seen.setdefault(row.tobytes(), set()).add(i)
    shared: dict[tuple[int, int], int] = {}
    for idx in seen.values():
        if len(idx) > 1:
            for a in idx:
                for b in idx:
                    if a < b:
                        shared[(a, b)] = shared.get((a, b), 0) + 1
    partners: list[set[int]] = [set() for _ in exps]
    for a, ea in enumerate(exps):
        for b, eb in enumerate(exps):
            if a != b and ea.t_start and eb.t_start and ea.t_start <= eb.t_end and eb.t_start <= ea.t_end:
                partners[a].add(b)
    for (a, b), n in shared.items():
        if n >= min_rows:
            partners[a].add(b)
            partners[b].add(a)
    return partners


def supervised_probs(exps: list[Experiment], hold_out) -> list[np.ndarray]:
    """hold_out(i, j) -> True if experiment j may be used to train the model that scores experiment i.
    With no allowed training experiment (e.g. the very first one chronologically) the model stays silent."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    F = [supervised_features(e) for e in exps]
    out = []
    for i in range(len(exps)):
        tr = [j for j in range(len(exps)) if hold_out(i, j)]
        ys = [exps[j].y for j in tr]
        if not tr or len(set(np.concatenate(ys).tolist())) < 2:
            out.append(np.zeros(len(exps[i].X) - TRAIN_ROWS))
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
        clf.fit(np.vstack([F[j] for j in tr]), np.concatenate(ys))
        out.append(clf.predict_proba(F[i][TRAIN_ROWS:])[:, 1])
    return out


def pooled_auc(fs_list: list[FileScores], s: Settings, trues: list[np.ndarray]) -> float:
    """Threshold-free ranking quality: ROC-AUC of the normalised scores pooled over all experiments."""
    from sklearn.metrics import roc_auc_score
    sc = np.concatenate([fs.normalised(s) for fs in fs_list])
    return round(float(roc_auc_score(np.concatenate(trues), sc)), 3)


def prob_scores(probs: list[np.ndarray]) -> list[FileScores]:
    return [FileScores(np.zeros(1), p, 0.0, fixed_limit=0.5) for p in probs]


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def _segments(mask):
    mask = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def metrics(preds: list[np.ndarray], trues: list[np.ndarray]) -> dict:
    """SKAB leaderboard metrics (pooled point-wise) + operator-facing event metrics."""
    TP = FP = TN = FN = 0
    caught = n_events = fa_events = normal_rows = 0
    delays = []
    for p, t in zip(preds, trues):
        p, t = np.asarray(p, bool), np.asarray(t, bool)
        TP += int((p & t).sum()); FP += int((p & ~t).sum())
        TN += int((~p & ~t).sum()); FN += int((~p & t).sum())
        normal_rows += int((~t).sum())
        for a, b in _segments(t):
            n_events += 1
            hit = np.flatnonzero(p[a:b])
            if len(hit):
                caught += 1
                delays.append(int(hit[0]))
        fa_events += sum(1 for a, b in _segments(p) if not t[a:b].any())  # alarm runs that never touch a fault
    f1 = TP / (TP + (FN + FP) / 2) if TP + FN + FP else 0.0
    return {
        "f1": round(f1, 3), "far": round(100 * FP / max(1, FP + TN), 2), "mar": round(100 * FN / max(1, FN + TP), 2),
        "precision": round(TP / max(1, TP + FP), 3), "recall": round(TP / max(1, TP + FN), 3),
        "faults_caught": caught, "faults": n_events,
        "median_delay_s": float(np.median(delays)) if delays else None,
        "false_alarm_episodes": fa_events,
        "false_alarm_episodes_per_h": round(fa_events / max(1e-9, normal_rows / 3600), 2),
        "tp": TP, "fp": FP, "tn": TN, "fn": FN,
    }


# ----------------------------------------------------------------------
# tuning without peeking: grouped cross-validation over the 3 fault families
# ----------------------------------------------------------------------
GRID = [Settings(q, f, sm, pe) for q in (0.95, 0.99, 0.999) for f in (1.0, 4 / 3, 1.5, 2.0)
        for sm in (1, 5, 15) for pe in (1, 5, 15)]


def objective(m: dict) -> float:
    """What we optimise. F1 alone rewards 'alarm on everything' on SKAB (54% of test rows are faulty,
    so an always-on alarm scores F1 = 0.70), so false alarms are penalised explicitly:
    every 10 points of FAR cost 0.05 F1, every 10 false-alarm episodes per hour cost 0.02 F1."""
    return m["f1"] - 0.5 * m["far"] / 100 - 0.002 * min(m["false_alarm_episodes_per_h"], 100)


def tune_cv(views: dict[str, list[FileScores]], exps: list[Experiment], grid=GRID):
    """views: {view name: per-experiment scores}. For each fault family pick (view, settings) on the
    OTHER families only, then apply to this one. Returns predictions and the choice per family."""
    preds: list = [None] * len(exps)
    chosen = {}
    for g in sorted({e.group for e in exps}):
        tr = [i for i, e in enumerate(exps) if e.group != g]
        yt = [exps[i].yte for i in tr]
        best = max(((v, s) for v in views for s in grid),
                   key=lambda vs: objective(metrics([views[vs[0]][i].flags(vs[1]) for i in tr], yt)))
        chosen[g] = {"view": best[0], **best[1].as_dict()}
        for i, e in enumerate(exps):
            if e.group == g:
                preds[i] = views[best[0]][i].flags(best[1])
    return preds, chosen


# ----------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------
def run(data_dir: str, models: list[str] | None = None, with_ark: bool = True, with_supervised: bool = True,
        log=print) -> dict:
    exps = load(data_dir)
    trues = [e.yte for e in exps]
    models = models or list(DETECTORS)
    rows, preds_out, scores, auc = [], {}, {}, {}

    def add(key, label, family, mode, preds, settings, seconds=None):
        m = metrics(preds, trues)
        rows.append({"key": key, "label": label, "family": family, "mode": mode, "settings": settings,
                     "seconds": seconds, **m})
        preds_out[key] = preds
        log(f"  {label[:44]:44s} {mode[:22]:22s} F1 {m['f1']:.2f}  FAR {m['far']:5.1f}%  MAR {m['mar']:5.1f}%  "
            f"false-alarm episodes/h {m['false_alarm_episodes_per_h']:6.1f}  caught {m['faults_caught']}/{m['faults']}")

    for key in models:
        cls = DETECTORS[key]
        t0 = time.time()
        views = {v: [score_file(cls, e, tau) for e in exps] for v, tau in VIEWS.items()}
        scores[key] = views
        secs = round(time.time() - t0, 1)
        auc[key] = {v: pooled_auc(views[v], cls.defaults, trues) for v in VIEWS}
        add(key, cls.label, cls.family, "plain", [fs.flags(cls.defaults) for fs in views["raw"]],
            {"view": "raw", **cls.defaults.as_dict()}, secs)
        grid = [Settings(1.0, 1.0, sm, pe) for sm in (1, 5, 15) for pe in (1, 5, 15)] if cls.fixed_limit else GRID
        p, chosen = tune_cv(views, exps, grid)
        add(key + "+", cls.label, cls.family, "improved (grouped CV)", p, chosen)

    members = [k for k in ENSEMBLE if k in scores]
    if len(members) >= 2:
        ens_views = {v: [ensemble_file([scores[k][v][i] for k in members], [DETECTORS[k].defaults for k in members])
                         for i in range(len(exps))] for v in VIEWS}
        scores["ensemble"] = ens_views
        auc["ensemble"] = {v: pooled_auc(ens_views[v], Settings(0.99, 1.0), trues) for v in VIEWS}
        label = f"Ensemble ({len(members)} unsupervised models)"
        add("ensemble", label, "ensemble", "plain", [fs.flags(Settings(0.99, 1.0)) for fs in ens_views["raw"]],
            {"view": "raw", "limit": "mean of normalised scores > 1"})
        p, chosen = tune_cv(ens_views, exps)
        add("ensemble+", label, "ensemble", "improved (grouped CV)", p, chosen)

    if with_supervised:
        partners = overlap_partners(exps)
        variants = [
            ("supervised", "leave-one-experiment-out", lambda i, j: j != i and j not in partners[i]),
            ("supervised_lfo", "unseen fault family", lambda i, j: exps[j].group != exps[i].group),
            ("supervised_past", "past experiments only", lambda i, j: bool(exps[j].t_end) and exps[j].t_end < exps[i].t_start),
        ]
        for key, mode, allowed in variants:
            t0 = time.time()
            fs = prob_scores(supervised_probs(exps, allowed))
            scores[key] = {"raw": fs}
            auc[key] = {"raw": pooled_auc(fs, SUPERVISED_SETTINGS, trues)}
            add(key, "Gradient boosting on labelled faults", "supervised ML", mode,
                [f.flags(SUPERVISED_SETTINGS) for f in fs],
                {"threshold": 0.5, "smooth": SUPERVISED_SETTINGS.smooth}, round(time.time() - t0, 1))

    if "ensemble+" in preds_out and "supervised" in preds_out:
        # ARK Predict v2: the supervised model recognises fault patterns it has seen before (-> URGENT),
        # the unsupervised ensemble flags anything unfamiliar (-> MONITOR). Alarm = either.
        yy = np.concatenate(trues).astype(bool)
        for sup, mode in (("supervised", "leave-one-experiment-out"), ("supervised_lfo", "unseen fault family"),
                          ("supervised_past", "past experiments only")):
            if sup not in preds_out:
                continue
            hyb = [a | b for a, b in zip(preds_out["ensemble+"], preds_out[sup])]
            add("hybrid" + sup[10:], "ARK Predict v2 (supervised + autoencoder ensemble)", "ours (hybrid)", mode, hyb,
                {"urgent": "supervised model", "monitor": "improved unsupervised ensemble"})
            urgent = np.concatenate(preds_out[sup])
            monitor = np.concatenate(preds_out["ensemble+"]) & ~urgent
            rows[-1]["tiers"] = {
                "urgent_rows": int(urgent.sum()),
                "urgent_precision": round(float(yy[urgent].mean()), 3) if urgent.any() else None,
                "monitor_only_rows": int(monitor.sum()),
                "monitor_only_precision": round(float(yy[monitor].mean()), 3) if monitor.any() else None}

    if with_ark:
        from .replay import replay_array
        t0 = time.time()
        preds = [replay_array(SENSORS, e.X, train_rows=TRAIN_ROWS, strict_train_rows=True)["flags"][TRAIN_ROWS:]
                 for e in exps]
        add("ark", "ARK Predict live pipeline (v1)", "ours", "as deployed", preds, {}, round(time.time() - t0, 1))

    always = metrics([np.ones_like(t) for t in trues], trues)
    return {"dataset": {"name": "SKAB v0.9", "files": len(exps), "test_rows": int(sum(len(t) for t in trues)),
                        "anomalous_share": round(float(np.mean(np.concatenate(trues))), 3),
                        "always_alarm_f1": always["f1"]},
            "rows": rows, "auc": auc, "leaderboard": [dict(zip(("label", "f1", "far", "mar"), r)) for r in LEADERBOARD],
            "experiments": [e.name for e in exps], "_preds": preds_out, "_scores": scores, "_exps": exps}


def markdown(res: dict) -> str:
    d = res["dataset"]
    out = [f"# SKAB benchmark — {d['files']} real experiments, {d['test_rows']:,} test rows", "",
           f"{100 * d['anomalous_share']:.0f}% of test rows are labelled faulty, so an alarm that is *always on* already "
           f"scores F1 = {d['always_alarm_f1']:.2f} (with FAR = 100%). Always read F1 next to FAR and false-alarm episodes.",
           "", "| Model | Family | Setting | F1 ↑ | FAR % ↓ | MAR % ↓ | Faults caught | False-alarm episodes / h ↓ | Median delay |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in res["rows"]:
        dl = "—" if r["median_delay_s"] is None else f"{r['median_delay_s']:.0f} s"
        out.append(f"| {r['label']} | {r['family']} | {r['mode']} | {r['f1']:.2f} | {r['far']:.1f} | {r['mar']:.1f} | "
                   f"{r['faults_caught']}/{r['faults']} | {r['false_alarm_episodes_per_h']:.1f} | {dl} |")
    if res.get("auc"):
        views = list(VIEWS)
        out += ["", "Threshold-free ranking quality — pooled ROC-AUC (1.0 = perfect, 0.5 = coin flip):", "",
                "| Model | " + " | ".join(views) + " |", "|---|" + "---:|" * len(views)]
        for k, a in res["auc"].items():
            name = DETECTORS[k].label if k in DETECTORS else k
            out.append(f"| {name} | " + " | ".join(f"{a[v]:.3f}" if v in a else "—" for v in views) + " |")
    out += ["", "Published SKAB leaderboard (same protocol):", "", "| Algorithm | F1 | FAR % | MAR % |", "|---|---:|---:|---:|"]
    out += [f"| {r['label']} | {r['f1']:.2f} | {r['far']:.2f} | {r['mar']:.2f} |" for r in res["leaderboard"]]
    return "\n".join(out) + "\n"


def public(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="SKAB data folder (git clone https://github.com/waico/SKAB)")
    ap.add_argument("--models", default=",".join(DETECTORS), help="comma list of " + ",".join(DETECTORS))
    ap.add_argument("--fast", action="store_true", help="skip the (slow) conv autoencoder")
    ap.add_argument("--no-ark", action="store_true", help="skip replaying the live ARK Predict pipeline")
    ap.add_argument("--no-supervised", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--export", nargs="?", const=os.path.join(os.path.dirname(__file__), "results", "skab_results.json"),
                    help="write the dashboard file (default: app/services/anomaly/results/skab_results.json)")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if not (a.fast and m == "conv_ae")]
    res = run(a.data, models, not a.no_ark, not a.no_supervised)
    md = markdown(res)
    print("\n" + md)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(public(res), f, indent=1)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md)
    if a.export:
        os.makedirs(os.path.dirname(a.export), exist_ok=True)
        export_dashboard(res, res["_exps"], a.export)
        print("dashboard file written:", a.export)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


# ----------------------------------------------------------------------
# export for the dashboard (ML Lab view): results + a replay of every experiment
# ----------------------------------------------------------------------
REPLAY_SENSORS = ("Volume Flow RateRMS", "Accelerometer1RMS", "Current", "Temperature")


def _segs_abs(mask, offset=0):
    return [[int(a) + offset, int(b) + offset] for a, b in _segments(mask)]


def export_dashboard(res: dict, exps: list[Experiment], path: str) -> dict:
    """One JSON file the /api/anomaly/benchmark endpoints serve (no dataset needed at demo time)."""
    P = res["_preds"]
    replay = {}
    for i, e in enumerate(exps):
        tracks = {"fixed": _segs_abs(P["fixed"][i], TRAIN_ROWS)}
        sup = "supervised_past" if "supervised_past" in P else "supervised"   # realistic: trained on earlier runs only
        if sup in P:
            tracks["urgent"] = _segs_abs(P[sup][i], TRAIN_ROWS)
        if "ensemble+" in P:
            mon = P["ensemble+"][i] & ~(P[sup][i] if sup in P else False)
            tracks["monitor"] = _segs_abs(mon, TRAIN_ROWS)
        if "ark" in P:
            tracks["v1"] = _segs_abs(P["ark"][i], TRAIN_ROWS)
        replay[e.name] = {
            "group": e.group, "rows": len(e.X), "train_rows": TRAIN_ROWS,
            "fault": _segs_abs(e.y.astype(bool)),
            "sensors": {s: [float(f"{v:.4g}") for v in e.X[:, SENSORS.index(s)]] for s in REPLAY_SENSORS},
            "alarms": tracks,
        }
    out = {**public(res), "replay": replay}
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return out
