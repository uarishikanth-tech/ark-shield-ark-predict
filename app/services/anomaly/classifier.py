"""
ARK Predict v2 — supervised fault classifier on the live forklift stream.

The statistical detectors and the unsupervised models decide *whether* a truck
looks wrong. This model decides *what* is wrong: it has learned from labelled
faults what a hydraulic leak, a wearing bearing, a stuck sensor … look like,
and names the fault with a confidence, every second, for every truck.

    features  per truck, from the last 60 s of the pipeline's own residuals
              (z = reading vs. expected-for-this-load, in the sensor's own noise units):
              z now · rolling mean / std over 10, 30, 60 s · change over the last 30 s ·
              largest |z| in 10 s · share of missing readings · how long a reading has
              been frozen · duty (load) now and averaged                      -> 57 numbers
    model     HistGradientBoostingClassifier (scikit-learn), 9 classes:
              normal + 8 fault types
    training  labelled faults injected into the simulator (ground truth), and —
              once technicians start giving verdicts — their labels too (see retrain()).
    output    per truck: most likely fault, confidence, smoothed over ~5 s;
              "confirmed" when the same fault stays >= 70 % for 10 of the last 15 s.

Train / evaluate from the command line:

    python -m app.services.anomaly.classifier --train          # ~2 min, writes results/fault_classifier.pkl
    python -m app.services.anomaly.classifier --evaluate       # accuracy on simulator runs it never saw
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import deque

import numpy as np

FAULT_CLASSES = ["normal", "drift", "dropout", "stuck", "bearing_wear", "hydraulic_leak",
                 "battery_fade", "oscillation", "overload"]
FAULT_LABEL = {
    "normal": "normal", "drift": "sensor drift (wear / fouling)", "dropout": "sensor dropout",
    "stuck": "stuck sensor", "bearing_wear": "bearing wear", "hydraulic_leak": "hydraulic leak",
    "battery_fade": "battery fade", "oscillation": "mast / chain oscillation", "overload": "overload lift",
}
IGNORED_SCENARIOS = ("spike", "spike_burst")   # handled by the rule-based spike logic; rows not used for training
N_SENSORS = 5
HIST = 60
RESULTS = os.path.join(os.path.dirname(__file__), "results")
MODEL_PATH = os.path.join(RESULTS, "fault_classifier.pkl.gz")        # the fitted model (small)
DATA_PATH = os.path.join(RESULTS, "fault_classifier_data.npz")      # its training data (for retraining)


class FeatureTracker:
    """Rolling 60-second memory of one truck; turns it into the classifier's feature vector."""

    def __init__(self, n: int = N_SENSORS):
        self.n = n
        self.z = deque(maxlen=HIST)
        self.miss = deque(maxlen=HIST)
        self.duty = deque(maxlen=HIST)
        self.last_raw = [None] * n
        self.frozen = [0] * n

    def update(self, zvec, raw, duty) -> None:
        self.z.append([0.0 if v is None else max(-10.0, min(10.0, v)) for v in zvec])
        self.miss.append([1.0 if r is None else 0.0 for r in raw])
        self.duty.append(float(duty))
        for j, r in enumerate(raw):
            if r is not None and r == self.last_raw[j]:
                self.frozen[j] += 1
            else:
                self.frozen[j] = 0
            if r is not None:
                self.last_raw[j] = r

    def features(self) -> np.ndarray:
        Z = np.asarray(self.z, dtype=float)
        M = np.asarray(self.miss, dtype=float)
        D = np.asarray(self.duty, dtype=float)
        f = [Z[-1]]
        for k in (10, 30, 60):
            w = Z[-k:]
            f += [w.mean(axis=0), w.std(axis=0)]
        a, b = Z[-15:].mean(axis=0), (Z[-30:-15].mean(axis=0) if len(Z) > 15 else Z[:1].mean(axis=0))
        f.append(a - b)
        f.append(np.abs(Z[-10:]).max(axis=0))
        f.append(M[-30:].mean(axis=0))
        f.append(np.minimum(np.array(self.frozen, dtype=float), 60.0) / 60.0)
        f.append(np.array([D[-1], D[-30:].mean()]))
        return np.concatenate(f)


N_FEATURES = 5 + 3 * 10 + 5 + 5 + 5 + 5 + 2   # 57


# ----------------------------------------------------------------------
# data from the simulator (labels = injected ground truth)
# ----------------------------------------------------------------------
def collect(seed: int, seconds: int, rate: float = 1.5, cfg=None):
    """Run the real pipeline on a simulated fleet with injected faults; return X, y (class index), meta."""
    from .service import AnomalyService
    svc = AnomalyService(seed=seed, anomaly_rate_per_min=rate, cfg=cfg)
    svc.engine.clf_record = []
    svc.ensure_ready()
    svc.step(seconds)
    rec = svc.engine.clf_record
    by_machine: dict = {}
    for a in svc.sim.anomalies:
        by_machine.setdefault(a.machine, []).append(a)
    X, y, meta = [], [], []
    for t, m, feats in rec:
        active = [a for a in by_machine.get(m, []) if a.start <= t <= a.end]
        if any(a.scenario in IGNORED_SCENARIOS for a in active):
            continue
        cls = active[0].scenario if active else "normal"
        if cls not in FAULT_CLASSES:
            continue
        X.append(feats)
        y.append(FAULT_CLASSES.index(cls))
        meta.append((t, m, active[0].id if active else 0))
    return np.array(X), np.array(y), meta, svc.sim.anomalies


class FaultClassifier:
    def __init__(self):
        self.model = None
        self.version = 0
        self.info: dict = {}
        self.base_X = self.base_y = None          # simulator training data (kept for retraining)
        self.feedback_X: list = []
        self.feedback_y: list = []

    @property
    def trained(self) -> bool:
        return self.model is not None

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, class_weight="balanced", random_state=0)
        m.fit(X, y, sample_weight=sample_weight)
        self.model = m
        self.version += 1
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilities for all FAULT_CLASSES (columns aligned even if a class was absent in training)."""
        P = self.model.predict_proba(np.atleast_2d(X))
        out = np.zeros((P.shape[0], len(FAULT_CLASSES)))
        out[:, self.model.classes_.astype(int)] = P
        return out

    # -- technician labels ----------------------------------------------
    def add_feedback(self, X: np.ndarray, cls: str) -> int:
        if cls not in FAULT_CLASSES or not len(X):
            return 0
        self.feedback_X.append(np.asarray(X))
        self.feedback_y.append(np.full(len(X), FAULT_CLASSES.index(cls)))
        return len(X)

    def retrain(self, feedback_weight: float = 3.0) -> dict:
        """Refit on the simulator data + every technician-labelled window (weighted up: real verdicts
        from the field count more than simulated examples)."""
        if self.base_X is None:
            raise RuntimeError("no base training data loaded")
        Xs, ys, ws = [self.base_X], [self.base_y], [np.ones(len(self.base_y))]
        for Xf, yf in zip(self.feedback_X, self.feedback_y):
            Xs.append(Xf)
            ys.append(yf)
            ws.append(np.full(len(yf), feedback_weight))
        t0 = time.time()
        self.fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))
        self.info.update({"feedbackRows": int(sum(len(y) for y in self.feedback_y)),
                          "feedbackWindows": len(self.feedback_y), "trainRows": int(sum(len(y) for y in ys)),
                          "lastTrainSeconds": round(time.time() - t0, 1)})
        return self.status()

    def status(self) -> dict:
        return {"trained": self.trained, "version": self.version, "classes": FAULT_CLASSES,
                "labels": FAULT_LABEL, **{k: v for k, v in self.info.items() if not k.startswith("_")}}

    # -- persistence ----------------------------------------------------
    def save(self, model_path: str = MODEL_PATH, data_path: str = DATA_PATH) -> None:
        import gzip
        import sklearn
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with gzip.open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "info": self.info, "sklearn": sklearn.__version__}, f)
        np.savez_compressed(data_path, X=self.base_X.astype(np.float16), y=self.base_y.astype(np.int8))

    @classmethod
    def load(cls, model_path: str = MODEL_PATH, data_path: str = DATA_PATH, refit_if_stale: bool = True):
        """Load the saved model. If it was pickled by another scikit-learn version it is refitted from the
        saved training data (about a minute) so predictions are always from this machine's library."""
        import gzip
        import sklearn
        c = cls()
        if os.path.exists(data_path):
            d = np.load(data_path)
            c.base_X, c.base_y = d["X"].astype(float), d["y"].astype(int)
        with gzip.open(model_path, "rb") as f:
            d = pickle.load(f)
        c.info = d.get("info", {})
        if d.get("sklearn") == sklearn.__version__ or not refit_if_stale or c.base_X is None:
            c.model, c.version = d["model"], 1
        else:
            c.fit(c.base_X, c.base_y)
            c.info["refitted"] = f"model saved with scikit-learn {d.get('sklearn')}, refitted for {sklearn.__version__}"
        return c


_shared: dict = {}


def shared_classifier(background: bool = True):
    """One classifier per process (the live service and the API share it). Loaded — or, when the
    saved model can't be used as-is, refitted — in a background thread so server start-up is instant."""
    import threading
    if "clf" in _shared:
        return _shared["clf"]
    holder = FaultClassifier()          # untrained until loading finishes
    holder.info = {"loading": True}
    _shared["clf"] = holder

    def _load():
        try:
            if not os.path.exists(MODEL_PATH):
                holder.info = {"loading": False, "error": "no saved model — run: python -m app.services.anomaly.classifier --train"}
                return
            c = FaultClassifier.load()
            holder.base_X, holder.base_y, holder.info, holder.version = c.base_X, c.base_y, c.info, c.version
            holder.model = c.model          # set last: the engine starts using it from the next second
        except Exception as e:  # pragma: no cover - defensive
            holder.info = {"loading": False, "error": f"could not load the classifier: {e}"}

    if background:
        threading.Thread(target=_load, daemon=True).start()
    else:
        _load()
    return holder


# ----------------------------------------------------------------------
# training + honest evaluation (train and test on different simulator runs)
# ----------------------------------------------------------------------
TRAIN_SEEDS = (101, 102, 103)
TEST_SEEDS = (201, 202)


def build(seconds: int = 5400, seeds=TRAIN_SEEDS, log=print) -> FaultClassifier:
    Xs, ys = [], []
    for s in seeds:
        t0 = time.time()
        X, y, _, _ = collect(s, seconds)
        Xs.append(X)
        ys.append(y)
        log(f"  simulated fleet seed {s}: {len(y):,} labelled truck-seconds ({time.time() - t0:.0f} s)")
    X, y = np.vstack(Xs), np.concatenate(ys)
    rng = np.random.default_rng(0)
    normal = np.flatnonzero(y == 0)
    keep = np.sort(np.concatenate([np.flatnonzero(y > 0), rng.choice(normal, min(len(normal), 20000), replace=False)]))
    clf = FaultClassifier()
    clf.base_X, clf.base_y = X[keep], y[keep]
    t0 = time.time()
    clf.fit(clf.base_X, clf.base_y)
    counts = np.bincount(clf.base_y, minlength=len(FAULT_CLASSES))
    clf.info = {"trainRows": int(len(clf.base_y)), "trainSeeds": list(seeds), "simSecondsPerSeed": seconds,
                "classCounts": dict(zip(FAULT_CLASSES, map(int, counts))),
                "lastTrainSeconds": round(time.time() - t0, 1), "feedbackRows": 0, "feedbackWindows": 0}
    return clf


def evaluate_clf(clf: FaultClassifier, seconds: int = 3600, seeds=TEST_SEEDS, log=print) -> dict:
    """Per-second accuracy and per-FAULT accuracy (what does it call each injected fault, by majority
    vote over the fault's second half) on simulator runs never used for training."""
    y_all, p_all, per_fault = [], [], []
    for s in seeds:
        X, y, meta, anomalies = collect(s, seconds)
        P = clf.predict_proba(X)
        y_all.append(y)
        p_all.append(P)
        pred = P.argmax(axis=1)
        by_id: dict = {}
        for (t, m, aid), yy, pp in zip(meta, y, pred):
            if aid:
                by_id.setdefault(aid, []).append((t, yy, pp))
        for aid, rows in by_id.items():
            rows.sort()
            half = rows[len(rows) // 2:]
            true = FAULT_CLASSES[half[0][1]]
            votes = np.bincount([p for _, _, p in half], minlength=len(FAULT_CLASSES))
            votes[0] = 0 if votes[1:].sum() else votes[0]     # "what fault is it", given it is one
            per_fault.append((true, FAULT_CLASSES[int(votes.argmax())]))
    y = np.concatenate(y_all)
    P = np.vstack(p_all)
    pred = P.argmax(axis=1)
    fault_rows = y > 0
    healthy_fp = float((pred[~fault_rows] > 0).mean())
    classes = {}
    for c in FAULT_CLASSES[1:]:
        pairs = [(t, p) for t, p in per_fault if t == c]
        if pairs:
            classes[c] = {"faults": len(pairs), "correct": sum(t == p for t, p in pairs)}
    conf = {c: {} for c in FAULT_CLASSES[1:]}
    for t, p in per_fault:
        conf[t][p] = conf[t].get(p, 0) + 1
    res = {
        "testSeeds": list(seeds), "simSecondsPerSeed": seconds,
        "secondAccuracy": round(float((pred == y).mean()), 3),
        "faultTypeAccuracy": round(sum(t == p for t, p in per_fault) / max(1, len(per_fault)), 3),
        "faultsTested": len(per_fault),
        "healthySecondsFlagged": round(healthy_fp, 4),
        "perClass": classes, "confusion": conf,
    }
    log(f"  unseen runs: fault type right for {res['faultTypeAccuracy']:.0%} of {len(per_fault)} faults · "
        f"healthy seconds wrongly called a fault: {healthy_fp:.2%}")
    return res


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--seconds", type=int, default=5400, help="simulated seconds per training seed")
    a = ap.parse_args()
    if a.train:
        clf = build(a.seconds)
        ev = evaluate_clf(clf)
        clf.info["evaluation"] = ev
        clf.save()
        with open(os.path.join(RESULTS, "fault_classifier_eval.json"), "w") as f:
            json.dump(ev, f, indent=1)
        print("saved", MODEL_PATH)
    elif a.evaluate:
        print(json.dumps(evaluate_clf(FaultClassifier.load()), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
