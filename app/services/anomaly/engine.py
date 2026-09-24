"""
AnomalyEngine — ties the layers together for one stream of frames:

    frame -> per-sensor detectors (statistical)
          -> fleet ML: Isolation Forest + multivariate Gaussian (joint, ~10 s smoothed residuals)
                       + temporal autoencoder (30 s windows)
          -> incident manager / severity / routing -> notifications

It is pure, synchronous and deterministic (given a seed), so the same
code runs inside the FastAPI service, the offline evaluator and tests.
A naive fixed-threshold alerter runs alongside purely as a *baseline*
for the alert-fatigue comparison — it never drives any routing.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import MACHINES, SENSORS, EngineConfig
from .detectors import SensorDetector, features
from .ml import MachineModel, SequenceAutoencoder
from .classifier import FAULT_CLASSES, FeatureTracker
from .router import AlertRouter

SENSOR_KEYS = [s.key for s in SENSORS]


class AnomalyEngine:
    def __init__(self, cfg: EngineConfig | None = None, machines=MACHINES, seed: int = 0) -> None:
        self.cfg = cfg or EngineConfig()
        self.machines = [m for m, _ in machines]
        self.detectors = {m: {s.key: SensorDetector(s, self.cfg) for s in SENSORS} for m in self.machines}
        self.duty_fast = {m: 0.0 for m in self.machines}
        self.duty_slow = {m: 0.0 for m in self.machines}
        self.model = MachineModel(self.cfg.ml_contamination, seed)                       # Isolation Forest
        self.gauss = MachineModel(self.cfg.ml_contamination, seed, kind="mahalanobis",
                                  leave_one_out=True)                                     # joint-shift Gaussian
        self.ae = SequenceAutoencoder(len(SENSOR_KEYS), self.cfg.ae_window, seed=seed) if self.cfg.use_autoencoder else None
        self._win = {m: deque(maxlen=self.cfg.ae_window) for m in self.machines}
        self._zs = {m: [0.0] * len(SENSOR_KEYS) for m in self.machines}   # smoothed z (IF input)
        self.if_last = {m: 0.0 for m in self.machines}
        self.ae_last = {m: 0.0 for m in self.machines}
        self.g_last = {m: 0.0 for m in self.machines}
        self.ae_share = {m: [0.0] * len(SENSOR_KEYS) for m in self.machines}
        self.router = AlertRouter(self.cfg)
        # ARK Predict v2: supervised fault classifier (names *which* fault); attach with set_classifier()
        self.clf = None
        self.clf_record: list | None = None          # set to [] to record (t, machine, features) for training
        self.trackers = {m: FeatureTracker(len(SENSOR_KEYS)) for m in self.machines}
        self.diag = {m: self._empty_diag() for m in self.machines}
        self.feat_hist = {m: deque(maxlen=1800) for m in self.machines}   # recent features, for technician labels
        n = self.cfg.history_points
        self.history = {m: {s: deque(maxlen=n) for s in SENSOR_KEYS} for m in self.machines}
        self.ml_history = {m: deque(maxlen=n) for m in self.machines}
        self.ml_last = {m: 0.0 for m in self.machines}
        self.t = 0
        self.fitted = False
        self.processed_readings = 0
        # naive baseline: one fixed |x - mean| > 3*std threshold per sensor on raw values
        self._raw_stats: dict = {}
        self._baseline_state: dict = {}
        self.baseline_alerts: list[tuple[int, str, str]] = []

    # ------------------------------------------------------------------
    def _x(self, machine: str, duty: float) -> np.ndarray:
        cfg = self.cfg
        self.duty_fast[machine] += cfg.ewma_fast * (duty - self.duty_fast[machine])
        self.duty_slow[machine] += cfg.ewma_slow * (duty - self.duty_slow[machine])
        return features(duty, self.duty_fast[machine], self.duty_slow[machine])

    def fit(self, frames: list[dict]) -> None:
        """Learn every sensor's context baseline + the fleet ML model from healthy warm-up frames."""
        X = {m: [] for m in self.machines}
        Y = {m: {s: [] for s in SENSOR_KEYS} for m in self.machines}
        for fr in frames:
            for m in self.machines:
                row = fr["readings"][m]
                X[m].append(self._x(m, row["duty"]))
                for s in SENSOR_KEYS:
                    Y[m][s].append(np.nan if row[s] is None else row[s])
        skip = min(120, len(frames) // 5)  # let the slow duty average settle
        vecs = []
        for m in self.machines:
            Xm = np.array(X[m])[skip:]
            for s in SENSOR_KEYS:
                y = np.array(Y[m][s], dtype=float)[skip:]
                ok = ~np.isnan(y)
                self.detectors[m][s].fit(Xm[ok], y[ok])
                self._raw_stats[(m, s)] = (float(np.nanmean(y)), float(np.nanstd(y)))
            Z = np.column_stack([
                (np.array(Y[m][s], dtype=float)[skip:] - Xm @ self.detectors[m][s].coef) / self.detectors[m][s].sigma
                for s in SENSOR_KEYS
            ])
            vecs.append(np.column_stack([np.clip(np.nan_to_num(Z), -10, 10), Xm[:, 1]]))
        feats = np.vstack([self._if_features(v) for v in vecs])
        if self.cfg.use_isolation_forest:
            self.model.fit(feats)
        if self.cfg.use_gaussian:
            self.gauss.fit(feats)
        if self.ae is not None:
            wins = [SequenceAutoencoder.windows(np.column_stack([np.clip(v[:, :-1], -3, 3), v[:, -1:]]), self.ae.window)
                    for v in vecs]
            self.ae.fit(np.vstack(wins))
        self.fitted = True

    # ------------------------------------------------------------------
    def process(self, frame: dict, ts_fn=None) -> list[dict]:
        if not self.fitted:
            raise RuntimeError("engine not fitted — call fit() with warm-up frames first")
        t, ts = frame["t"], frame["ts"]
        self.t = t
        signals: dict = {}
        ml_rows = []
        for m in self.machines:
            row = frame["readings"][m]
            x = self._x(m, row["duty"])
            zvec = []
            z_raw = []
            for s in SENSOR_KEYS:
                v = row[s]
                obs = self.detectors[m][s].update(t, v, x)
                z_raw.append(obs.z if v is not None else None)
                if v is not None:
                    self.processed_readings += 1
                if obs.signals:
                    signals[(m, s)] = obs.signals
                zvec.append(0.0 if obs.z is None else max(-10.0, min(10.0, obs.z)))
                self.history[m][s].append((t, v, round(obs.expected, 3), None if obs.z is None else round(obs.z, 2)))
                self._baseline(t, m, s, v)
            zs = self._zs[m]
            for k_, zz in enumerate(zvec):
                zs[k_] += self.cfg.if_smooth * (max(-6.0, min(6.0, zz)) - zs[k_])
            ml_rows.append(list(zs) + [row["duty"]])
            # the autoencoder looks for *shapes inside normal limits*; anything beyond ±3σ is the
            # statistical detectors' job, so it is clipped here and can't masquerade as a pattern
            self._win[m].append([max(-3.0, min(3.0, zz)) for zz in zvec] + [row["duty"]])
            self.trackers[m].update(z_raw, [row[s] for s in SENSOR_KEYS], row["duty"])
        self._classify(t)
        P = np.array(ml_rows)
        if_scores = self._ml_scores(P) if self.cfg.use_isolation_forest else [0.0] * len(self.machines)
        g_scores = self._scores(self.gauss, P) if self.cfg.use_gaussian else [0.0] * len(self.machines)
        ae_scores, ae_share = [0.0] * len(self.machines), [None] * len(self.machines)
        ready = [i for i, m in enumerate(self.machines) if len(self._win[m]) == self.cfg.ae_window]
        if self.ae is not None and self.ae.trained and ready:
            W = np.array([np.ravel(self._win[self.machines[i]]) for i in ready])
            sc, sh = self.ae.score(W)
            for j, i in enumerate(ready):
                ae_scores[i], ae_share[i] = float(sc[j]), sh[j]
        ml_scores, ml_info = {}, {}
        for i, m in enumerate(self.machines):
            fs, ae, gs = float(if_scores[i]), float(ae_scores[i]), float(g_scores[i])
            self.if_last[m], self.ae_last[m], self.g_last[m] = fs, ae, gs
            if ae_share[i] is not None:
                self.ae_share[m] = [float(x) for x in ae_share[i]]
            sc = max(fs, ae, gs)   # any model can raise the flag; all share the same normalized scale
            ml_scores[m] = sc
            ml_info[m] = {"if": fs, "ae": ae, "gauss": gs, "ae_share": dict(zip(SENSOR_KEYS, self.ae_share[m])),
                          "zs": dict(zip(SENSOR_KEYS, self._zs[m])), "diag": self.diag[m]}
            self.ml_last[m] = sc
            self.ml_history[m].append((t, round(fs, 2), round(ae, 2), round(gs, 2)))
        return self.router.process(t, ts, signals, ml_scores, self.detectors, ts_fn or (lambda _t: ts), ml_info)

    # ------------------------------------------------------------------
    # ARK Predict v2: supervised fault classifier
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_diag() -> dict:
        return {"fault": "normal", "confidence": 0.0, "confirmed": False, "probs": None,
                "_ew": None, "_hist": deque(maxlen=15)}

    def set_classifier(self, clf) -> None:
        self.clf = clf
        for m in self.machines:
            self.diag[m] = self._empty_diag()

    def _classify(self, t: int) -> None:
        want_rec = self.clf_record is not None
        use = self.clf is not None and self.clf.trained
        if not (want_rec or use):
            return
        F = np.array([self.trackers[m].features() for m in self.machines])
        if want_rec:
            for m, f in zip(self.machines, F):
                self.clf_record.append((t, m, f.astype(np.float32)))
        if not use:
            return
        for m, f in zip(self.machines, F):
            self.feat_hist[m].append((t, f.astype(np.float32)))
        P = self.clf.predict_proba(F)
        c = self.cfg
        for m, p in zip(self.machines, P):
            d = self.diag[m]
            d["_ew"] = p if d["_ew"] is None else d["_ew"] + c.clf_smooth * (p - d["_ew"])
            ew = d["_ew"]
            k = int(np.argmax(ew[1:])) + 1            # most likely *fault*
            d["fault"], d["confidence"] = FAULT_CLASSES[k], float(ew[k])
            d["_hist"].append(FAULT_CLASSES[k] if ew[k] >= c.clf_confirm_p else None)
            same = sum(1 for h in d["_hist"] if h == d["fault"])
            d["confirmed"] = same >= c.clf_confirm_n
            d["probs"] = {FAULT_CLASSES[i]: round(float(v), 3) for i, v in enumerate(ew)}

    def diagnosis(self, m: str) -> dict:
        d = self.diag[m]
        return {k: v for k, v in d.items() if not k.startswith("_")}

    def _if_features(self, v: np.ndarray) -> np.ndarray:
        """Training-time IF input: the per-sensor residuals smoothed over ~10 s (EWMA), plus duty.
        Smoothing removes single-sample noise, so a *combination* of small sustained shifts
        (e.g. overload: pressure up, current up, voltage down, each under its own limit)
        stands out as a point far from the healthy cloud."""
        a = self.cfg.if_smooth
        Z = np.clip(v[:, :-1], -6, 6)
        S = np.zeros_like(Z)
        acc = np.zeros(Z.shape[1])
        for i in range(len(Z)):
            acc += a * (Z[i] - acc)
            S[i] = acc
        return np.column_stack([S, v[:, -1:]])

    def _scores(self, model, X: np.ndarray) -> list[float]:
        if not model.trained:
            return [0.0] * len(X)
        raw = model._raw(np.nan_to_num(X))
        return [max(0.0, (r - model._p50) / max(1e-9, model._p995 - model._p50)) for r in raw]

    def _ml_scores(self, X: np.ndarray) -> list[float]:
        m = self.model
        if not m.trained:
            return [0.0] * len(X)
        raw = m._raw(np.nan_to_num(X))
        return [max(0.0, (r - m._p50) / max(1e-9, m._p995 - m._p50)) for r in raw]

    def _baseline(self, t: int, m: str, s: str, v) -> None:
        mu, sd = self._raw_stats[(m, s)]
        exceed = v is not None and abs(v - mu) > 3 * sd
        was = self._baseline_state.get((m, s), False)
        if exceed and not was:
            self.baseline_alerts.append((t, m, s))
        self._baseline_state[(m, s)] = exceed
