"""
ML layer — per-forklift multivariate model.

An Isolation Forest (scikit-learn) is trained per forklift on warm-up
vectors of [residual z of every sensor, duty]. It learns what "all the
sensors together" normally look like, so it catches patterns no single
sensor rule sees (e.g. two sensors each only mildly off, but in a
combination that never happens on a healthy truck).

Its raw score is normalized against the warm-up distribution:
    1.0 = as unusual as the 99.5th percentile of healthy data,
    >1  = beyond anything seen while healthy.
The normalized score (a) boosts severity and (b) opens a "multivariate"
incident when it stays high with no single-sensor explanation.

If scikit-learn is unavailable the module falls back to a robust
Mahalanobis-distance model (numpy only), with the same normalization,
so the demo still runs on a minimal install.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - import guard
    from sklearn.ensemble import IsolationForest
    HAVE_SKLEARN = True
except Exception:  # noqa: BLE001
    IsolationForest = None
    HAVE_SKLEARN = False


def _avg_path(n) -> np.ndarray:
    """Average path length of an unsuccessful BST search (Isolation Forest's c(n))."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    m = n > 2
    out[m] = 2.0 * (np.log(n[m] - 1.0) + np.euler_gamma) - 2.0 * (n[m] - 1.0) / n[m]
    return out


class MachineModel:
    def __init__(self, contamination: float = 0.01, seed: int = 0, kind: str | None = None,
                 leave_one_out: bool = False) -> None:
        """kind: "isolation_forest" (default when scikit-learn is present) or "mahalanobis"
        (multivariate Gaussian: distance from the healthy mean, scaled by the healthy covariance).

        leave_one_out (Gaussian only): score = the distance that remains after dropping the single
        most deviant sensor. A lone drifting sensor is the per-sensor detectors' job; what is left
        is a *joint* shift that no single sensor explains (e.g. overload)."""
        self.contamination = contamination
        self.seed = seed
        self.kind = kind or ("isolation_forest" if HAVE_SKLEARN else "mahalanobis")
        self.leave_one_out = leave_one_out and self.kind == "mahalanobis"
        self._model = None
        self._mu = None
        self._inv = None
        self._p50 = 0.0
        self._p995 = 1.0
        self.trained = False

    def _raw(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        if self.kind == "isolation_forest":
            return self._fast_iforest(X)
        d = X - self._mu
        if not self.leave_one_out:
            return np.sqrt(np.einsum("ij,jk,ik->i", d, self._inv, d))
        # min over "drop sensor j" of the Mahalanobis distance on the remaining dimensions
        best = None
        for keep, inv in self._loo:
            dj = d[:, keep]
            v = np.sqrt(np.einsum("ij,jk,ik->i", dj, inv, dj))
            best = v if best is None else np.minimum(best, v)
        return best

    def fit(self, X: np.ndarray) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        if self.kind == "isolation_forest":
            self._model = IsolationForest(n_estimators=150, contamination=self.contamination,
                                          random_state=self.seed)
            self._model.fit(X)
            self._compile_forest()
        else:
            self._mu = np.median(X, axis=0)
            cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-3
            self._inv = np.linalg.inv(cov)
            if self.leave_one_out:
                n = X.shape[1]
                sensors = range(n - 1)          # last column is duty: never dropped
                self._loo = []
                for j in sensors:
                    keep = [k for k in range(n) if k != j]
                    self._loo.append((keep, np.linalg.inv(cov[np.ix_(keep, keep)])))
        s = self._raw(X)
        self._p50 = float(np.percentile(s, 50))
        self._p995 = float(np.percentile(s, 99.5))
        self.trained = True

    # ------------------------------------------------------------------
    # Streaming needs ~5 rows scored every second; sklearn's per-tree
    # joblib dispatch costs ~20 ms per call. So the fitted forest is
    # compiled once into padded numpy arrays and traversed for all trees
    # at once — numerically identical to IsolationForest.score_samples
    # (verified in tests), ~50x faster for small batches.
    # ------------------------------------------------------------------
    def _compile_forest(self) -> None:
        trees = self._model.estimators_
        feats = self._model.estimators_features_
        n = max(t.tree_.node_count for t in trees)
        k = len(trees)
        self._L = np.full((k, n), -1, dtype=np.int64)
        self._R = np.full((k, n), -1, dtype=np.int64)
        self._F = np.zeros((k, n), dtype=np.int64)
        self._T = np.zeros((k, n), dtype=float)
        self._D = np.zeros((k, n), dtype=float)   # depth(node) + c(n_samples(node)) for leaves
        for i, (est, fmap) in enumerate(zip(trees, feats)):
            tr = est.tree_
            c = tr.node_count
            self._L[i, :c], self._R[i, :c] = tr.children_left, tr.children_right
            f = tr.feature.copy()
            f[f < 0] = 0
            self._F[i, :c] = np.asarray(fmap)[f]
            self._T[i, :c] = tr.threshold
            depth = np.zeros(c)
            for node in range(c):
                for ch in (tr.children_left[node], tr.children_right[node]):
                    if ch >= 0:
                        depth[ch] = depth[node] + 1
            self._D[i, :c] = depth + _avg_path(tr.n_node_samples)
        self._max_depth = int(max(t.tree_.max_depth for t in trees)) + 1
        self._norm = float(_avg_path(np.array([self._model.max_samples_]))[0])

    def _fast_iforest(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        k = self._L.shape[0]
        rows = np.arange(k)[None, :]
        node = np.zeros((X.shape[0], k), dtype=np.int64)
        for _ in range(self._max_depth):
            left = self._L[rows, node]
            leaf = left < 0
            if leaf.all():
                break
            fx = np.take_along_axis(X, self._F[rows, node], axis=1)
            go_left = fx <= self._T[rows, node]
            nxt = np.where(go_left, left, self._R[rows, node])
            node = np.where(leaf, node, nxt)
        depths = self._D[rows, node].mean(axis=1)
        return 2.0 ** (-depths / self._norm)

    def score(self, x: np.ndarray) -> float:
        """Normalized anomaly score (0 = typical, 1 = 99.5th pct of healthy data)."""
        if not self.trained:
            return 0.0
        x = np.nan_to_num(np.asarray(x, dtype=float)).reshape(1, -1)
        raw = float(self._raw(x)[0])
        return max(0.0, (raw - self._p50) / max(1e-9, self._p995 - self._p50))


# ======================================================================
# Deep learning layer: temporal autoencoder
# ======================================================================
try:  # pragma: no cover - import guard
    from sklearn.neural_network import MLPRegressor
except Exception:  # noqa: BLE001
    MLPRegressor = None


class SequenceAutoencoder:
    """
    Temporal autoencoder over a sliding window of each truck's recent
    behaviour: the last `window` seconds of [z of every sensor, duty].

    Architecture (encoder -> bottleneck -> decoder), tanh activations:
        (window x features) -> 64 -> 12 -> 64 -> (window x features)

    It is trained only on healthy warm-up windows to reproduce its input.
    A window it cannot reproduce well (high reconstruction error) has a
    *shape over time* it never saw while healthy — e.g. an oscillation that
    stays inside every single-sample limit, which neither the per-sample
    detectors nor the point-wise Isolation Forest can see.

    Per-sensor reconstruction error gives attribution ("the autoencoder
    could not explain mast vibration"), which feeds the root-cause hint.

    Training uses scikit-learn's MLPRegressor (Adam, early stopping); live
    scoring is a plain numpy forward pass (~0.1 ms for the whole fleet).
    Without scikit-learn it falls back to a linear autoencoder (PCA via SVD,
    12 components) with the same interface.
    """

    def __init__(self, n_sensors: int, window: int = 30, hidden: tuple = (64, 12, 64), seed: int = 0) -> None:
        self.n_sensors = n_sensors
        self.n_feat = n_sensors + 1          # z of each sensor + duty
        self.window = window
        self.hidden = hidden
        self.seed = seed
        self.kind = "mlp_autoencoder" if MLPRegressor is not None else "pca_autoencoder"
        self.trained = False
        self._p50 = 0.0
        self._p995 = 1.0
        self.train_info: dict = {}

    # ------------------------------------------------------------------
    @staticmethod
    def windows(seq: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
        """seq: (T, F) -> (N, window*F) sliding windows."""
        T = seq.shape[0]
        idx = np.arange(0, T - window + 1, stride)
        return np.stack([seq[i:i + window].ravel() for i in idx]) if len(idx) else np.zeros((0, window * seq.shape[1]))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "mlp_autoencoder":
            h = X
            for i, (W, b) in enumerate(zip(self._W, self._b)):
                h = h @ W + b
                if i < len(self._W) - 1:
                    h = np.tanh(h)
            return h
        return (X - self._mu) @ self._P @ self._P.T + self._mu

    def _center(self, X: np.ndarray) -> np.ndarray:
        """Remove each sensor's mean over the window: the autoencoder models *shape over time*,
        while level shifts (drifts) are left to the drift detector and the Gaussian model."""
        W = X.reshape(len(X), self.window, self.n_feat).copy()
        W[:, :, : self.n_sensors] -= W[:, :, : self.n_sensors].mean(axis=1, keepdims=True)
        return W.reshape(len(X), -1)

    def _errors(self, X: np.ndarray) -> np.ndarray:
        """Per-window, per-feature mean squared reconstruction error: (N, F)."""
        X = self._center(X)
        R = (X - self._forward(X)) ** 2
        return R.reshape(len(X), self.window, self.n_feat).mean(axis=1)

    def fit(self, X: np.ndarray) -> None:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        Xc = self._center(X)
        if self.kind == "mlp_autoencoder":
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh", solver="adam",
                               learning_rate_init=1e-3, batch_size=128, max_iter=120, early_stopping=True,
                               validation_fraction=0.15, n_iter_no_change=8, alpha=1e-4, random_state=self.seed)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.fit(Xc, Xc)
            self._net = net   # kept for verification (tests compare our numpy forward pass to net.predict)
            self._W = [np.asarray(w) for w in net.coefs_]
            self._b = [np.asarray(b) for b in net.intercepts_]
            self.train_info = {"epochs": int(net.n_iter_), "val_score": round(float(net.best_validation_score_ or 0), 4)}
        else:
            self._mu = Xc.mean(axis=0)
            _, _, Vt = np.linalg.svd(Xc - self._mu, full_matrices=False)
            self._P = Vt[: self.hidden[len(self.hidden) // 2]].T
            self.train_info = {"components": int(self._P.shape[1])}
        # calibrate each sensor's reconstruction error separately: a fault usually lives on
        # one sensor, and averaging it with four healthy ones would dilute it 5x
        E = self._errors(X)[:, : self.n_sensors]
        self._p50 = np.percentile(E, 50, axis=0)
        self._p995 = np.percentile(E, 99.5, axis=0)
        self.train_info.update({"windows": int(len(X)), "input_dim": int(X.shape[1])})
        self.trained = True

    def score(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X: (N, window*F). Returns (score per window, per-sensor share of the excess error (N, n_sensors)).
        Score = the worst sensor's reconstruction error, normalized so 1.0 = that sensor's 99.5th
        percentile on healthy data."""
        if not self.trained or len(X) == 0:
            return np.zeros(len(X)), np.zeros((len(X), self.n_sensors))
        E = self._errors(np.nan_to_num(np.asarray(X, dtype=float)))[:, : self.n_sensors]
        per = np.maximum(0.0, (E - self._p50) / np.maximum(self._p995 - self._p50, 1e-9))
        norm = per.max(axis=1)                      # worst-reconstructed sensor, in healthy-percentile units
        share = per / np.maximum(per.sum(axis=1, keepdims=True), 1e-12)
        return norm, share
