"""
Small deep-learning toolkit in pure numpy — no torch / tensorflow needed.

Two sequence autoencoders, the same families that top the SKAB leaderboard:

  * ConvAutoencoder  — 1-D convolutional encoder/decoder (the "Conv-AE" of SKAB)
      window (T, C) -> Conv(32, k7, stride 2) -> Conv(16, k7, stride 2)
                    -> Upsample x2 -> Conv(16, k7) -> Upsample x2 -> Conv(32, k7) -> Conv(C, k7)
  * LSTMAutoencoder  — LSTM encoder squeezes the window into one vector, an LSTM decoder
      rebuilds it step by step (the "LSTM-AE" of SKAB)

Both are trained with Adam on mean-squared reconstruction error and expose
``fit(windows)`` / ``reconstruct(windows)``. Every layer has a hand-written
backward pass; ``tests/test_deepnp.py`` checks each one against numerical gradients.

Why numpy: the whole ARK Predict stack runs anywhere scikit-learn runs (the
hackathon laptop, a Raspberry Pi gateway), with nothing to download.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------
class Layer:
    params: dict
    grads: dict

    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x, train=False):  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, g):  # pragma: no cover - interface
        raise NotImplementedError


class Conv1D(Layer):
    """'same'-padded 1-D convolution on (batch, time, channels), any stride, via im2col."""

    def __init__(self, c_in: int, c_out: int, k: int = 7, stride: int = 1, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        self.k, self.s, self.c_in, self.c_out = k, stride, c_in, c_out
        lim = np.sqrt(6.0 / (k * c_in + c_out))  # Glorot uniform
        self.params = {"W": rng.uniform(-lim, lim, (k * c_in, c_out)), "b": np.zeros(c_out)}

    def _index(self, T):
        out_t = -(-T // self.s)
        pad = max((out_t - 1) * self.s + self.k - T, 0)
        left = pad // 2
        idx = np.arange(out_t)[:, None] * self.s + np.arange(self.k)[None, :]
        return out_t, left, pad - left, idx

    def forward(self, x, train=False):
        B, T, C = x.shape
        out_t, left, right, idx = self._index(T)
        xp = np.pad(x, ((0, 0), (left, right), (0, 0)))
        cols = xp[:, idx, :].reshape(B, out_t, self.k * C)
        self._cache = (x.shape, xp.shape, idx, left, cols)
        return cols @ self.params["W"] + self.params["b"]

    def backward(self, g):
        (B, T, C), xp_shape, idx, left, cols = self._cache
        out_t = g.shape[1]
        self.grads["W"] = cols.reshape(-1, cols.shape[-1]).T @ g.reshape(-1, self.c_out)
        self.grads["b"] = g.sum(axis=(0, 1))
        dcols = (g @ self.params["W"].T).reshape(B, out_t, self.k, C)
        dxp = np.zeros(xp_shape)
        for j in range(self.k):  # scatter each kernel tap back (rows idx[:, j] are distinct for a fixed j)
            dxp[:, idx[:, j], :] += dcols[:, :, j, :]
        return dxp[:, left:left + T, :]


class Upsample1D(Layer):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.f = factor

    def forward(self, x, train=False):
        return np.repeat(x, self.f, axis=1)

    def backward(self, g):
        B, T, C = g.shape
        return g.reshape(B, T // self.f, self.f, C).sum(axis=2)


class ReLU(Layer):
    def forward(self, x, train=False):
        self._m = x > 0
        return x * self._m

    def backward(self, g):
        return g * self._m


class Dropout(Layer):
    def __init__(self, rate=0.2, seed=0):
        super().__init__()
        self.rate, self.rng = rate, np.random.default_rng(seed)

    def forward(self, x, train=False):
        if not train or self.rate <= 0:
            self._m = None
            return x
        self._m = (self.rng.random(x.shape) >= self.rate) / (1 - self.rate)
        return x * self._m

    def backward(self, g):
        return g if self._m is None else g * self._m


class Dense(Layer):
    """Applied to the last axis (works per time step on (B, T, C))."""

    def __init__(self, n_in, n_out, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        lim = np.sqrt(6.0 / (n_in + n_out))
        self.params = {"W": rng.uniform(-lim, lim, (n_in, n_out)), "b": np.zeros(n_out)}

    def forward(self, x, train=False):
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, g):
        x = self._x
        self.grads["W"] = x.reshape(-1, x.shape[-1]).T @ g.reshape(-1, g.shape[-1])
        self.grads["b"] = g.reshape(-1, g.shape[-1]).sum(axis=0)
        return g @ self.params["W"].T


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTM(Layer):
    """Standard LSTM over (B, T, C). return_sequences=False returns only the last hidden state."""

    def __init__(self, n_in, n_hidden, return_sequences=True, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        H = n_hidden
        lim = np.sqrt(6.0 / (n_in + 4 * H))
        # gate order: input, forget, cell, output
        self.params = {"Wx": rng.uniform(-lim, lim, (n_in, 4 * H)),
                       "Wh": rng.uniform(-lim, lim, (H, 4 * H)),
                       "b": np.concatenate([np.zeros(H), np.ones(H), np.zeros(H), np.zeros(H)])}
        self.H, self.seq = H, return_sequences

    def forward(self, x, train=False):
        B, T, _ = x.shape
        H = self.H
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        xs = x @ Wx + b
        cache = []
        hs = np.zeros((B, T, H))
        for t in range(T):
            a = xs[:, t] + h @ Wh
            i, f, g, o = _sig(a[:, :H]), _sig(a[:, H:2 * H]), np.tanh(a[:, 2 * H:3 * H]), _sig(a[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            cache.append((h, c, i, f, g, o, tc))
            h, c = h_new, c_new
            hs[:, t] = h
        self._cache = (x, cache)
        return hs if self.seq else h

    def backward(self, gout):
        x, cache = self._cache
        B, T, _ = x.shape
        H = self.H
        Wx, Wh = self.params["Wx"], self.params["Wh"]
        if not self.seq:
            gh_seq = np.zeros((B, T, H))
            gh_seq[:, -1] = gout
        else:
            gh_seq = gout
        dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros(4 * H)
        dx = np.zeros_like(x)
        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in reversed(range(T)):
            h_prev, c_prev, i, f, g, o, tc = cache[t]
            dh = gh_seq[:, t] + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            di, df, dg = dc * g, dc * c_prev, dc * i
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
            dWx += x[:, t].T @ da
            dWh += h_prev.T @ da
            db += da.sum(axis=0)
            dx[:, t] = da @ Wx.T
            dh_next = da @ Wh.T
            dc_next = dc * f
        self.grads = {"Wx": dWx, "Wh": dWh, "b": db}
        return dx


class RepeatVector(Layer):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x, train=False):
        return np.repeat(x[:, None, :], self.n, axis=1)

    def backward(self, g):
        return g.sum(axis=1)


# ----------------------------------------------------------------------
# model + training
# ----------------------------------------------------------------------
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=False):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def param_refs(self):
        return [(layer, k) for layer in self.layers for k in layer.params]


class Adam:
    def __init__(self, refs, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.refs, self.lr, self.b1, self.b2, self.eps, self.clip = refs, lr, b1, b2, eps, clip
        self.m = [np.zeros_like(layer.params[k]) for layer, k in refs]
        self.v = [np.zeros_like(layer.params[k]) for layer, k in refs]
        self.t = 0

    def step(self):
        self.t += 1
        gs = [layer.grads[k] for layer, k in self.refs]
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in gs))
        scale = min(1.0, self.clip / (norm + 1e-12))
        for n, ((layer, k), g) in enumerate(zip(self.refs, gs)):
            g = g * scale
            self.m[n] = self.b1 * self.m[n] + (1 - self.b1) * g
            self.v[n] = self.b2 * self.v[n] + (1 - self.b2) * g * g
            mh = self.m[n] / (1 - self.b1 ** self.t)
            vh = self.v[n] / (1 - self.b2 ** self.t)
            layer.params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class _AE:
    """Shared fit / reconstruct for the two autoencoders."""
    net: Sequential

    def __init__(self, epochs=40, batch=32, lr=1e-3, val_split=0.1, patience=6, seed=0):
        self.epochs, self.batch, self.lr, self.val_split, self.patience = epochs, batch, lr, val_split, patience
        self.rng = np.random.default_rng(seed)
        self.history: list[tuple[float, float]] = []

    def fit(self, W: np.ndarray) -> "_AE":
        W = np.asarray(W, dtype=float)
        n_val = int(len(W) * self.val_split)
        tr, va = (W[:-n_val], W[-n_val:]) if n_val >= 5 else (W, W)
        opt = Adam(self.net.param_refs(), lr=self.lr)
        best, best_params, bad = np.inf, None, 0
        for _ in range(self.epochs):
            order = self.rng.permutation(len(tr))
            tl = 0.0
            for s in range(0, len(tr), self.batch):
                xb = tr[order[s:s + self.batch]]
                out = self.net.forward(xb, train=True)
                diff = out - xb
                tl += float((diff ** 2).sum())
                self.net.backward(2 * diff / diff.size)
                opt.step()
            vl = float(((self.net.forward(va) - va) ** 2).mean())
            self.history.append((tl / tr.size, vl))
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_params = [layer.params[k].copy() for layer, k in opt.refs]
            else:
                bad += 1
                if bad >= self.patience:  # early stopping, restore best weights
                    break
        if best_params is not None:
            for (layer, k), p in zip(opt.refs, best_params):
                layer.params[k] = p
        return self

    def reconstruct(self, W: np.ndarray, chunk: int = 512) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        return np.concatenate([self.net.forward(W[s:s + chunk]) for s in range(0, len(W), chunk)]) if len(W) else W


class ConvAutoencoder(_AE):
    def __init__(self, n_features: int, window: int = 60, seed: int = 0, **kw):
        super().__init__(seed=seed, **kw)
        assert window % 4 == 0, "window must be divisible by 4 (two stride-2 convolutions)"
        r = np.random.default_rng(seed)
        self.window = window
        self.net = Sequential([
            Conv1D(n_features, 32, 7, 2, r), ReLU(), Dropout(0.2, seed),
            Conv1D(32, 16, 7, 2, r), ReLU(),
            Upsample1D(2), Conv1D(16, 16, 7, 1, r), ReLU(), Dropout(0.2, seed + 1),
            Upsample1D(2), Conv1D(16, 32, 7, 1, r), ReLU(),
            Conv1D(32, n_features, 7, 1, r),
        ])


class LSTMAutoencoder(_AE):
    def __init__(self, n_features: int, window: int = 10, hidden: int = 32, seed: int = 0, **kw):
        super().__init__(seed=seed, **kw)
        r = np.random.default_rng(seed)
        self.window = window
        self.net = Sequential([
            LSTM(n_features, hidden, return_sequences=False, rng=r),
            RepeatVector(window),
            LSTM(hidden, hidden, return_sequences=True, rng=r),
            Dense(hidden, n_features, rng=r),
        ])


def windows(X: np.ndarray, T: int) -> np.ndarray:
    """All length-T windows of X (n, C) -> (n - T + 1, T, C); window i ends at row i + T - 1."""
    X = np.asarray(X, dtype=float)
    if len(X) < T:
        return np.zeros((0, T, X.shape[1]))
    return np.lib.stride_tricks.sliding_window_view(X, (T, X.shape[1]))[:, 0].copy()
