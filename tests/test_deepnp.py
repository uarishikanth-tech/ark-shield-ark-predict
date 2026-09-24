"""
Gradient checks for the numpy deep-learning layers (app/services/anomaly/deepnp.py):
every hand-written backward pass is compared with a numerical derivative.

    python tests/test_deepnp.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.anomaly.deepnp import (  # noqa: E402
    LSTM, ConvAutoencoder, Conv1D, Dense, LSTMAutoencoder, RepeatVector, ReLU, Sequential, Upsample1D, windows,
)


def _check(net: Sequential, x: np.ndarray, eps=1e-5, tol=1e-5):
    rng = np.random.default_rng(1)
    R = rng.normal(size=net.forward(x).shape)

    def loss():
        return float((net.forward(x) * R).sum())

    net.forward(x)
    dx = net.backward(R)
    # input gradient
    for _ in range(6):
        idx = tuple(rng.integers(0, s) for s in x.shape)
        old = x[idx]
        x[idx] = old + eps; lp = loss()
        x[idx] = old - eps; lm = loss()
        x[idx] = old
        num = (lp - lm) / (2 * eps)
        assert abs(num - dx[idx]) < tol * max(1, abs(num)), ("dx", num, dx[idx])
    # parameter gradients
    net.forward(x)
    net.backward(R)
    for layer, k in net.param_refs():
        P, G = layer.params[k], layer.grads[k].copy()
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps; lp = loss()
            P[idx] = old - eps; lm = loss()
            P[idx] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - G[idx]) < tol * max(1, abs(num)), (type(layer).__name__, k, num, G[idx])


def test_conv_stride_upsample_gradients():
    r = np.random.default_rng(0)
    net = Sequential([Conv1D(3, 5, 7, 2, r), ReLU(), Upsample1D(2), Conv1D(5, 3, 7, 1, r)])
    _check(net, r.normal(size=(2, 12, 3)))


def test_lstm_gradients():
    r = np.random.default_rng(0)
    net = Sequential([LSTM(3, 4, return_sequences=False, rng=r), RepeatVector(5),
                      LSTM(4, 4, return_sequences=True, rng=r), Dense(4, 3, rng=r)])
    _check(net, r.normal(size=(2, 5, 3)))


def test_autoencoders_learn_a_signal():
    t = np.arange(900)
    X = np.column_stack([np.sin(t / 7), np.cos(t / 11), np.sin(t / 5) * 0.5])
    X += np.random.default_rng(0).normal(scale=0.05, size=X.shape)
    for model in (ConvAutoencoder(3, window=20, epochs=15, seed=0), LSTMAutoencoder(3, window=10, hidden=16, epochs=15)):
        W = windows(X, model.window)
        before = float(((model.reconstruct(W) - W) ** 2).mean())
        model.fit(W)
        after = float(((model.reconstruct(W) - W) ** 2).mean())
        assert after < before * 0.35, (type(model).__name__, before, after)


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
