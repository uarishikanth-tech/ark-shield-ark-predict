"""
Per-sensor streaming detectors (statistical layer).

Each sensor gets its own *context-aware baseline*: a small linear model,
fitted during warm-up, that predicts the value expected at the forklift's
current duty (instantaneous, fast-averaged and slow-averaged, so thermal
lag is captured). Detectors work on the standardized residual

        z = (reading - expected) / sigma_sensor

so "6 sigma" means the same thing for a 0.06 V battery sensor and a
1.6 bar pressure sensor, and a heavy lift never looks like a fault.

Four detectors, each tuned for one anomaly *shape* (no single fixed
threshold anywhere):

  * spike    |z| >= spike_z for at most 3 samples, then returns.
  * drift    two-sided CUSUM on z (spike-aware: samples that turn out to be
             spikes are never fed to it), confirmed only when the EWMA level
             also clears drift_confirm_z, then tracked until it settles back.
  * dropout  the sensor stops reporting for >= dropout_min_s seconds.
  * stuck    the sensor repeats the exact same value for >= stuck_window
             samples (real analog signals always carry noise).

Noise level adapts slowly (gated: only on quiet, non-anomalous samples),
so the baseline follows normal sensor ageing without absorbing faults.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import EngineConfig, SensorSpec


def features(duty: float, duty_fast: float, duty_slow: float) -> np.ndarray:
    return np.array([1.0, duty, duty_fast, duty_slow])


@dataclass
class Signal:
    type: str          # spike | drift | dropout | stuck
    t: int
    z: float = 0.0     # signed peak / current z
    direction: int = 0
    info: dict = field(default_factory=dict)


@dataclass
class Observation:
    value: float | None
    expected: float
    z: float | None
    signals: list[Signal]


class SensorDetector:
    def __init__(self, spec: SensorSpec, cfg: EngineConfig) -> None:
        self.spec = spec
        self.cfg = cfg
        self.coef = np.zeros(4)
        self.sigma = spec.noise
        self._var = spec.noise ** 2
        # state
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.ewma_z = 0.0
        self.drift_active = False
        self.drift_dir = 0
        self.drift_start = 0
        self.drift_quiet = 0
        self.drift_peak = 0.0
        self.missing_run = 0
        self.dropout_start = 0
        self.last_value: float | None = None
        self.repeat_run = 0
        self.stuck_start = 0
        self.spike_pending: list[tuple[int, float]] = []
        self.z_hist: deque = deque(maxlen=90)   # recent z (nan when missing) — slope + correlation
        self.trained = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        mad = float(np.median(np.abs(resid - np.median(resid))))
        sigma = max(1.4826 * mad, float(np.std(resid)) * 0.8, 1e-6)
        self.coef = coef
        self.sigma = sigma
        self._var = sigma ** 2
        self.trained = True

    def expected(self, x: np.ndarray) -> float:
        return float(x @ self.coef)

    # ------------------------------------------------------------------
    @property
    def stuck(self) -> bool:
        return self.repeat_run + 1 >= self.spec.stuck_window

    @property
    def dropout(self) -> bool:
        return self.missing_run >= self.cfg.dropout_min_s

    def slope_per_s(self, n: int = 30) -> float:
        z = np.array(list(self.z_hist)[-n:], dtype=float)
        z = z[~np.isnan(z)]
        if len(z) < 8:
            return 0.0
        tt = np.arange(len(z), dtype=float)
        return float(np.polyfit(tt, z, 1)[0])

    # ------------------------------------------------------------------
    def update(self, t: int, value: float | None, x: np.ndarray) -> Observation:
        spec, cfg = self.spec, self.cfg
        exp = self.expected(x)
        signals: list[Signal] = []

        # ---------------- dropout ----------------
        if value is None or (isinstance(value, float) and math.isnan(value)):
            if self.missing_run == 0:
                self.dropout_start = t
            self.missing_run += 1
            self.z_hist.append(float("nan"))
            signals += self._flush_spikes(t, force=True)
            if self.dropout:
                signals.append(Signal("dropout", t, info={"missing_s": self.missing_run, "since": self.dropout_start}))
            if self.drift_active:
                signals.append(Signal("drift", t, self.ewma_z, self.drift_dir, self._drift_info(t)))
            return Observation(None, exp, None, signals)
        self.missing_run = 0

        z = (value - exp) / self.sigma
        self.z_hist.append(z)

        # ---------------- stuck (exact repeats) ----------------
        if self.last_value is not None and abs(value - self.last_value) <= 1e-9:
            if self.repeat_run == 0:
                self.stuck_start = t - 1
            self.repeat_run += 1
        else:
            self.repeat_run = 0
        self.last_value = value
        if self.repeat_run >= 1:
            # a flat-lining sensor must never be mistaken for a spike or a drift
            self.spike_pending.clear()
            if self.stuck:
                signals.append(Signal("stuck", t, z, info={"stuck_s": self.repeat_run + 1, "since": self.stuck_start,
                                                          "value": value}))
            if self.drift_active:
                signals.append(Signal("drift", t, self.ewma_z, self.drift_dir, self._drift_info(t)))
            return Observation(value, exp, z, signals)

        # ---------------- spike (short excursions) ----------------
        feed: list[float] = []
        # spikes are measured against the current level (so a big drift is not "a spike every sample")
        level = self.ewma_z if self.drift_active else 0.0
        if abs(z - level) >= spec.spike_z:
            self.spike_pending.append((t, z))
            if len(self.spike_pending) > 3:
                # not a spike: a sustained shift -> hand the held samples to the CUSUM
                feed = [pz for _, pz in self.spike_pending]
                self.spike_pending.clear()
        else:
            signals += self._flush_spikes(t)
            feed = [z]

        # ---------------- drift (CUSUM + EWMA tracking) ----------------
        for zz in feed:
            zc = max(-10.0, min(10.0, zz))
            self.ewma_z += cfg.ewma_fast * (zc - self.ewma_z)
            if not self.drift_active:
                self.cusum_pos = max(0.0, self.cusum_pos + zc - spec.cusum_k)
                self.cusum_neg = max(0.0, self.cusum_neg - zc - spec.cusum_k)
                # CUSUM arms the alarm; it only fires once the smoothed level also
                # clears drift_confirm_z (residuals are autocorrelated in real
                # machines, and CUSUM alone over-fires on slow healthy wander).
                armed = self.cusum_pos > spec.cusum_h or self.cusum_neg > spec.cusum_h
                self.cusum_pos = min(self.cusum_pos, spec.cusum_h * 1.5)
                self.cusum_neg = min(self.cusum_neg, spec.cusum_h * 1.5)
                if armed and abs(self.ewma_z) >= cfg.drift_confirm_z:
                    self.drift_active = True
                    self.drift_dir = 1 if self.cusum_pos > self.cusum_neg else -1
                    self.drift_start = t
                    self.drift_quiet = 0
                    self.drift_peak = 0.0
                    self.ewma_z = max(-10.0, min(10.0, zz))
            else:
                if abs(self.ewma_z) < 1.0 or self.ewma_z * self.drift_dir < 0:
                    self.drift_quiet += 1
                else:
                    self.drift_quiet = 0
                self.drift_peak = max(self.drift_peak, abs(self.ewma_z))
                if self.drift_quiet >= 15:
                    self.drift_active = False
                    self.cusum_pos = self.cusum_neg = 0.0
        if self.drift_active:
            signals.append(Signal("drift", t, self.ewma_z, self.drift_dir, self._drift_info(t)))

        # ---------------- gated noise adaptation ----------------
        if not self.drift_active and not self.spike_pending and abs(z) < 3.0:
            r = value - exp
            self._var += cfg.sigma_adapt * (r * r - self._var)
            self.sigma = max(math.sqrt(self._var), spec.noise * 0.5)

        return Observation(value, exp, z, signals)

    # ------------------------------------------------------------------
    def _flush_spikes(self, t: int, force: bool = False) -> list[Signal]:
        if not self.spike_pending:
            return []
        pending, self.spike_pending = self.spike_pending, []
        if len(pending) <= 3:
            t0, _ = pending[0]
            peak = max(pending, key=lambda p: abs(p[1]))[1]
            return [Signal("spike", t0, peak, 1 if peak > 0 else -1, {"width_s": len(pending)})]
        return []

    def _drift_info(self, t: int) -> dict:
        slope = self.slope_per_s()
        crit = self.cfg.severity.critical_z
        cur = abs(self.ewma_z)
        toward = slope * self.drift_dir > 0
        if cur >= crit:
            ttl = 0.0
        elif toward and abs(slope) > 1e-4:
            ttl = (crit - cur) / abs(slope)
        else:
            ttl = None
        return {"since": self.drift_start, "slope_z_per_min": round(slope * 60, 3),
                "time_to_limit_s": None if ttl is None else round(ttl, 1), "ewma_z": round(self.ewma_z, 2)}
