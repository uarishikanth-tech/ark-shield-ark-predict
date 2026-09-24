"""
Synthetic streaming sensor feed for the forklift fleet, with
controllable, ground-truth-labelled anomaly injection.

Each forklift has a latent *duty cycle* (idle -> travel -> lift -> lower)
that drives every sensor at once, the way real machine signals move
together: lifting raises hydraulic pressure and motor current, sags the
battery voltage, shakes the mast a little more, and slowly heats the
motor. The truck controller reports that duty as a context channel, the
same way a real CAN-bus gateway reports lift/drive demand.

On top of that normal behaviour the simulator injects labelled faults:

    spike          1–2 sample jump (noise / EMI / pothole)           -> truth IGNORE
    spike_burst    repeated spikes on one sensor (loose connector)   -> MONITOR
    drift          slow ramp on one sensor (wear, fouling)  <12 sigma MONITOR, >=12 sigma URGENT
    dropout        sensor stops reporting (None)   <15 s IGNORE, 15–90 s MONITOR, >90 s URGENT
    stuck          sensor repeats one value (frozen)       <120 s MONITOR, >=120 s URGENT
    bearing_wear   vibration + motor temp drift together             -> URGENT
    hydraulic_leak pressure falls while motor current rises          -> URGENT
    battery_fade   voltage sags below what the load explains         -> MONITOR
    oscillation    slow periodic wobble that stays inside every per-sample
                   limit (slack mast chain, resonance). Only a model of the
                   signal's *shape over time* can see it  -> MONITOR
    overload       lifting above rated capacity: pressure and current a
                   little high and voltage a little low *together*, each
                   too small to trip its own detector. Only a model of the
                   joint pattern sees it                  -> MONITOR

Every injected fault is kept as an `InjectedAnomaly` so the evaluator
can score detection, type classification and severity calibration.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .config import MACHINES, SENSOR_BY_KEY, SENSORS

SCENARIOS = (
    "spike", "spike_burst", "drift", "dropout", "stuck",
    "bearing_wear", "hydraulic_leak", "battery_fade", "oscillation", "overload",
)

# Direction a *real* degradation pushes each sensor.
DRIFT_SIGN = {
    "motor_temp": +1, "hydraulic_pressure": -1, "mast_vibration": +1,
    "battery_voltage": -1, "motor_current": +1,
}

DUTY_STATES = {
    # name: (target duty, min seconds, max seconds)
    "idle": (0.06, 20, 80),
    "travel": (0.42, 15, 60),
    "lift": (0.92, 6, 16),
    "lower": (0.28, 5, 10),
}
DUTY_NEXT = {"idle": ["travel"], "travel": ["lift", "idle", "travel"], "lift": ["lower"], "lower": ["travel", "idle"]}


@dataclass
class InjectedAnomaly:
    id: int
    scenario: str
    machine: str
    start: int
    end: int
    truth_tier: str
    # per-sensor effect: sensor -> dict(kind=offset|ramp|nan|freeze|spikes, ...)
    effects: dict = field(default_factory=dict)
    # per-sensor expected detection type (what a correct detector should call it)
    expected_types: dict = field(default_factory=dict)
    source: str = "random"   # "random" | "manual"

    def as_dict(self) -> dict:
        return {
            "id": self.id, "scenario": self.scenario, "machine": self.machine,
            "start": self.start, "end": self.end, "truthTier": self.truth_tier,
            "sensors": sorted(self.effects), "expectedTypes": self.expected_types, "source": self.source,
        }


@dataclass
class _MachineState:
    duty_state: str = "idle"
    duty_left: int = 30
    duty: float = 0.06
    duty_slow: float = 0.06
    ar: dict = field(default_factory=dict)


class SensorFleetSimulator:
    def __init__(self, seed: int | None = 7, anomaly_rate_per_min: float = 1.2,
                 start_time: datetime | None = None, machines=MACHINES) -> None:
        self.rng = random.Random(seed)
        self.machines = [m for m, _ in machines]
        self.asset_numbers = dict(machines)
        self.t = 0
        self.start_time = start_time or datetime.now(timezone.utc).replace(microsecond=0)
        self.anomaly_rate_per_min = anomaly_rate_per_min
        self.random_injection = True
        self.state = {m: _MachineState(ar={s.key: 0.0 for s in SENSORS}) for m in self.machines}
        self.anomalies: list[InjectedAnomaly] = []
        self._next_id = 1
        self._last_clean: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    def timestamp(self, t: int | None = None) -> str:
        return (self.start_time + timedelta(seconds=self.t if t is None else t)).isoformat()

    def active_anomalies(self, t: int | None = None) -> list[InjectedAnomaly]:
        t = self.t if t is None else t
        return [a for a in self.anomalies if a.start <= t <= a.end]

    def _busy(self, machine: str, sensors: list[str], margin: int = 90) -> bool:
        for a in self.anomalies:
            if a.machine != machine:
                continue
            if a.end + margin >= self.t and (set(a.effects) & set(sensors)):
                return True
            # keep compound faults (whole-machine) cleanly separated
            if a.end + margin >= self.t and a.scenario in ("bearing_wear", "hydraulic_leak"):
                return True
        return False

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------
    def inject(self, scenario: str, machine: str | None = None, sensor: str | None = None,
               duration: int | None = None, magnitude: float | None = None,
               start_in: int = 1, source: str = "manual", force: bool = True) -> InjectedAnomaly | None:
        """Schedule an anomaly. magnitude is in multiples of the sensor's noise sigma."""
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
        rng = self.rng
        machine = machine or rng.choice(self.machines)
        if machine not in self.state:
            raise ValueError(f"unknown machine {machine!r}")
        start = self.t + start_in
        effects: dict = {}
        expected: dict = {}

        def pick_sensor(default_pool):
            if sensor:
                if sensor not in SENSOR_BY_KEY:
                    raise ValueError(f"unknown sensor {sensor!r}")
                return sensor
            return rng.choice(default_pool)

        all_keys = [s.key for s in SENSORS]
        if scenario == "spike":
            s = pick_sensor(all_keys)
            mag = magnitude or rng.uniform(9, 16)
            width = duration or rng.choice([1, 1, 2])
            effects[s] = {"kind": "spikes", "times": [start], "width": width,
                          "mag": mag * rng.choice([-1, 1])}
            expected[s] = "spike"
            end, tier = start + width, "IGNORE"
        elif scenario == "spike_burst":
            s = pick_sensor(all_keys)
            mag = magnitude or rng.uniform(9, 14)
            n = 4
            gaps = [rng.randint(25, 80) for _ in range(n - 1)]
            times = [start]
            for g in gaps:
                times.append(times[-1] + g)
            effects[s] = {"kind": "spikes", "times": times, "width": 1, "mag": mag}
            expected[s] = "spike_burst"
            end, tier = times[-1] + 1, "MONITOR"
        elif scenario == "drift":
            s = pick_sensor(["motor_temp", "hydraulic_pressure", "mast_vibration", "motor_current"])
            dur = duration or rng.randint(180, 420)
            mag = magnitude or rng.uniform(7, 24)
            effects[s] = {"kind": "ramp", "mag": DRIFT_SIGN[s] * mag, "ramp_s": dur}
            expected[s] = "drift"
            end = start + dur
            tier = "URGENT" if mag >= 12 else "MONITOR"
        elif scenario == "battery_fade":
            s = "battery_voltage"
            dur = duration or rng.randint(200, 400)
            mag = magnitude or rng.uniform(7, 11)
            effects[s] = {"kind": "ramp", "mag": -mag, "ramp_s": dur}
            expected[s] = "drift"
            end, tier = start + dur, "MONITOR"
        elif scenario == "dropout":
            s = pick_sensor(all_keys)
            dur = duration or rng.choice([rng.randint(5, 12), rng.randint(20, 80), rng.randint(100, 160)])
            effects[s] = {"kind": "nan"}
            expected[s] = "dropout"
            end = start + dur - 1
            tier = "IGNORE" if dur < 15 else ("URGENT" if dur > 90 else "MONITOR")
        elif scenario == "stuck":
            s = pick_sensor(all_keys)
            dur = duration or rng.randint(40, 240)
            effects[s] = {"kind": "freeze"}
            expected[s] = "stuck"
            end = start + dur - 1
            tier = "URGENT" if dur >= 120 else "MONITOR"
        elif scenario == "bearing_wear":
            dur = duration or rng.randint(300, 480)
            mag = magnitude or rng.uniform(16, 24)
            effects["mast_vibration"] = {"kind": "ramp", "mag": mag, "ramp_s": dur}
            effects["motor_temp"] = {"kind": "ramp", "mag": mag * 0.8, "ramp_s": dur - 60, "delay": 60}
            expected = {"mast_vibration": "drift", "motor_temp": "drift"}
            end, tier = start + dur, "URGENT"
        elif scenario == "hydraulic_leak":
            dur = duration or rng.randint(240, 420)
            mag = magnitude or rng.uniform(14, 22)
            effects["hydraulic_pressure"] = {"kind": "ramp", "mag": -mag, "ramp_s": dur}
            effects["motor_current"] = {"kind": "ramp", "mag": mag * 0.55, "ramp_s": dur - 30, "delay": 30}
            expected = {"hydraulic_pressure": "drift", "motor_current": "drift"}
            end, tier = start + dur, "URGENT"
        elif scenario == "oscillation":
            s = pick_sensor(["mast_vibration", "mast_vibration", "hydraulic_pressure", "motor_current"])
            dur = duration or rng.randint(150, 300)
            mag = magnitude or rng.uniform(2.2, 2.8)
            effects[s] = {"kind": "osc", "mag": mag, "period": rng.uniform(6, 12)}
            expected[s] = "multivariate"
            end, tier = start + dur, "MONITOR"
        elif scenario == "overload":
            dur = duration or rng.randint(150, 300)
            mag = magnitude or rng.uniform(1.8, 2.2)
            for s_, sign in (("hydraulic_pressure", 1), ("motor_current", 1), ("battery_voltage", -1)):
                effects[s_] = {"kind": "load_offset", "mag": sign * mag}
                # usually only the joint-pattern model sees it; on a heavy duty cycle the
                # per-sensor drift detectors may catch pressure/current first — both are correct
                expected[s_] = ("multivariate", "drift")
            end, tier = start + dur, "MONITOR"
        else:  # pragma: no cover
            raise ValueError(scenario)

        if not force and self._busy(machine, list(effects)):
            return None
        a = InjectedAnomaly(self._next_id, scenario, machine, start, end, tier, effects, expected, source)
        self._next_id += 1
        self.anomalies.append(a)
        return a

    def _maybe_random_inject(self) -> None:
        if not self.random_injection or self.anomaly_rate_per_min <= 0:
            return
        if self.rng.random() >= self.anomaly_rate_per_min / 60.0:
            return
        weights = {"spike": 5, "spike_burst": 1.2, "drift": 2, "dropout": 2.2, "stuck": 1.6,
                   "bearing_wear": 1, "hydraulic_leak": 1, "battery_fade": 0.8, "oscillation": 1, "overload": 1}
        scen = self.rng.choices(list(weights), weights=list(weights.values()))[0]
        machine = self.rng.choice(self.machines)
        self.inject(scen, machine=machine, source="random", force=False, start_in=1)

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------
    def step(self) -> dict:
        """Advance one second; returns {"t", "ts", "readings": {machine: {sensor: value|None, "duty": d}}}."""
        self.t += 1
        self._maybe_random_inject()
        rng = self.rng
        readings: dict = {}
        active = self.active_anomalies()
        for m in self.machines:
            ms = self.state[m]
            ms.duty_left -= 1
            if ms.duty_left <= 0:
                ms.duty_state = rng.choice(DUTY_NEXT[ms.duty_state])
                _, lo, hi = DUTY_STATES[ms.duty_state]
                ms.duty_left = rng.randint(lo, hi)
            target = DUTY_STATES[ms.duty_state][0]
            ms.duty += 0.35 * (target - ms.duty) + rng.gauss(0, 0.012)
            ms.duty = min(1.0, max(0.0, ms.duty))
            ms.duty_slow += 0.02 * (ms.duty - ms.duty_slow)

            row: dict = {"duty": round(ms.duty, 4)}
            for spec in SENSORS:
                ms.ar[spec.key] = 0.95 * ms.ar[spec.key] + rng.gauss(0, 0.15 * spec.noise)
                driver = ms.duty_slow if spec.thermal else ms.duty
                v = spec.base + spec.load_gain * driver + ms.ar[spec.key] + rng.gauss(0, spec.noise)
                row[spec.key] = v
            # apply anomalies
            for a in active:
                if a.machine != m:
                    continue
                for s, eff in a.effects.items():
                    row[s] = self._apply(a, s, eff, row[s])
            for spec in SENSORS:
                v = row[spec.key]
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    row[spec.key] = round(v, spec.decimals + 1)
                    if not any(a.machine == m and spec.key in a.effects and a.effects[spec.key]["kind"] == "freeze"
                               for a in active):
                        self._last_clean[(m, spec.key)] = row[spec.key]
                else:
                    row[spec.key] = None
            readings[m] = row
        return {"t": self.t, "ts": self.timestamp(), "readings": readings}

    def _apply(self, a: InjectedAnomaly, sensor: str, eff: dict, v: float):
        noise = SENSOR_BY_KEY[sensor].noise
        kind = eff["kind"]
        if kind == "nan":
            return None
        if kind == "freeze":
            if "value" not in eff:
                eff["value"] = self._last_clean.get((a.machine, sensor), v)
            return eff["value"]
        if kind == "spikes":
            for t0 in eff["times"]:
                if t0 <= self.t < t0 + eff["width"]:
                    return v + eff["mag"] * noise
            return v
        if kind == "load_offset":
            # a heavy load shows whenever the truck is carrying it (travel and lift), not at idle
            duty = self.state[a.machine].duty
            return v + eff["mag"] * noise * min(1.0, max(0.0, (duty - 0.1) / 0.4))
        if kind == "osc":
            return v + eff["mag"] * noise * math.sin(2 * math.pi * (self.t - a.start) / eff["period"])
        if kind == "ramp":
            t0 = a.start + eff.get("delay", 0)
            if self.t < t0:
                return v
            frac = min(1.0, (self.t - t0) / max(1, eff["ramp_s"]))
            return v + eff["mag"] * noise * frac
        return v
