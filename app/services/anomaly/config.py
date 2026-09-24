"""
ARK Predict — Machine-Health Anomaly Detection (HTH-ML-10) — configuration.

Everything a plant engineer might want to tune lives here, in one place,
as plain dataclasses: per-sensor detector sensitivities, sensor
criticality, severity weights, tier cut-offs and alert-fatigue rules.
There is deliberately NO single global threshold anywhere — every
detector threshold is expressed per sensor, per anomaly type, and in
units of that sensor's own learned noise level (sigma).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorSpec:
    key: str
    label: str
    unit: str
    base: float           # nominal value at idle
    load_gain: float      # how much the value moves with the forklift's duty (0..1)
    noise: float          # white measurement noise (1 sigma)
    thermal: bool = False  # responds to *slow* (averaged) duty, like a temperature
    criticality: float = 0.8  # 0..1 — how bad a fault on this sensor is for safety
    decimals: int = 1
    # Per-sensor detector sensitivities (all in sigma units of this sensor)
    spike_z: float = 6.0
    cusum_k: float = 0.75
    cusum_h: float = 12.0
    stuck_window: int = 12


# Forklift machine-health sensors. Values are illustrative, not OEM specs.
SENSORS: tuple[SensorSpec, ...] = (
    SensorSpec("motor_temp", "Drive-motor temperature", "°C", base=48.0, load_gain=26.0, noise=0.35,
               thermal=True, criticality=0.9, spike_z=6.5),
    SensorSpec("hydraulic_pressure", "Hydraulic pressure", "bar", base=38.0, load_gain=150.0, noise=1.6,
               criticality=1.0, spike_z=6.0),
    SensorSpec("mast_vibration", "Mast / drivetrain vibration", "mm/s", base=1.1, load_gain=2.6, noise=0.09,
               criticality=0.9, decimals=2, spike_z=6.0),
    SensorSpec("battery_voltage", "Traction battery voltage", "V", base=50.6, load_gain=-3.2, noise=0.06,
               criticality=0.7, decimals=2, spike_z=7.0),
    SensorSpec("motor_current", "Drive-motor current", "A", base=22.0, load_gain=190.0, noise=2.2,
               criticality=0.8, spike_z=6.5),
)
SENSOR_BY_KEY = {s.key: s for s in SENSORS}

# Forklifts monitored — IDs match the ARK Shield command-center demo fleet.
MACHINES: tuple[tuple[str, str], ...] = (
    ("ARK-F001", "FE-10231"),
    ("ARK-F002", "FE-10232"),
    ("ARK-F003", "FE-10233"),
    ("ARK-F004", "FE-10234"),
    ("ARK-F005", "FE-10235"),
)

TIERS = ("IGNORE", "MONITOR", "URGENT")
TIER_RANK = {t: i for i, t in enumerate(TIERS)}

# Routing destinations for each tier (what "send it to the right place" means).
ROUTES = {
    "IGNORE": "Logged only — no human notified",
    "MONITOR": "Maintenance queue — review this shift",
    "URGENT": "Page on-call technician + floor supervisor",
}


@dataclass
class SeverityConfig:
    # Type prior: how worrying each anomaly *kind* is on its own.
    type_prior: dict = field(default_factory=lambda: {
        "spike": 0.05,
        "spike_burst": 0.45,
        "dropout": 0.70,
        "stuck": 0.50,
        "drift": 0.55,
        "multivariate": 0.60,
    })
    # Weights of the 0..1 components (sum to 1).
    w_type: float = 0.30
    w_magnitude: float = 0.20
    w_persistence: float = 0.15
    w_trend: float = 0.10
    w_spread: float = 0.15
    w_ml: float = 0.10
    magnitude_full_z: float = 12.0      # |z| at which magnitude component saturates
    persistence_full_s: float = 240.0   # duration at which persistence saturates
    critical_z: float = 12.0            # |z| treated as the "hard limit" for time-to-limit trend
    monitor_at: float = 30.0            # severity >= this -> MONITOR
    urgent_at: float = 60.0             # severity >= this -> URGENT


@dataclass
class RoutingConfig:
    group_window_s: float = 180.0    # new incidents on a machine with an open alert are grouped into it
    reopen_window_s: float = 300.0   # a resolved incident that recurs within this window is re-opened, not re-alerted
    close_after_s: float = 20.0      # an incident with no detector activity for this long is resolved
    spike_burst_count: int = 3       # N spikes on one sensor ...
    spike_burst_window_s: float = 600.0  # ... within this window becomes a "spike_burst" incident
    min_confirm_s: float = 3.0       # a MONITOR incident must persist this long before it notifies
    pattern_confirm_s: float = 20.0  # ML "abnormal pattern" incidents wait this long (attribution settles)


@dataclass
class EngineConfig:
    warmup_s: int = 900              # anomaly-free seconds used to learn each sensor's baseline
    history_points: int = 600        # points kept per sensor for charts
    ewma_fast: float = 0.10
    ewma_slow: float = 0.02
    sigma_adapt: float = 0.002       # slow, gated adaptation of each sensor's noise level
    dropout_min_s: int = 3
    drift_confirm_z: float = 2.5     # smoothed |z| a CUSUM alarm must also reach to count as drift
    use_isolation_forest: bool = True
    use_gaussian: bool = True        # multivariate Gaussian (Mahalanobis) on smoothed residuals — joint shifts
    use_autoencoder: bool = True     # temporal (deep) autoencoder over the last ae_window seconds
    ae_window: int = 30
    if_smooth: float = 0.1           # EWMA factor for the point-wise models' input (≈10 s memory)
    ml_contamination: float = 0.01
    ml_multivariate_score: float = 1.6   # normalized ML score (max of IF / Gaussian / autoencoder) that, sustained, opens a pattern incident
    ml_multivariate_s: int = 20          # ... for at least this many seconds
    ml_multivariate_window: int = 30     # ... out of the last this many seconds
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
