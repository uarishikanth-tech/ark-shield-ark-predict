"""
Root-cause hints.

Two sources of evidence, combined into one plain-language hint:

1. Co-movement: correlation of the last ~90 s of residuals between the
   alerting sensor and every other sensor on the same truck (residuals,
   not raw values, so "both rise when lifting" does not count).
2. Fault signatures: known multi-sensor patterns, keyed on which sensors
   are abnormal together and in which direction.

Output example:
   "Probable drive-motor bearing wear — mast vibration ↑ with motor
    temperature ↑ (r=0.91). Schedule bearing inspection / lubrication."
"""
from __future__ import annotations

import numpy as np

from .config import SENSOR_BY_KEY

ARROW = {1: "↑", -1: "↓", 0: ""}

# (set of (sensor, direction)) -> (title, action)
SIGNATURES = [
    ({("mast_vibration", 1), ("motor_temp", 1)},
     "Probable drive-motor bearing wear / lubrication failure",
     "Inspect bearings, check lubrication; limit truck to light duty"),
    ({("hydraulic_pressure", -1), ("motor_current", 1)},
     "Probable hydraulic leak or pump cavitation — pump working harder while pressure falls",
     "Remove from lifting duty; check hoses, seals and fluid level"),
    ({("hydraulic_pressure", 1), ("motor_current", 1)},
     "Probable overload — lifting above rated capacity",
     "Check the load weight against the capacity plate; brief the operator"),
    ({("battery_voltage", -1), ("motor_current", 1)},
     "Battery under abnormal load — high internal resistance or dragging brake",
     "Check brake drag and battery cell balance"),
]
SINGLE = {
    ("motor_temp", 1): ("Motor running hot for its load — cooling fan / airflow issue",
                         "Clean cooling fins, check fan"),
    ("hydraulic_pressure", -1): ("Hydraulic pressure low for the lift demand — early leak or relief-valve wear",
                                 "Check fluid level and relief valve"),
    ("hydraulic_pressure", 1): ("Hydraulic pressure high for the lift demand — blocked filter / overload",
                                "Check filter and load weight"),
    ("mast_vibration", 1): ("Rising mast/drivetrain vibration — wear, misalignment or loose fasteners",
                            "Inspect mast rollers, chains and fasteners"),
    ("battery_voltage", -1): ("Battery voltage sagging more than the load explains — cell degradation",
                              "Schedule battery health test / equalize charge"),
    ("motor_current", 1): ("Motor drawing excess current for its duty — mechanical drag",
                           "Check brakes, wheels, drivetrain drag"),
}


# Temporal patterns only the autoencoder sees (oscillation / hunting inside normal limits)
PATTERN = {
    "mast_vibration": ("Oscillating mast vibration — slack lift chain, worn mast rollers or resonance",
                       "Check chain tension and mast rollers before the next heavy lift"),
    "hydraulic_pressure": ("Hydraulic pressure hunting — sticky control valve or air in the fluid",
                           "Bleed the hydraulic system; check the control valve"),
    "motor_current": ("Drive current hunting — controller instability or intermittent drag",
                      "Check drive-controller tuning, brakes and wheel bearings"),
    "motor_temp": ("Motor temperature cycling abnormally — cooling fan switching fault",
                   "Check fan relay and airflow"),
    "battery_voltage": ("Unstable battery voltage — loose terminal or failing cell",
                        "Check battery terminals and cell voltages"),
}


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 15:
        return None
    a, b = a[m], b[m]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def hint_for(inc, machine_detectors: dict, open_incidents: list) -> dict:
    """machine_detectors: sensor -> SensorDetector for the incident's truck.
    open_incidents: all currently-open incidents (whole fleet)."""
    machine = inc.machine
    sensor = inc.sensor
    evidence: list[str] = []

    # --- sensor/telemetry faults: not a process problem ---
    if inc.type in ("dropout", "stuck"):
        same_truck_dropouts = [i for i in open_incidents if i.machine == machine and i.type == "dropout"]
        if inc.type == "dropout" and len({i.sensor for i in same_truck_dropouts}) >= 3:
            return {"title": "Telematics gateway / CAN bus offline — several sensors silent on this truck",
                    "action": "Check the truck's telematics unit and CAN wiring", "evidence": evidence}
        others_ok = [s for s, d in machine_detectors.items()
                     if s != sensor and not d.drift_active and not d.dropout and not d.stuck]
        evidence.append(f"{len(others_ok)} other sensors on {machine} behaving normally")
        if inc.type == "dropout":
            return {"title": f"Sensor / telemetry fault on {SENSOR_BY_KEY[sensor].label.lower()} — not a process fault",
                    "action": "Check transmitter power, connector and CAN node", "evidence": evidence}
        return {"title": f"Frozen transmitter on {SENSOR_BY_KEY[sensor].label.lower()} — readings can't be trusted",
                "action": "Power-cycle / replace the transmitter; treat this signal as unavailable", "evidence": evidence}

    if inc.type == "spike":
        return {"title": "Isolated transient (EMI, pothole impact) — no correlated movement",
                "action": "No action", "evidence": evidence}
    if inc.type == "spike_burst":
        return {"title": "Intermittent spiking — likely loose connector or chafed cable",
                "action": "Inspect sensor wiring and connector", "evidence": evidence}

    # --- process anomalies: co-movement + signatures ---
    abnormal = {(i.sensor, i.direction) for i in open_incidents
                if i.machine == machine and i.type == "drift" and i.sensor}
    if inc.sensor and inc.type == "drift":
        abnormal.add((inc.sensor, inc.direction))

    corr_list = []
    if sensor in machine_detectors:
        base = np.array(list(machine_detectors[sensor].z_hist), dtype=float)
        for s, d in machine_detectors.items():
            if s == sensor:
                continue
            other = np.array(list(d.z_hist), dtype=float)
            n = min(len(base), len(other))
            if n < 15:
                continue
            r = _corr(base[-n:], other[-n:])
            if r is not None and abs(r) >= 0.6:
                corr_list.append((s, r))
    corr_list.sort(key=lambda p: -abs(p[1]))
    for s, r in corr_list[:2]:
        evidence.append(f"correlated with {machine} {SENSOR_BY_KEY[s].label.lower()} (r={r:+.2f})")

    for sig, title, action in SIGNATURES:
        if sig <= abnormal:
            parts = [f"{SENSOR_BY_KEY[s].label.lower()} {ARROW[d]}" for s, d in sorted(sig)]
            return {"title": title, "action": action, "evidence": [" with ".join(parts)] + evidence}

    # same sensor drifting on several trucks -> shared cause
    fleet_same = {i.machine for i in open_incidents if i.sensor == sensor and i.type == "drift"}
    if len(fleet_same) >= 2:
        return {"title": f"Fleet-wide: {SENSOR_BY_KEY[sensor].label.lower()} drifting on {len(fleet_same)} trucks — "
                         "shared cause (ambient, charger, fluid batch)",
                "action": "Check shared infrastructure before individual trucks", "evidence": evidence}

    if inc.sensor and (inc.sensor, inc.direction) in SINGLE:
        title, action = SINGLE[(inc.sensor, inc.direction)]
        if not corr_list:
            evidence.append("no other sensor on this truck moves with it — isolated to this component")
        return {"title": title, "action": action, "evidence": evidence}

    if inc.type == "multivariate":
        info = inc.info
        keys = info.get("top_sensor_keys", [])
        dirs = info.get("_dirs") or {}
        joint = {(k, d) for k, d in dirs.items()}
        if {("hydraulic_pressure", 1), ("motor_current", 1), ("battery_voltage", -1)} <= joint or \
           ({("hydraulic_pressure", 1), ("motor_current", 1)} <= joint and info.get("model") != "autoencoder"):
            return {"title": "Probable overload — lifting above rated capacity",
                    "action": "Check the load weight against the capacity plate; brief the operator",
                    "evidence": ["pressure ↑, motor current ↑, battery voltage ↓ together during lifts, "
                                 "each still inside its own limit",
                                 f"joint-pattern models: Gaussian {info.get('gauss_score', 0):.1f}, "
                                 f"Isolation Forest {info.get('if_score', 0):.1f}"] + evidence}
        if info.get("model") == "autoencoder" and keys:
            k = keys[0]
            title, action = PATTERN.get(k, (f"Abnormal signal shape on {SENSOR_BY_KEY[k].label.lower()}",
                                            "Inspect this component at the next stop"))
            ev = [f"temporal autoencoder could not reproduce the last 30 s of {SENSOR_BY_KEY[k].label.lower()}",
                  f"autoencoder score {info.get('ae_score', 0):.1f} vs Isolation Forest {info.get('if_score', 0):.1f} "
                  "— each single reading looks normal, the shape over time does not"]
            return {"title": title, "action": action, "evidence": ev + evidence}
        tops = info.get("top_sensors", [])
        return {"title": "Unusual combination of sensor readings (no single sensor out of range)",
                "action": "Review truck at next stop; " + (f"start with {', '.join(tops)}" if tops else "full inspection"),
                "evidence": evidence}
    return {"title": "Deviation isolated to this sensor", "action": "Inspect at next service", "evidence": evidence}
