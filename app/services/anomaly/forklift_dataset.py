"""
Labelled forklift dataset in SKAB format — so the exact SKAB benchmark (10 models vs the
fixed limit, same protocol, same metrics) also runs on forklift sensors.

Each CSV is one simulated forklift run, one row per second:

    datetime;motor_temp;hydraulic_pressure;mast_vibration;battery_voltage;motor_current;duty;anomaly;fault_type
    2026-01-05 08:00:01;58.214;141.52;0.3105;48.117;84.91;0.061;0;normal
    ...

  * the first 400 s are healthy (training rows, like SKAB), then one fault is injected
    (drift, dropout, stuck, bearing wear, hydraulic leak, battery fade, oscillation, overload)
  * `anomaly` = 1 while the injected fault is active (ground truth), `fault_type` names it
  * `duty` is the truck's load / work state as reported by its controller (a context column:
    models may use it, the fixed limit ignores it — nobody sets an alarm on "how hard the truck works")
  * an empty cell = the sensor sent nothing that second (dropout)
  * folders = fault families, used for grouped cross-validation exactly like SKAB's valve1 / valve2 / other:
        sensor/  dropout, stuck                      (the sensor is broken, not the truck)
        wear/    drift, bearing_wear, battery_fade   (slow degradation)
        load/    hydraulic_leak, overload, oscillation  (hydraulics / load handling)

    python -m app.services.anomaly.forklift_dataset data/forklift            # 40 runs (~44k rows)
    python -m app.services.anomaly.skab data/forklift --fast                  # benchmark it
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta, timezone

from .config import SENSORS
from .simulator import SensorFleetSimulator

FAMILIES = {
    "sensor": ("dropout", "stuck"),
    "wear": ("drift", "bearing_wear", "battery_fade"),
    "load": ("hydraulic_leak", "overload", "oscillation"),
}
COLUMNS = [s.key for s in SENSORS] + ["duty"]
MACHINE = [("ARK-F001", "FL-001")]


def one_run(scenario: str, seed: int, start: datetime, train_rows: int = 400) -> tuple[list[str], dict]:
    rng = random.Random(seed)
    sim = SensorFleetSimulator(seed=seed, anomaly_rate_per_min=0, start_time=start, machines=MACHINE)
    sim.random_injection = False
    rows = []
    burn = 120                                            # let the duty cycle settle before recording
    for _ in range(burn):
        sim.step()
    lead = train_rows + rng.randint(60, 240)
    fault = None
    total = None
    t_rec = 0
    while total is None or t_rec < total:
        if t_rec == lead:
            kw = {"duration": rng.randint(20, 80)} if scenario == "dropout" else {}   # a meaningful outage
            fault = sim.inject(scenario, machine="ARK-F001", start_in=1, source="dataset", **kw)
            total = (fault.end - sim.t) + lead + rng.randint(150, 350)
        fr = sim.step()
        t_rec += 1
        r = fr["readings"]["ARK-F001"]
        active = fault is not None and fault.start <= sim.t <= fault.end
        ts = (start + timedelta(seconds=t_rec)).strftime("%Y-%m-%d %H:%M:%S")
        vals = ["" if r[c] is None else f"{r[c]:.5g}" for c in COLUMNS]
        rows.append(";".join([ts] + vals + ["1" if active else "0", scenario if active else "normal"]))
    return rows, fault.as_dict()


def generate(out_dir: str, runs_per_scenario: int = 5, seed: int = 11, log=print) -> list[dict]:
    header = ";".join(["datetime"] + COLUMNS + ["anomaly", "fault_type"])
    day = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
    made = []
    jobs = [(fam, scen, i) for fam, scens in FAMILIES.items() for scen in scens for i in range(runs_per_scenario)]
    order = list(range(len(jobs)))
    random.Random(seed).shuffle(order)                     # recording order mixes fault types, like a real fleet
    for k, (family, scen, i) in enumerate(jobs):
        os.makedirs(os.path.join(out_dir, family), exist_ok=True)
        start = day + timedelta(hours=2 * order[k])           # runs never overlap in time
        rows, fault = one_run(scen, seed * 1000 + k + 1, start)
        path = os.path.join(out_dir, family, f"{scen}_{i}.csv")
        with open(path, "w", newline="") as f:
            f.write(header + "\n" + "\n".join(rows) + "\n")
        made.append({"file": f"{family}/{scen}_{i}.csv", "rows": len(rows), "fault": fault})
    log(f"wrote {len(made)} forklift runs ({sum(m['rows'] for m in made):,} rows) to {out_dir}")
    return made


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="data/forklift")
    ap.add_argument("--runs", type=int, default=5, help="runs per fault type")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    generate(a.out, a.runs, a.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
