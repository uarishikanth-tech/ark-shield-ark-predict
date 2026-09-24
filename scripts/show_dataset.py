"""
Show the real dataset the ML models are fed (SKAB) — for demos / judges.

    python scripts/show_dataset.py                 # summary of all 34 experiments
    python scripts/show_dataset.py valve1/1        # + the first rows of one experiment
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.anomaly import skab  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SKAB", "data")


def main():
    exps = skab.load(DATA)
    rows = sum(len(e.y) for e in exps)
    print(f"\nDataset folder : {DATA}")
    print(f"Experiments    : {len(exps)} CSV files (one real pump run each)")
    print(f"Total rows     : {rows:,} (one row = one second)")
    print(f"Sensors fed in : {', '.join(skab.SENSORS)}")
    print(f"Label column   : anomaly (0 = normal, 1 = fault)")
    print(f"Train / test   : first {skab.TRAIN_ROWS} rows of each file train the models, the rest is test\n")

    print(f"{'experiment':<11} {'rows':>5} {'train':>5} {'test':>5} {'fault rows':>10}  fault seconds")
    print("-" * 62)
    for e in exps:
        segs = skab._segments(e.y.astype(bool))
        span = ", ".join(f"{a}-{b}" for a, b in segs)
        print(f"{e.name:<11} {len(e.y):>5} {skab.TRAIN_ROWS:>5} {len(e.y) - skab.TRAIN_ROWS:>5} {int(e.y.sum()):>10}  {span}")

    if len(sys.argv) > 1:
        name = sys.argv[1]
        e = next((x for x in exps if x.name == name), None)
        if e is None:
            sys.exit(f"unknown experiment {name}")
        a = int(e.y.argmax()) if e.y.any() else 0
        short = ["Accel1", "Accel2", "Current", "Pressure", "Temp", "Thermo", "Voltage", "Flow"]
        print(f"\nFirst rows of {name} — exactly what the models receive:")
        print("  sec " + " ".join(f"{s:>9}" for s in short) + "  anomaly")
        for i in list(range(5)) + list(range(max(5, a - 2), a + 3)):
            if i == max(5, a - 2):
                print("  ...")
            print(f"{i:5d} " + " ".join(f"{v:9.4g}" for v in e.X[i]) + f"  {e.y[i]:>7}")
        print(f"\n(the fault starts at second {a}: the anomaly column switches from 0 to 1)")


if __name__ == "__main__":
    main()
