"""
Replay real, labelled sensor data through the ARK Predict detectors.

Built for the SKAB benchmark (Skoltech Anomaly Benchmark — the dataset the
HTH-ML-10 brief suggests): one CSV per experiment, 1 row per second,
';'-separated, numeric sensor columns plus an `anomaly` (0/1) label.
Any CSV with that shape works (comma or semicolon, any sensor columns).

    # 1) download SKAB:  https://github.com/waico/SKAB  (folder data/)
    # 2) run:
    python -m app.services.anomaly.replay path/to/SKAB/data            # every CSV under the folder
    python -m app.services.anomaly.replay one_file.csv --train-rows 400

    # no dataset handy? generate a SKAB-format file from the simulator and replay it:
    python -m app.services.anomaly.replay --make-demo demo_skab.csv && \\
    python -m app.services.anomaly.replay demo_skab.csv

Per file: the first --train-rows rows (only those labelled normal) fit each
column's baseline, the Isolation Forest, the Gaussian and the autoencoder;
the rest is streamed row by row exactly like the live service. A row is
flagged when any detector (spike / drift / dropout / stuck) or the ML
pattern rule (score >= 1.6 for 20 of the last 30 s) is active.

Reported against the labels: point-wise precision / recall / F1, event recall
(labelled anomaly segments that got at least one flag), median detection
delay, and false-alarm events per hour — next to a fixed 3-sigma-on-raw-value
baseline on the same rows. No duty/load channel exists in generic data, so
each column's "expected value" is simply its healthy mean.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import deque

import numpy as np

from .config import EngineConfig, SensorSpec
from .detectors import SensorDetector
from .ml import MachineModel, SequenceAutoencoder

LABEL_COLS = ("anomaly", "label", "is_anomaly")
IGNORE_COLS = ("changepoint", "datetime", "timestamp", "time", "date")


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_csv(path: str, columns: list[str] | None = None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        head = f.read(4096)
        f.seek(0)
        delim = ";" if head.count(";") > head.count(",") else ","
        rows = list(csv.reader(f, delimiter=delim))
    header, body = [h.strip() for h in rows[0]], rows[1:]
    low = [h.lower() for h in header]
    label_idx = next((i for i, h in enumerate(low) if h in LABEL_COLS), None)
    feats = []
    for i, h in enumerate(header):
        if i == label_idx or low[i] in IGNORE_COLS:
            continue
        if columns and h not in columns:
            continue
        try:
            float(body[0][i])
            feats.append(i)
        except (ValueError, IndexError):
            continue
    X = np.full((len(body), len(feats)), np.nan)
    y = np.zeros(len(body), dtype=int)
    for r, row in enumerate(body):
        for c, i in enumerate(feats):
            try:
                X[r, c] = float(row[i])
            except (ValueError, IndexError):
                pass
        if label_idx is not None:
            try:
                y[r] = int(float(row[label_idx]) > 0.5)
            except (ValueError, IndexError):
                pass
    return [header[i] for i in feats], X, y, label_idx is not None


# ----------------------------------------------------------------------
# one file through the pipeline
# ----------------------------------------------------------------------
def replay_array(names: list[str], X: np.ndarray, train_rows: int = 400, cfg: EngineConfig | None = None,
                 use_ml: bool = True, strict_train_rows: bool = False) -> dict:
    cfg = cfg or EngineConfig()
    n, k = X.shape
    if not strict_train_rows:
        train_rows = min(train_rows, max(60, n // 3))
    Xtr = X[:train_rows]

    specs, dets = [], []
    for j, name in enumerate(names):
        col = Xtr[:, j]
        col = col[~np.isnan(col)]
        mad = float(np.median(np.abs(col - np.median(col)))) * 1.4826 if len(col) else 1.0
        spec = SensorSpec(key=name, label=name, unit="", base=float(np.median(col)) if len(col) else 0.0,
                          load_gain=0.0, noise=max(mad, 1e-6))
        d = SensorDetector(spec, cfg)
        feats = np.tile([1.0, 0.0, 0.0, 0.0], (len(col), 1))
        d.fit(feats, col)
        specs.append(spec)
        dets.append(d)
    x_ctx = np.array([1.0, 0.0, 0.0, 0.0])

    # training z for the ML models
    Ztr = np.column_stack([(Xtr[:, j] - d.expected(x_ctx)) / d.sigma for j, d in enumerate(dets)])
    Ztr = np.nan_to_num(Ztr)
    smooth = np.zeros_like(Ztr)
    acc = np.zeros(k)
    for i in range(len(Ztr)):
        acc += cfg.if_smooth * (np.clip(Ztr[i], -6, 6) - acc)
        smooth[i] = acc
    P = np.column_stack([smooth, np.zeros(len(smooth))])
    iforest = MachineModel(cfg.ml_contamination, 0)
    gauss = MachineModel(cfg.ml_contamination, 0, kind="mahalanobis", leave_one_out=True)
    ae = SequenceAutoencoder(k, cfg.ae_window, seed=0)
    if use_ml:
        iforest.fit(P)
        gauss.fit(P)
        W = SequenceAutoencoder.windows(np.column_stack([np.clip(Ztr, -3, 3), np.zeros(len(Ztr))]), cfg.ae_window)
        if len(W) >= 50:
            ae.fit(W)

    def norm(model, row):
        raw = float(model._raw(row.reshape(1, -1))[0])
        return max(0.0, (raw - model._p50) / max(1e-9, model._p995 - model._p50))

    flags = np.zeros(n, dtype=bool)
    reasons = [""] * n
    zs = np.zeros(k)
    win: deque = deque(maxlen=cfg.ae_window)
    hist: deque = deque(maxlen=cfg.ml_multivariate_window)
    for t in range(train_rows, n):
        zrow = np.zeros(k)
        why = []
        for j, d in enumerate(dets):
            v = X[t, j]
            obs = d.update(t, None if math.isnan(v) else float(v), x_ctx)
            zrow[j] = 0.0 if obs.z is None else obs.z
            for sg in obs.signals:
                why.append(f"{sg.type}:{names[j]}")
        if use_ml:
            zs += cfg.if_smooth * (np.clip(zrow, -6, 6) - zs)
            p = np.append(zs, 0.0)
            score = max(norm(iforest, p), norm(gauss, p))
            win.append(list(np.clip(zrow, -3, 3)) + [0.0])
            if ae.trained and len(win) == cfg.ae_window:
                sc, _ = ae.score(np.array([np.ravel(win)]))
                score = max(score, float(sc[0]))
            hist.append(score >= cfg.ml_multivariate_score)
            if sum(hist) >= cfg.ml_multivariate_s:
                why.append("ml_pattern")
        if why:
            flags[t] = True
            reasons[t] = ",".join(sorted(set(w.split(":")[0] for w in why)))
    # naive baseline: fixed |x - mean| > 3 std on raw values, per column
    mu, sd = np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0) + 1e-9
    base = np.zeros(n, dtype=bool)
    base[train_rows:] = np.nan_to_num(np.abs(X[train_rows:] - mu) > 3 * sd).any(axis=1)
    return {"flags": flags, "baseline": base, "train_rows": train_rows, "reasons": reasons}


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def score(flags: np.ndarray, y: np.ndarray, t0: int, slack: int = 30) -> dict:
    f, yy = flags[t0:], y[t0:]
    tp = int((f & (yy == 1)).sum())
    fp = int((f & (yy == 0)).sum())
    fn = int((~f & (yy == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    segs = _segments(yy == 1)
    hit, delays = 0, []
    for a, b in segs:
        idx = np.where(f[a:b + 1])[0]
        if len(idx):
            hit += 1
            delays.append(int(idx[0]))
    # false-alarm events: flagged runs that don't touch any labelled segment (± slack)
    near = np.zeros(len(yy), dtype=bool)
    for a, b in segs:
        near[max(0, a - slack): b + slack + 1] = True
    fa = sum(1 for a, b in _segments(f) if not near[a:b + 1].any())
    hours = len(yy) / 3600
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "segments": len(segs), "segmentsHit": hit,
            "medianDelayS": (sorted(delays)[len(delays) // 2] if delays else None),
            "falseAlarmEvents": fa, "falseAlarmsPerHour": round(fa / hours, 2) if hours else None, "rows": len(yy)}


def run_path(path: str, train_rows: int = 400, columns: list[str] | None = None, use_ml: bool = True) -> dict:
    files = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(".csv")]
    else:
        files = [path]
    per, tot = [], {"ours": [0, 0, 0, 0, 0, 0, 0.0], "base": [0, 0, 0, 0, 0, 0, 0.0]}
    for fp in files:
        names, X, y, labelled = load_csv(fp, columns)
        if not names or not labelled or y.sum() == 0:
            continue
        res = replay_array(names, X, train_rows, use_ml=use_ml)
        t0 = res["train_rows"]
        ours, base = score(res["flags"], y, t0), score(res["baseline"], y, t0)
        per.append({"file": os.path.relpath(fp, path) if os.path.isdir(path) else os.path.basename(fp),
                    "columns": len(names), "ours": ours, "baseline": base})
        for key, s in (("ours", ours), ("base", base)):
            acc = tot[key]
            acc[0] += s["tp"]; acc[1] += s["fp"]; acc[2] += s["fn"]
            acc[3] += s["segments"]; acc[4] += s["segmentsHit"]; acc[5] += s["falseAlarmEvents"]; acc[6] += s["rows"] / 3600

    def agg(a):
        tp, fp, fn, segs, hit, fa, hrs = a
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                "eventRecall": round(hit / segs, 3) if segs else None,
                "falseAlarmsPerHour": round(fa / hrs, 2) if hrs else None}
    return {"files": per, "overall": {"ours": agg(tot["ours"]), "baseline": agg(tot["base"])}}


# ----------------------------------------------------------------------
# demo data in SKAB format (from the simulator)
# ----------------------------------------------------------------------
def make_demo(path: str, seed: int = 5, seconds: int = 1500) -> None:
    from .service import AnomalyService
    svc = AnomalyService(seed=seed, anomaly_rate_per_min=0)
    svc.ensure_ready()
    m = "ARK-F002"
    plan = [(450, "drift", "motor_temp", 200), (800, "stuck", "hydraulic_pressure", 90),
            (1050, "oscillation", "mast_vibration", 180), (1300, "dropout", "motor_current", 40)]
    rows, t0 = [], svc.sim.t
    for step in range(seconds):
        for at, scen, sen, dur in plan:
            if step == at:
                svc.sim.inject(scen, machine=m, sensor=sen, duration=dur)
        fr = svc.sim.step()
        r = fr["readings"][m]
        lab = int(any(a.machine == m and a.start <= svc.sim.t <= a.end for a in svc.sim.anomalies))
        rows.append([fr["ts"]] + ["" if r[s] is None else r[s] for s in
                                  ("motor_temp", "hydraulic_pressure", "mast_vibration", "battery_voltage", "motor_current")]
                    + [r["duty"], lab, 0])
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["datetime", "Temperature", "Pressure", "Accelerometer1RMS", "Voltage", "Current", "Load",
                    "anomaly", "changepoint"])
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows, {sum(r[-2] for r in rows)} labelled anomalous (SKAB format)")


def _fmt(x):
    return "—" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="a CSV file or a folder of CSVs (e.g. SKAB/data)")
    ap.add_argument("--train-rows", type=int, default=400, help="leading rows used to learn 'healthy' (default 400)")
    ap.add_argument("--columns", nargs="*", help="only use these sensor columns")
    ap.add_argument("--no-ml", action="store_true", help="statistical detectors only (ablation)")
    ap.add_argument("--make-demo", metavar="OUT_CSV", help="write a SKAB-format demo file from the simulator and exit")
    args = ap.parse_args()
    if args.make_demo:
        make_demo(args.make_demo)
        return
    if not args.path:
        ap.error("give a CSV file or folder (or --make-demo OUT.csv)")
    res = run_path(args.path, args.train_rows, args.columns, use_ml=not args.no_ml)
    if not res["files"]:
        print("No labelled CSVs found (need a numeric sensor table with an 'anomaly' column).")
        sys.exit(1)
    print(f"# ARK Predict replay — {len(res['files'])} file(s)\n")
    print("| File | Rows | F1 (ours) | F1 (fixed 3σ) | Events hit | Delay | False-alarm events |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for f in res["files"]:
        o, b = f["ours"], f["baseline"]
        print(f"| {f['file']} | {o['rows']} | {o['f1']:.2f} | {b['f1']:.2f} | {o['segmentsHit']}/{o['segments']} | "
              f"{_fmt(o['medianDelayS'])} s | {o['falseAlarmEvents']} (fixed: {b['falseAlarmEvents']}) |")
    o, b = res["overall"]["ours"], res["overall"]["baseline"]
    print("\n| Overall | Precision | Recall | F1 | Event recall | False alarms / h |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, s in (("ARK Predict", o), ("Fixed 3σ threshold", b)):
        print(f"| {name} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | {_fmt(s['eventRecall'])} | "
              f"{_fmt(s['falseAlarmsPerHour'])} |")


if __name__ == "__main__":  # pragma: no cover
    main()
