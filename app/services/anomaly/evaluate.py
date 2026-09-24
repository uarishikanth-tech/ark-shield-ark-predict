"""
Offline evaluation against ground truth.

    python -m app.services.anomaly.evaluate --hours 6 --seed 11
    python -m app.services.anomaly.evaluate --compare      # ML ablation: none / IF / Gaussian / AE / all

Runs the simulator with random fault injection, streams it through the
engine exactly as the live service does, then scores:

  * multi-type detection   per scenario: detected?  correct type?  latency
  * false alarms           incidents / notifications with no injected fault
  * severity calibration   assigned max tier vs ground-truth tier (confusion)
  * alert fatigue          detector hits -> incidents -> notifications, vs a
                           naive fixed-threshold alerter on the same stream
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from .config import TIER_RANK, TIERS

MATCH_AFTER_S = 90   # an incident opened up to this long after a fault ends still counts


def _overlaps(inc, a) -> bool:
    end = inc.closed_t if inc.closed_t is not None else inc.last_active_t
    return inc.opened_t <= a.end + MATCH_AFTER_S and end >= a.start - 2


def evaluate(anomalies, incidents, notifications, baseline_alerts, t_start: int, t_end: int,
             stats: dict | None = None) -> dict:
    scored = [a for a in anomalies if a.start >= t_start and a.end <= t_end - MATCH_AFTER_S]
    by_ms = defaultdict(list)
    for inc in incidents:
        by_ms[(inc.machine, inc.sensor)].append(inc)
        if inc.sensor is None:
            by_ms[(inc.machine, "*")].append(inc)

    per_type = defaultdict(lambda: Counter())
    latencies = defaultdict(list)
    confusion = {tt: {pt: 0 for pt in TIERS + ("MISSED",)} for tt in TIERS}
    matched_inc_ids: set = set()
    rows = []
    for a in scored:
        sensor_hits, type_hits, tiers, lat = 0, 0, [], []
        for s, exp_type in a.expected_types.items():
            cands = [i for i in by_ms[(a.machine, s)] + by_ms[(a.machine, "*")] if _overlaps(i, a)]
            if cands:
                sensor_hits += 1
                matched_inc_ids.update(i.id for i in cands)
                ok_types = exp_type if isinstance(exp_type, (tuple, list)) else (exp_type,)
                good = [i for i in cands if i.type in ok_types]
                # calibrate against the incident that *is* this fault when we have one
                tiers += [i.max_tier for i in (good or cands)]
                if good:
                    type_hits += 1
                    lat.append(max(0, min(i.detected_t for i in good) - a.start))
        detected = sensor_hits > 0
        type_ok = type_hits == len(a.expected_types)
        c = per_type[a.scenario]
        c["n"] += 1
        c["detected"] += int(detected)
        c["type_correct"] += int(type_ok)
        if lat:
            latencies[a.scenario].append(min(lat))
        pred = max(tiers, key=lambda x: TIER_RANK[x]) if tiers else "MISSED"
        # an IGNORE-truth fault that we logged-only or never surfaced is a correct outcome
        if a.truth_tier == "IGNORE" and pred == "MISSED":
            pred = "IGNORE"
        confusion[a.truth_tier][pred] += 1
        rows.append({**a.as_dict(), "detected": detected, "typeCorrect": type_ok, "assignedTier": pred})

    hours = max(1e-9, (t_end - t_start) / 3600)
    fp_inc = [i for i in incidents if i.id not in matched_inc_ids and i.opened_t >= t_start
              and not any(a.machine == i.machine and (i.sensor is None or i.sensor in a.effects)
                          and _overlaps(i, a) for a in anomalies)]
    fp_notified = [i for i in fp_inc if i.notified_tier]

    def _real(machine, sensor, t):
        return any(a.machine == machine and (sensor is None or sensor in a.effects)
                   and a.start - 2 <= t <= a.end + MATCH_AFTER_S for a in anomalies)

    notes = [n for n in notifications if n["t"] >= t_start]
    note_real = sum(1 for n in notes if _real(n["machine"], n["sensor"], n["t"]))
    base = [b for b in baseline_alerts if b[0] >= t_start]
    base_real = sum(1 for (t, m, s) in base if _real(m, s, t))

    n_total = sum(c["n"] for c in per_type.values())
    det_total = sum(c["detected"] for c in per_type.values())
    type_total = sum(c["type_correct"] for c in per_type.values())
    cal_total = sum(sum(v.values()) for v in confusion.values())
    cal_exact = sum(confusion[t][t] for t in TIERS)
    cal_within1 = sum(v for tt in TIERS for pt, v in confusion[tt].items()
                      if pt != "MISSED" and abs(TIER_RANK[tt] - TIER_RANK[pt]) <= 1)
    urgent_truth = sum(confusion["URGENT"].values())
    urgent_caught = confusion["URGENT"]["URGENT"]
    must_notify = sum(v for tt in ("MONITOR", "URGENT") for pt, v in confusion[tt].items() if pt in ("MONITOR", "URGENT"))
    must_total = sum(sum(confusion[tt].values()) for tt in ("MONITOR", "URGENT"))

    return {
        "window": {"startT": t_start, "endT": t_end, "hours": round(hours, 2)},
        "faultsScored": n_total,
        "detectionRate": round(det_total / n_total, 3) if n_total else None,
        "typeAccuracy": round(type_total / n_total, 3) if n_total else None,
        "perScenario": {
            k: {"n": c["n"], "detected": c["detected"], "typeCorrect": c["type_correct"],
                "medianLatencyS": (sorted(latencies[k])[len(latencies[k]) // 2] if latencies[k] else None)}
            for k, c in sorted(per_type.items())
        },
        "severityCalibration": {
            "confusion": confusion,
            "exactTierAccuracy": round(cal_exact / cal_total, 3) if cal_total else None,
            "withinOneTier": round(cal_within1 / cal_total, 3) if cal_total else None,
            "urgentRecall": round(urgent_caught / urgent_truth, 3) if urgent_truth else None,
            "actionableRecall": round(must_notify / must_total, 3) if must_total else None,
        },
        "falseAlarms": {
            "incidentsPerHour": round(len(fp_inc) / hours, 2),
            "notificationsPerHour": round(len(fp_notified) / hours, 2),
        },
        "alertFatigue": {
            "detectorHits": (stats or {}).get("detector_hits"),
            "incidents": sum(1 for i in incidents if i.opened_t >= t_start),
            "notifications": len(notes),
            "notificationsPerHour": round(len(notes) / hours, 1),
            "notificationPrecision": round(note_real / len(notes), 3) if notes else None,
            "baselineFixedThresholdAlerts": len(base),
            "baselinePerHour": round(len(base) / hours, 1),
            "baselinePrecision": round(base_real / len(base), 3) if base else None,
            "reductionVsBaseline": round(1 - len(notes) / len(base), 3) if base else None,
            "suppressed": {k.replace("suppressed_", ""): v for k, v in (stats or {}).items()
                           if k.startswith("suppressed_")},
        },
        "faults": rows,
    }


ML_CHOICES = ("all", "none", "if", "gauss", "ae")


def _service(hours, seed, rate, ml="all", v2=False):
    from .config import EngineConfig
    from .service import AnomalyService
    cfg = EngineConfig()
    cfg.use_isolation_forest = ml in ("all", "both", "if")
    cfg.use_gaussian = ml in ("all", "both", "gauss")
    cfg.use_autoencoder = ml in ("all", "both", "ae")
    clf = None
    if v2:
        from .classifier import FaultClassifier
        clf = FaultClassifier.load()
    svc = AnomalyService(seed=seed, anomaly_rate_per_min=rate, cfg=cfg, classifier=clf)
    svc.ensure_ready()
    svc.step(int(hours * 3600))
    return svc


def run(hours: float = 4.0, seed: int = 11, rate: float = 1.2, ml: str = "all", v2: bool = False) -> dict:
    """ml: 'all' (default: Isolation Forest + Gaussian + autoencoder), 'none' (statistics only),
    or one model alone: 'if', 'gauss', 'ae'. v2: also run the supervised fault classifier (ARK Predict v2)."""
    return _service(hours, seed, rate, ml, v2).summary(include_faults=True)


def v2_extras(svc) -> dict:
    """What v2 adds on top of the summary: is the named fault right, and how fast are dangerous faults paged?"""
    from .classifier import FAULT_CLASSES
    r = svc.engine.router
    diag_right = diag_total = diag_named = 0
    per = {}
    ttu = []
    for a in svc.sim.anomalies:
        if a.start <= svc.warmup_end_t or a.scenario not in FAULT_CLASSES:
            continue
        sens = set(a.effects)
        incs = [i for i in r.incidents.values() if i.machine == a.machine and i.opened_t <= a.end + 60
                and (i.closed_t or i.last_active_t) >= a.start and i.notified_tier
                and (i.sensor in sens or (i.sensor is None and (i.sensor_hint in sens or not i.sensor_hint)))]
        named = [i.diagnosis["fault"] for i in incs if i.diagnosis]
        if incs:
            diag_total += 1
            ok = bool(named) and max(set(named), key=named.count) == a.scenario
            diag_right += ok
            diag_named += bool(named)
            p = per.setdefault(a.scenario, [0, 0])
            p[0] += ok
            p[1] += 1
        if a.truth_tier == "URGENT":
            urg = [n["t"] for n in r.notifications if n["machine"] == a.machine and n["tier"] == "URGENT"
                   and a.start <= n["t"] <= a.end + 60]
            if urg:
                ttu.append(min(urg) - a.start)
    return {"diagnosisAccuracy": round(diag_right / diag_total, 3) if diag_total else None,
            "diagnosisNamed": round(diag_named / diag_total, 3) if diag_total else None,
            "diagnosisPrecision": round(diag_right / diag_named, 3) if diag_named else None,
            "diagnosedFaults": diag_total, "perScenario": {k: f"{v[0]}/{v[1]}" for k, v in per.items()},
            "medianSecondsToUrgent": float(np.median(ttu)) if ttu else None, "urgentPaged": len(ttu)}


def compare_v2(hours: float = 3.0, seed: int = 11, rate: float = 1.2) -> dict:
    """ARK Predict v1 (detectors + unsupervised ML) vs v2 (+ supervised fault classifier), same stream."""
    out = {}
    for name, v2 in (("v1", False), ("v2", True)):
        svc = _service(hours, seed, rate, "all", v2)
        ev = svc.summary(include_faults=True)["evaluation"]
        out[name] = {"detectionRate": ev["detectionRate"], "typeAccuracy": ev["typeAccuracy"],
                     "exactTier": ev["severityCalibration"]["exactTierAccuracy"],
                     "urgentRecall": ev["severityCalibration"]["urgentRecall"],
                     "falseAlarmsPerHour": ev["falseAlarms"]["notificationsPerHour"],
                     "alertsSent": ev["alertFatigue"]["notifications"],
                     "alertPrecision": ev["alertFatigue"]["notificationPrecision"],
                     "faults": ev["faultsScored"], **v2_extras(svc)}
    return out


def compare(hours: float = 3.0, seed: int = 11, rate: float = 1.2) -> dict:
    """Ablation: same stream, same faults, four model configurations."""
    out = {}
    for ml in ("none", "if", "gauss", "ae", "all"):
        ev = run(hours, seed, rate, ml)["evaluation"]
        osc = ev["perScenario"].get("oscillation", {"n": 0, "detected": 0, "typeCorrect": 0})
        ovl = ev["perScenario"].get("overload", {"n": 0, "detected": 0, "typeCorrect": 0})
        out[ml] = {"detectionRate": ev["detectionRate"], "typeAccuracy": ev["typeAccuracy"],
                   "exactTier": ev["severityCalibration"]["exactTierAccuracy"],
                   "urgentRecall": ev["severityCalibration"]["urgentRecall"],
                   "falseAlarmsPerHour": ev["falseAlarms"]["notificationsPerHour"],
                   "oscillation": f"{osc['detected']}/{osc['n']}", "overload": f"{ovl['detected']}/{ovl['n']}",
                   "faults": ev["faultsScored"]}
    return out


def compare_markdown(hours: float = 3.0, seed: int = 11, rate: float = 1.2) -> str:
    names = {"none": "Statistics only", "if": "+ Isolation Forest", "gauss": "+ Gaussian", "ae": "+ Autoencoder",
             "all": "+ all three (default)"}
    res = compare(hours, seed, rate)
    lines = [f"# ML ablation — {hours} simulated h, seed {seed}", "",
             "| Models | Faults found | Right type | Right tier | Urgent caught | False alarms/h | Oscillation | Overload |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    pct = lambda x: "—" if x is None else f"{x * 100:.0f}%"
    for k, v in res.items():
        lines.append(f"| {names[k]} | {pct(v['detectionRate'])} | {pct(v['typeAccuracy'])} | {pct(v['exactTier'])} | "
                     f"{pct(v['urgentRecall'])} | {v['falseAlarmsPerHour']} | {v['oscillation']} | {v['overload']} |")
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--rate", type=float, default=1.2, help="random faults injected per simulated minute (fleet)")
    ap.add_argument("--ml", choices=list(ML_CHOICES), default="all",
                    help="which ML models to run: all, none (statistics only), if / gauss / ae (one model alone)")
    ap.add_argument("--compare", action="store_true", help="ablation: statistics only, each model alone, all together")
    ap.add_argument("--json", action="store_true", help="print the full JSON report")
    args = ap.parse_args()
    if args.compare:
        print(compare_markdown(args.hours, args.seed, args.rate))
        return
    rep = run(args.hours, args.seed, args.rate, args.ml)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return
    from .service import report_markdown
    print(report_markdown(rep))


if __name__ == "__main__":  # pragma: no cover
    main()
