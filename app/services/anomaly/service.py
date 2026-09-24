"""
AnomalyService — the live, in-process runtime behind /api/anomaly.

Owns one simulator + one engine, advances them on an asyncio task
(sharing the FastAPI event loop, like the forklift SimulationEngine),
broadcasts routed alerts over Socket.IO as "anomaly:alert", and exposes
plain-dict views the router (and the dashboard) consume.

It needs no database: the anomaly pipeline is fully in-memory, so the
ARK Predict dashboard works even before Postgres is set up.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone

import numpy as np

from .config import MACHINES, ROUTES, SENSOR_BY_KEY, SENSORS, TIERS, EngineConfig
from .engine import AnomalyEngine
from .evaluate import evaluate
from .simulator import SCENARIOS, SensorFleetSimulator


class AnomalyService:
    def __init__(self, seed: int | None = 7, anomaly_rate_per_min: float = 0.8,
                 tick_seconds: float = 0.5, speed: int = 2, cfg: EngineConfig | None = None,
                 oracle_feedback: bool = False, classifier=None) -> None:
        # classifier: ARK Predict v2 fault classifier — None (off), "shared" (the saved model, loaded in
        # the background) or a FaultClassifier instance
        # oracle_feedback: a simulated technician labels every alerted incident when it closes,
        # using the injected ground truth — used to measure how fast the feedback loop learns.
        self.oracle_feedback = oracle_feedback
        self._classifier_src = classifier
        self.clf_labels: list = []          # technician verdicts turned into classifier training windows
        self.seed = seed
        self.rate = anomaly_rate_per_min
        self.tick_seconds = tick_seconds
        self.speed = speed              # simulated seconds per tick
        self.cfg = cfg or EngineConfig()
        self.running = False
        self._task: asyncio.Task | None = None
        self._sio = None
        self._lock = threading.RLock()
        self._build()

    def _build(self) -> None:
        self.sim = SensorFleetSimulator(seed=self.seed, anomaly_rate_per_min=self.rate)
        self.engine = AnomalyEngine(self.cfg, seed=self.seed or 0)
        clf = self.classifier
        if clf is not None and self.cfg.use_classifier:
            self.engine.set_classifier(clf)
        self.warmup_end_t = 0
        self.ready = False
        self.started_at = datetime.now(timezone.utc).isoformat()

    def ensure_ready(self) -> None:
        """Warm-up: stream healthy data (no faults) and fit every baseline + the ML model."""
        with self._lock:
            if self.ready:
                return
            self.sim.random_injection = False
            frames = [self.sim.step() for _ in range(self.cfg.warmup_s)]
            self.engine.fit(frames)
            self.warmup_end_t = self.sim.t
            self.sim.random_injection = True
            self.ready = True

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------
    def step(self, n: int = 1) -> list[dict]:
        self.ensure_ready()
        notes = []
        with self._lock:
            for _ in range(n):
                frame = self.sim.step()
                notes += self.engine.process(frame, self.sim.timestamp)
                r = self.engine.router
                if r.resolved_notified:
                    done, r.resolved_notified = r.resolved_notified, []
                    if self.oracle_feedback:
                        for inc in done:
                            r.feedback.record(inc, self._oracle_label(inc), source="simulated technician", t=self.sim.t)
        return notes

    def _oracle_label(self, inc) -> str:
        """What a technician who knows the truth would say about a closed, alerted incident."""
        from .config import TIER_RANK
        sensors = {inc.sensor} if inc.sensor else set(inc.info.get("top_sensor_keys", [])) or {inc.sensor_hint}
        end = inc.closed_t or inc.last_active_t
        truth = [a for a in self.sim.anomalies if a.machine == inc.machine and (set(a.effects) & sensors)
                 and inc.opened_t <= a.end + 90 and end >= a.start - 2]
        if not truth:
            return "false_alarm"
        want = max(TIER_RANK[a.truth_tier] for a in truth)
        got = TIER_RANK[inc.max_tier]
        return "right_call" if got == want else ("too_high" if got > want else "too_low")

    # ------------------------------------------------------------------
    # technician feedback
    # ------------------------------------------------------------------
    @property
    def classifier(self):
        src = self._classifier_src
        if src == "shared":
            from .classifier import shared_classifier
            return shared_classifier()
        return src

    def give_feedback(self, incident_id: int, label: str, fault_type: str | None = None) -> dict:
        with self._lock:
            inc = self.engine.router.incidents.get(incident_id)
            if inc is None:
                raise KeyError(incident_id)
            entry = self.engine.router.feedback.record(inc, label, t=self.sim.t)
            lab = self._label_window(inc, label, fault_type)
        return {**entry, "table": self.engine.router.feedback.table(), "classifierLabel": lab}

    def _label_window(self, inc, label: str, fault_type: str | None) -> dict | None:
        """Technician verdict -> training data for the fault classifier (ARK Predict v2).
        'False alarm' teaches it that this window was normal; any other verdict confirms a real fault,
        of the type the technician names (or, if none given, the type the classifier itself suggested)."""
        from .classifier import FAULT_CLASSES
        clf = self.classifier
        if clf is None:
            return None
        if label == "false_alarm":
            cls = "normal"
        else:
            cls = fault_type or (inc.diagnosis or {}).get("fault")
        if cls not in FAULT_CLASSES:
            return {"added": 0, "reason": "no fault type given — pick what it actually was to teach the classifier"}
        t0, t1 = inc.opened_t, inc.closed_t or inc.last_active_t
        X = [f for t, f in self.engine.feat_hist.get(inc.machine, []) if t0 <= t <= t1]
        n = clf.add_feedback(np.array(X), cls) if X else 0
        rec = {"incidentId": inc.id, "machine": inc.machine, "faultType": cls, "rows": n, "verdict": label}
        if n:
            self.clf_labels.append(rec)
        return {"added": n, "faultType": cls, "pendingWindows": len(clf.feedback_y)}

    def model_status(self) -> dict:
        clf = self.classifier
        if clf is None:
            return {"enabled": False}
        return {"enabled": True, **clf.status(), "pendingLabels": list(reversed(self.clf_labels[-20:])),
                "inUse": self.engine.clf is clf and clf.trained}

    def retrain_classifier(self) -> dict:
        clf = self.classifier
        if clf is None or clf.base_X is None:
            raise RuntimeError("fault classifier not loaded")
        return clf.retrain()

    def feedback_table(self) -> dict:
        fb = self.engine.router.feedback
        return {"learned": fb.table(), "recent": list(reversed(fb.log[-30:]))}

    async def start(self, sio=None) -> None:
        if sio is not None:
            self._sio = sio
        self.ensure_ready()
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        print(f"[anomaly] started (x{self.speed} sim-seconds every {self.tick_seconds}s, ML={self.ml_label()})")

    async def pause(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def reset(self, seed: int | None = None) -> None:
        was_running = self.running
        await self.pause()
        with self._lock:
            if seed is not None:
                self.seed = seed
            self._build()
            self.ensure_ready()
        if was_running:
            await self.start()

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.tick_seconds)
                try:
                    notes = self.step(self.speed)
                    for n in notes:
                        await self._emit("anomaly:alert", n)
                except Exception as err:  # noqa: BLE001 — keep the stream alive across one bad tick
                    print(f"[anomaly] tick error: {err!r}")
        except asyncio.CancelledError:
            pass

    async def _emit(self, event: str, payload) -> None:
        if self._sio is not None:
            try:
                await self._sio.emit(event, payload)
            except Exception as err:  # noqa: BLE001
                print(f"[anomaly] emit failed: {err!r}")

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    def set_speed(self, speed: int) -> None:
        self.speed = max(1, min(30, int(speed)))

    def set_rate(self, rate: float) -> None:
        self.rate = max(0.0, min(10.0, float(rate)))
        self.sim.anomaly_rate_per_min = self.rate

    def inject(self, scenario: str, machine: str | None = None, sensor: str | None = None,
               duration: int | None = None, magnitude: float | None = None) -> dict:
        self.ensure_ready()
        with self._lock:
            a = self.sim.inject(scenario, machine=machine, sensor=sensor, duration=duration,
                                magnitude=magnitude, source="manual")
        return a.as_dict()

    # ------------------------------------------------------------------
    # views (plain dicts — camelCase, like the rest of the ARK Shield API)
    # ------------------------------------------------------------------
    def status(self) -> dict:
        r = self.engine.router
        return {
            "running": self.running, "ready": self.ready, "simT": self.sim.t,
            "simTime": self.sim.timestamp(), "monitoredS": max(0, self.sim.t - self.warmup_end_t),
            "speed": self.speed, "tickSeconds": self.tick_seconds, "faultRatePerMin": self.rate,
            "mlModel": self.ml_label(), "machines": len(self.engine.machines),
            "sensorsPerMachine": len(SENSORS), "readingsProcessed": self.engine.processed_readings,
            "detectorHits": r.stats["detector_hits"], "incidents": r.stats["incidents"],
            "openIncidents": len(r.open), "notifications": len(r.notifications),
            "urgentOpen": sum(1 for i in r.open.values() if i.tier == "URGENT"),
            "monitorOpen": sum(1 for i in r.open.values() if i.tier == "MONITOR"),
            "baselineAlerts": sum(1 for b in self.engine.baseline_alerts if b[0] > self.warmup_end_t),
            "suppressed": {k.replace("suppressed_", ""): v for k, v in r.stats.items() if k.startswith("suppressed_")},
            "feedbackLabels": len(r.feedback.log),
        }

    def ml_label(self) -> str:
        parts = []
        if self.cfg.use_isolation_forest:
            parts.append("Isolation Forest" if self.engine.model.kind == "isolation_forest" else "Mahalanobis")
        if self.cfg.use_gaussian:
            parts.append("Gaussian")
        if self.engine.ae is not None:
            parts.append("Autoencoder (MLP 64-12-64)" if self.engine.ae.kind == "mlp_autoencoder" else "Linear autoencoder (PCA)")
        clf = self.engine.clf
        if clf is not None and clf.trained:
            parts.append("Fault classifier (v2)")
        return " + ".join(parts) or "none"

    def ml_info(self) -> dict:
        ae = self.engine.ae
        return {
            "isolationForest": {"enabled": self.cfg.use_isolation_forest, "kind": self.engine.model.kind,
                                "trees": 150, "inputs": "z of 5 sensors smoothed over ~10 s + duty"},
            "gaussian": {"enabled": self.cfg.use_gaussian, "kind": "multivariate Gaussian (Mahalanobis distance)",
                         "inputs": "z of 5 sensors smoothed over ~10 s + duty"},
            "autoencoder": None if ae is None else {
                "kind": ae.kind, "layers": [ae.window * ae.n_feat, *ae.hidden, ae.window * ae.n_feat],
                "window_s": ae.window, "inputs": "last 30 s of [z of 5 sensors + duty]", **ae.train_info},
            "combine": "max of the normalized scores (1.0 = 99.5th percentile of healthy data)",
            "patternTrigger": {"score": self.cfg.ml_multivariate_score, "sustainS": self.cfg.ml_multivariate_s},
        }

    def config(self) -> dict:
        sc = self.cfg.severity
        return {
            "machines": [{"id": m, "assetNumber": a} for m, a in MACHINES],
            "sensors": [{"key": s.key, "label": s.label, "unit": s.unit, "criticality": s.criticality,
                         "decimals": s.decimals, "spikeZ": s.spike_z, "cusumK": s.cusum_k, "cusumH": s.cusum_h,
                         "stuckWindow": s.stuck_window} for s in SENSORS],
            "scenarios": list(SCENARIOS), "tiers": list(TIERS), "routes": ROUTES,
            "tierThresholds": {"monitorAt": sc.monitor_at, "urgentAt": sc.urgent_at},
            "typePrior": sc.type_prior,
            "ml": self.ml_info() if self.ready else None,
        }

    def machines(self) -> list[dict]:
        r = self.engine.router
        out = []
        for m, asset in MACHINES:
            incs = [i for i in r.open.values() if i.machine == m]
            worst = max(incs, key=lambda i: i.severity, default=None)
            last = {}
            for s in SENSORS:
                h = self.engine.history[m][s.key]
                if h:
                    t, v, exp, z = h[-1]
                    d = self.engine.detectors[m][s.key]
                    state = ("dropout" if d.dropout else "stuck" if d.stuck else "drift" if d.drift_active else "ok")
                    last[s.key] = {"value": v, "expected": exp, "z": z, "state": state}
            out.append({
                "id": m, "assetNumber": asset,
                "tier": worst.tier if worst else "OK", "severity": worst.severity if worst else 0,
                "openIncidents": len(incs),
                "headline": r._title(worst) if worst else "All sensors within expected range",
                "rootCause": worst.root_cause if worst else None,
                "diagnosis": self.engine.diagnosis(m) if (self.engine.clf is not None and self.engine.clf.trained) else None,
                "mlScore": round(self.engine.ml_last[m], 2), "ifScore": round(self.engine.if_last[m], 2),
                "aeScore": round(self.engine.ae_last[m], 2), "gaussScore": round(self.engine.g_last[m], 2),
                "latest": last,
            })
        return out

    def series(self, machine: str, points: int = 240) -> dict:
        if machine not in self.engine.history:
            raise KeyError(machine)
        points = max(30, min(self.cfg.history_points, points))
        r = self.engine.router
        t_now = self.sim.t
        t0 = t_now - points + 1
        sensors = {}
        for s in SENSORS:
            h = list(self.engine.history[machine][s.key])[-points:]
            marks = []
            for inc in r.incidents.values():
                if inc.machine != machine or inc.sensor != s.key:
                    continue
                end = inc.closed_t if inc.closed_t is not None else t_now
                if end < t0:
                    continue
                marks.append({"from": max(inc.opened_t, t0), "to": end, "type": inc.type, "tier": inc.max_tier,
                              "severity": inc.max_severity, "incidentId": inc.id, "open": inc.status == "OPEN"})
            sensors[s.key] = {
                "t": [p[0] for p in h], "value": [p[1] for p in h], "expected": [p[2] for p in h],
                "z": [p[3] for p in h], "incidents": marks[-12:],
            }
        truth = [a.as_dict() for a in self.sim.anomalies if a.machine == machine and a.end >= t0 and a.start <= t_now]
        ml = list(self.engine.ml_history[machine])[-points:]
        # pattern (multivariate) incidents belong to the whole truck: shade them on the sensor the model points at
        for inc in r.incidents.values():
            if inc.machine == machine and inc.sensor is None and inc.sensor_hint in sensors:
                end = inc.closed_t if inc.closed_t is not None else t_now
                if end >= t0:
                    sensors[inc.sensor_hint]["incidents"].append(
                        {"from": max(inc.opened_t, t0), "to": end, "type": inc.type, "tier": inc.max_tier,
                         "severity": inc.max_severity, "incidentId": inc.id, "open": inc.status == "OPEN"})
        return {"machine": machine, "tNow": t_now, "sensors": sensors, "injected": truth,
                "ml": {"t": [p[0] for p in ml], "score": [max(p[1], p[2], p[3]) for p in ml],
                       "iforest": [p[1] for p in ml], "autoencoder": [p[2] for p in ml], "gaussian": [p[3] for p in ml],
                       "trigger": self.cfg.ml_multivariate_score}}

    def alerts(self, limit: int = 50) -> list[dict]:
        return list(reversed(self.engine.router.notifications[-limit:]))

    def feed(self, limit: int = 80) -> list[dict]:
        return list(self.engine.router.feed)[:limit]

    def incidents(self, status: str = "open", limit: int = 100) -> list[dict]:
        r = self.engine.router
        items = list(r.open.values()) if status == "open" else list(r.incidents.values())
        items.sort(key=lambda i: (-i.severity if status == "open" else -i.opened_t))
        return [i.as_dict(self.sim.timestamp) for i in items[:limit]]

    def summary(self, include_faults: bool = False) -> dict:
        r = self.engine.router
        ev = evaluate(self.sim.anomalies, list(r.incidents.values()), r.notifications,
                      self.engine.baseline_alerts, self.warmup_end_t + 1, self.sim.t, dict(r.stats))
        if not include_faults:
            ev.pop("faults", None)
        by_tier = {t: 0 for t in TIERS}
        by_type: dict = {}
        by_machine: dict = {}
        for i in r.incidents.values():
            by_tier[i.max_tier] += 1
            by_type[i.type] = by_type.get(i.type, 0) + 1
            by_machine.setdefault(i.machine, {t: 0 for t in TIERS})[i.max_tier] += 1
        top = sorted((i for i in r.incidents.values() if i.max_tier == "URGENT"),
                     key=lambda i: -i.max_severity)[:8]
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": self.status(),
            "incidentsByTier": by_tier, "incidentsByType": by_type, "incidentsByMachine": by_machine,
            "topUrgent": [i.as_dict(self.sim.timestamp) for i in top],
            "evaluation": ev,
            "method": {
                "statistical": "context-aware per-sensor baselines (duty-cycle regression) + spike / CUSUM-drift / "
                               "dropout / stuck detectors, each with its own sigma-scaled threshold",
                "ml": self.ml_label() + ": Isolation Forest and a leave-one-out multivariate Gaussian score ~10 s-smoothed "
                      "residuals of all 5 sensors (joint shifts, e.g. overload); a temporal autoencoder "
                      "(MLP 180-64-12-64-180) reconstructs the shape of the last 30 s (e.g. oscillation); any of them "
                      "can open an 'abnormal pattern' incident",
                "severity": "explainable 0-100 score: type, magnitude, persistence, trend, spread, ML, "
                            "plus a bounded correction learned from technician verdicts",
                "routing": "IGNORE→log, MONITOR→maintenance queue, URGENT→page; confirmation, escalation-only "
                           "re-alerts, per-truck grouping, flap suppression, spike-burst aggregation",
            },
        }


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def report_markdown(rep: dict) -> str:
    st, ev = rep["status"], rep["evaluation"]
    af, cal = ev["alertFatigue"], ev["severityCalibration"]
    lines = [
        "# ARK Predict — Anomaly Detection Run Report",
        "",
        f"Generated {rep['generatedAt']} · simulated time monitored: {st['monitoredS'] / 3600:.2f} h · "
        f"{st['machines']} forklifts × {st['sensorsPerMachine']} sensors · ML model: {st['mlModel']}",
        "",
        "## Headline",
        "",
        f"- **{st['readingsProcessed']:,}** sensor readings processed",
        f"- **{af['detectorHits']:,}** raw detector hits → **{af['incidents']}** incidents → "
        f"**{af['notifications']}** notifications ({af['notificationsPerHour']}/h)",
        f"- Naive fixed-threshold alerter on the same stream: **{af['baselineFixedThresholdAlerts']}** alerts "
        f"({af['baselinePerHour']}/h, precision {_pct(af['baselinePrecision'])}) → "
        f"**{_pct(af['reductionVsBaseline'])} fewer alerts**, precision {_pct(af['notificationPrecision'])}",
        f"- Fault detection rate **{_pct(ev['detectionRate'])}**, anomaly-type accuracy **{_pct(ev['typeAccuracy'])}** "
        f"over {ev['faultsScored']} injected faults",
        f"- Severity: exact tier {_pct(cal['exactTierAccuracy'])}, within one tier {_pct(cal['withinOneTier'])}, "
        f"URGENT recall {_pct(cal['urgentRecall'])}",
        f"- False alarms: {ev['falseAlarms']['incidentsPerHour']} incidents/h, "
        f"{ev['falseAlarms']['notificationsPerHour']} notifications/h",
        "",
        "## Detection by fault type",
        "",
        "| Scenario | Injected | Detected | Correct type | Median latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in ev["perScenario"].items():
        lat = "—" if v["medianLatencyS"] is None else f"{v['medianLatencyS']} s"
        lines.append(f"| {k} | {v['n']} | {v['detected']} | {v['typeCorrect']} | {lat} |")
    lines += ["", "## Severity calibration (rows = ground truth, columns = assigned)", "",
              "| truth \\ assigned | IGNORE | MONITOR | URGENT | MISSED |", "|---|---:|---:|---:|---:|"]
    for tt, row in cal["confusion"].items():
        lines.append(f"| {tt} | {row['IGNORE']} | {row['MONITOR']} | {row['URGENT']} | {row['MISSED']} |")
    sup = af.get("suppressed") or {}
    lines += ["", "## Alert-fatigue controls", "",
              f"- Logged only (IGNORE tier): {sup.get('ignore_tier', 0)}",
              f"- Grouped into an existing truck alert: {sup.get('grouped', 0)}",
              f"- Recurrences re-opened instead of re-alerted: {sup.get('reopened', 0)}",
              "", "## Top urgent incidents", ""]
    for i in rep["topUrgent"]:
        rc = (i.get("rootCause") or {}).get("title", "")
        lines.append(f"- **#{i['id']} {i['machine']} · {i['sensorLabel']} — {i['type']}** "
                     f"(severity {i['maxSeverity']}): {rc}")
    if not rep["topUrgent"]:
        lines.append("- none")
    lines += ["", "## Method", ""] + [f"- **{k}**: {v}" for k, v in rep["method"].items()]
    return "\n".join(lines) + "\n"


anomaly_service = AnomalyService(classifier="shared")
