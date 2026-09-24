"""
Incident manager + tiered alert routing with alert-fatigue controls.

Detector *signals* (which fire every second while a condition lasts)
are turned into *incidents* (one per truck + sensor + anomaly type), and
incidents into *notifications* only when a human actually needs to act:

  1. Severity tiers     IGNORE -> logged only; MONITOR -> maintenance queue;
                        URGENT -> page on-call technician + supervisor.
  2. Confirmation       a MONITOR incident must persist min_confirm_s
                        before anyone is told (URGENT goes immediately).
  3. One alert per incident, re-notified only on ESCALATION
                        (MONITOR -> URGENT), never on de-escalation.
  4. Machine grouping   a new incident on a truck that already has an open,
                        notified incident is attached to that alert (as
                        extra root-cause evidence) instead of paging again,
                        unless it raises the truck's tier.
  5. Flap suppression   an incident that recurs within reopen_window_s of
                        resolving is re-opened, not re-alerted.
  6. Spike bursts       lone spikes are logged; N spikes on one sensor within
                        a window become a single "spike_burst" incident.

Every suppression is counted by reason, so the dashboard can show
exactly how much noise was kept away from operators.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from .config import MACHINES, ROUTES, SENSOR_BY_KEY, TIER_RANK, EngineConfig
from .rootcause import hint_for
from .feedback import FeedbackLearner
from .severity import PROCESS_TYPES, score_incident, tier_for
from .classifier import FAULT_LABEL

# ARK Predict v2: known faults that are dangerous enough to page someone as soon as the
# supervised classifier is confident it recognises them
DANGEROUS_FAULTS = {"bearing_wear", "hydraulic_leak"}

ASSET = dict(MACHINES)


@dataclass
class Incident:
    id: int
    machine: str
    sensor: str | None
    type: str
    opened_t: int
    last_active_t: int
    status: str = "OPEN"
    closed_t: int | None = None
    direction: int = 0
    peak_z: float = 0.0
    peak_z_abs: float = 0.0
    spike_count: int = 0
    ml_peak: float = 0.0
    severity: int = 0
    tier: str = "IGNORE"
    max_tier: str = "IGNORE"
    max_severity: int = 0
    notified_tier: str | None = None
    group_id: int | None = None
    components: dict = field(default_factory=dict)
    explanation: str = ""
    root_cause: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    tier_history: list = field(default_factory=list)
    reopened: int = 0
    sensor_hint: str | None = None   # multivariate incidents: sensor the ML model points at
    detected_t: int | None = None   # when the pipeline first knew (opened_t may be back-dated, e.g. burst start)
    feedback: str | None = None      # technician label, if any (feedback.py)
    diagnosis: dict | None = None    # ARK Predict v2: what the supervised classifier says it is

    def __post_init__(self):
        if self.detected_t is None:
            self.detected_t = self.last_active_t

    @property
    def key(self):
        return (self.machine, self.sensor, self.type)

    def as_dict(self, ts_fn=None) -> dict:
        spec = SENSOR_BY_KEY.get(self.sensor)
        return {
            "id": self.id, "machine": self.machine, "assetNumber": ASSET.get(self.machine),
            "sensor": self.sensor, "sensorHint": self.sensor_hint,
            "sensorLabel": spec.label if spec else (SENSOR_BY_KEY[self.sensor_hint].label + " (pattern)"
                                                    if self.sensor_hint in SENSOR_BY_KEY else "Multiple sensors"),
            "type": self.type, "status": self.status,
            "openedT": self.opened_t, "detectedT": self.detected_t, "lastActiveT": self.last_active_t,
            "closedT": self.closed_t,
            "openedAt": ts_fn(self.opened_t) if ts_fn else None,
            "durationS": self.last_active_t - self.opened_t + 1,
            "direction": self.direction, "peakZ": round(self.peak_z, 2),
            "severity": self.severity, "maxSeverity": self.max_severity,
            "tier": self.tier, "maxTier": self.max_tier, "route": ROUTES[self.tier],
            "notifiedTier": self.notified_tier, "groupId": self.group_id,
            "components": self.components, "explanation": self.explanation,
            "rootCause": self.root_cause, "info": self.info, "tierHistory": self.tier_history[-6:],
            "feedback": self.feedback, "mlDiagnosis": self.diagnosis,
        }


class AlertRouter:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.incidents: dict[int, Incident] = {}
        self.open: dict[tuple, Incident] = {}
        self.recent_closed: dict[tuple, Incident] = {}
        self.notifications: list[dict] = []
        self.feed: deque = deque(maxlen=400)  # notifications + resolutions + logged-only events
        self._next_inc = 1
        self._next_note = 1
        self.spike_times: dict[tuple, deque] = defaultdict(deque)
        self.ml_run: dict[str, int] = defaultdict(int)
        self._recent_spikes: deque = deque(maxlen=200)
        self._ml_hist: dict[str, deque] = {}
        self.feedback = FeedbackLearner(cfg.severity.monitor_at, cfg.severity.urgent_at)
        self.resolved_notified: list = []     # incidents a human was told about that just closed (for review)
        self._votes: dict = {}                # incident id -> {fault: (confirmed seconds, sum of confidence)}
        self.stats = defaultdict(int)

    # ------------------------------------------------------------------
    def _open(self, t: int, machine: str, sensor: str | None, typ: str) -> Incident:
        key = (machine, sensor, typ)
        if key in self.open:
            return self.open[key]
        prev = self.recent_closed.get(key)
        if prev and prev.closed_t is not None and t - prev.closed_t <= self.cfg.routing.reopen_window_s:
            prev.status, prev.closed_t = "OPEN", None
            prev.reopened += 1
            prev.last_active_t = t
            self.open[key] = prev
            del self.recent_closed[key]
            self.stats["suppressed_reopened"] += 1
            return prev
        inc = Incident(self._next_inc, machine, sensor, typ, t, t)
        self._next_inc += 1
        self.incidents[inc.id] = inc
        self.open[key] = inc
        self.stats["incidents"] += 1
        return inc

    # ------------------------------------------------------------------
    def process(self, t: int, ts: str, signals: dict, ml_scores: dict, detectors: dict, ts_fn,
                ml_info: dict | None = None) -> list[dict]:
        """signals: {(machine, sensor): [Signal]}; ml_scores: {machine: float};
        detectors: {machine: {sensor: SensorDetector}}. Returns new notifications."""
        cfg, rcfg = self.cfg, self.cfg.routing
        touched: set = set()

        for (machine, sensor), sigs in signals.items():
            for sg in sigs:
                self.stats["detector_hits"] += 1
                if sg.type == "spike":
                    self._handle_spike(t, machine, sensor, sg, ts_fn)
                    continue
                inc = self._open(t, machine, sensor, sg.type)
                inc.last_active_t = t
                if sg.type == "drift":
                    inc.direction = sg.direction
                    if abs(sg.z) >= inc.peak_z_abs:
                        inc.peak_z_abs, inc.peak_z = abs(sg.z), sg.z
                    inc.info = dict(sg.info)
                else:
                    inc.info = dict(sg.info)
                    if abs(sg.z) > inc.peak_z_abs:
                        inc.peak_z_abs, inc.peak_z = abs(sg.z), sg.z
                touched.add(inc.key)

        # multivariate (ML) incidents: sustained, with no single-sensor explanation
        for machine, score in ml_scores.items():
            mi0 = (ml_info or {}).get(machine, {})
            point_score = mi0.get("if", score)   # evidence for single-sensor incidents: the point-wise model
            for inc in self.open.values():
                if inc.machine == machine and inc.type != "multivariate":
                    inc.ml_peak = max(inc.ml_peak, point_score)
            # "at least ml_multivariate_s of the last ml_multivariate_window seconds" — tolerant of
            # patterns that only show while the truck works (e.g. overload shows during lifts only)
            hist = self._ml_hist.setdefault(machine, deque(maxlen=cfg.ml_multivariate_window))
            hist.append(score >= cfg.ml_multivariate_score)
            self.ml_run[machine] = sum(hist)
            # which model is speaking, and which sensors does it point at?
            mi = mi0
            point = max(mi.get("if", 0.0), mi.get("gauss", 0.0))
            zs_now = mi.get("zs") or {}
            joint = [s_ for s_, v in zs_now.items() if abs(v) >= 0.8]
            # a genuine joint level shift (several sensors off together) is the Gaussian's call even
            # if the autoencoder is also elevated — the autoencoder is for *shape* anomalies
            prefer_point = mi.get("gauss", 0.0) >= cfg.ml_multivariate_score and len(joint) >= 2
            dirs = {}
            if not prefer_point and mi.get("ae", 0.0) >= point and mi.get("ae_share"):
                ranked = sorted(mi["ae_share"].items(), key=lambda p: -p[1])      # reconstruction error
                src = "autoencoder"
                top = [s_ for s_, sh in ranked[:2] if sh >= 0.25] or [ranked[0][0]]
            else:
                zs = mi.get("zs") or {s_: (d.z_hist[-1] if d.z_hist and d.z_hist[-1] == d.z_hist[-1] else 0.0)
                                      for s_, d in detectors[machine].items()}
                ranked = sorted(((s_, abs(v)) for s_, v in zs.items()), key=lambda p: -p[1])   # smoothed deviation
                src = "gaussian" if (prefer_point or mi.get("gauss", 0.0) >= mi.get("if", 0.0)) else "isolation_forest"
                dirs = {s_: (1 if v > 0 else -1) for s_, v in zs.items() if abs(v) >= 0.8}
                top = [s_ for s_, v in ranked[:3] if v >= 0.8] or [ranked[0][0]]
            # the pattern is "explained" when a statistical detector recently fired on one of *those*
            # sensors (the autoencoder's 30 s window still remembers a spike, for instance)
            recent = t - cfg.ae_window - 5
            covered = set(top)
            explained = any(i.machine == machine and i.type != "multivariate" and i.sensor in covered
                            and i.last_active_t >= recent
                            for i in list(self.open.values()) + list(self.recent_closed.values()) + list(self._recent_spikes))
            key = (machine, None, "multivariate")
            if key in self.open and explained:
                pass   # a statistical detector now explains it: let the pattern incident close, don't double-report
            elif self.ml_run[machine] >= cfg.ml_multivariate_s and (not explained or key in self.open):
                inc = self._open(t, machine, None, "multivariate")
                inc.last_active_t = t
                inc.ml_peak = max(inc.ml_peak, score)
                models = sorted(set(inc.info.get("models", [])) | {src})
                # average smoothed deviation over the whole incident -> stable directions for the root cause
                zsum = inc.info.get("_zsum", {})
                for s_, v in zs_now.items():
                    zsum[s_] = zsum.get(s_, 0.0) + v
                zn = inc.info.get("_zn", 0) + 1
                avg_dirs = {s_: (1 if v > 0 else -1) for s_, v in zsum.items() if abs(v / zn) >= 0.5}
                if src == "autoencoder" and inc.info.get("model") in ("gaussian", "isolation_forest") \
                        and len(avg_dirs) >= 2:
                    src = inc.info["model"]      # a joint shift already identified: keep that reading
                inc.info = {"top_sensor_keys": top, "top_sensors": [SENSOR_BY_KEY[s_].label.lower() for s_ in top],
                            "ml_score": round(score, 2), "model": src, "models": models,
                            "if_score": round(mi.get("if", 0.0), 2), "gauss_score": round(mi.get("gauss", 0.0), 2),
                            "ae_score": round(mi.get("ae", 0.0), 2),
                            "_dirs": avg_dirs if src != "autoencoder" else (dirs or {}), "_zsum": zsum, "_zn": zn}
                if top:
                    inc.sensor_hint = top[0]
                touched.add(key)

        # close quiet incidents
        for key, inc in list(self.open.items()):
            quiet_limit = 150 if inc.type == "spike_burst" else rcfg.close_after_s
            if t - inc.last_active_t > quiet_limit:
                inc.status, inc.closed_t = "RESOLVED", t
                del self.open[key]
                self.recent_closed[key] = inc
                if inc.notified_tier:
                    self._feed(t, ts, inc, "resolved", f"Resolved: {self._title(inc)}", ts_fn)
                    self.resolved_notified.append(inc)

        # score + route
        new_notes = []
        open_list = list(self.open.values())
        for inc in open_list:
            if inc.type in PROCESS_TYPES:
                spread_n = len({i.sensor for i in open_list if i.machine == inc.machine and i is not inc
                                and i.type in PROCESS_TYPES})
            else:
                spread_n = len({i.sensor for i in open_list if i.machine == inc.machine and i is not inc
                                and i.type == inc.type})
            inc.severity, inc.components, inc.explanation = score_incident(
                inc, cfg.severity, spread_n, inc.ml_peak, self.feedback.offset(inc))
            self._apply_diagnosis(inc, (ml_info or {}).get(inc.machine, {}).get("diag"))
            new_tier = tier_for(inc.severity, cfg.severity)
            if new_tier != inc.tier:
                inc.tier_history.append({"t": t, "tier": new_tier, "severity": inc.severity})
            inc.tier = new_tier
            inc.max_severity = max(inc.max_severity, inc.severity)
            if TIER_RANK[new_tier] > TIER_RANK[inc.max_tier]:
                inc.max_tier = new_tier
            if inc.notified_tier and (t - inc.opened_t) % 15 == 0:
                # evidence keeps arriving (a second sensor, a model changing its mind): keep the hint current
                inc.root_cause = hint_for(inc, detectors.get(inc.machine, {}), open_list)
            note = self._route(t, ts, inc, open_list, detectors, ts_fn)
            if note:
                new_notes.append(note)
            if inc.diagnosis:
                self._diag_evidence(inc)
        return new_notes

    # ------------------------------------------------------------------
    def _apply_diagnosis(self, inc: Incident, diag: dict | None) -> None:
        """ARK Predict v2. The detectors decided *that* something is wrong; the supervised classifier says
        *what* it is. A confirmed diagnosis is attached to every open incident on the truck (and kept at its
        most confident reading); a confirmed, dangerous known fault lifts the incident to URGENT."""
        cfg = self.cfg
        if diag and diag.get("confirmed") and diag.get("fault") not in (None, "normal"):
            # every confirmed second is a vote; the incident's diagnosis is the fault with most votes
            votes = self._votes.setdefault(inc.id, {})
            f = diag["fault"]
            n, csum = votes.get(f, (0, 0.0))
            votes[f] = (n + 1, csum + float(diag["confidence"]))
            best = max(votes, key=lambda k: votes[k][0])
            bn, bc = votes[best]
            inc.diagnosis = {"fault": best, "label": FAULT_LABEL.get(best, best), "confidence": round(bc / bn, 2),
                             "seconds": bn}
        d = inc.diagnosis
        if d and cfg.clf_escalate and d["fault"] in DANGEROUS_FAULTS and d["confidence"] >= 0.8 \
                and inc.type in PROCESS_TYPES and inc.severity < cfg.severity.urgent_at:
            inc.components = {**inc.components, "known_fault": cfg.severity.urgent_at - inc.severity}
            inc.severity = cfg.severity.urgent_at
            inc.explanation += (f" · ARK v2: the fault classifier recognises {d['label']} "
                                f"({d['confidence']:.0%}) → URGENT")

    @staticmethod
    def _diag_evidence(inc: Incident) -> None:
        d = inc.diagnosis
        if not inc.root_cause or not d:
            return
        line = f"Fault classifier (supervised ML): {d['label']}, {d['confidence']:.0%} confident"
        ev = [e for e in inc.root_cause.get("evidence", []) if not e.startswith("Fault classifier")]
        inc.root_cause = {**inc.root_cause, "evidence": [line] + ev, "mlDiagnosis": d}

    # ------------------------------------------------------------------
    def _handle_spike(self, t, machine, sensor, sg, ts_fn):
        rcfg = self.cfg.routing
        dq = self.spike_times[(machine, sensor)]
        dq.append(sg.t)
        while dq and sg.t - dq[0] > rcfg.spike_burst_window_s:
            dq.popleft()
        if len(dq) >= rcfg.spike_burst_count or (machine, sensor, "spike_burst") in self.open:
            inc = self._open(t, machine, sensor, "spike_burst")
            if inc.spike_count == 0:
                inc.opened_t = min(inc.opened_t, dq[0])
            inc.spike_count = len(dq)
            inc.last_active_t = t
            inc.peak_z_abs = max(inc.peak_z_abs, abs(sg.z))
            inc.info = {"spikes_in_window": len(dq), "window_s": rcfg.spike_burst_window_s}
            return
        # lone spike: a closed, logged-only incident
        inc = Incident(self._next_inc, machine, sensor, "spike", sg.t, sg.t, status="RESOLVED", closed_t=t,
                       direction=sg.direction, peak_z=sg.z, peak_z_abs=abs(sg.z), detected_t=t)
        self._next_inc += 1
        self.incidents[inc.id] = inc
        self._recent_spikes.append(inc)
        self.stats["incidents"] += 1
        inc.severity, inc.components, inc.explanation = score_incident(inc, self.cfg.severity, 0, 0.0,
                                                                       self.feedback.offset(inc))
        inc.tier = inc.max_tier = tier_for(inc.severity, self.cfg.severity)
        inc.max_severity = inc.severity
        inc.root_cause = hint_for(inc, {}, [])
        self.stats["suppressed_ignore_tier"] += 1
        self._feed(t, ts_fn(t), inc, "logged", f"Logged: {self._title(inc)}", ts_fn)

    def _route(self, t, ts, inc: Incident, open_list, detectors, ts_fn):
        rcfg = self.cfg.routing
        tier = inc.tier
        if tier == "IGNORE":
            return None
        if inc.notified_tier and TIER_RANK[tier] <= TIER_RANK[inc.notified_tier]:
            return None  # already told someone at this level
        if tier == "MONITOR" and (t - inc.opened_t + 1) < rcfg.min_confirm_s:
            return None
        if inc.type == "multivariate" and (t - inc.opened_t + 1) < rcfg.pattern_confirm_s:
            return None   # let all three models weigh in before naming a cause

        inc.root_cause = hint_for(inc, detectors.get(inc.machine, {}), open_list)

        # machine grouping: is there already a notified incident on this truck?
        group_peers = [i for i in open_list if i is not inc and i.machine == inc.machine and i.notified_tier
                       and (i.tier == "URGENT" or t - i.opened_t <= rcfg.group_window_s)]
        if group_peers and not inc.notified_tier:
            lead = max(group_peers, key=lambda i: (TIER_RANK[i.notified_tier], -i.opened_t))
            inc.group_id = lead.group_id or lead.id
            if TIER_RANK[tier] <= TIER_RANK[lead.notified_tier]:
                inc.notified_tier = tier
                self.stats["suppressed_grouped"] += 1
                self._feed(t, ts, inc, "grouped",
                           f"Grouped into alert #{inc.group_id}: {self._title(inc)}", ts_fn)
                # refresh the lead's root cause with the new evidence
                lead.root_cause = hint_for(lead, detectors.get(lead.machine, {}), open_list)
                return None

        kind = "escalation" if inc.notified_tier else "new"
        inc.notified_tier = tier
        if inc.group_id is None:
            inc.group_id = inc.id
        note = {
            "id": self._next_note, "t": t, "ts": ts, "kind": kind, "tier": tier, "route": ROUTES[tier],
            "incidentId": inc.id, "groupId": inc.group_id, "machine": inc.machine,
            "assetNumber": ASSET.get(inc.machine), "sensor": inc.sensor,
            "sensorLabel": SENSOR_BY_KEY[inc.sensor].label if inc.sensor in SENSOR_BY_KEY else
                           (SENSOR_BY_KEY[inc.sensor_hint].label + " (pattern)" if inc.sensor_hint in SENSOR_BY_KEY
                            else "Multiple sensors"),
            "type": inc.type, "severity": inc.severity, "title": self._title(inc),
            "explanation": inc.explanation, "components": inc.components, "rootCause": inc.root_cause,
        }
        self._next_note += 1
        self.notifications.append(note)
        self.stats[f"notified_{tier.lower()}"] += 1
        self.feed.appendleft({**note, "feedKind": "alert"})
        return note

    def _title(self, inc: Incident) -> str:
        label = (SENSOR_BY_KEY[inc.sensor].label if inc.sensor in SENSOR_BY_KEY
                 else SENSOR_BY_KEY[inc.sensor_hint].label if inc.sensor_hint in SENSOR_BY_KEY else "Multi-sensor pattern")
        arrow = {1: " rising", -1: " falling"}.get(inc.direction, "") if inc.type == "drift" else ""
        typ = {"spike_burst": "repeated spikes", "multivariate": "abnormal pattern (ML)"}.get(inc.type, inc.type)
        return f"{inc.machine} · {label}{arrow} — {typ}"

    def _feed(self, t, ts, inc, kind, title, ts_fn):
        self.feed.appendleft({"feedKind": kind, "t": t, "ts": ts, "tier": inc.tier, "severity": inc.severity,
                              "incidentId": inc.id, "groupId": inc.group_id, "machine": inc.machine,
                              "sensor": inc.sensor, "type": inc.type, "title": title,
                              "explanation": inc.explanation, "rootCause": inc.root_cause})
