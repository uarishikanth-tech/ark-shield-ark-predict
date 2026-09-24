"""
Technician feedback loop — the system learns from the people who act on its alerts.

When a technician closes an incident they can mark it:

    right_call   the tier was right                 -> desired correction 0
    too_high     should have been one tier lower    -> correction that drops it one tier
    too_low      should have been one tier higher   -> correction that lifts it one tier
    false_alarm  nothing was wrong                   -> correction that drops it to IGNORE

Corrections are learned per (anomaly type, sensor) as an exponential moving
average, so one angry click cannot swing the system and conflicting labels
average out. The learned offset is added to future severity scores for that
kind of incident (shown as its own "feedback" line in the explanation), and
is bounded to ±30 points so feedback can tune the model but never silence it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LABELS = ("right_call", "too_high", "too_low", "false_alarm")
MAX_OFFSET = 30.0


@dataclass
class _Stat:
    offset: float = 0.0
    n: int = 0
    counts: dict = field(default_factory=lambda: {k: 0 for k in LABELS})


class FeedbackLearner:
    def __init__(self, monitor_at: float, urgent_at: float, alpha: float = 0.35) -> None:
        self.monitor_at = monitor_at
        self.urgent_at = urgent_at
        self.alpha = alpha
        self.stats: dict[tuple, _Stat] = {}
        self.log: list[dict] = []

    @staticmethod
    def key(inc) -> tuple:
        return (inc.type, inc.sensor or inc.sensor_hint or "*")

    def offset(self, inc) -> float:
        st = self.stats.get(self.key(inc))
        return st.offset if st else 0.0

    def _desired(self, severity: float, label: str) -> float:
        """Correction (in severity points) that would have produced the right tier."""
        m, u = self.monitor_at, self.urgent_at
        if label == "right_call":
            return 0.0
        if label == "false_alarm":
            return min(0.0, (m - 5) - severity)
        if label == "too_high":
            target = (u - 5) if severity >= u else (m - 5)
            return min(0.0, target - severity)
        if label == "too_low":
            target = (u + 5) if severity >= m else (m + 5)
            return max(0.0, target - severity)
        raise ValueError(f"label must be one of {LABELS}")

    def record(self, inc, label: str, source: str = "technician", t: int | None = None) -> dict:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        k = self.key(inc)
        st = self.stats.setdefault(k, _Stat())
        base = inc.max_severity - st.offset          # what the model said before any learned offset
        want = self._desired(base, label)
        st.offset = max(-MAX_OFFSET, min(MAX_OFFSET, st.offset + self.alpha * (want - st.offset)))
        st.n += 1
        st.counts[label] += 1
        inc.feedback = label
        entry = {"t": t, "incidentId": inc.id, "type": k[0], "sensor": k[1], "label": label,
                 "learnedOffset": round(st.offset, 1), "source": source}
        self.log.append(entry)
        return entry

    def table(self) -> list[dict]:
        rows = []
        for (typ, sen), st in sorted(self.stats.items(), key=lambda p: -abs(p[1].offset)):
            useful = st.counts["right_call"] + st.counts["too_high"] + st.counts["too_low"]
            rows.append({"type": typ, "sensor": sen, "offset": round(st.offset, 1), "labels": st.n,
                         "counts": dict(st.counts),
                         "precision": round(useful / st.n, 2) if st.n else None})
        return rows
