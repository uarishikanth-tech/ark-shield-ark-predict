"""
Tests for ARK Predict v2: the supervised fault classifier, its live wiring and technician-label retraining.

    python tests/test_classifier.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.anomaly.classifier import FAULT_CLASSES, N_FEATURES, FaultClassifier, FeatureTracker  # noqa: E402
from app.services.anomaly.service import AnomalyService  # noqa: E402


class StubClassifier(FaultClassifier):
    """Always 'sees' the same fault — lets us test the live wiring without training a model."""

    def __init__(self, fault: str, p: float = 0.95):
        super().__init__()
        self.fault, self.p = fault, p
        self.model = "stub"

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        out = np.full((len(X), len(FAULT_CLASSES)), (1 - self.p) / (len(FAULT_CLASSES) - 1))
        out[:, FAULT_CLASSES.index(self.fault)] = self.p
        return out


def test_feature_tracker():
    tr = FeatureTracker(5)
    for i in range(70):
        tr.update([0.1 * i, None, 0.0, 1.0, -1.0], [1.0 + i, None, 5.0, 2.0, 3.0], 0.5)
    f = tr.features()
    assert f.shape == (N_FEATURES,)
    assert tr.frozen[2] == 69 and tr.frozen[0] == 0      # sensor 2 repeats the same raw value
    miss_block = f[5 + 30 + 5 + 5: 5 + 30 + 5 + 5 + 5]
    assert miss_block[1] == 1.0 and miss_block[0] == 0.0  # sensor 1 silent the whole time


def _drift_incident(clf):
    svc = AnomalyService(seed=3, anomaly_rate_per_min=0, classifier=clf)
    svc.ensure_ready()
    svc.step(60)
    svc.sim.inject("drift", machine="ARK-F002", sensor="mast_vibration", duration=300, magnitude=8)
    svc.step(360)
    incs = [i for i in svc.engine.router.incidents.values() if i.machine == "ARK-F002" and i.type == "drift"]
    return svc, incs[0]


def test_dangerous_known_fault_escalates_to_urgent():
    _, plain = _drift_incident(None)
    assert plain.max_tier == "MONITOR" and plain.diagnosis is None
    svc, inc = _drift_incident(StubClassifier("bearing_wear"))
    assert inc.diagnosis["fault"] == "bearing_wear" and inc.diagnosis["confidence"] >= 0.9
    assert inc.max_tier == "URGENT", "a confident bearing-wear diagnosis should page someone"
    assert "fault classifier" in inc.explanation
    assert any(e.startswith("Fault classifier") for e in inc.root_cause.get("evidence", []))
    assert svc.machines()[1]["diagnosis"]["fault"] == "bearing_wear"


def test_harmless_diagnosis_only_names_the_fault():
    _, inc = _drift_incident(StubClassifier("drift"))
    assert inc.diagnosis["fault"] == "drift"
    assert inc.max_tier == "MONITOR", "only dangerous known faults change the tier"


def test_no_incidents_from_the_classifier_alone():
    """The classifier names faults; it never opens an incident on its own (so it can't add false alarms)."""
    svc = AnomalyService(seed=1, anomaly_rate_per_min=0, classifier=StubClassifier("hydraulic_leak"))
    svc.ensure_ready()
    svc.step(900)
    assert len(svc.engine.router.notifications) == 0


def test_technician_labels_become_training_data_and_retrain():
    rng = np.random.default_rng(0)
    clf = FaultClassifier()
    clf.base_X = rng.normal(size=(600, N_FEATURES))
    clf.base_y = np.repeat(np.arange(len(FAULT_CLASSES)), 600 // len(FAULT_CLASSES) + 1)[:600]
    clf.fit(clf.base_X, clf.base_y)
    svc = AnomalyService(seed=4, anomaly_rate_per_min=0, classifier=clf)
    svc.ensure_ready()
    svc.step(60)
    svc.sim.inject("stuck", machine="ARK-F001", sensor="battery_voltage", duration=70)
    svc.step(130)
    inc = [i for i in svc.engine.router.incidents.values() if i.type == "stuck"][0]
    r = svc.give_feedback(inc.id, "right_call", "stuck")
    assert r["classifierLabel"]["added"] > 30 and r["classifierLabel"]["faultType"] == "stuck"
    r2 = svc.give_feedback(inc.id, "false_alarm")
    assert r2["classifierLabel"]["faultType"] == "normal"
    st = svc.retrain_classifier()
    assert st["version"] == 2 and st["feedbackWindows"] == 2 and st["feedbackRows"] > 60
    assert svc.model_status()["inUse"]


def test_saved_model_if_present():
    from app.services.anomaly.classifier import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        return
    c = FaultClassifier.load()
    ev = c.info.get("evaluation", {})
    assert ev.get("faultTypeAccuracy", 0) >= 0.8, "saved model should name >= 80% of unseen faults correctly"
    P = c.predict_proba(np.zeros((3, N_FEATURES)))
    assert P.shape == (3, len(FAULT_CLASSES)) and np.allclose(P.sum(axis=1), 1)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
