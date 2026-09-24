"""
Tests for ARK Predict (HTH-ML-10 anomaly pipeline).

    python -m pytest tests/test_anomaly.py -q      # with pytest
    python tests/test_anomaly.py                   # without pytest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.anomaly.ml import HAVE_SKLEARN, MachineModel  # noqa: E402
from app.services.anomaly.service import AnomalyService, report_markdown  # noqa: E402


def _clean_service(seed=3):
    svc = AnomalyService(seed=seed, anomaly_rate_per_min=0)
    svc.ensure_ready()
    svc.step(60)
    return svc


def _run_fault(scenario, machine="ARK-F002", sensor=None, duration=None, magnitude=None, after=150, seed=3):
    svc = _clean_service(seed)
    a = svc.sim.inject(scenario, machine=machine, sensor=sensor, duration=duration, magnitude=magnitude)
    svc.step((a.end - svc.sim.t) + after)
    incs = [i for i in svc.engine.router.incidents.values() if i.machine == machine]
    return svc, a, incs


def test_fast_isolation_forest_matches_sklearn():
    if not HAVE_SKLEARN:
        return
    rng = np.random.default_rng(0)
    m = MachineModel()
    m.fit(rng.normal(size=(2000, 6)))
    q = rng.normal(size=(40, 6)) * 2
    assert np.allclose(-m._model.score_samples(q), m._fast_iforest(q))


def test_autoencoder_numpy_forward_matches_sklearn():
    from app.services.anomaly.ml import MLPRegressor, SequenceAutoencoder
    if MLPRegressor is None:
        return
    svc = AnomalyService(seed=3, anomaly_rate_per_min=0)
    svc.ensure_ready()
    ae = svc.engine.ae
    X = np.random.default_rng(1).normal(size=(20, ae.window * ae.n_feat)) * 0.5
    assert np.allclose(ae._net.predict(X), ae._forward(X))
    assert ae.train_info["windows"] > 1000 and ae.train_info["input_dim"] == ae.window * ae.n_feat


def _oscillation_found(ml, sensor="mast_vibration", seed=10):
    from app.services.anomaly.config import EngineConfig
    cfg = EngineConfig()
    cfg.use_isolation_forest = ml in ("both", "if")
    cfg.use_gaussian = ml in ("both", "gauss")
    cfg.use_autoencoder = ml in ("both", "ae")
    svc = AnomalyService(seed=seed, anomaly_rate_per_min=0, cfg=cfg)
    svc.ensure_ready()
    svc.step(60)
    svc.sim.inject("oscillation", machine="ARK-F003", sensor=sensor, duration=240)
    svc.step(300)
    return [i for i in svc.engine.router.incidents.values() if i.machine == "ARK-F003" and i.type == "multivariate"]


def test_autoencoder_catches_oscillation_that_isolation_forest_misses():
    for sensor in ("mast_vibration", "hydraulic_pressure"):
        assert _oscillation_found("if", sensor) == [], "IF alone should not see an in-limits oscillation"
        found = _oscillation_found("both", sensor)
        assert found, f"autoencoder missed the oscillation on {sensor}"
        inc = found[0]
        assert inc.sensor_hint == sensor, "autoencoder should attribute the pattern to the right sensor"
        assert inc.max_tier == "MONITOR"
        assert "autoencoder" in " ".join(inc.root_cause.get("evidence", []))


def _run_cfg(ml, scenario, seed, machine="ARK-F002", **kw):
    from app.services.anomaly.config import EngineConfig
    cfg = EngineConfig()
    cfg.use_isolation_forest = ml in ("all", "if")
    cfg.use_gaussian = ml in ("all", "gauss")
    cfg.use_autoencoder = ml in ("all", "ae")
    svc = AnomalyService(seed=seed, anomaly_rate_per_min=0, cfg=cfg)
    svc.ensure_ready()
    svc.step(60)
    svc.sim.inject(scenario, machine=machine, **kw)
    svc.step(300)
    return [i for i in svc.engine.router.incidents.values() if i.machine == machine]


def test_gaussian_catches_overload_that_statistics_miss():
    for seed in (32, 35):
        assert _run_cfg("none", "overload", seed, duration=240, magnitude=2.0) == [], \
            "each overload shift is under its own sensor's limit — statistics alone should stay quiet"
        found = [i for i in _run_cfg("all", "overload", seed, duration=240, magnitude=2.0) if i.type == "multivariate"]
        assert found, "the joint-shift Gaussian should flag the overload"
        assert "overload" in found[0].root_cause.get("title", "").lower()
        assert found[0].max_tier == "MONITOR"


def test_feedback_loop_learns_and_is_bounded():
    from app.services.anomaly.feedback import MAX_OFFSET
    svc = _clean_service(seed=4)
    svc.sim.inject("stuck", machine="ARK-F001", sensor="battery_voltage", duration=70)
    svc.step(130)
    first = [i for i in svc.engine.router.incidents.values() if i.type == "stuck"][0]
    assert first.notified_tier == "MONITOR"
    for _ in range(6):
        svc.give_feedback(first.id, "false_alarm")
    learned = svc.feedback_table()["learned"][0]
    assert learned["type"] == "stuck" and learned["sensor"] == "battery_voltage"
    assert -MAX_OFFSET <= learned["offset"] < -10
    svc.step(360)   # past the re-open window, so the next one is a new incident
    svc.sim.inject("stuck", machine="ARK-F001", sensor="battery_voltage", duration=70)
    svc.step(130)
    second = [i for i in svc.engine.router.incidents.values() if i.type == "stuck" and i.id != first.id][0]
    assert second.max_severity < first.max_severity - 10, "same fault should now score lower"
    assert second.notified_tier is None, "after repeated 'false alarm' verdicts nobody is paged for it"
    assert "technician feedback" in second.explanation
    # other kinds of incident are unaffected
    svc.sim.inject("stuck", machine="ARK-F001", sensor="hydraulic_pressure", duration=70)
    svc.step(130)
    other = [i for i in svc.engine.router.incidents.values() if i.sensor == "hydraulic_pressure"][0]
    assert other.notified_tier == "MONITOR"


def test_replay_skab_format_csv(tmp_path=None):
    """The real-data path: a SKAB-format CSV (';', sensor columns, 'anomaly' label) streams through
    the same detectors and is scored against its labels."""
    import tempfile
    from app.services.anomaly.replay import make_demo, run_path
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    path = os.path.join(d, "demo_skab.csv")
    make_demo(path)
    with_ml = run_path(path)["overall"]
    stats_only = run_path(path, use_ml=False)["overall"]
    assert with_ml["ours"]["eventRecall"] >= 0.75
    assert with_ml["ours"]["f1"] > with_ml["baseline"]["f1"]
    assert with_ml["ours"]["f1"] >= stats_only["ours"]["f1"]
    assert with_ml["ours"]["falseAlarmsPerHour"] <= 5


def test_no_false_alarms_on_healthy_stream():
    svc = AnomalyService(seed=1, anomaly_rate_per_min=0)
    svc.ensure_ready()
    svc.step(3600)
    assert len(svc.engine.router.notifications) == 0
    assert len(svc.engine.router.incidents) == 0
    # while the naive fixed-threshold baseline fires constantly on normal load changes
    assert len(svc.engine.baseline_alerts) > 50


def test_each_anomaly_type_is_detected_and_typed():
    cases = [
        ("spike", "hydraulic_pressure", None, 12, "spike", "IGNORE"),
        ("drift", "motor_temp", 300, 18, "drift", "URGENT"),
        ("drift", "mast_vibration", 300, 8, "drift", "MONITOR"),
        ("dropout", "motor_current", 45, None, "dropout", "MONITOR"),
        ("dropout", "hydraulic_pressure", 150, None, "dropout", "URGENT"),
        ("stuck", "battery_voltage", 60, None, "stuck", "MONITOR"),
        ("stuck", "hydraulic_pressure", 200, None, "stuck", "URGENT"),
    ]
    for scen, sensor, dur, mag, want_type, want_tier in cases:
        _, a, incs = _run_fault(scen, sensor=sensor, duration=dur, magnitude=mag)
        hits = [i for i in incs if i.sensor == sensor and i.type == want_type]
        assert hits, f"{scen} on {sensor} not detected as {want_type}: {[(i.type, i.sensor) for i in incs]}"
        assert hits[0].max_tier == want_tier, f"{scen}/{sensor}: tier {hits[0].max_tier} != {want_tier}"


def test_spike_burst_aggregates_and_notifies_once():
    svc, a, incs = _run_fault("spike_burst", sensor="mast_vibration", after=60)
    bursts = [i for i in incs if i.type == "spike_burst"]
    assert len(bursts) == 1 and bursts[0].max_tier == "MONITOR"
    notes = [n for n in svc.engine.router.notifications if n["incidentId"] == bursts[0].id]
    assert len(notes) == 1


def test_compound_fault_bearing_wear_is_urgent_grouped_and_explained():
    svc, a, incs = _run_fault("bearing_wear", duration=360, magnitude=20)
    sensors = {i.sensor for i in incs if i.type == "drift"}
    assert {"mast_vibration", "motor_temp"} <= sensors
    assert max(i.max_tier for i in incs if i.type == "drift") == "URGENT"
    notes = [n for n in svc.engine.router.notifications if n["machine"] == "ARK-F002"]
    # one truck, one problem: at most one "new" alert (+ escalations), the second sensor is grouped
    assert sum(1 for n in notes if n["kind"] == "new") == 1
    lead = [i for i in incs if i.type == "drift" and i.notified_tier == "URGENT"]
    assert any("bearing" in i.root_cause.get("title", "").lower() for i in lead)


def test_hydraulic_leak_root_cause():
    svc, a, incs = _run_fault("hydraulic_leak", duration=300, magnitude=18)
    titles = " ".join(i.root_cause.get("title", "") for i in incs if i.root_cause)
    assert "hydraulic leak" in titles.lower()


def test_dropout_root_cause_is_sensor_fault():
    _, a, incs = _run_fault("dropout", sensor="motor_temp", duration=60)
    d = [i for i in incs if i.type == "dropout"][0]
    assert "sensor" in d.root_cause["title"].lower()


def test_views_are_json_and_report_renders():
    svc = AnomalyService(seed=5, anomaly_rate_per_min=2)
    svc.ensure_ready()
    svc.step(1800)
    for view in (svc.status(), svc.config(), svc.machines(), svc.series("ARK-F001", 120),
                 svc.alerts(), svc.feed(), svc.incidents("open"), svc.incidents("all")):
        json.dumps(view, default=str)
    rep = svc.summary()
    json.dumps(rep, default=str)
    md = report_markdown(rep)
    assert "Severity calibration" in md and "Alert-fatigue" in md


def test_offline_evaluation_quality():
    svc = AnomalyService(seed=11, anomaly_rate_per_min=1.2)
    svc.ensure_ready()
    svc.step(2 * 3600)
    ev = svc.summary()["evaluation"]
    assert ev["detectionRate"] >= 0.95
    assert ev["typeAccuracy"] >= 0.9
    assert ev["severityCalibration"]["urgentRecall"] >= 0.95
    assert ev["severityCalibration"]["exactTierAccuracy"] >= 0.85
    assert ev["alertFatigue"]["reductionVsBaseline"] >= 0.6


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
