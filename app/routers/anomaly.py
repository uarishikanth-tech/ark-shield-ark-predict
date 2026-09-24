"""
ARK Predict — machine-health anomaly detection API (HTH-ML-10).

Live, severity-aware streaming anomaly detection over forklift
machine-health sensors (motor temperature, hydraulic pressure, mast
vibration, battery voltage, motor current). Backed by the in-memory
AnomalyService — no database needed, so it works before Postgres is set up.

Access: with ANOMALY_PUBLIC_DEMO=true (the default, for the hackathon
demo UI which has no real JWT) every route is open. Set it to false and
reads require any logged-in user, controls require SAFETY_OR_FLEET.
"""
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.camel import CamelModel
from app.config import settings
from app.errors import AppError
from app.services.anomaly import anomaly_service, report_markdown
from app.services.anomaly.config import SENSOR_BY_KEY
from app.services.anomaly.simulator import SCENARIOS

router = APIRouter(prefix="/api/anomaly", tags=["anomaly (ARK Predict)"])
_bearer = HTTPBearer(auto_error=False)


def _viewer(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    if settings.anomaly_public_demo:
        return None
    from app.deps import get_current_user

    return get_current_user(credentials)


def _operator(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    if settings.anomaly_public_demo:
        return None
    from app.deps import get_current_user
    from app.types import ROLE_GROUPS

    user = get_current_user(credentials)
    if user.role not in ROLE_GROUPS["SAFETY_OR_FLEET"]:
        raise AppError.forbidden(f"Role {user.role} cannot control the anomaly stream")
    return user


# ------------------------------------------------------------------
# Reads
# ------------------------------------------------------------------
@router.get("/status")
async def status(_u=Depends(_viewer)):
    return anomaly_service.status()


@router.get("/config")
async def config(_u=Depends(_viewer)):
    return anomaly_service.config()


@router.get("/machines")
async def machines(_u=Depends(_viewer)):
    return anomaly_service.machines()


@router.get("/series")
async def series(machine_id: str = Query(..., alias="machineId"), points: int = 240, _u=Depends(_viewer)):
    try:
        return anomaly_service.series(machine_id, points)
    except KeyError:
        raise AppError.not_found("Machine")


@router.get("/alerts")
async def alerts(limit: int = Query(50, ge=1, le=500), _u=Depends(_viewer)):
    """Routed notifications only (what a human was actually told), newest first."""
    return anomaly_service.alerts(limit)


@router.get("/feed")
async def feed(limit: int = Query(80, ge=1, le=400), _u=Depends(_viewer)):
    """Everything the router decided: alerts, grouped, logged-only and resolved events."""
    return anomaly_service.feed(limit)


@router.get("/incidents")
async def incidents(status: str = Query("open", pattern="^(open|all)$"),
                    limit: int = Query(100, ge=1, le=1000), _u=Depends(_viewer)):
    return anomaly_service.incidents(status, limit)


@router.get("/incidents/{incident_id}")
async def incident(incident_id: int, _u=Depends(_viewer)):
    inc = anomaly_service.engine.router.incidents.get(incident_id)
    if not inc:
        raise AppError.not_found("Incident")
    return inc.as_dict(anomaly_service.sim.timestamp)


@router.get("/summary")
async def summary(include_faults: bool = Query(False, alias="includeFaults"), _u=Depends(_viewer)):
    """End-of-run summary: counts, top urgent incidents and evaluation against injected ground truth."""
    return anomaly_service.summary(include_faults)


@router.get("/report.md", response_class=PlainTextResponse)
async def report(_u=Depends(_viewer)):
    md = report_markdown(anomaly_service.summary())
    return PlainTextResponse(md, headers={"Content-Disposition": 'attachment; filename="ark-predict-report.md"'})


# ------------------------------------------------------------------
# ML Lab — results on the real SKAB dataset (precomputed by
# `python -m app.services.anomaly.skab SKAB/data --export`)
# ------------------------------------------------------------------
_RESULTS = Path(__file__).resolve().parent.parent / "services" / "anomaly" / "results"
_BENCH_FILES = {"skab": "skab_results.json", "forklift": "forklift_results.json"}
_bench_cache: dict = {}


def _bench(dataset: str = "skab") -> dict:
    if dataset not in _BENCH_FILES:
        raise AppError(f"dataset must be one of {', '.join(_BENCH_FILES)}", 400)
    if dataset not in _bench_cache:
        path = _RESULTS / _BENCH_FILES[dataset]
        if not path.exists():
            raise AppError(f"No {dataset} benchmark results yet — run: python scripts/review3.py", 404)
        _bench_cache[dataset] = json.loads(path.read_text())
    return _bench_cache[dataset]


@router.get("/benchmark")
async def benchmark(dataset: str = Query("skab", description="skab (real pump data) | forklift (labelled forklift runs)"),
                    _u=Depends(_viewer)):
    """Model comparison: every model vs the fixed-limit baseline (+ the published leaderboard for SKAB)."""
    return {k: v for k, v in _bench(dataset).items() if k != "replay"}


@router.get("/benchmark/replay")
async def benchmark_replay(experiment: str = Query(..., description="e.g. valve1/3"), dataset: str = Query("skab"),
                           _u=Depends(_viewer)):
    """One experiment: sensor traces, the labelled fault, and when each system raised an alarm."""
    rep = _bench(dataset).get("replay", {})
    if experiment not in rep:
        raise AppError.not_found(f"Experiment {experiment}")
    return {"experiment": experiment, "dataset": dataset, **rep[experiment]}


# ------------------------------------------------------------------
# ARK Predict v2 — the supervised fault classifier
# ------------------------------------------------------------------
@router.get("/model")
async def model_status(_u=Depends(_viewer)):
    """The fault classifier: version, training data, accuracy on unseen runs, technician labels waiting."""
    st = anomaly_service.model_status()
    ev = _RESULTS / "fault_classifier_eval.json"
    if ev.exists() and "evaluation" not in st:
        st["evaluation"] = json.loads(ev.read_text())
    return st


@router.post("/model/retrain")
async def model_retrain(_u=Depends(_operator)):
    """Retrain the fault classifier on its simulator data + every technician-labelled window (about 30–60 s)."""
    import asyncio

    try:
        return await asyncio.to_thread(anomaly_service.retrain_classifier)
    except RuntimeError as e:
        raise AppError(str(e), 409)


# ------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------
class ResetBody(CamelModel):
    seed: Optional[int] = None


class SpeedBody(CamelModel):
    speed: int


class RateBody(CamelModel):
    rate_per_min: float


class FeedbackBody(CamelModel):
    incident_id: int
    label: str   # right_call | too_high | too_low | false_alarm
    fault_type: Optional[str] = None   # what it actually was (teaches the v2 fault classifier)


class InjectBody(CamelModel):
    scenario: str
    machine_id: Optional[str] = None
    sensor: Optional[str] = None
    duration_s: Optional[int] = None
    magnitude: Optional[float] = None


@router.post("/start")
async def start(_u=Depends(_operator)):
    from app.realtime.socket import sio

    await anomaly_service.start(sio)
    return anomaly_service.status()


@router.post("/pause")
async def pause(_u=Depends(_operator)):
    await anomaly_service.pause()
    return anomaly_service.status()


@router.post("/reset")
async def reset(body: ResetBody | None = None, _u=Depends(_operator)):
    await anomaly_service.reset(body.seed if body else None)
    return anomaly_service.status()


@router.post("/speed")
async def speed(body: SpeedBody, _u=Depends(_operator)):
    anomaly_service.set_speed(body.speed)
    return anomaly_service.status()


@router.post("/fault-rate")
async def fault_rate(body: RateBody, _u=Depends(_operator)):
    anomaly_service.set_rate(body.rate_per_min)
    return anomaly_service.status()


@router.post("/feedback")
async def feedback(body: FeedbackBody, _u=Depends(_operator)):
    """Technician verdict on an incident. The system learns a bounded severity correction per
    (anomaly type, sensor) and applies it to future incidents of that kind."""
    from app.services.anomaly.feedback import LABELS

    if body.label not in LABELS:
        raise AppError(f"label must be one of {', '.join(LABELS)}", 400)
    try:
        return anomaly_service.give_feedback(body.incident_id, body.label, body.fault_type)
    except KeyError:
        raise AppError.not_found("Incident")


@router.get("/feedback")
async def feedback_table(_u=Depends(_viewer)):
    """What the system has learned from technicians so far."""
    return anomaly_service.feedback_table()


@router.post("/inject")
async def inject(body: InjectBody, _u=Depends(_operator)):
    """DEMO — schedule a labelled fault (spike, spike_burst, drift, dropout, stuck, bearing_wear,
    hydraulic_leak, battery_fade, oscillation, overload) on a simulated forklift sensor, starting next second."""
    if body.scenario not in SCENARIOS:
        raise AppError(f"scenario must be one of {', '.join(SCENARIOS)}", 400)
    if body.sensor and body.sensor not in SENSOR_BY_KEY:
        raise AppError(f"sensor must be one of {', '.join(SENSOR_BY_KEY)}", 400)
    try:
        injected = anomaly_service.inject(body.scenario, body.machine_id, body.sensor, body.duration_s, body.magnitude)
    except ValueError as err:
        raise AppError(str(err), 400)
    return {"ok": True, "injected": injected, "note": "DEMO ONLY — simulated sensor fault"}
