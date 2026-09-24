"""
ARK SHIELD server (Python) — application bootstrap.

Run with:
    uvicorn app.main:app --reload --port 4000

`app` here is the combined ASGI application (FastAPI + Socket.IO) —
Socket.IO is mounted at /socket.io/ alongside the REST API, the same
way src/index.ts hosts both on one Express + Socket.IO server.

The root path ("/") serves the interactive demo UI (app/static/index.html)
so this one project is both the API and the thing you open in a browser.
That demo page runs its own self-contained, client-side simulation (it
was originally built to need no backend at all) — it does not call this
server's REST API or Socket.IO. Wiring the demo's UI to real data from
this backend instead of its built-in simulation is exactly the kind of
frontend work described in the README's "Building a real frontend
against this API" section.
"""
from contextlib import asynccontextmanager
from pathlib import Path

import socketio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from app.config import settings
from app.database import AsyncSessionLocal, engine as db_engine
from app.errors import (
    AppError,
    app_error_handler,
    integrity_error_handler,
    not_found_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.realtime.socket import sio
from app.routers import (
    alerts,
    analytics,
    auth,
    cameras,
    forklifts,
    hierarchy,
    operators,
    safety_events,
    simulation_control,
    telemetry,
    uwb,
    vision,
    workers,
    zones,
)
from app.services.simulation import simulation_engine
from app.routers import anomaly
from app.services.anomaly import anomaly_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup
    if settings.auto_start_simulation:
        try:
            await simulation_engine.start(sio)
        except Exception as err:  # noqa: BLE001 — e.g. Postgres not running yet
            print(f"[simulation] not started ({err!r}) — is the database up? ARK Predict still runs without it.")
    # ARK Predict (HTH-ML-10): in-memory streaming anomaly detection, no database needed.
    if settings.anomaly_auto_start:
        if settings.anomaly_seed != anomaly_service.seed:
            await anomaly_service.reset(settings.anomaly_seed)
        anomaly_service.set_speed(settings.anomaly_speed)
        anomaly_service.set_rate(settings.anomaly_fault_rate_per_min)
        await anomaly_service.start(sio)
    yield
    # Shutdown — port of src/index.ts's SIGINT/SIGTERM graceful shutdown.
    await anomaly_service.pause()
    await simulation_engine.pause()
    await db_engine.dispose()


fastapi_app = FastAPI(title="ARK Shield API", version="1.0.0", lifespan=lifespan)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.web_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Centralized error handling (port of src/middleware/errorHandler.ts)
# ------------------------------------------------------------------
fastapi_app.add_exception_handler(AppError, app_error_handler)
fastapi_app.add_exception_handler(RequestValidationError, validation_error_handler)
fastapi_app.add_exception_handler(NoResultFound, not_found_handler)
fastapi_app.add_exception_handler(IntegrityError, integrity_error_handler)
fastapi_app.add_exception_handler(Exception, unhandled_error_handler)


# ------------------------------------------------------------------
# Demo UI — serves app/static/index.html at the root path. A specific
# GET route (not a StaticFiles Mount) is used so it can never shadow
# /api/*, /health, /docs, or /socket.io/, regardless of route order.
# ------------------------------------------------------------------
DEMO_HTML_PATH = Path(__file__).parent / "static" / "index.html"


@fastapi_app.get("/", include_in_schema=False)
async def serve_demo():
    return FileResponse(DEMO_HTML_PATH)


# ------------------------------------------------------------------
# Health checks
# ------------------------------------------------------------------
@fastapi_app.get("/health")
async def health():
    """Liveness probe — does not touch the database, so it stays useful
    even if Postgres is down."""
    return {"status": "ok", "service": "ark-shield-server-python", "simulatedDataOnly": True,
            "arkPredict": {"running": anomaly_service.running, "ready": anomaly_service.ready}}


@fastapi_app.get("/health/db")
async def health_db():
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as err:  # noqa: BLE001
        print(f"DB health check failed: {err!r}")
        return {"status": "error", "database": "unreachable"}


# ------------------------------------------------------------------
# Routes — everything under /api requires a valid JWT except
# /api/auth/login, enforced inside each router via get_current_user.
# ------------------------------------------------------------------
fastapi_app.include_router(auth.router)
fastapi_app.include_router(forklifts.router)
fastapi_app.include_router(operators.router)
fastapi_app.include_router(zones.router)
fastapi_app.include_router(safety_events.router)
fastapi_app.include_router(alerts.router)
fastapi_app.include_router(telemetry.router)
fastapi_app.include_router(vision.router)
fastapi_app.include_router(uwb.router)
fastapi_app.include_router(workers.router)
fastapi_app.include_router(analytics.router)
fastapi_app.include_router(cameras.router)
fastapi_app.include_router(hierarchy.router)  # /api/organizations/me, /api/sites, /api/warehouses
fastapi_app.include_router(simulation_control.router)
fastapi_app.include_router(anomaly.router)  # /api/anomaly — ARK Predict machine-health anomaly detection

# ------------------------------------------------------------------
# Combine FastAPI (REST) with python-socketio (real-time) into one
# ASGI app. This — not fastapi_app — is what uvicorn should serve.
# ------------------------------------------------------------------
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app, socketio_path="socket.io")
