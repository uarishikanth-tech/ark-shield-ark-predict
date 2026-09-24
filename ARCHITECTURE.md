# ARK Shield — Architecture (Python / FastAPI)

This mirrors the Node version's architecture exactly at the design level;
this document focuses on how each piece is implemented in Python. Read the
Node version's `ARCHITECTURE.md` first for the product-level rationale
(why the risk engine is structured as it is, why simulated data is tagged,
why the hardware-abstraction endpoints look the way they do) — it isn't
repeated in full here.

## 1. System overview

```
                        ┌─────────────────────────────┐
                        │        Frontend(s)          │
                        └───────────┬─────────┬────────┘
                                    │ REST     │ Socket.IO
                                    │ (JWT)    │ (JWT handshake)
                        ┌───────────▼─────────▼────────┐
                        │   FastAPI + python-socketio    │
                        │   combined into one ASGI app   │
                        │   (app/main.py: `app`)         │
                        └───────────┬────────────────────┘
                                    │ SQLAlchemy (async)
                        ┌───────────▼────────────────────┐
                        │          PostgreSQL             │
                        └──────────────────────────────────┘
                                    ▲
                        ┌───────────┴────────────────────┐
                        │  SimulationEngine (asyncio.Task) │
                        │  started from app.main's lifespan,│
                        │  or standalone via                │
                        │  `python -m app.services.simulation`│
                        └──────────────────────────────────┘
```

`app/main.py` builds a plain FastAPI app (`fastapi_app`), then wraps it with
`socketio.ASGIApp(sio, other_asgi_app=fastapi_app)` to produce the single
ASGI callable (`app`) that Uvicorn serves — REST and Socket.IO share one
process and one event loop. The simulation engine runs as a plain
`asyncio.Task` inside that same event loop (started in the FastAPI
`lifespan` context manager), ticking on `asyncio.sleep(...)` rather than a
separate thread or process, so it shares the loop cleanly with request
handling and Socket.IO's own async I/O.

## 2. Data model

Identical shape to the Node/Prisma schema (see its `ARCHITECTURE.md` §2 for
the full hierarchy diagram). Implementation notes specific to this port:

- **snake_case columns, camelCase JSON.** SQLAlchemy models
  (`app/models.py`) use idiomatic Python/Postgres snake_case throughout.
  Every Pydantic request/response schema subclasses `CamelModel`
  (`app/camel.py`), which auto-generates a camelCase alias for each
  snake_case field name and accepts either casing on input. FastAPI's
  default `response_model_by_alias=True` means responses serialize using
  the camelCase aliases automatically — no per-route configuration needed.
- **Tenant isolation** works the same way as the Node version: every query
  that touches warehouse-scoped data joins `Warehouse` → `Site` and filters
  on `Site.organization_id == <the JWT's org>`. This is written out
  per-router as a small `_org_scope()` helper (or inline joins) rather than
  a single shared function, because the join path differs by how many hops
  a table is from `Warehouse` (`SafetyEvent`/`Alert` join through
  `Warehouse` directly; `Telemetry`/`Detection`/`ProximityEvent` join
  through `Forklift` first). `app/utils/tenancy.get_owned_warehouse()` is
  the one shared helper, used specifically to validate a `warehouse_id` in
  a request body before attaching a new child row to it.
- **Native PostgreSQL enums.** Every Prisma enum became a Python
  `class X(str, Enum)` mapped via SQLAlchemy's `Enum` type, which creates a
  native Postgres enum type on migration. Because each enum subclasses
  `str`, enum instances satisfy plain `str`-typed Pydantic fields directly
  with no extra coercion code.
- **Async session per request.** `app/database.get_db()` is a FastAPI
  dependency yielding one `AsyncSession` per request, closed automatically
  when the request finishes. Relationships that a response schema needs are
  eager-loaded explicitly with SQLAlchemy's `selectinload()` in the route's
  query — async SQLAlchemy does not support implicit lazy-loading outside
  an active session, so a schema that includes a relationship the query
  didn't eager-load would raise `MissingGreenlet` at serialization time.
  Because of this, most resources have **two response shapes**: a flat one
  (matching the ORM's own columns, used for create/update/delete) and an
  expanded one adding `Optional[...Brief]` relation fields (used for list
  and get-by-id, where the query's `selectinload()` calls match exactly
  what the schema declares).

## 3. Authentication & RBAC

- `app/security.py`: bcrypt password hashing, PyJWT for signing/verifying.
- `app/deps.py`: `get_current_user` (a FastAPI dependency reading the
  `Authorization: Bearer` header via `fastapi.security.HTTPBearer`) and
  `require_role(allowed_roles)` (a dependency *factory* — call it with a
  role list from `app.types.ROLE_GROUPS` to get a dependency that 403s
  anyone outside that list). Both are the direct equivalents of
  `requireAuth`/`requireRole` in the Node version's middleware.
- `app/realtime/socket.py` authenticates Socket.IO connections with the
  same JWT via the handshake's `auth.token` field, and places each
  connection in an `org:<organizationId>` room so broadcasts never cross
  tenants — mirroring the REST side's org-scoping.

## 4–6. ARK Vision, ARK Proximity, the risk engine

Behaviorally identical to the Node version (`app/routers/vision.py`,
`app/routers/uwb.py`, `app/services/risk_engine.py`) — same fixed request
shapes for the two hardware-ingestion endpoints, same
`compute_risk(RiskInput) -> RiskResult` interface designed so a trained
model can later replace the function body without touching any caller, same
`isSimulated`/`is_simulated` convention so simulated and real data can never
be confused. `app/types.proximity_state_for()` centralizes the
SAFE/WARNING/DANGER classification exactly as the Node version's
`proximityStateFor()` does.

## 7. Simulation engine

`app/services/simulation.py`'s `SimulationEngine` class is a close,
tick-for-tick port of the Node version's engine: a deterministic ping-pong
route per forklift between two zone centers, a bounded random walk with
mild RED-zone repulsion for workers, and the same per-tick sequence
(move → telemetry → speed-limit check → nearest-worker proximity/risk →
occasional detection → restricted-zone check) with the same per-key
cooldowns to avoid re-firing the same transition every tick.

Implementation differences from the Node version, both consequences of
Python's concurrency model rather than behavior changes:
- The tick loop is `asyncio.create_task` + `asyncio.sleep(...)` instead of
  `setInterval`; `start()`/`pause()`/`reset()` create/cancel that task.
- Each tick opens one `AsyncSession`, does all of that tick's writes
  through it, and commits once at the end, rather than issuing one
  Prisma call per write. `session.flush()` (not `commit()`) is used after
  creating a `SafetyEvent`/`Alert` mid-tick specifically to obtain the
  new row's generated ID for the following `Alert.safety_event_id` link,
  without ending the transaction early.
- Movement state (`ForkliftAgent`/`WorkerAgent` dataclasses) is kept as
  plain in-memory Python objects, not ORM instances, precisely so it can
  survive across the many short-lived sessions each tick opens — an ORM
  object becomes unusable ("detached") once its originating session
  closes, so the engine never holds one across a tick boundary.

## 8. Real-time layer

`python-socketio`'s `AsyncServer` speaks the same Socket.IO/Engine.IO wire
protocol as the Node `socket.io` package, so it needs no client-side
changes. Event names and payloads (`simulation:tick`, `simulation:status`,
`safety_event:new`, `alert:new`, `alert:updated`) are identical to the Node
version. `app.realtime.socket.emit_to_org()` is the equivalent of the Node
version emitting to `` `org:${organizationId}` ``.

## 9. Future hardware integration path

Identical to the Node version — see its `ARCHITECTURE.md` §9. The three
ingestion endpoints (`POST /api/telemetry`, `POST /api/vision/detections`,
`POST /api/uwb/events`) are the entire surface a real integration needs;
nothing downstream of them issues a command back to a forklift.

## 10. ARK Predict

**Now implemented as machine-health anomaly detection (HTH-ML-10)** — see
[ANOMALY.md](./ANOMALY.md). `app/services/anomaly/` holds a pure-Python,
framework-independent pipeline (simulator → per-sensor detectors → Isolation
Forest + joint-shift Gaussian + temporal autoencoder → incidents/severity/routing
→ root-cause hints → technician feedback). `AnomalyService` runs
it as an asyncio task in the same event loop as the forklift simulation, keeps
state in memory (no database dependency), serves `/api/anomaly/*`
(`app/routers/anomaly.py`) and emits `anomaly:alert` over Socket.IO. The
ARK Predict page in `app/static/index.html` is the one view of the demo UI
that talks to the real backend.

The original, collision-prediction idea for ARK Predict below remains future work:
no endpoint exists for it, matching the browser demo's original
"Predictive model not trained" placeholder. The natural home for it later
is a new `app/services/predict.py` behind its own router, consuming the
`SafetyEvent`/`MaintenanceRecord`/`Telemetry` history this schema already
accumulates.

## 11. Error handling & operational concerns

- `app/errors.py` centralizes error responses via FastAPI exception
  handlers: `AppError` (expected conditions — 404/403/401/400, with
  `AppError.not_found()`/`.forbidden()`/`.unauthorized()` helpers),
  `RequestValidationError` (Pydantic validation → 422 with a field-level
  breakdown), `IntegrityError` (Postgres constraint violations → 409 for
  unique conflicts, 500 otherwise), and a catch-all for anything else
  (logged server-side, never leaked to the client).
- `GET /health` is a dependency-free liveness probe; `GET /health/db` runs
  `SELECT 1` to confirm the database connection specifically.
- The FastAPI `lifespan` context manager in `app/main.py` starts the
  simulation engine on boot (if `AUTO_START_SIMULATION`) and, on shutdown,
  pauses it and disposes the SQLAlchemy engine's connection pool — the
  async equivalent of the Node version's `SIGINT`/`SIGTERM` handlers.
