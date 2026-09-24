# ARK Shield — Server (Python / FastAPI)

A self-hostable backend for **ARK Shield: Intelligent Safety for Industrial Mobility**,
built with FastAPI, SQLAlchemy 2.0 (async) and PostgreSQL. This is a direct,
file-for-file Python port of the Node/Express version of this backend —
same data model, same REST endpoints, same real-time events, same
simulation behavior — for anyone who'd rather run (or extend) this stack
in Python.

> **Read this before anything else:** see
> [**"What's fully built vs. scaffolded"**](#whats-fully-built-vs-scaffolded)
> below. This is a genuine, working vertical slice — not a mockup — but it
> has **not been run against a live PostgreSQL database** in the environment
> that produced it (that sandbox has no network access, so `pip install` and
> `alembic upgrade` could not be executed there). It was built by porting the
> already-reviewed Node/Prisma version field-for-field and then statically
> checking every cross-module import and every `model_validate()` call site
> against the actual SQLAlchemy models — not by running it. Budget time for
> a first real `pip install && alembic upgrade head && python seed.py &&
> uvicorn app.main:app` pass.

## Same API, different stack

This server intentionally speaks the **same JSON contract** as the Node
version: identical URL paths, identical camelCase field names in requests
and responses, identical Socket.IO event names and payloads. Internally,
Python/SQLAlchemy code uses idiomatic snake_case column and variable names;
`app/camel.py` is the small shared layer that translates between the two at
the API boundary. Practically, this means:

- A frontend built against the Node backend's API can talk to this one
  without any changes.
- If you were given a link to the published **ARK Shield — Command Center**
  HTML demo artifact, it's the same product tour described in the Node
  version's README — a separate, fully client-side, self-contained page
  that needs no backend at all.

The one genuinely new capability here relative to the Node version: FastAPI
auto-generates interactive API docs. Once the server is running, open
`http://localhost:4000/docs` (Swagger UI) or `http://localhost:4000/redoc`.

## One project, one thing to run

This server also serves the interactive demo UI directly — open
`http://localhost:4000/` once the server is running (see "Getting started"
below) and you get the same **ARK Shield — Command Center** demo described
above, from the same process as the API. There's nothing extra to start.

**Important nuance:** that demo page runs its own self-contained,
client-side simulation in JavaScript — it was originally built to work with
no backend at all, and this project serves it as-is rather than rewriting
its UI. It does **not** call this server's REST API or Socket.IO; opening
it doesn't touch your database. What you get from running this project is
both things side by side: a real backend at `/api/*` you can build a
frontend against, and the existing demo at `/` you can click through
immediately. Pointing the demo's UI at real data from this backend instead
of its built-in simulation is exactly the kind of frontend work described
in "Building a real frontend against this API" below.

## ARK Predict — streaming anomaly detection (HTH-ML-10)

The **ARK Predict** page is now a working machine-health module: severity-aware
streaming anomaly detection over forklift sensors (motor temperature, hydraulic
pressure, mast vibration, battery voltage, motor current). It detects spike /
drift / dropout / stuck anomalies with per-sensor, load-aware statistical
detectors plus three ML models (Isolation Forest, a joint-shift Gaussian and a
deep temporal autoencoder, which catch overloads and oscillations no per-reading
rule can see), learns from technician verdicts, scores severity 0–100 with a breakdown of
why, suggests a likely root cause, and routes each incident to
IGNORE (log) / MONITOR (maintenance queue) / URGENT (page) while holding back
alert noise. It runs fully in memory, so **it works before Postgres is set up**.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 4000   # then open http://localhost:4000/ → ARK Predict
python -m app.services.anomaly.evaluate      # offline benchmark vs ground truth
python tests/test_anomaly.py                 # tests
```

Full write-up, results, demo script and API: **[ANOMALY.md](./ANOMALY.md)**.

**ARK Predict v2 (Review 3):** a supervised fault classifier now runs live on the stream and names *what* is
wrong (right fault type for 92 % of faults it never saw), fast-tracks bearing wear / hydraulic leaks to URGENT,
and retrains from technician verdicts. Every result is rebuilt by **`python scripts/review3.py`** →
**[FINAL_REPORT.md](./FINAL_REPORT.md)**.

**Tested on real industrial data:** on the SKAB benchmark (34 real pump experiments, official leaderboard
protocol) the fixed 3σ limit flags 44 % of normal time and fires 95 false alarms per hour. ARK Predict v2
(supervised model for known faults + deep-learning autoencoders for new ones) catches all 34 faults with F1 0.81,
a 24 % false-alarm rate and 12 false-alarm episodes per hour — trained only on earlier experiments. Details:
**[SKAB_RESULTS.md](./SKAB_RESULTS.md)**; live in the dashboard's **ML Lab** page.

## Tech stack

- **Runtime:** Python 3.11+, FastAPI, Uvicorn
- **Database:** PostgreSQL via SQLAlchemy 2.0 (async, `asyncpg` driver) + Alembic migrations
- **Real-time:** python-socketio (speaks the same Socket.IO wire protocol as
  the Node/Engine.IO server — any `socket.io-client` frontend connects
  unmodified)
- **Auth:** JWT (PyJWT), bcrypt-hashed passwords, 6-tier RBAC
- **Validation:** Pydantic v2

## Requirements

- Python 3.11+
- A PostgreSQL 14+ database (local install, Docker, or a hosted instance)

## Getting started

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# then edit .env — at minimum set DATABASE_URL and ALEMBIC_DATABASE_URL to
# point at your Postgres instance (same database, two driver URLs — see
# the comments in .env.example), and set JWT_SECRET to a real random value:
#   python -c "import secrets; print(secrets.token_hex(48))"

# 3. Create the database schema
alembic revision --autogenerate -m "init"
alembic upgrade head

# 4. Seed demo data (one org/site/warehouse, 6 demo users, 5 forklifts,
#    5 operators, zones, cameras, UWB anchors/tags, maintenance history)
python seed.py

# 5. Start the API server in dev mode (auto-restarts on changes)
uvicorn app.main:app --reload --port 4000
```

The server listens on the port you pass to uvicorn (4000 above, matching
`PORT` in `.env.example`). Open `http://localhost:4000/` for the demo UI,
check `http://localhost:4000/health` for a liveness probe, and
`http://localhost:4000/health/db` to confirm the database connection is
live. Interactive API docs live at `http://localhost:4000/docs`.

**Why an explicit `alembic revision --autogenerate` step:** this repository
ships Alembic's configuration (`alembic.ini`, `alembic/env.py`) but no
migration files, since none have ever been generated against a real
database. The command above creates the first migration from
`app/models.py`; from then on, `alembic upgrade head` alone is enough after
future model changes (followed by another `--autogenerate` revision).

### Demo logins

After seeding, these accounts exist (all roles, one user each), all sharing
the password `ArkShieldDemo!2026`:

| Role | Email |
|---|---|
| SUPER_ADMIN | admin@arkshield.demo |
| SAFETY_MANAGER | safety@arkshield.demo |
| FLEET_MANAGER | fleet@arkshield.demo |
| SUPERVISOR | supervisor@arkshield.demo |
| OPERATOR | operator@arkshield.demo |
| VIEWER | viewer@arkshield.demo |

**Change or remove these before any non-local deployment.**

### Simulation mode

With `AUTO_START_SIMULATION=true` (the default), the server starts
generating simulated telemetry/detections/proximity events/safety
events/alerts automatically on boot, on the first warehouse it finds. Every
record the simulation writes has `isSimulated: true` set explicitly.

Control it via REST (see [API.md](./API.md#simulation-control)) or run it as
its own process, independent of the HTTP/Socket.IO server:

```bash
python -m app.services.simulation
```

## Project structure

```
alembic/                Migration environment (no migrations committed yet — see above)
seed.py                 Demo data (port of prisma/seed.ts)
app/
  main.py               FastAPI + Socket.IO bootstrap — `app` is the ASGI entrypoint
  static/index.html      The demo UI, served at "/" (see "One project, one thing to run")
  config.py              Settings (pydantic-settings, reads .env)
  database.py            Async engine/session, declarative Base
  models.py               Full SQLAlchemy data model (port of schema.prisma)
  camel.py                 camelCase <-> snake_case translation for the API boundary
  types.py                 RBAC role groups, risk/proximity constants & helpers
  errors.py                AppError + FastAPI exception handlers
  security.py              Password hashing, JWT create/verify
  deps.py                   FastAPI dependencies: get_current_user, require_role
  services/
    risk_engine.py           Conceptual 0–100 risk score (see disclaimer below)
    simulation.py            Simulation engine (movement, risk, event generation)
  realtime/socket.py         python-socketio setup, JWT auth, per-org rooms
  routers/                  One file per resource (see API.md)
  utils/
    pagination.py            Shared pagination helper
    tenancy.py               Shared "does this warehouse belong to my org" check
```

## What's fully built vs. scaffolded

Identical scope to the Node version — see its README for the full
breakdown. In short, **fully built and wired end-to-end**: the multi-tenant
data model, JWT auth + 6-tier RBAC, org-scoped queries everywhere, CRUD for
forklifts/operators/zones/cameras/safety events/alerts, the three
hardware-abstraction ingestion endpoints, the risk engine, the simulation
engine, and the Socket.IO real-time layer. **Scaffolded / out of scope**:
admin CRUD for organizations/sites/users, notification-channel
configuration, ARK Predict (no endpoint — matches the demo's "not trained"
placeholder), and an automated test suite.

**Not executed in this environment:** `pip install`, `alembic upgrade`, and
the running server itself — the sandbox this was built in has no network
access. It was produced by porting the already-reviewed Node/Prisma version
line-by-line and then statically verifying (via AST parsing, not execution)
that every cross-module import resolves and every Pydantic schema's fields
match the SQLAlchemy model it validates from. Two real bugs were caught and
fixed this way during the port itself: a place where an eagerly-unloaded
ORM relationship would have crashed the async event loop on first access,
and a copied-and-pasted query helper that joined through the wrong table
when checking warehouse ownership for a plain `Warehouse` lookup. Treat this
as a strong, carefully-reviewed first draft, not pre-verified code.

## Two small, deliberate differences from the Node/Prisma version

- **Primary keys** are `uuid4().hex` strings instead of Prisma's `cuid()`.
  Both are non-sequential unique string IDs; nothing depends on the exact
  format.
- **Three foreign-key columns** that were unenforced plain columns in the
  Prisma schema (`SafetyEvent.operator_id`, `Alert.zone_id`,
  `Alert.acknowledged_by_user_id`) get a real `ForeignKey` constraint here
  for referential integrity, without an ORM relationship attribute — nothing
  reads them as a relation, so this only tightens the database, it doesn't
  change any route's behavior.

## Building a real frontend against this API

Identical to the Node version:

1. `POST /api/auth/login` with `{ email, password }` → receive a JWT.
2. Send `Authorization: Bearer <token>` on every subsequent REST call.
3. Open a Socket.IO connection with `io(url, { auth: { token } })` using the
   same JWT, and listen for `simulation:tick`, `safety_event:new`,
   `alert:new`, `alert:updated`, `simulation:status`.
4. Point `NEXT_PUBLIC_API_URL` / `API_URL` at this server.

See [API.md](./API.md) for the full endpoint reference and
[ARCHITECTURE.md](./ARCHITECTURE.md) for how the pieces fit together.

## Safety disclaimer

Identical to the Node version — this is a decision-support and monitoring
aid, not a safety-certified collision-prevention or vehicle-control system.
The risk engine (`app/services/risk_engine.py`) is an illustrative, weighted
formula, **not validated against real-world collision data**. Nothing in
this codebase actuates, brakes, or otherwise controls a real forklift — the
three ingestion endpoints are one-directional. **Emergency stop systems,
physical safety interlocks, and any life-safety-critical function must be
independently engineered, tested and certified** to applicable industrial
safety standards; they are explicitly out of scope here. Simulated data is
always marked `isSimulated: true` so it can never be mistaken for a real
sensor reading in an audit or incident report.

## Troubleshooting

- **`sqlalchemy.exc.OperationalError` / connection refused** — check
  `DATABASE_URL` in `.env` and that Postgres is running and reachable.
- **Alembic can't connect** — Alembic uses `ALEMBIC_DATABASE_URL` (the sync
  `psycopg2` driver), a separate setting from `DATABASE_URL` (the async
  `asyncpg` driver) — make sure both point at the same database.
- **500 on every request, "JWT_SECRET"-related** — set a real `JWT_SECRET`
  in `.env`.
- **Login returns 401 for a seeded demo user** — confirm `python seed.py`
  ran after `alembic upgrade head`, and that you're using the exact password
  `ArkShieldDemo!2026`.
- **Simulation isn't generating anything** — it needs at least one
  warehouse with zones and forklifts in the database; run `python seed.py`
  first, or check `GET /api/simulation/status`.
