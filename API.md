# ARK Shield — API Reference (Python / FastAPI)

This is the same REST + Socket.IO contract as the Node version of this
backend — every path, field name, and event name below is identical, so
this document is self-contained if you only have this package. Interactive,
always-up-to-date docs are also generated automatically at
`http://localhost:4000/docs` (Swagger UI) once the server is running.

Base URL: `http://localhost:4000` (or whatever you pass to `uvicorn --port`).

## Auth

Every route below except `POST /api/auth/login` requires:

```
Authorization: Bearer <token>
```

Tokens come from `POST /api/auth/login` and expire per `JWT_EXPIRES_HOURS`
(default 12h). All data is scoped to the organization embedded in the
token.

## Conventions

- **Pagination**: list endpoints accept `?page=1&pageSize=25` (max
  `pageSize` 100) and return `{ data, page, pageSize, total, totalPages }`.
- **Errors**: `{ "error": "message" }`, plus `{ "details": [...] }` on
  validation failures (422). Status codes: 400 bad request, 401
  unauthenticated, 403 forbidden, 404 not found, 409 conflict, 422
  validation, 500/503 server/database error.
- **Roles**: SUPER_ADMIN, SAFETY_MANAGER, FLEET_MANAGER, SUPERVISOR,
  OPERATOR, VIEWER.
- **Casing**: request and response bodies use camelCase, matching the Node
  version, even though this server is Python underneath.

---

## Auth

### `POST /api/auth/login`
```json
// request
{ "email": "safety@arkshield.demo", "password": "ArkShieldDemo!2026" }
// response
{ "token": "...", "user": { "id", "email", "name", "role", "organizationId" } }
```

### `GET /api/auth/me`
Returns the current user resolved fresh from the database.

---

## Organizations / Sites / Warehouses (read-only)

- `GET /api/organizations/me`
- `GET /api/sites`
- `GET /api/warehouses?siteId=`

## Forklifts — `/api/forklifts`

- `GET /` — `?status=&warehouseId=&search=&page=&pageSize=`
- `GET /{id}` — includes operator, camera, UWB device, recent maintenance
- `GET /{id}/events` — paginated safety events for this forklift
- `POST /` — **FLEET_MANAGER, SUPER_ADMIN**: `{ warehouseId, assetNumber,
  manufacturer, model, type, capacityLbs, status?, operatorId?,
  fuelPercent?, batteryPercent? }` (`type` ∈ INTERNAL_COMBUSTION | ELECTRIC
  | LPG | OTHER)
- `PATCH /{id}` — **FLEET_MANAGER, SUPER_ADMIN**
- `DELETE /{id}` — **FLEET_MANAGER, SUPER_ADMIN**

## Operators — `/api/operators`

- `GET /` — `?warehouseId=&authStatus=&page=&pageSize=`
- `GET /{id}`
- `POST /` — **FLEET_MANAGER, SUPER_ADMIN**: `{ warehouseId, externalId,
  name, authStatus?, trainingStatus? }`
- `PATCH /{id}` / `DELETE /{id}` — **FLEET_MANAGER, SUPER_ADMIN**

## Zones — `/api/zones`

- `GET /` — `?warehouseId=`
- `GET /{id}`
- `POST /` — **SAFETY_MANAGER, FLEET_MANAGER, SUPER_ADMIN**: `{ warehouseId,
  name, kind?, severity?, speedLimitKmh?, x, y, width, height }`
- `PATCH /{id}` / `DELETE /{id}` — same roles

## Safety Events — `/api/safety-events`

- `GET /` — `?warehouseId=&severity=&type=&status=&forkliftId=&operatorId=
  &zoneId=&from=&to=&page=&pageSize=`
- `GET /{id}` — includes forklift, zone, linked alerts
- `PATCH /{id}` — **SAFETY_MANAGER, FLEET_MANAGER, SUPER_ADMIN**:
  `{ status }` (OPEN | REVIEWED | CLOSED)

`severity` ∈ INFO | LOW | MEDIUM | HIGH | CRITICAL.
`type` ∈ PEDESTRIAN_PROXIMITY | COLLISION_RISK | RESTRICTED_ZONE |
SPEED_VIOLATION | IMPACT | UNAUTHORIZED_OPERATION | CAMERA_ALERT | UWB_ALERT.

## Alerts — `/api/alerts`

- `GET /` — `?warehouseId=&status=&severity=&forkliftId=&page=&pageSize=`
- `POST /{id}/acknowledge` — **SUPERVISOR and above**; broadcasts
  `alert:updated`
- `POST /{id}/resolve` — **SUPERVISOR and above**; broadcasts `alert:updated`

## Telemetry — `/api/telemetry`

- `GET /` — `?forkliftId=&page=&pageSize=`
- `POST /` — hardware ingestion: `{ forkliftId, uwbTagId?, x, y, speedKmh,
  isSimulated? }` (defaults `isSimulated` false)

## ARK Vision — `/api/vision`

- `GET /detections?forkliftId=` — last 50 detections
- `POST /detections` — `{ cameraId, forkliftId, objectType, confidence,
  distanceM?, direction?, speedKmh?, convergingTrajectory?, zoneSeverity?,
  isSimulated? }` → `{ detection, risk }`

## ARK Proximity / UWB — `/api/uwb`

- `GET /anchors?warehouseId=`
- `GET /tags?warehouseId=&kind=`
- `GET /events?forkliftId=` — last 100 readings
- `POST /events` — `{ forkliftId, zoneId?, workerLabel, distanceM, safeM?,
  warningM?, isSimulated? }`

## Workers — `/api/workers`

- `GET /?warehouseId=` — `{ tagId, workerLabel, status, nearestForklift,
  distanceM, riskState, lastSeenAt }` per worker tag. No biometric data.

## Cameras — `/api/cameras`

- `GET /?warehouseId=`
- `PATCH /{id}` — **SAFETY_MANAGER, FLEET_MANAGER, SUPER_ADMIN**:
  `{ status?, forkliftId? }`

## Analytics — `/api/analytics`

- `GET /overview?range=today|7d|30d&from=&to=&forkliftId=&operatorId=
  &severity=&type=`

## Simulation control — `/api/simulation`

- `GET /status`
- `POST /start` / `/pause` / `/reset` / `/increase-traffic` — **SAFETY_MANAGER,
  FLEET_MANAGER, SUPER_ADMIN**
- `POST /thresholds` — same roles: `{ safeM, warningM }`
- `POST /trigger-warning` / `/trigger-critical` — same roles, **demo only**

---

## Socket.IO

```js
const socket = io('http://localhost:4000', { auth: { token } });
socket.on('simulation:tick', (payload) => { /* forklift + worker positions */ });
socket.on('simulation:status', (payload) => { /* { running: boolean } */ });
socket.on('safety_event:new', (payload) => { /* { id, type, severity, forkliftId } */ });
socket.on('alert:new', (alert) => { /* full Alert row */ });
socket.on('alert:updated', (alert) => { /* full Alert row, after ack/resolve */ });
```

python-socketio speaks the same wire protocol as the Node server, so this
client code is identical either way.

---

## Health

- `GET /health` — process liveness, no database dependency
- `GET /health/db` — runs `SELECT 1` to confirm the database connection

---

## ARK Predict — anomaly detection — `/api/anomaly`

In-memory, no database required. Open without a JWT while
`ANOMALY_PUBLIC_DEMO=true` (default); otherwise reads need any logged-in
user and POST controls need **SAFETY_OR_FLEET**. Full reference and
payload notes: [ANOMALY.md](./ANOMALY.md#api-apianomaly).

- `GET /status` · `GET /config` · `GET /machines`
- `GET /series?machineId=&points=` — chart data incl. incident bands + injected ground truth
- `GET /alerts?limit=` — routed notifications · `GET /feed?limit=` — every routing decision
- `GET /incidents?status=open|all` · `GET /incidents/{id}`
- `GET /summary` · `GET /report.md` — end-of-run report + evaluation vs ground truth
- `POST /start` · `/pause` · `/reset {seed?}` · `/speed {speed}` · `/fault-rate {ratePerMin}`
- `POST /inject {scenario, machineId?, sensor?, durationS?, magnitude?}` — DEMO fault injection
- `POST /feedback {incidentId, label}` — technician verdict (`right_call` · `too_high` · `too_low` · `false_alarm`); `GET /feedback` — what has been learned

Socket.IO event: `anomaly:alert` (same payload as `/alerts` items).
