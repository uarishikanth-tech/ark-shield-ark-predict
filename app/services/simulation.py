"""
ARK SHIELD — Simulation Engine (Python port of src/services/simulation.ts)
------------------------------------------------------------------
Generates realistic forklift/worker movement, UWB proximity data and
camera detections, then writes them through SQLAlchemy exactly as
real hardware telemetry eventually would. Every record this engine
writes sets is_simulated = True so it can never be confused with real
sensor data.

Runs two ways, same as the Node version:
  1. Embedded — app/main.py calls engine.start() on startup so ticks
     broadcast over the same Socket.IO server as the REST API.
  2. Standalone — `python -m app.services.simulation` runs the engine
     with no HTTP/Socket.IO server; ticks are only logged.

This module has NOT been executed against a live PostgreSQL instance
in the environment that produced it — see the README.
"""
import asyncio
import math
import random
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import AsyncSessionLocal, engine as db_engine
from app.errors import AppError
from app.models import (
    Alert,
    Camera,
    Detection,
    ProximityEvent,
    SafetyEvent,
    Telemetry,
    Warehouse,
)
from app.services.risk_engine import RiskInput, compute_risk
from app.types import DEFAULT_SAFE_M, DEFAULT_WARNING_M, proximity_state_for


@dataclass
class ZoneInfo:
    id: str
    x: float
    y: float
    width: float
    height: float
    severity: str
    speed_limit_kmh: int


@dataclass
class ForkliftAgent:
    id: str
    asset_number: str
    status: str
    operator_id: str | None
    route: list[tuple[float, float]]
    route_index: int = 0
    x: float = 0.0
    y: float = 0.0
    speed_kmh: float = 0.0


@dataclass
class WorkerAgent:
    label: str
    x: float
    y: float
    vx: float
    vy: float


class SimulationEngine:
    def __init__(self) -> None:
        self.running = False
        self._task: asyncio.Task | None = None
        self._sio = None  # set by start(sio) when run embedded

        self.warehouse_id: str | None = None
        self.organization_id: str | None = None
        self.bounds = {"width": 960.0, "height": 600.0}
        self.zones: list[ZoneInfo] = []
        self.forklift_agents: list[ForkliftAgent] = []
        self.worker_agents: list[WorkerAgent] = []
        self.thresholds = {"safe_m": DEFAULT_SAFE_M, "warning_m": DEFAULT_WARNING_M}

        self._last_proximity_state: dict[str, str] = {}
        self._last_alert_at: dict[str, float] = {}
        self.traffic_level = 1

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    async def load_warehouse(self) -> bool:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Warehouse)
                .options(
                    selectinload(Warehouse.zones),
                    selectinload(Warehouse.forklifts),
                    selectinload(Warehouse.uwb_tags),
                    selectinload(Warehouse.site),
                )
                .order_by(Warehouse.created_at.asc())
                .limit(1)
            )
            warehouse = result.scalars().first()
            if warehouse is None:
                print("[simulation] No warehouse found — run `python seed.py` first.")
                return False

            self.warehouse_id = warehouse.id
            self.organization_id = warehouse.site.organization_id
            self.zones = [
                ZoneInfo(z.id, z.x, z.y, z.width, z.height, z.severity.value, z.speed_limit_kmh)
                for z in warehouse.zones
            ]

            max_x = max([z.x + z.width for z in self.zones] + [900.0])
            max_y = max([z.y + z.height for z in self.zones] + [560.0])
            self.bounds = {"width": max_x + 40, "height": max_y + 40}

            non_restricted = [z for z in self.zones if z.severity != "RED"] or self.zones
            self.forklift_agents = []
            for i, f in enumerate(warehouse.forklifts):
                zone_a = non_restricted[i % len(non_restricted)]
                zone_b = non_restricted[(i + 1) % len(non_restricted)]

                def center(z: ZoneInfo) -> tuple[float, float]:
                    return (z.x + z.width / 2, z.y + z.height / 2)

                route = [center(zone_a), center(zone_b)]
                self.forklift_agents.append(
                    ForkliftAgent(
                        id=f.id,
                        asset_number=f.asset_number,
                        status=f.status.value,
                        operator_id=f.operator_id,
                        route=route,
                        x=route[0][0],
                        y=route[0][1],
                    )
                )

            worker_tags = [t for t in warehouse.uwb_tags if t.kind == "WORKER"]
            self.worker_agents = [
                WorkerAgent(
                    label=t.worker_label or f"W-{i+1}",
                    x=40 + random.random() * (self.bounds["width"] - 80),
                    y=40 + random.random() * (self.bounds["height"] - 80),
                    vx=(random.random() - 0.5) * 2,
                    vy=(random.random() - 0.5) * 2,
                )
                for i, t in enumerate(worker_tags)
            ]
            return True

    async def start(self, sio=None) -> None:
        if self.running:
            return
        if sio is not None:
            self._sio = sio

        if not self.warehouse_id:
            ok = await self.load_warehouse()
            if not ok:
                return

        self.running = True
        await self._emit("simulation:status", {"running": True})
        self._task = asyncio.create_task(self._loop())
        print(f"[simulation] started (tick every {settings.simulation_tick_seconds}s)")

    async def pause(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None
        await self._emit("simulation:status", {"running": False})
        print("[simulation] paused")

    async def reset(self) -> None:
        await self.pause()
        self.warehouse_id = None
        self._last_proximity_state.clear()
        self._last_alert_at.clear()
        self.traffic_level = 1
        await self.load_warehouse()
        await self._emit("simulation:status", {"running": False, "reset": True})
        print("[simulation] reset")

    def increase_traffic(self) -> None:
        self.traffic_level = min(self.traffic_level + 1, 4)
        for agent in self.forklift_agents:
            if agent.status not in ("ACTIVE", "MAINTENANCE"):
                agent.status = "ACTIVE"
                break
        print(f"[simulation] traffic level -> {self.traffic_level}")

    def set_thresholds(self, safe_m: float, warning_m: float) -> None:
        if warning_m >= safe_m:
            raise AppError("warningM must be less than safeM", 400)
        self.thresholds = {"safe_m": safe_m, "warning_m": warning_m}

    async def trigger_warning(self) -> None:
        """DEMO ONLY — forces a WARNING-level proximity alert."""
        if not self.forklift_agents:
            return
        async with AsyncSessionLocal() as session:
            await self._record_proximity(session, self.forklift_agents[0], "DEMO-W", self.thresholds["warning_m"] + 1)
            await session.commit()

    async def trigger_critical(self) -> None:
        """DEMO ONLY — forces a CRITICAL safety event. Never actuates real machinery."""
        if not self.forklift_agents:
            return
        risk = compute_risk(
            RiskInput(distance_m=1.2, speed_kmh=12, converging_trajectory=True, zone_severity="RED", object_type="pedestrian")
        )
        async with AsyncSessionLocal() as session:
            await self._record_safety_event(
                session,
                self.forklift_agents[0],
                "COLLISION_RISK",
                "CRITICAL",
                detected_object="pedestrian",
                distance_m=1.2,
                risk_score=risk.score,
                forced_demo=True,
            )
            await session.commit()

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "warehouseId": self.warehouse_id,
            "trafficLevel": self.traffic_level,
            "forklifts": len(self.forklift_agents),
            "workers": len(self.worker_agents),
            "thresholds": {"safeM": self.thresholds["safe_m"], "warningM": self.thresholds["warning_m"]},
        }

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(settings.simulation_tick_seconds)
                try:
                    await self._tick()
                except Exception as err:  # noqa: BLE001 - keep the loop alive across a single bad tick
                    print(f"[simulation] tick error: {err!r}")
        except asyncio.CancelledError:
            pass

    async def _tick(self) -> None:
        if not self.warehouse_id:
            return

        self._move_forklifts()
        self._move_workers()

        async with AsyncSessionLocal() as session:
            for agent in self.forklift_agents:
                if agent.status != "ACTIVE":
                    continue
                await self._tick_forklift(session, agent)
            await session.commit()

        await self._emit(
            "simulation:tick",
            {
                "forklifts": [
                    {
                        "id": a.id,
                        "assetNumber": a.asset_number,
                        "x": a.x,
                        "y": a.y,
                        "speedKmh": round(a.speed_kmh, 1),
                        "status": a.status,
                    }
                    for a in self.forklift_agents
                ],
                "workers": [{"label": w.label, "x": w.x, "y": w.y} for w in self.worker_agents],
                "isSimulated": True,
            },
        )

    async def _tick_forklift(self, session: AsyncSession, agent: ForkliftAgent) -> None:
        session.add(
            Telemetry(forklift_id=agent.id, x=agent.x, y=agent.y, speed_kmh=agent.speed_kmh, is_simulated=True)
        )

        zone = self._zone_at(agent.x, agent.y)

        if zone and agent.speed_kmh > zone.speed_limit_kmh + 1:
            await self._record_safety_event(session, agent, "SPEED_VIOLATION", "MEDIUM", zone_id=zone.id)

        nearest = self._nearest_worker(agent)
        if nearest:
            worker_label, distance_m, wx, wy = nearest
            await self._record_proximity(session, agent, worker_label, distance_m, zone)

            risk = compute_risk(
                RiskInput(
                    distance_m=distance_m,
                    speed_kmh=agent.speed_kmh,
                    converging_trajectory=distance_m < 10 and random.random() < 0.4,
                    zone_severity=(zone.severity if zone else "GREEN"),
                    object_type="pedestrian",
                )
            )

            if random.random() < 0.5:
                camera_result = await session.execute(select(Camera).where(Camera.forklift_id == agent.id))
                camera = camera_result.scalars().first()
                if camera:
                    session.add(
                        Detection(
                            camera_id=camera.id,
                            forklift_id=agent.id,
                            object_type="pedestrian",
                            confidence=0.7 + random.random() * 0.29,
                            distance_m=distance_m,
                            direction=self._direction_label(agent, wx, wy),
                            risk_score=risk.score,
                            is_simulated=True,
                        )
                    )

            if risk.state in ("HIGH", "CRITICAL"):
                await self._record_safety_event(
                    session,
                    agent,
                    "COLLISION_RISK",
                    risk.state,
                    detected_object="pedestrian",
                    distance_m=distance_m,
                    risk_score=risk.score,
                    zone_id=zone.id if zone else None,
                )

        if zone and zone.severity == "RED":
            await self._record_safety_event(session, agent, "RESTRICTED_ZONE", "HIGH", zone_id=zone.id)

    # ----------------------------------------------------------------
    # Movement (pure in-memory, no DB)
    # ----------------------------------------------------------------
    def _move_forklifts(self) -> None:
        for agent in self.forklift_agents:
            if agent.status != "ACTIVE":
                agent.speed_kmh = 0
                continue
            target = agent.route[agent.route_index]
            dx, dy = target[0] - agent.x, target[1] - agent.y
            dist = math.hypot(dx, dy)

            base_speed = 1.5 + self.traffic_level * 0.5
            if dist < base_speed:
                agent.route_index = (agent.route_index + 1) % len(agent.route)
            else:
                agent.x += (dx / dist) * base_speed
                agent.y += (dy / dist) * base_speed
            agent.speed_kmh = 4 + self.traffic_level * 1.5 + random.random() * 2

    def _move_workers(self) -> None:
        restricted = [z for z in self.zones if z.severity == "RED"]
        for w in self.worker_agents:
            w.x += w.vx
            w.y += w.vy

            if w.x < 20 or w.x > self.bounds["width"] - 20:
                w.vx *= -1
            if w.y < 20 or w.y > self.bounds["height"] - 20:
                w.vy *= -1

            for rz in restricted:
                cx, cy = rz.x + rz.width / 2, rz.y + rz.height / 2
                dx, dy = w.x - cx, w.y - cy
                dist = math.hypot(dx, dy) or 1
                temptation = 0.05 * self.traffic_level
                if dist < 120 and random.random() > temptation:
                    w.vx += (dx / dist) * 0.3
                    w.vy += (dy / dist) * 0.3

            speed = math.hypot(w.vx, w.vy)
            max_speed = 1.2
            if speed > max_speed:
                w.vx, w.vy = (w.vx / speed) * max_speed, (w.vy / speed) * max_speed
            w.x = max(10.0, min(self.bounds["width"] - 10, w.x))
            w.y = max(10.0, min(self.bounds["height"] - 10, w.y))

    def _zone_at(self, x: float, y: float) -> ZoneInfo | None:
        for z in self.zones:
            if z.x <= x <= z.x + z.width and z.y <= y <= z.y + z.height:
                return z
        return None

    def _nearest_worker(self, agent: ForkliftAgent) -> tuple[str, float, float, float] | None:
        best = None
        for w in self.worker_agents:
            dist_px = math.hypot(w.x - agent.x, w.y - agent.y)
            distance_m = round(dist_px * 0.08, 1)  # ~0.08m per layout px, illustrative warehouse scale
            if best is None or distance_m < best[1]:
                best = (w.label, distance_m, w.x, w.y)
        return best

    def _direction_label(self, agent: ForkliftAgent, wx: float, wy: float) -> str:
        angle = math.degrees(math.atan2(wy - agent.y, wx - agent.x))
        dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
        idx = round(((angle + 360) % 360) / 45) % 8
        return dirs[idx]

    # ----------------------------------------------------------------
    # Persistence helpers
    # ----------------------------------------------------------------
    def _cooldown_ok(self, key: str, cooldown_seconds: float = 15.0) -> bool:
        now = time.monotonic()
        last = self._last_alert_at.get(key, 0.0)
        if now - last < cooldown_seconds:
            return False
        self._last_alert_at[key] = now
        return True

    async def _record_proximity(
        self, session: AsyncSession, agent: ForkliftAgent, worker_label: str, distance_m: float, zone: ZoneInfo | None = None
    ) -> None:
        state = proximity_state_for(distance_m, self.thresholds["safe_m"], self.thresholds["warning_m"])
        key = f"{agent.id}:{worker_label}"
        prev = self._last_proximity_state.get(key)

        session.add(
            ProximityEvent(
                forklift_id=agent.id,
                zone_id=zone.id if zone else None,
                worker_label=worker_label,
                distance_m=distance_m,
                state=state,
                is_simulated=True,
            )
        )

        if state != prev and state in ("WARNING", "DANGER") and self._cooldown_ok(key):
            severity = "HIGH" if state == "DANGER" else "MEDIUM"
            event = await self._record_safety_event(
                session,
                agent,
                "PEDESTRIAN_PROXIMITY",
                severity,
                detected_object="pedestrian",
                distance_m=distance_m,
                zone_id=zone.id if zone else None,
            )
            await self._create_alert(
                session,
                agent,
                event.id if event else None,
                severity,
                title=f"DANGER — PEDESTRIAN TOO CLOSE ({worker_label})"
                if state == "DANGER"
                else f"PROXIMITY WARNING ({worker_label})",
                worker_label=worker_label,
                distance_m=distance_m,
            )
        self._last_proximity_state[key] = state

    async def _record_safety_event(
        self,
        session: AsyncSession,
        agent: ForkliftAgent,
        type_: str,
        severity: str,
        detected_object: str | None = None,
        distance_m: float | None = None,
        risk_score: int | None = None,
        zone_id: str | None = None,
        forced_demo: bool = False,
    ) -> SafetyEvent | None:
        cooldown_key = f"event:{agent.id}:{type_}"
        if not forced_demo and not self._cooldown_ok(cooldown_key, 20.0):
            return None

        event = SafetyEvent(
            warehouse_id=self.warehouse_id,
            type=type_,
            severity=severity,
            forklift_id=agent.id,
            operator_id=agent.operator_id,
            zone_id=zone_id,
            detected_object=detected_object,
            distance_m=distance_m,
            risk_score=risk_score,
            source="ARK_PROXIMITY" if type_ in ("PEDESTRIAN_PROXIMITY", "UWB_ALERT") else "ARK_VISION",
            is_simulated=True,
        )
        session.add(event)
        await session.flush()  # assigns event.id without committing the transaction

        # PEDESTRIAN_PROXIMITY alerts are created explicitly by
        # _record_proximity (which also fires for MEDIUM/WARNING, not
        # just HIGH/CRITICAL) — skip the generic auto-alert here so a
        # DANGER-level proximity reading doesn't produce two alerts.
        if type_ != "PEDESTRIAN_PROXIMITY" and severity in ("HIGH", "CRITICAL"):
            await self._create_alert(
                session,
                agent,
                event.id,
                severity,
                title=f"{severity} — {type_.replace('_', ' ')}",
                distance_m=distance_m,
                risk_score=risk_score,
            )

        await self._emit("safety_event:new", {"id": event.id, "type": type_, "severity": severity, "forkliftId": agent.id})
        return event

    async def _create_alert(
        self,
        session: AsyncSession,
        agent: ForkliftAgent,
        safety_event_id: str | None,
        severity: str,
        title: str,
        worker_label: str | None = None,
        distance_m: float | None = None,
        risk_score: int | None = None,
    ) -> None:
        alert = Alert(
            warehouse_id=self.warehouse_id,
            safety_event_id=safety_event_id,
            severity=severity,
            title=title,
            forklift_id=agent.id,
            worker_label=worker_label,
            distance_m=distance_m,
            risk_score=risk_score,
            status="ACTIVE",
        )
        session.add(alert)
        await session.flush()
        await self._emit(
            "alert:new",
            {
                "id": alert.id,
                "warehouseId": alert.warehouse_id,
                "severity": severity,
                "title": alert.title,
                "forkliftId": alert.forklift_id,
                "workerLabel": alert.worker_label,
                "distanceM": alert.distance_m,
                "riskScore": alert.risk_score,
                "status": "ACTIVE",
            },
        )

    async def _emit(self, event: str, payload) -> None:
        if self._sio is None:
            return
        if self.organization_id:
            await self._sio.emit(event, payload, room=f"org:{self.organization_id}")
        else:
            await self._sio.emit(event, payload)


simulation_engine = SimulationEngine()


async def start_simulation(sio=None) -> None:
    await simulation_engine.start(sio)


# ------------------------------------------------------------------
# Standalone entrypoint: `python -m app.services.simulation`
# ------------------------------------------------------------------
async def _standalone_main() -> None:
    print("[simulation] running in standalone mode (no HTTP/Socket.IO server)")
    await simulation_engine.start()
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(_standalone_main())
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(db_engine.dispose())
