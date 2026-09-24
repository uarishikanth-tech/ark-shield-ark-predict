"""
ARK SHIELD — seed script (port of prisma/seed.ts)
Populates one demo organization/site/warehouse with fleet, operators,
devices, zones and a handful of historical maintenance records so the
app is not empty on first run.

Run with: python seed.py
"""
import asyncio
from datetime import datetime, timezone

import bcrypt

from app.database import AsyncSessionLocal, engine as db_engine
from app.models import (
    Camera,
    DeviceStatus,
    Forklift,
    ForkliftStatus,
    ForkliftType,
    MaintenanceRecord,
    MaintenanceStatus,
    Operator,
    OperatorAuthStatus,
    Organization,
    Role,
    Site,
    TrainingStatus,
    User,
    UWBAnchor,
    UWBTag,
    Warehouse,
    Zone,
    ZoneKind,
    ZoneSeverity,
)


def d(iso_date: str) -> datetime:
    return datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc)


async def main() -> None:
    print("Seeding ARK Shield demo data...")

    async with AsyncSessionLocal() as session:
        org = Organization(name="Meridian Logistics Group")
        session.add(org)
        await session.flush()

        site = Site(name="Riverside Distribution Site", organization_id=org.id)
        session.add(site)
        await session.flush()

        warehouse = Warehouse(name="Warehouse 4 — Riverside DC", site_id=site.id)
        session.add(warehouse)
        await session.flush()

        # --- Users (one per role, for demo login) ---
        password_hash = bcrypt.hashpw(b"ArkShieldDemo!2026", bcrypt.gensalt()).decode("utf-8")
        role_defs = [
            (Role.SUPER_ADMIN, "admin@arkshield.demo", "Jordan Reyes"),
            (Role.SAFETY_MANAGER, "safety@arkshield.demo", "Alicia Chen"),
            (Role.FLEET_MANAGER, "fleet@arkshield.demo", "Ben Ortiz"),
            (Role.SUPERVISOR, "supervisor@arkshield.demo", "Kim Patel"),
            (Role.OPERATOR, "operator@arkshield.demo", "Daniel Osei"),
            (Role.VIEWER, "viewer@arkshield.demo", "Guest Viewer"),
        ]
        for role, email, name in role_defs:
            session.add(User(role=role, email=email, name=name, password_hash=password_hash, organization_id=org.id))
        await session.flush()

        # --- Zones ---
        zone_defs = [
            dict(name="Loading Bay", kind=ZoneKind.LOADING_AREA, severity=ZoneSeverity.AMBER, speed_limit_kmh=8, x=20, y=20, width=250, height=170),
            dict(name="Warehouse A", kind=ZoneKind.STORAGE_RACK_AREA, severity=ZoneSeverity.GREEN, speed_limit_kmh=10, x=300, y=20, width=330, height=250),
            dict(name="Warehouse B", kind=ZoneKind.STORAGE_RACK_AREA, severity=ZoneSeverity.GREEN, speed_limit_kmh=10, x=300, y=290, width=330, height=250),
            dict(name="Charging Station", kind=ZoneKind.CHARGING_AREA, severity=ZoneSeverity.GREEN, speed_limit_kmh=5, x=660, y=20, width=150, height=120),
            dict(name="Restricted Maintenance Area", kind=ZoneKind.RESTRICTED_ZONE, severity=ZoneSeverity.RED, speed_limit_kmh=0, x=660, y=160, width=150, height=120),
        ]
        for z in zone_defs:
            session.add(Zone(warehouse_id=warehouse.id, **z))
        await session.flush()

        # --- Operators ---
        operator_defs = [
            dict(external_id="OP-001", name="Daniel Osei", auth_status=OperatorAuthStatus.AUTHORIZED, training_status=TrainingStatus.CURRENT, operating_hours=1840),
            dict(external_id="OP-002", name="Priya Nandakumar", auth_status=OperatorAuthStatus.AUTHORIZED, training_status=TrainingStatus.CURRENT, operating_hours=2210),
            dict(external_id="OP-003", name="Marcus Webb", auth_status=OperatorAuthStatus.AUTHORIZED, training_status=TrainingStatus.DUE_FOR_RENEWAL, operating_hours=1590),
            dict(external_id="OP-004", name="Elena Cruz", auth_status=OperatorAuthStatus.AUTHORIZED, training_status=TrainingStatus.CURRENT, operating_hours=980),
            dict(external_id="OP-005", name="Sam Whitfield", auth_status=OperatorAuthStatus.SUSPENDED, training_status=TrainingStatus.EXPIRED, operating_hours=410),
        ]
        operators = []
        for o in operator_defs:
            op = Operator(warehouse_id=warehouse.id, **o)
            session.add(op)
            operators.append(op)
        await session.flush()

        # --- Forklifts ---
        forklift_defs = [
            dict(asset_number="FE-10231", manufacturer="Hyster", model="H50FT", type=ForkliftType.INTERNAL_COMBUSTION, capacity_lbs=5000, status=ForkliftStatus.ACTIVE, operator_idx=0, fuel_percent=68, last_service_at=d("2026-08-02"), next_service_at=d("2026-11-02")),
            dict(asset_number="FE-10232", manufacturer="Toyota", model="8FBE20", type=ForkliftType.ELECTRIC, capacity_lbs=4000, status=ForkliftStatus.ACTIVE, operator_idx=1, battery_percent=74, last_service_at=d("2026-07-18"), next_service_at=d("2026-10-18")),
            dict(asset_number="FE-10233", manufacturer="Crown", model="FC 5200", type=ForkliftType.ELECTRIC, capacity_lbs=4500, status=ForkliftStatus.ACTIVE, operator_idx=2, battery_percent=55, last_service_at=d("2026-08-21"), next_service_at=d("2026-11-21")),
            dict(asset_number="FE-10234", manufacturer="Crown", model="RC 5500", type=ForkliftType.ELECTRIC, capacity_lbs=5000, status=ForkliftStatus.CHARGING, operator_idx=3, battery_percent=32, last_service_at=d("2026-06-30"), next_service_at=d("2026-09-30")),
            dict(asset_number="FE-10235", manufacturer="Hyster", model="H50FT", type=ForkliftType.INTERNAL_COMBUSTION, capacity_lbs=5000, status=ForkliftStatus.OFFLINE, operator_idx=None, fuel_percent=40, last_service_at=d("2026-05-14"), next_service_at=d("2026-08-14")),
        ]
        forklifts = []
        for f in forklift_defs:
            operator_idx = f.pop("operator_idx")
            forklift = Forklift(
                warehouse_id=warehouse.id,
                operator_id=operators[operator_idx].id if operator_idx is not None else None,
                **f,
            )
            session.add(forklift)
            forklifts.append(forklift)
        await session.flush()

        # --- Cameras (one per forklift) ---
        for i, forklift in enumerate(forklifts):
            session.add(
                Camera(
                    external_id=f"CAM-0{i + 1}",
                    warehouse_id=warehouse.id,
                    forklift_id=forklift.id,
                    status=DeviceStatus.OFFLINE if forklift.status == ForkliftStatus.OFFLINE else DeviceStatus.ONLINE,
                )
            )

        # --- UWB anchors ---
        anchor_defs = [
            ("ANCH-01", 60, 50), ("ANCH-02", 520, 40),
            ("ANCH-03", 520, 280), ("ANCH-04", 520, 520),
            ("ANCH-05", 730, 60), ("ANCH-06", 900, 460),
        ]
        for external_id, x, y in anchor_defs:
            session.add(UWBAnchor(external_id=external_id, x=x, y=y, warehouse_id=warehouse.id, status=DeviceStatus.ONLINE))

        # --- Worker UWB tags (workers are tracked by tag only — no personal data) ---
        for i in range(1, 9):
            session.add(
                UWBTag(
                    external_id=f"TAG-40{i + 10}",
                    warehouse_id=warehouse.id,
                    kind="WORKER",
                    worker_label=f"W-{i:02d}",
                    status=DeviceStatus.ONLINE,
                )
            )

        # --- Maintenance records ---
        maint_defs = [
            (0, "Scheduled Service", "250-hr service, hydraulic fluid, filters", MaintenanceStatus.COMPLETE, d("2026-08-02")),
            (1, "Tire Replacement", "Front load wheels replaced", MaintenanceStatus.COMPLETE, d("2026-07-18")),
            (2, "Scheduled Service", "500-hr service", MaintenanceStatus.COMPLETE, d("2026-08-21")),
            (3, "Battery Inspection", "Cell voltage check, terminals cleaned", MaintenanceStatus.COMPLETE, d("2026-06-30")),
            (4, "Scheduled Service", "750-hr service — overdue, forklift taken offline", MaintenanceStatus.OVERDUE, None),
        ]
        for idx, type_, notes, status, performed_at in maint_defs:
            session.add(
                MaintenanceRecord(forklift_id=forklifts[idx].id, type=type_, notes=notes, status=status, performed_at=performed_at)
            )

        await session.commit()

        print("Seed complete.")
        print("Demo logins (all use password: ArkShieldDemo!2026):")
        for role, email, _name in role_defs:
            print(f"  {role.value.ljust(15)} {email}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(db_engine.dispose())
