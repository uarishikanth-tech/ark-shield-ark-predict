from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import (
    Forklift,
    MaintenanceRecord,
    ProximityEvent,
    SafetyEvent,
    SafetyEventType,
    Severity,
    Site,
    Warehouse,
)
from app.types import JwtPayload

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _resolve_range(range_: Optional[str], from_: Optional[datetime], to: Optional[datetime]):
    if from_ or to:
        return from_, to
    now = datetime.now(timezone.utc)
    if range_ == "7d":
        start = now - timedelta(days=7)
    elif range_ == "30d":
        start = now - timedelta(days=30)
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, now


@router.get("/overview")
async def analytics_overview(
    range: Optional[str] = None,
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = None,
    forklift_id: Optional[str] = Query(None, alias="forkliftId"),
    operator_id: Optional[str] = Query(None, alias="operatorId"),
    severity: Optional[Severity] = None,
    type: Optional[SafetyEventType] = None,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Powers the Analytics page's charts: utilization, safety incidents
    by severity/type, near misses (proximity WARNING/DANGER), and
    maintenance events, all respecting a Today/7D/30D/custom-range
    filter, plus optional forklift/operator/severity/event-type filters.
    """
    range_start, range_end = _resolve_range(range, from_, to)
    org_id = user.organization_id

    # Forklifts in scope (optionally narrowed to one).
    forklift_query = (
        select(Forklift)
        .join(Warehouse, Forklift.warehouse_id == Warehouse.id)
        .join(Site, Warehouse.site_id == Site.id)
        .where(Site.organization_id == org_id)
    )
    if forklift_id:
        forklift_query = forklift_query.where(Forklift.id == forklift_id)
    forklifts = (await db.execute(forklift_query)).scalars().all()
    forklift_ids = [f.id for f in forklifts]

    # Shared safety-event filters, applied to both group-by queries below.
    event_filters = [
        Site.organization_id == org_id,
        SafetyEvent.created_at >= range_start,
        SafetyEvent.created_at <= range_end,
    ]
    if forklift_id:
        event_filters.append(SafetyEvent.forklift_id == forklift_id)
    if operator_id:
        event_filters.append(SafetyEvent.operator_id == operator_id)
    if severity:
        event_filters.append(SafetyEvent.severity == severity)
    if type:
        event_filters.append(SafetyEvent.type == type)

    severity_counts = (
        await db.execute(
            select(SafetyEvent.severity, func.count(SafetyEvent.id))
            .join(Warehouse, SafetyEvent.warehouse_id == Warehouse.id)
            .join(Site, Warehouse.site_id == Site.id)
            .where(*event_filters)
            .group_by(SafetyEvent.severity)
        )
    ).all()

    type_counts = (
        await db.execute(
            select(SafetyEvent.type, func.count(SafetyEvent.id))
            .join(Warehouse, SafetyEvent.warehouse_id == Warehouse.id)
            .join(Site, Warehouse.site_id == Site.id)
            .where(*event_filters)
            .group_by(SafetyEvent.type)
        )
    ).all()

    proximity_by_state: dict[str, int] = {}
    maintenance_by_status: dict[str, int] = {}
    if forklift_ids:
        proximity_counts = (
            await db.execute(
                select(ProximityEvent.state, func.count(ProximityEvent.id))
                .where(
                    ProximityEvent.forklift_id.in_(forklift_ids),
                    ProximityEvent.recorded_at >= range_start,
                    ProximityEvent.recorded_at <= range_end,
                )
                .group_by(ProximityEvent.state)
            )
        ).all()
        proximity_by_state = {state: count for state, count in proximity_counts}

        maintenance_counts = (
            await db.execute(
                select(MaintenanceRecord.status, func.count(MaintenanceRecord.id))
                .where(MaintenanceRecord.forklift_id.in_(forklift_ids))
                .group_by(MaintenanceRecord.status)
            )
        ).all()
        maintenance_by_status = {status.value: count for status, count in maintenance_counts}

    total_operating_hours = sum(f.operating_hours for f in forklifts)
    status_counts: dict[str, int] = {}
    for f in forklifts:
        status_counts[f.status.value] = status_counts.get(f.status.value, 0) + 1
    total_forklifts = len(forklifts) or 1
    utilization_pct = round((status_counts.get("ACTIVE", 0) / total_forklifts) * 100)

    return {
        "range": range or "today",
        "fleet": {
            "total": len(forklifts),
            "byStatus": status_counts,
            "utilizationPct": utilization_pct,
            "totalOperatingHours": total_operating_hours,
        },
        "safetyEvents": {
            "bySeverity": {sev.value: count for sev, count in severity_counts},
            "byType": {t.value: count for t, count in type_counts},
        },
        "proximity": {
            "byState": proximity_by_state,
            "nearMisses": sum(c for s, c in proximity_by_state.items() if s in ("WARNING", "DANGER")),
        },
        "maintenance": {"byStatus": maintenance_by_status},
    }
