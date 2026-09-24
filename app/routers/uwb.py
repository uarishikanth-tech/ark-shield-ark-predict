from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user
from app.errors import AppError
from app.models import Forklift, ProximityEvent, Site, UWBAnchor, UWBTag, Warehouse
from app.types import DEFAULT_SAFE_M, DEFAULT_WARNING_M, JwtPayload, proximity_state_for

router = APIRouter(prefix="/api/uwb", tags=["uwb"])


class AnchorOut(CamelModel):
    id: str
    warehouse_id: str
    external_id: str
    x: float
    y: float
    status: str


class TagOut(CamelModel):
    id: str
    warehouse_id: str
    external_id: str
    kind: str
    worker_label: Optional[str]
    forklift_id: Optional[str]
    status: str


class ProximityEventOut(CamelModel):
    id: str
    forklift_id: str
    zone_id: Optional[str]
    worker_label: Optional[str]
    distance_m: float
    state: str
    is_simulated: bool
    recorded_at: datetime


@router.get("/anchors", response_model=list[AnchorOut])
async def list_anchors(
    warehouse_id: Optional[str] = None,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(UWBAnchor)
        .join(Warehouse, UWBAnchor.warehouse_id == Warehouse.id)
        .join(Site, Warehouse.site_id == Site.id)
        .where(Site.organization_id == user.organization_id)
    )
    if warehouse_id:
        query = query.where(UWBAnchor.warehouse_id == warehouse_id)
    rows = (await db.execute(query.order_by(UWBAnchor.external_id.asc()))).scalars().all()
    return [AnchorOut.model_validate(r) for r in rows]


@router.get("/tags", response_model=list[TagOut])
async def list_tags(
    warehouse_id: Optional[str] = None,
    kind: Optional[str] = None,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(UWBTag)
        .join(Warehouse, UWBTag.warehouse_id == Warehouse.id)
        .join(Site, Warehouse.site_id == Site.id)
        .where(Site.organization_id == user.organization_id)
    )
    if warehouse_id:
        query = query.where(UWBTag.warehouse_id == warehouse_id)
    if kind:
        query = query.where(UWBTag.kind == kind)
    rows = (await db.execute(query.order_by(UWBTag.external_id.asc()))).scalars().all()
    return [TagOut.model_validate(r) for r in rows]


@router.get("/events", response_model=list[ProximityEventOut])
async def list_proximity_events(
    forklift_id: Optional[str] = Query(None, alias="forkliftId"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(ProximityEvent)
        .join(Forklift, ProximityEvent.forklift_id == Forklift.id)
        .join(Warehouse, Forklift.warehouse_id == Warehouse.id)
        .join(Site, Warehouse.site_id == Site.id)
        .where(Site.organization_id == user.organization_id)
    )
    if forklift_id:
        query = query.where(ProximityEvent.forklift_id == forklift_id)
    rows = (await db.execute(query.order_by(ProximityEvent.recorded_at.desc()).limit(100))).scalars().all()
    return [ProximityEventOut.model_validate(r) for r in rows]


class ProximityEventIngest(CamelModel):
    forklift_id: str
    zone_id: Optional[str] = None
    worker_label: str
    distance_m: float
    safe_m: Optional[float] = None
    warning_m: Optional[float] = None
    is_simulated: bool = False


@router.post("/events", response_model=ProximityEventOut, status_code=201)
async def ingest_proximity_event(
    body: ProximityEventIngest,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    The UWB hardware-abstraction ingestion point: a real position
    engine derives worker-forklift distance from anchors + tags and
    posts it here. The proximity state (SAFE/WARNING/DANGER) is
    computed server-side so the definition of "too close" lives in
    one place.
    """
    forklift = (
        await db.execute(
            select(Forklift)
            .join(Warehouse, Forklift.warehouse_id == Warehouse.id)
            .join(Site, Warehouse.site_id == Site.id)
            .where(Forklift.id == body.forklift_id, Site.organization_id == user.organization_id)
        )
    ).scalars().first()
    if not forklift:
        raise AppError.not_found("Forklift")

    safe_m = body.safe_m if body.safe_m and body.warning_m else DEFAULT_SAFE_M
    warning_m = body.warning_m if body.safe_m and body.warning_m else DEFAULT_WARNING_M
    state = proximity_state_for(body.distance_m, safe_m, warning_m)

    event = ProximityEvent(
        forklift_id=body.forklift_id,
        zone_id=body.zone_id,
        worker_label=body.worker_label,
        distance_m=body.distance_m,
        state=state,
        is_simulated=body.is_simulated,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return ProximityEventOut.model_validate(event)
