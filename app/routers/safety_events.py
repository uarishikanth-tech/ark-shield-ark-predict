from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user, require_role
from app.errors import AppError
from app.models import EventStatus, SafetyEvent, SafetyEventType, Severity, Site, Warehouse
from app.types import ROLE_GROUPS, JwtPayload
from app.utils.pagination import build_paginated_result, parse_pagination

router = APIRouter(prefix="/api/safety-events", tags=["safety-events"])


def _org_scope(query, organization_id: str):
    return query.join(Warehouse, SafetyEvent.warehouse_id == Warehouse.id).join(Site, Warehouse.site_id == Site.id).where(
        Site.organization_id == organization_id
    )


class ForkliftBrief(CamelModel):
    asset_number: str


class ZoneBrief(CamelModel):
    name: str


class AlertBrief(CamelModel):
    id: str
    severity: str
    status: str
    title: str


class SafetyEventOut(CamelModel):
    id: str
    warehouse_id: str
    type: str
    severity: str
    forklift_id: Optional[str]
    operator_id: Optional[str]
    zone_id: Optional[str]
    detected_object: Optional[str]
    distance_m: Optional[float]
    risk_score: Optional[int]
    source: str
    status: str
    is_simulated: bool
    created_at: datetime
    updated_at: datetime


class SafetyEventListItem(SafetyEventOut):
    forklift: Optional[ForkliftBrief] = None
    zone: Optional[ZoneBrief] = None


class SafetyEventDetail(SafetyEventListItem):
    alerts: list[AlertBrief] = []


@router.get("/", response_model=dict)
async def list_safety_events(
    warehouse_id: Optional[str] = None,
    severity: Optional[Severity] = None,
    type: Optional[SafetyEventType] = None,
    status: Optional[EventStatus] = None,
    forklift_id: Optional[str] = Query(None, alias="forkliftId"),
    operator_id: Optional[str] = Query(None, alias="operatorId"),
    zone_id: Optional[str] = Query(None, alias="zoneId"),
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = Query(None, alias="pageSize"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pagination = parse_pagination(page, page_size)
    query = _org_scope(
        select(SafetyEvent).options(selectinload(SafetyEvent.forklift), selectinload(SafetyEvent.zone)),
        user.organization_id,
    )
    count_query = _org_scope(select(func.count(SafetyEvent.id)), user.organization_id)

    filters = []
    if warehouse_id:
        filters.append(SafetyEvent.warehouse_id == warehouse_id)
    if severity:
        filters.append(SafetyEvent.severity == severity)
    if type:
        filters.append(SafetyEvent.type == type)
    if status:
        filters.append(SafetyEvent.status == status)
    if forklift_id:
        filters.append(SafetyEvent.forklift_id == forklift_id)
    if operator_id:
        filters.append(SafetyEvent.operator_id == operator_id)
    if zone_id:
        filters.append(SafetyEvent.zone_id == zone_id)
    if from_:
        filters.append(SafetyEvent.created_at >= from_)
    if to:
        filters.append(SafetyEvent.created_at <= to)

    for f in filters:
        query = query.where(f)
        count_query = count_query.where(f)

    total = (await db.execute(count_query)).scalar_one()
    rows = (
        (await db.execute(query.order_by(SafetyEvent.created_at.desc()).offset(pagination.offset).limit(pagination.limit)))
        .scalars()
        .all()
    )
    return build_paginated_result([SafetyEventListItem.model_validate(r) for r in rows], total, pagination)


@router.get("/{event_id}", response_model=SafetyEventDetail)
async def get_safety_event(event_id: str, user: JwtPayload = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    query = _org_scope(
        select(SafetyEvent).where(SafetyEvent.id == event_id),
        user.organization_id,
    ).options(selectinload(SafetyEvent.forklift), selectinload(SafetyEvent.zone), selectinload(SafetyEvent.alerts))
    event = (await db.execute(query)).scalars().first()
    if not event:
        raise AppError.not_found("Safety event")
    return SafetyEventDetail.model_validate(event)


class StatusUpdate(CamelModel):
    status: EventStatus


@router.patch("/{event_id}", response_model=SafetyEventOut)
async def update_safety_event_status(
    event_id: str,
    body: StatusUpdate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    query = _org_scope(select(SafetyEvent).where(SafetyEvent.id == event_id), user.organization_id)
    event = (await db.execute(query)).scalars().first()
    if not event:
        raise AppError.not_found("Safety event")

    event.status = body.status
    await db.commit()
    await db.refresh(event)
    return SafetyEventOut.model_validate(event)
