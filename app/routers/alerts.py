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
from app.models import Alert, AlertStatus, Severity, Site, Warehouse
from app.realtime.socket import emit_to_org
from app.types import ROLE_GROUPS, JwtPayload
from app.utils.pagination import build_paginated_result, parse_pagination

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _org_scope(query, organization_id: str):
    return query.join(Warehouse, Alert.warehouse_id == Warehouse.id).join(Site, Warehouse.site_id == Site.id).where(
        Site.organization_id == organization_id
    )


class ForkliftBrief(CamelModel):
    asset_number: str


class SafetyEventBrief(CamelModel):
    id: str
    type: str
    severity: str


class AlertOut(CamelModel):
    id: str
    warehouse_id: str
    safety_event_id: Optional[str]
    severity: str
    title: str
    forklift_id: Optional[str]
    worker_label: Optional[str]
    distance_m: Optional[float]
    risk_score: Optional[int]
    zone_id: Optional[str]
    status: str
    acknowledged_by_user_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class AlertListItem(AlertOut):
    forklift: Optional[ForkliftBrief] = None
    safety_event: Optional[SafetyEventBrief] = None


@router.get("/", response_model=dict)
async def list_alerts(
    warehouse_id: Optional[str] = None,
    status: Optional[AlertStatus] = None,
    severity: Optional[Severity] = None,
    forklift_id: Optional[str] = Query(None, alias="forkliftId"),
    page: Optional[int] = None,
    page_size: Optional[int] = Query(None, alias="pageSize"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pagination = parse_pagination(page, page_size)
    query = _org_scope(
        select(Alert).options(selectinload(Alert.forklift), selectinload(Alert.safety_event)),
        user.organization_id,
    )
    count_query = _org_scope(select(func.count(Alert.id)), user.organization_id)

    filters = []
    if warehouse_id:
        filters.append(Alert.warehouse_id == warehouse_id)
    if status:
        filters.append(Alert.status == status)
    if severity:
        filters.append(Alert.severity == severity)
    if forklift_id:
        filters.append(Alert.forklift_id == forklift_id)
    for f in filters:
        query = query.where(f)
        count_query = count_query.where(f)

    total = (await db.execute(count_query)).scalar_one()
    rows = (
        (await db.execute(query.order_by(Alert.created_at.desc()).offset(pagination.offset).limit(pagination.limit)))
        .scalars()
        .all()
    )
    return build_paginated_result([AlertListItem.model_validate(r) for r in rows], total, pagination)


async def _get_owned_alert(db: AsyncSession, alert_id: str, organization_id: str) -> Alert:
    alert = (
        await db.execute(_org_scope(select(Alert).where(Alert.id == alert_id), organization_id))
    ).scalars().first()
    if not alert:
        raise AppError.not_found("Alert")
    return alert


@router.post("/{alert_id}/acknowledge", response_model=AlertOut)
async def acknowledge_alert(
    alert_id: str,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["CAN_ACK_ALERTS"])),
    db: AsyncSession = Depends(get_db),
):
    alert = await _get_owned_alert(db, alert_id, user.organization_id)
    alert.status = AlertStatus.ACKNOWLEDGED
    alert.acknowledged_by_user_id = user.user_id
    await db.commit()
    await db.refresh(alert)
    out = AlertOut.model_validate(alert)
    await emit_to_org("alert:updated", out.model_dump(mode="json", by_alias=True), user.organization_id)
    return out


@router.post("/{alert_id}/resolve", response_model=AlertOut)
async def resolve_alert(
    alert_id: str,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["CAN_ACK_ALERTS"])),
    db: AsyncSession = Depends(get_db),
):
    alert = await _get_owned_alert(db, alert_id, user.organization_id)
    alert.status = AlertStatus.RESOLVED
    await db.commit()
    await db.refresh(alert)
    out = AlertOut.model_validate(alert)
    await emit_to_org("alert:updated", out.model_dump(mode="json", by_alias=True), user.organization_id)
    return out
