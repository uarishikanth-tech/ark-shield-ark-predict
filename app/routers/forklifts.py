from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user, require_role
from app.errors import AppError
from app.models import Forklift, ForkliftStatus, ForkliftType, Site, Warehouse
from app.types import ROLE_GROUPS, JwtPayload
from app.utils.pagination import build_paginated_result, parse_pagination
from app.utils.tenancy import get_owned_warehouse

router = APIRouter(prefix="/api/forklifts", tags=["forklifts"])


class OperatorBrief(CamelModel):
    id: str
    external_id: str
    name: str
    auth_status: str
    training_status: str


class CameraBrief(CamelModel):
    id: str
    external_id: str
    status: str


class UWBTagBrief(CamelModel):
    id: str
    external_id: str
    status: str


class MaintenanceRecordOut(CamelModel):
    id: str
    forklift_id: str
    type: str
    notes: Optional[str]
    status: str
    performed_at: Optional[datetime]
    created_at: datetime


class ForkliftOut(CamelModel):
    """Flat shape — no relations loaded. Used for create/update/delete
    responses, matching the Node version's `prisma.forklift.create()`
    (no `include`) so callers never trigger an unawaited lazy-load."""

    id: str
    warehouse_id: str
    asset_number: str
    manufacturer: str
    model: str
    type: str
    capacity_lbs: int
    status: str
    operator_id: Optional[str]
    operating_hours: float
    fuel_percent: Optional[float]
    battery_percent: Optional[float]
    last_service_at: Optional[datetime]
    next_service_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class ForkliftWithRelations(ForkliftOut):
    """Used for list/detail responses, where the query eager-loads
    these relations via selectinload() before this schema reads them."""

    operator: Optional[OperatorBrief] = None
    camera: Optional[CameraBrief] = None
    uwb_device: Optional[UWBTagBrief] = None


class ForkliftDetailOut(ForkliftWithRelations):
    maintenance_records: list[MaintenanceRecordOut] = []


def _org_scope(query, organization_id: str):
    return query.join(Warehouse, Forklift.warehouse_id == Warehouse.id).join(Site, Warehouse.site_id == Site.id).where(
        Site.organization_id == organization_id
    )


@router.get("/", response_model=dict)
async def list_forklifts(
    status: Optional[ForkliftStatus] = None,
    warehouse_id: Optional[str] = None,
    search: Optional[str] = None,
    page: Optional[int] = Query(None),
    page_size: Optional[int] = Query(None, alias="pageSize"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pagination = parse_pagination(page, page_size)

    query = _org_scope(
        select(Forklift).options(
            selectinload(Forklift.operator), selectinload(Forklift.camera), selectinload(Forklift.uwb_device)
        ),
        user.organization_id,
    )
    count_query = _org_scope(select(func.count(Forklift.id)), user.organization_id)

    if status:
        query = query.where(Forklift.status == status)
        count_query = count_query.where(Forklift.status == status)
    if warehouse_id:
        query = query.where(Forklift.warehouse_id == warehouse_id)
        count_query = count_query.where(Forklift.warehouse_id == warehouse_id)
    if search:
        like = f"%{search}%"
        clause = or_(Forklift.asset_number.ilike(like), Forklift.manufacturer.ilike(like), Forklift.model.ilike(like))
        query = query.where(clause)
        count_query = count_query.where(clause)

    total = (await db.execute(count_query)).scalar_one()
    rows = (
        (await db.execute(query.order_by(Forklift.asset_number.asc()).offset(pagination.offset).limit(pagination.limit)))
        .scalars()
        .all()
    )

    return build_paginated_result([ForkliftWithRelations.model_validate(r) for r in rows], total, pagination)


async def _get_owned_forklift(db: AsyncSession, forklift_id: str, organization_id: str, *options) -> Forklift:
    query = _org_scope(select(Forklift).where(Forklift.id == forklift_id), organization_id)
    for opt in options:
        query = query.options(opt)
    result = await db.execute(query)
    forklift = result.scalars().first()
    if not forklift:
        raise AppError.not_found("Forklift")
    return forklift


@router.get("/{forklift_id}", response_model=ForkliftDetailOut)
async def get_forklift(forklift_id: str, user: JwtPayload = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    forklift = await _get_owned_forklift(
        db,
        forklift_id,
        user.organization_id,
        selectinload(Forklift.operator),
        selectinload(Forklift.camera),
        selectinload(Forklift.uwb_device),
        selectinload(Forklift.maintenance_records),
    )
    # Match the Node API: only the 20 most recent maintenance records.
    forklift.maintenance_records = sorted(forklift.maintenance_records, key=lambda m: m.created_at, reverse=True)[:20]
    return ForkliftDetailOut.model_validate(forklift)


class SafetyEventBrief(CamelModel):
    id: str
    type: str
    severity: str
    created_at: datetime


@router.get("/{forklift_id}/events", response_model=dict)
async def get_forklift_events(
    forklift_id: str,
    page: Optional[int] = Query(None),
    page_size: Optional[int] = Query(None, alias="pageSize"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.models import SafetyEvent  # local import avoids a circular top-level import

    await _get_owned_forklift(db, forklift_id, user.organization_id)
    pagination = parse_pagination(page, page_size)

    base = select(SafetyEvent).where(SafetyEvent.forklift_id == forklift_id)
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        (await db.execute(base.order_by(SafetyEvent.created_at.desc()).offset(pagination.offset).limit(pagination.limit)))
        .scalars()
        .all()
    )
    return build_paginated_result([SafetyEventBrief.model_validate(r) for r in rows], total, pagination)


class ForkliftCreate(CamelModel):
    warehouse_id: str
    asset_number: str
    manufacturer: str
    model: str
    type: ForkliftType
    capacity_lbs: int
    status: Optional[ForkliftStatus] = None
    operator_id: Optional[str] = None
    fuel_percent: Optional[float] = None
    battery_percent: Optional[float] = None


class ForkliftUpdate(CamelModel):
    asset_number: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    type: Optional[ForkliftType] = None
    capacity_lbs: Optional[int] = None
    status: Optional[ForkliftStatus] = None
    operator_id: Optional[str] = None
    fuel_percent: Optional[float] = None
    battery_percent: Optional[float] = None


@router.post("/", response_model=ForkliftOut, status_code=201)
async def create_forklift(
    body: ForkliftCreate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    if not await get_owned_warehouse(db, body.warehouse_id, user.organization_id):
        raise AppError.not_found("Warehouse")

    # exclude_unset so an omitted optional field (e.g. status) lets the
    # column's own default apply, instead of the ORM inserting an
    # explicit None into a NOT NULL column.
    forklift = Forklift(**body.model_dump(exclude_unset=True))
    db.add(forklift)
    await db.commit()
    await db.refresh(forklift)
    return ForkliftOut.model_validate(forklift)


@router.patch("/{forklift_id}", response_model=ForkliftOut)
async def update_forklift(
    forklift_id: str,
    body: ForkliftUpdate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    forklift = await _get_owned_forklift(db, forklift_id, user.organization_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(forklift, field, value)
    await db.commit()
    await db.refresh(forklift)
    return ForkliftOut.model_validate(forklift)


@router.delete("/{forklift_id}", status_code=204)
async def delete_forklift(
    forklift_id: str,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    forklift = await _get_owned_forklift(db, forklift_id, user.organization_id)
    await db.delete(forklift)
    await db.commit()
