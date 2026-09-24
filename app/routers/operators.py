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
from app.models import Operator, OperatorAuthStatus, Site, TrainingStatus, Warehouse
from app.types import ROLE_GROUPS, JwtPayload
from app.utils.pagination import build_paginated_result, parse_pagination
from app.utils.tenancy import get_owned_warehouse

router = APIRouter(prefix="/api/operators", tags=["operators"])


def _org_scope(query, organization_id: str):
    return query.join(Warehouse, Operator.warehouse_id == Warehouse.id).join(Site, Warehouse.site_id == Site.id).where(
        Site.organization_id == organization_id
    )


class ForkliftBrief(CamelModel):
    id: str
    asset_number: str
    status: str


class OperatorOut(CamelModel):
    id: str
    warehouse_id: str
    external_id: str
    name: str
    auth_status: str
    training_status: str
    operating_hours: float
    created_at: datetime
    updated_at: datetime


class OperatorWithForklifts(OperatorOut):
    forklifts: list[ForkliftBrief] = []


@router.get("/", response_model=dict)
async def list_operators(
    warehouse_id: Optional[str] = None,
    auth_status: Optional[OperatorAuthStatus] = Query(None, alias="authStatus"),
    page: Optional[int] = None,
    page_size: Optional[int] = Query(None, alias="pageSize"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pagination = parse_pagination(page, page_size)
    query = _org_scope(select(Operator).options(selectinload(Operator.forklifts)), user.organization_id)
    count_query = _org_scope(select(func.count(Operator.id)), user.organization_id)

    if warehouse_id:
        query = query.where(Operator.warehouse_id == warehouse_id)
        count_query = count_query.where(Operator.warehouse_id == warehouse_id)
    if auth_status:
        query = query.where(Operator.auth_status == auth_status)
        count_query = count_query.where(Operator.auth_status == auth_status)

    total = (await db.execute(count_query)).scalar_one()
    rows = (
        (await db.execute(query.order_by(Operator.external_id.asc()).offset(pagination.offset).limit(pagination.limit)))
        .scalars()
        .all()
    )
    return build_paginated_result([OperatorWithForklifts.model_validate(r) for r in rows], total, pagination)


async def _get_owned_operator(db: AsyncSession, operator_id: str, organization_id: str, *options) -> Operator:
    query = _org_scope(select(Operator).where(Operator.id == operator_id), organization_id)
    for opt in options:
        query = query.options(opt)
    operator = (await db.execute(query)).scalars().first()
    if not operator:
        raise AppError.not_found("Operator")
    return operator


@router.get("/{operator_id}", response_model=OperatorWithForklifts)
async def get_operator(operator_id: str, user: JwtPayload = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    operator = await _get_owned_operator(db, operator_id, user.organization_id, selectinload(Operator.forklifts))
    return OperatorWithForklifts.model_validate(operator)


class OperatorCreate(CamelModel):
    warehouse_id: str
    external_id: str
    name: str
    auth_status: Optional[OperatorAuthStatus] = None
    training_status: Optional[TrainingStatus] = None


class OperatorUpdate(CamelModel):
    external_id: Optional[str] = None
    name: Optional[str] = None
    auth_status: Optional[OperatorAuthStatus] = None
    training_status: Optional[TrainingStatus] = None


@router.post("/", response_model=OperatorOut, status_code=201)
async def create_operator(
    body: OperatorCreate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    if not await get_owned_warehouse(db, body.warehouse_id, user.organization_id):
        raise AppError.not_found("Warehouse")

    operator = Operator(**body.model_dump(exclude_unset=True))
    db.add(operator)
    await db.commit()
    await db.refresh(operator)
    return OperatorOut.model_validate(operator)


@router.patch("/{operator_id}", response_model=OperatorOut)
async def update_operator(
    operator_id: str,
    body: OperatorUpdate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    operator = await _get_owned_operator(db, operator_id, user.organization_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(operator, field, value)
    await db.commit()
    await db.refresh(operator)
    return OperatorOut.model_validate(operator)


@router.delete("/{operator_id}", status_code=204)
async def delete_operator(
    operator_id: str,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    operator = await _get_owned_operator(db, operator_id, user.organization_id)
    await db.delete(operator)
    await db.commit()
