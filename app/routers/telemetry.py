from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user
from app.errors import AppError
from app.models import Forklift, Site, Telemetry, Warehouse
from app.types import JwtPayload
from app.utils.pagination import build_paginated_result, parse_pagination

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


class TelemetryOut(CamelModel):
    id: str
    forklift_id: str
    uwb_tag_id: Optional[str]
    x: float
    y: float
    speed_kmh: float
    is_simulated: bool
    recorded_at: datetime


@router.get("/", response_model=dict)
async def list_telemetry(
    forklift_id: Optional[str] = Query(None, alias="forkliftId"),
    page: Optional[int] = None,
    page_size: Optional[int] = Query(None, alias="pageSize"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    pagination = parse_pagination(page, page_size)
    base = (
        select(Telemetry)
        .join(Forklift, Telemetry.forklift_id == Forklift.id)
        .join(Warehouse, Forklift.warehouse_id == Warehouse.id)
        .join(Site, Warehouse.site_id == Site.id)
        .where(Site.organization_id == user.organization_id)
    )
    if forklift_id:
        base = base.where(Telemetry.forklift_id == forklift_id)

    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (
        (await db.execute(base.order_by(Telemetry.recorded_at.desc()).offset(pagination.offset).limit(pagination.limit)))
        .scalars()
        .all()
    )
    return build_paginated_result([TelemetryOut.model_validate(r) for r in rows], total, pagination)


class TelemetryIngest(CamelModel):
    forklift_id: str
    uwb_tag_id: Optional[str] = None
    x: float
    y: float
    speed_kmh: float
    # Real hardware integrations must explicitly opt in; the simulation
    # engine always sets True. Defaulting False here means a forgetful
    # integration doesn't silently mislabel real data as simulated.
    is_simulated: bool = False


@router.post("/", response_model=TelemetryOut, status_code=201)
async def ingest_telemetry(
    body: TelemetryIngest,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Hardware-abstraction ingestion point for a real fleet telemetry
    gateway. Records data for display/analytics only — never drives
    any vehicle control."""
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

    telemetry = Telemetry(**body.model_dump())
    db.add(telemetry)
    await db.commit()
    await db.refresh(telemetry)
    return TelemetryOut.model_validate(telemetry)
