from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user, require_role
from app.errors import AppError
from app.models import Site, Warehouse, Zone, ZoneKind, ZoneSeverity
from app.types import ROLE_GROUPS, JwtPayload
from app.utils.tenancy import get_owned_warehouse

router = APIRouter(prefix="/api/zones", tags=["zones"])


def _org_scope(query, organization_id: str):
    return query.join(Warehouse, Zone.warehouse_id == Warehouse.id).join(Site, Warehouse.site_id == Site.id).where(
        Site.organization_id == organization_id
    )


class ZoneOut(CamelModel):
    id: str
    warehouse_id: str
    name: str
    kind: str
    severity: str
    speed_limit_kmh: int
    x: float
    y: float
    width: float
    height: float
    created_at: datetime
    updated_at: datetime


@router.get("/", response_model=list[ZoneOut])
async def list_zones(
    warehouse_id: Optional[str] = None,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = _org_scope(select(Zone), user.organization_id)
    if warehouse_id:
        query = query.where(Zone.warehouse_id == warehouse_id)
    rows = (await db.execute(query.order_by(Zone.name.asc()))).scalars().all()
    return [ZoneOut.model_validate(z) for z in rows]


async def _get_owned_zone(db: AsyncSession, zone_id: str, organization_id: str) -> Zone:
    zone = (await db.execute(_org_scope(select(Zone).where(Zone.id == zone_id), organization_id))).scalars().first()
    if not zone:
        raise AppError.not_found("Zone")
    return zone


@router.get("/{zone_id}", response_model=ZoneOut)
async def get_zone(zone_id: str, user: JwtPayload = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    zone = await _get_owned_zone(db, zone_id, user.organization_id)
    return ZoneOut.model_validate(zone)


class ZoneCreate(CamelModel):
    warehouse_id: str
    name: str
    kind: Optional[ZoneKind] = None
    severity: Optional[ZoneSeverity] = None
    speed_limit_kmh: Optional[int] = None
    x: float
    y: float
    width: float
    height: float


class ZoneUpdate(CamelModel):
    name: Optional[str] = None
    kind: Optional[ZoneKind] = None
    severity: Optional[ZoneSeverity] = None
    speed_limit_kmh: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


@router.post("/", response_model=ZoneOut, status_code=201)
async def create_zone(
    body: ZoneCreate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    """Zones directly affect safety behavior (speed limits, RED restricted
    areas), so only Safety/Fleet managers and Super Admin may create/edit them."""
    if not await get_owned_warehouse(db, body.warehouse_id, user.organization_id):
        raise AppError.not_found("Warehouse")

    zone = Zone(**body.model_dump(exclude_unset=True))
    db.add(zone)
    await db.commit()
    await db.refresh(zone)
    return ZoneOut.model_validate(zone)


@router.patch("/{zone_id}", response_model=ZoneOut)
async def update_zone(
    zone_id: str,
    body: ZoneUpdate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    zone = await _get_owned_zone(db, zone_id, user.organization_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(zone, field, value)
    await db.commit()
    await db.refresh(zone)
    return ZoneOut.model_validate(zone)


@router.delete("/{zone_id}", status_code=204)
async def delete_zone(
    zone_id: str,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    zone = await _get_owned_zone(db, zone_id, user.organization_id)
    await db.delete(zone)
    await db.commit()
