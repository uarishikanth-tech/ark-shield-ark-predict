from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user
from app.models import Organization, Site, Warehouse
from app.types import JwtPayload

router = APIRouter(prefix="/api", tags=["hierarchy"])


class OrganizationOut(CamelModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime


class SiteOut(CamelModel):
    id: str
    name: str
    organization_id: str
    created_at: datetime
    updated_at: datetime


class WarehouseOut(CamelModel):
    id: str
    name: str
    site_id: str
    created_at: datetime
    updated_at: datetime


@router.get("/organizations/me", response_model=Optional[OrganizationOut])
async def get_my_organization(user: JwtPayload = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Organization).where(Organization.id == user.organization_id))
    org = result.scalars().first()
    return OrganizationOut.model_validate(org) if org else None


@router.get("/sites", response_model=list[SiteOut])
async def list_sites(user: JwtPayload = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Site).where(Site.organization_id == user.organization_id).order_by(Site.name.asc())
    )
    return [SiteOut.model_validate(s) for s in result.scalars().all()]


@router.get("/warehouses", response_model=list[WarehouseOut])
async def list_warehouses(
    site_id: Optional[str] = None,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Warehouse).join(Site).where(Site.organization_id == user.organization_id)
    if site_id:
        query = query.where(Warehouse.site_id == site_id)
    result = await db.execute(query.order_by(Warehouse.name.asc()))
    return [WarehouseOut.model_validate(w) for w in result.scalars().all()]
