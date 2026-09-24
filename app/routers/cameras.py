from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user, require_role
from app.errors import AppError
from app.models import Camera, DeviceStatus, Site, Warehouse
from app.types import ROLE_GROUPS, JwtPayload

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


def _org_scope(query, organization_id: str):
    return query.join(Warehouse, Camera.warehouse_id == Warehouse.id).join(Site, Warehouse.site_id == Site.id).where(
        Site.organization_id == organization_id
    )


class ForkliftBrief(CamelModel):
    id: str
    asset_number: str


class CameraOut(CamelModel):
    id: str
    warehouse_id: str
    external_id: str
    mount: str
    forklift_id: Optional[str]
    status: str


class CameraWithForklift(CameraOut):
    forklift: Optional[ForkliftBrief] = None


@router.get("/", response_model=list[CameraWithForklift])
async def list_cameras(
    warehouse_id: Optional[str] = None,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = _org_scope(select(Camera).options(selectinload(Camera.forklift)), user.organization_id)
    if warehouse_id:
        query = query.where(Camera.warehouse_id == warehouse_id)
    rows = (await db.execute(query.order_by(Camera.external_id.asc()))).scalars().all()
    return [CameraWithForklift.model_validate(r) for r in rows]


class CameraUpdate(CamelModel):
    status: Optional[DeviceStatus] = None
    forklift_id: Optional[str] = None


@router.patch("/{camera_id}", response_model=CameraOut)
async def update_camera(
    camera_id: str,
    body: CameraUpdate,
    user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"])),
    db: AsyncSession = Depends(get_db),
):
    camera = (
        await db.execute(_org_scope(select(Camera).where(Camera.id == camera_id), user.organization_id))
    ).scalars().first()
    if not camera:
        raise AppError.not_found("Camera")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)
    await db.commit()
    await db.refresh(camera)
    return CameraOut.model_validate(camera)
