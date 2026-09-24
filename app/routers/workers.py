from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user
from app.models import Forklift, ProximityEvent, Site, UWBTag, Warehouse
from app.types import JwtPayload

router = APIRouter(prefix="/api/workers", tags=["workers"])


class ForkliftBrief(CamelModel):
    id: str
    asset_number: str


class WorkerOut(CamelModel):
    tag_id: str
    worker_label: Optional[str]
    status: str
    nearest_forklift: Optional[ForkliftBrief]
    distance_m: Optional[float]
    risk_state: Optional[str]
    last_seen_at: Optional[datetime]


@router.get("/", response_model=list[WorkerOut])
async def list_workers(
    warehouse_id: Optional[str] = None,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Workers are tracked only via their UWB tag + a non-identifying
    label (no biometric identification by design), so this reads
    worker tags and folds in each worker's most recent proximity
    reading.
    """
    query = (
        select(UWBTag)
        .join(Warehouse, UWBTag.warehouse_id == Warehouse.id)
        .join(Site, Warehouse.site_id == Site.id)
        .where(UWBTag.kind == "WORKER", Site.organization_id == user.organization_id)
    )
    if warehouse_id:
        query = query.where(UWBTag.warehouse_id == warehouse_id)
    tags = (await db.execute(query.order_by(UWBTag.external_id.asc()))).scalars().all()

    workers = []
    for tag in tags:
        latest = None
        if tag.worker_label:
            result = await db.execute(
                select(ProximityEvent, Forklift)
                .join(Forklift, ProximityEvent.forklift_id == Forklift.id)
                .where(ProximityEvent.worker_label == tag.worker_label)
                .order_by(ProximityEvent.recorded_at.desc())
                .limit(1)
            )
            latest = result.first()

        proximity, forklift = latest if latest else (None, None)
        workers.append(
            WorkerOut(
                tag_id=tag.external_id,
                worker_label=tag.worker_label,
                status=tag.status.value,
                nearest_forklift=ForkliftBrief(id=forklift.id, asset_number=forklift.asset_number) if forklift else None,
                distance_m=proximity.distance_m if proximity else None,
                risk_state=proximity.state if proximity else None,
                last_seen_at=proximity.recorded_at if proximity else None,
            )
        )
    return workers
