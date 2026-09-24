from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Site, Warehouse


async def get_owned_warehouse(db: AsyncSession, warehouse_id: str, organization_id: str) -> Optional[Warehouse]:
    """Returns the warehouse if it belongs to `organization_id`, else None.

    Used before creating any child resource (forklift, operator, zone,
    camera, ...) so a caller can never attach a new row to a warehouse
    outside their own organization by guessing/reusing an ID.
    """
    result = await db.execute(
        select(Warehouse)
        .join(Site, Warehouse.site_id == Site.id)
        .where(Warehouse.id == warehouse_id, Site.organization_id == organization_id)
    )
    return result.scalars().first()
