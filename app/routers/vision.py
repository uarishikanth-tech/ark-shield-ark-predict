from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user
from app.errors import AppError
from app.models import Camera, Detection, Forklift, Site, Warehouse
from app.services.risk_engine import RiskInput, compute_risk
from app.types import JwtPayload

router = APIRouter(prefix="/api/vision", tags=["vision"])

ObjectType = Literal["pedestrian", "forklift", "vehicle", "obstacle"]
ZoneSeverityInput = Literal["GREEN", "AMBER", "RED"]


class DetectionOut(CamelModel):
    id: str
    camera_id: str
    forklift_id: str
    object_type: str
    confidence: float
    distance_m: Optional[float]
    direction: Optional[str]
    risk_score: Optional[int]
    is_simulated: bool
    recorded_at: datetime


async def _get_owned_forklift(db: AsyncSession, forklift_id: str, organization_id: str) -> Forklift:
    forklift = (
        await db.execute(
            select(Forklift)
            .join(Warehouse, Forklift.warehouse_id == Warehouse.id)
            .join(Site, Warehouse.site_id == Site.id)
            .where(Forklift.id == forklift_id, Site.organization_id == organization_id)
        )
    ).scalars().first()
    if not forklift:
        raise AppError.not_found("Forklift")
    return forklift


@router.get("/detections", response_model=list[DetectionOut])
async def list_detections(
    forklift_id: str = Query(..., alias="forkliftId"),
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Recent detections for the ARK Vision camera dashboard's bounding-box overlay."""
    await _get_owned_forklift(db, forklift_id, user.organization_id)
    rows = (
        await db.execute(
            select(Detection).where(Detection.forklift_id == forklift_id).order_by(Detection.recorded_at.desc()).limit(50)
        )
    ).scalars().all()
    return [DetectionOut.model_validate(r) for r in rows]


class DetectionIngest(CamelModel):
    camera_id: str
    forklift_id: str
    object_type: ObjectType
    confidence: float
    distance_m: Optional[float] = None
    direction: Optional[str] = None
    speed_kmh: float = 0
    converging_trajectory: bool = False
    zone_severity: ZoneSeverityInput = "GREEN"
    is_simulated: bool = False


class RiskOut(CamelModel):
    score: int
    state: str
    factors: dict


class DetectionIngestResponse(CamelModel):
    detection: DetectionOut
    risk: Optional[RiskOut]


@router.post("/detections", response_model=DetectionIngestResponse, status_code=201)
async def ingest_detection(
    body: DetectionIngest,
    user: JwtPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    The modular computer-vision service interface: any future
    YOLO/OpenCV/TensorRT/edge-AI inference process posts its
    detections here in a fixed shape, and this endpoint computes the
    ARK Risk Score and persists the detection. Nothing about this
    route is tied to a specific model implementation.
    """
    forklift = await _get_owned_forklift(db, body.forklift_id, user.organization_id)
    camera = (await db.execute(select(Camera).where(Camera.id == body.camera_id))).scalars().first()
    if not camera:
        raise AppError.not_found("Camera")

    risk = None
    if body.distance_m is not None:
        risk = compute_risk(
            RiskInput(
                distance_m=body.distance_m,
                speed_kmh=body.speed_kmh,
                converging_trajectory=body.converging_trajectory,
                zone_severity=body.zone_severity,
                object_type=body.object_type,
            )
        )

    detection = Detection(
        camera_id=body.camera_id,
        forklift_id=forklift.id,
        object_type=body.object_type,
        confidence=body.confidence,
        distance_m=body.distance_m,
        direction=body.direction,
        risk_score=risk.score if risk else None,
        is_simulated=body.is_simulated,
    )
    db.add(detection)
    await db.commit()
    await db.refresh(detection)

    return DetectionIngestResponse(
        detection=DetectionOut.model_validate(detection),
        risk=RiskOut(score=risk.score, state=risk.state, factors=risk.factors) if risk else None,
    )
