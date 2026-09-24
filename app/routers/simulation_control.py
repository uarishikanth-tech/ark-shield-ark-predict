from fastapi import APIRouter, Depends

from app.camel import CamelModel
from app.deps import get_current_user, require_role
from app.services.simulation import simulation_engine
from app.types import ROLE_GROUPS, JwtPayload

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.get("/status")
async def get_status(_user: JwtPayload = Depends(get_current_user)):
    return simulation_engine.get_status()


@router.post("/start")
async def start(user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"]))):
    from app.realtime.socket import sio  # local import avoids a circular import at module load time

    await simulation_engine.start(sio)
    return simulation_engine.get_status()


@router.post("/pause")
async def pause(_user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"]))):
    await simulation_engine.pause()
    return simulation_engine.get_status()


@router.post("/reset")
async def reset(_user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"]))):
    await simulation_engine.reset()
    return simulation_engine.get_status()


@router.post("/increase-traffic")
async def increase_traffic(_user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"]))):
    simulation_engine.increase_traffic()
    return simulation_engine.get_status()


class ThresholdsUpdate(CamelModel):
    safe_m: float
    warning_m: float


@router.post("/thresholds")
async def set_thresholds(
    body: ThresholdsUpdate, _user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"]))
):
    simulation_engine.set_thresholds(body.safe_m, body.warning_m)
    return simulation_engine.get_status()


@router.post("/trigger-warning")
async def trigger_warning(_user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"]))):
    """DEMO ONLY — forces a WARNING-level proximity alert."""
    await simulation_engine.trigger_warning()
    return {"ok": True, "note": "DEMO ONLY — simulated warning event triggered"}


@router.post("/trigger-critical")
async def trigger_critical(_user: JwtPayload = Depends(require_role(ROLE_GROUPS["SAFETY_OR_FLEET"]))):
    """DEMO ONLY — forces a CRITICAL safety event. Never actuates real machinery."""
    await simulation_engine.trigger_critical()
    return {"ok": True, "note": "DEMO ONLY — simulated critical event triggered"}
