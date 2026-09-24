"""Shared types and constants used across the server (port of src/types.ts)."""
from pydantic import BaseModel

from app.models import Role

# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------
class JwtPayload(BaseModel):
    user_id: str
    organization_id: str
    role: Role
    email: str


# ------------------------------------------------------------------
# Role -> permission groups (mirrors spec §23)
# ------------------------------------------------------------------
ROLE_GROUPS = {
    "ALL": [Role.SUPER_ADMIN],
    "SAFETY": [Role.SUPER_ADMIN, Role.SAFETY_MANAGER],
    "FLEET": [Role.SUPER_ADMIN, Role.FLEET_MANAGER],
    "SAFETY_OR_FLEET": [Role.SUPER_ADMIN, Role.SAFETY_MANAGER, Role.FLEET_MANAGER],
    "OPERATIONAL": [Role.SUPER_ADMIN, Role.SAFETY_MANAGER, Role.FLEET_MANAGER, Role.SUPERVISOR],
    "CAN_ACK_ALERTS": [Role.SUPER_ADMIN, Role.SAFETY_MANAGER, Role.FLEET_MANAGER, Role.SUPERVISOR],
    "ANY": [
        Role.SUPER_ADMIN,
        Role.SAFETY_MANAGER,
        Role.FLEET_MANAGER,
        Role.SUPERVISOR,
        Role.OPERATOR,
        Role.VIEWER,
    ],
}


# ------------------------------------------------------------------
# Risk engine constants (spec §8) — conceptual demo model, NOT a
# validated safety system. See app/services/risk_engine.py.
# ------------------------------------------------------------------
RISK_LOW_MAX = 29
RISK_CAUTION_MAX = 59
RISK_HIGH_MAX = 79
# 80-100 => CRITICAL


def risk_state_for(score: float) -> str:
    if score <= RISK_LOW_MAX:
        return "LOW"
    if score <= RISK_CAUTION_MAX:
        return "CAUTION"
    if score <= RISK_HIGH_MAX:
        return "HIGH"
    return "CRITICAL"


# ------------------------------------------------------------------
# UWB proximity thresholds (spec §10) — configurable, not universal
# safe distances. Defaults only.
# ------------------------------------------------------------------
DEFAULT_SAFE_M = 15.0
DEFAULT_WARNING_M = 8.0


def proximity_state_for(distance_m: float, safe_m: float = DEFAULT_SAFE_M, warning_m: float = DEFAULT_WARNING_M) -> str:
    if distance_m >= safe_m:
        return "SAFE"
    if distance_m >= warning_m:
        return "WARNING"
    return "DANGER"
