"""
ARK RISK ENGINE (Python port)
------------------------------------------------------------------
Mirrors the conceptual model used by the browser demo and the Node
backend, so a frontend sees the same risk scores regardless of which
backend it talks to.

IMPORTANT — this is a DEMO CONCEPTUAL MODEL ONLY. It is NOT a
scientifically validated collision-prediction system, carries no
safety certification, and must never be presented to users as one.
compute_risk() is deliberately the *only* thing callers depend on, so
this implementation can later be swapped for a trained/validated
model without changing any calling code.
"""
from dataclasses import dataclass, field
from typing import Literal

from app.types import risk_state_for

DetectedObjectType = Literal["pedestrian", "forklift", "vehicle", "obstacle"]
ZoneSeverityInput = Literal["GREEN", "AMBER", "RED"]


@dataclass
class RiskInput:
    distance_m: float
    speed_kmh: float
    converging_trajectory: bool
    zone_severity: ZoneSeverityInput
    object_type: DetectedObjectType


@dataclass
class RiskResult:
    score: int
    state: str
    factors: dict = field(default_factory=dict)


def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def compute_risk(input: RiskInput) -> RiskResult:
    """Conceptual weighted sum, clamped to 0-100 (see module docstring)."""

    # Closer => higher risk. Saturates at 0 beyond 20m, maxes out inside 1m.
    distance_factor = _clamp((1 - _clamp(input.distance_m, 0, 20) / 20) * 100, 0, 100)

    # Faster forklift => higher risk. Saturates around 18 km/h (typical indoor limit).
    speed_factor = _clamp((input.speed_kmh / 18) * 100, 0, 100)

    # Converging trajectories are treated as a meaningful flat bump.
    trajectory_factor = 70 if input.converging_trajectory else 15

    # Zone severity contributes directly.
    zone_factor = 90 if input.zone_severity == "RED" else 45 if input.zone_severity == "AMBER" else 10

    # Pedestrians are weighted highest; forklift-forklift lowest.
    object_type_factor = (
        90
        if input.object_type == "pedestrian"
        else 55
        if input.object_type == "vehicle"
        else 30
        if input.object_type == "obstacle"
        else 40
    )

    # Weighted blend (weights sum to 1) reflecting the product spec's five factors.
    weighted = (
        distance_factor * 0.35
        + speed_factor * 0.2
        + trajectory_factor * 0.15
        + zone_factor * 0.15
        + object_type_factor * 0.15
    )

    score = round(_clamp(weighted, 0, 100))

    return RiskResult(
        score=score,
        state=risk_state_for(score),
        factors={
            "distanceFactor": distance_factor,
            "speedFactor": speed_factor,
            "trajectoryFactor": trajectory_factor,
            "zoneFactor": zone_factor,
            "objectTypeFactor": object_type_factor,
        },
    )
