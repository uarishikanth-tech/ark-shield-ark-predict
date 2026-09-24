"""
ARK SHIELD — SQLAlchemy models
------------------------------------------------------------------
Direct port of prisma/schema.prisma (see the Node version of this
project) to SQLAlchemy 2.0. Multi-tenant hierarchy:

    Organization -> Site -> Warehouse -> Zone / Forklift / Operator /
                                          Camera / UWB Anchor / UWB Tag

Two intentional differences from the Prisma schema, both cosmetic or
tightening rather than behavioral:
  - Primary keys are UUID hex strings (uuid4().hex) instead of cuid();
    both are non-sequential unique string IDs, so nothing downstream
    depends on the exact ID format.
  - Column names are snake_case (Python/Postgres convention) rather
    than camelCase. The REST API still speaks camelCase JSON — see
    app/camel.py — so a frontend built against the Node version's API
    can talk to this one unmodified.
  - A few columns that were plain (unenforced) foreign keys in the
    Node schema — SafetyEvent.operator_id, Alert.zone_id,
    Alert.acknowledged_by_user_id — get a real ForeignKey constraint
    here for referential integrity, without an ORM relationship
    attribute (nothing reads them as a relation, same as before).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def gen_id() -> str:
    return uuid.uuid4().hex


class IdMixin:
    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_id)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ------------------------------------------------------------------
# Enums (values match the Prisma enums exactly)
# ------------------------------------------------------------------
class Role(str, PyEnum):
    SUPER_ADMIN = "SUPER_ADMIN"
    SAFETY_MANAGER = "SAFETY_MANAGER"
    FLEET_MANAGER = "FLEET_MANAGER"
    SUPERVISOR = "SUPERVISOR"
    OPERATOR = "OPERATOR"
    VIEWER = "VIEWER"


class ForkliftStatus(str, PyEnum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    CHARGING = "CHARGING"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"


class ForkliftType(str, PyEnum):
    INTERNAL_COMBUSTION = "INTERNAL_COMBUSTION"
    ELECTRIC = "ELECTRIC"
    LPG = "LPG"
    OTHER = "OTHER"


class OperatorAuthStatus(str, PyEnum):
    AUTHORIZED = "AUTHORIZED"
    SUSPENDED = "SUSPENDED"
    REVOKED = "REVOKED"


class TrainingStatus(str, PyEnum):
    CURRENT = "CURRENT"
    DUE_FOR_RENEWAL = "DUE_FOR_RENEWAL"
    EXPIRED = "EXPIRED"


class DeviceStatus(str, PyEnum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class ZoneSeverity(str, PyEnum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


class ZoneKind(str, PyEnum):
    LOADING_AREA = "LOADING_AREA"
    STORAGE_RACK_AREA = "STORAGE_RACK_AREA"
    CHARGING_AREA = "CHARGING_AREA"
    RESTRICTED_ZONE = "RESTRICTED_ZONE"
    GENERAL = "GENERAL"


class SafetyEventType(str, PyEnum):
    PEDESTRIAN_PROXIMITY = "PEDESTRIAN_PROXIMITY"
    COLLISION_RISK = "COLLISION_RISK"
    RESTRICTED_ZONE = "RESTRICTED_ZONE"
    SPEED_VIOLATION = "SPEED_VIOLATION"
    IMPACT = "IMPACT"
    UNAUTHORIZED_OPERATION = "UNAUTHORIZED_OPERATION"
    CAMERA_ALERT = "CAMERA_ALERT"
    UWB_ALERT = "UWB_ALERT"


class Severity(str, PyEnum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventStatus(str, PyEnum):
    OPEN = "OPEN"
    REVIEWED = "REVIEWED"
    CLOSED = "CLOSED"


class AlertStatus(str, PyEnum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class EventSource(str, PyEnum):
    ARK_VISION = "ARK_VISION"
    ARK_PROXIMITY = "ARK_PROXIMITY"
    ARK_FLEET = "ARK_FLEET"
    SYSTEM = "SYSTEM"


class MaintenanceStatus(str, PyEnum):
    SCHEDULED = "SCHEDULED"
    COMPLETE = "COMPLETE"
    OVERDUE = "OVERDUE"


# ------------------------------------------------------------------
# Tenancy hierarchy
# ------------------------------------------------------------------
class Organization(Base, IdMixin, TimestampMixin):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String, index=True)

    sites: Mapped[list["Site"]] = relationship(back_populates="organization", cascade="all, delete-orphan")
    users: Mapped[list["User"]] = relationship(back_populates="organization", cascade="all, delete-orphan")


class Site(Base, IdMixin, TimestampMixin):
    __tablename__ = "sites"

    name: Mapped[str] = mapped_column(String)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)

    organization: Mapped["Organization"] = relationship(back_populates="sites")
    warehouses: Mapped[list["Warehouse"]] = relationship(back_populates="site", cascade="all, delete-orphan")


class Warehouse(Base, IdMixin, TimestampMixin):
    __tablename__ = "warehouses"

    name: Mapped[str] = mapped_column(String)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"), index=True)

    site: Mapped["Site"] = relationship(back_populates="warehouses")
    zones: Mapped[list["Zone"]] = relationship(back_populates="warehouse", cascade="all, delete-orphan")
    forklifts: Mapped[list["Forklift"]] = relationship(back_populates="warehouse", cascade="all, delete-orphan")
    operators: Mapped[list["Operator"]] = relationship(back_populates="warehouse", cascade="all, delete-orphan")
    cameras: Mapped[list["Camera"]] = relationship(back_populates="warehouse", cascade="all, delete-orphan")
    uwb_anchors: Mapped[list["UWBAnchor"]] = relationship(back_populates="warehouse", cascade="all, delete-orphan")
    uwb_tags: Mapped[list["UWBTag"]] = relationship(back_populates="warehouse", cascade="all, delete-orphan")
    safety_events: Mapped[list["SafetyEvent"]] = relationship(back_populates="warehouse")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="warehouse")


# ------------------------------------------------------------------
# Users & RBAC
# ------------------------------------------------------------------
class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    role: Mapped[Role] = mapped_column(SqlEnum(Role, name="role"), default=Role.VIEWER)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    organization: Mapped["Organization"] = relationship(back_populates="users")


# ------------------------------------------------------------------
# Zones
# ------------------------------------------------------------------
class Zone(Base, IdMixin, TimestampMixin):
    __tablename__ = "zones"

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String)
    kind: Mapped[ZoneKind] = mapped_column(SqlEnum(ZoneKind, name="zone_kind"), default=ZoneKind.GENERAL)
    severity: Mapped[ZoneSeverity] = mapped_column(
        SqlEnum(ZoneSeverity, name="zone_severity"), default=ZoneSeverity.GREEN
    )
    speed_limit_kmh: Mapped[int] = mapped_column(Integer, default=10)
    # Simple rectangular geometry for the MVP map; could move to polygon/GeoJSON later.
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)

    warehouse: Mapped["Warehouse"] = relationship(back_populates="zones")
    safety_events: Mapped[list["SafetyEvent"]] = relationship(back_populates="zone")
    proximity_events: Mapped[list["ProximityEvent"]] = relationship(back_populates="zone")


# ------------------------------------------------------------------
# Fleet
# ------------------------------------------------------------------
class Forklift(Base, IdMixin, TimestampMixin):
    __tablename__ = "forklifts"
    __table_args__ = (
        UniqueConstraint("warehouse_id", "asset_number", name="uq_forklift_warehouse_asset"),
        Index("ix_forklift_warehouse_status", "warehouse_id", "status"),
    )

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"))
    asset_number: Mapped[str] = mapped_column(String)
    manufacturer: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    type: Mapped[ForkliftType] = mapped_column(SqlEnum(ForkliftType, name="forklift_type"))
    capacity_lbs: Mapped[int] = mapped_column(Integer)
    status: Mapped[ForkliftStatus] = mapped_column(
        SqlEnum(ForkliftStatus, name="forklift_status"), default=ForkliftStatus.OFFLINE
    )
    operator_id: Mapped[Optional[str]] = mapped_column(ForeignKey("operators.id"), nullable=True)
    operating_hours: Mapped[float] = mapped_column(Float, default=0)
    fuel_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    battery_percent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_service_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    next_service_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    warehouse: Mapped["Warehouse"] = relationship(back_populates="forklifts")
    operator: Mapped[Optional["Operator"]] = relationship(back_populates="forklifts")
    camera: Mapped[Optional["Camera"]] = relationship(back_populates="forklift", uselist=False)
    uwb_device: Mapped[Optional["UWBTag"]] = relationship(
        back_populates="forklift", uselist=False, foreign_keys="UWBTag.forklift_id"
    )
    telemetry: Mapped[list["Telemetry"]] = relationship(back_populates="forklift")
    detections: Mapped[list["Detection"]] = relationship(back_populates="forklift")
    safety_events: Mapped[list["SafetyEvent"]] = relationship(back_populates="forklift")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="forklift")
    maintenance_records: Mapped[list["MaintenanceRecord"]] = relationship(back_populates="forklift")
    proximity_events: Mapped[list["ProximityEvent"]] = relationship(back_populates="forklift")


class Operator(Base, IdMixin, TimestampMixin):
    __tablename__ = "operators"
    __table_args__ = (UniqueConstraint("warehouse_id", "external_id", name="uq_operator_warehouse_external"),)

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str] = mapped_column(String)  # e.g. OP-001, business-facing identifier
    name: Mapped[str] = mapped_column(String)
    auth_status: Mapped[OperatorAuthStatus] = mapped_column(
        SqlEnum(OperatorAuthStatus, name="operator_auth_status"), default=OperatorAuthStatus.AUTHORIZED
    )
    training_status: Mapped[TrainingStatus] = mapped_column(
        SqlEnum(TrainingStatus, name="training_status"), default=TrainingStatus.CURRENT
    )
    operating_hours: Mapped[float] = mapped_column(Float, default=0)

    warehouse: Mapped["Warehouse"] = relationship(back_populates="operators")
    forklifts: Mapped[list["Forklift"]] = relationship(back_populates="operator", foreign_keys="Forklift.operator_id")


# ------------------------------------------------------------------
# Hardware abstraction: cameras & UWB
# ------------------------------------------------------------------
class Camera(Base, IdMixin, TimestampMixin):
    __tablename__ = "cameras"
    __table_args__ = (UniqueConstraint("warehouse_id", "external_id", name="uq_camera_warehouse_external"),)

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str] = mapped_column(String)  # e.g. CAM-01
    mount: Mapped[str] = mapped_column(String, default="Forklift-mounted")
    forklift_id: Mapped[Optional[str]] = mapped_column(ForeignKey("forklifts.id"), unique=True, nullable=True)
    status: Mapped[DeviceStatus] = mapped_column(SqlEnum(DeviceStatus, name="device_status"), default=DeviceStatus.OFFLINE)

    warehouse: Mapped["Warehouse"] = relationship(back_populates="cameras")
    forklift: Mapped[Optional["Forklift"]] = relationship(back_populates="camera", foreign_keys=[forklift_id])
    detections: Mapped[list["Detection"]] = relationship(back_populates="camera")


class UWBAnchor(Base, IdMixin, TimestampMixin):
    __tablename__ = "uwb_anchors"
    __table_args__ = (UniqueConstraint("warehouse_id", "external_id", name="uq_uwb_anchor_warehouse_external"),)

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str] = mapped_column(String)  # e.g. ANCH-01
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    status: Mapped[DeviceStatus] = mapped_column(SqlEnum(DeviceStatus, name="device_status"), default=DeviceStatus.OFFLINE)

    warehouse: Mapped["Warehouse"] = relationship(back_populates="uwb_anchors")


class UWBTag(Base, IdMixin, TimestampMixin):
    __tablename__ = "uwb_tags"
    __table_args__ = (UniqueConstraint("warehouse_id", "external_id", name="uq_uwb_tag_warehouse_external"),)

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"), index=True)
    external_id: Mapped[str] = mapped_column(String)  # e.g. TAG-4001
    kind: Mapped[str] = mapped_column(String, default="WORKER")  # WORKER | FORKLIFT
    worker_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # e.g. W-01 — no biometric data
    forklift_id: Mapped[Optional[str]] = mapped_column(ForeignKey("forklifts.id"), unique=True, nullable=True)
    status: Mapped[DeviceStatus] = mapped_column(SqlEnum(DeviceStatus, name="device_status"), default=DeviceStatus.OFFLINE)

    warehouse: Mapped["Warehouse"] = relationship(back_populates="uwb_tags")
    forklift: Mapped[Optional["Forklift"]] = relationship(
        back_populates="uwb_device", foreign_keys=[forklift_id]
    )
    telemetry: Mapped[list["Telemetry"]] = relationship(back_populates="uwb_tag")


# ------------------------------------------------------------------
# Telemetry / detections / proximity (high-volume, append-only)
# ------------------------------------------------------------------
class Telemetry(Base, IdMixin):
    __tablename__ = "telemetry"
    __table_args__ = (Index("ix_telemetry_forklift_recorded", "forklift_id", "recorded_at"),)

    forklift_id: Mapped[str] = mapped_column(ForeignKey("forklifts.id", ondelete="CASCADE"))
    uwb_tag_id: Mapped[Optional[str]] = mapped_column(ForeignKey("uwb_tags.id"), nullable=True)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    speed_kmh: Mapped[float] = mapped_column(Float)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    forklift: Mapped["Forklift"] = relationship(back_populates="telemetry")
    uwb_tag: Mapped[Optional["UWBTag"]] = relationship(back_populates="telemetry")


class Detection(Base, IdMixin):
    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detection_forklift_recorded", "forklift_id", "recorded_at"),
        Index("ix_detection_camera_recorded", "camera_id", "recorded_at"),
    )

    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    forklift_id: Mapped[str] = mapped_column(ForeignKey("forklifts.id", ondelete="CASCADE"))
    object_type: Mapped[str] = mapped_column(String)  # pedestrian | forklift | vehicle | obstacle
    confidence: Mapped[float] = mapped_column(Float)
    distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    direction: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    camera: Mapped["Camera"] = relationship(back_populates="detections")
    forklift: Mapped["Forklift"] = relationship(back_populates="detections")


class ProximityEvent(Base, IdMixin):
    __tablename__ = "proximity_events"
    __table_args__ = (Index("ix_proximity_forklift_recorded", "forklift_id", "recorded_at"),)

    forklift_id: Mapped[str] = mapped_column(ForeignKey("forklifts.id", ondelete="CASCADE"))
    zone_id: Mapped[Optional[str]] = mapped_column(ForeignKey("zones.id"), nullable=True)
    worker_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    distance_m: Mapped[float] = mapped_column(Float)
    state: Mapped[str] = mapped_column(String)  # SAFE | WARNING | DANGER
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    forklift: Mapped["Forklift"] = relationship(back_populates="proximity_events")
    zone: Mapped[Optional["Zone"]] = relationship(back_populates="proximity_events")


# ------------------------------------------------------------------
# Safety events & alerts
# ------------------------------------------------------------------
class SafetyEvent(Base, IdMixin, TimestampMixin):
    __tablename__ = "safety_events"
    __table_args__ = (
        Index("ix_safetyevent_warehouse_created", "warehouse_id", "created_at"),
        Index("ix_safetyevent_severity", "severity"),
    )

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"), index=True)
    type: Mapped[SafetyEventType] = mapped_column(SqlEnum(SafetyEventType, name="safety_event_type"))
    severity: Mapped[Severity] = mapped_column(SqlEnum(Severity, name="severity"))
    forklift_id: Mapped[Optional[str]] = mapped_column(ForeignKey("forklifts.id"), nullable=True, index=True)
    # Plain FK column (no ORM relationship) — informational, not traversed as a relation anywhere.
    operator_id: Mapped[Optional[str]] = mapped_column(ForeignKey("operators.id"), nullable=True)
    zone_id: Mapped[Optional[str]] = mapped_column(ForeignKey("zones.id"), nullable=True)
    detected_object: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source: Mapped[EventSource] = mapped_column(SqlEnum(EventSource, name="event_source"))
    status: Mapped[EventStatus] = mapped_column(SqlEnum(EventStatus, name="event_status"), default=EventStatus.OPEN)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=True)

    forklift: Mapped[Optional["Forklift"]] = relationship(back_populates="safety_events")
    zone: Mapped[Optional["Zone"]] = relationship(back_populates="safety_events")
    warehouse: Mapped["Warehouse"] = relationship(back_populates="safety_events")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="safety_event")


class Alert(Base, IdMixin, TimestampMixin):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alert_warehouse_status", "warehouse_id", "status"),)

    warehouse_id: Mapped[str] = mapped_column(ForeignKey("warehouses.id", ondelete="CASCADE"), index=True)
    safety_event_id: Mapped[Optional[str]] = mapped_column(ForeignKey("safety_events.id"), nullable=True)
    severity: Mapped[Severity] = mapped_column(SqlEnum(Severity, name="severity"))
    title: Mapped[str] = mapped_column(String)
    forklift_id: Mapped[Optional[str]] = mapped_column(ForeignKey("forklifts.id"), nullable=True, index=True)
    worker_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Plain FK columns (no ORM relationship) — same as the Node schema.
    zone_id: Mapped[Optional[str]] = mapped_column(ForeignKey("zones.id"), nullable=True)
    acknowledged_by_user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[AlertStatus] = mapped_column(SqlEnum(AlertStatus, name="alert_status"), default=AlertStatus.ACTIVE)

    safety_event: Mapped[Optional["SafetyEvent"]] = relationship(back_populates="alerts")
    forklift: Mapped[Optional["Forklift"]] = relationship(back_populates="alerts")
    warehouse: Mapped["Warehouse"] = relationship(back_populates="alerts")


# ------------------------------------------------------------------
# Maintenance
# ------------------------------------------------------------------
class MaintenanceRecord(Base, IdMixin, TimestampMixin):
    __tablename__ = "maintenance_records"
    __table_args__ = (Index("ix_maintenance_forklift", "forklift_id"),)

    forklift_id: Mapped[str] = mapped_column(ForeignKey("forklifts.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[MaintenanceStatus] = mapped_column(
        SqlEnum(MaintenanceStatus, name="maintenance_status"), default=MaintenanceStatus.SCHEDULED
    )
    performed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    forklift: Mapped["Forklift"] = relationship(back_populates="maintenance_records")
