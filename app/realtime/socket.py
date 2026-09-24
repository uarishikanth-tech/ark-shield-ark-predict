"""
Real-time layer for the Alert Center / live map — port of
src/realtime/socket.ts. Uses python-socketio, which speaks the same
Socket.IO wire protocol as the Node/Engine.IO server, so any existing
socket.io-client frontend can connect to this backend unmodified.

Events broadcast here (identical names/payloads to the Node version):
    "simulation:tick"   -> forklift/worker positions each tick
    "simulation:status" -> { running: bool }
    "safety_event:new"  -> a SafetyEvent was just created
    "alert:new"         -> an Alert was just created
    "alert:updated"     -> an Alert was acknowledged/resolved

Clients authenticate with the same JWT used for the REST API via the
Socket.IO handshake's `auth` payload:
    io(url, { auth: { token } })
Connections are placed in an "org:<organizationId>" room so one
tenant never receives another tenant's real-time data.
"""
import jwt
import socketio

from app.config import settings
from app.security import decode_access_token

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.web_origin,
)


@sio.event
async def connect(sid: str, _environ: dict, auth: dict | None):
    token = (auth or {}).get("token")
    if not token:
        raise socketio.exceptions.ConnectionRefusedError("Unauthorized: missing token")
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        raise socketio.exceptions.ConnectionRefusedError("Unauthorized: invalid token")

    await sio.save_session(sid, {"email": payload.email, "organization_id": payload.organization_id})
    await sio.enter_room(sid, f"org:{payload.organization_id}")
    print(f"[socket] {payload.email} connected, joined org:{payload.organization_id}")


@sio.event
async def disconnect(sid: str):
    try:
        session = await sio.get_session(sid)
        print(f"[socket] {session.get('email')} disconnected")
    except KeyError:
        pass


async def emit_to_org(event: str, payload, organization_id: str) -> None:
    """Broadcasts only to sockets in this organization's room."""
    await sio.emit(event, payload, room=f"org:{organization_id}")
