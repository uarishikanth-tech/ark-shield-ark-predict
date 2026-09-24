from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import settings
from app.types import JwtPayload


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(payload: JwtPayload) -> str:
    to_encode = payload.model_dump(mode="json")
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expires_hours)
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> JwtPayload:
    """Raises jwt.PyJWTError (caught by callers) if the token is invalid or expired."""
    raw = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return JwtPayload(
        user_id=raw["user_id"],
        organization_id=raw["organization_id"],
        role=raw["role"],
        email=raw["email"],
    )
