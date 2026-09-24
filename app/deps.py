from typing import Iterable

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.errors import AppError
from app.models import Role
from app.security import decode_access_token
from app.types import JwtPayload

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> JwtPayload:
    """Verifies the Bearer JWT and returns the decoded payload. Port of requireAuth."""
    if credentials is None or not credentials.credentials:
        raise AppError.unauthorized("Missing or malformed Authorization header")
    try:
        return decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise AppError.unauthorized("Invalid or expired token")


def require_role(allowed: Iterable[Role]):
    """Dependency factory — port of requireRole(allowed)."""
    allowed_set = set(allowed)

    def _check(user: JwtPayload = Depends(get_current_user)) -> JwtPayload:
        if user.role not in allowed_set:
            raise AppError.forbidden(f"Role {user.role} cannot access this resource")
        return user

    return _check
