from fastapi import APIRouter, Depends
from pydantic import EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.camel import CamelModel
from app.database import get_db
from app.deps import get_current_user
from app.errors import AppError
from app.models import Role, User
from app.security import create_access_token, verify_password
from app.types import JwtPayload

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(CamelModel):
    email: EmailStr
    password: str


class UserOut(CamelModel):
    id: str
    email: str
    name: str
    role: Role
    organization_id: str


class LoginResponse(CamelModel):
    token: str
    user: UserOut


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Demo-only credential auth. Passwords are bcrypt-hashed at rest (see seed.py)."""
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalars().first()

    def invalid():
        return AppError("Invalid email or password", 401)

    if not user or not user.active:
        raise invalid()
    if not verify_password(body.password, user.password_hash):
        raise invalid()

    token = create_access_token(
        JwtPayload(user_id=user.id, organization_id=user.organization_id, role=user.role, email=user.email)
    )
    return LoginResponse(token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
async def me(current_user: JwtPayload = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Resolve the current token to a fresh user record."""
    result = await db.execute(select(User).where(User.id == current_user.user_id))
    user = result.scalars().first()
    if not user:
        raise AppError.not_found("User")
    return UserOut.model_validate(user)
