"""Centralized error handling — port of src/middleware/errorHandler.ts."""
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, NoResultFound


class AppError(Exception):
    """
    Raised deliberately by routes/services for expected error
    conditions (not found, forbidden, bad input, etc). Anything that
    is NOT an AppError is treated as unexpected and logged
    server-side without leaking details to the client.
    """

    def __init__(self, message: str, status_code: int = 400, details=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    @classmethod
    def not_found(cls, resource: str = "Resource") -> "AppError":
        return cls(f"{resource} not found", status.HTTP_404_NOT_FOUND)

    @classmethod
    def forbidden(cls, message: str = "You do not have permission to perform this action") -> "AppError":
        return cls(message, status.HTTP_403_FORBIDDEN)

    @classmethod
    def unauthorized(cls, message: str = "Authentication required") -> "AppError":
        return cls(message, status.HTTP_401_UNAUTHORIZED)


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    body = {"error": exc.message}
    if exc.details is not None:
        body["details"] = exc.details
    return JSONResponse(status_code=exc.status_code, content=body)


async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"path": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Validation failed", "details": details})


async def not_found_handler(_request: Request, _exc: NoResultFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "Resource not found"})


async def integrity_error_handler(_request: Request, exc: IntegrityError) -> JSONResponse:
    # Never leak raw SQL/constraint details to the client.
    print(f"Database integrity error: {exc}")
    if "unique" in str(exc.orig).lower():
        return JSONResponse(status_code=409, content={"error": "A record with that value already exists"})
    return JSONResponse(status_code=500, content={"error": "Database error"})


async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    print(f"Unhandled error: {exc!r}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
