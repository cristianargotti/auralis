"""AURALIS authentication — JWT-based username/password security.

Environment variables:
- AURALIS_AUTH_USERNAME: admin username (required)
- AURALIS_AUTH_PASSWORD_HASH: bcrypt hash of the password (required)
- AURALIS_JWT_SECRET: secret key for JWT signing (required)

Generate a password hash:
    python -c "from passlib.hash import bcrypt; print(bcrypt.hash('your-password'))"
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.hash import bcrypt  # type: ignore[import-untyped]
from pydantic import BaseModel

from auralis.config import settings

# OAuth2 scheme — points to the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# JWT config
_ALGORITHM = "HS256"
_TOKEN_EXPIRE_HOURS = 24


class TokenResponse(BaseModel):
    """Response from the login endpoint."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int


class UserPayload(BaseModel):
    """Decoded JWT user information."""

    username: str
    exp: datetime


def _create_token(username: str) -> tuple[str, int]:
    """Create a signed JWT token for the given username."""
    expires_delta = timedelta(hours=_TOKEN_EXPIRE_HOURS)
    expire = datetime.now(UTC) + expires_delta
    payload: dict[str, Any] = {"sub": username, "exp": expire}
    token = jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)
    return token, int(expires_delta.total_seconds())


def verify_password(plain_password: str) -> bool:
    """Verify a plain password against the stored hash."""
    if not settings.auth_password_hash:
        return False
    result: bool = bcrypt.verify(plain_password, settings.auth_password_hash)  # type: ignore[no-untyped-call]
    return result


def authenticate_user(username: str, password: str) -> bool:
    """Check username and password against config."""
    if username != settings.auth_username:
        return False
    return verify_password(password)


async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenResponse:
    """Authenticate and return a JWT token."""
    if not authenticate_user(form.username, form.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = _create_token(form.username)
    return TokenResponse(access_token=token, expires_in=expires_in)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPayload:
    """Decode and validate the JWT token — FastAPI dependency."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
        return UserPayload(username=username, exp=datetime.fromtimestamp(payload["exp"], tz=UTC))
    except JWTError:
        raise credentials_exception from None
