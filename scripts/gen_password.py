#!/usr/bin/env python3
"""Generate bcrypt password hash and JWT secret for AURALIS auth.

Usage:
    uv run python scripts/gen_password.py <password>
"""

import secrets
import sys

import bcrypt


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/gen_password.py <password>")
        sys.exit(1)

    password = sys.argv[1].encode()
    password_hash = bcrypt.hashpw(password, bcrypt.gensalt()).decode()
    jwt_secret = secrets.token_hex(32)

    print("AURALIS_AUTH_USERNAME=admin")
    print(f"AURALIS_AUTH_PASSWORD_HASH={password_hash}")
    print(f"AURALIS_JWT_SECRET={jwt_secret}")


if __name__ == "__main__":
    main()
