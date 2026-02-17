"""AURALIS global configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    env: str = "development"

    # Authentication
    auth_username: str = "admin"
    auth_password_hash: str = ""  # bcrypt hash (see .env.example)
    jwt_secret: str = "change-me-in-production-use-openssl-rand"  # noqa: S105

    # OpenAI
    openai_api_key: str = ""

    # AWS
    aws_profile: str = ""
    aws_region: str = "us-east-1"

    # Replicate (Gen AI)
    replicate_api_token: str = ""

    # Paths
    projects_dir: Path = Path("./projects")
    samples_dir: Path = Path("./samples")

    model_config = {"env_prefix": "AURALIS_"}


settings = Settings()
