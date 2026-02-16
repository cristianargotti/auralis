"""AURALIS global configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    env: str = "development"

    # OpenAI
    openai_api_key: str = ""

    # AWS
    aws_profile: str = ""
    aws_region: str = "us-east-1"

    # Paths
    projects_dir: Path = Path("./projects")
    samples_dir: Path = Path("./samples")

    model_config = {"env_prefix": "AURALIS_"}


settings = Settings()
