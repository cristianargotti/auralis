"""Test AURALIS API health check."""

from fastapi.testclient import TestClient

from auralis.api.server import app

client = TestClient(app)


def test_health_check() -> None:
    """Health endpoint returns OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "auralis"


def test_version() -> None:
    """Package has correct version."""
    from auralis import __version__

    assert __version__ == "0.1.0"


def test_settings_defaults() -> None:
    """Settings load with correct defaults."""
    from auralis.config import Settings

    s = Settings()
    assert s.port == 8000
    assert s.env == "development"
