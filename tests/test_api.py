"""Tests for AURALIS API — health, auth, console, and QC endpoints."""

from __future__ import annotations

import wave
import struct
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from auralis.api.server import app

client = TestClient(app)


# ── Helpers ──────────────────────────────────────────────


def _generate_test_wav(path: str, duration_s: float = 0.5, sr: int = 44100) -> None:
    """Generate a minimal stereo WAV file with a sine wave."""
    n_samples = int(sr * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n_samples):
            val = int(16000 * math.sin(2 * math.pi * 440 * i / sr))
            wf.writeframes(struct.pack("<hh", val, val))


def _get_token() -> str:
    """Get a valid JWT token."""
    response = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "AuralisEngine2026!"},
    )
    if response.status_code != 200:
        pytest.skip("Auth not configured for tests")
    return response.json()["access_token"]


# ── Health & Version ─────────────────────────────────────


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


def test_api_info() -> None:
    """Info endpoint returns available endpoints."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "endpoints" in data


# ── Console Module Tests ─────────────────────────────────


def test_mastering_presets_exist() -> None:
    """Mastering presets are properly defined."""
    from auralis.console.mastering import PRESETS

    assert "mood_check" in PRESETS
    assert "streaming" in PRESETS
    assert "club" in PRESETS
    assert PRESETS["streaming"].target_lufs == -14.0
    assert PRESETS["club"].target_lufs == -8.0


def test_master_config_defaults() -> None:
    """MasterConfig has correct defaults."""
    from auralis.console.mastering import MasterConfig

    config = MasterConfig()
    assert config.target_lufs == -8.0
    assert config.ceiling_db == -0.3
    assert config.bit_depth == 24


def test_master_audio_integration() -> None:
    """Master audio function processes a test WAV file."""
    from auralis.console.mastering import master_audio

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "test_input.wav")
        output_path = str(Path(tmpdir) / "test_output.wav")
        _generate_test_wav(input_path, duration_s=0.5)

        result = master_audio(input_path, output_path, preset="mood_check")

        assert Path(result.output_path).exists()
        assert result.clipping_samples >= 0
        assert len(result.stages_applied) > 0


# ── QC Module Tests ──────────────────────────────────────


def test_qc_analysis() -> None:
    """QC analysis runs on a test WAV file."""
    from auralis.console.qc import run_qc

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "test.wav")
        _generate_test_wav(input_path, duration_s=0.5)

        report = run_qc(input_path)

        assert report.pass_fail in ("PASS", "FAIL")
        assert report.dynamics is not None
        assert report.spectrum is not None
        assert report.sample_rate == 44100
        assert report.channels == 2


def test_qc_dynamics_values() -> None:
    """QC dynamics metrics are within reasonable range."""
    from auralis.console.qc import run_qc

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "test.wav")
        _generate_test_wav(input_path, duration_s=0.2)

        report = run_qc(input_path)

        assert -60 < report.dynamics.peak_db < 0
        assert -60 < report.dynamics.rms_db < 0
        assert report.dynamics.crest_factor_db >= 0


def test_qc_spectrum_bands() -> None:
    """QC spectrum returns all 7 bands."""
    from auralis.console.qc import run_qc

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "test.wav")
        _generate_test_wav(input_path, duration_s=0.2)

        report = run_qc(input_path)

        for band in ("sub", "bass", "low_mid", "mid", "upper_mid", "presence", "brilliance"):
            assert hasattr(report.spectrum, band)
            assert isinstance(getattr(report.spectrum, band), float)


# ── Visualization Tests ──────────────────────────────────


def test_qc_radar_generation() -> None:
    """QC radar chart generates a valid PNG."""
    from auralis.console.visualize import generate_qc_radar

    png = generate_qc_radar(
        peak_db=-1.5, rms_db=-10.0, crest_db=8.5,
        stereo_corr=0.85, stereo_width=0.35,
        lufs=-14.0, dynamic_range=8.0,
    )

    assert isinstance(png, bytes)
    assert len(png) > 1000  # Valid PNG is at least a few KB
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_loudness_timeline_generation() -> None:
    """Loudness timeline generates a valid PNG."""
    from auralis.console.visualize import generate_loudness_timeline

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "test.wav")
        _generate_test_wav(input_path, duration_s=1.0)

        png = generate_loudness_timeline(input_path)

        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_waveform_comparison_generation() -> None:
    """Waveform comparison generates a valid PNG."""
    from auralis.console.visualize import generate_waveform_comparison

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = str(Path(tmpdir) / "a.wav")
        path_b = str(Path(tmpdir) / "b.wav")
        _generate_test_wav(path_a, duration_s=0.5)
        _generate_test_wav(path_b, duration_s=0.5)

        png = generate_waveform_comparison(path_a, path_b)

        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_spectrum_comparison_generation() -> None:
    """Spectrum comparison generates a valid PNG."""
    from auralis.console.visualize import generate_spectrum_comparison

    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = str(Path(tmpdir) / "a.wav")
        path_b = str(Path(tmpdir) / "b.wav")
        _generate_test_wav(path_a, duration_s=0.5)
        _generate_test_wav(path_b, duration_s=0.5)

        png = generate_spectrum_comparison(path_a, path_b)

        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ── API Endpoint Tests ───────────────────────────────────


def test_presets_endpoint() -> None:
    """Presets endpoint returns available presets."""
    try:
        token = _get_token()
    except Exception:
        pytest.skip("Auth not available")
        return

    response = client.get(
        "/api/console/presets",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "streaming" in data
    assert "club" in data
    assert "target_lufs" in data["streaming"]
