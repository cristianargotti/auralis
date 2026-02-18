"""AURALIS DSP Safety Tests — Verifies all pipeline audit fixes.

Tests allpass filter overflow, NaN guards, shimmer reverb stability,
compressor safety, mixer NaN protection, and mastering gain caps.
"""

import numpy as np
import pytest


# ── Test 1: Allpass filter no longer overflows ────────────

def test_reverb_allpass_no_overflow():
    """The old allpass formula had feedback > 1.0 causing NaN. Fixed."""
    from auralis.hands.effects import apply_reverb, ReverbConfig

    audio = np.random.randn(44100).astype(np.float64) * 0.5
    # Worst case: max room, min damping
    result = apply_reverb(audio, ReverbConfig(room_size=0.95, damping=0.1, wet=0.5))

    assert not np.any(np.isnan(result)), "NaN in reverb output"
    assert not np.any(np.isinf(result)), "Inf in reverb output"
    assert np.max(np.abs(result)) < 10.0, f"Peak too high: {np.max(np.abs(result))}"


# ── Test 2: process_chain NaN safety net ─────────────────

def test_process_chain_nan_guard():
    """NaN/Inf injected into chain must be cleaned at output."""
    from auralis.hands.effects import process_chain, EffectChain

    corrupted = np.array([0.5, np.nan, 0.3, np.inf, -0.5, float("nan")])
    chain = EffectChain(name="test_chain")
    result = process_chain(corrupted, chain)

    assert not np.any(np.isnan(result)), "NaN leaked through process_chain"
    assert not np.any(np.isinf(result)), "Inf leaked through process_chain"


# ── Test 3: Shimmer reverb accumulation clipped ──────────

def test_shimmer_reverb_no_runaway():
    """Shimmer reverb with long decay must not accumulate to overflow."""
    from auralis.hands.effects import apply_shimmer_reverb

    audio = np.random.randn(44100 * 2).astype(np.float64) * 0.8
    result = apply_shimmer_reverb(audio, decay_s=4.0, wet=0.5)

    assert not np.any(np.isnan(result)), "NaN in shimmer reverb"
    assert np.max(np.abs(result)) < 20.0, f"Shimmer peak too high: {np.max(np.abs(result))}"


# ── Test 4: Full effect chain with reverb ────────────────

def test_full_chain_with_reverb():
    """Complete chain including reverb must produce clean audio."""
    from auralis.hands.effects import process_chain, EffectChain, ReverbConfig

    chain = EffectChain(
        name="full_test",
        reverb=ReverbConfig(room_size=0.9, damping=0.2, wet=0.4),
    )
    audio = np.random.randn(44100).astype(np.float64) * 0.5
    result = process_chain(audio, chain)

    assert not np.any(np.isnan(result)), "NaN in full chain output"
    assert np.max(np.abs(result)) < 10.0


# ── Test 5: Compressor doesn't divide by zero ───────────

def test_compressor_near_zero_envelope():
    """Near-silent audio must not cause division by zero in compressor."""
    from auralis.hands.effects import apply_compressor, CompressorConfig

    # Extremely quiet audio
    audio = np.ones(44100, dtype=np.float64) * 1e-12
    config = CompressorConfig(threshold_db=-60.0, ratio=8.0)
    result = apply_compressor(audio, config)

    assert not np.any(np.isnan(result)), "NaN from compressor on quiet audio"
    assert not np.any(np.isinf(result)), "Inf from compressor on quiet audio"


# ── Test 6: Mixer NaN guard ─────────────────────────────

def test_mixer_nan_guard():
    """Mixer must clean NaN before writing to disk."""
    from auralis.hands.mixer import Mixer, MixConfig
    import tempfile, os

    mixer = Mixer(MixConfig(sample_rate=44100))
    # Inject NaN audio
    bad_audio = np.array([0.5, np.nan, 0.3, np.nan, -0.5], dtype=np.float64)
    mixer.add_track("test", bad_audio)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        result = mixer.mix(output_path=tmp_path)
        import soundfile as sf
        data, _ = sf.read(tmp_path)
        assert not np.any(np.isnan(data)), "NaN written to WAV by mixer"
    finally:
        os.unlink(tmp_path)


# ── Test 7: Mastering gain cap ───────────────────────────

def test_mastering_gain_cap():
    """Near-silent input must not get +60dB of gain applied."""
    from auralis.console.mastering import master_audio, MasterConfig
    import tempfile, os, soundfile as sf

    # Create a near-silent WAV
    sr = 44100
    silence = np.ones((sr, 2), dtype=np.float64) * 1e-8
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        in_path = f.name
    sf.write(in_path, silence, sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    try:
        config = MasterConfig(target_lufs=-8.0)
        result = master_audio(in_path, out_path, config=config)
        data, _ = sf.read(out_path)
        peak = np.max(np.abs(data))
        # Should NOT be amplified to full scale
        assert peak < 0.5, f"Silent input was amplified to peak={peak}"
        assert not np.any(np.isnan(data)), "NaN in mastered output"
    finally:
        os.unlink(in_path)
        os.unlink(out_path)


# ── Test 8: Comb filter feedback clamped ─────────────────

def test_comb_filter_feedback_clamped():
    """Comb filter feedback must be clamped ≤0.95 regardless of params."""
    from auralis.hands.effects import apply_reverb, ReverbConfig

    # Max room_size, min damping → highest possible feedback
    audio = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float64)
    result = apply_reverb(audio, ReverbConfig(room_size=1.0, damping=0.0, wet=1.0))

    assert not np.any(np.isnan(result)), "Comb filter overflowed to NaN"
    assert not np.any(np.isinf(result)), "Comb filter overflowed to Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
