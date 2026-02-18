"""Verify Living Timbre upgrade in timbre_cloner.py."""
import sys
sys.path.insert(0, "/Users/cristian.reyes/code/auralis")

# Mock structlog (not installed in this env)
from unittest.mock import MagicMock
sys.modules["structlog"] = MagicMock()
import numpy as np
from auralis.hands.timbre_cloner import analyze_timbre, clone_render, morph_timbres, TimbreModel

print("=== LIVING TIMBRE VERIFICATION ===\n")

sr = 22050
t = np.linspace(0, 1.0, sr, endpoint=False)

# Pluck-like: bright attack decaying to warm sustain
attack_env = np.exp(-t * 5)
audio = (
    np.sin(2 * np.pi * 220 * t) * 0.5 +
    np.sin(2 * np.pi * 440 * t) * 0.3 * attack_env +
    np.sin(2 * np.pi * 660 * t) * 0.2 * attack_env +
    np.random.randn(len(t)) * 0.02 * attack_env
)

model = analyze_timbre(audio, sr=sr, name="test_pluck")

# Check new fields
print(f"harmonic_amps: {len(model.harmonic_amps)} harmonics")
print(f"harmonic_frames: {len(model.harmonic_frames)} frames x {len(model.harmonic_frames[0]) if model.harmonic_frames else 0}")
print(f"transient_profile: {len(model.transient_profile)} bins")
print(f"noise_ratio (static): {model.noise_ratio}")
print(f"noise_envelope: {len(model.noise_envelope)} points -> {[round(v,3) for v in model.noise_envelope]}")

errors = 0

# 1. Harmonic evolution
if model.harmonic_frames and len(model.harmonic_frames) >= 2:
    first, last = model.harmonic_frames[0], model.harmonic_frames[-1]
    if len(first) >= 3:
        h2_diff = first[1] - last[1]
        print(f"\n2nd harmonic: frame[0]={first[1]:.3f} -> frame[-1]={last[1]:.3f} (delta={h2_diff:.3f})")
        if h2_diff > 0.01:
            print("  ✅ Harmonics evolve over time (living sound)")
        else:
            print("  ⚠️  Small evolution")
else:
    print("❌ No harmonic frames"); errors += 1

# 2. Render
rendered = clone_render(model, midi_note=60, duration_s=0.5, velocity=0.8, sr=44100)
if len(rendered) > 0 and np.max(np.abs(rendered)) > 0:
    print(f"✅ clone_render: {len(rendered)} samples, peak={np.max(np.abs(rendered)):.3f}")
else:
    print("❌ clone_render failed"); errors += 1

# 3. Morph
model2 = analyze_timbre(np.sin(2 * np.pi * 440 * t), sr=sr, name="pure_sine")
morphed = morph_timbres(model, model2, blend=0.5)
if morphed.harmonic_frames and morphed.noise_envelope:
    print(f"✅ morph: frames={len(morphed.harmonic_frames)}, noise_env={len(morphed.noise_envelope)}, transient={len(morphed.transient_profile)}")
else:
    print("❌ morph missing new fields"); errors += 1

# 4. Serialization
d = model.to_dict()
m2 = TimbreModel.from_dict(d)
assert len(m2.harmonic_frames) == len(model.harmonic_frames)
assert len(m2.noise_envelope) == len(model.noise_envelope)
assert len(m2.transient_profile) == len(model.transient_profile)
print("✅ Serialization roundtrip OK")

# 5. Backward compat
old = TimbreModel(name="old", harmonic_amps=[1.0, 0.5, 0.3], noise_ratio=0.1)
r = clone_render(old, midi_note=60, duration_s=0.3, velocity=0.8)
print(f"✅ Backward compat: {len(r)} samples")

print(f"\n{'🎯 ALL PASSED' if errors == 0 else f'⚠️ {errors} FAILED'}")
sys.exit(errors)
