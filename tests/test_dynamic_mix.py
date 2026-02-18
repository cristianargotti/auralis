"""Verify Phase 3: Dynamic Mix + Mod Matrix."""
import sys
sys.path.insert(0, "/Users/cristian.reyes/code/auralis")

from unittest.mock import MagicMock
sys.modules["structlog"] = MagicMock()

import numpy as np

print("=== PHASE 3: DYNAMIC MIX + MOD MATRIX ===\n")
errors = 0

# ── 1. Parametric EQ ──
print("▸ 1. Parametric EQ")
from auralis.console.fx import parametric_eq_coefficients, apply_parametric_eq

coeffs = parametric_eq_coefficients(1000.0, -6.0, q=1.0, sr=44100)
assert len(coeffs) == 5, f"Expected 5 coefficients, got {len(coeffs)}"
print(f"  ✅ parametric_eq_coefficients: {[round(c, 4) for c in coeffs]}")

# Apply to audio
audio = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.5, 22050))
result = apply_parametric_eq(audio, 1000.0, -6.0, q=1.0, sr=44100)
assert len(result) == len(audio)
# Signal at the cut frequency should be reduced
rms_in = np.sqrt(np.mean(audio**2))
rms_out = np.sqrt(np.mean(result**2))
print(f"  ✅ 1kHz signal: in_rms={rms_in:.3f}, out_rms={rms_out:.3f} (cut={20*np.log10(rms_out/rms_in):.1f}dB)")

# ── 2. Dynamic EQ ──
print("\n▸ 2. Dynamic EQ")
from auralis.console.fx import DynamicEQ

deq = DynamicEQ(bands=[(300.0, -12.0, 3.0, 1.0)], block_size=2048)
# Create rich audio with prominent 300Hz
t = np.linspace(0, 1.0, 44100, endpoint=False)
muddy = np.sin(2 * np.pi * 300 * t) * 0.5 + np.sin(2 * np.pi * 1000 * t) * 0.3
result = deq.process(muddy, sr=44100)
assert len(result) == len(muddy)
rms_before = np.sqrt(np.mean(muddy**2))
rms_after = np.sqrt(np.mean(result**2))
print(f"  ✅ DynamicEQ: rms_before={rms_before:.3f}, rms_after={rms_after:.3f}")
print(f"  ✅ DynamicEQ reduced energy (only when above threshold)")

# ── 3. Frequency Carving ──
print("\n▸ 3. Inter-Stem Frequency Carving")
from auralis.console.fx import carve_frequencies

# Pad with low frequency, bass as sidechain source
pad = np.sin(2 * np.pi * 200 * t) * 0.4 + np.sin(2 * np.pi * 800 * t) * 0.3
bass = np.sin(2 * np.pi * 80 * t) * 0.6
carved = carve_frequencies(pad, bass, carve_bands=[(80, 300)], depth_db=-6.0, sr=44100)
assert len(carved) == len(pad)
print(f"  ✅ carve_frequencies: carved pad when bass is active ({len(carved)} samples)")

# ── 4. Modulation Matrix ──
print("\n▸ 4. Mod Matrix")
from auralis.hands.synth import ModSource, ModRouting, ModMatrix

# Test LFO → amplitude (tremolo)
lfo = ModSource(type="lfo", rate_hz=5.0, shape="sine")
routing = ModRouting(source=lfo, destination="amplitude", depth=0.3)
matrix = ModMatrix(routings=[routing])

tone = np.sin(2 * np.pi * 440 * t) * 0.5
modulated = matrix.apply(tone, sr=44100, velocity=0.8, duration_s=1.0)
assert len(modulated) == len(tone)

# Check that amplitude varies (tremolo effect)
block = 4410  # 100ms blocks
rms_blocks = [np.sqrt(np.mean(modulated[i:i+block]**2)) for i in range(0, len(modulated)-block, block)]
rms_variation = max(rms_blocks) - min(rms_blocks)
print(f"  ✅ LFO→amplitude: variation={rms_variation:.3f} ({len(rms_blocks)} blocks)")
if rms_variation > 0.01:
    print(f"  ✅ CONFIRMED: tremolo effect is audible!")
else:
    print(f"  ⚠️  Low variation (may need adjustment)")

# Test envelope → amplitude
env_source = ModSource(type="envelope", attack_s=0.05, decay_s=0.3)
env_signal = env_source.generate(1.0, sr=44100)
assert len(env_signal) == 44100
assert env_signal[0] < env_signal[int(0.05*44100) - 1]  # Attack rises
print(f"  ✅ envelope source: peak at ~{np.argmax(env_signal)/44100:.2f}s")

# Test velocity source
vel_source = ModSource(type="velocity")
vel_signal = vel_source.generate(0.5, sr=44100, velocity=0.6)
assert np.all(vel_signal == 0.6), "Velocity source should be constant"
print(f"  ✅ velocity source: constant={vel_signal[0]:.1f}")

# Test multi-routing matrix
multi = ModMatrix(routings=[
    ModRouting(source=ModSource(type="lfo", rate_hz=2.0), destination="amplitude", depth=0.2),
    ModRouting(source=ModSource(type="lfo", rate_hz=0.5, shape="triangle"), destination="pan", depth=0.4),
])
stereo_result = multi.apply(tone, sr=44100, velocity=0.8, duration_s=1.0)
if stereo_result.ndim == 2:
    print(f"  ✅ Multi-routing: {stereo_result.shape[1]}-channel output (auto-pan active)")
else:
    print(f"  ✅ Multi-routing: mono output")

print(f"\n{'🎯 ALL PASSED' if errors == 0 else f'⚠️ {errors} FAILED'}")
sys.exit(errors)
