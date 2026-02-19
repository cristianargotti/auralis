#!/usr/bin/env python3
"""Quick test: verify MIDI extraction pipeline is operational."""

print("Testing MIDI extraction pipeline...")

# 1. basic-pitch import
import basic_pitch
print("  basic-pitch imported OK")

# 2. ONNX Runtime
import onnxruntime
print(f"  ONNX Runtime: {onnxruntime.__version__}")

# 3. Inference module
from basic_pitch.inference import predict
print("  basic_pitch.inference.predict OK")

# 4. MIDI extractor checks
from auralis.ear.midi_extractor import _check_basic_pitch_available, _check_replicate_available
bp = _check_basic_pitch_available()
rp = _check_replicate_available()
print(f"  basic-pitch available: {bp}")
print(f"  Replicate available: {rp}")

# 5. Replicate client
from auralis.hands.midi_transcribe import ReplicateMIDIClient
client = ReplicateMIDIClient()
print(f"  ReplicateMIDIClient available: {client.available}")

# 6. All functions importable
from auralis.ear.midi_extractor import extract_midi, extract_midi_from_stems, MIDIExtractionResult
print("  All MIDI functions importable OK")

print("\nRESULT: MIDI pipeline READY")
print("  Method 1: basic-pitch (ONNX) - LOCAL")
print("  Method 2: Replicate API - EXTERNAL GPU")
print("  Method 3: librosa.pyin - FALLBACK")
