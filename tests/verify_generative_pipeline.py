
import sys
import os
from pathlib import Path
import structlog

# Add project root
sys.path.append("/Users/cristian.reyes/code/auralis")

from auralis.console.stem_decisions import StemDecision
from auralis.console.stem_generator import generate_stem

logger = structlog.get_logger()

def test_generative_pipeline():
    output_dir = Path("tests/output/gen_ai_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Decision: Replace Bass with AI
    decision = StemDecision(
        stem_name="bass",
        action="replace",
        quality_score=10.0,
        reason="Testing AI Generation",
        synth_patch="analog_moog",
        pattern_style="melodic_techno"
    )

    print("🚀 Starting AI Generation Test (Replicate)...")
    print(f"Prompt context: {decision.pattern_style} {decision.stem_name} loop, 124 BPM, Gm")

    try:
        path = generate_stem(
            decision=decision,
            bpm=124.0,
            key="G",
            scale="minor",
            duration_s=6.0, # Short duration for speed/cost
            output_dir=output_dir,
        )

        if path and path.exists():
            print(f"✅ SUCCESS: Generated AI Stem!")
            print(f"📂 Path: {path}")
            print(f"📦 Size: {path.stat().st_size / 1024:.2f} KB")
        else:
            print("❌ FAILED: No file generated (check logs)")

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_generative_pipeline()
