"""Verify Emotional Brain upgrade in dna_brain.py and brain/memory.py."""
import sys
sys.path.insert(0, "/Users/cristian.reyes/code/auralis")

from unittest.mock import MagicMock
sys.modules["structlog"] = MagicMock()

print("=== EMOTIONAL BRAIN VERIFICATION ===\n")
errors = 0

# ── 1. Stochastic selection ──
print("▸ 1. Stochastic selection")
from auralis.console.dna_brain import _select_with_creativity

candidates = {"style_a": 80.0, "style_b": 78.0, "style_c": 40.0}

# Temperature 0 = deterministic
det = _select_with_creativity(candidates, temperature=0.0)
assert det == "style_a", f"Expected deterministic=style_a, got {det}"
print(f"  ✅ temperature=0.0: always '{det}' (deterministic)")

# Temperature 0.3 = controlled creativity — run 20 times
results = set()
for _ in range(20):
    results.add(_select_with_creativity(candidates, temperature=0.3))
# Should mostly pick top candidates
print(f"  ✅ temperature=0.3: picked {results} across 20 runs")
if len(results) >= 2:
    print("  ✅ CONFIRMED: creativity introduces variety!")
else:
    print("  ⚠️  All same (OK at low temp, stochastic still works)")

# ── 2. Emotional Arc ──
print("\n▸ 2. Emotional Arc")
from auralis.console.dna_brain import EmotionalArc, StemPlan

# Breakdown = low energy
arc_bd = EmotionalArc("breakdown")
assert arc_bd.energy == 0.2, f"Expected energy=0.2, got {arc_bd.energy}"
print(f"  ✅ breakdown energy={arc_bd.energy}")

# Test stem adjustment — breakdown should widen stereo
plan = StemPlan(stem_name="other", stereo_width=1.0, compression={"ratio": 4.0})
adj = arc_bd.adjust_stem_plan(plan, "other")
assert plan.stereo_width > 1.0, f"Expected wider stereo, got {plan.stereo_width}"
assert plan.compression["ratio"] <= 2.0, f"Expected gentle ratio, got {plan.compression['ratio']}"
print(f"  ✅ breakdown adjustments: width={plan.stereo_width:.1f}, ratio={plan.compression['ratio']:.1f}")
for a in adj:
    print(f"    → {a}")

# Drop = high energy
arc_drop = EmotionalArc("drop")
assert arc_drop.energy == 1.0
bass_plan = StemPlan(stem_name="bass", sidechain=False, compression={"ratio": 2.0, "attack_ms": 20.0})
adj2 = arc_drop.adjust_stem_plan(bass_plan, "bass")
assert bass_plan.sidechain is True, "Expected sidechain=True for drop"
assert bass_plan.compression["ratio"] >= 4.0
assert bass_plan.compression["attack_ms"] <= 10.0
print(f"  ✅ drop adjustments: sidechain=True, ratio={bass_plan.compression['ratio']:.1f}, attack={bass_plan.compression['attack_ms']:.0f}ms")

# ── 3. Session Memory ──
print("\n▸ 3. Session Memory")
import tempfile
import os
from auralis.brain import memory as mem_mod

# Override memory path for testing
original_dir = mem_mod.MEMORY_DIR
original_file = mem_mod.MEMORY_FILE
tmpdir = tempfile.mkdtemp()
mem_mod.MEMORY_DIR = type(original_dir)(tmpdir)
mem_mod.MEMORY_FILE = type(original_file)(os.path.join(tmpdir, "test_memory.json"))

try:
    from auralis.brain.memory import SessionMemory
    sm = SessionMemory()
    assert sm.session_count == 0
    print(f"  ✅ Empty memory: {sm.session_count} sessions")

    # Record high-QC session
    sm.record_session(
        bpm=128.0, key="C", scale="minor",
        stem_choices=[
            {"stem_name": "drums", "style": "four_on_floor", "qc_score": 92.0},
            {"stem_name": "bass", "patch": "sub_bass", "qc_score": 88.0},
        ],
        overall_qc=90.0,
    )
    assert sm.session_count == 1
    print(f"  ✅ Recorded session: {sm.session_count} sessions")

    # Check preference bonus
    bonus = sm.preference_bonus("drums", "four_on_floor", bpm=128.0)
    assert bonus > 0, f"Expected bonus > 0, got {bonus}"
    print(f"  ✅ preference_bonus (drums, four_on_floor) = {bonus}")

    # No bonus for unused choice
    no_bonus = sm.preference_bonus("drums", "trap", bpm=128.0)
    assert no_bonus == 0.0
    print(f"  ✅ preference_bonus (drums, trap) = {no_bonus} (no history)")

    # Persistence test
    sm2 = SessionMemory()
    assert sm2.session_count == 1, f"Expected 1, got {sm2.session_count}"
    print(f"  ✅ Persistence: reloaded {sm2.session_count} sessions from disk")

finally:
    mem_mod.MEMORY_DIR = original_dir
    mem_mod.MEMORY_FILE = original_file
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

# ── 4. Full DNABrain.think() with new params ──
print("\n▸ 4. DNABrain.think() integration")
from auralis.console.dna_brain import DNABrain

brain = DNABrain()
report = brain.think(deep_profile=None, section_type="breakdown", creativity=0.3)

assert report.stem_plans, "No stem plans generated"
assert report.reasoning_chain, "No reasoning chain"
has_arc = any("ARC" in r or "Emotional" in r for r in report.reasoning_chain)
has_section = any("breakdown" in r for r in report.reasoning_chain)
assert has_arc, "No emotional arc in reasoning"
assert has_section, "Section type not mentioned"
print(f"  ✅ {len(report.stem_plans)} stem plans, {len(report.reasoning_chain)} reasoning steps")
print(f"  ✅ Emotional arc present in reasoning: {has_arc}")
for r in report.reasoning_chain[:6]:
    print(f"    {r}")

print(f"\n{'🎯 ALL PASSED' if errors == 0 else f'⚠️ {errors} FAILED'}")
sys.exit(errors)
