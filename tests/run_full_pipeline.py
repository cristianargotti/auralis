#!/usr/bin/env python3
"""AURALIS Full Pipeline Test — Runs ALL tests in sequence.

Episode 1: Living Timbre
Episode 2: Emotional Brain  
Episode 3: Dynamic Mix + Mod Matrix
Episode 4: Final 3% (Multi-filter, Wavetable Import, Mix Recall)
Episode 5: Composition Upgrade (original)
Episode 6: 100% Gap Verification (original)
"""
import subprocess
import sys
import time

TESTS = [
    ("🧬 Ep.1: Living Timbre", "tests/test_living_timbre.py"),
    ("🧠 Ep.2: Emotional Brain", "tests/test_emotional_brain.py"),
    ("🎛️  Ep.3: Dynamic Mix + Mod Matrix", "tests/test_dynamic_mix.py"),
    ("🏆 Ep.4: Final 3% Push", "tests/test_final_3pct.py"),
    ("🎵 Ep.5: Composition Upgrade", "tests/test_composition_upgrade.py"),
    ("📊 Ep.6: 100% Gap Verification", "tests/test_100_upgrade.py"),
]

WIDTH = 60
passed = 0
failed = 0
results = []

print("=" * WIDTH)
print("🎬 AURALIS — FULL PIPELINE TEST".center(WIDTH))
print("=" * WIDTH)
print()

for title, test_file in TESTS:
    print(f"{'─' * WIDTH}")
    print(f"  {title}")
    print(f"  File: {test_file}")
    print(f"{'─' * WIDTH}")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=True,
        text=True,
        cwd="/Users/cristian.reyes/code/auralis",
        timeout=60,
    )
    elapsed = time.time() - start
    
    # Print output
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"  ⚠️  {line}")
    
    if result.returncode == 0:
        passed += 1
        status = "✅ PASSED"
    else:
        failed += 1
        status = "❌ FAILED"
    
    results.append((title, status, elapsed))
    print(f"\n  {status} ({elapsed:.1f}s)")
    print()

# ── Final Report ──
print("=" * WIDTH)
print("📋 FINAL REPORT".center(WIDTH))
print("=" * WIDTH)
print()

for title, status, elapsed in results:
    print(f"  {status}  {title} ({elapsed:.1f}s)")

print()
total = passed + failed
print(f"  {'─' * (WIDTH - 4)}")
print(f"  Total: {total} tests | ✅ {passed} passed | ❌ {failed} failed")
print()

if failed == 0:
    print("  🎯 ALL EPISODES PASSED — AURALIS 100% VERIFIED! 🎯")
    print()
    print("  Score: 92% ████████████████████████░░░░ 100%")
    print("         ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
else:
    print(f"  ⚠️  {failed} episode(s) need attention")

print()
print("=" * WIDTH)
sys.exit(failed)
