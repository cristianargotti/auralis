#!/usr/bin/env python3
"""AURALIS Readiness Audit — verifies the full intelligence chain."""

import importlib
import inspect
import sys
from pathlib import Path

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition


# ══════════════════════════════════════════════════════
# 1. MODULE IMPORTS
# ══════════════════════════════════════════════════════
print("\n═══ 1. MODULE IMPORTS ═══")

modules = {
    "dna_brain": "auralis.console.dna_brain",
    "stem_decisions": "auralis.console.stem_decisions",
    "stem_recipes": "auralis.console.stem_recipes",
    "stem_generator": "auralis.console.stem_generator",
    "mastering": "auralis.console.mastering",
    "auto_correct": "auralis.console.auto_correct",
    "organic_pack": "auralis.console.organic_pack",
}

loaded = {}
for name, mod_path in modules.items():
    try:
        loaded[name] = importlib.import_module(mod_path)
        check(f"Import {name}", True)
    except Exception as e:
        check(f"Import {name}", False, str(e))

# ══════════════════════════════════════════════════════
# 2. DATA STRUCTURES — StemPlan has all fields
# ══════════════════════════════════════════════════════
print("\n═══ 2. STEMPLAN FIELDS ═══")

brain = loaded.get("dna_brain")
if brain:
    sp = brain.StemPlan(stem_name="test")
    required_fields = [
        "stem_name", "patch", "style", "fx_chain", "eq_adjustments",
        "compression", "volume_db", "sidechain", "sidechain_depth",
        "reverb_wet", "delay_wet", "saturation_drive", "stereo_width",
        "use_organic", "organic_category",
        "ai_prompt_hints", "reasoning", "confidence",
    ]
    for f in required_fields:
        check(f"StemPlan.{f}", hasattr(sp, f))

    # Check to_dict includes all
    d = sp.to_dict()
    for f in required_fields:
        check(f"to_dict has '{f}'", f in d)

# ══════════════════════════════════════════════════════
# 3. BRAIN INTELLIGENCE — all thinkers produce StemPlans
# ══════════════════════════════════════════════════════
print("\n═══ 3. BRAIN THINKERS ═══")

if brain:
    # Create minimal evidence
    ev = brain._gather_evidence(None, {"tempo": 125, "key": "C", "scale": "minor"})

    # Test all thinkers
    drums = brain._think_drums(ev)
    check("_think_drums() returns StemPlan", isinstance(drums, brain.StemPlan))
    check("drums has style", bool(drums.style), drums.style)
    check("drums has compression", bool(drums.compression))

    bass = brain._think_bass(ev)
    check("_think_bass() returns StemPlan", isinstance(bass, brain.StemPlan))
    check("bass has patch", bool(bass.patch), bass.patch)

    vocals = brain._think_vocals(ev)
    check("_think_vocals() returns StemPlan", isinstance(vocals, brain.StemPlan))

    other = brain._think_other(ev)
    check("_think_other() returns StemPlan", isinstance(other, brain.StemPlan))
    check("other has patch", bool(other.patch), other.patch)
    check("other has delay_wet", other.delay_wet >= 0)
    check("other has saturation_drive", other.saturation_drive > 0)

    master = brain._think_master(ev)
    check("_think_master() returns MasterPlan", isinstance(master, brain.MasterPlan))
    check("master has target_lufs", master.target_lufs < 0)

    # Test organic detection with evidence
    ev_organic = brain._gather_evidence(
        {"percussion": {"palette": {"Conga": 5, "Shaker": 3}, "dominant": ["Conga", "Shaker"], "avg_density": 4.0}},
        {"tempo": 128, "key": "Am", "scale": "minor"},
    )
    drums_organic = brain._think_drums(ev_organic)
    check("🌿 Organic detection (conga in evidence)", drums_organic.use_organic, drums_organic.organic_category)

# ══════════════════════════════════════════════════════
# 4. ORGANIC PACK
# ══════════════════════════════════════════════════════
print("\n═══ 4. ORGANIC PACK ═══")

op = loaded.get("organic_pack")
if op:
    pack = op.OrganicPack.load()
    check("OrganicPack loads", len(pack.samples) > 0, f"{len(pack.samples)} samples")
    check("Has congas", pack.has_organic("conga"))
    check("Has shakers", pack.has_organic("shaker"))
    check("Has atmosphere", pack.has_organic("atmosphere"))
    check("No kicks (AI handles)", not pack.has_organic("kick"))
    check("No bass (AI handles)", not pack.has_organic("bass"))
    check("No snare (AI handles)", not pack.has_organic("snare"))

    conga = pack.find_best("conga", bpm=128)
    check("find_best conga @ 128bpm", conga is not None, conga.filename if conga else "")
    if conga:
        check("Conga file exists on disk", conga.exists)

# ══════════════════════════════════════════════════════
# 5. RECIPE BUILDER — brain params used
# ══════════════════════════════════════════════════════
print("\n═══ 5. RECIPE BUILDER ═══")

recipes = loaded.get("stem_recipes")
if recipes:
    # Check build_recipe_for_stem accepts stem_plan
    sig = inspect.signature(recipes.build_recipe_for_stem)
    check("build_recipe_for_stem has 'stem_plan' param", "stem_plan" in sig.parameters)

    # Check drum recipe uses brain reverb
    src = inspect.getsource(recipes.build_drum_recipe)
    check("Drums reverb from brain", "stem_plan" in src and "reverb_wet" in src)

    # Check other recipe uses brain pan/sends
    src_other = inspect.getsource(recipes.build_other_recipe)
    check("Other pan from brain (stereo_width)", "stereo_width" in src_other)
    check("Other sends from brain (reverb_wet)", "reverb_wet" in src_other)

# ══════════════════════════════════════════════════════
# 6. STEM GENERATOR — organic priority chain
# ══════════════════════════════════════════════════════
print("\n═══ 6. STEM GENERATOR ═══")

gen = loaded.get("stem_generator")
if gen:
    src = inspect.getsource(gen.generate_stem)
    check("Organic-first check exists", "use_organic" in src)
    check("OrganicPack imported in generator", "OrganicPack" in src)
    check("AI generation after organic", src.index("use_organic") < src.index("AI GENERATION"))
    check("generate_pad_stem has stem_plan", "stem_plan" in inspect.signature(gen.generate_pad_stem).parameters)

# ══════════════════════════════════════════════════════
# 7. MASTERING — brain-guided
# ══════════════════════════════════════════════════════
print("\n═══ 7. MASTERING ═══")

mastering = loaded.get("mastering")
if mastering:
    check("master_audio exists", hasattr(mastering, "master_audio"))
    sig = inspect.signature(mastering.master_audio)
    check("master_audio has 'config' param", "config" in sig.parameters)
    src_eq = inspect.getsource(mastering.apply_ms_eq)
    check("EQ brain-guided", "brain_plan" in src_eq)
    src_comp = inspect.getsource(mastering.apply_multiband_compression)
    check("Compression brain-guided", "brain_plan" in src_comp)

# ══════════════════════════════════════════════════════
# 8. AUTO-CORRECT — feedback loop
# ══════════════════════════════════════════════════════
print("\n═══ 8. AUTO-CORRECT ═══")

ac = loaded.get("auto_correct")
if ac:
    check("evaluate_and_correct exists", hasattr(ac, "evaluate_and_correct"))
    check("CorrectionReport exists", hasattr(ac, "CorrectionReport"))
    sig = inspect.signature(ac.evaluate_and_correct)
    check("Has deep_profile param", "deep_profile" in sig.parameters)

# ══════════════════════════════════════════════════════
# 9. RECONSTRUCT PIPELINE — all stages connected
# ══════════════════════════════════════════════════════
print("\n═══ 9. PIPELINE INTEGRATION ═══")

try:
    recon_src = Path("auralis/api/routes/reconstruct.py").read_text()
    check("brain_report initialized before Stage 4", "brain_report = None" in recon_src)
    check("brain_stem_plans passed to mix", "brain_stem_plans" in recon_src)
    check("_master key for bus config", '"_master"' in recon_src)
    check("Auto-correct called after mastering", "evaluate_and_correct" in recon_src)
    check("Corrections merged before re-master", "corrected_plan" in recon_src)
    check("Bus config from brain (brain_room)", "brain_room" in recon_src)
except Exception as e:
    check("Read reconstruct.py", False, str(e))

# ══════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
total = len(results)
print(f"AUDIT RESULT: {passed}/{total} passed, {failed} failed")

if failed == 0:
    print("🎯 SYSTEM IS READY — 100% intelligent pipeline")
else:
    print(f"⚠️  {failed} issues need attention:")
    for s, name, detail in results:
        if s == FAIL:
            print(f"   {FAIL} {name}: {detail}")

print("=" * 60)
