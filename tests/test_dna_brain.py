"""Smoke test for DNABrain integration."""

from auralis.console.dna_brain import DNABrain
from auralis.console.auto_correct import evaluate_and_correct
from auralis.console.mastering import MasterConfig
from auralis.console.stem_decisions import make_decisions
from auralis.console.stem_recipes import build_recipe_for_stem

# Mock deep profile
DEEP = {
    "percussion": {
        "dominant_palette": "Electronic Tight",
        "hat_style": "Closed 16th",
        "kick_character": "Deep Sub",
        "swing_pct": 0,
        "confidence": 4,
    },
    "bass": {
        "dominant_type": "Sub Bass",
        "sustain": "Long",
        "sidechain": "Common",
        "confidence": 5,
    },
    "vocals": {"has_vocals": False, "confidence": 0},
    "instruments": {
        "palette": "Pads + Arps",
        "stereo_width": "Wide",
        "texture": "Atmospheric",
        "confidence": 3,
    },
    "master": {
        "lufs": -8.0,
        "dynamic_range_db": 6,
        "freq_balance": {"low": 40, "mid": 35, "high": 25},
        "stereo_width_correlation": 0.7,
    },
    "fx": {"sidechain_style": "Common"},
    "arrangement": {},
}

STEM_ANALYSIS = {
    "drums": {"rms_db": -18, "peak_db": -6, "freq_balance": {"low": 45, "mid": 30, "high": 25}},
    "bass": {"rms_db": -20, "peak_db": -8, "freq_balance": {"low": 60, "mid": 25, "high": 15}},
    "other": {"rms_db": -22, "peak_db": -10, "freq_balance": {"low": 20, "mid": 40, "high": 40}},
}

EAR = {"bpm": 126.0, "key": "Am", "scale": "minor"}


def test_brain_think():
    brain = DNABrain()
    report = brain.think(DEEP, STEM_ANALYSIS, {}, EAR)

    assert report.stem_plans, "Should have stem plans"
    assert report.master_plan is not None, "Should have master plan"
    assert len(report.reasoning_chain) > 0, "Should have reasoning"

    d = report.to_dict()
    assert "stem_plans" in d
    assert "master_plan" in d
    print(f"✅ Brain: {len(report.stem_plans)} plans, {len(report.reasoning_chain)} reasons")
    for line in report.reasoning_chain[:3]:
        print(f"   {line}")


def test_imports():
    assert "brain_plan" in MasterConfig.__dataclass_fields__
    assert "brain_report" in make_decisions.__code__.co_varnames
    assert "stem_plan" in build_recipe_for_stem.__code__.co_varnames
    print("✅ All signatures accept brain params")


if __name__ == "__main__":
    test_imports()
    test_brain_think()
    print("\n✅ All tests passed")
