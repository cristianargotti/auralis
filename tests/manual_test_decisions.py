
import sys
from pathlib import Path

# Add project root to path
sys.path.append("/Users/cristian.reyes/code/auralis")

from auralis.console.gap_analyzer import GapReport, StemGap, FreqGap
from auralis.console.stem_decisions import make_decisions, format_decisions_for_logs

# Mock data
mock_ear = {
    "bpm": 128.0,
    "key": "Cm",
    "scale": "minor",
    "duration": 180.0
}

# Create a mock GapReport with diverse scenarios
mock_gaps = {
    # Scenario 1: High quality -> KEEP
    "vocals": StemGap(
        stem_name="vocals",
        quality_score=85.0,
        rms_gap_db=-1.0,
        peak_gap_db=-0.5,
        energy_gap_pct=-2.0,
        dynamic_range_gap_db=1.0,
        freq_gaps=[],
        suggestions=[]
    ),
    
    # Scenario 2: Medium quality -> CORRECT (needs FX)
    "bass": StemGap(
        stem_name="bass",
        quality_score=65.0,
        rms_gap_db=-5.0,  # Needs sidechain/saturation
        peak_gap_db=-2.0,
        energy_gap_pct=-10.0,
        dynamic_range_gap_db=2.0,
        freq_gaps=[FreqGap(band="mid", ref_pct=50.0, your_pct=56.0, gap_pct=6.0, action="Cut mid")], # Muddy
        suggestions=[]
    ),
    
    # Scenario 3: Low quality -> ENHANCE (add support layer)
    "drums": StemGap(
        stem_name="drums",
        quality_score=40.0,
        rms_gap_db=-8.0,
        peak_gap_db=-4.0,
        energy_gap_pct=-15.0,
        dynamic_range_gap_db=4.0,
        freq_gaps=[],
        suggestions=["Weak transient response"]
    ),
    
    # Scenario 4: Very low quality -> REPLACE (regenerate)
    "other": StemGap(
        stem_name="other",
        quality_score=15.0,
        rms_gap_db=-12.0,
        peak_gap_db=-6.0,
        energy_gap_pct=-25.0,
        dynamic_range_gap_db=6.0,
        freq_gaps=[],
        suggestions=["Severe phasing", "Missing fundamental"]
    ),

    # Scenario 5: Harmful -> MUTE
    "artifacts": StemGap( # Using a dummy name, but structure valid
        stem_name="artifacts",
        quality_score=10.0,
        rms_gap_db=0.0,
        peak_gap_db=0.0,
        energy_gap_pct=30.0, # Way too loud
        dynamic_range_gap_db=0.0,
        freq_gaps=[],
        suggestions=["Extreme energy excess"]
    )
}

report = GapReport(
    reference_count=5,
    stem_gaps=mock_gaps,
    overall_score=43.0
)

# Run decision engine
print("Running Stem Decision Engine...")
decisions = make_decisions(report, mock_ear, bank=None)

# Print formatted logs
print("\n--- DECISION LOGS ---\n")
for line in format_decisions_for_logs(decisions):
    print(line)

print("\n--- JSON OUTPUT ---")
import json
print(json.dumps(decisions.to_dict(), indent=2))
