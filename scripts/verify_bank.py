#!/usr/bin/env python3
"""Verify Reference Bank is working inside the container."""
import sys
sys.path.insert(0, "/app")

from auralis.ear.reference_bank import ReferenceBank
from auralis.config import settings

bank = ReferenceBank(settings.projects_dir)
print("=" * 50)
print("REFERENCE BANK VERIFICATION")
print("=" * 50)
print(f"Bank loaded: {bank.count()} references")
print()

for ref in bank.list_references():
    name = ref["name"][:55]
    key = ref["key"]
    lufs = ref["lufs"]
    deep = ref["deep"]
    print(f"  [{name}]")
    print(f"    Key: {key} | LUFS: {lufs} | Deep: {deep}")

profile = bank.get_deep_profile()
print()
print("DEEP PROFILE:")
print(f"  Refs: {profile['reference_count']}")
print(f"  Deep count: {profile['deep_count']}")
print(f"  Dominant key: {profile['dominant_key']}")
print(f"  Bass type: {profile['bass']['dominant_type']}")
perc_hits = profile["percussion"]["total_hits_across_refs"]
print(f"  Percussion hits: {perc_hits}")
instruments = profile["instruments"]["palette"][:5]
print(f"  Instruments: {instruments}")
sc = profile["arrangement"]["sidechain_ratio"]
print(f"  Sidechain ratio: {sc}")

avgs = bank.get_master_averages()
print()
print("MASTER AVERAGES:")
print(f"  LUFS: {avgs['lufs']}")
print(f"  BPM: {avgs['bpm']}")

print()
print("BRAIN CHECK:")
try:
    from auralis.console.dna_brain import DNABrain
    brain = DNABrain()
    print(f"  DNABrain imported OK")
    print(f"  Brain ready: True")
except Exception as e:
    print(f"  DNABrain error: {e}")

print()
if bank.count() >= 3:
    print("RESULT: AI WILL USE REFERENCES DURING RECONSTRUCT")
else:
    print("RESULT: NOT ENOUGH REFERENCES")
