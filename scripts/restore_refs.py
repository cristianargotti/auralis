"""Restore DNA references from completed jobs."""
import json
import sys
sys.path.insert(0, "/app")

from auralis.ear.reference_bank import ReferenceBank
from pathlib import Path

projects_dir = Path("/app/projects")
bank = ReferenceBank(projects_dir)

with open(projects_dir / "jobs.json") as f:
    data = json.load(f)

completed = [
    (jid, j) for jid, j in data.items()
    if j.get("status") == "completed"
    and j.get("mode") == "full"
    and j.get("result", {}).get("analysis")
]

print(f"Found {len(completed)} completed jobs with analysis")
added = 0
for jid, j in completed:
    result = j.get("result", {})
    analysis = result.get("analysis", {})
    stem_analysis = result.get("stem_analysis", {})
    name = j.get("original_name", f"Track {jid[:8]}")
    try:
        bank.add_reference(
            track_id=jid,
            name=name,
            ear_analysis=analysis,
            stem_analysis=stem_analysis,
        )
        print(f"  + {name}")
        added += 1
    except Exception as e:
        print(f"  SKIP {name}: {e}")

print(f"\nDone: {added} references added ({bank.count()} total in bank)")
