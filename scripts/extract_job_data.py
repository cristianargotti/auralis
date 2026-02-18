"""Extract brain reasoning, stem decisions and QC from Mono Aullador job."""
import json

JOB_ID = "fe87e9d8-6c70-4d5b-86f7-e788133470b9"

with open("/tmp/auralis_jobs.json") as f:
    jobs = json.load(f)

job = jobs.get(JOB_ID)
if not job:
    print(f"Job {JOB_ID} not found")
    exit(1)

r = job.get("result", {})
print(f"Title: {job.get('title', '?')}")
print(f"Status: {job.get('status')}")
print(f"Progress: {job.get('progress')}")
print()

# Brain report
br = r.get("brain_report", {})
if br:
    print("=== BRAIN REASONING CHAIN ===")
    for line in br.get("reasoning_chain", []):
        print(f"  {line}")
    print()
    print("=== STEM PLANS ===")
    for stem, plan in br.get("stem_plans", {}).items():
        print(f"\n  [{stem}]:")
        for k, v in plan.items():
            print(f"    {k}: {v}")
    mp = br.get("master_plan", {})
    if mp:
        print(f"\n  [MASTER PLAN]:")
        for k, v in mp.items():
            print(f"    {k}: {v}")
else:
    print("NO BRAIN REPORT FOUND")

# Stem decisions
sd = r.get("stem_decisions", {})
if sd:
    print()
    print("=== STEM DECISIONS ===")
    for stem, dec in sd.get("decisions", {}).items():
        print(f"\n  [{stem}]:")
        for k, v in dec.items():
            print(f"    {k}: {v}")
else:
    print("\nNO STEM DECISIONS FOUND")

# Gap report
gr = r.get("gap_report", {})
if gr:
    print()
    print("=== GAP REPORT ===")
    for k, v in gr.items():
        if k != "stem_gaps":
            print(f"  {k}: {v}")
    for stem, gap in gr.get("stem_gaps", {}).items():
        print(f"\n  [{stem}]:")
        for k, v in gap.items():
            print(f"    {k}: {v}")

# QC
qc = r.get("qc", {})
if qc:
    print()
    print("=== QC RESULTS ===")
    for k, v in qc.items():
        if k != "details":
            print(f"  {k}: {v}")
    details = qc.get("details", {})
    if details:
        for k, v in details.items():
            print(f"  {k}: {v}")

# Logs (last 30)
logs = job.get("log", [])
if logs:
    print()
    print("=== LAST 30 LOG ENTRIES ===")
    for entry in logs[-30:]:
        print(f"  {entry.get('message', '')}")
