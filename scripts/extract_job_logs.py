"""Show HANDS stage logs + full result dict keys."""
import json

JOB_ID = "fe87e9d8-6c70-4d5b-86f7-e788133470b9"

with open("/tmp/auralis_jobs.json") as f:
    jobs = json.load(f)

job = jobs.get(JOB_ID)
logs = job.get("logs", [])

print("=== ENTRIES 20-40 (HANDS stage area) ===")
for i in range(20, min(41, len(logs))):
    entry = logs[i]
    print(f"  [{i:03d}] [{entry.get('level')}] {entry.get('msg', '')}")

# Deep inspect the result dict
print()
print("=== FULL RESULT KEYS (recursive) ===")
result = job.get("result", {})

def show_keys(d, prefix=""):
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"  {prefix}{k}: dict({len(v)} keys)")
            if len(v) < 10:
                show_keys(v, prefix=f"  {prefix}")
        elif isinstance(v, list):
            print(f"  {prefix}{k}: list({len(v)} items)")
        else:
            val = str(v)
            if len(val) > 80:
                val = val[:80] + "..."
            print(f"  {prefix}{k}: {val}")

show_keys(result)
