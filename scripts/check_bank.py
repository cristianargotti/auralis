import json

with open("/app/projects/_reference_bank/bank.json") as f:
    d = json.load(f)

entries = d.get("entries", [])
print("Total references:", len(entries))
for e in entries:
    name = e.get("name", "?")
    bpm = e.get("bpm", "?")
    key = e.get("key", "?")
    scale = e.get("scale", "?")
    lufs = e.get("lufs", "?")
    dv = e.get("deep_version", 0)
    perc = len(e.get("percussion_palette", {}))
    stems = list(e.get("stems", {}).keys())
    print(f"  - {name}")
    print(f"    BPM:{bpm} Key:{key} {scale} LUFS:{lufs} Deep:v{dv} Perc:{perc} Stems:{stems}")
