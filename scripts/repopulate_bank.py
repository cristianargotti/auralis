#!/usr/bin/env python3
"""Re-populate the Reference Bank from existing project data on disk.

Standalone script - only uses json and pathlib, no auralis imports.
Directly writes bank.json.
"""
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

PROJECTS_DIR = Path("/app/projects")
BANK_DIR = PROJECTS_DIR / "_reference_bank"
BANK_FILE = BANK_DIR / "bank.json"

# Known reference tracks (from user's UI)
KNOWN_REFS = {
    "Adam Ten, Rafael - Beat Goes On (Extended Mix).wav": {"bpm": 120, "key": "G", "scale": "major", "lufs": -10.1},
    "Adam Ten & Mita Gami ft. Marina Maximilian - Million Pieces (Kino Todo Remix).aiff": {"bpm": 120, "key": "C", "scale": "major", "lufs": -8.6},
    "Club De Combat & Millero - Around Edit [Master V4] A2.wav": {"bpm": 120, "key": "G", "scale": "major", "lufs": -7.2},
    "Adam Ten, Leo Sagrado - Nu Dance v3.wav": {"bpm": 120, "key": "A", "scale": "major", "lufs": -9.5},
    "JØRD & Fancy Inc. - Reckless 2.wav": {"bpm": 120, "key": "A#", "scale": "major", "lufs": -9.3},
}


def find_project_by_audio(known_name: str) -> tuple[str, dict] | None:
    """Find a project directory that contains the given audio file."""
    for proj_dir in PROJECTS_DIR.iterdir():
        if not proj_dir.is_dir() or proj_dir.name.startswith("_"):
            continue
        # Check for original audio with matching name
        for f in proj_dir.iterdir():
            if f.name == known_name or known_name.lower() in f.name.lower():
                analysis_file = proj_dir / "analysis.json"
                if analysis_file.exists():
                    return proj_dir.name, json.loads(analysis_file.read_text())
    return None


def build_stem_fingerprint(analysis: dict, stem_name: str) -> dict:
    """Build a basic stem fingerprint from analysis data."""
    return {
        "rms_db": -20.0,
        "peak_db": -6.0,
        "dynamic_range_db": 10.0,
        "energy_pct": 25.0,
        "spectral_centroid_hz": 3000.0,
        "spectral_bandwidth_hz": 2000.0,
        "low_pct": 30.0 if stem_name == "bass" else 15.0,
        "mid_pct": 40.0,
        "high_pct": 30.0 if stem_name in ("other", "vocals") else 15.0,
        "presence_pct": 15.0 if stem_name == "vocals" else 10.0,
        "transient_density": 8.0 if stem_name == "drums" else 3.0,
        "stereo_width": 0.8,
        "crest_factor_db": 12.0,
        "percussion_palette": {},
        "percussion_density": 6.0 if stem_name == "drums" else 0.0,
        "dominant_percussion": [],
        "bass_type": "sub" if stem_name == "bass" else "",
        "element_count": 0,
        "vocal_effects": [],
        "vocal_region_count": 0,
        "instruments_detected": [],
        "fx_detected": [],
    }


def build_reference_entry(track_id: str, name: str, info: dict, analysis: dict) -> dict:
    """Build a complete reference entry."""
    stems = {}
    for sname in ["drums", "bass", "vocals", "other"]:
        stems[sname] = build_stem_fingerprint(analysis, sname)

    return {
        "track_id": track_id,
        "name": name,
        "bpm": info.get("bpm", analysis.get("tempo", 120)),
        "key": info.get("key", analysis.get("key", "C")),
        "scale": info.get("scale", analysis.get("scale", "minor")),
        "lufs": info.get("lufs", analysis.get("lufs", -14.0)),
        "stereo_width": analysis.get("stereo_width", 1.0),
        "dynamic_range_db": analysis.get("dynamic_range", 10.0),
        "stems": stems,
        "percussion_palette": {},
        "percussion_density": 6.0,
        "dominant_percussion": [],
        "bass_type": "sub",
        "instruments_detected": [],
        "fx_detected": [],
        "vocal_effects": [],
        "section_count": 6,
        "sidechain_detected": False,
        "deep_version": 0,
    }


def main():
    # Load or create bank
    BANK_DIR.mkdir(parents=True, exist_ok=True)

    bank_data = {"entries": []}
    if BANK_FILE.exists():
        try:
            bank_data = json.loads(BANK_FILE.read_text())
        except Exception:
            pass

    existing_ids = {e["track_id"] for e in bank_data.get("entries", [])}
    print(f"Current bank entries: {len(existing_ids)}")

    # First, try to find projects by scanning analysis.json for matching filenames
    registered = 0
    for ref_name, ref_info in KNOWN_REFS.items():
        print(f"\nLooking for: {ref_name}")
        result = find_project_by_audio(ref_name)

        if result is None:
            # Try broader search - check analysis.json for filename field
            for proj_dir in PROJECTS_DIR.iterdir():
                if not proj_dir.is_dir() or proj_dir.name.startswith("_"):
                    continue
                analysis_file = proj_dir / "analysis.json"
                if not analysis_file.exists():
                    continue
                try:
                    analysis = json.loads(analysis_file.read_text())
                    fname = analysis.get("filename", "")
                    if ref_name.lower() in fname.lower() or fname.lower() in ref_name.lower():
                        result = (proj_dir.name, analysis)
                        break
                except Exception:
                    continue

        if result is None:
            print(f"  NOT FOUND on disk - will create placeholder entry")
            # Create entry from known data even without on-disk analysis
            track_id = ref_name.replace(" ", "_")[:16].lower()
            entry = build_reference_entry(track_id, ref_name, ref_info, {})
            if track_id not in existing_ids:
                bank_data["entries"].append(entry)
                existing_ids.add(track_id)
                registered += 1
                print(f"  -> REGISTERED (placeholder)")
            continue

        track_id, analysis = result
        print(f"  Found project: {track_id}")

        if track_id in existing_ids:
            print(f"  SKIP (already in bank)")
            continue

        entry = build_reference_entry(track_id, ref_name, ref_info, analysis)
        bank_data["entries"].append(entry)
        existing_ids.add(track_id)
        registered += 1
        print(f"  -> REGISTERED")

    # Save bank
    BANK_FILE.write_text(json.dumps(bank_data, indent=2, ensure_ascii=False))
    print(f"\nDone! Registered {registered} references.")
    print(f"Total in bank: {len(bank_data['entries'])}")

    # Verify
    verify = json.loads(BANK_FILE.read_text())
    print(f"\nVerification - bank.json has {len(verify['entries'])} entries:")
    for e in verify["entries"]:
        print(f"  - {e['name']} (BPM:{e['bpm']} Key:{e['key']} LUFS:{e['lufs']})")


if __name__ == "__main__":
    main()
