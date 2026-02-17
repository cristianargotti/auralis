# Auralis Intelligent Stem Decision Engine (Phase 4)

This document details the implementation of the Intelligent Stem Decision Engine, which empowers Auralis to make sophisticated decisions about whether to Keep, Correct, Enhance, Replace, or Mute individual stems based on gap analysis against a reference DNA bank.

## 1. Overview

The engine acts as a decision layer between the **EAR** (Analysis) and **HANDS** (Mixing) stages. Instead of blindly processing every stem, it evaluates the quality of each stem relative to professional reference tracks and chooses the optimal reconstruction strategy.

### Decision Matrix

| Action | Condition | Strategy |
| :--- | :--- | :--- |
| **KEEP** | Score ≥ 80 | Apply standard reference-targeted recipe. Minimal intervention. |
| **CORRECT** | Score 50-79 | Apply aggressive recipe + **Smart FX** (see below). |
| **ENHANCE** | Score 25-49 | Keep original stem + layer a generated support stem (Synth or Organic). |
| **REPLACE** | Score < 25 | **Mute** original stem + generate a complete replacement. |
| **MUTE** | Harmful | Remove stems that actively degrade the mix (artifacts, extreme energy). |

---

## 2. Core Components

### A. Decision Logic (`auralis.console.stem_decisions`)

*   **Input:** `GapReport` from `gap_analyzer`.
*   **Output:** `DecisionReport` containing a `StemDecision` for each stem.
*   **Smart FX:** Automatically selects additional effects based on specific gaps:
    *   **Muddy Mids (>5%):** Adds `de_mud_eq` (-3dB @ 400Hz).
    *   **Missing Highs (<-5%):** Adds `air_boost` (+2.5dB @ 12kHz).
    *   **Narrow Stereo (< -8%):** Adds `chorus`.
    *   **Low Dynamics (> 4dB):** Adds `compress_harder`.
    *   **Weak Bass (< -4dB RMS):** Adds `sidechain` pumping.
    *   **Thin Sound (< -6dB RMS):** Adds `saturation`.

### B. Hybrid Generator (`auralis.console.stem_generator`)

Generates new audio content to support (Enhance) or replace (Replace) user stems. It uses a **Hybrid Approach**:

1.  **Organic Drums (One-Shot Extraction):**
    *   Locates the closest BMP-matched reference track in the bank.
    *   Extracts individual drum hits (kicks, snares, hats) from the reference *drum stem* using onset detection.
    *   **Re-sequences** these hits into a new, original pattern (e.g., "four_on_floor", "breakbeat") based on the user's track BPM.
    *   *Result:* Authentic, professional drum sounds in a new arrangement.

2.  **Tonal Synthesis (Bass, Pads, Leads):**
    *   Uses the Auralis **Grid** (MIDI) and **Hands** (Synth) engines.
    *   Generates MIDI patterns (basslines, chord progressions) in the user's Key and Scale.
    *   Synthesizes audio using curated presets (e.g., `bass_808`, `acid_303`, `pad_warm`, `supersaw`).
    *   *Result:* Clean, key-aligned tonal elements that fit the mix.

### C. Pipeline Integration (`auralis.api.routes.reconstruct`)

The engine is wired into the **Console** stage of the reconstruction pipeline (`_run_reconstruction`):

1.  **Analyze:** `gap_analyzer` produces a `GapReport`.
2.  **Decide:** `stem_decisions.make_decisions()` evaluates the report.
3.  **Generate:** For `ENHANCE` and `REPLACE` actions, `stem_generator` creates new audio files in `generated_stems/`.
4.  **Mute:** For `REPLACE` and `MUTE` actions, original stems are removed from the mix list.
5.  **Mix:** The `Mixer` receives the `StemDecision` data to apply **Smart FX** to the effect chains of active stems.

---

## 3. Usage

The system operates automatically during the reconstruction process if reference tracks exist in the bank.

**To enable:**
1.  Upload at least one reference track via `/api/reference/add`.
2.  Run a reconstruction job. The logs will show the decision process:

```text
── STEM DECISIONS ──────────────────────
  🥁 drums    → 🔄 REPLACE  (18/100)
     Quality 18/100 — replacing with generated drums_fm (four_on_floor)
     GEN: drums_fm pattern=four_on_floor

  🎸 bass     → 🔧 CORRECT  (55/100)
     Quality 55/100 — aggressive recipe + saturation, sidechain
     FX: saturation, sidechain

  🎹 other    → ✅ KEEP     (82/100)
     Quality 82/100 — ref-targeted recipe only
```

## 4. Future Improvements

*   **Deep Learning Extraction:** Replace onset-based one-shot extraction with a neural model (e.g., DOSE) for cleaner separation.
*   **Advanced Generation:** Integrate ACE-Step or Stable Audio for more complex texture generation.
*   **User Overrides:** Allow users to force specific decisions (e.g., "Always Replace Drums") via API parameters.
