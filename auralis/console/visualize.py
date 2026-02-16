"""AURALIS Console — High-resolution audio visualization.

Generates comparison images (PNG, 300dpi) for track QC and A/B analysis.
All open source: matplotlib + librosa + numpy + scipy.

Charts:
  - Spectrum comparison (7-band bar chart)
  - Full-resolution spectrogram comparison (mel-spectrogram)
  - Waveform comparison (original vs master)
  - QC radar (quality dimensions)
  - Loudness timeline (short-term LUFS over time)
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

import structlog

logger = structlog.get_logger()

# ── AURALIS Dark Theme ──────────────────────────────────
CYAN = "#22d3ee"
VIOLET = "#a78bfa"
EMERALD = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"
ROSE = "#fb7185"
BG = "#0a0a0f"
BG_CARD = "#111118"
GRID = "#1e1e2e"
TEXT = "#94a3b8"
WHITE = "#e2e8f0"

BAND_NAMES = ["Sub\n20-60", "Bass\n60-250", "Low Mid\n250-500",
              "Mid\n500-2k", "Upper Mid\n2k-4k", "Presence\n4k-8k", "Air\n8k-20k"]
BAND_COLORS_A = [CYAN] * 7
BAND_COLORS_B = [VIOLET] * 7

DPI = 300  # High-res output


def _setup_matplotlib():
    """Configure matplotlib with Agg backend and AURALIS theme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    })
    return plt


def _fig_to_bytes(fig, dpi: int = DPI) -> bytes:
    """Save figure to PNG bytes and close."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=BG, bbox_inches="tight",
                pad_inches=0.3)
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Spectrum Comparison ──────────────────────────────────

def generate_spectrum_comparison(
    track_a_path: str | Path,
    track_b_path: str | Path,
    label_a: str = "Original",
    label_b: str = "Master",
) -> bytes:
    """Generate a 7-band spectrum comparison bar chart (300dpi PNG)."""
    from auralis.console.qc import analyze_spectrum

    plt = _setup_matplotlib()

    data_a, sr_a = sf.read(str(track_a_path), dtype="float64")
    data_b, sr_b = sf.read(str(track_b_path), dtype="float64")
    spec_a = analyze_spectrum(data_a, sr_a)
    spec_b = analyze_spectrum(data_b, sr_b)

    bands_a = [spec_a.sub, spec_a.bass, spec_a.low_mid, spec_a.mid,
               spec_a.upper_mid, spec_a.presence, spec_a.brilliance]
    bands_b = [spec_b.sub, spec_b.bass, spec_b.low_mid, spec_b.mid,
               spec_b.upper_mid, spec_b.presence, spec_b.brilliance]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(BAND_NAMES))
    w = 0.35

    ax.bar(x - w / 2, bands_a, w, label=label_a, color=CYAN, alpha=0.85,
           edgecolor="none", zorder=3)
    ax.bar(x + w / 2, bands_b, w, label=label_b, color=VIOLET, alpha=0.85,
           edgecolor="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES, fontsize=8)
    ax.set_ylabel("Energy (dB)", fontsize=11)
    ax.set_title("7-Band Spectral Comparison", fontsize=16, color=WHITE,
                 fontweight="bold", pad=15)
    ax.legend(fontsize=10, facecolor=BG_CARD, edgecolor=GRID, labelcolor=TEXT,
              loc="upper right")
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5, zorder=0)

    # Difference annotations
    for i, (a, b) in enumerate(zip(bands_a, bands_b)):
        diff = a - b
        color = EMERALD if abs(diff) < 3 else AMBER if abs(diff) < 6 else RED
        ax.annotate(f"{diff:+.1f} dB", xy=(i, max(a, b) + 1.5), ha="center",
                    fontsize=7, color=color, fontweight="bold")

    return _fig_to_bytes(fig)


# ── Mel Spectrogram Comparison ───────────────────────────

def generate_spectrogram_comparison(
    track_a_path: str | Path,
    track_b_path: str | Path,
    label_a: str = "Original",
    label_b: str = "Master",
    n_mels: int = 128,
    fmax: int = 16000,
) -> bytes:
    """Generate side-by-side mel spectrograms (300dpi PNG).

    Uses librosa for mel-spectrogram computation (all open source).
    """
    plt = _setup_matplotlib()

    try:
        import librosa
        import librosa.display
    except ImportError:
        # Fallback: simple FFT spectrogram
        return _generate_fft_spectrogram_comparison(
            track_a_path, track_b_path, label_a, label_b
        )

    data_a, sr_a = sf.read(str(track_a_path), dtype="float32")
    data_b, sr_b = sf.read(str(track_b_path), dtype="float32")
    mono_a = np.mean(data_a, axis=1) if data_a.ndim == 2 else data_a
    mono_b = np.mean(data_b, axis=1) if data_b.ndim == 2 else data_b

    mel_a = librosa.feature.melspectrogram(y=mono_a, sr=sr_a, n_mels=n_mels, fmax=fmax)
    mel_b = librosa.feature.melspectrogram(y=mono_b, sr=sr_b, n_mels=n_mels, fmax=fmax)
    mel_a_db = librosa.power_to_db(mel_a, ref=np.max)
    mel_b_db = librosa.power_to_db(mel_b, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, mel_db, label, sr, color in [
        (ax1, mel_a_db, label_a, sr_a, "magma"),
        (ax2, mel_b_db, label_b, sr_b, "inferno"),
    ]:
        img = librosa.display.specshow(mel_db, sr=sr, fmax=fmax, x_axis="time",
                                        y_axis="mel", ax=ax, cmap=color)
        ax.set_title(label, fontsize=13, color=WHITE, fontweight="bold", pad=10)
        ax.set_ylabel("Frequency" if ax == ax1 else "", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02, aspect=30)

    fig.suptitle("Mel Spectrogram Comparison", fontsize=16, color=WHITE,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return _fig_to_bytes(fig)


def _generate_fft_spectrogram_comparison(
    track_a_path: str | Path,
    track_b_path: str | Path,
    label_a: str = "Original",
    label_b: str = "Master",
) -> bytes:
    """Fallback: scipy-based spectrogram when librosa is not installed."""
    from scipy import signal as scipy_signal

    plt = _setup_matplotlib()

    data_a, sr_a = sf.read(str(track_a_path), dtype="float64")
    data_b, sr_b = sf.read(str(track_b_path), dtype="float64")
    mono_a = np.mean(data_a, axis=1) if data_a.ndim == 2 else data_a
    mono_b = np.mean(data_b, axis=1) if data_b.ndim == 2 else data_b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, mono, sr, label in [
        (ax1, mono_a, sr_a, label_a),
        (ax2, mono_b, sr_b, label_b),
    ]:
        nperseg = min(4096, len(mono))
        f, t, sxx = scipy_signal.spectrogram(mono, sr, nperseg=nperseg,
                                              noverlap=nperseg // 2)
        mask = f <= 16000
        ax.pcolormesh(t, f[mask], 10 * np.log10(sxx[mask] + 1e-10),
                       shading="gouraud", cmap="magma", vmin=-80, vmax=0)
        ax.set_title(label, fontsize=13, color=WHITE, fontweight="bold")
        ax.set_ylabel("Frequency (Hz)" if ax == ax1 else "", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)

    fig.suptitle("Spectrogram Comparison", fontsize=16, color=WHITE,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── Waveform Comparison ─────────────────────────────────

def generate_waveform_comparison(
    track_a_path: str | Path,
    track_b_path: str | Path,
    label_a: str = "Original",
    label_b: str = "Master",
) -> bytes:
    """Generate stacked waveform comparison (300dpi PNG)."""
    plt = _setup_matplotlib()

    data_a, sr_a = sf.read(str(track_a_path), dtype="float64")
    data_b, sr_b = sf.read(str(track_b_path), dtype="float64")
    mono_a = np.mean(data_a, axis=1) if data_a.ndim == 2 else data_a
    mono_b = np.mean(data_b, axis=1) if data_b.ndim == 2 else data_b

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    for ax, mono, sr, label, color in [
        (ax1, mono_a, sr_a, label_a, CYAN),
        (ax2, mono_b, sr_b, label_b, VIOLET),
    ]:
        t = np.linspace(0, len(mono) / sr, len(mono))
        step = max(1, len(mono) // 10000)
        ax.fill_between(t[::step], mono[::step], color=color, alpha=0.6, linewidth=0)
        ax.plot(t[::step], mono[::step], color=color, alpha=0.3, linewidth=0.2)
        ax.set_ylabel(label, fontsize=10, color=WHITE)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color=GRID, linewidth=0.5)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax2.set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Waveform Comparison", fontsize=16, color=WHITE, fontweight="bold")
    plt.tight_layout()
    return _fig_to_bytes(fig)


# ── QC Radar ────────────────────────────────────────────

def generate_qc_radar(
    peak_db: float, rms_db: float, crest_db: float,
    stereo_corr: float, stereo_width: float,
    lufs: float, dynamic_range: float,
) -> bytes:
    """Generate a radar chart of QC quality dimensions (300dpi PNG)."""
    plt = _setup_matplotlib()

    labels = ["Peak\ndBTP", "RMS\ndB", "Crest\nFactor", "Stereo\nCorr",
              "Width", "LUFS", "Dynamic\nRange"]

    # Normalize to 0-1 (higher = better)
    values = [
        min(1, max(0, (-peak_db) / 3)),                # -3..0 → 1..0
        min(1, max(0, (rms_db + 20) / 14)),             # -20..-6 → 0..1
        min(1, max(0, crest_db / 20)),                  # 0..20 → 0..1
        min(1, max(0, stereo_corr)),                    # 0..1
        min(1, max(0, stereo_width / 0.8)),             # 0..0.8 → 0..1
        min(1, max(0, (lufs + 20) / 14)),               # -20..-6 → 0..1
        min(1, max(0, dynamic_range / 12)),             # 0..12 → 0..1
    ]
    values_closed = values + [values[0]]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, values_closed, color=CYAN, alpha=0.15)
    ax.plot(angles_closed, values_closed, color=CYAN, linewidth=2.5,
            marker="o", markersize=6, markeredgecolor=BG, markeredgewidth=1.5)

    # Score circles
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100,
                color=GRID, linewidth=0.5, alpha=0.7)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_title("Quality Dimensions", fontsize=16, color=WHITE,
                 fontweight="bold", pad=25)

    # Score label
    avg = np.mean(values) * 100
    color = EMERALD if avg >= 70 else AMBER if avg >= 50 else RED
    ax.text(0, 0, f"{avg:.0f}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=color)

    return _fig_to_bytes(fig)


# ── Loudness Timeline ───────────────────────────────────

def generate_loudness_timeline(
    track_path: str | Path,
    label: str = "Track",
    window_s: float = 0.4,
) -> bytes:
    """Generate short-term loudness (RMS) over time (300dpi PNG)."""
    plt = _setup_matplotlib()

    data, sr = sf.read(str(track_path), dtype="float64")
    mono = np.mean(data, axis=1) if data.ndim == 2 else data

    win = int(window_s * sr)
    hop = win // 2
    times, rms_vals = [], []

    for i in range(0, len(mono) - win, hop):
        chunk = mono[i:i + win]
        r = np.sqrt(np.mean(chunk ** 2))
        rms_db = 20 * np.log10(max(r, 1e-10))
        times.append((i + win / 2) / sr)
        rms_vals.append(rms_db)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(times, rms_vals, min(rms_vals), color=CYAN, alpha=0.3)
    ax.plot(times, rms_vals, color=CYAN, linewidth=1.2)

    # Target zones
    ax.axhspan(-16, -12, color=EMERALD, alpha=0.08, label="Streaming target")
    ax.axhspan(-10, -6, color=AMBER, alpha=0.08, label="Club target")

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("RMS (dB)", fontsize=10)
    ax.set_title(f"Loudness Timeline — {label}", fontsize=14, color=WHITE,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=GRID, labelcolor=TEXT,
              loc="lower right")
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.5)

    return _fig_to_bytes(fig)
