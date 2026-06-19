#!/usr/bin/env python3
"""
waveform.py
--------------------
Produces a comparative waveform diagram for two .wav files.

Usage:
    python compare_waveforms.py file_a.wav file_b.wav
    python compare_waveforms.py file_a.wav file_b.wav --output comparison.png
    python compare_waveforms.py file_a.wav file_b.wav --labels "Original" "Separated" --color1 "#4C9BE8" --color2 "#E8824C"

Requires: numpy, librosa, matplotlib, soundfile
    pip install numpy librosa matplotlib soundfile
"""

import argparse
import sys
import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_time(x, _):
    """Format x-axis ticks as mm:ss.s"""
    mins = int(x // 60)
    secs = x % 60
    return f"{mins}:{secs:04.1f}"


def load_audio(path):
    """Load audio file, returning (samples, sample_rate, duration, n_channels)."""
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, samples)
    duration = y.shape[1] / sr
    return y, sr, duration, y.shape[0]


def rms_db(y):
    """Compute per-frame RMS in dB."""
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db_vals = librosa.amplitude_to_db(rms, ref=np.max)
    times = librosa.frames_to_time(
        np.arange(len(rms_db_vals)), sr=len(y) // len(rms_db_vals) * 100, hop_length=hop_length
    )
    return rms_db_vals


def compute_rms_envelope(y, sr, hop_length=512):
    """RMS envelope with correct time axis."""
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    return times, librosa.amplitude_to_db(rms + 1e-9, ref=1.0)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_comparison(
    path_a, path_b,
    label_a="File A", label_b="File B",
    color_a="#4C9BE8", color_b="#E8824C",
    output_path="waveform_comparison.png",
    dpi=150,
):
    ya, sr_a, dur_a, ch_a = load_audio(path_a)
    yb, sr_b, dur_b, ch_b = load_audio(path_b)

    # Mix to mono for waveform and RMS (keep originals for stats)
    mono_a = ya.mean(axis=0)
    mono_b = yb.mean(axis=0)

    time_a = np.linspace(0, dur_a, len(mono_a))
    time_b = np.linspace(0, dur_b, len(mono_b))

    rms_t_a, rms_db_a = compute_rms_envelope(mono_a, sr_a)
    rms_t_b, rms_db_b = compute_rms_envelope(mono_b, sr_b)

    max_dur = max(dur_a, dur_b)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor="#0F1117")
    fig.suptitle(
        "Waveform Comparison",
        color="white", fontsize=16, fontweight="bold", y=0.97
    )

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.07, right=0.97, top=0.92, bottom=0.07
    )

    ax_wa = fig.add_subplot(gs[0, 0])   # waveform A
    ax_wb = fig.add_subplot(gs[0, 1])   # waveform B
    ax_ra = fig.add_subplot(gs[1, 0])   # RMS A
    ax_rb = fig.add_subplot(gs[1, 1])   # RMS B
    ax_ov = fig.add_subplot(gs[2, :])   # overlay waveforms
    ax_st = fig.add_subplot(gs[3, :])   # stats table

    panel_bg   = "#1A1D27"
    grid_color = "#2A2D3A"
    text_color = "#C8CDD8"

    def style_ax(ax, title, color):
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines[:].set_color(grid_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.set_title(title, color=color, fontsize=10, fontweight="semibold", pad=6)
        ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(FuncFormatter(format_time))

    # ── Waveform panels ────────────────────────────────────────────────────────
    # Downsample for plotting performance
    def downsample(arr, n=80_000):
        if len(arr) > n:
            step = len(arr) // n
            return arr[::step]
        return arr

    ds_mono_a = downsample(mono_a)
    ds_time_a = np.linspace(0, dur_a, len(ds_mono_a))
    ds_mono_b = downsample(mono_b)
    ds_time_b = np.linspace(0, dur_b, len(ds_mono_b))

    for ax, t, y, lbl, col in [
        (ax_wa, ds_time_a, ds_mono_a, label_a, color_a),
        (ax_wb, ds_time_b, ds_mono_b, label_b, color_b),
    ]:
        ax.fill_between(t, y, alpha=0.55, color=col)
        ax.plot(t, y, linewidth=0.4, color=col, alpha=0.85)
        style_ax(ax, f"{lbl} — Waveform", col)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_xlim(0, max_dur)
        peak = np.max(np.abs(y))
        ax.set_ylim(-peak * 1.15, peak * 1.15)

    # ── RMS panels ─────────────────────────────────────────────────────────────
    for ax, t, r, lbl, col in [
        (ax_ra, rms_t_a, rms_db_a, label_a, color_a),
        (ax_rb, rms_t_b, rms_db_b, label_b, color_b),
    ]:
        ax.fill_between(t, r, alpha=0.4, color=col)
        ax.plot(t, r, linewidth=0.9, color=col)
        style_ax(ax, f"{lbl} — RMS Envelope (dB)", col)
        ax.set_ylabel("dBFS", fontsize=8)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_xlim(0, max_dur)

    # ── Overlay ────────────────────────────────────────────────────────────────
    ax_ov.fill_between(ds_time_a, ds_mono_a, alpha=0.35, color=color_a, label=label_a)
    ax_ov.plot(ds_time_a, ds_mono_a, linewidth=0.5, color=color_a, alpha=0.8)
    ax_ov.fill_between(ds_time_b, ds_mono_b, alpha=0.35, color=color_b, label=label_b)
    ax_ov.plot(ds_time_b, ds_mono_b, linewidth=0.5, color=color_b, alpha=0.8)
    style_ax(ax_ov, "Overlay", "white")
    ax_ov.set_ylabel("Amplitude", fontsize=8, color=text_color)
    ax_ov.set_xlabel("Time", fontsize=8, color=text_color)
    ax_ov.set_xlim(0, max_dur)
    leg = ax_ov.legend(
        facecolor=panel_bg, edgecolor=grid_color,
        labelcolor=text_color, fontsize=9
    )

    # ── Stats table ────────────────────────────────────────────────────────────
    ax_st.set_facecolor(panel_bg)
    ax_st.axis("off")

    def peak_db(y):
        p = np.max(np.abs(y))
        return 20 * np.log10(p + 1e-9)

    def rms_full(y):
        return 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)

    def crest(y):
        return peak_db(y) - rms_full(y)

    stats = [
        ("Duration (s)",       f"{dur_a:.3f}",                   f"{dur_b:.3f}"),
        ("Sample Rate (Hz)",   f"{sr_a:,}",                      f"{sr_b:,}"),
        ("Channels",           str(ch_a),                        str(ch_b)),
        ("Peak (dBFS)",        f"{peak_db(mono_a):.2f}",         f"{peak_db(mono_b):.2f}"),
        ("RMS (dBFS)",         f"{rms_full(mono_a):.2f}",        f"{rms_full(mono_b):.2f}"),
        ("Crest Factor (dB)",  f"{crest(mono_a):.2f}",           f"{crest(mono_b):.2f}"),
        ("DC Offset",          f"{np.mean(mono_a):.6f}",         f"{np.mean(mono_b):.6f}"),
    ]

    col_labels = ["Metric", label_a, label_b]
    cell_text  = stats

    tbl = ax_st.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    header_color = "#2A3A5A"
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(header_color if row == 0 else panel_bg)
        cell.set_text_props(
            color=color_a if (col == 1 and row > 0) else
                  color_b if (col == 2 and row > 0) else
                  text_color
        )
        cell.set_edgecolor(grid_color)

    ax_st.set_title(
        "Audio Statistics", color="white", fontsize=10,
        fontweight="semibold", pad=8, loc="left"
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare waveforms of two WAV files and save a diagram."
    )
    parser.add_argument("file_a", help="Path to first WAV file")
    parser.add_argument("file_b", help="Path to second WAV file")
    parser.add_argument(
        "--labels", nargs=2, default=["Original Mix", "Accessible Mix"],
        metavar=("LABEL_A", "LABEL_B"),
        help="Display labels for the two files"
    )
    parser.add_argument(
        "--output", default="waveform_comparison.png",
        help="Output image path (default: waveform_comparison.png)"
    )
    parser.add_argument(
        "--color1", default="#4C9BE8",
        help="Hex colour for file A (default: #4C9BE8)"
    )
    parser.add_argument(
        "--color2", default="#E8824C",
        help="Hex colour for file B (default: #E8824C)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI (default: 150)"
    )
    args = parser.parse_args()

    plot_comparison(
        path_a=args.file_a,
        path_b=args.file_b,
        label_a=args.labels[0],
        label_b=args.labels[1],
        color_a=args.color1,
        color_b=args.color2,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
