#!/usr/bin/env python3

import argparse
import os
import sys

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_audio(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    y, sr = librosa.load(path, sr=None, mono=False)

    if y.ndim == 1:
        y = y[np.newaxis, :]

    return y, sr


def mono(y):
    return y.mean(axis=0)


def downsample(y, target=80000):
    if len(y) <= target:
        return y

    step = len(y) // target
    return y[::step]


def rms_envelope(y, sr):
    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=512
    )[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=512
    )

    rms_db = librosa.amplitude_to_db(
        rms + 1e-9,
        ref=1.0
    )

    return times, rms_db


def peak_db(y):
    return 20 * np.log10(
        np.max(np.abs(y)) + 1e-9
    )


def rms_db_full(y):
    return 20 * np.log10(
        np.sqrt(np.mean(y**2)) + 1e-9
    )


def crest_factor(y):
    return peak_db(y) - rms_db_full(y)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dsp", required=True)
    parser.add_argument("--generative", required=True)
    parser.add_argument("--combined", required=True)

    parser.add_argument(
        "--output",
        default="all_conditions_waveforms.png"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200
    )

    args = parser.parse_args()

    files = {
        "Baseline": args.baseline,
        "DSP": args.dsp,
        "Generative": args.generative,
        "Combined": args.combined,
    }

    colors = {
        "Baseline": "#B0B0B0",
        "DSP": "#4C9BE8",
        "Generative": "#E8824C",
        "Combined": "#6FCF97",
    }

    audio = {}

    for name, path in files.items():

        y, sr = load_audio(path)

        audio[name] = {
            "sr": sr,
            "mono": mono(y),
            "duration": y.shape[-1] / sr,
            "channels": y.shape[0],
        }

    max_duration = max(
        x["duration"] for x in audio.values()
    )

    # --------------------------------------------------------
    # Figure Layout
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(16, 22),
        facecolor="#0F1117"
    )

    fig.suptitle(
        "EEEM004 Audio Processing Comparison",
        color="white",
        fontsize=18,
        fontweight="bold"
    )

    gs = gridspec.GridSpec(
        10,
        1,
        figure=fig,
        hspace=0.55
    )

    panel_bg = "#1A1D27"
    grid_color = "#2A2D3A"
    text_color = "#C8CDD8"

    def style_ax(ax, title, color="white"):

        ax.set_facecolor(panel_bg)

        ax.tick_params(
            colors=text_color,
            labelsize=8
        )

        for spine in ax.spines.values():
            spine.set_color(grid_color)

        ax.grid(
            True,
            color=grid_color,
            alpha=0.5
        )

        ax.set_title(
            title,
            color=color,
            fontsize=10,
            fontweight="bold"
        )

    # --------------------------------------------------------
    # Waveforms
    # --------------------------------------------------------

    waveform_axes = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[2]),
        fig.add_subplot(gs[3]),
    ]

    for ax, name in zip(
        waveform_axes,
        ["Baseline", "DSP", "Generative", "Combined"]
    ):

        y = audio[name]["mono"]

        ds = downsample(y)

        t = np.linspace(
            0,
            audio[name]["duration"],
            len(ds)
        )

        ax.fill_between(
            t,
            ds,
            alpha=0.5,
            color=colors[name]
        )

        ax.plot(
            t,
            ds,
            linewidth=0.4,
            color=colors[name]
        )

        style_ax(
            ax,
            f"{name} Waveform",
            colors[name]
        )

        ax.set_xlim(0, max_duration)

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    ax_rms = fig.add_subplot(gs[4])

    for name in audio:

        t, rms = rms_envelope(
            audio[name]["mono"],
            audio[name]["sr"]
        )

        ax_rms.plot(
            t,
            rms,
            label=name,
            color=colors[name]
        )

    style_ax(
        ax_rms,
        "RMS Envelope Comparison"
    )

    ax_rms.legend()

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    ax_overlay = fig.add_subplot(gs[5])

    for name in audio:

        y = downsample(audio[name]["mono"])

        t = np.linspace(
            0,
            audio[name]["duration"],
            len(y)
        )

        ax_overlay.plot(
            t,
            y,
            linewidth=0.5,
            alpha=0.7,
            color=colors[name],
            label=name
        )

    style_ax(
        ax_overlay,
        "Waveform Overlay"
    )

    ax_overlay.legend()

    # --------------------------------------------------------
    # Difference Waveforms
    # --------------------------------------------------------

    baseline = audio["Baseline"]["mono"]

    diff_axes = [
        fig.add_subplot(gs[6]),
        fig.add_subplot(gs[7]),
        fig.add_subplot(gs[8]),
    ]

    diff_names = [
        "DSP",
        "Generative",
        "Combined"
    ]

    for ax, name in zip(diff_axes, diff_names):

        processed = audio[name]["mono"]

        length = min(
            len(processed),
            len(baseline)
        )

        diff = (
            processed[:length]
            - baseline[:length]
        )

        diff_ds = downsample(diff)

        t = np.linspace(
            0,
            length / audio[name]["sr"],
            len(diff_ds)
        )

        ax.plot(
            t,
            diff_ds,
            color=colors[name],
            linewidth=0.5
        )

        ax.fill_between(
            t,
            diff_ds,
            alpha=0.4,
            color=colors[name]
        )

        style_ax(
            ax,
            f"{name} − Baseline Difference",
            colors[name]
        )

    # --------------------------------------------------------
    # Statistics Table
    # --------------------------------------------------------

    ax_stats = fig.add_subplot(gs[9])

    ax_stats.axis("off")

    stats = []

    for name in [
        "Baseline",
        "DSP",
        "Generative",
        "Combined"
    ]:

        y = audio[name]["mono"]

        stats.append([
            name,
            f"{audio[name]['duration']:.2f}",
            f"{audio[name]['sr']}",
            f"{peak_db(y):.2f}",
            f"{rms_db_full(y):.2f}",
            f"{crest_factor(y):.2f}",
        ])

    table = ax_stats.table(
        cellText=stats,
        colLabels=[
            "Condition",
            "Duration(s)",
            "Sample Rate",
            "Peak(dBFS)",
            "RMS(dBFS)",
            "Crest(dB)"
        ],
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)

    plt.savefig(
        args.output,
        dpi=args.dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
