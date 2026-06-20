"""
EEEM004 — Single-Clip End-to-End Runner
=========================================
Louis Ilett | Supervisor: Prof Philip Jackson | University of Surrey

Runs ONE test clip through all four evaluation conditions
(baseline / dsp_only / generative / combined) and then computes
objective metrics comparing them — in one command.

This exists so you can validate the whole chain (separation → personalisation
→ evaluation) end-to-end on a single clip BEFORE scaling up to a full batch
of clips. Get this working first; batch processing is just a loop around
this once it's solid.

Prerequisites
-------------
You must already have three separated stems from SAM Audio (web), placed
in a folder, named exactly:
    <input_dir>/dialogue.wav
    <input_dir>/music.wav
    <input_dir>/sfx.wav

Usage
-----
    # Fast path — DSP only, no GPU needed, good for first smoke-test:
    python run_full_pipeline.py --input_dir stems/clip1 --clip_name clip1 --skip_generative

    # Full path — includes Parler-TTS + AudioLDM2 (needs GPU):
    python run_full_pipeline.py --input_dir stems/clip1 --clip_name clip1

Output structure
-----------------
    outputs/<clip_name>/baseline/...
    outputs/<clip_name>/dsp_only/...
    outputs/<clip_name>/generative/...      (skipped if --skip_generative)
    outputs/<clip_name>/combined/...        (skipped if --skip_generative)
    evaluation_results/<clip_name>/evaluation_results.json
    evaluation_results/<clip_name>/evaluation_table.tex
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import soundfile as sf
import numpy as np

from generative_pipeline import GenerativePipeline, PRESETS
from evaluation import EvaluationSuite


def check_stems(input_dir: Path) -> tuple[Path, Path, Path]:
    """Verify the three expected stem files exist before doing anything else."""
    dialogue = input_dir / "dialogue.wav"
    music = input_dir / "music.wav"
    sfx = input_dir / "sfx.wav"

    missing = [p.name for p in (dialogue, music, sfx) if not p.exists()]
    if missing:
        print(f"\n[ERROR] Missing stem file(s) in {input_dir}: {missing}")
        print("Expected exactly: dialogue.wav, music.wav, sfx.wav")
        print("(Export these from SAM Audio web and place them in --input_dir)")
        sys.exit(1)

    return dialogue, music, sfx


def write_baseline_mix(dialogue: Path, music: Path, sfx: Path,
                        output_dir: Path, clip_name: str, sr: int = 44100) -> dict:
    """
    Writes the unprocessed baseline condition: stems summed at unity gain,
    no DSP, no generative processing. This is the reference everything
    else is compared against in evaluation.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    d, _ = sf_load(dialogue, sr)
    m, _ = sf_load(music, sr)
    s, _ = sf_load(sfx, sr)
    n = min(len(d), len(m), len(s))
    d, m, s = d[:n], m[:n], s[:n]

    mix = d + m + s
    peak = np.max(np.abs(mix))
    if peak > 1e-9:
        mix = mix / peak * 0.95

    mix_path = output_dir / f"{clip_name}_baseline_accessible_mix.wav"
    dlg_path = output_dir / f"{clip_name}_baseline_dialogue_enhanced.wav"
    sfx_path = output_dir / f"{clip_name}_baseline_sfx_suppressed.wav"

    sf.write(mix_path, mix, sr)
    sf.write(dlg_path, d, sr)
    sf.write(sfx_path, s, sr)

    return {
        "condition": "baseline",
        "output_path": str(mix_path),
        "duration_s": n / sr,
    }


def sf_load(path: Path, sr: int):
    import librosa
    y, orig_sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), orig_sr


def main():
    parser = argparse.ArgumentParser(
        description="Run one clip through baseline/dsp_only/generative/combined "
                    "and evaluate the results.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing dialogue.wav, music.wav, sfx.wav")
    parser.add_argument("--clip_name", required=True,
                        help="Identifier for this clip, used in all output filenames")
    parser.add_argument("--output_root", default="outputs",
                        help="Root directory for pipeline outputs (default: outputs/)")
    parser.add_argument("--eval_root", default="evaluation_results",
                        help="Root directory for evaluation results (default: evaluation_results/)")
    parser.add_argument("--device", default="cuda",
                        help="Device for generative models (default: cuda). Use 'cpu' if no GPU "
                             "available, but Parler-TTS/AudioLDM2 will be very slow on CPU.")
    parser.add_argument("--skip_generative", action="store_true",
                        help="Skip the generative/combined conditions (Parler-TTS + AudioLDM2). "
                             "Use this for a fast first smoke-test of baseline vs dsp_only.")
    parser.add_argument("--sr", type=int, default=44100)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_root) / args.clip_name
    eval_dir = Path(args.eval_root) / args.clip_name

    dialogue_path, music_path, sfx_path = check_stems(input_dir)

    print("=" * 70)
    print(f"  EEEM004 single-clip pipeline run — clip: {args.clip_name}")
    print(f"  Conditions: baseline, dsp_only" +
          ("" if args.skip_generative else ", generative, combined"))
    print("=" * 70)

    t0 = time.time()

    # ── Condition: baseline (no processing) ────────────────────────────
    print("\n[1/4] Baseline (unprocessed reference)...")
    baseline_dir = output_dir / "baseline"
    write_baseline_mix(dialogue_path, music_path, sfx_path, baseline_dir, args.clip_name, args.sr)
    print(f"  Done -> {baseline_dir}")

    # ── Condition: dsp_only (fast, no GPU needed) ───────────────────────
    print("\n[2/4] DSP-only (conventional processing, no generative AI)...")
    dsp_dir = output_dir / "dsp_only"
    pipe = GenerativePipeline(device=args.device)
    # dsp_only preset has both generative flags False, so models are never
    # loaded for this condition — safe to run without a GPU.
    dsp_result = pipe.run(
        dialogue_path=str(dialogue_path),
        music_path=str(music_path),
        sfx_path=str(sfx_path),
        profile=PRESETS["dsp_only"],
        output_dir=str(dsp_dir),
        clip_name=args.clip_name,
    )
    print(f"  Done -> {dsp_result['output_path']}")

    conditions_run = ["dsp_only"]

    if not args.skip_generative:
        # ── Condition: generative (AI stages only) ──────────────────────
        print("\n[3/4] Generative (Parler-TTS + AudioLDM2, no DSP)...")
        print("  NOTE: this stage loads multiple large models and runs diffusion "
              "inference — expect this to take significantly longer than the "
              "DSP-only stage, especially on a single clip's first run.")
        gen_dir = output_dir / "generative"
        gen_t0 = time.time()
        gen_result = pipe.run(
            dialogue_path=str(dialogue_path),
            music_path=str(music_path),
            sfx_path=str(sfx_path),
            profile=PRESETS["generative"],
            output_dir=str(gen_dir),
            clip_name=args.clip_name,
        )
        print(f"  Done -> {gen_result['output_path']}  ({time.time()-gen_t0:.1f}s)")
        conditions_run.append("generative")

        # ── Condition: combined (AI + DSP stacked) ───────────────────────
        print("\n[4/4] Combined (generative AI + DSP)...")
        comb_dir = output_dir / "combined"
        comb_t0 = time.time()
        comb_result = pipe.run(
            dialogue_path=str(dialogue_path),
            music_path=str(music_path),
            sfx_path=str(sfx_path),
            profile=PRESETS["combined"],
            output_dir=str(comb_dir),
            clip_name=args.clip_name,
        )
        print(f"  Done -> {comb_result['output_path']}  ({time.time()-comb_t0:.1f}s)")
        conditions_run.append("combined")
    else:
        print("\n[3-4/4] Skipped (--skip_generative). Run again without this flag "
              "once dsp_only results look sensible, on a machine with a GPU.")

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Running objective evaluation...")
    print("=" * 70)

    suite = EvaluationSuite(
        reference_dir=str(baseline_dir),
        sample_rate=args.sr,
        clip_name=args.clip_name,
    )
    for cond in conditions_run:
        suite.add_condition(cond, str(output_dir / cond))

    results = suite.run()
    suite.report(results, str(eval_dir))

    elapsed = time.time() - t0
    print(f"\n[run_full_pipeline] Total time: {elapsed/60:.1f} min")
    print(f"[run_full_pipeline] Results: {eval_dir / 'evaluation_results.json'}")
    print(f"[run_full_pipeline] LaTeX table: {eval_dir / 'evaluation_table.tex'}")


if __name__ == "__main__":
    main()
