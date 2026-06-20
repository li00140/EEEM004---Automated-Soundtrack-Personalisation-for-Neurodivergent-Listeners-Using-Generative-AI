"""
EEEM004 — Objective Evaluation Module
======================================
Louis Ilett | Supervisor: Prof Philip Jackson | University of Surrey

Computes objective audio quality and accessibility metrics for comparing
system outputs across the four evaluation conditions:

    baseline        → unprocessed re-mix (no AI, no DSP)
    dsp_only        → conventional EQ + gain only
    generative      → AI stages only (Parler-TTS + AudioLDM2)
    combined        → AI + DSP stacked

Metrics
-------
1.  SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
    Standard in source separation literature. Measures how well the
    processing preserves the target signal vs introducing artefacts.
    Higher = better. [Vincent et al., 2006; Le Roux et al., 2019]

2.  PESQ-WB (Perceptual Evaluation of Speech Quality — Wideband)
    ITU-T P.862.2 reference implementation via pesq package.
    Applied to dialogue stem only. Range: -0.5 to 4.5.
    Higher = better perceived speech quality. [ITU-T P.862, 2001]

3.  STOI (Short-Time Objective Intelligibility)
    Correlates with intelligibility scores from listening tests.
    Applied to dialogue stem. Range: 0–1. Higher = more intelligible.
    [Taal et al., 2011]

4.  Spectral Centroid Shift (SCS)
    Mean shift of the spectral centroid between reference and processed
    audio, across the full mix. Captures how much frequency content
    has moved — relevant for high-frequency taming evaluation.
    Reported in Hz. Negative = shift toward lower frequencies.

5.  Transient Peak Reduction (TPR)
    Difference in peak dBFS between reference and processed audio,
    measured on the SFX stem only. Captures how much the suppression
    stage has reduced sudden peaks.
    Reported in dB. Positive = peaks reduced.

6.  RMS Dialogue Intelligibility Ratio (RDIR)
    RMS of dialogue stem divided by RMS of music+sfx in the final mix.
    Higher ratio = dialogue more prominent relative to background.
    Captures the practical accessibility goal.

Usage
-----
    from evaluation import EvaluationSuite
    suite = EvaluationSuite(reference_dir="outputs/baseline/",
                             sample_rate=44100)
    suite.add_condition("dsp_only",   "outputs/dsp_only/")
    suite.add_condition("generative", "outputs/generative/")
    suite.add_condition("combined",   "outputs/combined/")
    results = suite.run()
    suite.report(results, output_dir="evaluation_results/")

Dependencies (already in your venv from generative_pipeline install)
-----
    pip install pesq pystoi torchmetrics soundfile librosa numpy
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf


# ── Optional imports with graceful fallback ────────────────────────────────
try:
    from pesq import pesq
    _HAS_PESQ = True
except ImportError:
    _HAS_PESQ = False
    print("[evaluation] Warning: pesq not installed. PESQ scores will be skipped.")
    print("  Install with: pip install pesq")

try:
    from pystoi import stoi
    _HAS_STOI = True
except ImportError:
    _HAS_STOI = False
    print("[evaluation] Warning: pystoi not installed. STOI scores will be skipped.")
    print("  Install with: pip install pystoi")


# ═══════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio.
    Both arrays must be the same length and float32/float64.
    """
    reference = reference - reference.mean()
    estimate  = estimate  - estimate.mean()
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]

    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-9)
    target = alpha * reference
    noise  = estimate - target

    sdr = 10 * np.log10(
        (np.dot(target, target) + 1e-9) / (np.dot(noise, noise) + 1e-9)
    )
    return float(sdr)


def spectral_centroid_shift(reference: np.ndarray, estimate: np.ndarray,
                             sr: int) -> float:
    """
    Mean difference of spectral centroid between reference and estimate,
    in Hz. Negative means the estimate is spectrally darker (lower centroid).
    """
    n = min(len(reference), len(estimate))
    sc_ref = librosa.feature.spectral_centroid(y=reference[:n], sr=sr)[0]
    sc_est = librosa.feature.spectral_centroid(y=estimate[:n],  sr=sr)[0]
    return float(np.mean(sc_est) - np.mean(sc_ref))


def transient_peak_reduction(reference_sfx: np.ndarray,
                              estimate_sfx: np.ndarray) -> float:
    """
    Reduction in peak dBFS between reference SFX stem and processed SFX stem.
    Positive values mean the processed version has lower peaks.
    """
    def peak_dbfs(y):
        peak = np.max(np.abs(y))
        return 20 * np.log10(max(peak, 1e-9))

    ref_peak = peak_dbfs(reference_sfx)
    est_peak = peak_dbfs(estimate_sfx)
    return float(ref_peak - est_peak)  # positive = reduction


def rms_dialogue_intelligibility_ratio(dialogue: np.ndarray,
                                        music: np.ndarray,
                                        sfx: np.ndarray) -> float:
    """
    RMS of dialogue divided by RMS of (music + sfx).
    Higher = dialogue more prominent in the mix.
    """
    def rms(y):
        return float(np.sqrt(np.mean(y ** 2)) + 1e-9)

    n = min(len(dialogue), len(music), len(sfx))
    background = music[:n] + sfx[:n]
    return rms(dialogue[:n]) / rms(background)


def pesq_score(reference: np.ndarray, estimate: np.ndarray,
               sr: int) -> Optional[float]:
    """
    PESQ-WB score. Requires sr=16000 (PESQ resamples internally if needed).
    Returns None if pesq is not installed.
    """
    if not _HAS_PESQ:
        return None
    # PESQ wideband requires 16kHz
    target_sr = 16000
    if sr != target_sr:
        reference = librosa.resample(reference, orig_sr=sr, target_sr=target_sr)
        estimate  = librosa.resample(estimate,  orig_sr=sr, target_sr=target_sr)
    n = min(len(reference), len(estimate))
    try:
        score = pesq(target_sr, reference[:n], estimate[:n], "wb")
        return float(score)
    except Exception as e:
        print(f"  [PESQ] Error: {e}")
        return None


def stoi_score(reference: np.ndarray, estimate: np.ndarray,
               sr: int) -> Optional[float]:
    """STOI intelligibility. Returns None if pystoi is not installed."""
    if not _HAS_STOI:
        return None
    n = min(len(reference), len(estimate))
    try:
        score = stoi(reference[:n], estimate[:n], sr, extended=False)
        return float(score)
    except Exception as e:
        print(f"  [STOI] Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION SUITE
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationSuite:
    """
    Computes all metrics for a set of conditions relative to a baseline.

    Expected file layout per condition directory:
        <condition_dir>/<clip_name>_<condition>_accessible_mix.wav   (full mix)
        <condition_dir>/<clip_name>_<condition>_dialogue_enhanced.wav (dialogue stem)
        <condition_dir>/<clip_name>_<condition>_sfx_suppressed.wav   (sfx stem)

    If a stem file is missing (e.g. baseline has no _enhanced suffix),
    specify stem_paths explicitly when calling add_condition.
    """

    def __init__(self, reference_dir: str, sample_rate: int = 44100,
                 clip_name: str = "clip"):
        self.sr = sample_rate
        self.clip_name = clip_name
        self._ref_dir = Path(reference_dir)
        self._conditions: dict[str, dict] = {}

        # Load reference stems
        self._ref = self._load_stems(self._ref_dir, condition="baseline")
        print(f"[EvaluationSuite] Reference loaded from {reference_dir}")

    def _load_stems(self, directory: Path, condition: str,
                    mix_file: str = None,
                    dialogue_file: str = None,
                    sfx_file: str = None) -> dict:
        """Load mix, dialogue, and SFX stems from a condition directory."""
        d = Path(directory)
        cn = self.clip_name

        def _load(path):
            y, _ = librosa.load(str(path), sr=self.sr, mono=True)
            return y.astype(np.float32)

        # Infer filenames if not given
        mix_path = Path(mix_file) if mix_file else \
            d / f"{cn}_{condition}_accessible_mix.wav"
        dialogue_path = Path(dialogue_file) if dialogue_file else \
            d / f"{cn}_{condition}_dialogue_enhanced.wav"
        sfx_path = Path(sfx_file) if sfx_file else \
            d / f"{cn}_{condition}_sfx_suppressed.wav"

        # Fallback: try finding any .wav with the right keyword
        def _first_match(pattern):
            matches = list(d.glob(f"*{pattern}*.wav"))
            return matches[0] if matches else None

        if not mix_path.exists():
            m = _first_match("mix")
            if m:
                mix_path = m
                print(f"  [EvaluationSuite] Inferred mix: {m.name}")

        if not dialogue_path.exists():
            m = _first_match("dialogue")
            if m:
                dialogue_path = m
                print(f"  [EvaluationSuite] Inferred dialogue: {m.name}")

        if not sfx_path.exists():
            m = _first_match("sfx")
            if m:
                sfx_path = m
                print(f"  [EvaluationSuite] Inferred SFX: {m.name}")

        stems = {}
        for key, path in [("mix", mix_path), ("dialogue", dialogue_path),
                           ("sfx", sfx_path)]:
            if path.exists():
                stems[key] = _load(path)
            else:
                print(f"  [EvaluationSuite] Warning: {key} stem not found at {path}")
                stems[key] = None

        return stems

    def add_condition(self, name: str, directory: str,
                      mix_file: str = None,
                      dialogue_file: str = None,
                      sfx_file: str = None):
        """Register a condition to evaluate against the baseline."""
        stems = self._load_stems(
            Path(directory), condition=name,
            mix_file=mix_file,
            dialogue_file=dialogue_file,
            sfx_file=sfx_file,
        )
        self._conditions[name] = stems
        print(f"[EvaluationSuite] Registered condition: {name}")

    def _evaluate_condition(self, name: str, stems: dict) -> dict:
        """Compute all metrics for one condition vs the reference."""
        print(f"\n[EvaluationSuite] Evaluating condition: {name}")
        results = {"condition": name}

        ref_mix = self._ref.get("mix")
        est_mix = stems.get("mix")
        ref_dlg = self._ref.get("dialogue")
        est_dlg = stems.get("dialogue")
        ref_sfx = self._ref.get("sfx")
        est_sfx = stems.get("sfx")

        # 1. SI-SDR on full mix
        if ref_mix is not None and est_mix is not None:
            results["si_sdr_mix_db"] = round(si_sdr(ref_mix, est_mix), 3)
            print(f"  SI-SDR (mix):          {results['si_sdr_mix_db']:.2f} dB")
        else:
            results["si_sdr_mix_db"] = None

        # 2. SI-SDR on dialogue stem
        if ref_dlg is not None and est_dlg is not None:
            results["si_sdr_dialogue_db"] = round(si_sdr(ref_dlg, est_dlg), 3)
            print(f"  SI-SDR (dialogue):     {results['si_sdr_dialogue_db']:.2f} dB")
        else:
            results["si_sdr_dialogue_db"] = None

        # 3. PESQ on dialogue
        if ref_dlg is not None and est_dlg is not None:
            score = pesq_score(ref_dlg, est_dlg, self.sr)
            results["pesq_wb"] = round(score, 3) if score is not None else None
            if score is not None:
                print(f"  PESQ-WB (dialogue):    {score:.3f}")
        else:
            results["pesq_wb"] = None

        # 4. STOI on dialogue
        if ref_dlg is not None and est_dlg is not None:
            score = stoi_score(ref_dlg, est_dlg, self.sr)
            results["stoi"] = round(score, 4) if score is not None else None
            if score is not None:
                print(f"  STOI (dialogue):       {score:.4f}")
        else:
            results["stoi"] = None

        # 5. Spectral centroid shift on full mix
        if ref_mix is not None and est_mix is not None:
            scs = spectral_centroid_shift(ref_mix, est_mix, self.sr)
            results["spectral_centroid_shift_hz"] = round(scs, 1)
            print(f"  Spectral centroid Δ:   {scs:+.1f} Hz")
        else:
            results["spectral_centroid_shift_hz"] = None

        # 6. Transient peak reduction on SFX stem
        if ref_sfx is not None and est_sfx is not None:
            tpr = transient_peak_reduction(ref_sfx, est_sfx)
            results["transient_peak_reduction_db"] = round(tpr, 2)
            print(f"  Transient peak Δ:      {tpr:+.2f} dB")
        else:
            results["transient_peak_reduction_db"] = None

        # 7. RDIR — dialogue intelligibility ratio in the estimated mix
        # (compare ratio in baseline vs processed)
        ref_music = self._ref.get("music")
        est_music = stems.get("music")
        if ref_dlg is not None and ref_sfx is not None:
            ref_music_arr = (
                ref_music if ref_music is not None
                else np.zeros_like(ref_dlg)
            )

            reference_shape = est_dlg if est_dlg is not None else ref_dlg

            est_music_arr = (
                est_music if est_music is not None
                else np.zeros_like(reference_shape)
            )

            est_dlg_arr = (
                est_dlg if est_dlg is not None
                else ref_dlg
            )

            est_sfx_arr = (
                est_sfx if est_sfx is not None
                else ref_sfx
            )

            ref_rdir = rms_dialogue_intelligibility_ratio(ref_dlg, ref_music_arr, ref_sfx)
            est_rdir = rms_dialogue_intelligibility_ratio(est_dlg_arr, est_music_arr, est_sfx_arr)
            results["rdir_reference"] = round(ref_rdir, 4)
            results["rdir_processed"] = round(est_rdir, 4)
            results["rdir_improvement"] = round(est_rdir - ref_rdir, 4)
            print(f"  RDIR (ref → processed): {ref_rdir:.4f} → {est_rdir:.4f}  "
                  f"(Δ {est_rdir - ref_rdir:+.4f})")
        else:
            results["rdir_reference"] = results["rdir_processed"] = \
                results["rdir_improvement"] = None

        return results

    def run(self) -> list[dict]:
        """Run evaluation for all registered conditions."""
        all_results = []
        for name, stems in self._conditions.items():
            result = self._evaluate_condition(name, stems)
            all_results.append(result)
        return all_results

    def report(self, results: list[dict], output_dir: str):
        """
        Save results to JSON and print a formatted comparison table.
        The JSON is structured for easy import into your dissertation figures.
        """
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "evaluation_results.json")

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[EvaluationSuite] Results saved to {json_path}")

        # Print comparison table
        metrics = [
            ("si_sdr_mix_db",               "SI-SDR mix (dB)    ", "↑ better"),
            ("si_sdr_dialogue_db",           "SI-SDR dlg (dB)    ", "↑ better"),
            ("pesq_wb",                      "PESQ-WB            ", "↑ better"),
            ("stoi",                         "STOI               ", "↑ better"),
            ("spectral_centroid_shift_hz",   "Spectral centroid Δ", "↓ better"),
            ("transient_peak_reduction_db",  "Transient peak Δ   ", "↑ better"),
            ("rdir_improvement",             "RDIR improvement   ", "↑ better"),
        ]

        conds = [r["condition"] for r in results]
        print("\n" + "=" * 80)
        print(f"{'METRIC':<28} {'DIR':>9}  " +
              "  ".join(f"{c:>12}" for c in conds))
        print("=" * 80)

        for key, label, direction in metrics:
            row = f"{label:<28} {direction:>9}  "
            for r in results:
                val = r.get(key)
                if val is None:
                    row += f"{'—':>12}  "
                else:
                    row += f"{val:>12.3f}  "
            print(row)

        print("=" * 80)
        print("\nNote: all metrics computed relative to the baseline (unprocessed) mix.")
        print("PESQ and STOI applied to dialogue stem only.")
        print("Transient Peak Reduction applied to SFX stem only.")

        # Also write a LaTeX table for direct paste into dissertation
        latex_path = os.path.join(output_dir, "evaluation_table.tex")
        with open(latex_path, "w") as f:
            f.write("% Auto-generated by evaluation.py — paste into dissertation\n")
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{Objective evaluation metrics across processing conditions.}\n")
            f.write("\\label{tab:eval}\n")
            ncols = 2 + len(conds)
            f.write(f"\\begin{{tabular}}{{ll{'r' * len(conds)}}}\n\\hline\n")
            f.write("Metric & Dir. & " + " & ".join(conds) + " \\\\\n\\hline\n")
            for key, label, direction in metrics:
                row_vals = []
                for r in results:
                    val = r.get(key)
                    row_vals.append(f"—" if val is None else f"{val:.3f}")
                f.write(f"{label.strip()} & {direction} & " +
                        " & ".join(row_vals) + " \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        print(f"[EvaluationSuite] LaTeX table saved to {latex_path}")


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE USAGE — run directly to evaluate two directories
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EEEM004 objective evaluation — compare processed conditions "
                    "against a baseline.")
    parser.add_argument("--baseline", required=True,
                        help="Directory containing baseline (unprocessed) outputs")
    parser.add_argument("--conditions", nargs="+",
                        help="One or more 'name:directory' pairs, e.g. "
                             "dsp_only:outputs/dsp_only  generative:outputs/gen")
    parser.add_argument("--clip", default="clip",
                        help="Clip name prefix used in output filenames")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--output_dir", default="evaluation_results/")
    args = parser.parse_args()

    suite = EvaluationSuite(args.baseline, sample_rate=args.sr,
                             clip_name=args.clip)

    for cond_str in (args.conditions or []):
        name, directory = cond_str.split(":", 1)
        suite.add_condition(name, directory)

    results = suite.run()
    suite.report(results, args.output_dir)
