"""
EEEM004 — Stem Personalisation Engine
======================================
Takes separated stems (dialogue / music / sfx) and applies configurable
sensory-preference adjustments before remixing into a single accessible track.

Designed around real, named sensory accommodations rather than generic
"EQ knobs" — each parameter maps to a documented hyperacusis/sensory
processing concern so the choices are defensible in a dissertation write-up.

Core operations, applied per-stem then summed:
1. Frequency taming   — attenuate a configurable band (commonly cited
                         trigger range for hyperacusis/misophonia is
                         2-8 kHz: sibilance, cymbals, alarms, cutlery)
2. Transient limiting  — soft-knee compression on sudden level jumps
                         (sirens, jump-scares, slammed doors) without
                         crushing the overall dynamic range
3. Stem gain           — simple relative loudness balance (dialogue
                         boost / background reduce, as before)

This module contains NO Jupyter/display code — it is pure DSP so it can be
imported by the SAM pipeline notebook, a future test notebook, or eventually
a real interface, without modification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------

@dataclass
class FrequencyTamingConfig:
    """Attenuates a frequency band associated with sensory discomfort."""
    enabled: bool = True
    low_hz: float = 2000.0       # lower edge of the taming band
    high_hz: float = 8000.0      # upper edge of the taming band
    reduction_db: float = -12.0  # how much to attenuate within the band
    # Transition width avoids harsh filter artefacts (Hann-window crossfade
    # between passband and tamed band, in Hz)
    transition_hz: float = 500.0


@dataclass
class TransientLimiterConfig:
    """Soft-knee limiter targeting sudden onsets (sirens, bangs, jump-scares)."""
    enabled: bool = True
    threshold_db: float = -18.0   # level above which limiting kicks in
    ratio: float = 6.0            # compression ratio above threshold
    attack_ms: float = 5.0        # how fast the limiter engages
    release_ms: float = 120.0     # how fast it lets go


@dataclass
class StemGainConfig:
    """Relative loudness balance between stems in the final mix."""
    dialogue_gain: float = 1.5
    music_gain: float = 0.4
    sfx_gain: float = 0.4


@dataclass
class PersonalisationProfile:
    """A complete, named set of preferences — the unit a user would save/load."""
    name: str = "default"
    sample_rate: int = 48000
    frequency_taming: FrequencyTamingConfig = field(default_factory=FrequencyTamingConfig)
    transient_limiter: TransientLimiterConfig = field(default_factory=TransientLimiterConfig)
    stem_gain: StemGainConfig = field(default_factory=StemGainConfig)


# Example presets — these are illustrative starting points for your
# evaluation chapter, not validated clinical recommendations.
PRESETS = {
    "default": PersonalisationProfile(name="default"),
    "hyperacusis_high_freq": PersonalisationProfile(
        name="hyperacusis_high_freq",
        frequency_taming=FrequencyTamingConfig(low_hz=2000, high_hz=10000, reduction_db=-18),
        transient_limiter=TransientLimiterConfig(threshold_db=-20, ratio=8.0),
        stem_gain=StemGainConfig(dialogue_gain=1.6, music_gain=0.3, sfx_gain=0.25),
    ),
    "gentle": PersonalisationProfile(
        name="gentle",
        frequency_taming=FrequencyTamingConfig(low_hz=3000, high_hz=8000, reduction_db=-6),
        transient_limiter=TransientLimiterConfig(threshold_db=-14, ratio=3.0),
        stem_gain=StemGainConfig(dialogue_gain=1.2, music_gain=0.7, sfx_gain=0.6),
    ),
    "dialogue_only_focus": PersonalisationProfile(
        name="dialogue_only_focus",
        frequency_taming=FrequencyTamingConfig(low_hz=2000, high_hz=8000, reduction_db=-24),
        transient_limiter=TransientLimiterConfig(threshold_db=-22, ratio=10.0),
        stem_gain=StemGainConfig(dialogue_gain=2.0, music_gain=0.15, sfx_gain=0.1),
    ),
}


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------

def apply_frequency_taming(y: np.ndarray, sr: int, cfg: FrequencyTamingConfig) -> np.ndarray:
    """
    Attenuates [low_hz, high_hz] using an STFT mask with smooth (Hann-based)
    transition edges, to avoid ringing artefacts from a hard brick-wall cut.
    """
    if not cfg.enabled:
        return y

    n_fft = 2048
    hop = n_fft // 4
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    gain_db = np.zeros_like(freqs)
    band_mask = (freqs >= cfg.low_hz) & (freqs <= cfg.high_hz)
    gain_db[band_mask] = cfg.reduction_db

    # Smooth the transition into/out of the band so it isn't a hard cut
    trans_bins = max(1, int(cfg.transition_hz / (sr / n_fft)))
    if trans_bins > 1:
        window = np.hanning(2 * trans_bins)
        gain_db = np.convolve(gain_db, window / window.sum(), mode="same")

    gain_lin = 10 ** (gain_db / 20.0)
    S_tamed = S * gain_lin[:, np.newaxis]

    y_tamed = librosa.istft(S_tamed, hop_length=hop, length=len(y))
    return y_tamed.astype(np.float32)


def apply_transient_limiter(y: np.ndarray, sr: int, cfg: TransientLimiterConfig) -> np.ndarray:
    """
    Simple feed-forward soft-knee compressor/limiter operating on an
    envelope follower. Targets sudden onsets (sirens, slams, jump-scares)
    while leaving steady-state level largely untouched.
    """
    if not cfg.enabled:
        return y

    threshold_lin = 10 ** (cfg.threshold_db / 20.0)
    attack_coef = np.exp(-1.0 / (sr * cfg.attack_ms / 1000.0))
    release_coef = np.exp(-1.0 / (sr * cfg.release_ms / 1000.0))

    abs_y = np.abs(y)
    envelope = np.zeros_like(abs_y)
    env = 0.0
    for i, sample in enumerate(abs_y):
        coef = attack_coef if sample > env else release_coef
        env = coef * env + (1 - coef) * sample
        envelope[i] = env

    # Soft-knee gain reduction above threshold
    with np.errstate(divide="ignore"):
        over_db = 20 * np.log10(np.maximum(envelope, 1e-9) / threshold_lin)
    over_db = np.maximum(over_db, 0.0)
    reduction_db = over_db * (1 - 1 / cfg.ratio)
    gain_lin = 10 ** (-reduction_db / 20.0)

    return (y * gain_lin).astype(np.float32)


def normalise_peak(y: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Prevents clipping after gain staging / summing stems."""
    peak = np.max(np.abs(y))
    if peak < 1e-9:
        return y
    return (y / peak * target_peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def process_stem(y: np.ndarray, sr: int, profile: PersonalisationProfile,
                  gain: float) -> np.ndarray:
    """Applies frequency taming + transient limiting + gain to a single stem."""
    y = apply_frequency_taming(y, sr, profile.frequency_taming)
    y = apply_transient_limiter(y, sr, profile.transient_limiter)
    y = y * gain
    return y


def personalise_mix(
    dialogue_path: str,
    music_path: str,
    sfx_path: str,
    profile: PersonalisationProfile,
    output_path: str,
) -> dict:
    """
    Loads three stem files, applies the personalisation profile to each,
    sums them into a single accessible mix, and writes the result to disk.

    Returns a dict of metadata useful for logging / dissertation evaluation.
    """
    sr = profile.sample_rate

    dialogue, _ = librosa.load(dialogue_path, sr=sr, mono=True)
    music, _ = librosa.load(music_path, sr=sr, mono=True)
    sfx, _ = librosa.load(sfx_path, sr=sr, mono=True)

    n = min(len(dialogue), len(music), len(sfx))
    dialogue, music, sfx = dialogue[:n], music[:n], sfx[:n]

    dialogue_out = process_stem(dialogue, sr, profile, profile.stem_gain.dialogue_gain)
    music_out = process_stem(music, sr, profile, profile.stem_gain.music_gain)
    sfx_out = process_stem(sfx, sr, profile, profile.stem_gain.sfx_gain)

    mix = dialogue_out + music_out + sfx_out
    mix = normalise_peak(mix)

    sf.write(output_path, mix, sr)

    return {
        "profile": profile.name,
        "output_path": output_path,
        "duration_s": n / sr,
        "sample_rate": sr,
        "stem_gains": {
            "dialogue": profile.stem_gain.dialogue_gain,
            "music": profile.stem_gain.music_gain,
            "sfx": profile.stem_gain.sfx_gain,
        },
        "frequency_taming": {
            "enabled": profile.frequency_taming.enabled,
            "band_hz": [profile.frequency_taming.low_hz, profile.frequency_taming.high_hz],
            "reduction_db": profile.frequency_taming.reduction_db,
        },
        "transient_limiter": {
            "enabled": profile.transient_limiter.enabled,
            "threshold_db": profile.transient_limiter.threshold_db,
            "ratio": profile.transient_limiter.ratio,
        },
    }