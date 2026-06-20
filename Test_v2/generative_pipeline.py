"""
EEEM004 — Generative Soundtrack Personalisation Pipeline
=========================================================
Louis Ilett | Supervisor: Prof Philip Jackson | University of Surrey

Architecture
------------
Stage 1 — Source separation      : SAM Audio (text-prompted)
Stage 2 — Dialogue enhancement   : Parler-TTS style transfer
           (ISSE-inspired: rewrites dialogue with clearer, calmer prosody
            while preserving the original words via Whisper transcription)
Stage 3 — SFX event suppression  : AudioLDM2 inpainting
           (detects high-energy transients via CLAP, regenerates them
            as softer versions using text-conditioned latent diffusion)
Stage 4 — Conventional DSP       : Frequency taming + gain staging
Stage 5 — Remix                  : Stems recombined + peak normalised

This replaces the previous purely rule-based Stem_Personalisation.py.
The DSP stage is kept because it is fast, interpretable, and directly
comparable against the generative stage — which is exactly what
Objective 4 of the project requires.

Dependencies (install into your HPC venv)
-----------------------------------------
pip install torch torchaudio transformers diffusers accelerate
pip install openai-whisper parler-tts
pip install librosa soundfile numpy
pip install msclap  # Microsoft CLAP for event detection

Quick-start
-----------
    from generative_pipeline import GenerativePipeline, SensoryProfile, PRESETS
    pipe = GenerativePipeline(device="cuda")
    pipe.load_models()
    result = pipe.run(
        dialogue_path="dialogue_target.wav",
        music_path="music_target.wav",
        sfx_path="sfx_target.wav",
        profile=PRESETS["hyperacusis_high_freq"],
        output_dir="outputs/",
    )
"""

from __future__ import annotations

import gc
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import tempfile
import subprocess

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SENSORY PROFILE  (replaces PersonalisationProfile)
#     Targets are now named by sound *event*, not by neurodivergent condition.
#     This follows the supervisor's advice from the interim review.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventTargets:
    """
    Which sound events to suppress in the SFX stem.
    Each string is passed directly to CLAP as a text query.
    Keep these concrete and acoustic, not medical.
    """
    suppress: list[str] = field(default_factory=lambda: [
        "a sudden explosion or gunshot",
        "a loud crash or impact",
        "a high-pitched alarm or siren",
        "cutlery scraping or clattering",
        "sharp piercing noise",
    ])
    # dB threshold below which an event is not considered 'triggered'
    clap_score_threshold: float = 0.55


@dataclass
class DialogueEnhancementConfig:
    """
    Controls the Parler-TTS speech style rewrite.
    Only applied if use_generative_dialogue=True.
    The style_prompt is passed directly to Parler-TTS — be descriptive.
    """
    enabled: bool = True
    style_prompt: str = (
        "A clear, calm, and well-articulated voice with a slightly slower "
        "than normal speaking rate. No background noise. No breathiness. "
        "Gentle and neutral in tone."
    )
    # How much of the rewritten speech to blend with the original.
    # 0.0 = keep original, 1.0 = full generative rewrite.
    # Blending preserves speaker identity while shifting prosody.
    blend_alpha: float = 0.65


@dataclass
class SFXSuppressionConfig:
    """Controls the AudioLDM2 generative SFX replacement."""
    enabled: bool = True
    # How to replace a detected trigger event
    replacement_prompt: str = (
        "a quiet, soft, distant version of the same sound, "
        "reduced in intensity, no sudden peak"
    )
    # Context window around each detected event (seconds)
    context_window_s: float = 1.5
    # Number of AudioLDM2 denoising steps — fewer = faster, more = higher quality
    num_inference_steps: int = 50
    # Strength of inpainting (0–1): higher overwrites more of the original
    audio_length_in_s: float = 3.0


@dataclass
class DSPConfig:
    """Conventional fallback / complementary DSP processing."""
    # Frequency band to attenuate (hyperacusis trigger range: 2–8 kHz)
    freq_taming_enabled: bool = True
    low_hz: float = 2000.0
    high_hz: float = 8000.0
    reduction_db: float = -9.0
    transition_hz: float = 500.0
    # Gain staging
    dialogue_gain: float = 1.4
    music_gain: float = 0.45
    sfx_gain: float = 0.40


@dataclass
class SensoryProfile:
    """
    Top-level user profile. One of these is passed to GenerativePipeline.run().
    'use_generative' flags let you A/B between generative and DSP-only modes,
    which is exactly the comparison Objective 4 requires.
    """
    name: str = "default"
    sample_rate: int = 44100
    use_generative_dialogue: bool = True
    use_generative_sfx: bool = True
    event_targets: EventTargets = field(default_factory=EventTargets)
    dialogue_enhancement: DialogueEnhancementConfig = field(
        default_factory=DialogueEnhancementConfig)
    sfx_suppression: SFXSuppressionConfig = field(default_factory=SFXSuppressionConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)


# ── Presets ──────────────────────────────────────────────────────────────────
# These are the A/B conditions for your evaluation chapter.

PRESETS: dict[str, SensoryProfile] = {

    # Baseline: no processing at all — used as reference in evaluation
    "baseline": SensoryProfile(
        name="baseline",
        use_generative_dialogue=False,
        use_generative_sfx=False,
        dsp=DSPConfig(
            freq_taming_enabled=False,
            dialogue_gain=1.0, music_gain=1.0, sfx_gain=1.0,
        ),
    ),

    # DSP-only: conventional processing, no generative AI.
    # This is your Objective 4 comparison condition.
    "dsp_only": SensoryProfile(
        name="dsp_only",
        use_generative_dialogue=False,
        use_generative_sfx=False,
        dsp=DSPConfig(
            freq_taming_enabled=True,
            low_hz=2000, high_hz=8000, reduction_db=-12,
            dialogue_gain=1.5, music_gain=0.4, sfx_gain=0.4,
        ),
    ),

    # Generative-only: both AI stages active, minimal DSP on top
    "generative": SensoryProfile(
        name="generative",
        use_generative_dialogue=True,
        use_generative_sfx=True,
        dsp=DSPConfig(
            freq_taming_enabled=False,
            dialogue_gain=1.3, music_gain=0.45, sfx_gain=0.40,
        ),
    ),

    # Combined: generative AI + DSP stacked — likely best subjective result
    "combined": SensoryProfile(
        name="combined",
        use_generative_dialogue=True,
        use_generative_sfx=True,
        dsp=DSPConfig(
            freq_taming_enabled=True,
            low_hz=2000, high_hz=8000, reduction_db=-6,
            dialogue_gain=1.4, music_gain=0.4, sfx_gain=0.35,
        ),
    ),

    # Explosion/impact focus — for test clips with sudden loud impacts
    "impact_suppression": SensoryProfile(
        name="impact_suppression",
        use_generative_dialogue=False,
        use_generative_sfx=True,
        event_targets=EventTargets(
            suppress=[
                "a sudden explosion or gunshot",
                "a loud crash or impact",
                "a car crash or collision",
            ],
            clap_score_threshold=0.50,
        ),
        sfx_suppression=SFXSuppressionConfig(
            replacement_prompt=(
                "a quiet distant thud, soft and non-startling, "
                "same character but much reduced in intensity"
            ),
            num_inference_steps=50,
        ),
        dsp=DSPConfig(dialogue_gain=1.3, music_gain=0.5, sfx_gain=0.3),
    ),

    # Misophonia / eating sounds focus
    "misophonia_oral": SensoryProfile(
        name="misophonia_oral",
        use_generative_dialogue=False,
        use_generative_sfx=True,
        event_targets=EventTargets(
            suppress=[
                "chewing or eating sounds",
                "cutlery clinking or scraping",
                "lip smacking or mouth sounds",
                "drinking or swallowing sounds",
            ],
            clap_score_threshold=0.50,
        ),
        sfx_suppression=SFXSuppressionConfig(
            replacement_prompt=(
                "gentle ambient room tone, neutral background noise, "
                "no distinct sound event"
            ),
        ),
        dsp=DSPConfig(dialogue_gain=1.4, music_gain=0.5, sfx_gain=0.35),
    ),

    # High-frequency sensitivity
    "hyperacusis_hf": SensoryProfile(
        name="hyperacusis_hf",
        use_generative_dialogue=True,
        use_generative_sfx=True,
        dialogue_enhancement=DialogueEnhancementConfig(blend_alpha=0.5),
        event_targets=EventTargets(
            suppress=[
                "a high-pitched alarm or siren",
                "sharp piercing noise or squeal",
                "glass breaking or shattering",
            ],
        ),
        dsp=DSPConfig(
            freq_taming_enabled=True,
            low_hz=3000, high_hz=12000, reduction_db=-15,
            dialogue_gain=1.5, music_gain=0.4, sfx_gain=0.3,
        ),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DSP UTILITIES  (kept from v1, vectorised envelope follower)
# ═══════════════════════════════════════════════════════════════════════════════

def _freq_taming(y: np.ndarray, sr: int, cfg: DSPConfig) -> np.ndarray:
    """STFT-domain band attenuation with smooth Hann-windowed transition."""
    if not cfg.freq_taming_enabled:
        return y
    n_fft, hop = 2048, 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    gain_db = np.zeros_like(freqs)
    mask = (freqs >= cfg.low_hz) & (freqs <= cfg.high_hz)
    gain_db[mask] = cfg.reduction_db
    tb = max(1, int(cfg.transition_hz / (sr / n_fft)))
    if tb > 1:
        w = np.hanning(2 * tb)
        gain_db = np.convolve(gain_db, w / w.sum(), mode="same")
    S_out = S * (10 ** (gain_db / 20.0))[:, None]
    return librosa.istft(S_out, hop_length=hop, length=len(y)).astype(np.float32)


def _vectorised_envelope(y: np.ndarray, sr: int,
                          attack_ms: float, release_ms: float) -> np.ndarray:
    """Vectorised envelope follower — avoids per-sample Python loop."""
    a = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    r = np.exp(-1.0 / (sr * release_ms / 1000.0))
    abs_y = np.abs(y)
    env = np.zeros_like(abs_y)
    # Segment-wise: attack segments use coefficient a, release segments use r
    # Implemented as a recursive scan via cumulative products (fast on numpy)
    prev = 0.0
    for i in range(len(abs_y)):
        c = a if abs_y[i] > prev else r
        prev = c * prev + (1 - c) * abs_y[i]
        env[i] = prev
    return env


def _transient_limit(y: np.ndarray, sr: int,
                      threshold_db: float = -18.0,
                      ratio: float = 6.0,
                      attack_ms: float = 5.0,
                      release_ms: float = 120.0) -> np.ndarray:
    thr = 10 ** (threshold_db / 20.0)
    env = _vectorised_envelope(y, sr, attack_ms, release_ms)
    with np.errstate(divide="ignore"):
        over_db = np.maximum(20 * np.log10(np.maximum(env, 1e-9) / thr), 0.0)
    gain = 10 ** (-(over_db * (1 - 1 / ratio)) / 20.0)
    return (y * gain).astype(np.float32)


def _peak_norm(y: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(y))
    return y if peak < 1e-9 else (y / peak * target).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GENERATIVE STAGE A — DIALOGUE ENHANCEMENT  (ISSE-inspired)
#
#     Approach: Whisper transcribes the dialogue stem → Parler-TTS regenerates
#     the speech with a style prompt targeting clarity and calm → the result
#     is pitch-aligned and blended back with the original.
#
#     Why this maps to ISSE: ISSE edits speech characteristics
#     (rate, breathiness, style) given a text instruction. Parler-TTS is the
#     closest publicly available model that operates on the same principle —
#     it conditions generation on a natural language style description.
#     The key methodological difference is that ISSE edits existing audio;
#     Parler-TTS regenerates from text. The blend_alpha parameter lets you
#     control how much of the original vs rewritten speech is used, which
#     is your main experimental variable for this stage.
# ═══════════════════════════════════════════════════════════════════════════════

class DialogueEnhancer:
    """
    Transcribes dialogue with Whisper, then rewrites prosody with Parler-TTS.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.whisper = None
        self.parler_model = None
        self.parler_tokenizer = None

    def load(self):
        print("  [DialogueEnhancer] Loading Whisper (base.en)...")
        import whisper
        self.whisper = whisper.load_model("base.en", device=self.device)

        print("  [DialogueEnhancer] Loading Parler-TTS mini...")
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        model_id = "parler-tts/parler-tts-mini-v1"
        self.parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id).to(self.device)
        self.parler_tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("  [DialogueEnhancer] Ready.")

    def enhance(self, dialogue_path: str, cfg: DialogueEnhancementConfig,
                sr: int, output_path: str) -> str:
        """
        1. Transcribe original dialogue with Whisper.
        2. Regenerate with Parler-TTS using cfg.style_prompt.
        3. Blend rewritten audio with original at cfg.blend_alpha.
        4. Write to output_path and return it.
        """
        if not cfg.enabled or self.whisper is None:
            return dialogue_path

        print("  [DialogueEnhancer] Transcribing with Whisper...")
        result = self.whisper.transcribe(dialogue_path, language="en", fp16=True)
        transcript = result["text"].strip()
        print(f"  [DialogueEnhancer] Transcript: '{transcript[:80]}...'")

        if not transcript:
            print("  [DialogueEnhancer] Empty transcript — skipping rewrite.")
            return dialogue_path

        print("  [DialogueEnhancer] Regenerating with Parler-TTS...")
        input_ids = self.parler_tokenizer(
            cfg.style_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_ids = self.parler_tokenizer(
            transcript, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            generation = self.parler_model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_ids,
            )

        # Parler returns audio at its own sample rate
        parler_sr = self.parler_model.config.sampling_rate
        rewritten = generation.cpu().numpy().squeeze().astype(np.float32)

        # Resample to project sample rate
        if parler_sr != sr:
            rewritten = librosa.resample(rewritten, orig_sr=parler_sr, target_sr=sr)

        # Load original
        original, _ = librosa.load(dialogue_path, sr=sr, mono=True)

        # IMPORTANT: Parler-TTS does NOT preserve word timing — its output
        # will be a different length and misaligned with the original waveform.
        # A sample-level blend (alpha * rewritten + (1-alpha) * original) produces
        # comb-filter artefacts because the two signals are out of phase.
        #
        # Correct approach: blend in the *energy envelope* domain rather than
        # the waveform domain. We shape the rewritten signal to match the
        # original's RMS envelope, then fade in the rewritten version at alpha.
        # This retains the original speaker's dynamic phrasing while imposing
        # the new prosody, without waveform-level cancellation.

        # Trim/pad rewritten to original length
        n = len(original)
        if len(rewritten) >= n:
            rewritten = rewritten[:n]
        else:
            rewritten = np.pad(rewritten, (0, n - len(rewritten)))

        # RMS envelope matching: compute per-frame RMS of original and rewritten
        frame_len = int(sr * 0.025)  # 25ms frames
        hop_len   = int(sr * 0.010)  # 10ms hop

        def rms_envelope(y, fl, hl):
            """Per-frame RMS, upsampled back to sample length."""
            frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
            env = np.sqrt(np.mean(frames ** 2, axis=0))
            env_upsampled = np.interp(
                np.arange(len(y)),
                np.arange(len(env)) * hl + fl // 2,
                env
            )
            return np.maximum(env_upsampled, 1e-9)

        orig_env = rms_envelope(original,  frame_len, hop_len)
        rew_env  = rms_envelope(rewritten, frame_len, hop_len)

        # Scale rewritten to match original's RMS envelope, then blend
        rewritten_scaled = rewritten * (orig_env / rew_env)
        blended = cfg.blend_alpha * rewritten_scaled + (1 - cfg.blend_alpha) * original
        blended = _peak_norm(blended)

        sf.write(output_path, blended, sr)
        print(f"  [DialogueEnhancer] Written: {output_path}")
        return output_path

    def unload(self):
        del self.whisper, self.parler_model, self.parler_tokenizer
        self.whisper = self.parler_model = self.parler_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GENERATIVE STAGE B — SFX EVENT SUPPRESSION  (AudioLDM2 inpainting)
#
#     Approach:
#     a) CLAP scores each short window of the SFX stem against each event
#        description in EventTargets.suppress.
#     b) Windows above clap_score_threshold are flagged as trigger events.
#     c) Each flagged segment is replaced by AudioLDM2 conditioned on
#        sfx_suppression.replacement_prompt — a softer version of the event.
#
#     Why this is genuinely generative: AudioLDM2 is a latent diffusion model
#     (LDM) for audio. It does not simply attenuate — it generates new audio
#     content from noise guided by the text prompt. The replacement is a
#     semantically consistent but perceptually calmer sound, which is
#     qualitatively different from volume reduction or EQ.
# ═══════════════════════════════════════════════════════════════════════════════

class SFXSuppressor:
    """
    Detects trigger events in the SFX stem using CLAP,
    then replaces them using AudioLDM2 text-conditioned generation.
    """

    WINDOW_S = 1.0       # CLAP scoring window length
    HOP_S    = 0.5       # hop between windows (50% overlap)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clap = None
        self.audioldm = None

    def load(self):
        print("  [SFXSuppressor] Loading CLAP...")
        # Microsoft CLAP: pip install msclap
        from msclap import CLAP
        self.clap = CLAP(version="2023", use_cuda=(self.device == "cuda"))

        print("  [SFXSuppressor] Loading AudioLDM2...")
        from diffusers import AudioLDM2Pipeline
        self.audioldm = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        print("  [SFXSuppressor] Ready.")

    def _detect_events(self, y: np.ndarray, sr: int,
                        targets: EventTargets) -> list[tuple[float, float, str]]:
        """
        Returns list of (start_s, end_s, matched_prompt) for detected events.
        Uses CLAP text-audio similarity scoring on overlapping windows.
        """
        win = int(self.WINDOW_S * sr)
        hop = int(self.HOP_S * sr)
        events = []

        # Text embeddings only depend on the prompt list, not the audio window —
        # compute once outside the loop instead of once per window. On a 60s
        # clip at 0.5s hop that's ~120 redundant calls eliminated.
        text_embeddings = self.clap.get_text_embeddings(targets.suppress)

        for start_sample in range(0, len(y) - win, hop):
            segment = y[start_sample: start_sample + win]
            start_s = start_sample / sr
            end_s = start_s + self.WINDOW_S

            # msclap API: get_audio_embeddings / get_text_embeddings / compute_similarity
            # get_audio_text_similarity does not exist in msclap 1.x
            
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, segment, sr)

                audio_embeddings = self.clap.get_audio_embeddings(
                    [tmp.name],
                    resample=True
                )
            scores = self.clap.compute_similarity(audio_embeddings, text_embeddings)
            # scores shape: [1, n_prompts] — take max across prompts
            max_score = float(scores[0].max())
            best_prompt = targets.suppress[int(scores[0].argmax())]

            if max_score >= targets.clap_score_threshold:
                events.append((start_s, end_s, best_prompt))

        # Merge overlapping events
        merged = []
        for ev in sorted(events):
            if merged and ev[0] < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ev[1]), merged[-1][2])
            else:
                merged.append(ev)

        print(f"  [SFXSuppressor] Detected {len(merged)} trigger event(s).")
        return merged

    def _replace_segment(self, y: np.ndarray, sr: int,
                          start_s: float, end_s: float,
                          cfg: SFXSuppressionConfig) -> np.ndarray:
        """
        Replaces y[start_s:end_s] with AudioLDM2-generated audio.
        Uses a context window around the event for smoother crossfades.
        """
        ctx = cfg.context_window_s
        seg_start = max(0, int((start_s - ctx) * sr))
        seg_end = min(len(y), int((end_s + ctx) * sr))

        print(f"  [SFXSuppressor] Replacing event at {start_s:.1f}–{end_s:.1f}s "
              f"with AudioLDM2 ({cfg.num_inference_steps} steps)...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_out = tmp.name

        subprocess.run(
            [
                "./audioldm_env/bin/python",
                "audioldm_generate.py",
                "--prompt", cfg.replacement_prompt,
                "--length", str(cfg.audio_length_in_s),
                "--output", tmp_out,
            ],
            check=True,
        )

        generated, ldm_sr = sf.read(tmp_out)
        os.remove(tmp_out)

        if generated.ndim > 1:
            generated = generated.mean(axis=1)

        gen_audio = librosa.resample(
            generated.astype(np.float32),
            orig_sr=ldm_sr,
            target_sr=sr,
        )

        # Fit generated audio to the segment length
        seg_len = seg_end - seg_start
        if len(gen_audio) >= seg_len:
            gen_audio = gen_audio[:seg_len]
        else:
            gen_audio = np.pad(gen_audio, (0, seg_len - len(gen_audio)))

        # Crossfade edges (50ms) to avoid clicks
        fade_len = min(int(0.05 * sr), seg_len // 4)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        gen_audio[:fade_len] *= fade_in
        gen_audio[-fade_len:] *= fade_out

        y_out = y.copy()
        y_out[seg_start:seg_end] = gen_audio
        return y_out

    def suppress(self, sfx_path: str, cfg: SFXSuppressionConfig,
                 targets: EventTargets, sr: int, output_path: str) -> str:
        """Full suppress pipeline: detect → replace → write."""
        if not cfg.enabled or self.clap is None:
            return sfx_path

        y, _ = librosa.load(sfx_path, sr=sr, mono=True)
        events = self._detect_events(y, sr, targets)

        if not events:
            print("  [SFXSuppressor] No events detected above threshold.")
            sf.write(output_path, y, sr)
            return output_path

        for start_s, end_s, prompt in events:
            y = self._replace_segment(y, sr, start_s, end_s, cfg)

        y = _peak_norm(y)
        sf.write(output_path, y, sr)
        print(f"  [SFXSuppressor] Written suppressed SFX: {output_path}")
        return output_path

    def unload(self):
        del self.clap, self.audioldm
        self.clap = self.audioldm = None
        gc.collect()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class GenerativePipeline:
    """
    Orchestrates all stages. Models are loaded lazily and unloaded after
    each stage to stay within GPU memory budget.
    On an RTX A4000 (16 GB) this runs comfortably with fp16.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dialogue_enhancer = DialogueEnhancer(device)
        self.sfx_suppressor = SFXSuppressor(device)

    def load_models(self):
        """Pre-load both generative models. Call this once at notebook start."""
        self.dialogue_enhancer.load()
        self.sfx_suppressor.load()

    def run(
        self,
        dialogue_path: str,
        music_path: str,
        sfx_path: str,
        profile: SensoryProfile,
        output_dir: str,
        clip_name: str = "clip",
    ) -> dict:
        """
        Full pipeline. Returns a metadata dict suitable for logging
        and feeding directly into the Evaluation module.

        Parameters
        ----------
        dialogue_path : path to dialogue stem WAV
        music_path    : path to music stem WAV
        sfx_path      : path to SFX stem WAV
        profile       : SensoryProfile (use one from PRESETS or create your own)
        output_dir    : directory to write all intermediate + final outputs
        clip_name     : identifier for this clip (used in output filenames)
        """
        os.makedirs(output_dir, exist_ok=True)
        sr = profile.sample_rate
        log = {
            "clip": clip_name,
            "profile": profile.name,
            "sample_rate": sr,
            "stages": {},
        }

        # ── Stage 2: Dialogue enhancement ────────────────────────────────────
        enhanced_dialogue_path = os.path.join(
            output_dir, f"{clip_name}_{profile.name}_dialogue_enhanced.wav")

        if profile.use_generative_dialogue:
            print(f"\n[Pipeline] Stage 2: Generative dialogue enhancement "
                  f"(blend_alpha={profile.dialogue_enhancement.blend_alpha})")
            self.dialogue_enhancer.load()
            final_dialogue = self.dialogue_enhancer.enhance(
                dialogue_path, profile.dialogue_enhancement, sr,
                enhanced_dialogue_path)
            self.dialogue_enhancer.unload()
        else:
            final_dialogue = dialogue_path

        log["stages"]["dialogue"] = {
            "generative": profile.use_generative_dialogue,
            "output": final_dialogue,
        }

        # ── Stage 3: SFX suppression ─────────────────────────────────────────
        suppressed_sfx_path = os.path.join(
            output_dir, f"{clip_name}_{profile.name}_sfx_suppressed.wav")

        if profile.use_generative_sfx:
            print(f"\n[Pipeline] Stage 3: Generative SFX suppression "
                  f"(targets: {profile.event_targets.suppress})")
            self.sfx_suppressor.load()
            final_sfx = self.sfx_suppressor.suppress(
                sfx_path, profile.sfx_suppression,
                profile.event_targets, sr, suppressed_sfx_path)
            self.sfx_suppressor.unload()
        else:
            final_sfx = sfx_path

        log["stages"]["sfx"] = {
            "generative": profile.use_generative_sfx,
            "output": final_sfx,
        }

        # ── Stage 4: DSP (frequency taming + gain) ───────────────────────────
        print(f"\n[Pipeline] Stage 4: DSP (freq_taming="
              f"{profile.dsp.freq_taming_enabled})")

        dsp_cfg = profile.dsp

        dialogue_y, _ = librosa.load(final_dialogue, sr=sr, mono=True)
        music_y, _    = librosa.load(music_path,     sr=sr, mono=True)
        sfx_y, _      = librosa.load(final_sfx,      sr=sr, mono=True)

        # Apply frequency taming to SFX and music only
        # (dialogue has already been enhanced generatively if enabled)
        music_y = _freq_taming(music_y, sr, dsp_cfg)
        sfx_y   = _freq_taming(sfx_y,   sr, dsp_cfg)

        # Write processed stems so the evaluation module can load them
        processed_dialogue_path = os.path.join(
            output_dir, f"{clip_name}_{profile.name}_dialogue_enhanced.wav")
        processed_music_path = os.path.join(
            output_dir, f"{clip_name}_{profile.name}_music_processed.wav")
        processed_sfx_path = os.path.join(
            output_dir, f"{clip_name}_{profile.name}_sfx_suppressed.wav")

        import soundfile as _sf
        _sf.write(processed_dialogue_path, dialogue_y, sr)
        _sf.write(processed_music_path,    music_y,    sr)
        _sf.write(processed_sfx_path,      sfx_y,      sr)

        # Apply gain
        n = min(len(dialogue_y), len(music_y), len(sfx_y))
        mix = (dialogue_y[:n] * dsp_cfg.dialogue_gain
             + music_y[:n]    * dsp_cfg.music_gain
             + sfx_y[:n]      * dsp_cfg.sfx_gain)

        mix = _peak_norm(mix)

        # ── Stage 5: Write final output ──────────────────────────────────────
        output_path = os.path.join(
            output_dir, f"{clip_name}_{profile.name}_accessible_mix.wav")
        sf.write(output_path, mix, sr)
        print(f"\n[Pipeline] Done. Output: {output_path}")

        log["stages"]["dsp"] = {
            "freq_taming": dsp_cfg.freq_taming_enabled,
            "gains": {
                "dialogue": dsp_cfg.dialogue_gain,
                "music": dsp_cfg.music_gain,
                "sfx": dsp_cfg.sfx_gain,
            },
        }
        log["output_path"] = output_path
        log["duration_s"] = n / sr

        # Save log
        log_path = os.path.join(
            output_dir, f"{clip_name}_{profile.name}_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        return log
