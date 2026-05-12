"""
Acoustic feature extraction for SLI detection.

Two levels of features:
1. Prosodic features (handcrafted, interpretable)
   — speech rate, pause rate, F0 statistics, energy
2. wav2vec2 embeddings (learned, high-capacity)
   — facebook/wav2vec2-base, mean-pooled across time
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


PROSODIC_FEATURE_NAMES = [
    "speech_rate",        # words per second
    "pause_rate",         # pauses per utterance
    "mean_pause_dur",     # mean pause duration (s)
    "f0_mean",            # mean fundamental frequency (Hz)
    "f0_std",             # F0 standard deviation
    "f0_range",           # F0 range (max - min)
    "energy_mean",        # RMS energy mean
    "energy_std",         # RMS energy std
    "speaking_ratio",     # fraction of time with speech (vs silence)
    "mfcc_mean_1",        # first 4 MFCC means (spectral shape)
    "mfcc_mean_2",
    "mfcc_mean_3",
    "mfcc_mean_4",
]

WAV2VEC2_DIM = 768
WAV2VEC2_MODEL = "facebook/wav2vec2-base"


# ── Prosodic features ──────────────────────────────────────────────────────

def extract_prosodic(
    segments: list[tuple[np.ndarray, int]],
    utterance_texts: list[str],
) -> np.ndarray:
    """
    segments : list of (waveform, sample_rate) per CHI utterance
    utterance_texts : corresponding clean text strings
    Returns a 1-D array of length len(PROSODIC_FEATURE_NAMES).
    """
    if not _LIBROSA_AVAILABLE:
        raise ImportError("librosa is required: pip install librosa")

    if not segments:
        return np.zeros(len(PROSODIC_FEATURE_NAMES))

    sr = segments[0][1]
    total_words = sum(len(t.split()) for t in utterance_texts)
    total_dur = sum(len(seg) / sr for seg, _ in segments)

    speech_rate = total_words / max(total_dur, 1e-6)

    # pause detection via RMS silence
    pause_counts, pause_durs = [], []
    energy_vals = []
    for seg, _ in segments:
        rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=256)[0]
        energy_vals.extend(rms.tolist())
        threshold = np.percentile(rms, 20)
        is_silence = rms < threshold
        pauses = _count_runs(is_silence, min_frames=3)
        pause_counts.append(pauses['count'])
        pause_durs.extend(pauses['durations'])

    pause_rate = np.mean(pause_counts)
    mean_pause_dur = np.mean(pause_durs) * 256 / sr if pause_durs else 0.0

    # F0 (pitch) via pyin
    f0_all = []
    for seg, _ in segments:
        if len(seg) < sr // 4:
            continue
        f0, voiced_flag, _ = librosa.pyin(
            seg, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
        f0_all.extend(f0_voiced[~np.isnan(f0_voiced)].tolist())

    f0_mean = float(np.mean(f0_all)) if f0_all else 0.0
    f0_std = float(np.std(f0_all)) if f0_all else 0.0
    f0_range = float(np.max(f0_all) - np.min(f0_all)) if len(f0_all) > 1 else 0.0

    energy_arr = np.array(energy_vals)
    energy_mean = float(np.mean(energy_arr))
    energy_std = float(np.std(energy_arr))

    threshold_global = np.percentile(energy_arr, 20) if len(energy_arr) > 0 else 0
    speaking_ratio = float(np.mean(energy_arr > threshold_global))

    # MFCCs (first 4 coefficients, mean over time)
    mfcc_means = np.zeros(4)
    all_segs = np.concatenate([seg for seg, _ in segments])
    if len(all_segs) > 512:
        mfccs = librosa.feature.mfcc(y=all_segs, sr=sr, n_mfcc=4)
        mfcc_means = mfccs.mean(axis=1)

    return np.array([
        speech_rate, pause_rate, mean_pause_dur,
        f0_mean, f0_std, f0_range,
        energy_mean, energy_std, speaking_ratio,
        *mfcc_means,
    ])


def _count_runs(mask: np.ndarray, min_frames: int = 3) -> dict:
    """Count and measure runs of True in a boolean array."""
    count = 0
    durations = []
    run = 0
    for val in mask:
        if val:
            run += 1
        else:
            if run >= min_frames:
                count += 1
                durations.append(run)
            run = 0
    if run >= min_frames:
        count += 1
        durations.append(run)
    return {'count': count, 'durations': durations}


# ── wav2vec2 embeddings ────────────────────────────────────────────────────

class Wav2Vec2Encoder:
    """
    Encodes child speech segments into a fixed-size embedding using
    facebook/wav2vec2-base.  Mean-pools across time frames.
    """

    def __init__(self, model_name: str = WAV2VEC2_MODEL):
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        import torch

        self.device = (
            torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, segments: list[tuple[np.ndarray, int]]) -> np.ndarray:
        """
        Returns a (WAV2VEC2_DIM,) embedding for a list of audio segments.
        Segments are concatenated before encoding (max 30s to avoid OOM).
        """
        import torch

        if not segments:
            return np.zeros(WAV2VEC2_DIM)

        sr = segments[0][1]
        audio = np.concatenate([seg for seg, _ in segments])

        # truncate to 30 seconds
        max_samples = sr * 30
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        inputs = self.processor(audio, sampling_rate=sr, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state  # (1, T, 768)

        return hidden.squeeze(0).mean(0).cpu().numpy()   # (768,)

    def encode_batch(self, segment_lists: list[list[tuple[np.ndarray, int]]]) -> np.ndarray:
        """Encode a list of children, returning (N, WAV2VEC2_DIM)."""
        return np.vstack([self.encode(segs) for segs in segment_lists])
