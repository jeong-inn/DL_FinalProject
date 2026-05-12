"""
Audio segmentation utilities for CHILDES .cha files.

CHILDES audio files share the same stem as the .cha file.
Timestamps embedded in utterances (e.g. \x1512340_15678\x15) give
millisecond offsets into the audio, allowing per-utterance segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

from src.data.cha_parser import Utterance

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.aif', '.aiff', '.m4a', '.ogg')
TARGET_SR = 16_000   # wav2vec2 expects 16 kHz


def find_audio_file(cha_path: str | Path) -> Optional[Path]:
    cha_path = Path(cha_path)
    for ext in AUDIO_EXTENSIONS:
        candidate = cha_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def load_audio(audio_path: str | Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    if not _LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio loading. Run: pip install librosa")
    waveform, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return waveform, sr


def segment_utterance(waveform: np.ndarray, sr: int, start_ms: int, end_ms: int) -> np.ndarray:
    start_sample = int(start_ms / 1000 * sr)
    end_sample = int(end_ms / 1000 * sr)
    end_sample = min(end_sample, len(waveform))
    segment = waveform[start_sample:end_sample]
    return segment if len(segment) > 0 else np.zeros(sr // 10)   # 100ms silence fallback


def extract_child_segments(
    cha_path: str | Path,
    utterances: list[Utterance],
) -> Optional[list[tuple[np.ndarray, int]]]:
    """
    Return list of (waveform_segment, sample_rate) for each CHI utterance
    that has valid timestamps. Returns None if no audio file is found.
    """
    audio_path = find_audio_file(cha_path)
    if audio_path is None:
        return None

    waveform, sr = load_audio(audio_path)
    segments = []
    for utt in utterances:
        if utt.speaker == 'CHI' and utt.start_ms is not None and utt.end_ms is not None:
            seg = segment_utterance(waveform, sr, utt.start_ms, utt.end_ms)
            segments.append((seg, sr))

    return segments if segments else None


def concatenate_segments(segments: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int]:
    """Concatenate all child utterance segments into a single waveform."""
    sr = segments[0][1]
    silence = np.zeros(int(sr * 0.2))   # 200ms gap between utterances
    parts = []
    for seg, _ in segments:
        parts.append(seg)
        parts.append(silence)
    return np.concatenate(parts), sr
