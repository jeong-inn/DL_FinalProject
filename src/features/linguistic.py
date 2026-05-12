"""
Linguistic feature extraction for SLI detection.

Features are grounded in speech-language pathology literature:
- MLU-w  (Leadholm & Miller 1992; Rice & Wexler 1996)
- TTR / MATTR / CTTR  (Templin 1957; Covington & McFall 2010)
- NDW / TNW  (Miller 1981)
"""

import math
from collections import Counter

import numpy as np


FEATURE_NAMES = [
    "mlu_w",
    "ttr",
    "mattr",
    "cttr",
    "ndw",
    "tnw",
    "n_utterances",
    "utt_len_mean",
    "utt_len_std",
    "utt_len_max",
    "prop_short_utt",   # proportion of 1-2 word utterances
    "prop_long_utt",    # proportion of 5+ word utterances
]


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in text.split() if w.isalpha()]


def compute_mlu(utterances: list[str]) -> float:
    """Mean Length of Utterance in words."""
    lengths = [len(u.split()) for u in utterances if u.strip()]
    return float(np.mean(lengths)) if lengths else 0.0


def compute_ttr(utterances: list[str]) -> float:
    """Type-Token Ratio (sensitive to sample size)."""
    tokens = [t for u in utterances for t in _tokenize(u)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_mattr(utterances: list[str], window: int = 50) -> float:
    """Moving Average Type-Token Ratio — more stable than raw TTR."""
    tokens = [t for u in utterances for t in _tokenize(u)]
    if len(tokens) < window:
        return compute_ttr(utterances)
    ttrs = [
        len(set(tokens[i : i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return float(np.mean(ttrs))


def compute_ndw(utterances: list[str]) -> int:
    """Number of Different Words."""
    tokens = [t for u in utterances for t in _tokenize(u)]
    return len(set(tokens))


def compute_tnw(utterances: list[str]) -> int:
    """Total Number of Words."""
    return sum(len(_tokenize(u)) for u in utterances)


def compute_cttr(utterances: list[str]) -> float:
    """Corrected TTR = NDW / sqrt(2 * TNW)."""
    ndw = compute_ndw(utterances)
    tnw = compute_tnw(utterances)
    if tnw == 0:
        return 0.0
    return ndw / math.sqrt(2 * tnw)


def extract_features(utterances: list[str]) -> np.ndarray:
    """
    Extract all linguistic features from a list of clean utterances.
    Returns a 1-D numpy array of length len(FEATURE_NAMES).
    """
    non_empty = [u for u in utterances if u.strip()]
    if not non_empty:
        return np.zeros(len(FEATURE_NAMES))

    lengths = [len(u.split()) for u in non_empty]

    mlu_w = float(np.mean(lengths))
    ttr = compute_ttr(non_empty)
    mattr = compute_mattr(non_empty)
    cttr = compute_cttr(non_empty)
    ndw = float(compute_ndw(non_empty))
    tnw = float(compute_tnw(non_empty))
    n_utt = float(len(non_empty))
    utt_len_mean = mlu_w
    utt_len_std = float(np.std(lengths))
    utt_len_max = float(max(lengths))
    prop_short = sum(1 for l in lengths if l <= 2) / len(lengths)
    prop_long = sum(1 for l in lengths if l >= 5) / len(lengths)

    return np.array([
        mlu_w, ttr, mattr, cttr,
        ndw, tnw, n_utt,
        utt_len_mean, utt_len_std, utt_len_max,
        prop_short, prop_long,
    ])
