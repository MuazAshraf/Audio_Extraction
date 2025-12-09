from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import librosa


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    power: float = 2.0,
) -> np.ndarray:
    """Compute Mel spectrogram (power) with librosa.

    Returns an array of shape (n_mels, n_frames).
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    return S


def compute_log_mel(
    y: np.ndarray,
    sr: int,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    power: float = 2.0,
    ref: float | np.ndarray | None = None,
) -> np.ndarray:
    """Compute log-Mel spectrogram in dB.

    Returns an array of shape (n_mels, n_frames).
    """
    S = compute_mel_spectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    S_db = librosa.power_to_db(S, ref=np.max if ref is None else ref)
    return S_db


def compute_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    d1: bool = False,
    d2: bool = False,
) -> np.ndarray:
    """Compute MFCCs (optionally with delta and delta-delta).

    Returns shape:
      - (n_mfcc, n_frames) if d1==False and d2==False
      - (n_mfcc*2, n_frames) if d1==True and d2==False (stacked [mfcc; d_mfcc])
      - (n_mfcc*3, n_frames) if d1==True and d2==True (stacked [mfcc; d_mfcc; dd_mfcc])
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    feats = [mfcc]
    if d1:
        d_mfcc = librosa.feature.delta(mfcc, order=1)
        feats.append(d_mfcc)
    if d2:
        # If first delta wasn't requested, compute from base MFCCs
        base = feats[-1] if d1 else mfcc
        dd_mfcc = librosa.feature.delta(base, order=2)
        feats.append(dd_mfcc)

    if len(feats) == 1:
        return feats[0]
    return np.vstack(feats)
