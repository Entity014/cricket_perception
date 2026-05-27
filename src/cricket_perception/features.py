"""
features.py
===========
Step 2 of the Cricket Perception pipeline.

Extracts numerical feature vectors from audio segments:
    - MFCC (Mel-Frequency Cepstral Coefficients)  — timbre / texture
    - Spectral features (centroid, bandwidth, rolloff, entropy)
    - Chroma & Zero-Crossing Rate
    - RMS Energy                                   — overall loudness
    - Acoustic Complexity Index (ACI)              — bioacoustic diversity

Each function returns a 1-D numpy vector suitable for downstream
dimensionality reduction (UMAP) and clustering (HDBSCAN).

Usage:
    from cricket_perception.features import FeatureExtractor

    extractor = FeatureExtractor(sr=22050)
    vec = extractor.extract(segment)          # → 1-D float32 array
    matrix = extractor.extract_batch(segs)    # → shape [N, D]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)

# ── MFCC ───────────────────────────────────────────────────────────────────────

def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract MFCC mean + std → vector of length 2 * n_mfcc.

    Args:
        y:          Audio waveform (float32).
        sr:         Sample rate.
        n_mfcc:     Number of MFCC coefficients (default 13).
        n_fft:      FFT window size.
        hop_length: Hop size between frames.

    Returns:
        1-D array of shape [2 * n_mfcc] (mean then std of each coefficient).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


# ── Spectral Features ──────────────────────────────────────────────────────────

def extract_spectral(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract spectral summary features → vector of length 8.

    Features (mean + std for each):
        - Spectral Centroid  : perceived brightness / pitch center
        - Spectral Bandwidth : spread of frequency content
        - Spectral Rolloff   : frequency below which 85% of energy lies
        - Spectral Entropy   : disorder / complexity of spectrum

    Returns:
        1-D array of shape [8].
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    centroid    = librosa.feature.spectral_centroid(S=S, sr=sr, freq=freqs)[0]
    bandwidth   = librosa.feature.spectral_bandwidth(S=S, sr=sr, freq=freqs)[0]
    rolloff     = librosa.feature.spectral_rolloff(S=S, sr=sr, freq=freqs)[0]

    # Spectral entropy per frame
    S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    sp_entropy = scipy_entropy(S_norm + 1e-10, axis=0)

    features = []
    for arr in (centroid, bandwidth, rolloff, sp_entropy):
        features.extend([arr.mean(), arr.std()])

    return np.array(features, dtype=np.float32)


# ── Chroma & ZCR ──────────────────────────────────────────────────────────────

def extract_chroma_zcr(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract Chroma mean (12-dim) + ZCR mean+std → vector of length 14.

    Chroma captures harmonic/tonal content (useful for distinguishing
    calling song vs. aggressive irregular bursts).
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=n_fft).mean(axis=1)
    zcr    = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    return np.concatenate([chroma, [zcr.mean(), zcr.std()]]).astype(np.float32)


# ── RMS Energy ────────────────────────────────────────────────────────────────

def extract_rms(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract RMS energy statistics → vector of length 4.

    Features: [mean, std, min, max] of frame-level RMS.
    RMS is a key indicator for mortality detection (low RMS = few active crickets).
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return np.array([rms.mean(), rms.std(), rms.min(), rms.max()], dtype=np.float32)


# ── Acoustic Complexity Index (ACI) ───────────────────────────────────────────

def extract_aci(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute the Acoustic Complexity Index (ACI) → vector of length 1.

    ACI = sum(|dB[t+1] - dB[t]|) / sum(dB[t])  per frequency bin, then summed.

    High ACI → rich, varied biotic sounds (healthy colony).
    Low ACI  → monotonous or silent (dead/sick colony, or pure noise).

    Reference: Pieretti et al. (2011), Ecological Indicators.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) + 1e-10
    # ACI per frequency bin
    delta = np.abs(np.diff(S, axis=1))
    aci_per_bin = delta.sum(axis=1) / (S[:, :-1].sum(axis=1) + 1e-10)
    aci_total = aci_per_bin.sum()
    return np.array([aci_total], dtype=np.float32)


# ── Mel-Spectrogram Statistics ────────────────────────────────────────────────

def extract_melspec_stats(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Flatten mel-spectrogram statistics → vector of length n_mels * 2.

    Captures the overall frequency "shape" of the sound, useful as a
    complement to MFCCs for distinguishing soundscape states.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([mel_db.mean(axis=1),
                            mel_db.std(axis=1)]).astype(np.float32)


# ── Unified Extractor ──────────────────────────────────────────────────────────

@dataclass
class FeatureExtractor:
    """Configurable feature extractor that stacks all features into one vector.

    Args:
        sr:          Sample rate of the audio.
        n_mfcc:      Number of MFCC coefficients.
        n_mels:      Number of mel filter banks for mel-spec stats.
        n_fft:       FFT window size.
        hop_length:  Hop size between analysis frames.
        use_mfcc:       Include MFCC features.
        use_spectral:   Include spectral shape features.
        use_chroma_zcr: Include chroma + ZCR features.
        use_rms:        Include RMS energy features.
        use_aci:        Include Acoustic Complexity Index.
        use_melspec:    Include mel-spectrogram statistics.

    Example::

        extractor = FeatureExtractor(sr=22050)
        vec  = extractor.extract(segment)        # 1-D vector
        mat  = extractor.extract_batch(segments) # [N, D] matrix
        print("Feature dim:", extractor.feature_dim)
    """
    sr: int = 22_050
    n_mfcc: int = 13
    n_mels: int = 64
    n_fft: int = 2048
    hop_length: int = 512
    use_mfcc: bool = True
    use_spectral: bool = True
    use_chroma_zcr: bool = True
    use_rms: bool = True
    use_aci: bool = True
    use_melspec: bool = False   # off by default (high-dim, use for CNN inputs)

    @property
    def feature_dim(self) -> int:
        dim = 0
        if self.use_mfcc:       dim += 2 * self.n_mfcc   # 26
        if self.use_spectral:   dim += 8
        if self.use_chroma_zcr: dim += 14
        if self.use_rms:        dim += 4
        if self.use_aci:        dim += 1
        if self.use_melspec:    dim += 2 * self.n_mels    # 128
        return dim

    def extract(self, y: np.ndarray) -> np.ndarray:
        """Extract all enabled features from a single audio segment.

        Args:
            y: Waveform array (float32, shape [n_samples]).

        Returns:
            Feature vector (float32, shape [feature_dim]).
        """
        parts: list[np.ndarray] = []
        kw = dict(sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)

        if self.use_mfcc:
            parts.append(extract_mfcc(y, n_mfcc=self.n_mfcc, **kw))
        if self.use_spectral:
            parts.append(extract_spectral(y, **kw))
        if self.use_chroma_zcr:
            parts.append(extract_chroma_zcr(y, **kw))
        if self.use_rms:
            parts.append(extract_rms(y, sr=self.sr, hop_length=self.hop_length))
        if self.use_aci:
            parts.append(extract_aci(y, **kw))
        if self.use_melspec:
            parts.append(extract_melspec_stats(y, n_mels=self.n_mels, **kw))

        return np.concatenate(parts).astype(np.float32)

    def extract_batch(
        self,
        segments: list[np.ndarray],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract features from a list of audio segments.

        Args:
            segments:      List of waveform arrays.
            show_progress: Show tqdm progress bar.

        Returns:
            Feature matrix of shape [len(segments), feature_dim].
        """
        from tqdm import tqdm

        itr = tqdm(segments, desc="Extracting features", disable=not show_progress)
        vectors = [self.extract(seg) for seg in itr]
        return np.stack(vectors).astype(np.float32)
