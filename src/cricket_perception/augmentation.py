"""
augmentation.py
===============
Step 2b of the Cricket Perception pipeline.

Provides SoundscapeSynthesizer for mixture augmentation to bridge the domain gap
between single-insect recordings and multi-source commercial farm soundscapes.

Features:
    - Multi-source mixing (combining multiple segments with time jitter and gain)
    - Synthetic noise generation (white, pink, and brown noise)
    - Configurable source density range
    - Energy-based composition labeling (dominant label selection)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

logger = logging.getLogger(__name__)

@dataclass
class SoundscapeSynthesizer:
    """สร้าง synthetic multi-cricket soundscapes จาก single-insect recordings."""

    sr: int = 22050
    segment_sec: float = 5.0          # ความยาว output
    n_sources_range: tuple[int, int] = (3, 50)  # ช่วงจำนวนตัวที่ mix
    gain_range_db: tuple[float, float] = (-12.0, 6.0) # random gain per source
    offset_jitter_sec: float = 2.5     # random start offset
    noise_types: tuple[str, ...] = ("pink", "white", "brown") # ประเภท synthetic noise
    snr_range_db: tuple[float, float] = (5.0, 30.0)    # signal-to-noise ratio

    def generate_noise(self, noise_type: str, length: int) -> np.ndarray:
        """สร้าง synthetic noise (white, pink, brown) ความยาวตามต้องการ."""
        if noise_type == "white":
            noise = np.random.normal(0.0, 1.0, length)
        elif noise_type == "pink":
            # FFT method for pink noise (1/f power spectrum)
            white = np.random.normal(0.0, 1.0, length)
            white_fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(length)
            # Avoid division by zero at freq=0
            freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
            scaling = 1.0 / np.sqrt(freqs)
            scaling[0] = 0.0
            pink_fft = white_fft * scaling
            noise = np.fft.irfft(pink_fft, n=length)
        elif noise_type == "brown":
            # Cumulative sum of white noise (1/f^2 power spectrum)
            white = np.random.normal(0.0, 1.0, length)
            noise = np.cumsum(white)
            # Detrend by removing the mean
            noise = noise - np.mean(noise)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Normalize to standard RMS of 0.1 for stability
        rms = np.sqrt(np.mean(noise**2))
        if rms > 1e-6:
            noise = noise * (0.1 / rms)
        return noise.astype(np.float32)

    def load_and_preprocess_segment(self, file_path: Path, start_s: float, duration: float) -> np.ndarray:
        """โหลดและปรับความยาวเสียงจากไฟล์ต้นฉบับ."""
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # load segment using librosa (handles resampling internally)
        y, _ = librosa.load(str(file_path), sr=self.sr, offset=start_s, duration=duration)

        # Pad or truncate to ensure exact length
        target_len = int(duration * self.sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            y = y[:target_len]

        return y.astype(np.float32)

    def synthesize_one(self, sources: list[np.ndarray], labels: list[str]) -> tuple[np.ndarray, str]:
        """Mix N sources → one synthetic soundscape + label."""
        if not sources:
            # Fallback to pure noise if no sources provided
            length = int(self.segment_sec * self.sr)
            noise_type = np.random.choice(self.noise_types)
            noise = self.generate_noise(noise_type, length)
            peak = np.max(np.abs(noise))
            if peak > 1e-6:
                noise = noise / peak * 0.99
            return noise, "Quiet/Background"

        length = len(sources[0])
        y_sum = np.zeros(length, dtype=np.float32)
        
        # Calculate individual source gains and energies
        gains = []
        energies = []
        for src in sources:
            # Random gain in dB
            g_db = np.random.uniform(self.gain_range_db[0], self.gain_range_db[1])
            scale = 10.0 ** (g_db / 20.0)
            
            # Apply time jitter (circular roll)
            shift = np.random.randint(0, length)
            src_jittered = np.roll(src, shift)
            
            # Scale and sum
            src_scaled = src_jittered * scale
            y_sum += src_scaled
            
            gains.append(scale)
            # Energy is proportional to sum of squares
            energies.append(np.sum(src_scaled ** 2))

        # Composition-based labeling logic:
        # We group energies by class label to find the dominant behavior.
        label_energies = {}
        for label, energy in zip(labels, energies):
            label_energies[label] = label_energies.get(label, 0.0) + energy

        total_cricket_energy = sum(energies)
        
        if total_cricket_energy < 1e-6:
            final_label = "Quiet/Background"
        else:
            # If Aggressive Song makes up a significant portion (> 30%) of cricket energy,
            # we classify the soundscape as Aggressive Song (hunger warning signal).
            if label_energies.get("Aggressive Song", 0.0) / total_cricket_energy >= 0.3:
                final_label = "Aggressive Song"
            elif label_energies.get("Calling Song", 0.0) / total_cricket_energy >= 0.4:
                final_label = "Calling Song"
            elif label_energies.get("Courtship/Low Song", 0.0) / total_cricket_energy >= 0.4:
                final_label = "Courtship/Low Song"
            else:
                # Majority vote fallback
                final_label = max(label_energies, key=label_energies.get)

        # Inject background synthetic noise
        noise_type = np.random.choice(self.noise_types)
        snr = np.random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        
        signal_power = np.mean(y_sum ** 2)
        if signal_power < 1e-6:
            # If no signal, just return scaled noise
            noise = self.generate_noise(noise_type, length)
            peak = np.max(np.abs(noise))
            if peak > 1e-6:
                noise = noise / peak * 0.99
            return noise, "Quiet/Background"

        # SNR = 10 * log10(P_signal / P_noise) => P_noise = P_signal / (10 ** (SNR / 10))
        noise_power = signal_power / (10.0 ** (snr / 10.0))
        
        noise = self.generate_noise(noise_type, length)
        current_noise_power = np.mean(noise ** 2)
        if current_noise_power > 1e-6:
            noise = noise * np.sqrt(noise_power / current_noise_power)
            
        y_mixed = y_sum + noise
        
        # Peak normalization to 0.99 to prevent clipping
        peak = np.max(np.abs(y_mixed))
        if peak > 1e-6:
            y_mixed = y_mixed / peak * 0.99
            
        return np.clip(y_mixed, -1.0, 1.0).astype(np.float32), final_label

    def generate_dataset(
        self,
        segments_csv: Path,
        labels_csv: Path,
        dataset_dir: Path,
        n_samples: int = 500,
    ) -> tuple[list[np.ndarray], list[str]]:
        """สร้าง synthetic dataset (waveforms + labels) จาก segments metadata และ labels mapping."""
        # 1. Load segments and labels dataframes
        df_seg = pd.read_csv(segments_csv)
        df_lbl = pd.read_csv(labels_csv)

        # Map cluster ID to song_type
        cluster_to_type = dict(zip(df_lbl["cluster"], df_lbl["song_type"]))
        # Noise or outliers (-1) mapped to Noise
        cluster_to_type[-1] = "Noise"

        # Add song_type column to segments df
        df_seg["song_type"] = df_seg["cluster"].map(cluster_to_type).fillna("Noise")

        # Exclude Noise and Quiet/Background from clean sources to mix
        # since we want to mix actual bioacoustic signals, and we inject noise separately
        df_clean = df_seg[~df_seg["song_type"].isin(["Noise", "Quiet/Background"])].reset_index(drop=True)

        if len(df_clean) == 0:
            raise ValueError("No clean cricket song segments found in metadata to synthesize soundscapes.")

        # Cache segments to avoid re-loading files repeatedly
        cached_waveforms = []
        cached_labels = []
        logger.info("Caching clean segments from dataset...")
        
        for _, row in df_clean.iterrows():
            file_path = dataset_dir / row["file"]
            try:
                y = self.load_and_preprocess_segment(file_path, row["start_s"], self.segment_sec)
                cached_waveforms.append(y)
                cached_labels.append(row["song_type"])
            except Exception as e:
                logger.warning("Failed to load segment from %s: %s", file_path, e)

        if not cached_waveforms:
            raise RuntimeError("Failed to load any audio segments for dataset generation.")

        synthesized_waveforms = []
        synthesized_labels = []

        logger.info("Synthesizing %d soundscape samples...", n_samples)
        for _ in range(n_samples):
            # Select random number of sources to mix
            n_sources = np.random.randint(self.n_sources_range[0], self.n_sources_range[1] + 1)
            
            # Select random sources from cache
            idxs = np.random.choice(len(cached_waveforms), size=min(n_sources, len(cached_waveforms)), replace=True)
            sources = [cached_waveforms[i] for i in idxs]
            labels = [cached_labels[i] for i in idxs]
            
            y_synth, lbl_synth = self.synthesize_one(sources, labels)
            synthesized_waveforms.append(y_synth)
            synthesized_labels.append(lbl_synth)

        return synthesized_waveforms, synthesized_labels
