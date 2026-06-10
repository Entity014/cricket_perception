"""
tests/test_augmentation.py
===========================
Unit tests for cricket_perception.augmentation.SoundscapeSynthesizer
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from cricket_perception.augmentation import SoundscapeSynthesizer

SR = 22050
DURATION_SEC = 5.0
LENGTH = int(SR * DURATION_SEC)


def test_generate_noise():
    synth = SoundscapeSynthesizer(sr=SR, segment_sec=DURATION_SEC)
    
    for noise_type in ["white", "pink", "brown"]:
        noise = synth.generate_noise(noise_type, LENGTH)
        assert len(noise) == LENGTH
        assert noise.ndim == 1
        assert np.all(np.isfinite(noise))
        
        # Standard deviation (RMS) should be close to 0.1 because of normalization
        rms = np.sqrt(np.mean(noise**2))
        assert abs(rms - 0.1) < 1e-3


def test_synthesize_one_empty():
    synth = SoundscapeSynthesizer(sr=SR, segment_sec=DURATION_SEC)
    y, label = synth.synthesize_one([], [])
    
    assert len(y) == LENGTH
    assert label == "Quiet/Background"
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0


def test_synthesize_one_mixture():
    synth = SoundscapeSynthesizer(sr=SR, segment_sec=DURATION_SEC)
    
    # 3 dummy sources
    sources = [
        np.sin(2 * np.pi * 440.0 * np.linspace(0, DURATION_SEC, LENGTH, dtype=np.float32)),
        np.sin(2 * np.pi * 880.0 * np.linspace(0, DURATION_SEC, LENGTH, dtype=np.float32)),
        np.sin(2 * np.pi * 1200.0 * np.linspace(0, DURATION_SEC, LENGTH, dtype=np.float32))
    ]
    
    labels = ["Calling Song", "Aggressive Song", "Courtship/Low Song"]
    
    y, label = synth.synthesize_one(sources, labels)
    
    assert len(y) == LENGTH
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0
    assert label in ["Calling Song", "Aggressive Song", "Courtship/Low Song", "Quiet/Background"]


@patch("pathlib.Path.exists")
@patch("librosa.load")
def test_load_and_preprocess_segment(mock_load, mock_exists):
    mock_exists.return_value = True
    # Mock librosa.load to return a dummy array of length 2 seconds
    dummy_audio = np.ones(int(SR * 2.0), dtype=np.float32)
    mock_load.return_value = (dummy_audio, SR)
    
    synth = SoundscapeSynthesizer(sr=SR, segment_sec=DURATION_SEC)
    
    # Preprocess a 2-second mock file to 5.0 seconds
    y = synth.load_and_preprocess_segment(Path("dummy.wav"), start_s=0.0, duration=DURATION_SEC)
    
    assert len(y) == LENGTH
    # The last 3 seconds should be padded with zeros
    assert np.allclose(y[int(SR * 2.0):], 0.0)
    mock_load.assert_called_once_with("dummy.wav", sr=SR, offset=0.0, duration=DURATION_SEC)


@patch("pandas.read_csv")
@patch.object(SoundscapeSynthesizer, "load_and_preprocess_segment")
def test_generate_dataset(mock_load_segment, mock_read_csv):
    synth = SoundscapeSynthesizer(sr=SR, segment_sec=DURATION_SEC, n_sources_range=(2, 3))
    
    # Mock dataframes
    df_segments = pd.DataFrame({
        "file": ["file1.wav", "file2.wav", "file3.wav"],
        "species": ["sp1", "sp2", "sp3"],
        "segment": [0, 0, 0],
        "start_s": [0.0, 0.0, 0.0],
        "cluster": [1, 2, 3]
    })
    
    df_labels = pd.DataFrame({
        "cluster": [1, 2, 3],
        "song_type": ["Calling Song", "Aggressive Song", "Calling Song"]
    })
    
    mock_read_csv.side_effect = [df_segments, df_labels]
    
    # Mock load segment to return clean sines
    mock_load_segment.return_value = np.sin(2 * np.pi * 440.0 * np.linspace(0, DURATION_SEC, LENGTH, dtype=np.float32))
    
    waveforms, labels = synth.generate_dataset(
        segments_csv=Path("segments.csv"),
        labels_csv=Path("labels.csv"),
        dataset_dir=Path("dataset"),
        n_samples=10
    )
    
    assert len(waveforms) == 10
    assert len(labels) == 10
    for y, lbl in zip(waveforms, labels):
        assert len(y) == LENGTH
        assert lbl in ["Calling Song", "Aggressive Song", "Quiet/Background"]
