"""
tests/test_features.py
======================
Unit tests for cricket_perception.features

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import numpy as np
import pytest

# Synthetic audio helpers
SR = 22_050
DURATION_S = 5.0
N_SAMPLES = int(SR * DURATION_S)


def make_sine(freq: float = 440.0, sr: int = SR, duration: float = DURATION_S) -> np.ndarray:
    """Generate a pure sine wave for deterministic testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def make_noise(sr: int = SR, duration: float = DURATION_S, seed: int = 0) -> np.ndarray:
    """Generate white noise."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32)


def make_silence(sr: int = SR, duration: float = DURATION_S) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


# ── MFCC ───────────────────────────────────────────────────────────────────────

class TestExtractMFCC:
    from cricket_perception.features import extract_mfcc

    def test_output_shape_default(self):
        from cricket_perception.features import extract_mfcc
        y = make_sine()
        vec = extract_mfcc(y, sr=SR)
        assert vec.shape == (26,), f"Expected (26,), got {vec.shape}"  # 2 * 13

    def test_output_shape_custom_n_mfcc(self):
        from cricket_perception.features import extract_mfcc
        y = make_sine()
        vec = extract_mfcc(y, sr=SR, n_mfcc=20)
        assert vec.shape == (40,)  # 2 * 20

    def test_output_dtype(self):
        from cricket_perception.features import extract_mfcc
        vec = extract_mfcc(make_sine(), sr=SR)
        assert vec.dtype == np.float32

    def test_no_nan_inf(self):
        from cricket_perception.features import extract_mfcc
        vec = extract_mfcc(make_noise(), sr=SR)
        assert np.all(np.isfinite(vec)), "MFCC contains NaN or Inf"

    def test_silence_is_all_zeros(self):
        from cricket_perception.features import extract_mfcc
        # MFCCs of silence should not crash (may be near-zero)
        vec = extract_mfcc(make_silence(), sr=SR)
        assert np.all(np.isfinite(vec))


# ── Spectral ───────────────────────────────────────────────────────────────────

class TestExtractSpectral:
    def test_output_shape(self):
        from cricket_perception.features import extract_spectral
        vec = extract_spectral(make_sine(), sr=SR)
        assert vec.shape == (8,)

    def test_dtype(self):
        from cricket_perception.features import extract_spectral
        assert extract_spectral(make_noise(), sr=SR).dtype == np.float32

    def test_centroid_sine(self):
        from cricket_perception.features import extract_spectral
        # Spectral centroid of 440 Hz sine should be near 440
        vec = extract_spectral(make_sine(freq=440.0), sr=SR)
        centroid_mean = vec[0]  # first element = centroid mean
        assert 300 < centroid_mean < 600, f"Centroid {centroid_mean} not near 440 Hz"


# ── RMS Energy ────────────────────────────────────────────────────────────────

class TestExtractRMS:
    def test_output_shape(self):
        from cricket_perception.features import extract_rms
        vec = extract_rms(make_sine(), sr=SR)
        assert vec.shape == (4,)

    def test_silence_rms_is_zero(self):
        from cricket_perception.features import extract_rms
        vec = extract_rms(make_silence(), sr=SR)
        assert vec[0] < 1e-6, "RMS of silence should be ~0"

    def test_louder_signal_higher_rms(self):
        from cricket_perception.features import extract_rms
        quiet = make_sine() * 0.1
        loud  = make_sine() * 1.0
        rms_quiet = extract_rms(quiet, sr=SR)[0]
        rms_loud  = extract_rms(loud,  sr=SR)[0]
        assert rms_loud > rms_quiet


# ── ACI ───────────────────────────────────────────────────────────────────────

class TestExtractACI:
    def test_output_shape(self):
        from cricket_perception.features import extract_aci
        vec = extract_aci(make_sine(), sr=SR)
        assert vec.shape == (1,)

    def test_noise_higher_aci_than_silence(self):
        from cricket_perception.features import extract_aci
        aci_silence = extract_aci(make_silence(), sr=SR)[0]
        aci_noise   = extract_aci(make_noise(),   sr=SR)[0]
        # Silence has near-zero ACI; any real signal should be higher
        assert aci_noise > aci_silence, (
            f"Noise ACI ({aci_noise:.2f}) should > Silence ACI ({aci_silence:.2f})"
        )


# ── FeatureExtractor ───────────────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_feature_dim_default(self):
        from cricket_perception.features import FeatureExtractor
        ext = FeatureExtractor(sr=SR)
        # MFCC(26) + Spectral(8) + Chroma+ZCR(14) + RMS(4) + ACI(1) = 53
        assert ext.feature_dim == 53

    def test_extract_shape(self):
        from cricket_perception.features import FeatureExtractor
        ext = FeatureExtractor(sr=SR)
        vec = ext.extract(make_sine())
        assert vec.shape == (ext.feature_dim,)

    def test_extract_dtype(self):
        from cricket_perception.features import FeatureExtractor
        ext = FeatureExtractor(sr=SR)
        vec = ext.extract(make_noise())
        assert vec.dtype == np.float32

    def test_extract_batch_shape(self):
        from cricket_perception.features import FeatureExtractor
        ext = FeatureExtractor(sr=SR)
        segments = [make_sine(), make_noise(), make_sine(880.0)]
        mat = ext.extract_batch(segments, show_progress=False)
        assert mat.shape == (3, ext.feature_dim)

    def test_disable_features(self):
        from cricket_perception.features import FeatureExtractor
        ext = FeatureExtractor(
            sr=SR,
            use_mfcc=True,
            use_spectral=False,
            use_chroma_zcr=False,
            use_rms=False,
            use_aci=False,
        )
        assert ext.feature_dim == 26  # only MFCC
        vec = ext.extract(make_sine())
        assert vec.shape == (26,)

    def test_different_signals_differ(self):
        from cricket_perception.features import FeatureExtractor
        ext = FeatureExtractor(sr=SR)
        v_sine  = ext.extract(make_sine(440.0))
        v_noise = ext.extract(make_noise())
        assert not np.allclose(v_sine, v_noise), "Sine and noise should have different features"


# ── Dolbear's Law ─────────────────────────────────────────────────────────────

class TestDolbear:
    def test_known_value(self):
        from cricket_perception.behavior import temperature_from_chirps
        # N=80 chirps/15s → T_F = 50 + (80-40)/4 = 60°F = 15.56°C
        t = temperature_from_chirps(80.0)
        assert abs(t - 15.56) < 0.1

    def test_temperature_correction_factor(self):
        from cricket_perception.behavior import chirp_rate_correction_factor
        # Same temp → factor = 1.0
        factor = chirp_rate_correction_factor(28.0, reference_temp_c=28.0)
        assert abs(factor - 1.0) < 1e-6

        # 10°C cooler → factor ≈ 2.0 (Q10=2)
        factor_cold = chirp_rate_correction_factor(18.0, reference_temp_c=28.0)
        assert abs(factor_cold - 2.0) < 0.01
