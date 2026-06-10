"""
audio_utils.py
==============
Step 1 of the Cricket Perception pipeline.

Responsibilities:
    - Load audio files (WAV, FLAC, MP3)
    - Segment into fixed-length frames
    - **Stream** long audio files chunk-by-chunk (hours-long recordings)
    - Optional denoising (spectral subtraction via noisereduce)

Usage:
    from cricket_perception.audio_utils import load_audio, segment_audio, denoise_audio

    y, sr = load_audio("recording.wav")
    segments = segment_audio(y, sr, segment_sec=5.0)
    y_clean = denoise_audio(y, sr)

    # Stream a long file without loading into RAM:
    from cricket_perception.audio_utils import stream_audio

    for window, start_sec in stream_audio("24h_recording.wav", window_sec=2.5):
        process(window)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Default constants ──────────────────────────────────────────────────────────
DEFAULT_SR: int = 22_050          # resample rate (Hz)
DEFAULT_SEGMENT_SEC: float = 5.0  # segment length (seconds)
DEFAULT_HOP_SEC: float = 2.5      # hop between segments (50% overlap)
DEFAULT_READ_BLOCK_SEC: float = 30.0  # internal I/O read block size


# ── Load ───────────────────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    sr: int = DEFAULT_SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to `sr`.

    Args:
        path:  Path to audio file (.wav, .flac, .mp3, .ogg).
        sr:    Target sample rate in Hz.
        mono:  If True, convert stereo → mono by averaging channels.

    Returns:
        (y, sr): Audio time-series (float32, shape [n_samples]) and sample rate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    y, orig_sr = librosa.load(str(path), sr=sr, mono=mono, dtype=np.float32)
    logger.debug("Loaded '%s' | sr=%d | duration=%.2fs", path.name, sr, len(y) / sr)
    return y, sr


def load_folder(
    folder: str | Path,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg"),
    sr: int = DEFAULT_SR,
) -> dict[str, tuple[np.ndarray, int]]:
    """Load all audio files in a folder recursively.

    Returns:
        dict mapping relative-path-string → (y, sr)
    """
    folder = Path(folder)
    audio_files = [p for ext in extensions for p in folder.rglob(f"*{ext}")]
    audio_files.sort()

    logger.info("Found %d audio files in '%s'", len(audio_files), folder)
    results: dict[str, tuple[np.ndarray, int]] = {}
    for p in audio_files:
        try:
            y, sr_ = load_audio(p, sr=sr)
            results[str(p.relative_to(folder))] = (y, sr_)
        except Exception as exc:
            logger.warning("Skipping '%s': %s", p, exc)
    return results


# ── Segment ────────────────────────────────────────────────────────────────────

def segment_audio(
    y: np.ndarray,
    sr: int,
    segment_sec: float = DEFAULT_SEGMENT_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    pad_last: bool = True,
) -> list[np.ndarray]:
    """Split a waveform into overlapping fixed-length segments.

    Args:
        y:           Audio time-series.
        sr:          Sample rate.
        segment_sec: Length of each segment in seconds.
        hop_sec:     Hop size in seconds (controls overlap).
        pad_last:    If True, zero-pad the final segment to full length.

    Returns:
        List of numpy arrays, each of shape [segment_sec * sr].
    """
    segment_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)
    total = len(y)

    segments: list[np.ndarray] = []
    start = 0
    while start < total:
        end = start + segment_len
        chunk = y[start:end]

        if len(chunk) < segment_len:
            if pad_last:
                chunk = np.pad(chunk, (0, segment_len - len(chunk)))
            else:
                break  # discard incomplete last segment

        segments.append(chunk.astype(np.float32))
        start += hop_len

    logger.debug(
        "Segmented %d samples → %d segments (%.1fs each, %.1fs hop)",
        total, len(segments), segment_sec, hop_sec,
    )
    return segments


# ── Denoise ────────────────────────────────────────────────────────────────────

def denoise_audio(
    y: np.ndarray,
    sr: int,
    stationary: bool = False,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    """Apply spectral subtraction noise reduction.

    Uses the `noisereduce` library which estimates noise from a
    short baseline section and subtracts it in the frequency domain.

    Args:
        y:              Input waveform.
        sr:             Sample rate.
        stationary:     If True, assume stationary noise (fan/AC hum).
                        If False, use adaptive non-stationary mode.
        prop_decrease:  How much to attenuate noise (0.0–1.0).

    Returns:
        Denoised waveform (same shape as input).
    """
    try:
        import noisereduce as nr
        y_clean = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
        )
        return y_clean.astype(np.float32)
    except ImportError:
        logger.warning("noisereduce not installed — skipping denoising")
        return y


# ── Utilities ──────────────────────────────────────────────────────────────────

def save_audio(y: np.ndarray, sr: int, path: str | Path) -> None:
    """Save a waveform to disk as WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)
    logger.debug("Saved audio → '%s'", path)


def get_duration(y: np.ndarray, sr: int) -> float:
    """Return duration of a waveform in seconds."""
    return len(y) / sr


def trim_silence(
    y: np.ndarray,
    top_db: float = 30.0,
) -> np.ndarray:
    """Trim leading/trailing silence using librosa's energy-based trimming."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


# ── Streaming (memory-efficient long-file support) ─────────────────────────────

def get_file_duration(path: str | Path) -> float:
    """Get the duration of an audio file in seconds without loading it.

    Args:
        path: Path to audio file.

    Returns:
        Duration in seconds.
    """
    info = sf.info(str(path))
    return info.duration


def stream_audio(
    path: str | Path,
    sr: int = DEFAULT_SR,
    window_sec: float = 2.5,
    hop_sec: float = 1.0,
    read_block_sec: float = DEFAULT_READ_BLOCK_SEC,
) -> Iterator[tuple[np.ndarray, float]]:
    """Stream an audio file as overlapping windows without loading it all into RAM.

    Reads the file in blocks of ``read_block_sec`` seconds using
    ``soundfile.SoundFile``, resamples on-the-fly if needed, and yields
    sliding windows of ``window_sec`` seconds advancing by ``hop_sec``.

    Peak memory ≈ ``(read_block_sec + window_sec) × sr × 4`` bytes,
    regardless of total file length.  A 24-hour recording at 22 050 Hz
    uses ~2.9 MB instead of ~7.2 GB.

    Args:
        path:            Path to audio file (.wav, .flac, .mp3, .ogg).
        sr:              Target sample rate in Hz.
        window_sec:      Length of each output window in seconds.
        hop_sec:         Hop between consecutive windows in seconds.
        read_block_sec:  Internal I/O block size in seconds (larger = fewer
                         disk reads but more RAM; 30 s is a good default).

    Yields:
        ``(window, start_sec)`` tuples where *window* is a float32 array
        of shape ``[int(window_sec * sr)]`` and *start_sec* is the window's
        offset from the beginning of the file.

    Example::

        from cricket_perception.audio_utils import stream_audio

        for window, t in stream_audio("farm_24h.wav", window_sec=2.5, hop_sec=1.0):
            features = extractor.extract(window)
            print(f"t={t:.1f}s  features={features.shape}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    with sf.SoundFile(str(path)) as f:
        orig_sr = f.samplerate
        read_block_frames = int(read_block_sec * orig_sr)

        buffer = np.empty(0, dtype=np.float32)
        position_s = 0.0       # current window start time (in seconds)
        windows_yielded = 0
        total_frames = f.frames

        logger.info(
            "Streaming '%s' | orig_sr=%d target_sr=%d | "
            "duration=%.1fs | window=%.1fs hop=%.1fs | block=%.0fs",
            path.name, orig_sr, sr,
            total_frames / orig_sr, window_sec, hop_sec, read_block_sec,
        )

        while True:
            # ── Read a block from disk ──────────────────────────────────
            block = f.read(read_block_frames, dtype="float32")
            eof = len(block) == 0

            if not eof:
                # Stereo → mono
                if block.ndim > 1:
                    block = block.mean(axis=1)

                # Resample if needed
                if orig_sr != sr:
                    block = librosa.resample(
                        block, orig_sr=orig_sr, target_sr=sr,
                    ).astype(np.float32)

                buffer = np.concatenate([buffer, block])

            # ── Yield windows from buffer ───────────────────────────────
            while len(buffer) >= window_samples:
                yield buffer[:window_samples].copy(), position_s
                buffer = buffer[hop_samples:]
                position_s += hop_sec
                windows_yielded += 1

            if eof:
                break

    logger.info(
        "Stream complete: %d windows over %.1fs audio from '%s'",
        windows_yielded, total_frames / orig_sr, path.name,
    )
