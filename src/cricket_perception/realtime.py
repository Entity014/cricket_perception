"""
realtime.py
===========
Step 5 of the Cricket Perception pipeline.

Real-time audio monitoring that chains:
    FeatureExtractor → SongTypeClassifier → BehaviorMonitor

Supports two modes:
    1. ``RealTimeMonitor.from_file()``  — process a WAV file as if streaming
    2. ``RealTimeMonitor.from_mic()``   — live microphone input

The monitor runs inference every ``hop_s`` seconds on a sliding window
of ``window_s`` seconds, so alerts fire in near-real-time.

Usage (file mode)::

    from cricket_perception.realtime import RealTimeMonitor

    monitor = RealTimeMonitor.from_file(
        wav_path="farm_recording.wav",
        classifier_path="results/song_classifier.pkl",
        rms_baseline=0.014,
        aci_baseline=535.0,
    )
    for result in monitor.stream():
        if result.alerts:
            print(result.alerts)

Usage (microphone mode)::

    monitor = RealTimeMonitor.from_mic(
        device=None,          # None = system default mic
        classifier_path="results/song_classifier.pkl",
        rms_baseline=0.014,
        aci_baseline=535.0,
    )
    monitor.start(on_alert=lambda r: print(r.alerts))
    # ... runs in background ...
    monitor.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)


# ── Streaming Inference Result ────────────────────────────────────────────────


@dataclass
class StreamResult:
    """Result of a single inference window in the real-time stream."""

    timestamp: datetime
    window_start_s: float          # seconds from stream start
    window_dur_s: float

    # Classification
    song_type: str                 # predicted song type
    song_type_probas: dict[str, float]  # {song_type: probability}
    confidence: float              # max probability

    # Acoustic features
    rms: float
    aci: float
    temperature_c: float | None

    # Behavior
    hungry: bool
    mortality_risk: bool
    alerts: list[str]

    @property
    def is_critical(self) -> bool:
        return bool(self.alerts)

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        alert_str = " ⚠️ " + " | ".join(self.alerts) if self.alerts else " ✅"
        return (
            f"[{ts}] {self.song_type:<22s} "
            f"conf={self.confidence:.0%} "
            f"RMS={self.rms:.4f} ACI={self.aci:.0f}"
            f"{alert_str}"
        )


# ── Real-Time Monitor ─────────────────────────────────────────────────────────


@dataclass
class RealTimeMonitor:
    """Real-time acoustic monitor for cricket farms.

    Chains feature extraction → song-type classification → behavior analysis
    on a sliding audio window, suitable for both file playback and live mic.

    Args:
        classifier_path:   Path to a trained SongTypeClassifier pickle.
        rms_baseline:      Healthy colony RMS baseline.
        aci_baseline:      Healthy colony ACI baseline.
        aggressive_threshold: Fraction of aggressive song that triggers hunger.
        rms_low_threshold: RMS ratio below which mortality alert fires.
        aci_low_threshold: ACI ratio below which mortality alert fires.
        aggressive_song_types: Which predicted labels count as "aggressive".
        calling_song_types:    Which predicted labels count as "calling".
        window_s:          Analysis window length (seconds).
        hop_s:             Step between consecutive windows (seconds).
        sr:                Sample rate (Hz).
        temperature_c:     Current enclosure temperature (optional).
        reference_temp_c:  Calibration temperature for Q10 correction.
    """

    classifier_path: str | Path
    rms_baseline: float = 0.014
    aci_baseline: float = 535.0

    aggressive_threshold: float = 0.35
    rms_low_threshold: float = 0.40
    aci_low_threshold: float = 0.50

    aggressive_song_types: list[str] = field(
        default_factory=lambda: ["Aggressive Song"]
    )
    calling_song_types: list[str] = field(
        default_factory=lambda: ["Calling Song"]
    )

    window_s: float = 2.5
    hop_s: float = 1.0
    sr: int = 22050
    temperature_c: float | None = None
    reference_temp_c: float = 28.0

    # Internal state
    _clf: object = field(default=None, init=False, repr=False)
    _extractor: object = field(default=None, init=False, repr=False)
    _monitor: object = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False)
    _audio_queue: queue.Queue = field(default_factory=queue.Queue, init=False, repr=False)
    _result_queue: queue.Queue = field(default_factory=queue.Queue, init=False, repr=False)

    def __post_init__(self) -> None:
        from cricket_perception.classifier import SongTypeClassifier
        from cricket_perception.features import FeatureExtractor
        from cricket_perception.behavior import BehaviorMonitor

        self._clf = SongTypeClassifier.load(self.classifier_path)
        self._extractor = FeatureExtractor(sr=self.sr)
        self._monitor = BehaviorMonitor(
            rms_baseline=self.rms_baseline,
            aci_baseline=self.aci_baseline,
            reference_temp_c=self.reference_temp_c,
            aggressive_cluster_ids=[],   # not used — we use song_type label instead
            calling_cluster_ids=[],
            aggressive_threshold=self.aggressive_threshold,
            rms_low_threshold=self.rms_low_threshold,
            aci_low_threshold=self.aci_low_threshold,
        )
        logger.info(
            "RealTimeMonitor ready | window=%.1fs hop=%.1fs sr=%d",
            self.window_s, self.hop_s, self.sr,
        )

    # ── Inference on a single window ──────────────────────────────────────────

    def _infer_window(
        self,
        audio_window: np.ndarray,
        window_start_s: float,
        timestamp: datetime | None = None,
    ) -> StreamResult:
        """Run the full inference pipeline on one audio window."""

        ts = timestamp or datetime.now()

        # 1. Feature extraction
        feat = self._extractor.extract(audio_window)

        # 2. Classification
        song_type = self._clf.predict(feat)
        probas = self._clf.predict_proba(feat)
        confidence = probas.get(song_type, 0.0)

        # 3. Acoustic metrics
        # RMS: take mean of the RMS feature columns (indices 48-51)
        rms = float(feat[48:52].mean())
        aci = float(feat[52])

        # 4. Build cluster_fractions proxy from predicted song type
        # (BehaviorMonitor expects cluster_fractions but we now have song types)
        # We use a simple pass-through: map song type → probability
        agg_frac  = sum(probas.get(t, 0.0) for t in self.aggressive_song_types)
        call_frac = sum(probas.get(t, 0.0) for t in self.calling_song_types)

        # Build a fake cluster_fractions where "cluster 0" = aggressive fraction
        # and we pass directly to hunger_alert / mortality_alert instead
        from cricket_perception.behavior import hunger_alert, mortality_alert

        hunger = hunger_alert(
            cluster_fractions={0: agg_frac, 1: call_frac},
            aggressive_cluster_ids=[0],
            calling_cluster_ids=[1],
            aggressive_threshold=self.aggressive_threshold,
        )
        mortality = mortality_alert(
            rms=rms,
            aci=aci,
            rms_baseline=self.rms_baseline,
            aci_baseline=self.aci_baseline,
            rms_low_threshold=self.rms_low_threshold,
            aci_low_threshold=self.aci_low_threshold,
            temperature_c=self.temperature_c,
            reference_temp_c=self.reference_temp_c,
        )

        alerts = []
        if hunger["hungry"]:
            alerts.append(hunger["message"])
        if mortality["mortality_risk"]:
            alerts.append(mortality["message"])

        return StreamResult(
            timestamp=ts,
            window_start_s=window_start_s,
            window_dur_s=self.window_s,
            song_type=song_type,
            song_type_probas=probas,
            confidence=confidence,
            rms=rms,
            aci=aci,
            temperature_c=self.temperature_c,
            hungry=bool(hunger["hungry"]),
            mortality_risk=bool(mortality["mortality_risk"]),
            alerts=alerts,
        )

    # ── File-based streaming ───────────────────────────────────────────────────

    def stream_file(
        self,
        wav_path: str | Path,
        verbose: bool = True,
    ) -> Iterator[StreamResult]:
        """Process a WAV file as a sliding-window stream.

        Yields a StreamResult for each analysis window. Simulates real-time
        by stepping through the file in hop_s increments.

        Args:
            wav_path:  Path to a WAV file.
            verbose:   If True, print each result to stdout.

        Yields:
            StreamResult for each window.
        """
        import librosa

        wav_path = Path(wav_path)
        logger.info("Streaming file: %s", wav_path)

        y, _ = librosa.load(str(wav_path), sr=self.sr, mono=True)
        total_s = len(y) / self.sr
        win_samples = int(self.window_s * self.sr)
        hop_samples = int(self.hop_s * self.sr)

        start = 0
        window_idx = 0
        while start + win_samples <= len(y):
            window = y[start : start + win_samples]
            t_start = start / self.sr

            t0 = time.perf_counter()
            result = self._infer_window(window, window_start_s=t_start)
            latency_ms = (time.perf_counter() - t0) * 1000

            if verbose:
                print(f"{result}  [{latency_ms:.0f}ms]")

            yield result

            start += hop_samples
            window_idx += 1

        logger.info(
            "File stream complete: %d windows over %.1fs audio",
            window_idx, total_s,
        )

    # ── Microphone streaming ───────────────────────────────────────────────────

    def start_mic(
        self,
        device: int | str | None = None,
        on_result: Callable[[StreamResult], None] | None = None,
        on_alert: Callable[[StreamResult], None] | None = None,
        verbose: bool = True,
    ) -> None:
        """Start real-time microphone monitoring in a background thread.

        Requires ``sounddevice`` and ``libportaudio2`` to be installed.

        Args:
            device:    Microphone device index / name (None = system default).
            on_result: Callback fired on every window (receives StreamResult).
            on_alert:  Callback fired only when alerts are active.
            verbose:   Print each result to stdout.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice not installed. Run:\n"
                "  sudo apt-get install libportaudio2\n"
                "  pip install sounddevice"
            )

        if self._running:
            logger.warning("Monitor already running. Call stop() first.")
            return

        self._running = True
        win_samples = int(self.window_s * self.sr)
        hop_samples = int(self.hop_s * self.sr)
        ring_buffer: list[float] = []

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning("Sounddevice status: %s", status)
            # Flatten to mono
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            ring_buffer.extend(mono.tolist())

        def inference_loop():
            nonlocal ring_buffer
            window_start_s = 0.0

            while self._running:
                if len(ring_buffer) >= win_samples:
                    window = np.array(ring_buffer[:win_samples], dtype=np.float32)
                    ring_buffer = ring_buffer[hop_samples:]

                    t0 = time.perf_counter()
                    result = self._infer_window(window, window_start_s=window_start_s)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    window_start_s += self.hop_s

                    if verbose:
                        print(f"{result}  [{latency_ms:.0f}ms]")
                    if on_result:
                        on_result(result)
                    if on_alert and result.is_critical:
                        on_alert(result)

                    self._result_queue.put(result)
                else:
                    time.sleep(0.01)

        # Start audio capture
        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=device,
            callback=audio_callback,
            blocksize=int(self.sr * 0.1),  # 100ms blocks
        )
        self._stream.start()
        logger.info("🎙️  Microphone stream started (device=%s sr=%d)", device, self.sr)

        # Start inference thread
        self._inference_thread = threading.Thread(
            target=inference_loop, daemon=True, name="cricket-inference"
        )
        self._inference_thread.start()

    def stop_mic(self) -> list[StreamResult]:
        """Stop the microphone stream and return all buffered results.

        Returns:
            List of StreamResult collected since start_mic() was called.
        """
        self._running = False
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()
        if hasattr(self, "_inference_thread"):
            self._inference_thread.join(timeout=3.0)

        results = []
        while not self._result_queue.empty():
            results.append(self._result_queue.get_nowait())

        logger.info("🛑 Microphone stream stopped. %d windows processed.", len(results))
        return results

    # ── Class-method constructors ──────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        wav_path: str | Path,
        classifier_path: str | Path,
        **kwargs,
    ) -> "RealTimeMonitor":
        """Convenience constructor for file-based streaming.

        Creates a monitor and immediately starts streaming the given WAV file.
        Call ``monitor.stream_file(wav_path)`` to iterate results.
        """
        return cls(classifier_path=classifier_path, **kwargs)

    @classmethod
    def from_mic(
        cls,
        classifier_path: str | Path,
        **kwargs,
    ) -> "RealTimeMonitor":
        """Convenience constructor for microphone streaming."""
        return cls(classifier_path=classifier_path, **kwargs)
