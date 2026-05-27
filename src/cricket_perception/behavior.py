"""
behavior.py
===========
Step 4 of the Cricket Perception pipeline.

Maps cluster-level acoustic features to farm-actionable behavioral signals:

    1. hunger_alert()     — Detects food competition (Aggressive Song surge)
    2. mortality_alert()  — Detects abnormally low acoustic activity
    3. temperature_adjust() — Corrects for chirp-rate vs. temperature
                              dependency (Dolbear's Law)

These functions operate on time-series of feature statistics or cluster labels,
not on raw audio — they are the final interpretation layer.

Usage:
    from cricket_perception.behavior import BehaviorMonitor

    monitor = BehaviorMonitor(rms_baseline=0.05, aci_baseline=200.0)
    result  = monitor.analyze(
        rms=current_rms,
        aci=current_aci,
        temperature_c=28.5,
        cluster_fractions={0: 0.6, 1: 0.3, -1: 0.1},
    )
    print(result.alerts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Dolbear's Law ─────────────────────────────────────────────────────────────

def temperature_from_chirps(
    chirps_per_15s: float,
    species: str = "gryllus",
) -> float:
    """Estimate temperature (°C) from chirp rate using Dolbear's Law.

    Original (Fahrenheit): T_F = 50 + (N_15 - 40) / 4   [Gryllus bimaculatus]
    Converted to Celsius : T_C = (T_F - 32) * 5/9

    Args:
        chirps_per_15s: Number of chirps in a 15-second window.
        species:        Affects the formula coefficients.
                        Supported: "gryllus" (G. bimaculatus, common farm species).

    Returns:
        Estimated temperature in °C.

    Reference: Dolbear, A.E. (1897). The cricket as a thermometer.
               The American Naturalist, 31(371), 970–971.
    """
    if species == "gryllus":
        t_fahrenheit = 50.0 + (chirps_per_15s - 40.0) / 4.0
    else:
        raise ValueError(f"Unknown species: '{species}'. Supported: 'gryllus'")

    t_celsius = (t_fahrenheit - 32.0) * 5.0 / 9.0
    return round(t_celsius, 2)


def chirp_rate_correction_factor(
    measured_temp_c: float,
    reference_temp_c: float = 28.0,
    q10: float = 2.0,
) -> float:
    """Compute a Q10-based correction factor for chirp-rate vs. temperature.

    Cricket acoustic activity roughly doubles every 10°C (Q10 ≈ 2).
    If temperature drops, we expect fewer chirps — not starvation or death.

    Args:
        measured_temp_c:  Current enclosure temperature.
        reference_temp_c: Baseline "normal" temperature.
        q10:              Metabolic rate doubling factor (default 2.0).

    Returns:
        Correction factor to multiply observed chirp rate by,
        to normalise to reference temperature.
        Values > 1 mean current temp is cooler than baseline (activity is lower).
    """
    exponent = (reference_temp_c - measured_temp_c) / 10.0
    return float(q10 ** exponent)


# ── Hunger Detection ──────────────────────────────────────────────────────────

def hunger_alert(
    cluster_fractions: dict[int, float],
    aggressive_cluster_ids: list[int],
    calling_cluster_ids: list[int],
    aggressive_threshold: float = 0.35,
) -> dict:
    """Detect food competition by checking the Aggressive:Calling ratio.

    Theory (from research_notes.md §3.1):
        When food is scarce, Aggressive Song fraction increases sharply.
        If Aggressive / Calling > threshold → alert.

    Args:
        cluster_fractions:      Dict {cluster_id: fraction} from current window.
        aggressive_cluster_ids: Cluster IDs labelled as "Aggressive Song".
        calling_cluster_ids:    Cluster IDs labelled as "Calling Song".
        aggressive_threshold:   Fraction of Aggressive Song that triggers alert.

    Returns:
        Dict with keys:
            "hungry" (bool), "aggressive_frac", "calling_frac",
            "ratio", "message"
    """
    agg_frac  = sum(cluster_fractions.get(c, 0.0) for c in aggressive_cluster_ids)
    call_frac = sum(cluster_fractions.get(c, 0.0) for c in calling_cluster_ids)
    ratio = agg_frac / (call_frac + 1e-8)

    hungry = agg_frac >= aggressive_threshold
    message = (
        f"⚠️  HUNGER ALERT: Aggressive fraction {agg_frac:.1%} ≥ {aggressive_threshold:.1%}"
        if hungry
        else f"✅ Normal: Aggressive fraction {agg_frac:.1%}"
    )

    logger.info(message)
    return {
        "hungry": hungry,
        "aggressive_frac": round(agg_frac, 4),
        "calling_frac": round(call_frac, 4),
        "ratio": round(ratio, 4),
        "message": message,
    }


# ── Mortality Detection ───────────────────────────────────────────────────────

def mortality_alert(
    rms: float,
    aci: float,
    rms_baseline: float,
    aci_baseline: float,
    rms_low_threshold: float = 0.4,
    aci_low_threshold: float = 0.5,
    temperature_c: float | None = None,
    reference_temp_c: float = 28.0,
) -> dict:
    """Detect abnormally low acoustic activity suggesting high mortality.

    Theory (from research_notes.md §3.1):
        A pond with many dead/sick crickets will have very low RMS Energy
        and very low ACI compared to baseline.

    Both thresholds are expressed as fractions of baseline, e.g.:
        rms_low_threshold=0.4 → alert if RMS < 40% of baseline.

    Args:
        rms:                 Measured RMS (mean over current window).
        aci:                 Measured ACI (current window).
        rms_baseline:        Expected RMS at reference temperature.
        aci_baseline:        Expected ACI at reference temperature.
        rms_low_threshold:   RMS / baseline fraction below which alert fires.
        aci_low_threshold:   ACI / baseline fraction below which alert fires.
        temperature_c:       Current temperature (optional, for correction).
        reference_temp_c:    Baseline calibration temperature.

    Returns:
        Dict with keys:
            "mortality_risk" (bool), "rms_ratio", "aci_ratio",
            "temperature_correction", "message"
    """
    temp_correction = 1.0
    if temperature_c is not None:
        temp_correction = chirp_rate_correction_factor(temperature_c, reference_temp_c)

    # Apply temperature correction to thresholds
    effective_rms = rms * temp_correction
    effective_aci = aci * temp_correction

    rms_ratio = effective_rms / (rms_baseline + 1e-10)
    aci_ratio = effective_aci / (aci_baseline + 1e-10)

    at_risk = rms_ratio < rms_low_threshold or aci_ratio < aci_low_threshold
    message = (
        f"🚨 MORTALITY RISK: RMS ratio={rms_ratio:.2f}, ACI ratio={aci_ratio:.2f}"
        if at_risk
        else f"✅ Normal activity: RMS ratio={rms_ratio:.2f}, ACI ratio={aci_ratio:.2f}"
    )

    logger.info(message)
    return {
        "mortality_risk": at_risk,
        "rms_ratio": round(rms_ratio, 4),
        "aci_ratio": round(aci_ratio, 4),
        "temperature_correction": round(temp_correction, 4),
        "message": message,
    }


# ── Analysis Result ────────────────────────────────────────────────────────────

class BehaviorResult(NamedTuple):
    """Result of a single behavioral analysis window."""
    timestamp: datetime
    temperature_c: float | None
    hunger: dict
    mortality: dict

    @property
    def alerts(self) -> list[str]:
        """Return a list of active alert messages."""
        msgs = []
        if self.hunger.get("hungry"):
            msgs.append(self.hunger["message"])
        if self.mortality.get("mortality_risk"):
            msgs.append(self.mortality["message"])
        return msgs

    @property
    def is_critical(self) -> bool:
        return bool(self.alerts)


# ── Unified Monitor ────────────────────────────────────────────────────────────

@dataclass
class BehaviorMonitor:
    """High-level behavior monitoring interface.

    Calibrate once with healthy colony baseline values, then call
    `analyze()` on each new measurement window.

    Example::

        monitor = BehaviorMonitor(
            rms_baseline=0.05,
            aci_baseline=250.0,
            reference_temp_c=28.0,
            aggressive_cluster_ids=[2],
            calling_cluster_ids=[0],
        )

        result = monitor.analyze(
            rms=0.018,
            aci=90.0,
            temperature_c=26.5,
            cluster_fractions={0: 0.5, 1: 0.2, 2: 0.25, -1: 0.05},
        )

        for alert in result.alerts:
            print(alert)
    """

    # Baseline values (calibrate from a healthy, well-fed colony)
    rms_baseline: float = 0.05
    aci_baseline: float = 200.0
    reference_temp_c: float = 28.0

    # Cluster semantics (set after manual labelling of clusters)
    aggressive_cluster_ids: list[int] = field(default_factory=list)
    calling_cluster_ids: list[int]    = field(default_factory=lambda: [0])

    # Alert thresholds
    aggressive_threshold: float = 0.35
    rms_low_threshold: float    = 0.40
    aci_low_threshold: float    = 0.50

    def analyze(
        self,
        rms: float,
        aci: float,
        cluster_fractions: dict[int, float],
        temperature_c: float | None = None,
        timestamp: datetime | None = None,
    ) -> BehaviorResult:
        """Analyze a measurement window and return behavioral assessment.

        Args:
            rms:               Mean RMS energy of the window.
            aci:               Acoustic Complexity Index of the window.
            cluster_fractions: {cluster_id: fraction} for this window.
            temperature_c:     Current enclosure temperature (°C).
            timestamp:         Window timestamp (defaults to now).

        Returns:
            BehaviorResult with hunger, mortality results and any alerts.
        """
        ts = timestamp or datetime.now()

        hunger = hunger_alert(
            cluster_fractions=cluster_fractions,
            aggressive_cluster_ids=self.aggressive_cluster_ids,
            calling_cluster_ids=self.calling_cluster_ids,
            aggressive_threshold=self.aggressive_threshold,
        )
        mortality = mortality_alert(
            rms=rms,
            aci=aci,
            rms_baseline=self.rms_baseline,
            aci_baseline=self.aci_baseline,
            rms_low_threshold=self.rms_low_threshold,
            aci_low_threshold=self.aci_low_threshold,
            temperature_c=temperature_c,
            reference_temp_c=self.reference_temp_c,
        )

        result = BehaviorResult(
            timestamp=ts,
            temperature_c=temperature_c,
            hunger=hunger,
            mortality=mortality,
        )

        if result.is_critical:
            logger.warning("🚨 Critical alerts at %s: %s", ts.isoformat(), result.alerts)

        return result
