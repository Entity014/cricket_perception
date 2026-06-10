"""
cricket_perception
==================
Unsupervised clustering & behavior analysis of farm cricket acoustics.

Pipeline:
    Raw Audio → audio_utils → features → clustering → behavior
"""

__version__ = "0.1.0"
__author__ = "Xero"

from cricket_perception import audio_utils, features, clustering, behavior, augmentation
from cricket_perception.augmentation import SoundscapeSynthesizer

__all__ = ["audio_utils", "features", "clustering", "behavior", "augmentation", "SoundscapeSynthesizer"]
