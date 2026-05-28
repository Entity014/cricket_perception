"""
classifier.py
=============
Step 4b of the Cricket Perception pipeline.

Trains a lightweight supervised classifier on top of the unsupervised
cluster labels produced by clustering.py.

Workflow:
    1.  Run UMAP + HDBSCAN (clustering.py) → get cluster IDs
    2.  Label each cluster as a song type (manually or rule-based)
    3.  Train SongTypeClassifier on feature vectors + song-type labels
    4.  At inference time: extract 53-dim features → predict song type
        in <10 ms on CPU — suitable for Raspberry Pi / edge devices.

Usage::

    from cricket_perception.classifier import SongTypeClassifier, SONG_TYPES

    clf = SongTypeClassifier()
    clf.train(X_train, y_train)          # y_train: list of SONG_TYPES strings
    pred   = clf.predict(x_new)          # single feature vector
    probas = clf.predict_proba(x_new)    # {song_type: probability}

    clf.save("results/song_classifier.pkl")
    clf2 = SongTypeClassifier.load("results/song_classifier.pkl")
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ── Canonical Song Type Labels ─────────────────────────────────────────────────

SONG_TYPES = [
    "Calling Song",        # regular chirp, moderate RMS — normal colony activity
    "Aggressive Song",     # irregular bursts, high RMS — hunger / territory
    "Courtship/Low Song",  # soft trill, very low RMS — mate attraction
    "Quiet/Background",    # low energy, minimal activity
    "Noise",               # non-cricket sounds / silence
]

SONG_TYPE_SHORT = {
    "Calling Song":       "calling",
    "Aggressive Song":    "aggressive",
    "Courtship/Low Song": "courtship",
    "Quiet/Background":   "quiet",
    "Noise":              "noise",
}

# ── Classifier ────────────────────────────────────────────────────────────────


@dataclass
class SongTypeClassifier:
    """Supervised song-type classifier trained on cluster-labeled features.

    Wraps scikit-learn's SVC (RBF kernel) with StandardScaler pre-processing.
    Falls back to RandomForestClassifier if ``backend="rf"``.

    Example::

        clf = SongTypeClassifier(backend="svm")
        clf.train(X, y_labels)
        print(clf.predict(x_segment))
        print(clf.predict_proba(x_segment))
        clf.save("results/song_classifier.pkl")
    """

    backend: Literal["svm", "rf"] = "svm"

    # SVM hyperparams
    svm_C: float = 10.0
    svm_gamma: str = "scale"

    # RF hyperparams
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None

    # State (set after train())
    _scaler: object = field(default=None, init=False, repr=False)
    _model: object  = field(default=None, init=False, repr=False)
    _classes: list[str] = field(default_factory=list, init=False)
    _trained: bool  = field(default=False, init=False)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: list[str] | np.ndarray,
        cv_folds: int = 5,
    ) -> dict:
        """Train the classifier and return cross-validation metrics.

        Args:
            X:        Feature matrix [N, D] (raw, unscaled).
            y:        Song-type labels [N] — strings from ``SONG_TYPES``.
            cv_folds: Number of cross-validation folds (0 = skip CV).

        Returns:
            Dict with ``accuracy_mean``, ``accuracy_std``, ``train_time_s``,
            ``n_samples``, ``n_classes``, ``classes``.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        y = np.asarray(y)
        self._classes = sorted(set(y.tolist()))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self.backend == "svm":
            self._model = SVC(
                C=self.svm_C,
                kernel="rbf",
                gamma=self.svm_gamma,
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
        else:
            self._model = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )

        t0 = time.perf_counter()
        self._model.fit(X_scaled, y)
        train_time = time.perf_counter() - t0
        self._trained = True

        result: dict = {
            "n_samples": len(y),
            "n_classes": len(self._classes),
            "classes": self._classes,
            "train_time_s": round(train_time, 3),
            "backend": self.backend,
        }

        if cv_folds > 0 and len(y) >= cv_folds * 2:
            scores = cross_val_score(
                self._model, X_scaled, y,
                cv=cv_folds, scoring="balanced_accuracy",
            )
            result["accuracy_mean"] = round(float(scores.mean()), 4)
            result["accuracy_std"]  = round(float(scores.std()), 4)
            logger.info(
                "CV accuracy: %.3f ± %.3f (%d folds)",
                scores.mean(), scores.std(), cv_folds,
            )

        logger.info(
            "Trained %s on %d samples, %d classes in %.2fs",
            self.backend.upper(), len(y), len(self._classes), train_time,
        )
        return result

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> str:
        """Predict song type for a single feature vector.

        Args:
            x: Feature vector [D] or [1, D].

        Returns:
            Predicted song type string.
        """
        self._check_trained()
        x = np.atleast_2d(x)
        x_scaled = self._scaler.transform(x)
        return str(self._model.predict(x_scaled)[0])

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        """Return class probabilities for a single feature vector.

        Args:
            x: Feature vector [D] or [1, D].

        Returns:
            Dict mapping song_type → probability (sum = 1.0).
        """
        self._check_trained()
        x = np.atleast_2d(x)
        x_scaled = self._scaler.transform(x)
        proba = self._model.predict_proba(x_scaled)[0]
        classes = self._model.classes_
        return {str(c): round(float(p), 4) for c, p in zip(classes, proba)}

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict song types for a batch of feature vectors.

        Args:
            X: Feature matrix [N, D].

        Returns:
            Array of predicted song type strings [N].
        """
        self._check_trained()
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix for a batch.

        Returns:
            Array [N, n_classes] of probabilities.
        """
        self._check_trained()
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: list[str] | np.ndarray,
    ) -> dict:
        """Compute accuracy, balanced accuracy, and per-class metrics.

        Args:
            X_test: Test feature matrix [N, D].
            y_test: True song-type labels [N].

        Returns:
            Dict with accuracy, balanced_accuracy, classification_report.
        """
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
        )

        self._check_trained()
        y_pred = self.predict_batch(X_test)
        y_test = np.asarray(y_test)

        acc  = float(accuracy_score(y_test, y_pred))
        bacc = float(balanced_accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info("Accuracy: %.3f | Balanced: %.3f", acc, bacc)
        return {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bacc, 4),
            "classification_report": report,
            "y_pred": y_pred,
            "y_true": y_test,
        }

    def feature_importance(self) -> dict[int, float] | None:
        """Return feature importances (RF only).

        Returns:
            Dict mapping feature_index → importance, or None for SVM.
        """
        if self.backend != "rf" or not self._trained:
            return None
        importances = self._model.feature_importances_
        return {i: float(v) for i, v in enumerate(importances)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the classifier to disk.

        Args:
            path: Output file path (e.g. ``results/song_classifier.pkl``).
        """
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved SongTypeClassifier → '%s'", path)

    @classmethod
    def load(cls, path: str | Path) -> "SongTypeClassifier":
        """Load a pickled SongTypeClassifier.

        Args:
            path: Path to the .pkl file.

        Returns:
            Loaded SongTypeClassifier instance.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded SongTypeClassifier from '%s'", path)
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                "Classifier not trained yet. Call .train(X, y) first."
            )

    @property
    def is_trained(self) -> bool:
        """True if the classifier has been trained."""
        return self._trained

    @property
    def classes(self) -> list[str]:
        """List of song type classes the classifier knows about."""
        return self._classes


# ── Utility: Auto-label clusters from acoustic stats ─────────────────────────


def auto_label_clusters(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    rms_cols: slice = slice(48, 52),
    aci_col: int = 52,
) -> np.ndarray:
    """Assign song-type labels to segments via rule-based cluster heuristics.

    Used when manual labels are not yet available.
    Rules are based on RMS and ACI percentiles across all clusters.

    Args:
        X:              Feature matrix [N, D].
        cluster_labels: HDBSCAN cluster labels [N] (-1 = noise).
        rms_cols:       Slice of RMS feature columns in X.
        aci_col:        Index of the ACI feature column in X.

    Returns:
        Array of song-type strings [N] (one per segment).
    """
    import pandas as pd

    n = len(cluster_labels)
    song_type_labels = np.full(n, "Noise", dtype=object)

    # Noise points
    noise_mask = cluster_labels == -1
    song_type_labels[noise_mask] = "Noise"

    # Compute per-cluster RMS and ACI means
    unique_clusters = np.unique(cluster_labels[~noise_mask])
    cluster_rms = {}
    cluster_aci = {}
    for cid in unique_clusters:
        mask = cluster_labels == cid
        cluster_rms[cid] = float(X[mask][:, rms_cols].mean())
        cluster_aci[cid] = float(X[mask][:, aci_col].mean())

    rms_values = np.array(list(cluster_rms.values()))
    aci_values = np.array(list(cluster_aci.values()))

    rms_q25 = float(np.percentile(rms_values, 25))
    rms_q75 = float(np.percentile(rms_values, 75))
    aci_med  = float(np.median(aci_values))

    for cid in unique_clusters:
        rms = cluster_rms[cid]
        aci = cluster_aci[cid]
        mask = cluster_labels == cid

        if rms >= rms_q75:
            label = "Aggressive Song"
        elif rms >= rms_q25 and aci >= aci_med:
            label = "Calling Song"
        elif rms < rms_q25 and aci >= aci_med:
            label = "Courtship/Low Song"
        else:
            label = "Quiet/Background"

        song_type_labels[mask] = label

    return song_type_labels
