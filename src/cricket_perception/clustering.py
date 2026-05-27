"""
clustering.py
=============
Step 3 of the Cricket Perception pipeline.

Reduces high-dimensional feature vectors to 2D/3D with UMAP,
then discovers natural groupings with HDBSCAN — no need to
pre-specify the number of clusters K.

Supports both CPU (umap-learn + hdbscan) and GPU (cuML RAPIDS)
backends, auto-detected at import time.

Usage:
    from cricket_perception.clustering import ClusterPipeline

    pipe = ClusterPipeline()
    labels, embeddings_2d = pipe.fit(feature_matrix)

    pipe.plot(embeddings_2d, labels, title="Cricket soundscape clusters")
    pipe.save("results/cluster_model.pkl")
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Backend Detection ──────────────────────────────────────────────────────────

def _detect_backend() -> Literal["cuml", "cpu"]:
    """Auto-detect cuML (RAPIDS GPU) availability, fallback to CPU."""
    try:
        import cuml  # noqa: F401
        logger.info("🚀 cuML (RAPIDS) detected — using GPU backend")
        return "cuml"
    except ImportError:
        logger.info("cuML not found — using CPU backend (umap-learn + hdbscan)")
        return "cpu"


# ── UMAP Reduction ─────────────────────────────────────────────────────────────

def reduce_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    backend: Literal["auto", "cuml", "cpu"] = "auto",
) -> np.ndarray:
    """Reduce feature matrix to low-dimensional embedding using UMAP.

    Args:
        X:            Feature matrix [N, D].
        n_components: Target dimensions (2 for scatter plot, 3 for 3D).
        n_neighbors:  UMAP n_neighbors — controls local vs. global structure.
                      Small → local detail, Large → global shape.
        min_dist:     UMAP min_dist — controls how tightly points cluster.
        metric:       Distance metric (euclidean, cosine, etc.).
        random_state: For reproducibility.
        backend:      "auto" (detect GPU), "cuml" (force GPU), "cpu" (force CPU).

    Returns:
        Embedding array of shape [N, n_components].
    """
    if backend == "auto":
        backend = _detect_backend()

    logger.info("Running UMAP: N=%d, D=%d → %dD [%s]",
                X.shape[0], X.shape[1], n_components, backend)

    if backend == "cuml":
        from cuml.manifold import UMAP as cuUMAP
        reducer = cuUMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(X)
        return np.asarray(embedding)
    else:
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            low_memory=False,
        )
        return reducer.fit_transform(X)


# ── HDBSCAN Clustering ────────────────────────────────────────────────────────

def cluster_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",
    backend: Literal["auto", "cuml", "cpu"] = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster a (UMAP) embedding with HDBSCAN.

    Label -1 means "noise" (outlier not assigned to any cluster).

    Args:
        embedding:                 Low-dim array [N, d].
        min_cluster_size:          Minimum cluster size (raise to get fewer,
                                   larger clusters).
        min_samples:               Controls robustness to noise.
        cluster_selection_method:  "eom" (default) or "leaf".
        backend:                   "auto", "cuml", or "cpu".

    Returns:
        (labels, probabilities):
            labels        — int array [N], cluster id per point (-1 = noise).
            probabilities — float array [N], membership strength.
    """
    if backend == "auto":
        backend = _detect_backend()

    logger.info("Running HDBSCAN: N=%d [%s]", embedding.shape[0], backend)

    if backend == "cuml":
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        clusterer = cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(embedding)
        probs = np.ones(len(labels), dtype=np.float32)
        return np.asarray(labels), probs
    else:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            prediction_data=True,
        )
        clusterer.fit(embedding)
        return clusterer.labels_, clusterer.probabilities_


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str = "Cricket Soundscape Clusters",
    label_names: dict[int, str] | None = None,
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create a 2D scatter plot of UMAP embedding colored by cluster.

    Args:
        embedding:    2D array [N, 2] from UMAP.
        labels:       Integer cluster labels [N] (-1 = noise).
        title:        Plot title.
        label_names:  Optional dict mapping cluster id → human-readable name.
        figsize:      Figure size in inches.
        save_path:    If given, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])

    palette = sns.color_palette("husl", n_colors=max(n_clusters, 1))
    color_map = {lbl: palette[i] for i, lbl in enumerate(unique_labels[unique_labels >= 0])}
    color_map[-1] = (0.7, 0.7, 0.7)  # grey for noise

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    for lbl in unique_labels:
        mask = labels == lbl
        name = (label_names or {}).get(lbl, f"Cluster {lbl}" if lbl >= 0 else "Noise")
        color = color_map[lbl]
        alpha = 0.4 if lbl == -1 else 0.8
        size  = 10  if lbl == -1 else 20
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color], label=name, alpha=alpha, s=size, edgecolors="none",
        )

    ax.legend(
        loc="upper right", framealpha=0.2,
        labelcolor="white", fontsize=9,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    n_noise = (labels == -1).sum()
    ax.text(
        0.01, 0.01,
        f"N={len(labels)} | Clusters={n_clusters} | Noise={n_noise}",
        transform=ax.transAxes, fontsize=8, color="#aaaacc",
    )

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved cluster plot → '%s'", save_path)

    return fig


# ── Unified Pipeline ───────────────────────────────────────────────────────────

@dataclass
class ClusterPipeline:
    """End-to-end UMAP + HDBSCAN clustering pipeline.

    Example::

        from cricket_perception.clustering import ClusterPipeline

        pipe = ClusterPipeline(umap_n_neighbors=20, hdbscan_min_cluster_size=15)
        labels, emb = pipe.fit(feature_matrix)
        pipe.plot(emb, labels, title="Farm A — May 2026")
        pipe.save("results/cluster_model.pkl")
    """

    # UMAP params
    umap_n_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_random_state: int = 42

    # HDBSCAN params
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    hdbscan_method: str = "eom"

    # Backend
    backend: Literal["auto", "cuml", "cpu"] = "auto"

    # State (set after fit)
    embedding_: np.ndarray = field(default=None, init=False, repr=False)
    labels_: np.ndarray = field(default=None, init=False, repr=False)
    probabilities_: np.ndarray = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run UMAP → HDBSCAN on feature matrix X.

        Args:
            X: Feature matrix [N, D].

        Returns:
            (labels, embedding): Cluster labels and 2D embedding.
        """
        self.embedding_ = reduce_umap(
            X,
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=self.umap_random_state,
            backend=self.backend,
        )
        self.labels_, self.probabilities_ = cluster_hdbscan(
            self.embedding_,
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            cluster_selection_method=self.hdbscan_method,
            backend=self.backend,
        )

        n_clusters = len(np.unique(self.labels_[self.labels_ >= 0]))
        n_noise = (self.labels_ == -1).sum()
        logger.info("Clustering done: %d clusters, %d noise points", n_clusters, n_noise)

        return self.labels_, self.embedding_

    def plot(
        self,
        embedding: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot clusters. Uses fitted values if not provided."""
        emb = embedding if embedding is not None else self.embedding_
        lbl = labels    if labels    is not None else self.labels_
        return plot_clusters(emb, lbl, **kwargs)

    def save(self, path: str | Path) -> None:
        """Pickle the pipeline state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved ClusterPipeline → '%s'", path)

    @classmethod
    def load(cls, path: str | Path) -> "ClusterPipeline":
        """Load a pickled pipeline."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Cluster Summary ────────────────────────────────────────────────────────────

def summarise_clusters(
    labels: np.ndarray,
    metadata: list[dict] | None = None,
) -> dict[int, dict]:
    """Return a summary dict of each cluster.

    Args:
        labels:   Cluster labels [N].
        metadata: Optional list of dicts with per-segment metadata
                  (e.g. filename, timestamp, species).

    Returns:
        Dict mapping cluster_id → {count, fraction, ...}
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    summary = {}
    for lbl, cnt in zip(unique, counts):
        entry: dict = {
            "count": int(cnt),
            "fraction": round(float(cnt) / total, 4),
            "is_noise": bool(lbl == -1),
        }
        if metadata:
            idxs = np.where(labels == lbl)[0]
            entry["sample_files"] = [metadata[i].get("file", str(i))
                                      for i in idxs[:5]]  # first 5 examples
        summary[int(lbl)] = entry

    return summary
