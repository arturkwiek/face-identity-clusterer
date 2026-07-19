"""Embedding clustering: HDBSCAN (default) with DBSCAN / Agglomerative fallbacks.

Embeddings must be L2-normalized. For normalized vectors, euclidean distance is
a monotonic function of cosine distance (d_e = sqrt(2 - 2*cos_sim)), so HDBSCAN
with euclidean on normalized data behaves like cosine-based clustering — this
matters because the hdbscan library does not support the cosine metric natively.

Extra improvements over vanilla HDBSCAN:
- outlier purge: cluster members whose cosine distance to their own centroid
  exceeds a threshold are demoted to noise (HDBSCAN can absorb far-away points
  into a cluster because density is relative);
- small-cluster demotion: clusters below min_cluster_size become noise
  (agglomerative clustering otherwise emits singleton "identities");
- noise re-attachment: noise points are attached to the nearest cluster
  centroid if close enough, recovering valid faces HDBSCAN was too
  conservative about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import ClusteringConfig
from .embedder import l2_normalize

logger = logging.getLogger(__name__)

NOISE = -1


@dataclass
class ClusteringResult:
    labels: np.ndarray                        # (n,) int, -1 = noise
    centroids: dict[int, np.ndarray] = field(default_factory=dict)
    n_clusters: int = 0
    n_noise: int = 0
    reattached: int = 0
    purged: int = 0


def _cluster_hdbscan(x: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    """HDBSCAN via the `hdbscan` package, falling back to scikit-learn's port.

    The standalone `hdbscan` package needs a C++ toolchain and frequently fails
    to install on Windows. scikit-learn >= 1.3 ships an equivalent
    implementation, so prefer the dedicated package when present but treat
    sklearn as a first-class alternative — degrading straight to DBSCAN would
    silently drop the algorithm this pipeline is built around.
    """
    kwargs = dict(
        min_cluster_size=max(2, cfg.min_cluster_size),
        min_samples=max(1, cfg.min_samples),
        metric="euclidean",  # on L2-normalized vectors ~ cosine (see module docstring)
    )
    try:
        import hdbscan

        return hdbscan.HDBSCAN(**kwargs).fit_predict(x)
    except ImportError:
        pass

    from sklearn.cluster import HDBSCAN  # sklearn >= 1.3

    logger.debug("hdbscan package unavailable — using sklearn.cluster.HDBSCAN.")
    return HDBSCAN(**kwargs).fit_predict(x)


def _cluster_dbscan(x: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    return DBSCAN(
        eps=cfg.dbscan_eps, min_samples=max(1, cfg.min_samples), metric="cosine"
    ).fit_predict(x)


def _cluster_agglomerative(x: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=cfg.agglomerative_threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(x)


def compute_centroids(x: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for label in sorted(set(labels)):
        if label == NOISE:
            continue
        centroids[int(label)] = l2_normalize(x[labels == label].mean(axis=0))
    return centroids


def demote_small_clusters(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Clusters with fewer than min_size members become noise."""
    labels = labels.copy()
    for label in set(labels):
        if label == NOISE:
            continue
        if int((labels == label).sum()) < min_size:
            labels[labels == label] = NOISE
    return labels


def purge_outliers(
    x: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    max_cos_distance: float,
) -> tuple[np.ndarray, int]:
    """Demote members too far from their own centroid to noise."""
    labels = labels.copy()
    purged = 0
    for label, centroid in centroids.items():
        idx = np.where(labels == label)[0]
        sims = x[idx] @ centroid
        far = idx[(1.0 - sims) > max_cos_distance]
        if len(far):
            labels[far] = NOISE
            purged += len(far)
    return labels, purged


def reattach_noise(
    x: np.ndarray,
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
    max_cos_distance: float,
) -> tuple[np.ndarray, int]:
    """Attach noise points to the nearest centroid when close enough."""
    if not centroids:
        return labels, 0
    labels = labels.copy()
    ids = np.array(sorted(centroids), dtype=int)
    matrix = np.vstack([centroids[i] for i in ids])       # (k, d)
    noise_idx = np.where(labels == NOISE)[0]
    reattached = 0
    for i in noise_idx:
        sims = matrix @ x[i]                              # cosine sim (normalized)
        best = int(np.argmax(sims))
        if 1.0 - float(sims[best]) <= max_cos_distance:
            labels[i] = int(ids[best])
            reattached += 1
    return labels, reattached


def cluster_embeddings(
    embeddings: np.ndarray, cfg: ClusteringConfig | None = None
) -> ClusteringResult:
    """Cluster L2-normalized embeddings into identities."""
    cfg = cfg or ClusteringConfig()
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0:
        return ClusteringResult(labels=np.empty(0, dtype=int))

    if len(x) < max(2, cfg.min_cluster_size):
        # Too few samples to form any cluster — everything is noise.
        return ClusteringResult(
            labels=np.full(len(x), NOISE, dtype=int), n_noise=len(x)
        )

    algorithm = cfg.algorithm.lower()
    if algorithm == "hdbscan":
        try:
            labels = _cluster_hdbscan(x, cfg)
        except ImportError:
            logger.warning(
                "No HDBSCAN implementation available (needs the hdbscan package "
                "or scikit-learn >= 1.3) — falling back to DBSCAN."
            )
            labels = _cluster_dbscan(x, cfg)
    elif algorithm == "dbscan":
        labels = _cluster_dbscan(x, cfg)
    elif algorithm == "agglomerative":
        labels = _cluster_agglomerative(x, cfg)
    else:
        raise ValueError(f"Unknown clustering algorithm: {cfg.algorithm}")

    labels = np.asarray(labels, dtype=int)
    labels = demote_small_clusters(labels, cfg.min_cluster_size)
    centroids = compute_centroids(x, labels)

    purged = 0
    if cfg.purge_outliers and centroids:
        labels, purged = purge_outliers(x, labels, centroids, cfg.purge_threshold)
        if purged:
            labels = demote_small_clusters(labels, cfg.min_cluster_size)
            centroids = compute_centroids(x, labels)

    reattached = 0
    if cfg.reattach_noise and centroids:
        labels, reattached = reattach_noise(
            x, labels, centroids, cfg.reattach_threshold
        )
        if reattached:
            centroids = compute_centroids(x, labels)

    return ClusteringResult(
        labels=labels,
        centroids=centroids,
        n_clusters=len(centroids),
        n_noise=int((labels == NOISE).sum()),
        reattached=reattached,
        purged=purged,
    )
