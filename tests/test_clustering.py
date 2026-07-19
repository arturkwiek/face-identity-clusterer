import numpy as np
import pytest

from face_clusterer.clustering import (
    NOISE, cluster_embeddings, compute_centroids, reattach_noise,
)
from face_clusterer.config import ClusteringConfig


def _purity(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of clustered points whose cluster is dominated by one true id."""
    correct = total = 0
    for c in set(pred):
        if c == NOISE:
            continue
        members = true[pred == c]
        counts = np.bincount(members[members >= 0]) if (members >= 0).any() else []
        correct += int(np.max(counts)) if len(counts) else 0
        total += len(members)
    return correct / max(1, total)


@pytest.mark.parametrize("algorithm", ["hdbscan", "dbscan", "agglomerative"])
def test_recovers_identities(synthetic_identities, algorithm):
    x, true = synthetic_identities
    cfg = ClusteringConfig(algorithm=algorithm, min_cluster_size=4)
    result = cluster_embeddings(x, cfg)
    assert result.n_clusters == 3, f"{algorithm}: expected 3 identities"
    assert _purity(result.labels, true) >= 0.95


def test_empty_input():
    result = cluster_embeddings(np.empty((0, 512), dtype=np.float32))
    assert result.n_clusters == 0
    assert len(result.labels) == 0


def test_too_few_samples_all_noise():
    x = np.eye(3, 512, dtype=np.float32)
    result = cluster_embeddings(x, ClusteringConfig(min_cluster_size=5))
    assert result.n_clusters == 0
    assert (result.labels == NOISE).all()


def test_noise_reattachment():
    rng = np.random.default_rng(7)
    base = rng.normal(size=512)
    base /= np.linalg.norm(base)
    cluster = np.vstack([
        (base + rng.normal(scale=0.02, size=512)) for _ in range(6)
    ])
    cluster /= np.linalg.norm(cluster, axis=1, keepdims=True)
    near_noise = base + rng.normal(scale=0.05, size=512)
    near_noise /= np.linalg.norm(near_noise)

    x = np.vstack([cluster, near_noise]).astype(np.float32)
    labels = np.array([0] * 6 + [NOISE])
    centroids = compute_centroids(x, labels)
    new_labels, n = reattach_noise(x, labels, centroids, max_cos_distance=0.45)
    assert n == 1
    assert new_labels[-1] == 0


def test_far_noise_stays_noise(synthetic_identities):
    x, true = synthetic_identities
    result = cluster_embeddings(x, ClusteringConfig(min_cluster_size=4))
    # the 4 random-vector faces must remain unassigned
    noise_positions = np.where(true == -1)[0]
    assert (result.labels[noise_positions] == NOISE).all()
