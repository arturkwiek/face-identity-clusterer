import numpy as np

from face_clusterer.config import PrototypeConfig
from face_clusterer.prototypes import PrototypeStore


def _unit(rng, dim=512):
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_match_or_add_and_reid():
    rng = np.random.default_rng(0)
    store = PrototypeStore(PrototypeConfig(match_threshold=0.6))

    base = _unit(rng)
    pid, _, is_new = store.match_or_add(base)
    assert is_new

    # a slightly jittered embedding of the same person re-identifies
    jittered = base + rng.normal(scale=0.01, size=512).astype(np.float32)
    jittered /= np.linalg.norm(jittered)
    pid2, sim, is_new2 = store.match_or_add(jittered)
    assert pid2 == pid and not is_new2 and sim > 0.9

    # a completely different person creates a new prototype
    other = _unit(rng)
    pid3, _, is_new3 = store.match_or_add(other)
    assert is_new3 and pid3 != pid
    assert len(store.prototypes) == 2


def test_ema_update_moves_prototype():
    rng = np.random.default_rng(1)
    store = PrototypeStore(PrototypeConfig(ema_alpha=0.5))
    a, b = _unit(rng), _unit(rng)
    pid = store.add(a)
    before = store.prototypes[pid].vector.copy()
    store.update(pid, b)
    after = store.prototypes[pid].vector
    assert not np.allclose(before, after)
    assert np.isclose(np.linalg.norm(after), 1.0, atol=1e-5)  # stays normalized
    # moved towards b
    assert float(after @ b) > float(before @ b)


def test_absorb_clusters_maps_and_persists(tmp_path):
    rng = np.random.default_rng(2)
    path = str(tmp_path / "prototypes")
    store = PrototypeStore(PrototypeConfig(path=path))

    centroids = {0: _unit(rng), 1: _unit(rng)}
    mapping = store.absorb_clusters(centroids, counts={0: 5, 1: 8})
    assert len(mapping) == 2
    store.save()

    # reload in a fresh store: same persons come back, centroids re-identify
    store2 = PrototypeStore(PrototypeConfig(path=path))
    assert store2.load()
    assert set(store2.prototypes) == set(mapping.values())
    pid, sim = store2.match(centroids[0])
    assert pid == mapping[0] and sim > 0.9

    # new run with same centroid maps to the SAME person (cross-run re-id)
    mapping2 = store2.absorb_clusters({0: centroids[0]})
    assert mapping2[0] == mapping[0]
