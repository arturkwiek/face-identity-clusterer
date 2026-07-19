import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from face_clusterer.config import QualityConfig  # noqa: E402
from face_clusterer.embedder import FaceSample, l2_normalize  # noqa: E402
from face_clusterer.quality import assess, filter_samples, yaw_asymmetry  # noqa: E402
from face_clusterer.store import EmbeddingStore  # noqa: E402


def _sample(crop, det_score=0.9, size=100, landmarks=None):
    rng = np.random.default_rng(0)
    return FaceSample(
        bbox=np.array([0, 0, size, size], dtype=np.float32),
        det_score=det_score,
        embedding=l2_normalize(rng.normal(size=512)),
        landmarks=landmarks,
        crop=crop,
    )


def _sharp_crop(size=100):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)  # noisy = "sharp"


def _blurry_crop(size=100):
    return np.full((size, size, 3), 128, dtype=np.uint8)  # flat = zero Laplacian


def test_sharp_face_passes():
    result = assess(_sample(_sharp_crop()), QualityConfig())
    assert result.ok and result.score > 0.5


def test_blurry_face_rejected():
    result = assess(_sample(_blurry_crop()), QualityConfig())
    assert not result.ok
    assert any(r.startswith("blurry") for r in result.reasons)


def test_small_face_rejected():
    result = assess(_sample(_sharp_crop(30), size=30), QualityConfig(min_face_size=60))
    assert not result.ok
    assert any(r.startswith("too_small") for r in result.reasons)


def test_low_score_rejected():
    result = assess(_sample(_sharp_crop(), det_score=0.4), QualityConfig())
    assert not result.ok


def test_profile_rejected():
    # nose almost on top of the right eye -> strong yaw
    lm = np.array([[10, 50], [90, 50], [85, 60], [30, 80], [80, 80]], np.float32)
    assert yaw_asymmetry(lm) > 2.5
    result = assess(_sample(_sharp_crop(), landmarks=lm), QualityConfig())
    assert not result.ok


def test_filter_disabled_accepts_everything():
    samples = [_sample(_blurry_crop()), _sample(_sharp_crop(), det_score=0.1)]
    ok, rejected = filter_samples(samples, QualityConfig(enabled=False))
    assert len(ok) == 2 and not rejected


def test_store_roundtrip(tmp_path):
    db = tmp_path / "faces.sqlite"
    store = EmbeddingStore(db)
    samples = [_sample(None) for _ in range(5)]
    ids = store.add_samples(
        samples,
        cluster_ids=[0, 0, 1, 1, -1],
        person_ids=["p0", "p0", "p1", "p1", None],
    )
    assert len(ids) == 5

    embs, row_ids = store.all_embeddings()
    assert embs.shape == (5, 512) and len(row_ids) == 5
    np.testing.assert_allclose(embs[0], samples[0].embedding, atol=1e-6)

    summary = store.summary()
    assert summary["total_faces"] == 5
    assert summary["persons"] == {"p0": 2, "p1": 2}

    store.update_assignments(row_ids, [2] * 5, ["px"] * 5)
    assert store.summary()["persons"] == {"px": 5}
    store.close()


def test_clear_removes_all_faces(tmp_path):
    """Full re-analysis must replace stored faces, not duplicate them."""
    import numpy as np

    from face_clusterer.embedder import FaceSample
    from face_clusterer.store import EmbeddingStore

    store = EmbeddingStore(tmp_path / "faces.sqlite")
    sample = FaceSample(
        bbox=np.array([0, 0, 10, 10], dtype=np.float32), det_score=0.9,
        embedding=np.ones(512, dtype=np.float32), landmarks=None,
        source="a.jpg",
    )
    store.add_sample(sample, cluster_id=0, person_id="person_0000")
    store.add_sample(sample, cluster_id=0, person_id="person_0000")
    assert store.summary()["total_faces"] == 2

    assert store.clear() == 2
    assert store.summary()["total_faces"] == 0

    # the table still works after clearing
    store.add_sample(sample, cluster_id=0, person_id="person_0000")
    assert store.summary()["total_faces"] == 1
    store.close()
