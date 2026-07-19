import numpy as np

from face_clusterer.config import TrackerConfig
from face_clusterer.embedder import FaceSample
from face_clusterer.tracker import IouTracker, iou


def _unit(rng, dim=512):
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _sample(bbox, emb, frame_idx, quality=1.0):
    return FaceSample(
        bbox=np.asarray(bbox, dtype=np.float32),
        det_score=0.95,
        embedding=emb,
        landmarks=None,
        crop=None,
        frame_idx=frame_idx,
        quality=quality,
    )


def test_iou_basics():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30], dtype=np.float32)) == 0.0
    assert 0.0 < iou(a, np.array([5, 0, 15, 10], dtype=np.float32)) < 1.0


def test_track_continuity_and_averaging():
    rng = np.random.default_rng(3)
    emb_a, emb_b = _unit(rng), _unit(rng)
    tracker = IouTracker(TrackerConfig(min_hits=2, max_age=3))

    # two people moving slowly for 5 frames
    for f in range(5):
        samples = [
            _sample([10 + f, 10, 60 + f, 60], emb_a, f),
            _sample([200, 50 + f, 260, 110 + f], emb_b, f),
        ]
        tracker.update(samples)
        ids = {s.track_id for s in samples}
        assert len(ids) == 2  # ids stay distinct

    tracks = tracker.flush()
    assert len(tracks) == 2
    for tr in tracks:
        assert tr.hits == 5
        mean = tr.mean_embedding()
        assert np.isclose(np.linalg.norm(mean), 1.0, atol=1e-5)


def test_embedding_fallback_after_jump():
    """A face 'teleports' (IoU=0) but keeps its embedding -> same track."""
    rng = np.random.default_rng(4)
    emb = _unit(rng)
    tracker = IouTracker(TrackerConfig(min_hits=1, max_age=5))

    s1 = _sample([0, 0, 50, 50], emb, 0)
    tracker.update([s1])
    s2 = _sample([300, 300, 350, 350], emb, 1)  # no IoU overlap
    tracker.update([s2])
    assert s2.track_id == s1.track_id


def test_stale_tracks_are_retired():
    rng = np.random.default_rng(5)
    tracker = IouTracker(TrackerConfig(min_hits=1, max_age=2))
    emb = _unit(rng)
    tracker.update([_sample([0, 0, 50, 50], emb, 0)])
    # person disappears; run empty frames past max_age
    for f in range(1, 5):
        tracker.update([])
    assert len(tracker.active) == 0
    assert len(tracker.finished) == 1
