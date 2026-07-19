"""Integration test: full ImagesPipeline / VideoPipeline flow with a fake
embedder (no InsightFace model needed). Verifies wiring: quality filter ->
clustering -> prototypes -> sqlite store -> reports."""

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from face_clusterer.config import AppConfig  # noqa: E402
from face_clusterer.embedder import FaceSample, l2_normalize  # noqa: E402
from face_clusterer.pipeline import ImagesPipeline  # noqa: E402


class FakeEmbedder:
    """Deterministic embedder: person id is encoded in the image's blue channel."""

    def __init__(self, identities: dict[int, np.ndarray]):
        self.identities = identities
        self.rng = np.random.default_rng(0)

    def extract(self, frame_bgr, source="", frame_idx=-1, keep_crops=True):
        person = int(frame_bgr[0, 0, 0])  # blue channel encodes person id
        base = self.identities[person]
        emb = l2_normalize(
            base + self.rng.normal(scale=0.02, size=512).astype(np.float32)
        )
        crop = self.rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
        return [
            FaceSample(
                bbox=np.array([10, 10, 110, 110], dtype=np.float32),
                det_score=0.95,
                embedding=emb,
                landmarks=None,
                crop=crop,
                source=source,
                frame_idx=frame_idx,
            )
        ]


@pytest.fixture
def photo_dir(tmp_path):
    """8 photos per synthetic person, person id stored in the blue channel."""
    d = tmp_path / "photos"
    d.mkdir()
    for person in range(3):
        for i in range(8):
            img = np.full((200, 200, 3), 128, dtype=np.uint8)
            img[:, :, 0] = person
            cv2.imwrite(str(d / f"p{person}_{i}.png"), img)
    return d


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.output.dir = str(tmp_path / "output")
    cfg.output.db_path = str(tmp_path / "output" / "faces.sqlite")
    cfg.prototypes.path = str(tmp_path / "output" / "prototypes")
    cfg.quality.enabled = False  # fake crops are random noise
    cfg.clustering.min_cluster_size = 4
    return cfg


def test_images_pipeline_end_to_end(tmp_path, photo_dir):
    rng = np.random.default_rng(99)
    identities = {
        p: l2_normalize(rng.normal(size=512).astype(np.float32)) for p in range(3)
    }

    cfg = _config(tmp_path)
    pipeline = ImagesPipeline(cfg)
    pipeline.embedder = FakeEmbedder(identities)  # inject fake

    summary = pipeline.run(photo_dir)
    assert summary["total_faces"] == 24
    assert summary["identities"] == 3
    assert summary["noise_faces"] == 0

    out = tmp_path / "output"
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert saved["identities"] == 3
    assert (out / "faces.sqlite").exists()

    person_dirs = [p for p in (out / "persons").iterdir() if p.is_dir()]
    assert len(person_dirs) == 3
    for d in person_dirs:
        assert len(list(d.glob("*.jpg"))) == 8

    # ---- second run: same people must re-identify to the SAME person ids ----
    first_ids = set(saved["persons"])
    pipeline2 = ImagesPipeline(cfg)
    pipeline2.embedder = FakeEmbedder(identities)
    summary2 = pipeline2.run(photo_dir)
    assert set(summary2["persons"]) == first_ids  # cross-run re-identification


def test_noise_samples_are_recovered_via_prototypes(tmp_path, photo_dir):
    """A known person seen too rarely to cluster must still be identified.

    This is the video case: clustering runs on tracks, so someone appearing in
    a single track can never form a cluster of `min_cluster_size`. Without the
    noise fallback they would be discarded despite already being known.
    """
    rng = np.random.default_rng(7)
    identities = {
        p: l2_normalize(rng.normal(size=512).astype(np.float32)) for p in range(3)
    }

    cfg = _config(tmp_path)
    pipeline = ImagesPipeline(cfg)
    pipeline.embedder = FakeEmbedder(identities)
    first = pipeline.run(photo_dir)
    assert first["identities"] == 3
    known = set(first["persons"])

    # A second folder with a single photo per person: 1 sample each, far below
    # min_cluster_size, so HDBSCAN can only call all of them noise.
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    for person in range(3):
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        img[:, :, 0] = person
        cv2.imwrite(str(sparse / f"p{person}.png"), img)

    pipeline2 = ImagesPipeline(cfg)
    pipeline2.embedder = FakeEmbedder(identities)
    summary = pipeline2.run(sparse)

    assert summary["recovered_from_noise"] == 3
    assert set(summary["persons"]) <= known      # no new identities invented
    assert "_noise" not in summary["persons"]


def test_unknown_noise_stays_noise(tmp_path, photo_dir):
    """The fallback must not invent identities for people it has never seen."""
    rng = np.random.default_rng(11)
    identities = {
        p: l2_normalize(rng.normal(size=512).astype(np.float32)) for p in range(3)
    }
    cfg = _config(tmp_path)
    pipeline = ImagesPipeline(cfg)
    pipeline.embedder = FakeEmbedder(identities)
    pipeline.run(photo_dir)

    # a stranger: one photo of a person id the prototype store has never seen
    stranger_dir = tmp_path / "stranger"
    stranger_dir.mkdir()
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[:, :, 0] = 9
    cv2.imwrite(str(stranger_dir / "x.png"), img)

    identities[9] = l2_normalize(rng.normal(size=512).astype(np.float32))
    pipeline2 = ImagesPipeline(cfg)
    pipeline2.embedder = FakeEmbedder(identities)
    summary = pipeline2.run(stranger_dir)

    assert summary["recovered_from_noise"] == 0
    assert summary["persons"]["_noise"]["faces"] == 1


def test_recluster_store_recovers_known_persons(tmp_path):
    """Recluster must not un-assign known people whose samples can't cluster.

    Regression: update_assignments used to overwrite person_id with None for
    every row HDBSCAN called noise, silently erasing earlier identifications.
    """
    from face_clusterer.pipeline import recluster_store
    from face_clusterer.prototypes import PrototypeStore
    from face_clusterer.store import EmbeddingStore

    rng = np.random.default_rng(21)
    cfg = _config(tmp_path)

    def unit():
        v = rng.normal(size=512)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def jitter(base):
        e = base + rng.normal(scale=0.02, size=512)
        return (e / np.linalg.norm(e)).astype(np.float32)

    bases = [unit(), unit()]
    protos = PrototypeStore(cfg.prototypes)
    known = [protos.add(b) for b in bases]
    protos.save()

    store = EmbeddingStore(cfg.output.db_path)
    def add(emb, person):
        return store.add_sample(
            FaceSample(
                bbox=np.array([0, 0, 10, 10], dtype=np.float32), det_score=0.9,
                embedding=emb, landmarks=None, source="x.jpg",
            ),
            person_id=person,
        )
    # 2 samples per known person — below min_cluster_size=4, so pure noise
    # for HDBSCAN — plus one genuine stranger.
    for base, pid in zip(bases, known):
        add(jitter(base), pid)
        add(jitter(base), pid)
    stranger_row = add(unit(), None)
    store.close()

    summary = recluster_store(cfg)
    assert summary["recovered_from_noise"] == 4

    store = EmbeddingStore(cfg.output.db_path)
    rows = dict(store.conn.execute("SELECT id, person_id FROM faces").fetchall())
    store.close()
    assert rows[stranger_row] is None            # stranger stays unknown
    assigned = [p for r, p in rows.items() if r != stranger_row]
    assert sorted(set(assigned)) == sorted(known)  # nobody lost their identity
