import numpy as np
import pytest
from fastapi.testclient import TestClient

from face_clusterer.api import create_app
from face_clusterer.config import (
    ApiConfig, AppConfig, ClassifierConfig, OutputConfig, PrototypeConfig,
)
from face_clusterer.embedder import FaceSample
from face_clusterer.prototypes import PrototypeStore
from face_clusterer.store import EmbeddingStore

DIM = 512


def _unit(rng, dim=DIM):
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _jitter(rng, base, scale=0.02):
    e = base + rng.normal(scale=scale, size=len(base))
    return (e / np.linalg.norm(e)).astype(np.float32)


@pytest.fixture
def api(tmp_path):
    """An API backed by a temp store seeded with 3 persons x 9 faces."""
    rng = np.random.default_rng(7)
    cfg = AppConfig(
        prototypes=PrototypeConfig(path=str(tmp_path / "prototypes")),
        classifier=ClassifierConfig(
            enabled=True, path=str(tmp_path / "classifier")
        ),
        api=ApiConfig(embedding_dim=DIM),
        output=OutputConfig(
            dir=str(tmp_path), db_path=str(tmp_path / "faces.sqlite")
        ),
    )

    bases = [_unit(rng) for _ in range(3)]
    protos = PrototypeStore(cfg.prototypes)
    store = EmbeddingStore(cfg.output.db_path)
    for i, base in enumerate(bases):
        pid = protos.add(base)
        assert pid == f"person_{i:04d}"
        for k in range(9):
            store.add_sample(
                FaceSample(
                    bbox=np.array([0, 0, 100, 100], dtype=np.float32),
                    det_score=0.9, embedding=_jitter(rng, base), landmarks=None,
                    source=f"seed/p{i}_{k}.jpg", frame_idx=k, quality=0.8,
                ),
                cluster_id=i, person_id=pid,
            )
    protos.save()
    store.close()

    with TestClient(create_app(cfg)) as client:
        yield client, bases, rng


def test_health_reports_known_persons(api):
    client, _, _ = api
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["persons_known"] == 3
    assert body["classifier_enabled"] is True


def test_identify_known_person_via_prototype(api):
    client, bases, rng = api
    probe = _jitter(rng, bases[1]).tolist()
    body = client.post("/identify", json={"embedding": probe}).json()
    assert body["person_id"] == "person_0001"
    assert body["source"] == "prototype"        # classifier not trained yet
    assert body["committed"] is False


def test_identify_stranger_is_unknown(api):
    client, _, rng = api
    body = client.post("/identify", json={"embedding": _unit(rng).tolist()}).json()
    assert body["person_id"] is None and body["source"] == "unknown"


def test_identify_rejects_malformed_embeddings(api):
    client, _, _ = api
    assert client.post("/identify", json={"embedding": [0.1, 0.2]}).status_code == 422
    assert client.post(
        "/identify", json={"embedding": [0.0] * DIM}
    ).status_code == 422                        # zero vector has no direction


def test_identify_rejects_non_finite_values(api):
    """NaN/Infinity are not valid JSON, so they must be posted raw."""
    client, _, _ = api
    for literal in ("NaN", "Infinity"):
        values = ", ".join([literal] + ["0.0"] * (DIM - 1))
        response = client.post(
            "/identify",
            content=f'{{"embedding": [{values}]}}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422


def test_identify_accepts_unnormalized_vector(api):
    """Clients may post raw model output; the API normalizes it."""
    client, bases, rng = api
    scaled = (_jitter(rng, bases[0]) * 17.5).tolist()
    body = client.post("/identify", json={"embedding": scaled}).json()
    assert body["person_id"] == "person_0000" and body["score"] > 0.9


def test_identify_commit_updates_prototype(api):
    client, bases, rng = api
    probe = _jitter(rng, bases[2]).tolist()
    before = client.get("/persons/person_0002").json()["embeddings_absorbed"]

    body = client.post(
        "/identify", json={"embedding": probe, "commit": True}
    ).json()
    assert body["person_id"] == "person_0002" and body["committed"] is True

    after = client.get("/persons/person_0002").json()["embeddings_absorbed"]
    assert after == before + 1


def test_list_and_get_persons(api):
    client, _, _ = api
    persons = client.get("/persons").json()
    assert [p["person_id"] for p in persons] == [
        "person_0000", "person_0001", "person_0002"
    ]
    assert all(p["faces_in_store"] == 9 for p in persons)

    assert client.get("/persons/person_0000").json()["faces_in_store"] == 9
    assert client.get("/persons/nobody").status_code == 404


def test_label_person(api):
    client, _, _ = api
    body = client.patch(
        "/persons/person_0001", json={"label": "Anna"}
    ).json()
    assert body["label"] == "Anna"
    # the label survives and shows up in identification
    assert client.get("/persons/person_0001").json()["label"] == "Anna"
    assert client.patch("/persons/ghost", json={"label": "x"}).status_code == 404


def test_identify_returns_label(api):
    client, bases, rng = api
    client.patch("/persons/person_0000", json={"label": "Bartek"})
    body = client.post(
        "/identify", json={"embedding": _jitter(rng, bases[0]).tolist()}
    ).json()
    assert body["person_id"] == "person_0000" and body["label"] == "Bartek"


def test_delete_person(api):
    client, bases, rng = api
    assert client.delete("/persons/person_0001").json() == {
        "deleted": "person_0001"
    }
    assert client.get("/persons/person_0001").status_code == 404
    assert client.get("/health").json()["persons_known"] == 2
    # a face of the deleted person is no longer recognized
    body = client.post(
        "/identify", json={"embedding": _jitter(rng, bases[1]).tolist()}
    ).json()
    assert body["person_id"] is None
    assert client.delete("/persons/person_0001").status_code == 404


def test_update_prototype_endpoint(api):
    client, bases, rng = api
    probe = _jitter(rng, bases[0]).tolist()
    body = client.post(
        "/update-prototype",
        json={"person_id": "person_0000", "embedding": probe},
    ).json()
    assert body["person_id"] == "person_0000"

    missing = client.post(
        "/update-prototype", json={"person_id": "nope", "embedding": probe}
    )
    assert missing.status_code == 404


def test_clusters_endpoint(api):
    client, _, _ = api
    body = client.get("/clusters").json()
    assert body["total_faces"] == 27 and body["persons"] == 3
    assert {c["cluster_id"] for c in body["clusters"]} == {0, 1, 2}
    assert all(c["faces"] == 9 for c in body["clusters"])


def test_retrain_classifier_switches_cascade(api):
    client, bases, rng = api
    # before training the cascade resolves via prototypes
    probe = _jitter(rng, bases[1]).tolist()
    assert client.post("/identify", json={"embedding": probe}).json()[
        "source"
    ] == "prototype"

    report = client.post("/retrain-classifier").json()
    assert report["trained"] is True and report["n_persons"] == 3

    # afterwards the freshly trained classifier answers first
    body = client.post("/identify", json={"embedding": probe}).json()
    assert body["person_id"] == "person_0001" and body["source"] == "classifier"
    assert client.get("/health").json()["classifier_ready"] is True


def test_metrics_exposes_counters(api):
    client, bases, rng = api
    client.post("/identify", json={"embedding": _jitter(rng, bases[0]).tolist()})
    text = client.get("/metrics").text
    assert "fic_identifications_total" in text
    assert "fic_persons_known" in text


def test_detect_without_insightface_returns_503(api):
    """The detector stack is optional — the API must say so, not crash."""
    import cv2

    client, _, _ = api
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    assert ok

    response = client.post(
        "/detect", files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    )
    # 200 when InsightFace is installed (no faces in a blank frame), 503 when not
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert response.json()["faces"] == []
    else:
        assert "unavailable" in response.json()["detail"]


def test_detect_rejects_undecodable_upload(api):
    client, _, _ = api
    response = client.post(
        "/detect", files={"file": ("x.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 422
