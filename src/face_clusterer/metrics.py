"""Prometheus metrics (observability layer).

`prometheus_client` is optional: when it is missing every metric degrades to a
no-op, so importing this module never breaks a deployment that does not scrape
metrics. This mirrors how the package treats InsightFace and hdbscan.

What is measured maps to the four questions the system should be able to answer
in production: is it keeping up (frame/identify latency), is it seeing anything
(faces detected/rejected), are the embeddings any good (quality histogram), and
are the identities stable (persons known, identification source split).
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class _NoopMetric:
    """Stand-in with the prometheus_client surface used here."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, amount: float = 1):
        pass

    def observe(self, value: float):
        pass

    def set(self, value: float):
        pass


def _build():
    """Create the real metrics, or no-ops when prometheus_client is absent."""
    try:
        from prometheus_client import Counter, Gauge, Histogram
    except ImportError:  # pragma: no cover - exercised only without the dep
        logger.info("prometheus_client not installed — metrics disabled.")
        noop = _NoopMetric()
        return dict.fromkeys(
            (
                "faces_detected", "faces_rejected", "identifications",
                "identify_latency", "frame_latency", "face_quality",
                "persons_known", "clusters_total", "classifier_persons",
                "classifier_trainings", "prototype_updates",
            ),
            noop,
        )

    return {
        "faces_detected": Counter(
            "fic_faces_detected_total", "Faces detected and embedded."
        ),
        "faces_rejected": Counter(
            "fic_faces_rejected_total", "Faces dropped by the quality filter."
        ),
        "identifications": Counter(
            "fic_identifications_total",
            "Identification attempts by resolution source.",
            ["source"],                       # classifier | prototype | unknown
        ),
        "identify_latency": Histogram(
            "fic_identify_latency_seconds", "Latency of one identification."
        ),
        "frame_latency": Histogram(
            "fic_frame_latency_seconds",
            "Wall time to process one frame (FPS = 1 / this).",
        ),
        "face_quality": Histogram(
            "fic_face_quality", "Quality score of accepted faces.",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        ),
        "persons_known": Gauge(
            "fic_persons_known", "Person prototypes currently stored."
        ),
        "clusters_total": Gauge(
            "fic_clusters_total", "Clusters produced by the last clustering run."
        ),
        "classifier_persons": Gauge(
            "fic_classifier_persons", "Classes the online classifier can predict."
        ),
        "classifier_trainings": Counter(
            "fic_classifier_trainings_total", "Classifier training runs."
        ),
        "prototype_updates": Counter(
            "fic_prototype_updates_total", "EMA updates applied to prototypes."
        ),
    }


_M = _build()


# ---------------- recording helpers ----------------
# Call sites stay free of prometheus imports and of None-checks.

def record_faces(detected: int = 0, rejected: int = 0):
    if detected:
        _M["faces_detected"].inc(detected)
    if rejected:
        _M["faces_rejected"].inc(rejected)


def record_quality(score: float):
    _M["face_quality"].observe(float(score))


def record_identification(source: str, seconds: float | None = None):
    _M["identifications"].labels(source=source).inc()
    if seconds is not None:
        _M["identify_latency"].observe(seconds)


def record_prototype_update(count: int = 1):
    _M["prototype_updates"].inc(count)


def record_training(n_persons: int):
    _M["classifier_trainings"].inc()
    _M["classifier_persons"].set(n_persons)


def set_persons_known(count: int):
    _M["persons_known"].set(count)


def set_clusters(count: int):
    _M["clusters_total"].set(count)


@contextmanager
def time_frame():
    """Time one processed frame; FPS is derived from this histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _M["frame_latency"].observe(time.perf_counter() - start)


def render() -> bytes:
    """Serialize the registry in Prometheus text format."""
    try:
        from prometheus_client import REGISTRY, generate_latest
    except ImportError:  # pragma: no cover
        return b"# prometheus_client not installed\n"
    return generate_latest(REGISTRY)
