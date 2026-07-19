"""Face detection + ArcFace embedding via InsightFace (ONNX Runtime, CPU-friendly).

InsightFace is imported lazily so the rest of the package (clustering,
prototypes, tracker) works without it — e.g. in unit tests or when
re-clustering previously stored embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import DetectorConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False)  # identity-based equality (fields contain numpy arrays)
class FaceSample:
    """A single detected face with its embedding and metadata."""

    bbox: np.ndarray                  # (4,) float32: x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray             # (512,) float32, L2-normalized
    landmarks: np.ndarray | None      # (5, 2) float32 or None
    crop: np.ndarray | None = None    # BGR crop of the face (for reports)
    source: str = ""                  # file path / camera id
    frame_idx: int = -1
    track_id: int = -1
    quality: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(min(x2 - x1, y2 - y1))


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize an embedding (required for cosine-based clustering)."""
    vec = np.asarray(vec, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + eps)


class FaceEmbedder:
    """Wraps insightface.app.FaceAnalysis: detection + alignment + embedding."""

    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self._app = None

    def _ensure_model(self):
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "insightface is not installed. Run: pip install -r requirements.txt"
            ) from exc
        logger.info("Loading InsightFace model pack '%s'...", self.cfg.model_pack)
        self._app = FaceAnalysis(
            name=self.cfg.model_pack,
            providers=list(self.cfg.providers),
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=(self.cfg.det_size, self.cfg.det_size))
        logger.info("Model ready.")

    def extract(
        self,
        frame_bgr: np.ndarray,
        source: str = "",
        frame_idx: int = -1,
        keep_crops: bool = True,
    ) -> list[FaceSample]:
        """Detect all faces in a BGR frame and return embedded samples."""
        self._ensure_model()
        h, w = frame_bgr.shape[:2]
        samples: list[FaceSample] = []
        for face in self._app.get(frame_bgr):
            if face.det_score < self.cfg.det_score_threshold:
                continue
            if getattr(face, "embedding", None) is None:
                continue
            bbox = np.asarray(face.bbox, dtype=np.float32)
            crop = None
            if keep_crops:
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame_bgr[y1:y2, x1:x2].copy()
            samples.append(
                FaceSample(
                    bbox=bbox,
                    det_score=float(face.det_score),
                    embedding=l2_normalize(face.embedding),
                    landmarks=(
                        np.asarray(face.kps, dtype=np.float32)
                        if getattr(face, "kps", None) is not None
                        else None
                    ),
                    crop=crop,
                    source=source,
                    frame_idx=frame_idx,
                )
            )
        return samples
