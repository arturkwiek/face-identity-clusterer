"""Face quality filtering: reject blurry, tiny, low-confidence and extreme-profile faces.

Poor samples distort embeddings and merge clusters, so filtering them out
noticeably improves cluster purity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import QualityConfig
from .embedder import FaceSample


@dataclass
class QualityResult:
    ok: bool
    score: float          # 0..1, aggregated quality
    reasons: list[str]    # why the face was rejected (empty if ok)


def blur_variance(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian — a standard sharpness measure."""
    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def yaw_asymmetry(landmarks: np.ndarray) -> float:
    """Ratio of eye-to-nose horizontal distances; large values = strong profile.

    landmarks: (5, 2) = [left_eye, right_eye, nose, mouth_left, mouth_right]
    """
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    d_left = abs(float(nose[0] - left_eye[0]))
    d_right = abs(float(right_eye[0] - nose[0]))
    lo, hi = sorted((d_left, d_right))
    if lo < 1e-3:
        return float("inf")
    return hi / lo


def assess(sample: FaceSample, cfg: QualityConfig) -> QualityResult:
    """Evaluate a face sample against the quality thresholds."""
    if not cfg.enabled:
        return QualityResult(True, 1.0, [])

    reasons: list[str] = []
    scores: list[float] = []

    # 1. Detection confidence
    scores.append(min(1.0, sample.det_score))
    if sample.det_score < cfg.min_det_score:
        reasons.append(f"low_det_score({sample.det_score:.2f})")

    # 2. Face size
    size = sample.size
    scores.append(min(1.0, size / max(1.0, 2.0 * cfg.min_face_size)))
    if size < cfg.min_face_size:
        reasons.append(f"too_small({size:.0f}px)")

    # 3. Sharpness (blur)
    if sample.crop is not None and sample.crop.size > 0:
        bv = blur_variance(sample.crop)
        scores.append(min(1.0, bv / max(1e-6, 2.0 * cfg.min_blur_variance)))
        if bv < cfg.min_blur_variance:
            reasons.append(f"blurry({bv:.0f})")

    # 4. Extreme profile (yaw)
    if sample.landmarks is not None and len(sample.landmarks) >= 3:
        asym = yaw_asymmetry(sample.landmarks)
        if asym > cfg.max_yaw_asymmetry:
            reasons.append(f"profile(asym={asym:.1f})")
            scores.append(0.3)

    score = float(np.mean(scores)) if scores else 0.0
    return QualityResult(ok=not reasons, score=score, reasons=reasons)


def filter_samples(
    samples: list[FaceSample], cfg: QualityConfig
) -> tuple[list[FaceSample], list[tuple[FaceSample, QualityResult]]]:
    """Split samples into (accepted, rejected-with-reasons); stores quality score."""
    accepted: list[FaceSample] = []
    rejected: list[tuple[FaceSample, QualityResult]] = []
    for s in samples:
        result = assess(s, cfg)
        s.quality = result.score
        if result.ok:
            accepted.append(s)
        else:
            rejected.append((s, result))
    return accepted, rejected
