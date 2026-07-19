"""Lightweight IoU-based multi-face tracker for video streams.

Zero heavy dependencies (no torch / DeepSORT needed). Faces are linked across
frames by bbox IoU with an embedding-similarity fallback, and each track
accumulates embeddings so they can be averaged into one stable vector per
person appearance — this dramatically reduces noise before clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TrackerConfig
from .embedder import FaceSample, l2_normalize


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    embeddings: list[np.ndarray] = field(default_factory=list)
    qualities: list[float] = field(default_factory=list)
    best_crop: np.ndarray | None = None
    best_quality: float = -1.0
    hits: int = 0
    age: int = 0                     # frames since last match
    first_frame: int = -1
    last_frame: int = -1

    def add(self, sample: FaceSample, max_embeddings: int):
        self.bbox = sample.bbox
        self.hits += 1
        self.age = 0
        self.last_frame = sample.frame_idx
        if self.first_frame < 0:
            self.first_frame = sample.frame_idx
        if len(self.embeddings) < max_embeddings:
            self.embeddings.append(sample.embedding)
            self.qualities.append(sample.quality)
        if sample.quality > self.best_quality and sample.crop is not None:
            self.best_quality = sample.quality
            self.best_crop = sample.crop

    def mean_embedding(self) -> np.ndarray:
        """Quality-weighted average of the track's embeddings, re-normalized."""
        embs = np.vstack(self.embeddings)
        weights = np.asarray(self.qualities, dtype=np.float32)
        if weights.sum() <= 1e-6:
            weights = np.ones_like(weights)
        mean = (embs * weights[:, None]).sum(axis=0) / weights.sum()
        return l2_normalize(mean)


class IouTracker:
    """Greedy IoU matcher with embedding-similarity fallback."""

    EMB_FALLBACK_SIM = 0.55  # cosine similarity accepted when IoU fails (fast motion)

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self._next_id = 0
        self.active: list[Track] = []
        self.finished: list[Track] = []

    def update(self, samples: list[FaceSample]) -> list[FaceSample]:
        """Assign track ids to this frame's samples; returns the same samples."""
        unmatched = list(samples)

        # 1. Greedy IoU matching (best pair first)
        pairs: list[tuple[float, Track, FaceSample]] = []
        for tr in self.active:
            for s in unmatched:
                score = iou(tr.bbox, s.bbox)
                if score >= self.cfg.iou_threshold:
                    pairs.append((score, tr, s))
        pairs.sort(key=lambda p: p[0], reverse=True)

        used_tracks: set[int] = set()
        for score, tr, s in pairs:
            if tr.track_id in used_tracks or s not in unmatched:
                continue
            self._assign(tr, s)
            used_tracks.add(tr.track_id)
            unmatched.remove(s)

        # 2. Embedding fallback for tracks that lost IoU (fast motion / stride)
        for tr in self.active:
            if tr.track_id in used_tracks or not unmatched or not tr.embeddings:
                continue
            proto = tr.mean_embedding()
            sims = [float(np.dot(proto, s.embedding)) for s in unmatched]
            best = int(np.argmax(sims))
            if sims[best] >= self.EMB_FALLBACK_SIM:
                s = unmatched.pop(best)
                self._assign(tr, s)
                used_tracks.add(tr.track_id)

        # 3. New tracks for remaining detections
        for s in unmatched:
            tr = Track(track_id=self._next_id, bbox=s.bbox)
            self._next_id += 1
            self._assign(tr, s)
            self.active.append(tr)
            used_tracks.add(tr.track_id)

        # 4. Age unmatched tracks; retire the stale ones
        still_active: list[Track] = []
        for tr in self.active:
            if tr.track_id not in used_tracks:
                tr.age += 1
            if tr.age > self.cfg.max_age:
                if tr.hits >= self.cfg.min_hits:
                    self.finished.append(tr)
            else:
                still_active.append(tr)
        self.active = still_active
        return samples

    def _assign(self, tr: Track, s: FaceSample):
        s.track_id = tr.track_id
        tr.add(s, self.cfg.max_embeddings_per_track)

    def flush(self) -> list[Track]:
        """End of stream: move confirmed active tracks to finished, return all."""
        for tr in self.active:
            if tr.hits >= self.cfg.min_hits:
                self.finished.append(tr)
        self.active = []
        return self.finished
