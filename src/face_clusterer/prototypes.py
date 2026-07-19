"""Person prototypes: the "online learning" layer.

The embedding model stays frozen; what adapts over time are per-person
prototype vectors (EMA-updated mean embeddings). New faces are first matched
against known prototypes (fast re-identification); only unmatched faces go to
clustering. The store persists to disk (JSON metadata + NPZ vectors), so
identities survive across runs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PrototypeConfig
from .embedder import l2_normalize

logger = logging.getLogger(__name__)


@dataclass
class Prototype:
    person_id: str
    vector: np.ndarray               # (512,) L2-normalized
    count: int = 1                   # embeddings absorbed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    label: str | None = None         # optional human-assigned name


class PrototypeStore:
    """Cosine-similarity prototype matcher with EMA updates and persistence."""

    def __init__(self, cfg: PrototypeConfig | None = None):
        self.cfg = cfg or PrototypeConfig()
        self.prototypes: dict[str, Prototype] = {}
        self._next_num = 0

    # ---------------- matching / updates ----------------

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (person_id, similarity) of the best match, or (None, sim)."""
        if not self.prototypes:
            return None, 0.0
        ids = list(self.prototypes)
        matrix = np.vstack([self.prototypes[i].vector for i in ids])
        sims = matrix @ np.asarray(embedding, dtype=np.float32)
        best = int(np.argmax(sims))
        sim = float(sims[best])
        if sim >= self.cfg.match_threshold:
            return ids[best], sim
        return None, sim

    def update(self, person_id: str, embedding: np.ndarray):
        """EMA-update an existing prototype with a new embedding."""
        proto = self.prototypes[person_id]
        a = self.cfg.ema_alpha
        proto.vector = l2_normalize((1.0 - a) * proto.vector + a * embedding)
        proto.count += 1
        proto.updated_at = time.time()

    def add(self, embedding: np.ndarray, count: int = 1) -> str:
        """Register a new person; returns the generated person_id."""
        person_id = f"person_{self._next_num:04d}"
        self._next_num += 1
        self.prototypes[person_id] = Prototype(
            person_id=person_id, vector=l2_normalize(embedding), count=count
        )
        return person_id

    def match_or_add(self, embedding: np.ndarray) -> tuple[str, float, bool]:
        """Match to an existing person (and update it) or create a new one.

        Returns (person_id, similarity, is_new).
        """
        person_id, sim = self.match(embedding)
        if person_id is not None:
            self.update(person_id, embedding)
            return person_id, sim, False
        return self.add(embedding), sim, True

    def absorb_clusters(
        self, centroids: dict[int, np.ndarray], counts: dict[int, int] | None = None
    ) -> dict[int, str]:
        """Map cluster centroids to persons (re-id across runs); returns cluster->person."""
        mapping: dict[int, str] = {}
        for cluster_id, centroid in sorted(centroids.items()):
            person_id, sim = self.match(centroid)
            if person_id is not None:
                self.update(person_id, centroid)
                logger.info(
                    "Cluster %d re-identified as %s (sim=%.3f)",
                    cluster_id, person_id, sim,
                )
            else:
                person_id = self.add(
                    centroid, count=(counts or {}).get(cluster_id, 1)
                )
                logger.info("Cluster %d -> new %s", cluster_id, person_id)
            mapping[cluster_id] = person_id
        return mapping

    # ---------------- persistence ----------------

    def save(self, path: str | Path | None = None):
        base = Path(path or self.cfg.path)
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "next_num": self._next_num,
            "prototypes": [
                {
                    "person_id": p.person_id,
                    "count": p.count,
                    "created_at": p.created_at,
                    "updated_at": p.updated_at,
                    "label": p.label,
                }
                for p in self.prototypes.values()
            ],
        }
        Path(f"{base}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            f"{base}.npz",
            **{p.person_id: p.vector for p in self.prototypes.values()},
        )

    def load(self, path: str | Path | None = None) -> bool:
        base = Path(path or self.cfg.path)
        json_path, npz_path = Path(f"{base}.json"), Path(f"{base}.npz")
        if not (json_path.exists() and npz_path.exists()):
            return False
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        vectors = np.load(npz_path)
        self.prototypes = {}
        for entry in meta["prototypes"]:
            pid = entry["person_id"]
            if pid not in vectors:
                continue
            self.prototypes[pid] = Prototype(
                person_id=pid,
                vector=l2_normalize(vectors[pid]),
                count=int(entry.get("count", 1)),
                created_at=float(entry.get("created_at", 0.0)),
                updated_at=float(entry.get("updated_at", 0.0)),
                label=entry.get("label"),
            )
        self._next_num = int(meta.get("next_num", len(self.prototypes)))
        logger.info("Loaded %d prototypes from %s", len(self.prototypes), base)
        return True
