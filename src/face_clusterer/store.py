"""SQLite embedding store: every accepted face with metadata + embedding blob.

Lets you re-cluster later without re-running detection, audit assignments, and
feed offline retraining datasets.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from .embedder import FaceSample

_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    frame_idx INTEGER,
    track_id INTEGER,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    det_score REAL,
    quality REAL,
    embedding BLOB NOT NULL,
    cluster_id INTEGER,
    person_id TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
"""


class EmbeddingStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def add_sample(
        self,
        sample: FaceSample,
        cluster_id: int | None = None,
        person_id: str | None = None,
    ) -> int:
        x1, y1, x2, y2 = (float(v) for v in sample.bbox)
        cur = self.conn.execute(
            "INSERT INTO faces (source, frame_idx, track_id, x1, y1, x2, y2,"
            " det_score, quality, embedding, cluster_id, person_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sample.source, sample.frame_idx, sample.track_id,
                x1, y1, x2, y2,
                sample.det_score, sample.quality,
                np.asarray(sample.embedding, dtype=np.float32).tobytes(),
                cluster_id, person_id, time.time(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_samples(self, samples, cluster_ids=None, person_ids=None) -> list[int]:
        ids = []
        for i, s in enumerate(samples):
            ids.append(
                self.add_sample(
                    s,
                    cluster_id=None if cluster_ids is None else int(cluster_ids[i]),
                    person_id=None if person_ids is None else person_ids[i],
                )
            )
        return ids

    def clear(self) -> int:
        """Drop every stored face. Used before a full re-analysis.

        Re-running analysis over the same folder would otherwise insert the same
        faces again: after a month of nightly full rebuilds each face would sit
        in the table thirty times, which silently distorts density-based
        clustering and inflates the classifier's training set.
        """
        removed = self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        self.conn.execute("DELETE FROM faces")
        self.conn.commit()
        return int(removed)

    def all_embeddings(self) -> tuple[np.ndarray, list[int]]:
        """Return (embeddings matrix, row ids) for re-clustering."""
        rows = self.conn.execute("SELECT id, embedding FROM faces").fetchall()
        if not rows:
            return np.empty((0, 512), dtype=np.float32), []
        ids = [int(r[0]) for r in rows]
        embs = np.vstack(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        )
        return embs, ids

    def labeled_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Return (embeddings, person_ids) for rows already assigned to a person.

        This is the training set for the online classifier — the labels come
        from clustering + prototype re-identification, not from humans.
        """
        rows = self.conn.execute(
            "SELECT embedding, person_id FROM faces WHERE person_id IS NOT NULL"
        ).fetchall()
        if not rows:
            return np.empty((0, 512), dtype=np.float32), []
        embs = np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
        return embs, [str(r[1]) for r in rows]

    def update_assignments(self, row_ids, cluster_ids, person_ids):
        self.conn.executemany(
            "UPDATE faces SET cluster_id = ?, person_id = ? WHERE id = ?",
            [
                (int(c), p, int(rid))
                for rid, c, p in zip(row_ids, cluster_ids, person_ids)
            ],
        )
        self.conn.commit()

    def cluster_summary(self) -> list[dict]:
        """Per-cluster face counts and the person each cluster maps to."""
        rows = self.conn.execute(
            "SELECT cluster_id, person_id, COUNT(*) FROM faces"
            " WHERE cluster_id IS NOT NULL AND cluster_id >= 0"
            " GROUP BY cluster_id, person_id ORDER BY cluster_id"
        ).fetchall()
        return [
            {"cluster_id": int(c), "person_id": p, "faces": int(n)}
            for c, p, n in rows
        ]

    def person_faces(self, person_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE person_id = ?", (person_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def summary(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        persons = self.conn.execute(
            "SELECT person_id, COUNT(*) FROM faces WHERE person_id IS NOT NULL"
            " GROUP BY person_id ORDER BY COUNT(*) DESC"
        ).fetchall()
        return {"total_faces": int(total), "persons": dict(persons)}

    def close(self):
        self.conn.close()
