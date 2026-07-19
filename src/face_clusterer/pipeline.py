"""Pipelines: batch photos, video files, and live camera.

Common flow: detect -> embed -> quality filter -> (track for video) ->
cluster -> map clusters to persistent person prototypes -> reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import metrics
from .classifier import IdentityResolver, OnlineClassifier, TrainingReport
from .clustering import NOISE, cluster_embeddings
from .config import AppConfig
from .embedder import FaceEmbedder, FaceSample
from .prototypes import PrototypeStore
from .quality import filter_samples
from .report import save_crops, save_montages, save_summary
from .store import EmbeddingStore
from .tracker import IouTracker, Track

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BasePipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.embedder = FaceEmbedder(cfg.detector)
        self.prototypes = PrototypeStore(cfg.prototypes)
        if cfg.prototypes.enabled:
            self.prototypes.load()

        self.classifier: OnlineClassifier | None = None
        if cfg.classifier.enabled:
            self.classifier = OnlineClassifier(cfg.classifier)
            self.classifier.load()
        self.resolver = IdentityResolver(self.prototypes, self.classifier)

    # ---------- shared post-processing ----------

    def _finalize(
        self,
        samples: list[FaceSample],
        embeddings: np.ndarray,
        extra: dict | None = None,
        reset_store: bool = False,
    ) -> dict:
        """Cluster embeddings, map to persons, persist and report.

        `reset_store` replaces the stored faces instead of appending to them —
        correct when this run re-analyzes the entire dataset. Prototypes are
        deliberately *not* reset, so person ids stay stable across rebuilds.
        """
        out_dir = Path(self.cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = cluster_embeddings(embeddings, self.cfg.clustering)
        logger.info(
            "Clustering: %d identities, %d noise (%d reattached, %d purged) "
            "from %d samples",
            result.n_clusters, result.n_noise, result.reattached,
            result.purged, len(samples),
        )
        metrics.set_clusters(result.n_clusters)
        for s in samples:
            metrics.record_quality(s.quality)

        person_ids: list[str | None]
        if self.cfg.prototypes.enabled:
            counts = {
                int(c): int((result.labels == c).sum()) for c in result.centroids
            }
            mapping = self.prototypes.absorb_clusters(result.centroids, counts)
            person_ids = [
                mapping.get(int(lbl)) if lbl != NOISE else None
                for lbl in result.labels
            ]
            # Samples HDBSCAN called noise still get a chance through the
            # cascade: a face can be un-clusterable (a single track in a video,
            # a person seen once) and still be a person we already know. Without
            # this, known people are silently discarded whenever they appear too
            # few times to form a cluster of their own.
            recovered = 0
            for i, label in enumerate(result.labels):
                if label != NOISE:
                    continue
                ident = self.resolver.resolve(embeddings[i])
                if ident.person_id is not None:
                    self.resolver.commit(ident, embeddings[i])
                    person_ids[i] = ident.person_id
                    recovered += 1
            if recovered:
                logger.info(
                    "Recovered %d noise samples via %s.",
                    recovered,
                    "classifier/prototypes" if self.classifier else "prototypes",
                )
            extra = {**(extra or {}), "recovered_from_noise": recovered}

            self.prototypes.save()
            metrics.set_persons_known(len(self.prototypes.prototypes))
        else:
            person_ids = [
                f"cluster_{lbl}" if lbl != NOISE else None for lbl in result.labels
            ]

        # Persist embeddings + assignments
        store = EmbeddingStore(self.cfg.output.db_path)
        if reset_store:
            removed = store.clear()
            logger.info("Full re-analysis: replaced %d stored faces.", removed)
        store.add_samples(
            samples,
            cluster_ids=result.labels,
            person_ids=person_ids,
        )
        store.close()

        # Reports
        if self.cfg.output.save_crops:
            save_crops(
                samples, person_ids, out_dir, self.cfg.output.max_crops_per_cluster
            )
            save_montages(out_dir)
        summary = save_summary(
            out_dir, samples, person_ids,
            extra={
                **(extra or {}),
                "reattached_noise": result.reattached,
                "purged_outliers": result.purged,
            },
        )
        return summary


class ImagesPipeline(BasePipeline):
    """Batch clustering of a folder of photos."""

    def run(self, images_dir: str | Path, reset_store: bool = False) -> dict:
        images_dir = Path(images_dir)
        paths = sorted(
            p for p in images_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {images_dir}")
        logger.info("Processing %d images from %s", len(paths), images_dir)

        import cv2

        accepted: list[FaceSample] = []
        n_rejected = 0
        for i, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                logger.warning("Cannot read %s — skipped", path)
                continue
            with metrics.time_frame():
                samples = self.embedder.extract(frame, source=str(path), frame_idx=i)
                ok, rejected = filter_samples(samples, self.cfg.quality)
            metrics.record_faces(detected=len(ok), rejected=len(rejected))
            accepted.extend(ok)
            n_rejected += len(rejected)

        logger.info(
            "Faces: %d accepted, %d rejected by quality filter",
            len(accepted), n_rejected,
        )
        if not accepted:
            return {"total_faces": 0, "identities": 0}
        embeddings = np.vstack([s.embedding for s in accepted])
        return self._finalize(
            accepted, embeddings,
            extra={"images": len(paths), "rejected_faces": n_rejected},
            reset_store=reset_store,
        )


class VideoPipeline(BasePipeline):
    """Video file: tracking + per-track quality-weighted embedding averaging.

    Clustering runs on one averaged embedding per track (not per frame), which
    is both faster and far more robust to per-frame noise.
    """

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg"}

    def _extract_tracks(
        self, video_path: Path
    ) -> tuple[list[FaceSample], int, int]:
        """Track faces through one video; returns (samples, frames, rejected).

        One sample per track, not per frame: the track's quality-weighted mean
        embedding. A fresh tracker per video prevents track ids from one
        recording leaking into the next.
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        tracker = IouTracker(self.cfg.tracker)
        stride = max(1, self.cfg.video.frame_stride)
        frame_idx, n_rejected = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                with metrics.time_frame():
                    samples = self.embedder.extract(
                        frame, source=str(video_path), frame_idx=frame_idx
                    )
                    ok, rejected = filter_samples(samples, self.cfg.quality)
                    tracker.update(ok)
                metrics.record_faces(detected=len(ok), rejected=len(rejected))
                n_rejected += len(rejected)
                if self.cfg.video.display:
                    self._draw(frame, ok)
                    cv2.imshow("face-identity-clusterer", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            frame_idx += 1

        cap.release()
        if self.cfg.video.display:
            cv2.destroyAllWindows()

        tracks = [t for t in tracker.flush() if t.embeddings]
        logger.info(
            "%s: %d frames, %d confirmed tracks, %d rejected faces",
            video_path.name, frame_idx, len(tracks), n_rejected,
        )
        samples = [self._track_to_sample(t, str(video_path)) for t in tracks]
        return samples, frame_idx, n_rejected

    def run(self, video_path: str | Path, reset_store: bool = False) -> dict:
        """Process one video file, or every video in a folder."""
        path = Path(video_path)
        if path.is_dir():
            return self.run_many(sorted(
                p for p in path.rglob("*")
                if p.suffix.lower() in self.VIDEO_EXTENSIONS
            ), reset_store=reset_store)

        samples, frames, n_rejected = self._extract_tracks(path)
        if not samples:
            return {"total_faces": 0, "identities": 0}

        embeddings = np.vstack([s.embedding for s in samples])
        return self._finalize(
            samples, embeddings,
            extra={
                "frames": frames,
                "tracks": len(samples),
                "rejected_faces": n_rejected,
            },
            reset_store=reset_store,
        )

    def run_many(
        self, video_paths: list[Path], reset_store: bool = False
    ) -> dict:
        """Process several videos and cluster their tracks together.

        Clustering once over all videos — rather than once per file — is what
        lets the same person appearing in two recordings land in one identity.
        """
        if not video_paths:
            raise FileNotFoundError("No video files found.")
        logger.info("Processing %d video files", len(video_paths))

        all_samples: list[FaceSample] = []
        total_frames = total_rejected = 0
        processed, failed = [], []
        for path in video_paths:
            try:
                samples, frames, rejected = self._extract_tracks(path)
            except IOError as exc:
                # One unreadable file must not discard the whole batch.
                logger.warning("Skipping %s: %s", path.name, exc)
                failed.append(path.name)
                continue
            all_samples.extend(samples)
            total_frames += frames
            total_rejected += rejected
            processed.append(path.name)

        if not all_samples:
            return {"total_faces": 0, "identities": 0, "videos_failed": failed}

        embeddings = np.vstack([s.embedding for s in all_samples])
        return self._finalize(
            all_samples, embeddings,
            extra={
                "videos": len(processed),
                "videos_failed": failed,
                "frames": total_frames,
                "tracks": len(all_samples),
                "rejected_faces": total_rejected,
            },
            reset_store=reset_store,
        )

    @staticmethod
    def _track_to_sample(track: Track, source: str) -> FaceSample:
        return FaceSample(
            bbox=track.bbox,
            det_score=1.0,
            embedding=track.mean_embedding(),
            landmarks=None,
            crop=track.best_crop,
            source=source,
            frame_idx=track.first_frame,
            track_id=track.track_id,
            quality=float(np.mean(track.qualities)) if track.qualities else 1.0,
            meta={"n_frames": len(track.embeddings)},
        )

    @staticmethod
    def _draw(frame, samples: list[FaceSample]):
        import cv2

        for s in samples:
            x1, y1, x2, y2 = s.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"t{s.track_id}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )


class LivePipeline(BasePipeline):
    """Live camera: instant re-identification against prototypes + periodic clustering.

    Known persons are labeled in real time via the prototype store; unknown
    faces accumulate and get clustered every `recluster_every` processed frames.
    """

    def run(self, camera: int | str = 0, recluster_every: int = 300) -> dict:
        import cv2

        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera: {camera}")

        tracker = IouTracker(self.cfg.tracker)
        stride = max(1, self.cfg.video.frame_stride)
        display = self.cfg.video.display
        # Every accepted face is persisted as it is seen (design notes, layer
        # 6): live hours must feed the same store that recluster and classifier
        # training read, or that material is simply lost.
        store = EmbeddingStore(self.cfg.output.db_path)
        pending: list[FaceSample] = []
        pending_rows: list[int] = []
        frame_idx = processed = stored = 0
        source = f"camera:{camera}"

        if display:
            logger.info("Live mode — press ESC in the preview window to stop.")
        else:
            logger.info("Live mode (headless) — Ctrl+C stops.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    with metrics.time_frame():
                        samples = self.embedder.extract(
                            frame, source=source, frame_idx=frame_idx
                        )
                        ok, rejected = filter_samples(samples, self.cfg.quality)
                        tracker.update(ok)
                    metrics.record_faces(
                        detected=len(ok), rejected=len(rejected)
                    )
                    for s in ok:
                        metrics.record_quality(s.quality)
                        ident = self.resolver.resolve(s.embedding)
                        metrics.record_identification(ident.source)
                        if ident.person_id is not None:
                            self.resolver.commit(ident, s.embedding)
                            s.meta["person"] = ident.person_id
                            s.meta["sim"] = ident.score
                            s.meta["via"] = ident.source
                            store.add_sample(s, person_id=ident.person_id)
                        else:
                            row_id = store.add_sample(s)
                            pending.append(s)
                            pending_rows.append(row_id)
                        stored += 1
                    if display:
                        self._draw_live(frame, ok)
                    processed += 1
                    if processed % recluster_every == 0 and pending:
                        self._recluster_pending(pending, pending_rows, store)
                        pending, pending_rows = [], []
                if display:
                    cv2.imshow("face-identity-clusterer [live]", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                frame_idx += 1
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

            # Shutdown work sits inside finally so Ctrl+C — the normal way to
            # stop a headless live run — still clusters the tail of pending
            # faces and persists prototypes.
            if pending:
                self._recluster_pending(pending, pending_rows, store)
            if self.cfg.prototypes.enabled:
                self.prototypes.save()
            store.close()
        return {
            "frames": frame_idx,
            "faces_stored": stored,
            "persons_known": len(self.prototypes.prototypes),
        }

    def _recluster_pending(
        self,
        pending: list[FaceSample],
        pending_rows: list[int],
        store: EmbeddingStore,
    ):
        embeddings = np.vstack([s.embedding for s in pending])
        result = cluster_embeddings(embeddings, self.cfg.clustering)
        if result.centroids:
            counts = {
                int(c): int((result.labels == c).sum()) for c in result.centroids
            }
            mapping = self.prototypes.absorb_clusters(result.centroids, counts)
            # Update the rows stored earlier as unknown with their new persons.
            assigned = [
                (row, int(lbl), mapping[int(lbl)])
                for row, lbl in zip(pending_rows, result.labels)
                if lbl != NOISE
            ]
            if assigned:
                rows, labels, persons = zip(*assigned)
                store.update_assignments(rows, labels, persons)
            self.prototypes.save()
            logger.info(
                "Reclustered %d pending faces -> %d identities "
                "(known persons total: %d)",
                len(pending), result.n_clusters, len(self.prototypes.prototypes),
            )

    @staticmethod
    def _draw_live(frame, samples: list[FaceSample]):
        import cv2

        for s in samples:
            x1, y1, x2, y2 = s.bbox.astype(int)
            person = s.meta.get("person")
            color = (0, 255, 0) if person else (0, 165, 255)
            label = person or "unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )


def recluster_store(cfg: AppConfig) -> dict:
    """Re-cluster every stored embedding without re-running detection.

    Noise goes through the same classifier -> prototypes cascade as in
    `_finalize`. Without this, `update_assignments` would overwrite the
    person_id of every row HDBSCAN calls noise with None — so a person
    correctly identified in an earlier run could be silently *un*-assigned
    just by running recluster over a store that has too few of their samples
    to form a cluster.
    """
    store = EmbeddingStore(cfg.output.db_path)
    try:
        embeddings, row_ids = store.all_embeddings()
        if len(row_ids) == 0:
            return {"total_faces": 0, "identities": 0}

        result = cluster_embeddings(embeddings, cfg.clustering)

        prototypes = PrototypeStore(cfg.prototypes)
        prototypes.load()
        classifier = None
        if cfg.classifier.enabled:
            classifier = OnlineClassifier(cfg.classifier)
            classifier.load()
        resolver = IdentityResolver(prototypes, classifier)

        counts = {
            int(c): int((result.labels == c).sum()) for c in result.centroids
        }
        mapping = prototypes.absorb_clusters(result.centroids, counts)
        person_ids: list[str | None] = [
            mapping.get(int(lbl)) if lbl != NOISE else None
            for lbl in result.labels
        ]

        recovered = 0
        for i, label in enumerate(result.labels):
            if label != NOISE:
                continue
            ident = resolver.resolve(embeddings[i])
            if ident.person_id is not None:
                resolver.commit(ident, embeddings[i])
                person_ids[i] = ident.person_id
                recovered += 1
        if recovered:
            logger.info("Recluster: recovered %d noise samples.", recovered)

        store.update_assignments(row_ids, result.labels, person_ids)
        prototypes.save()
        metrics.set_persons_known(len(prototypes.prototypes))
        metrics.set_clusters(result.n_clusters)

        summary = store.summary()
        summary.update(
            {
                "identities": result.n_clusters,
                "noise_faces": int(result.n_noise),
                "recovered_from_noise": recovered,
            }
        )
        return summary
    finally:
        store.close()


def train_classifier(cfg: AppConfig) -> TrainingReport:
    """Refit the online classifier on every person-assigned face in the store.

    Labels come from clustering + prototype re-identification, so this is
    self-supervised: no human annotation is involved.
    """
    store = EmbeddingStore(cfg.output.db_path)
    try:
        embeddings, person_ids = store.labeled_embeddings()
    finally:
        store.close()

    classifier = OnlineClassifier(cfg.classifier)
    report = classifier.fit(embeddings, person_ids)
    if report.trained:
        classifier.save()
        metrics.record_training(report.n_persons)
    else:
        logger.warning("Classifier not trained: %s", report.reason)
    return report
