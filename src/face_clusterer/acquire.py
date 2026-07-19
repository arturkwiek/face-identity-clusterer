"""Dataset acquisition — stage 1 of the workflow, before any analysis.

Collecting material and clustering it are separate jobs with different needs, so
this module is deliberately independent of the model stack: interval sampling,
sharpness gating and change detection all run on OpenCV alone. Acquisition
can therefore run on a camera box that has no InsightFace installed, and the
models are only needed later, when the collected frames are analyzed.

Three filters decide whether a frame is kept, cheapest first:

1. **interval** — one frame per `interval_seconds` of *stream* time (derived
   from the video's FPS for files, wall clock for cameras), because
   neighbouring frames of the same person add nothing but disk usage;
2. **sharpness** — variance of the Laplacian, the same measure the quality
   filter uses later, so blurry frames are dropped at the source;
3. **change detection** — the fraction of the picture that differs from the
   last *kept* frame, which is what stops an empty room from filling the disk
   overnight.

Change is measured as "how much of the frame changed", not "by how much did the
frame change on average". Averaging dilutes a person who occupies a few percent
of a wide camera view: measured on a 640x360 scene, a subject crossing the
entire frame moved the mean absolute difference only from 0.0 to 4.0, so every
useful threshold sat within noise of every other. Counting pixels that moved by
more than `_CHANGE_DELTA` separates the same cases as 0% / 0.9% / 5.6%, which is
both wider and directly interpretable as a percentage of the image.

Output is a plain folder of JPEGs plus a manifest, so a session is consumable by
`main.py images <dataset>` with no import step, and inspectable with any file
browser.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import AcquisitionConfig, AppConfig

logger = logging.getLogger(__name__)

_THUMB = (64, 64)          # size used for the change-detection signature
_CHANGE_DELTA = 12.0       # per-pixel delta (0-255) that counts as "changed"


@dataclass
class AcquisitionReport:
    session: str
    source: str
    frames_saved: int = 0
    frames_seen: int = 0
    skipped_interval: int = 0
    skipped_blurry: int = 0
    skipped_unchanged: int = 0
    skipped_no_face: int = 0
    duration_seconds: float = 0.0
    directory: str = ""
    frames: list[dict] = field(default_factory=list)

    def as_summary(self) -> dict:
        """Report without the per-frame list (that lives in the manifest)."""
        data = asdict(self)
        data.pop("frames")
        return data


def _sharpness(frame_bgr: np.ndarray) -> float:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _signature(frame_bgr: np.ndarray) -> np.ndarray:
    """Small grayscale thumbnail used to compare consecutive kept frames.

    Downscaling also suppresses sensor and compression noise, which would
    otherwise register as change on a completely static scene.
    """
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, _THUMB).astype(np.float32)


def _change_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the picture (0..1) that visibly changed between two frames."""
    return float(np.mean(np.abs(a - b) > _CHANGE_DELTA))


class DatasetAcquisition:
    """Capture frames from a camera, video file or stream into a dataset folder."""

    def __init__(self, cfg: AppConfig | None = None):
        cfg = cfg or AppConfig()
        self.app_cfg = cfg
        self.cfg: AcquisitionConfig = cfg.acquisition
        self._embedder = None          # built lazily, only when face_gate is on

    # ---------------- face gate (optional) ----------------

    def _has_face(self, frame_bgr: np.ndarray) -> bool:
        if self._embedder is None:
            from .embedder import FaceEmbedder

            self._embedder = FaceEmbedder(self.app_cfg.detector)
        return bool(self._embedder.extract(frame_bgr, keep_crops=False))

    # ---------------- capture ----------------

    def run(
        self,
        source: int | str = 0,
        session: str | None = None,
        label: str | None = None,
    ) -> AcquisitionReport:
        """Capture from `source` into a new session folder.

        `label` records who/what the session contains (e.g. a person's name for
        an enrollment session). It is metadata only — nothing downstream reads
        it as ground truth, but it makes a collected dataset auditable.
        """
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open capture source: {source}")

        # Webcams default to 640x480; a larger frame gives the detector more
        # pixels per face, which matters as soon as people are not close-up.
        if self.cfg.capture_width and self.cfg.capture_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.capture_height)
            actual = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            if actual != (self.cfg.capture_width, self.cfg.capture_height):
                # Cameras silently fall back to a supported mode.
                logger.warning(
                    "Requested %dx%d but the camera provides %dx%d.",
                    self.cfg.capture_width, self.cfg.capture_height, *actual,
                )

        session = session or time.strftime("session_%Y%m%d_%H%M%S")
        out_dir = Path(self.cfg.dir) / session
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        is_file = not isinstance(source, int) and Path(str(source)).exists()
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        report = AcquisitionReport(
            session=session, source=str(source), directory=str(out_dir)
        )
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.jpeg_quality)
        ]

        started = time.time()
        last_kept_at = -float("inf")     # stream time of the last saved frame
        last_signature: np.ndarray | None = None
        frame_idx = 0

        logger.info("Acquisition started: %s -> %s", source, out_dir)
        if self.cfg.display:
            logger.info("Press ESC in the preview window to stop.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                report.frames_seen += 1

                # Stream time: derived from FPS for files (so a fast-decoding
                # file samples like real time), wall clock for live sources.
                now = (
                    frame_idx / fps if (is_file and fps > 0)
                    else time.time() - started
                )

                if self.cfg.max_duration_seconds and now >= self.cfg.max_duration_seconds:
                    break

                keep = now - last_kept_at >= self.cfg.interval_seconds
                if not keep:
                    report.skipped_interval += 1
                else:
                    sharpness = _sharpness(frame)
                    if (
                        self.cfg.min_blur_variance > 0
                        and sharpness < self.cfg.min_blur_variance
                    ):
                        report.skipped_blurry += 1
                        keep = False

                    signature = _signature(frame) if keep else None
                    if keep and last_signature is not None:
                        changed = _change_fraction(signature, last_signature)
                        if changed < self.cfg.min_change_fraction:
                            report.skipped_unchanged += 1
                            keep = False

                    if keep and self.cfg.face_gate and not self._has_face(frame):
                        report.skipped_no_face += 1
                        keep = False

                    if keep:
                        name = f"{report.frames_saved:06d}.jpg"
                        cv2.imwrite(str(frames_dir / name), frame, encode_params)
                        report.frames.append(
                            {
                                "file": f"frames/{name}",
                                "frame_idx": frame_idx,
                                "stream_time": round(now, 3),
                                "sharpness": round(sharpness, 1),
                            }
                        )
                        report.frames_saved += 1
                        last_kept_at = now
                        last_signature = signature

                if self.cfg.display:
                    self._draw(frame, report)
                    cv2.imshow("face-identity-clusterer [capture]", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        logger.info("Stopped by user.")
                        break

                if self.cfg.max_frames and report.frames_saved >= self.cfg.max_frames:
                    break
        except KeyboardInterrupt:      # Ctrl+C must still leave a valid manifest
            logger.info("Interrupted — writing manifest for what was collected.")
        finally:
            cap.release()
            if self.cfg.display:
                cv2.destroyAllWindows()

        report.duration_seconds = round(time.time() - started, 2)
        self._write_manifest(out_dir, report, label)
        logger.info(
            "Acquisition done: %d frames saved from %d seen "
            "(interval %d, blurry %d, unchanged %d, no-face %d) -> %s",
            report.frames_saved, report.frames_seen, report.skipped_interval,
            report.skipped_blurry, report.skipped_unchanged,
            report.skipped_no_face, out_dir,
        )
        return report

    def _write_manifest(
        self, out_dir: Path, report: AcquisitionReport, label: str | None
    ):
        manifest = {
            **report.as_summary(),
            "label": label,
            "created_at": time.time(),
            "settings": asdict(self.cfg),
            "frames": report.frames,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _draw(frame, report: AcquisitionReport):
        import cv2

        cv2.putText(
            frame, f"saved: {report.frames_saved}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )


def dataset_stats(dataset_dir: str | Path) -> dict:
    """Summarize every capture session in a dataset folder."""
    dataset_dir = Path(dataset_dir)
    sessions = []
    for manifest_path in sorted(dataset_dir.glob("*/manifest.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable manifest %s: %s", manifest_path, exc)
            continue
        sessions.append(
            {
                "session": data.get("session", manifest_path.parent.name),
                "label": data.get("label"),
                "frames_saved": data.get("frames_saved", 0),
                "source": data.get("source"),
                "duration_seconds": data.get("duration_seconds", 0.0),
            }
        )

    # Count images on disk too: frames may be added or pruned by hand, and the
    # analysis stage reads the folder, not the manifest.
    images_on_disk = sum(
        1 for p in dataset_dir.rglob("*.jpg")
    ) + sum(1 for p in dataset_dir.rglob("*.png"))

    return {
        "dataset": str(dataset_dir),
        "sessions": sessions,
        "total_sessions": len(sessions),
        "total_frames_in_manifests": sum(s["frames_saved"] for s in sessions),
        "images_on_disk": images_on_disk,
    }
