import json

import numpy as np
import pytest

from face_clusterer.acquire import (
    DatasetAcquisition, _change_fraction, _signature, dataset_stats,
)
from face_clusterer.config import AcquisitionConfig, AppConfig

FPS = 10.0
SIZE = (240, 320)          # h, w


def _textured(rng, h=SIZE[0], w=SIZE[1]):
    """A sharp, high-variance frame (survives the blur gate)."""
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _flat(value=128, h=SIZE[0], w=SIZE[1]):
    """A perfectly smooth frame — zero Laplacian variance, i.e. 'blurry'."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _write_video(path, frames, fps=FPS):
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    if not writer.isOpened():
        pytest.skip("no MJPG video writer available in this OpenCV build")
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def _config(tmp_path, **overrides):
    settings = dict(dir=str(tmp_path / "dataset"), interval_seconds=0.0,
                    min_blur_variance=0.0, min_change_fraction=0.0)
    settings.update(overrides)
    return AppConfig(acquisition=AcquisitionConfig(**settings))


def test_captures_every_frame_with_filters_off(tmp_path):
    rng = np.random.default_rng(0)
    video = _write_video(tmp_path / "v.avi", [_textured(rng) for _ in range(6)])

    report = DatasetAcquisition(_config(tmp_path)).run(str(video), session="s1")

    assert report.frames_seen == 6 and report.frames_saved == 6
    saved = sorted((tmp_path / "dataset" / "s1" / "frames").glob("*.jpg"))
    assert [p.name for p in saved] == [f"{i:06d}.jpg" for i in range(6)]


def test_interval_subsamples_by_stream_time(tmp_path):
    """At 10 FPS a 1-second interval keeps roughly every tenth frame."""
    rng = np.random.default_rng(1)
    video = _write_video(tmp_path / "v.avi", [_textured(rng) for _ in range(30)])

    report = DatasetAcquisition(
        _config(tmp_path, interval_seconds=1.0)
    ).run(str(video), session="s1")

    assert report.frames_seen == 30
    assert report.frames_saved == 3
    assert report.skipped_interval == 27


def test_blur_gate_rejects_flat_frames(tmp_path):
    rng = np.random.default_rng(2)
    frames = [_textured(rng), _flat(), _flat(), _textured(rng)]
    video = _write_video(tmp_path / "v.avi", frames)

    report = DatasetAcquisition(
        _config(tmp_path, min_blur_variance=45.0)
    ).run(str(video), session="s1")

    assert report.frames_saved == 2 and report.skipped_blurry == 2


def _write_png_sequence(dir_path, frames):
    """Lossless capture source.

    OpenCV's MJPG writer only settles on a stable quantization after a few
    frames, so identical inputs decode to slightly different pixels early in a
    video — useless for asserting on exact-duplicate detection. A PNG sequence
    is lossless and still exercises the real VideoCapture path.
    """
    import cv2

    dir_path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(dir_path / f"{i:06d}.png"), frame)
    return dir_path / "%06d.png"


def test_change_fraction_separates_identical_from_distinct():
    rng = np.random.default_rng(30)
    a, b = _textured(rng), _textured(rng)
    assert _change_fraction(_signature(a), _signature(a)) == 0.0
    # a different scene changes far more than the 2% default threshold
    assert _change_fraction(_signature(a), _signature(b)) > 0.02


def test_change_fraction_detects_a_small_moving_subject():
    """The reason this metric replaced mean-absolute-difference.

    A subject occupying a few percent of a wide frame barely moves the mean, so
    a threshold that catches it sits within noise of one that catches nothing.
    Measured as a share of changed pixels, the same motion is unambiguous.
    """
    import cv2

    background = np.full((360, 640, 3), 90, dtype=np.uint8)
    cv2.randn(background, 90, 20)                    # mild texture, like a room
    before = background.copy()
    after = background.copy()
    cv2.rectangle(after, (100, 120), (190, 260), (30, 200, 30), -1)

    changed = _change_fraction(_signature(before), _signature(after))
    assert changed > 0.02                            # detected as real change


def test_duplicate_rejection_skips_identical_frames(tmp_path):
    rng = np.random.default_rng(3)
    still, other = _textured(rng), _textured(rng)
    source = _write_png_sequence(
        tmp_path / "seq", [still, still, still, other]
    )

    report = DatasetAcquisition(
        _config(tmp_path, min_change_fraction=0.02)
    ).run(str(source), session="s1")

    assert report.frames_saved == 2
    assert report.skipped_unchanged == 2


def test_max_frames_stops_capture(tmp_path):
    rng = np.random.default_rng(4)
    video = _write_video(tmp_path / "v.avi", [_textured(rng) for _ in range(20)])

    report = DatasetAcquisition(
        _config(tmp_path, max_frames=5)
    ).run(str(video), session="s1")

    assert report.frames_saved == 5
    assert report.frames_seen < 20          # stopped early, did not read it all


def test_max_duration_stops_capture(tmp_path):
    rng = np.random.default_rng(5)
    video = _write_video(tmp_path / "v.avi", [_textured(rng) for _ in range(30)])

    report = DatasetAcquisition(
        _config(tmp_path, max_duration_seconds=1.0)
    ).run(str(video), session="s1")

    # 10 FPS => 1 second of stream time is ~10 frames
    assert report.frames_seen <= 11 and report.frames_saved <= 10


def test_manifest_records_session_metadata(tmp_path):
    rng = np.random.default_rng(6)
    video = _write_video(tmp_path / "v.avi", [_textured(rng) for _ in range(4)])

    DatasetAcquisition(_config(tmp_path)).run(
        str(video), session="enroll", label="Anna"
    )

    manifest = json.loads(
        (tmp_path / "dataset" / "enroll" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["label"] == "Anna"
    assert manifest["session"] == "enroll"
    assert manifest["frames_saved"] == 4
    assert len(manifest["frames"]) == 4
    assert manifest["frames"][0]["file"] == "frames/000000.jpg"
    assert manifest["settings"]["interval_seconds"] == 0.0


def test_unopenable_source_raises(tmp_path):
    with pytest.raises(IOError):
        DatasetAcquisition(_config(tmp_path)).run(str(tmp_path / "missing.mp4"))


def test_dataset_stats_aggregates_sessions(tmp_path):
    rng = np.random.default_rng(7)
    video = _write_video(tmp_path / "v.avi", [_textured(rng) for _ in range(3)])
    cfg = _config(tmp_path)
    DatasetAcquisition(cfg).run(str(video), session="a", label="Anna")
    DatasetAcquisition(cfg).run(str(video), session="b", label="Bartek")

    stats = dataset_stats(tmp_path / "dataset")
    assert stats["total_sessions"] == 2
    assert stats["total_frames_in_manifests"] == 6
    assert stats["images_on_disk"] == 6
    assert {s["label"] for s in stats["sessions"]} == {"Anna", "Bartek"}


def test_dataset_stats_survives_broken_manifest(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "broken").mkdir(parents=True)
    (dataset / "broken" / "manifest.json").write_text("{not json", encoding="utf-8")

    stats = dataset_stats(dataset)
    assert stats["total_sessions"] == 0      # skipped, not crashed


def test_dataset_stats_on_empty_folder(tmp_path):
    stats = dataset_stats(tmp_path / "nothing-here")
    assert stats["total_sessions"] == 0 and stats["images_on_disk"] == 0
