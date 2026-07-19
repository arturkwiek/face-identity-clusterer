#!/usr/bin/env python3
"""face-identity-clusterer CLI.

Examples:
    python main.py images ./photos                # cluster a folder of photos
    python main.py video ./recording.mp4          # process a video file
    python main.py video ./recording.mp4 --display
    python main.py live --camera 0                # live webcam re-identification
    python main.py recluster                      # re-cluster stored embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from face_clusterer.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-identity-clusterer",
        description="Unsupervised face identity clustering "
                    "(detect -> embed -> cluster -> re-identify).",
    )
    parser.add_argument("-c", "--config", default=None,
                        help="Path to config.yaml (defaults are used if omitted)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser(
        "capture", help="STAGE 1: collect frames into a dataset (no models needed)"
    )
    p_cap.add_argument("--camera", default=None,
                       help="Camera index (0) or RTSP/HTTP URL")
    p_cap.add_argument("--video", default=None,
                       help="Capture from a video file instead of a camera")
    p_cap.add_argument("--out", default=None,
                       help="Dataset folder (default: dataset/)")
    p_cap.add_argument("--session", default=None,
                       help="Session name (default: timestamp)")
    p_cap.add_argument("--label", default=None,
                       help="What this session contains, e.g. a person's name")
    p_cap.add_argument("--interval", type=float, default=None,
                       help="Seconds between saved frames (default 1.0)")
    p_cap.add_argument("--max-frames", type=int, default=None,
                       help="Stop after N saved frames")
    p_cap.add_argument("--max-seconds", type=float, default=None,
                       help="Stop after N seconds of stream time")
    p_cap.add_argument("--min-sharpness", type=float, default=None,
                       help="Reject frames below this Laplacian variance (0 = off)")
    p_cap.add_argument("--face-gate", action="store_true",
                       help="Keep only frames containing a face (needs InsightFace)")
    p_cap.add_argument("--display", action="store_true",
                       help="Show a preview window (ESC stops capture)")

    p_run = sub.add_parser(
        "schedule",
        help="Run the day/night supervisor: capture 07-15, analyze 15-07",
    )
    p_run.add_argument("--camera", default=None,
                       help="Camera index (0) or RTSP/HTTP URL")
    p_run.add_argument("--capture-start", default=None, help="HH:MM")
    p_run.add_argument("--capture-end", default=None, help="HH:MM")
    p_run.add_argument("--status", action="store_true",
                       help="Print what the scheduler would do now, then exit")
    p_run.add_argument("--analyze-now", action="store_true",
                       help="Run the nightly analysis job once, then exit")

    p_ds = sub.add_parser("dataset", help="Summarize collected capture sessions")
    p_ds.add_argument("path", nargs="?", default=None,
                      help="Dataset folder (default: dataset/)")

    p_img = sub.add_parser("images", help="Cluster faces from a folder of photos")
    p_img.add_argument("path", help="Folder with images (searched recursively)")
    p_img.add_argument("--reset", action="store_true",
                       help="Replace stored faces instead of appending "
                            "(use when re-analyzing the same folder)")

    p_vid = sub.add_parser("video", help="Process video files with tracking")
    p_vid.add_argument("path",
                       help="Video file (mp4/avi/mkv/mov/webm) or a folder of them")
    p_vid.add_argument("--display", action="store_true",
                       help="Show a preview window while processing")
    p_vid.add_argument("--stride", type=int, default=None,
                       help="Process every Nth frame (default 3)")
    p_vid.add_argument("--reset", action="store_true",
                       help="Replace stored faces instead of appending")
    p_vid.add_argument("--min-cluster-size", type=int, default=None,
                       help="Min TRACKS forming an identity (default 4). Video "
                            "clusters one sample per track, so a person seen "
                            "once needs a low value to be enrolled.")

    p_live = sub.add_parser("live", help="Live camera with re-identification")
    p_live.add_argument("--camera", default="0",
                        help="Camera index (0) or RTSP/HTTP URL")
    p_live.add_argument("--recluster-every", type=int, default=300,
                        help="Cluster unknown faces every N processed frames")
    p_live.add_argument("--headless", action="store_true",
                        help="No preview window (services/autostart); Ctrl+C stops")

    sub.add_parser("recluster",
                   help="Re-cluster embeddings already stored in the database")

    sub.add_parser("train",
                   help="Train the online classifier on stored person-labeled faces")

    p_serve = sub.add_parser("serve", help="Run the REST API (FastAPI + uvicorn)")
    p_serve.add_argument("--host", default=None,
                         help="Bind address (default 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=None,
                         help="Bind port (default 8000)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    default_cfg = Path(__file__).parent / "config.yaml"
    cfg_path = args.config or (default_cfg if default_cfg.exists() else None)
    cfg = load_config(cfg_path)

    if args.command == "capture":
        summary = _capture(cfg, args)
    elif args.command == "schedule":
        summary = _schedule(cfg, args)
    elif args.command == "dataset":
        from face_clusterer.acquire import dataset_stats
        summary = dataset_stats(args.path or cfg.acquisition.dir)
    elif args.command == "images":
        from face_clusterer.pipeline import ImagesPipeline
        summary = ImagesPipeline(cfg).run(args.path, reset_store=args.reset)
    elif args.command == "video":
        from face_clusterer.pipeline import VideoPipeline
        if args.display:
            cfg.video.display = True
        if args.stride:
            cfg.video.frame_stride = args.stride
        if args.min_cluster_size:
            cfg.clustering.min_cluster_size = args.min_cluster_size
        summary = VideoPipeline(cfg).run(args.path, reset_store=args.reset)
    elif args.command == "live":
        from face_clusterer.pipeline import LivePipeline
        camera = int(args.camera) if str(args.camera).isdigit() else args.camera
        cfg.video.display = not args.headless
        summary = LivePipeline(cfg).run(
            camera, recluster_every=args.recluster_every
        )
    elif args.command == "recluster":
        from face_clusterer.pipeline import recluster_store
        summary = recluster_store(cfg)
    elif args.command == "train":
        from dataclasses import asdict

        from face_clusterer.pipeline import train_classifier
        summary = asdict(train_classifier(cfg))
    elif args.command == "serve":
        from face_clusterer.api import serve

        if args.host:
            cfg.api.host = args.host
        if args.port:
            cfg.api.port = args.port
        serve(cfg)          # blocks until the server stops
        return 0
    else:  # pragma: no cover
        raise SystemExit(2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _capture(cfg, args) -> dict:
    """Stage 1: collect frames. Runs without the detection/embedding stack."""
    from face_clusterer.acquire import DatasetAcquisition

    if args.video and args.camera:
        raise SystemExit("Use either --video or --camera, not both.")

    acq = cfg.acquisition
    for attr, value in (
        ("dir", args.out),
        ("interval_seconds", args.interval),
        ("max_frames", args.max_frames),
        ("max_duration_seconds", args.max_seconds),
        ("min_blur_variance", args.min_sharpness),
    ):
        if value is not None:
            setattr(acq, attr, value)
    if args.face_gate:
        acq.face_gate = True
    if args.display:
        acq.display = True

    if args.video:
        source: int | str = args.video
    else:
        camera = args.camera if args.camera is not None else "0"
        source = int(camera) if str(camera).isdigit() else camera

    report = DatasetAcquisition(cfg).run(
        source, session=args.session, label=args.label
    )
    return report.as_summary()


def _schedule(cfg, args) -> dict:
    """Day/night supervisor (stage 1 and stage 2 separated in time)."""
    from face_clusterer.scheduler import DayNightScheduler, schedule_status

    if args.camera is not None:
        cfg.schedule.camera = args.camera
    if args.capture_start:
        cfg.schedule.capture_start = args.capture_start
    if args.capture_end:
        cfg.schedule.capture_end = args.capture_end

    if args.status:
        return schedule_status(cfg)
    if args.analyze_now:
        return DayNightScheduler(cfg).analyze()
    return DayNightScheduler(cfg).run()      # blocks until interrupted


if __name__ == "__main__":
    raise SystemExit(main())
