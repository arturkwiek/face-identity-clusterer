"""Configuration: dataclasses + YAML loader with sane defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectorConfig:
    model_pack: str = "buffalo_l"        # InsightFace model pack (RetinaFace + ArcFace 512D)
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    det_size: int = 640                  # detector input size (square)
    det_score_threshold: float = 0.5     # minimum detection confidence


@dataclass
class QualityConfig:
    enabled: bool = True
    min_face_size: int = 60              # px, min(width, height) of the face bbox
    min_det_score: float = 0.6           # stricter than detector threshold
    min_blur_variance: float = 45.0      # variance of Laplacian on the gray crop
    max_yaw_asymmetry: float = 2.5       # eye-to-nose distance ratio (profile rejection)


@dataclass
class TrackerConfig:
    enabled: bool = True                 # only used for video / live sources
    iou_threshold: float = 0.3
    max_age: int = 15                    # frames a track survives without a match
    min_hits: int = 3                    # frames before a track is considered confirmed
    max_embeddings_per_track: int = 30


@dataclass
class ClusteringConfig:
    algorithm: str = "hdbscan"           # hdbscan | dbscan | agglomerative
    min_cluster_size: int = 4            # HDBSCAN: min faces to form an identity
    min_samples: int = 2
    dbscan_eps: float = 0.45             # cosine distance (DBSCAN fallback)
    agglomerative_threshold: float = 0.55
    reattach_noise: bool = True          # attach HDBSCAN noise to nearest centroid
    reattach_threshold: float = 0.45     # max cosine distance for reattachment
    purge_outliers: bool = True          # demote members far from their centroid
    purge_threshold: float = 0.45        # max cosine distance member<->centroid


@dataclass
class PrototypeConfig:
    enabled: bool = True
    match_threshold: float = 0.60        # min cosine similarity to re-identify a person
    ema_alpha: float = 0.10              # prototype update rate
    path: str = "output/prototypes"      # persisted store (json + npz)


@dataclass
class ClassifierConfig:
    enabled: bool = False                # opt-in: needs labeled data to be useful
    model: str = "logreg"                # logreg | mlp
    confidence_threshold: float = 0.75   # below this the classifier abstains
    min_samples_per_person: int = 5      # classes with fewer examples are dropped
    min_persons: int = 2                 # a classifier needs at least two classes
    regularization_c: float = 10.0       # logreg only
    mlp_hidden: int = 128                # mlp only
    max_iter: int = 1000
    path: str = "output/classifier"      # persisted model (joblib)


@dataclass
class VideoConfig:
    frame_stride: int = 3                # process every Nth frame
    display: bool = False                # live preview window (needs GUI)


@dataclass
class AcquisitionConfig:
    """Dataset capture — stage 1, deliberately independent of the model stack."""

    dir: str = "dataset"                 # sessions are created as subfolders
    interval_seconds: float = 1.0        # min wall/stream time between saved frames
    max_frames: int = 0                  # 0 = unlimited
    max_duration_seconds: float = 0.0    # 0 = until the source ends / ESC
    # Camera-failure guard only — a lens cap, a black frame or a grossly
    # defocused camera, which all score near zero. NOT a quality filter, and
    # not comparable to QualityConfig.min_blur_variance, which is measured on a
    # face crop. Whole-frame variance turned out to depend on two things that
    # have nothing to do with face sharpness: background texture (the same face
    # scored 133 against foliage and 54 against a plain wall) and resolution
    # (that same indoor scene scored 54 at 640x480 and 22 at 1280x720). It is
    # therefore useless as an absolute quality threshold — judging real
    # sharpness is the quality filter's job, on the crop, during analysis.
    min_blur_variance: float = 8.0       # 0 disables
    capture_width: int = 0               # 0 = camera default; e.g. 1280
    capture_height: int = 0              # 0 = camera default; e.g. 720
    min_change_fraction: float = 0.02    # min share of the frame that must differ
                                         # from the last kept frame (0 disables)
    face_gate: bool = False              # keep only frames with a detected face
    jpeg_quality: int = 95
    display: bool = False                # preview window (needs a GUI)


@dataclass
class ScheduleConfig:
    """Day/night split: capture during the day, analyze at night."""

    capture_start: str = "07:00"         # local time, HH:MM
    capture_end: str = "15:00"           # analysis runs from here until start
    camera: str = "0"                    # USB camera index, or an RTSP/HTTP URL
    session_minutes: int = 60            # one capture session per N minutes
    analysis_enabled: bool = True        # run the nightly analysis job
    train_after_analysis: bool = True    # refit the classifier once clustered
    retry_seconds: float = 60.0          # wait before retrying a failed camera


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"              # bind loopback by default, not 0.0.0.0
    port: int = 8000
    embedding_dim: int = 512             # ArcFace output size; requests must match
    metrics_enabled: bool = True         # expose /metrics


@dataclass
class OutputConfig:
    dir: str = "output"
    save_crops: bool = True
    max_crops_per_cluster: int = 40
    db_path: str = "output/faces.sqlite"


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_into(dc: Any, data: dict[str, Any]) -> Any:
    """Recursively overlay a dict onto a dataclass instance."""
    for f in fields(dc):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(dc, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into(current, value)
        else:
            setattr(dc, f.name, value)
    return dc


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config; missing file or keys fall back to defaults."""
    cfg = AppConfig()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return _merge_into(cfg, data)
