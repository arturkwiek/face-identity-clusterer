"""face_clusterer — unsupervised face identity clustering pipeline.

Pipeline: detection (RetinaFace via InsightFace) -> alignment + ArcFace embedding
-> L2 normalization -> quality filtering -> (tracking for video) -> HDBSCAN
clustering -> person prototypes with re-identification.
"""

__version__ = "0.1.0"

from .config import AppConfig, load_config  # noqa: F401
