"""Lightweight online classifier over embeddings — the second "learning" layer.

The embedding model stays frozen (as in prototypes.py); this layer adds a small
supervised head (logistic regression / MLP) trained on embeddings already
assigned to persons. It is a *refit-from-scratch* design rather than
`partial_fit`: with a few thousand 512D vectors a logistic regression fits in
well under a second, and refitting sidesteps the main drawback of incremental
sklearn estimators — `partial_fit` requires the full class list up front, which
is exactly what an open-set system does not have (new persons appear at any
time).

Identification cascade (as specified): classifier -> prototypes -> clustering.
The classifier is a fast, discriminative first stage; prototypes remain the
authority for anyone the classifier has not seen enough of; unmatched faces
fall through to HDBSCAN, which is what creates new identities in the first
place. The classifier never invents a person — below `confidence_threshold`,
or before it has enough data to be meaningful, it abstains and the cascade
moves on.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ClassifierConfig
from .embedder import l2_normalize
from .prototypes import PrototypeStore

logger = logging.getLogger(__name__)


@dataclass
class Identification:
    """Outcome of one identification attempt."""

    person_id: str | None
    score: float
    source: str                      # "classifier" | "prototype" | "unknown"


@dataclass
class TrainingReport:
    trained: bool
    n_samples: int = 0
    n_persons: int = 0
    classes: list[str] = field(default_factory=list)
    reason: str = ""                 # why training was skipped, if it was


class OnlineClassifier:
    """Supervised head over frozen embeddings, refit on demand."""

    def __init__(self, cfg: ClassifierConfig | None = None):
        self.cfg = cfg or ClassifierConfig()
        self._model = None
        self.classes: list[str] = []
        self.trained_at: float = 0.0
        self.n_samples: int = 0

    @property
    def is_ready(self) -> bool:
        return self._model is not None and len(self.classes) >= 2

    # ---------------- training ----------------

    def _build_model(self):
        if self.cfg.model == "mlp":
            from sklearn.neural_network import MLPClassifier

            return MLPClassifier(
                hidden_layer_sizes=(self.cfg.mlp_hidden,),
                max_iter=self.cfg.max_iter,
                random_state=0,
            )
        if self.cfg.model == "logreg":
            from sklearn.linear_model import LogisticRegression

            # Embeddings are L2-normalized, so features are already comparable
            # in scale; C is the only knob that matters much here.
            return LogisticRegression(
                C=self.cfg.regularization_c,
                max_iter=self.cfg.max_iter,
            )
        raise ValueError(f"Unknown classifier model: {self.cfg.model}")

    def fit(self, embeddings: np.ndarray, person_ids: list[str]) -> TrainingReport:
        """Refit on labeled embeddings; persons with too few samples are dropped.

        Returns a report instead of raising: too little data is the normal state
        of a fresh system, not an error — the cascade simply falls back to
        prototypes until enough labeled faces accumulate.
        """
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or len(x) != len(person_ids) or len(x) == 0:
            return TrainingReport(trained=False, reason="no labeled samples")

        # Drop under-represented persons: a class with 1-2 examples teaches the
        # model nothing and inflates its confidence on that identity.
        counts: dict[str, int] = {}
        for pid in person_ids:
            counts[pid] = counts.get(pid, 0) + 1
        keep = {p for p, c in counts.items() if c >= self.cfg.min_samples_per_person}
        mask = np.array([pid in keep for pid in person_ids], dtype=bool)

        if len(keep) < max(2, self.cfg.min_persons):
            return TrainingReport(
                trained=False,
                n_samples=int(mask.sum()),
                n_persons=len(keep),
                reason=(
                    f"need >= {max(2, self.cfg.min_persons)} persons with "
                    f">= {self.cfg.min_samples_per_person} samples, have {len(keep)}"
                ),
            )

        x_fit = x[mask]
        y_fit = [pid for pid, m in zip(person_ids, mask) if m]

        model = self._build_model()
        model.fit(x_fit, y_fit)

        self._model = model
        self.classes = [str(c) for c in model.classes_]
        self.trained_at = time.time()
        self.n_samples = len(x_fit)
        logger.info(
            "Classifier trained: %d samples, %d persons (%s)",
            self.n_samples, len(self.classes), self.cfg.model,
        )
        return TrainingReport(
            trained=True,
            n_samples=self.n_samples,
            n_persons=len(self.classes),
            classes=list(self.classes),
        )

    # ---------------- inference ----------------

    def predict(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (person_id, confidence), or (None, confidence) when unsure."""
        if not self.is_ready:
            return None, 0.0
        vec = l2_normalize(np.asarray(embedding, dtype=np.float32)).reshape(1, -1)
        probs = self._model.predict_proba(vec)[0]
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        if confidence < self.cfg.confidence_threshold:
            return None, confidence
        return str(self._model.classes_[best]), confidence

    # ---------------- persistence ----------------

    def save(self, path: str | Path | None = None):
        import joblib

        base = Path(path or self.cfg.path)
        base.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "classes": self.classes,
                "trained_at": self.trained_at,
                "n_samples": self.n_samples,
                "config_model": self.cfg.model,
            },
            f"{base}.joblib",
        )

    def load(self, path: str | Path | None = None) -> bool:
        import joblib

        base = Path(path or self.cfg.path)
        blob_path = Path(f"{base}.joblib")
        if not blob_path.exists():
            return False
        try:
            blob = joblib.load(blob_path)
        except Exception as exc:  # pickled sklearn models are version-fragile
            logger.warning("Cannot load classifier from %s: %s", blob_path, exc)
            return False
        self._model = blob.get("model")
        self.classes = list(blob.get("classes", []))
        self.trained_at = float(blob.get("trained_at", 0.0))
        self.n_samples = int(blob.get("n_samples", 0))
        logger.info(
            "Loaded classifier: %d persons, %d samples",
            len(self.classes), self.n_samples,
        )
        return self.is_ready


class IdentityResolver:
    """The classifier -> prototypes -> unknown cascade.

    Resolution is read-only; committing a match (EMA-updating the prototype) is
    a separate, explicit step so that callers which only *ask* who someone is
    — e.g. an /identify endpoint — cannot silently mutate stored identities.
    """

    def __init__(
        self,
        prototypes: PrototypeStore,
        classifier: OnlineClassifier | None = None,
    ):
        self.prototypes = prototypes
        self.classifier = classifier

    def resolve(self, embedding: np.ndarray) -> Identification:
        if self.classifier is not None and self.classifier.is_ready:
            person_id, confidence = self.classifier.predict(embedding)
            # Trust the classifier only for persons the prototype store still
            # knows about; a stale model can name someone since removed.
            if person_id is not None and person_id in self.prototypes.prototypes:
                return Identification(person_id, confidence, "classifier")

        person_id, sim = self.prototypes.match(embedding)
        if person_id is not None:
            return Identification(person_id, sim, "prototype")

        return Identification(None, sim, "unknown")

    def commit(self, identification: Identification, embedding: np.ndarray):
        """Apply an accepted match back to the prototype store (online update)."""
        if identification.person_id is not None:
            self.prototypes.update(identification.person_id, embedding)
