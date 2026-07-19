"""REST API over the pipeline (FastAPI).

Two ways in, deliberately:

* `/identify` takes a raw embedding. This keeps the API usable without
  InsightFace installed and matches the microservice split in the design notes,
  where an embedding-service posts vectors to an identity-service.
* `/detect` takes an image and runs the full detect -> embed -> identify path.
  It needs InsightFace and answers 503 when the model stack is unavailable,
  rather than failing at import time.

State (prototypes, classifier) lives in the app and is mutated under a lock:
FastAPI runs sync handlers in a threadpool, so concurrent `/identify?commit=true`
calls would otherwise interleave EMA updates on the same prototype. SQLite is
opened per request instead, because connections are not shareable across
threads.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import metrics
from .classifier import IdentityResolver, OnlineClassifier
from .config import AppConfig
from .embedder import FaceEmbedder, l2_normalize
from .prototypes import PrototypeStore
from .quality import filter_samples
from .store import EmbeddingStore

logger = logging.getLogger(__name__)


# ---------------- request / response models ----------------

class IdentifyRequest(BaseModel):
    embedding: list[float] = Field(..., description="Face embedding (ArcFace 512D)")
    commit: bool = Field(
        False,
        description="Apply the match back to the prototype (online EMA update)",
    )


class IdentifyResponse(BaseModel):
    person_id: str | None
    label: str | None = None
    score: float
    source: str                      # classifier | prototype | unknown
    committed: bool = False


class PersonResponse(BaseModel):
    person_id: str
    label: str | None
    embeddings_absorbed: int
    faces_in_store: int
    created_at: float
    updated_at: float


class LabelRequest(BaseModel):
    label: str | None = Field(..., description="Human-readable name, or null to clear")


class UpdatePrototypeRequest(BaseModel):
    person_id: str
    embedding: list[float]


class DetectedFace(BaseModel):
    bbox: list[float]
    det_score: float
    quality: float
    identity: IdentifyResponse


class DetectResponse(BaseModel):
    faces: list[DetectedFace]
    rejected: int


# ---------------- application state ----------------

class ApiState:
    """Shared, mutable identity state guarded by a single lock."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.prototypes = PrototypeStore(cfg.prototypes)
        self.prototypes.load()

        self.classifier: OnlineClassifier | None = None
        if cfg.classifier.enabled:
            self.classifier = OnlineClassifier(cfg.classifier)
            self.classifier.load()
        self.resolver = IdentityResolver(self.prototypes, self.classifier)
        self.embedder = FaceEmbedder(cfg.detector)   # model loads lazily
        metrics.set_persons_known(len(self.prototypes.prototypes))

    def store(self) -> EmbeddingStore:
        return EmbeddingStore(self.cfg.output.db_path)

    def vector(self, values: list[float]) -> np.ndarray:
        """Validate and normalize an incoming embedding."""
        expected = self.cfg.api.embedding_dim
        if len(values) != expected:
            raise HTTPException(
                status_code=422,
                detail=f"embedding must have {expected} values, got {len(values)}",
            )
        vec = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(vec)):
            raise HTTPException(
                status_code=422, detail="embedding contains NaN or infinite values"
            )
        if float(np.linalg.norm(vec)) == 0.0:
            raise HTTPException(status_code=422, detail="embedding is a zero vector")
        # Clients may send un-normalized vectors; the cascade requires L2 norm.
        return l2_normalize(vec)


def get_state(request: Request) -> ApiState:
    return request.app.state.fic


# ---------------- application ----------------

def create_app(cfg: AppConfig | None = None) -> FastAPI:
    cfg = cfg or AppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.fic = ApiState(cfg)
        logger.info(
            "API ready: %d persons known, classifier=%s",
            len(app.state.fic.prototypes.prototypes),
            "ready" if (app.state.fic.classifier
                        and app.state.fic.classifier.is_ready) else "off",
        )
        yield

    app = FastAPI(
        title="face-identity-clusterer",
        version="0.2.0",
        description="Face identity clustering and re-identification.",
        lifespan=lifespan,
    )

    # ---------- health & metrics ----------

    @app.get("/health")
    def health(state: ApiState = Depends(get_state)) -> dict:
        classifier_ready = bool(state.classifier and state.classifier.is_ready)
        return {
            "status": "ok",
            "persons_known": len(state.prototypes.prototypes),
            "classifier_enabled": state.cfg.classifier.enabled,
            "classifier_ready": classifier_ready,
            "classifier_persons": (
                len(state.classifier.classes) if classifier_ready else 0
            ),
        }

    @app.get("/metrics")
    def prometheus_metrics(state: ApiState = Depends(get_state)):
        if not state.cfg.api.metrics_enabled:
            raise HTTPException(status_code=404, detail="metrics are disabled")
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)

    # ---------- identification ----------

    @app.post("/identify", response_model=IdentifyResponse)
    def identify(
        req: IdentifyRequest, state: ApiState = Depends(get_state)
    ) -> IdentifyResponse:
        """Resolve an embedding to a person via classifier -> prototypes."""
        vec = state.vector(req.embedding)
        started = time.perf_counter()

        with state.lock:
            ident = state.resolver.resolve(vec)
            committed = False
            if req.commit and ident.person_id is not None:
                state.resolver.commit(ident, vec)
                state.prototypes.save()
                committed = True
                metrics.record_prototype_update()

            label = None
            if ident.person_id is not None:
                label = state.prototypes.prototypes[ident.person_id].label

        metrics.record_identification(ident.source, time.perf_counter() - started)
        return IdentifyResponse(
            person_id=ident.person_id,
            label=label,
            score=ident.score,
            source=ident.source,
            committed=committed,
        )

    @app.post("/detect", response_model=DetectResponse)
    def detect(
        file: UploadFile = File(...), state: ApiState = Depends(get_state)
    ) -> DetectResponse:
        """Detect every face in an uploaded image and identify each one."""
        import cv2

        raw = np.frombuffer(file.file.read(), dtype=np.uint8)
        if raw.size == 0:
            raise HTTPException(status_code=422, detail="empty upload")
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=422, detail="cannot decode image")

        with metrics.time_frame():
            try:
                samples = state.embedder.extract(
                    frame, source=file.filename or "upload", keep_crops=False
                )
            except ImportError as exc:
                # The detector stack is optional; say so instead of a 500.
                raise HTTPException(
                    status_code=503,
                    detail=f"face detection unavailable: {exc}",
                ) from exc
            accepted, rejected = filter_samples(samples, state.cfg.quality)

        metrics.record_faces(detected=len(accepted), rejected=len(rejected))

        faces = []
        for s in accepted:
            metrics.record_quality(s.quality)
            started = time.perf_counter()
            with state.lock:
                ident = state.resolver.resolve(s.embedding)
                label = (
                    state.prototypes.prototypes[ident.person_id].label
                    if ident.person_id is not None else None
                )
            metrics.record_identification(
                ident.source, time.perf_counter() - started
            )
            faces.append(
                DetectedFace(
                    bbox=[float(v) for v in s.bbox],
                    det_score=float(s.det_score),
                    quality=float(s.quality),
                    identity=IdentifyResponse(
                        person_id=ident.person_id, label=label,
                        score=ident.score, source=ident.source,
                    ),
                )
            )
        return DetectResponse(faces=faces, rejected=len(rejected))

    # ---------- persons ----------

    def _person_payload(state: ApiState, person_id: str, faces: int) -> PersonResponse:
        p = state.prototypes.prototypes[person_id]
        return PersonResponse(
            person_id=p.person_id, label=p.label, embeddings_absorbed=p.count,
            faces_in_store=faces, created_at=p.created_at, updated_at=p.updated_at,
        )

    @app.get("/persons", response_model=list[PersonResponse])
    def list_persons(state: ApiState = Depends(get_state)) -> list[PersonResponse]:
        store = state.store()
        try:
            counts = store.summary()["persons"]
        finally:
            store.close()
        with state.lock:
            return [
                _person_payload(state, pid, int(counts.get(pid, 0)))
                for pid in sorted(state.prototypes.prototypes)
            ]

    @app.get("/persons/{person_id}", response_model=PersonResponse)
    def get_person(
        person_id: str, state: ApiState = Depends(get_state)
    ) -> PersonResponse:
        with state.lock:
            if person_id not in state.prototypes.prototypes:
                raise HTTPException(status_code=404, detail=f"unknown {person_id}")
        store = state.store()
        try:
            faces = store.person_faces(person_id)
        finally:
            store.close()
        with state.lock:
            return _person_payload(state, person_id, faces)

    @app.patch("/persons/{person_id}", response_model=PersonResponse)
    def label_person(
        person_id: str, req: LabelRequest, state: ApiState = Depends(get_state)
    ) -> PersonResponse:
        """Attach a human-readable name to an auto-discovered identity."""
        with state.lock:
            if person_id not in state.prototypes.prototypes:
                raise HTTPException(status_code=404, detail=f"unknown {person_id}")
            state.prototypes.prototypes[person_id].label = req.label
            state.prototypes.save()
        store = state.store()
        try:
            faces = store.person_faces(person_id)
        finally:
            store.close()
        with state.lock:
            return _person_payload(state, person_id, faces)

    @app.delete("/persons/{person_id}")
    def delete_person(person_id: str, state: ApiState = Depends(get_state)) -> dict:
        """Forget an identity. Stored faces keep their rows for auditability."""
        with state.lock:
            if person_id not in state.prototypes.prototypes:
                raise HTTPException(status_code=404, detail=f"unknown {person_id}")
            del state.prototypes.prototypes[person_id]
            state.prototypes.save()
            metrics.set_persons_known(len(state.prototypes.prototypes))
        return {"deleted": person_id}

    @app.post("/update-prototype", response_model=PersonResponse)
    def update_prototype(
        req: UpdatePrototypeRequest, state: ApiState = Depends(get_state)
    ) -> PersonResponse:
        """Explicit online update of one prototype (EMA)."""
        vec = state.vector(req.embedding)
        with state.lock:
            if req.person_id not in state.prototypes.prototypes:
                raise HTTPException(
                    status_code=404, detail=f"unknown {req.person_id}"
                )
            state.prototypes.update(req.person_id, vec)
            state.prototypes.save()
            metrics.record_prototype_update()
        store = state.store()
        try:
            faces = store.person_faces(req.person_id)
        finally:
            store.close()
        with state.lock:
            return _person_payload(state, req.person_id, faces)

    # ---------- clusters & training ----------

    @app.get("/clusters")
    def list_clusters(state: ApiState = Depends(get_state)) -> dict:
        store = state.store()
        try:
            clusters = store.cluster_summary()
            summary = store.summary()
        finally:
            store.close()
        metrics.set_clusters(len(clusters))
        return {
            "clusters": clusters,
            "total_faces": summary["total_faces"],
            "persons": len(summary["persons"]),
        }

    @app.post("/retrain-classifier")
    def retrain_classifier(state: ApiState = Depends(get_state)) -> dict:
        """Refit the online classifier and hot-swap it into the live cascade."""
        store = state.store()
        try:
            embeddings, person_ids = store.labeled_embeddings()
        finally:
            store.close()

        classifier = OnlineClassifier(state.cfg.classifier)
        report = classifier.fit(embeddings, person_ids)
        if report.trained:
            classifier.save()
            with state.lock:
                state.classifier = classifier
                state.resolver.classifier = classifier
            metrics.record_training(report.n_persons)
        return {
            "trained": report.trained,
            "n_samples": report.n_samples,
            "n_persons": report.n_persons,
            "classes": report.classes,
            "reason": report.reason,
        }

    return app


def serve(cfg: AppConfig | None = None):  # pragma: no cover - process entrypoint
    import uvicorn

    cfg = cfg or AppConfig()
    uvicorn.run(create_app(cfg), host=cfg.api.host, port=cfg.api.port)
