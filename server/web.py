"""FastAPI web app factory, separated from the Modal layer.

Modal's :class:`modal.App` is built in ``app.py``; this module builds the
**HTTP contract** in isolation so it can be tested with a FastAPI
``TestClient`` and mocked store/embed/job layers — no Modal SDK, no GPU, no
network. ``app.py`` injects the real Modal-backed closures at runtime.

The factory is the single source of truth for the HTTP API contract documented
in ``docs/MODAL_EMBEDDING.md``. Additive changes go here and are mirrored in
``src/modal-client.ts``.

Import discipline
-----------------
``fastapi`` / ``pydantic`` are imported **inside** :func:`build_app` (not at
module top level). The Modal CLI imports this module locally to serialize the
App — on a laptop without the ML/web stack, the top-level import must stay
light so ``modal deploy`` works. ``MemoryJobStore`` (stdlib only) is safe at
module level.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def build_app(
    *,
    cfg,
    store,
    embed_fn: Callable[..., list[list[float]]],
    get_spec_fn: Callable[[str], Any],
    registry: dict,
    read_job: Callable[[str], dict | None],
    write_job: Callable[[str, dict], None],
    list_job_docs: Callable[[int], list[dict]],
    spawn_job: Callable[[str, str, str, int | None, dict], object],
    create_graph_job: Callable[[str, str, dict | None], dict] | None = None,
    append_graph_batch: Callable[[str, list[dict], list[dict]], dict] | None = None,
    stage_graph_artifact: Callable[[str, str, str, str], dict] | None = None,
    start_graph_job: Callable[[str], dict] | None = None,
    read_graph_job: Callable[[str], dict | None] | None = None,
    list_graph_jobs: Callable[[int], list[dict]] | None = None,
    loaded_models: Callable[[], list[str]] | None = None,
    reload_fn: Callable[[], None] | None = None,
) -> Any:
    """Construct the ASGI app with all dependencies injected.

    Returns a ``fastapi.FastAPI`` instance. Heavy web imports are deferred to
    here so importing this module (e.g. during ``modal deploy`` serialization)
    never requires ``fastapi`` / ``pydantic``.

    Parameters
    ----------
    cfg : module
        ``config`` — provides DEFAULT_MODEL, DEFAULT_DIM, EXPORT_LIMIT_MAX, etc.
        plus ``API_TOKEN`` (bound in by ``app.py`` from the Modal Secret).
    store : module
        ``store`` — list_collections, export_since, export_since_arrow, stats.
    embed_fn : callable
        ``embedders.embed`` (or a mock returning canned vectors).
    get_spec_fn : callable
        ``embedders.get_spec`` (or a mock).
    registry : dict
        ``enabled_registry()`` — used to list models on ``GET /models``.
    read_job/write_job/list_job_docs : callable
        Job-persistence closures backed by the Volume in production.
    spawn_job : callable
        Spawns the bulk worker; returns an object with ``.object_id``.
    loaded_models : callable, optional
        Names of loaded embedder models for ``GET /stats``.
    reload_fn : callable, optional
        Invoked before store reads (``/stats``, ``/sync/collections``,
        ``/sync/export``) so the web container sees writes committed by
        the separate bulk-worker container. In production this is
        ``vectors_volume.reload``; None in tests (in-memory store).
    """
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import Response as RawResponse

    from schemas import EmbedRequest, GraphArtifactUploadRequest, GraphChunkRequest, GraphImportRequest, GraphJobBatchRequest, GraphJobRequest, JobRequest, ReembedSource, V1EmbeddingRequest

    # Route handlers are defined as closures below, so their `__globals__`
    # is *this module's* globals. FastAPI resolves body-model annotations via
    # get_type_hints against __globals__, so the request models must live in
    # the module globals (not build_app's locals) to be bound as bodies
    # rather than falling back to query params (422).
    globals().update(
        EmbedRequest=EmbedRequest, GraphArtifactUploadRequest=GraphArtifactUploadRequest,
        GraphChunkRequest=GraphChunkRequest, GraphImportRequest=GraphImportRequest,
        GraphJobBatchRequest=GraphJobBatchRequest, GraphJobRequest=GraphJobRequest, JobRequest=JobRequest,
        ReembedSource=ReembedSource, V1EmbeddingRequest=V1EmbeddingRequest,
    )

    web = FastAPI(title="pi-vault-mind embeddings", version="0.1.0")
    api_token = getattr(cfg, "API_TOKEN", "") or ""

    def require_auth(authorization: str | None = Header(default=None)):
        if not api_token:  # misconfigured deploy → fail closed
            raise HTTPException(503, "Server auth not configured")
        expected = f"Bearer {api_token}"
        if authorization != expected:
            raise HTTPException(401, "Invalid or missing bearer token")

    @web.get("/health")
    def health():
        return {"ok": True, "default_model": cfg.DEFAULT_MODEL}

    @web.get("/models")
    def models():
        return {
            "default": cfg.DEFAULT_MODEL,
            "default_dim": cfg.DEFAULT_DIM,
            "models": [s.to_public() for s in registry.values()],
        }

    @web.get("/stats", dependencies=[Depends(require_auth)])
    def stats():
        if reload_fn is not None:
            reload_fn()
        data = store.stats()
        data["gpu"] = (cfg.GPU or "cpu") if hasattr(cfg, "GPU") else "cpu"
        data["default_model"] = cfg.DEFAULT_MODEL
        if loaded_models is not None:
            data["loaded_models"] = loaded_models()
        return data

    @web.post("/embed", dependencies=[Depends(require_auth)])
    def embed(req: EmbedRequest):
        if req.task not in ("query", "document"):
            raise HTTPException(400, "task must be 'query' or 'document'")
        if not req.texts:
            return {"model": cfg.DEFAULT_MODEL, "dim": 0, "vectors": []}
        model = req.model or cfg.DEFAULT_MODEL
        try:
            spec = get_spec_fn(model)
            out_dim = spec.resolve_dim(req.dim)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        vectors = embed_fn(
            spec, req.texts, task=req.task, dim=out_dim,
            cache_dir=getattr(cfg, "CACHE_DIR", "/cache"),
            batch_size=getattr(cfg, "BATCH_SIZE", 64),
        )
        return {"model": spec.key, "dim": out_dim, "vectors": vectors}

    @web.post("/v1/embeddings", dependencies=[Depends(require_auth)])
    def v1_embeddings(req: V1EmbeddingRequest):
        # Normalize input to list of strings
        if isinstance(req.input, str):
            texts = [req.input]
        elif isinstance(req.input, list) and req.input and isinstance(req.input[0], int):
            # Tokenized input — not supported, return error
            raise HTTPException(400, "Tokenized input not supported. Use string input.")
        elif isinstance(req.input, list) and req.input and isinstance(req.input[0], list):
            raise HTTPException(400, "Tokenized input not supported. Use string input.")
        else:
            texts = req.input  # type: list[str]

        if not texts:
            return {
                "object": "list",
                "data": [],
                "model": req.model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        try:
            spec = get_spec_fn(req.model)
            out_dim = spec.native_dim
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

        vectors = embed_fn(
            spec, texts, task="document", dim=out_dim,
            cache_dir=getattr(cfg, "CACHE_DIR", "/cache"),
            batch_size=getattr(cfg, "BATCH_SIZE", 64),
        )

        data = [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ]
        return {
            "object": "list",
            "data": data,
            "model": spec.key,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @web.post("/jobs", dependencies=[Depends(require_auth)])
    def submit_job(req: JobRequest):
        try:
            spec = get_spec_fn(req.model or cfg.DEFAULT_MODEL)
            resolved = spec.resolve_dim(req.dim)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

        set_count = sum(bool(x) for x in (req.records, req.records_file, req.reembed_from))
        if set_count != 1:
            raise HTTPException(
                400,
                "Provide exactly one of: records (inline), records_file "
                "(Volume JSONL), or reembed_from (existing namespace).",
            )
        if req.records is not None:
            for r in req.records:
                if "id" not in r or "text" not in r:
                    raise HTTPException(400, "Each record needs 'id' and 'text'.")
            total = len(req.records)
        else:
            total = 0  # file/re-embed total filled in by the worker

        job_id = _new_job_id()
        write_job(
            job_id,
            {
                "status": "queued",
                "collection": req.collection,
                "model": spec.key,
                "dim": resolved,
                "total": total,
                "processed": 0,
            },
        )
        call = spawn_job(job_id, req.collection, spec.key, req.dim, req.model_dump())
        call_id = getattr(call, "object_id", getattr(call, "call_id", ""))
        return {"job_id": job_id, "call_id": call_id, "total": total}

    @web.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def job_status(job_id: str):
        doc = read_job(job_id)
        if doc is None:
            raise HTTPException(404, "Unknown job_id")
        return doc

    @web.get("/jobs", dependencies=[Depends(require_auth)])
    def list_jobs(limit: int = getattr(cfg, "JOB_LIST_LIMIT", 50)):
        limit = max(1, min(limit, 500))
        docs = list_job_docs(limit)
        return {"jobs": docs, "count": len(docs)}

    @web.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
    def cancel_job(job_id: str):
        doc = read_job(job_id)
        if doc is None:
            raise HTTPException(404, "Unknown job_id")
        if doc.get("status") in ("done", "error", "cancelled"):
            raise HTTPException(409, f"Job already {doc['status']}")
        write_job(job_id, {"cancel_requested": True})
        return {"job_id": job_id, "cancel_requested": True}

    @web.get("/sync/collections", dependencies=[Depends(require_auth)])
    def sync_collections():
        if reload_fn is not None:
            reload_fn()
        return {"collections": store.list_collections()}

    @web.get("/sync/export", dependencies=[Depends(require_auth)])
    def sync_export(
        collection: str,
        model: str | None = None,
        dim: int | None = None,
        since: int = 0,
        limit: int = getattr(cfg, "EXPORT_LIMIT_DEFAULT", 500),
        format: str = "json",
    ):
        if reload_fn is not None:
            reload_fn()
        if format not in ("json", "arrow"):
            raise HTTPException(400, "format must be 'json' or 'arrow'")
        try:
            spec = get_spec_fn(model or cfg.DEFAULT_MODEL)
            out_dim = spec.resolve_dim(dim)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

        limit = max(1, min(limit, getattr(cfg, "EXPORT_LIMIT_MAX", 5000)))

        if format == "arrow":
            payload, next_wm, done, count = store.export_since_arrow(
                collection, spec.key, out_dim, since, limit
            )
            return RawResponse(
                content=payload,
                media_type="application/vnd.apache.arrow.stream",
                headers={
                    "X-Next-Watermark": str(next_wm),
                    "X-Done": str(done).lower(),
                    "X-Count": str(count),
                },
            )
        # JSON path — rows always carry vector.
        return store.export_since(collection, spec.key, out_dim, since, limit)

    @web.post("/graph/import", dependencies=[Depends(require_auth)])
    def graph_import(req: GraphImportRequest):
        if reload_fn is not None:
            reload_fn()
        result = store.replace_graph_snapshot(req.graph_id, req.entities, req.relations)
        return {"ok": True, **result}

    @web.post("/graph/upsert", dependencies=[Depends(require_auth)])
    def graph_upsert(req: GraphChunkRequest):
        if reload_fn is not None:
            reload_fn()
        result = store.upsert_graph_snapshot(req.graph_id, req.entities, req.relations)
        return {"ok": True, **result}

    @web.get("/graph/export-page", dependencies=[Depends(require_auth)])
    def graph_export_page(graph_id: str, kind: str, offset: int = 0, limit: int = 500):
        if reload_fn is not None:
            reload_fn()
        if kind not in ("entities", "relations"):
            raise HTTPException(400, "kind must be 'entities' or 'relations'")
        return store.export_graph_page(graph_id, kind, max(0, offset), max(1, min(limit, 5000)))

    @web.get("/graph/export", dependencies=[Depends(require_auth)])
    def graph_export(graph_id: str):
        if reload_fn is not None:
            reload_fn()
        return store.export_graph_snapshot(graph_id)

    @web.post("/graph/artifacts", dependencies=[Depends(require_auth)])
    def graph_artifact_upload(req: GraphArtifactUploadRequest):
        if stage_graph_artifact is None:
            raise HTTPException(501, "graph artifacts not configured")
        return stage_graph_artifact(req.family, req.kind, req.rows, req.sha256)

    @web.post("/graph/jobs", dependencies=[Depends(require_auth)])
    def graph_job_create(req: GraphJobRequest):
        if create_graph_job is None:
            raise HTTPException(501, "graph jobs not configured")
        artifact_manifest = req.manifest.model_dump() if req.manifest else None
        return create_graph_job(req.graph_id, req.mode, artifact_manifest)

    @web.post("/graph/jobs/{job_id}/batches", dependencies=[Depends(require_auth)])
    def graph_job_batches(job_id: str, req: GraphJobBatchRequest):
        if append_graph_batch is None:
            raise HTTPException(501, "graph jobs not configured")
        total = len(req.entities) + len(req.relations)
        if total <= 0:
            raise HTTPException(400, "Provide at least one entity or relation row.")
        if total > getattr(cfg, "GRAPH_UPLOAD_BATCH_MAX", 64):
            raise HTTPException(400, "Graph batch exceeds GRAPH_UPLOAD_BATCH_MAX.")
        return append_graph_batch(job_id, req.entities, req.relations, req.batch_key)

    @web.post("/graph/jobs/{job_id}/start", dependencies=[Depends(require_auth)])
    def graph_job_start(job_id: str):
        if start_graph_job is None:
            raise HTTPException(501, "graph jobs not configured")
        return start_graph_job(job_id)

    @web.get("/graph/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def graph_job_status(job_id: str):
        if read_graph_job is None:
            raise HTTPException(501, "graph jobs not configured")
        doc = read_graph_job(job_id)
        if doc is None:
            raise HTTPException(404, "Unknown graph job_id")
        return doc

    @web.get("/graph/jobs", dependencies=[Depends(require_auth)])
    def graph_job_list(limit: int = getattr(cfg, "JOB_LIST_LIMIT", 50)):
        if list_graph_jobs is None:
            raise HTTPException(501, "graph jobs not configured")
        jobs = list_graph_jobs(max(1, min(limit, 500)))
        return {"jobs": jobs, "count": len(jobs)}

    return web


def _new_job_id() -> str:
    import uuid

    return uuid.uuid4().hex


# Convenience: a minimal in-memory job store for tests / local-entrypoint use.
class MemoryJobStore:
    """Stdlib-only in-memory job doc store (no FastAPI/pydantic needed)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    def write(self, job_id: str, fields: dict) -> None:
        doc = self.docs.get(job_id, {})
        doc.update(fields)
        doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.docs[job_id] = doc

    def read(self, job_id: str) -> dict | None:
        return self.docs.get(job_id)

    def list(self, limit: int) -> list[dict]:
        docs = sorted(self.docs.values(),
                      key=lambda d: d.get("updated_at", ""), reverse=True)
        return docs[:limit]


def write_job_merger(store: MemoryJobStore) -> Callable[[str, dict], None]:
    """Adapt the merge-style ``write_job(job_id, **fields)`` API to MemoryJobStore."""

    def _write(job_id: str, fields: dict) -> None:
        store.write(job_id, fields)

    return _write