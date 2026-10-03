"""FastAPI web app factory, separated from the Modal layer.

Modal's :class:`modal.App` is built in ``app.py``; this module builds the
**HTTP contract** in isolation so it can be tested with a FastAPI
``TestClient`` and mocked store/embed/job layers: no Modal SDK, no GPU, no
network. ``app.py`` injects the real Modal-backed closures at runtime.

The factory is the single source of truth for the HTTP API contract. Additive
changes go here and are mirrored in the client examples.

Import discipline
-----------------
``fastapi`` / ``pydantic`` are imported **inside** :func:`build_app` (not at
module top level). The Modal CLI imports this module locally to serialize the
App: on a laptop without the ML/web stack, the top-level import must stay
light so ``modal deploy`` works. ``MemoryJobStore`` (stdlib only) and
``libs.hooks`` (stdlib only) are safe at module level.

Lifecycle hooks
---------------
This module owns the package's hooks INSTANCE and fires it at the HTTP
lifecycle boundaries. Closed tag set (new tags = this package's releases):

- ``request.pre``  : at the entry of every route that requires auth,
  before any handler logic - (method, path)
- ``request.post`` : at the return of every route handler - (method,
  path, status_code)
- ``embed.post``   : after a successful embed response is built -
  ({"path", "count"})
- ``job.pre``      : just before a new bulk job is accepted - ({"job_id"})
- ``job.post``     : when a bulk job reaches a terminal state in the
  worker - ({"job_id", "status"})

``fire()`` contains handler errors (returns + records them via
``last_errors(tag)``); it never breaks a request.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from libs.hooks import Hooks

# The package's hook seam (family contract): one shared instance, closed tag
# set above; lanes register with @hooks.on(...) / hooks.register(...) and the
# fire sites below contain handler errors, never the host request path.
hooks = Hooks(
    ("request.pre", "request.post", "embed.post", "job.pre", "job.post"),
    name="modal-embedding-server",
)


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
        ``config``: provides DEFAULT_MODEL, DEFAULT_DIM, EXPORT_LIMIT_MAX, etc.
        plus ``API_TOKEN`` (bound in by ``app.py`` from the Modal Secret).
    store : module
        ``store``: list_collections, export_since, export_since_arrow, stats.
    embed_fn : callable
        ``embedders.embed`` (or a mock returning canned vectors).
    get_spec_fn : callable
        ``embedders.get_spec`` (or a mock).
    registry : dict
        ``enabled_registry()``: used to list models on ``GET /models``.
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

    from schemas import (
        EmbedRequest,
        GraphArtifactUploadRequest,
        GraphChunkRequest,
        GraphImportRequest,
        GraphJobBatchRequest,
        GraphJobRequest,
        JobRequest,
        ReembedSource,
        V1EmbeddingRequest,
    )

    # Route handlers are defined as closures below, so their `__globals__`
    # is *this module's* globals. FastAPI resolves body-model annotations via
    # get_type_hints against __globals__, so the request models must live in
    # the module globals (not build_app's locals) to be bound as bodies
    # rather than falling back to query params (422).
    globals().update(
        EmbedRequest=EmbedRequest,
        GraphArtifactUploadRequest=GraphArtifactUploadRequest,
        GraphChunkRequest=GraphChunkRequest,
        GraphImportRequest=GraphImportRequest,
        GraphJobBatchRequest=GraphJobBatchRequest,
        GraphJobRequest=GraphJobRequest,
        JobRequest=JobRequest,
        ReembedSource=ReembedSource,
        V1EmbeddingRequest=V1EmbeddingRequest,
    )

    web = FastAPI(title="modal-embedding-server embeddings", version="0.1.0")
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
        hooks.fire("request.pre", "GET", "/stats")
        if reload_fn is not None:
            reload_fn()
        data = store.stats()
        data["gpu"] = (cfg.GPU or "cpu") if hasattr(cfg, "GPU") else "cpu"
        data["default_model"] = cfg.DEFAULT_MODEL
        if loaded_models is not None:
            data["loaded_models"] = loaded_models()
        hooks.fire("request.post", "GET", "/stats", 200)
        return data

    @web.post("/embed", dependencies=[Depends(require_auth)])
    def embed(req: EmbedRequest):
        hooks.fire("request.pre", "POST", "/embed")
        if req.task not in ("query", "document"):
            raise HTTPException(400, "task must be 'query' or 'document'")
        if not req.texts:
            hooks.fire("request.post", "POST", "/embed", 200)
            return {"model": cfg.DEFAULT_MODEL, "dim": 0, "vectors": []}
        model = req.model or cfg.DEFAULT_MODEL
        try:
            spec = get_spec_fn(model)
            out_dim = spec.resolve_dim(req.dim)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        vectors = embed_fn(
            spec,
            req.texts,
            task=req.task,
            dim=out_dim,
            cache_dir=getattr(cfg, "CACHE_DIR", "/cache"),
            batch_size=getattr(cfg, "BATCH_SIZE", 64),
        )
        hooks.fire("embed.post", {"path": "/embed", "count": len(req.texts)})
        hooks.fire("request.post", "POST", "/embed", 200)
        return {"model": spec.key, "dim": out_dim, "vectors": vectors}

    @web.post("/v1/embeddings", dependencies=[Depends(require_auth)])
    def v1_embeddings(req: V1EmbeddingRequest):
        hooks.fire("request.pre", "POST", "/v1/embeddings")
        # Normalize input to list of strings
        if isinstance(req.input, str):
            texts = [req.input]
        elif (
            isinstance(req.input, list) and req.input and isinstance(req.input[0], int)
        ):
            # Tokenized input: not supported, return error
            raise HTTPException(400, "Tokenized input not supported. Use string input.")
        elif (
            isinstance(req.input, list) and req.input and isinstance(req.input[0], list)
        ):
            raise HTTPException(400, "Tokenized input not supported. Use string input.")
        else:
            texts = req.input  # type: list[str]

        if not texts:
            hooks.fire("request.post", "POST", "/v1/embeddings", 200)
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
            spec,
            texts,
            task="document",
            dim=out_dim,
            cache_dir=getattr(cfg, "CACHE_DIR", "/cache"),
            batch_size=getattr(cfg, "BATCH_SIZE", 64),
        )

        data = [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ]
        hooks.fire("request.post", "POST", "/v1/embeddings", 200)
        return {
            "object": "list",
            "data": data,
            "model": spec.key,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @web.post("/jobs", dependencies=[Depends(require_auth)])
    def submit_job(req: JobRequest):
        hooks.fire("request.pre", "POST", "/jobs")
        try:
            spec = get_spec_fn(req.model or cfg.DEFAULT_MODEL)
            resolved = spec.resolve_dim(req.dim)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))

        set_count = sum(
            bool(x) for x in (req.records, req.records_file, req.reembed_from)
        )
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
        hooks.fire("job.pre", {"job_id": job_id})
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
        hooks.fire("request.post", "POST", "/jobs", 200)
        return {"job_id": job_id, "call_id": call_id, "total": total}

    @web.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def job_status(job_id: str):
        hooks.fire("request.pre", "GET", "/jobs/{job_id}")
        doc = read_job(job_id)
        if doc is None:
            raise HTTPException(404, "Unknown job_id")
        hooks.fire("request.post", "GET", "/jobs/{job_id}", 200)
        return doc

    @web.get("/jobs", dependencies=[Depends(require_auth)])
    def list_jobs(limit: int = getattr(cfg, "JOB_LIST_LIMIT", 50)):
        hooks.fire("request.pre", "GET", "/jobs")
        limit = max(1, min(limit, 500))
        docs = list_job_docs(limit)
        hooks.fire("request.post", "GET", "/jobs", 200)
        return {"jobs": docs, "count": len(docs)}

    @web.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
    def cancel_job(job_id: str):
        hooks.fire("request.pre", "POST", "/jobs/{job_id}/cancel")
        doc = read_job(job_id)
        if doc is None:
            raise HTTPException(404, "Unknown job_id")
        if doc.get("status") in ("done", "error", "cancelled"):
            raise HTTPException(409, f"Job already {doc['status']}")
        write_job(job_id, {"cancel_requested": True})
        hooks.fire("request.post", "POST", "/jobs/{job_id}/cancel", 200)
        return {"job_id": job_id, "cancel_requested": True}

    @web.get("/sync/collections", dependencies=[Depends(require_auth)])
    def sync_collections():
        hooks.fire("request.pre", "GET", "/sync/collections")
        if reload_fn is not None:
            reload_fn()
        hooks.fire("request.post", "GET", "/sync/collections", 200)
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
        hooks.fire("request.pre", "GET", "/sync/export")
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
            hooks.fire("request.post", "GET", "/sync/export", 200)
            return RawResponse(
                content=payload,
                media_type="application/vnd.apache.arrow.stream",
                headers={
                    "X-Next-Watermark": str(next_wm),
                    "X-Done": str(done).lower(),
                    "X-Count": str(count),
                },
            )
        # JSON path: rows always carry vector.
        hooks.fire("request.post", "GET", "/sync/export", 200)
        return store.export_since(collection, spec.key, out_dim, since, limit)

    @web.post("/graph/import", dependencies=[Depends(require_auth)])
    def graph_import(req: GraphImportRequest):
        hooks.fire("request.pre", "POST", "/graph/import")
        if reload_fn is not None:
            reload_fn()
        result = store.replace_graph_snapshot(req.graph_id, req.entities, req.relations)
        hooks.fire("request.post", "POST", "/graph/import", 200)
        return {"ok": True, **result}

    @web.post("/graph/upsert", dependencies=[Depends(require_auth)])
    def graph_upsert(req: GraphChunkRequest):
        hooks.fire("request.pre", "POST", "/graph/upsert")
        if reload_fn is not None:
            reload_fn()
        result = store.upsert_graph_snapshot(req.graph_id, req.entities, req.relations)
        hooks.fire("request.post", "POST", "/graph/upsert", 200)
        return {"ok": True, **result}

    @web.get("/graph/export-page", dependencies=[Depends(require_auth)])
    def graph_export_page(graph_id: str, kind: str, offset: int = 0, limit: int = 500):
        hooks.fire("request.pre", "GET", "/graph/export-page")
        if reload_fn is not None:
            reload_fn()
        if kind not in ("entities", "relations"):
            raise HTTPException(400, "kind must be 'entities' or 'relations'")
        page = store.export_graph_page(
            graph_id, kind, max(0, offset), max(1, min(limit, 5000))
        )
        hooks.fire("request.post", "GET", "/graph/export-page", 200)
        return page

    @web.get("/graph/export", dependencies=[Depends(require_auth)])
    def graph_export(graph_id: str):
        hooks.fire("request.pre", "GET", "/graph/export")
        if reload_fn is not None:
            reload_fn()
        snapshot = store.export_graph_snapshot(graph_id)
        hooks.fire("request.post", "GET", "/graph/export", 200)
        return snapshot

    @web.post("/graph/artifacts", dependencies=[Depends(require_auth)])
    def graph_artifact_upload(req: GraphArtifactUploadRequest):
        hooks.fire("request.pre", "POST", "/graph/artifacts")
        if stage_graph_artifact is None:
            raise HTTPException(501, "graph artifacts not configured")
        staged = stage_graph_artifact(req.family, req.kind, req.rows, req.sha256)
        hooks.fire("request.post", "POST", "/graph/artifacts", 200)
        return staged

    @web.post("/graph/jobs", dependencies=[Depends(require_auth)])
    def graph_job_create(req: GraphJobRequest):
        hooks.fire("request.pre", "POST", "/graph/jobs")
        if create_graph_job is None:
            raise HTTPException(501, "graph jobs not configured")
        artifact_manifest = req.manifest.model_dump() if req.manifest else None
        created = create_graph_job(req.graph_id, req.mode, artifact_manifest)
        hooks.fire("request.post", "POST", "/graph/jobs", 200)
        return created

    @web.post("/graph/jobs/{job_id}/batches", dependencies=[Depends(require_auth)])
    def graph_job_batches(job_id: str, req: GraphJobBatchRequest):
        hooks.fire("request.pre", "POST", "/graph/jobs/{job_id}/batches")
        if append_graph_batch is None:
            raise HTTPException(501, "graph jobs not configured")
        total = len(req.entities) + len(req.relations)
        if total <= 0:
            raise HTTPException(400, "Provide at least one entity or relation row.")
        if total > getattr(cfg, "GRAPH_UPLOAD_BATCH_MAX", 64):
            raise HTTPException(400, "Graph batch exceeds GRAPH_UPLOAD_BATCH_MAX.")
        appended = append_graph_batch(
            job_id, req.entities, req.relations, req.batch_key
        )
        hooks.fire("request.post", "POST", "/graph/jobs/{job_id}/batches", 200)
        return appended

    @web.post("/graph/jobs/{job_id}/start", dependencies=[Depends(require_auth)])
    def graph_job_start(job_id: str):
        hooks.fire("request.pre", "POST", "/graph/jobs/{job_id}/start")
        if start_graph_job is None:
            raise HTTPException(501, "graph jobs not configured")
        started = start_graph_job(job_id)
        hooks.fire("request.post", "POST", "/graph/jobs/{job_id}/start", 200)
        return started

    @web.get("/graph/jobs/{job_id}", dependencies=[Depends(require_auth)])
    def graph_job_status(job_id: str):
        hooks.fire("request.pre", "GET", "/graph/jobs/{job_id}")
        if read_graph_job is None:
            raise HTTPException(501, "graph jobs not configured")
        doc = read_graph_job(job_id)
        if doc is None:
            raise HTTPException(404, "Unknown graph job_id")
        hooks.fire("request.post", "GET", "/graph/jobs/{job_id}", 200)
        return doc

    @web.get("/graph/jobs", dependencies=[Depends(require_auth)])
    def graph_job_list(limit: int = getattr(cfg, "JOB_LIST_LIMIT", 50)):
        hooks.fire("request.pre", "GET", "/graph/jobs")
        if list_graph_jobs is None:
            raise HTTPException(501, "graph jobs not configured")
        jobs = list_graph_jobs(max(1, min(limit, 500)))
        hooks.fire("request.post", "GET", "/graph/jobs", 200)
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
        docs = sorted(
            self.docs.values(), key=lambda d: d.get("updated_at", ""), reverse=True
        )
        return docs[:limit]


def write_job_merger(store: MemoryJobStore) -> Callable[[str, dict], None]:
    """Adapt the merge-style ``write_job(job_id, **fields)`` API to MemoryJobStore."""

    def _write(job_id: str, fields: dict) -> None:
        store.write(job_id, fields)

    return _write
