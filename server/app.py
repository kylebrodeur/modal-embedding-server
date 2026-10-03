"""pi-vault-mind embedding service on Modal.

Three capabilities, one App:

1. **On-demand embedding** - low-latency ``POST /embed`` for interactive
   ``wiki_search`` queries and small appends.
2. **Bulk background jobs** - ``POST /jobs`` spawns a GPU worker that embeds
   many records and writes them to the Volume-backed LanceDB; poll
   ``GET /jobs/{id}`` for progress, ``GET /jobs`` to list, ``POST /jobs/{id}/cancel``
   for a cooperative cancel.
3. **Sync** - ``GET /sync/collections`` and ``GET /sync/export`` (JSON or Arrow
   IPC) let the local extension pull new vectors down into its own ``.lancedb``.

Bulk ingestion paths
--------------------
A job may be submitted from any one of:
- **inline records** - ``records: [{id, text, metadata?}]``,
- **a JSONL file staged on the Volume** - ``records_file: "/vectors/.../x.jsonl"``
  (for very large corpora that shouldn't round-trip through the web layer), or
- **re-embed of an existing namespace** - ``reembed_from: {collection, model, dim}``
  to re-embed a source table's text into a new ``model__dim`` namespace.

Jobs are idempotent (rows keyed by ``id``, merge-inserted) and resumable
(re-submitting the same records upserts rather than duplicates).

The HTTP contract lives in ``web.py`` (importable without the Modal SDK so it
can be tested with a ``TestClient``); this file wires it to Modal Volumes,
Secrets, and the GPU bulk worker.

Deploy (uvx runs the Modal CLI in an ephemeral env - no global install):
    uvx modal deploy modal/app.py
Run locally against the deployed app:
    uv run modal/client_example.py

See modal/README.md for secret setup.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

import modal

import config
from embedders import enabled_registry, get_spec
from web import build_app

# Structured logging (one JSON object per line).
logging.basicConfig(
    level=os.environ.get("MODAL_EMBED_LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("pvm.modal")

# Image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=5.0.0",
        "transformers>=4.56.0",
        "torch>=2.4.0",
        "lancedb>=0.16.0",
        "pyarrow>=17.0.0",
        "fastapi[standard]>=0.115.0",
        "huggingface_hub>=0.25.0",
    )
    .env({"HF_HOME": config.CACHE_DIR, "SENTENCE_TRANSFORMERS_HOME": config.CACHE_DIR})
    .add_local_python_source("config", "embedders", "store", "web", "schemas")
)

app = modal.App(config.APP_NAME, image=image)

vectors_volume = modal.Volume.from_name(
    config.VECTORS_VOLUME_NAME, create_if_missing=True,
    version=config.VECTORS_VOLUME_VERSION,
)
cache_volume = modal.Volume.from_name(config.CACHE_VOLUME_NAME, create_if_missing=True)

VOLUMES = {config.VECTORS_DIR: vectors_volume, config.CACHE_DIR: cache_volume}

# Secrets are optional at import time so the file still loads for `modal run`
# in environments where they're not yet created; the functions that need them
# declare them explicitly below.
# Secrets. Default: attach the named secrets so a normal deploy is
# fail-closed-by-configuration (the web service reads API_TOKEN from the
# auth secret; without it every protected route returns 503, and the bulk
# worker can't load gated models without HF_TOKEN). To stand up the infra
# *before* the secrets exist (create them, then redeploy), set
# PVM_ATTACH_SECRETS=0 — the Secret objects are then not created at all, so
# Modal doesn't register them as code deps (which would otherwise mismatch
# and crash-loop the container).
_ATTACH_SECRETS = os.environ.get("MODAL_EMBED_ATTACH_SECRETS", "1") not in ("", "0", "false")
auth_secret = modal.Secret.from_name(config.AUTH_SECRET_NAME) if _ATTACH_SECRETS else None
hf_secret = modal.Secret.from_name(config.HF_SECRET_NAME) if _ATTACH_SECRETS else None


# Job status helpers (persisted on the Volume)
def _job_path(job_id: str) -> str:
    return f"{config.JOBS_DIR}/{job_id}.json"


def _write_job(job_id: str, **fields) -> None:
    jobs_dir = Path(config.JOBS_DIR)
    if not jobs_dir.exists():
        try:
            jobs_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass
    path = _job_path(job_id)
    doc: dict = {}
    if os.path.exists(path):
        with open(path) as fh:
            doc = json.load(fh)
    doc.update(fields)
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp_path = str(jobs_dir / f"{job_id}.{uuid.uuid4().hex}.tmp")
    with open(tmp_path, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp_path, path)
    vectors_volume.commit()

def _reload_volume_best_effort() -> None:
    try:
        vectors_volume.reload()
    except RuntimeError:
        pass

def _read_job(job_id: str) -> dict | None:
    _reload_volume_best_effort()
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _list_job_docs(limit: int) -> list[dict]:
    _reload_volume_best_effort()
    jobs_dir = Path(config.JOBS_DIR)
    if not jobs_dir.exists():
        return []

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    # Sort paths by mtime (newest first) and parse only the top `limit`, so a
    # large job history doesn't mean reading + JSON-parsing every file on each
    # /jobs call. mtime tracks the last status write, matching `updated_at`.
    paths = sorted(jobs_dir.glob("*.json"), key=_mtime, reverse=True)
    docs: list[dict] = []
    for p in paths:
        if len(docs) >= limit:
            break
        try:
            with open(p) as fh:
                docs.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    return docs


def _loaded_models() -> list[str]:
    """Names of currently-loaded embedder models in this container."""
    try:
        from embedders import _loaded
        return list(_loaded.keys())
    except Exception:  # noqa: BLE001
        return []


# Record sources for bulk jobs
def _iter_jsonl(path: str, batch: int):
    """Stream records from a JSONL file on the Volume, batched."""
    with open(path) as fh:
        buf: list[dict] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            buf.append(json.loads(line))
            if len(buf) >= batch:
                yield buf
                buf = []
        if buf:
            yield buf


def _iter_reembed(source_collection: str, source_model: str, source_dim: int, batch: int):
    """Stream (id, text, metadata) from an existing namespace to re-embed."""
    import lancedb
    import pyarrow as pa  # noqa: F401 - lancedb needs pyarrow at runtime

    from store import VECTORS_DIR, table_name, _tables

    db = lancedb.connect(VECTORS_DIR())
    name = table_name(source_collection, source_model, source_dim)
    if name not in _tables(db):
        raise ValueError(f"Re-embed source table '{name}' does not exist")
    tbl = db.open_table(name)
    # Stream the source namespace in batches instead of materializing the whole
    # table — a large vault would otherwise load every row into memory at once.
    for record_batch in tbl.search().select(["id", "text", "metadata"]).to_batches(batch):
        records = []
        for r in record_batch.to_pylist():
            meta = r.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            records.append({"id": r["id"], "text": r.get("text", ""), "metadata": meta})
        if records:
            yield records


# Bulk workers

@app.function(
    gpu=config.GPU,
    volumes=VOLUMES,
    secrets=[hf_secret] if _ATTACH_SECRETS else [],
    timeout=60 * 60,  # long jobs allowed
)
def embed_batch(job_id: str, collection: str, model: str, dim: int | None, req: dict) -> dict:
    """Embed records from ``req`` and upsert to the store.

    ``req`` is the JSON of the original JobRequest so the worker can resolve
    inline / file / re-embed sources without a second round-trip.
    Cooperative cancellation: if the job doc has ``cancel_requested=True``
    (set by ``POST /jobs/{id}/cancel``), the worker stops after the current
    batch and writes ``status=cancelled``.
    """
    import store

    spec = get_spec(model)
    resolved_dim = spec.resolve_dim(dim)

    total = 0
    source_kind = None
    if req.get("records"):
        source_kind = "inline"
        total = len(req["records"])
    elif req.get("records_file"):
        source_kind = "file"
    elif req.get("reembed_from"):
        source_kind = "reembed"

    _write_job(
        job_id,
        status="running",
        collection=collection,
        model=spec.key,
        dim=resolved_dim,
        total=total,
        processed=0,
        source=source_kind,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    try:
        from embedders import embed as embed_texts

        def batches():
            if req.get("records"):
                recs = req["records"]
                for i in range(0, len(recs), config.BATCH_SIZE):
                    yield recs[i : i + config.BATCH_SIZE]
            elif req.get("records_file"):
                yield from _iter_jsonl(req["records_file"], config.BATCH_SIZE)
            elif req.get("reembed_from"):
                src = req["reembed_from"]
                yield from _iter_reembed(
                    src["collection"], src["model"], int(src["dim"]), config.BATCH_SIZE
                )
            else:
                return

        processed = 0
        for chunk in batches():
            # Cooperative cancel check.
            latest = _read_job(job_id) or {}
            if latest.get("cancel_requested"):
                _write_job(
                    job_id,
                    status="cancelled",
                    processed=processed,
                    cancelled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                log.info("job %s cancelled at %d/%d", job_id, processed, total)
                return {"job_id": job_id, "status": "cancelled", "processed": processed}

            if total == 0 and source_kind == "file":
                # We didn't know the file length up-front; update total lazily.
                total = processed + len(chunk)
                _write_job(job_id, total=total)

            texts = [r.get("text", "") for r in chunk]
            vectors = embed_texts(
                spec, texts, task="document", dim=resolved_dim,
                cache_dir=config.CACHE_DIR, batch_size=config.BATCH_SIZE,
            )
            rows = [
                {
                    "id": r["id"],
                    "text": r.get("text", ""),
                    "vector": v,
                    "metadata": r.get("metadata", {}),
                    "created_at": r.get("created_at"),
                }
                for r, v in zip(chunk, vectors)
            ]
            store.upsert(collection, spec.key, resolved_dim, rows)
            vectors_volume.commit()
            processed += len(chunk)
            _write_job(job_id, processed=processed)
            log.info("job %s: %d/%d", job_id, processed, total)

        _write_job(
            job_id,
            status="done",
            processed=processed,
            total=total if total else processed,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return {"job_id": job_id, "status": "done", "processed": processed}
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client
        log.exception("job %s failed", job_id)
        _write_job(job_id, status="error", error=str(exc))
        raise


# On-demand + sync web service
@app.cls(
    gpu=config.GPU,
    volumes=VOLUMES,
    secrets=[s for s in (auth_secret, hf_secret) if s],
    scaledown_window=config.SCALEDOWN_WINDOW,
    # min_containers=1,  # uncomment to keep one warm and kill cold starts
)
@modal.concurrent(max_inputs=config.MAX_CONCURRENT_INPUTS)
class EmbeddingService:
    @modal.enter()
    def _startup(self):
        # Warm the canonical model so the first request is fast. Run in a
        # background daemon thread so container readiness (and /health,
        # /models) is NOT blocked on a (possibly slow / gated) model download.
        # Only meaningful for the sentence-transformers backend; ollama/hf
        # backends are no-ops. Skip entirely with PVM_PREWARM=0.
        import threading

        if os.environ.get("MODAL_EMBED_PREWARM", "1") in ("", "0", "false"):
            log.info("pre-warm disabled by PVM_PREWARM")
            return

        def _warm():
            from embedders import load_model
            try:
                load_model(get_spec(config.DEFAULT_MODEL), config.CACHE_DIR)
                log.info("pre-warmed default model %s", config.DEFAULT_MODEL)
            except Exception as exc:  # noqa: BLE001 - never crash startup
                log.warning("could not pre-warm default model: %s", exc)

        threading.Thread(target=_warm, daemon=True).start()

    @modal.asgi_app()
    def fastapi_app(self):
        import store
        from embedders import embed as embed_texts

        api_token = os.environ.get("API_TOKEN", "")
        # Bind the API token into a config-like namespace the factory can read,
        # since the real config module doesn't carry it (it lives in the Secret).
        cfg = type(
            "_Cfg",
            (),
            {**{k: getattr(config, k) for k in dir(config) if k.isupper()},
             "API_TOKEN": api_token},
        )()

        def _write_job_dict(job_id: str, fields: dict) -> None:
            _write_job(job_id, **fields)

        def _spawn_job(job_id, collection, model, dim, req_dict):
            return embed_batch.spawn(job_id, collection, model, dim, req_dict)

        return build_app(
            cfg=cfg,
            store=store,
            embed_fn=embed_texts,
            get_spec_fn=get_spec,
            registry=enabled_registry(),
            read_job=_read_job,
            loaded_models=_loaded_models,
            reload_fn=vectors_volume.reload,
        )


# Local entrypoint for quick smoke testing
@app.local_entrypoint()
def main():
    """`modal run modal/app.py` -> embeds records and runs a tiny bulk job."""
    print("Submitting a 3-record bulk job to the 'main' collection...")
    result = embed_batch.remote(
        job_id=uuid.uuid4().hex,
        collection="main",
        model=config.DEFAULT_MODEL,
        dim=None,
        req={
            "collection": "main",
            "records": [
                {"id": "1", "text": "JWT tokens expire after one hour.", "metadata": {"tag": "auth"}},
                {"id": "2", "text": "Refresh tokens live for 30 days.", "metadata": {"tag": "auth"}},
                {"id": "3", "text": "The vault watcher polls every 2 seconds.", "metadata": {"tag": "watcher"}},
            ],
        },
    )
    print("Bulk job result:", result)