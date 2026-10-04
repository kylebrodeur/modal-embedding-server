"""API contract tests via FastAPI TestClient with the model layer mocked.

No Modal SDK, no GPU. The store is real (temp-dir LanceDB from conftest); the
embed function is a fake returning deterministic vectors. This locks the HTTP
contract in docs/MODAL_EMBEDDING.md + src/modal-client.ts.
"""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

import store
from embedders import enabled_registry, get_spec
from web import MemoryJobStore, build_app


def _fake_embed(spec, texts, task, dim, cache_dir, batch_size=64):
    """Deterministic fake: a unit vector of the right dim per text."""
    return [[(i + 1) / dim for i in range(dim)] for _ in texts]


def _make_client(api_token: str = "testtoken", gpu: str | None = "L4"):
    cfg = types.SimpleNamespace(
        DEFAULT_MODEL="embeddinggemma",
        DEFAULT_DIM=None,
        CACHE_DIR="/cache",
        BATCH_SIZE=64,
        JOB_LIST_LIMIT=50,
        EXPORT_LIMIT_DEFAULT=500,
        EXPORT_LIMIT_MAX=5000,
        GPU=gpu,
        API_TOKEN=api_token,
    )
    jobs = MemoryJobStore()

    def spawn_job(job_id, collection, model, dim, req_dict):
        # Simulate the worker: embed the inline records with the fake and write
        # them to the (temp-dir) store so sync/export has real rows to return.
        records = req_dict.get("records") or []
        from embedders import get_spec as _gs
        spec = _gs(model)
        out_dim = spec.resolve_dim(dim)
        vectors = _fake_embed(spec, [r.get("text", "") for r in records],
                              task="document", dim=out_dim, cache_dir="/cache")
        rows = [{"id": r["id"], "text": r.get("text", ""), "vector": v,
                 "metadata": r.get("metadata", {})} for r, v in zip(records, vectors)]
        if rows:
            store.upsert(collection, spec.key, out_dim, rows)
        jobs.write(job_id, {"status": "done", "processed": len(records),
                            "total": len(records)})
        return types.SimpleNamespace(object_id=f"call-{job_id}")

    app = build_app(
        cfg=cfg,
        store=store,
        embed_fn=_fake_embed,
        get_spec_fn=get_spec,
        registry=enabled_registry(),
        read_job=jobs.read,
        write_job=jobs.write,
        list_job_docs=jobs.list,
        spawn_job=spawn_job,
        loaded_models=lambda: ["embeddinggemma"],
    )
    return TestClient(app), jobs


@pytest.fixture()
def client():
    c, _ = _make_client()
    return c


AUTH = {"Authorization": "Bearer testtoken"}


# health + models are public
def test_health_no_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["default_model"] == "embeddinggemma"


def test_models_public_lists_native_dim(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "embeddinggemma"
    keys = {m["key"] for m in body["models"]}
    assert "embeddinggemma" in keys
    gemma = next(m for m in body["models"] if m["key"] == "embeddinggemma")
    assert gemma["native_dim"] == 768
    assert gemma["backend"] == "sentence-transformers"


# auth gating
def test_embed_requires_auth(client):
    r = client.post("/embed", json={"texts": ["hi"]})
    assert r.status_code == 401


def test_fail_closed_when_no_token():
    c, _ = _make_client(api_token="")
    r = c.post("/embed", json={"texts": ["hi"]}, headers=AUTH)
    assert r.status_code == 503


# embed
def test_embed_returns_vectors(client):
    r = client.post("/embed", json={"texts": ["hello", "world"], "task": "query"},
                     headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "embeddinggemma"
    assert body["dim"] == 768
    assert len(body["vectors"]) == 2
    assert len(body["vectors"][0]) == 768


def test_embed_matryoshka_dim(client):
    r = client.post("/embed", json={"texts": ["x"], "dim": 256}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["dim"] == 256
    assert len(r.json()["vectors"][0]) == 256


def test_embed_bad_task(client):
    r = client.post("/embed", json={"texts": ["x"], "task": "nope"}, headers=AUTH)
    assert r.status_code == 400


def test_embed_unknown_model(client):
    r = client.post("/embed", json={"texts": ["x"], "model": "ghost"}, headers=AUTH)
    assert r.status_code == 400



# v1/embeddings
def test_v1_embeddings_requires_auth(client):
    r = client.post("/v1/embeddings", json={"model": "embeddinggemma", "input": ["hi"]})
    assert r.status_code == 401


def test_v1_embeddings_returns_vectors(client):
    r = client.post(
        "/v1/embeddings",
        json={"model": "embeddinggemma", "input": ["hello", "world"]},
        headers=AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "embeddinggemma"
    assert len(body["data"]) == 2
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 768
    assert "usage" in body


def test_v1_embeddings_bad_model(client):
    r = client.post(
        "/v1/embeddings",
        json={"model": "ghost", "input": ["x"]},
        headers=AUTH,
    )
    assert r.status_code == 400

# jobs
def test_submit_job_inline(client):
    r = client.post("/jobs", json={
        "collection": "main",
        "records": [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}],
    }, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["total"] == 2


def test_submit_job_requires_exactly_one_source(client):
    r = client.post("/jobs", json={"collection": "main"}, headers=AUTH)
    assert r.status_code == 400


def test_submit_job_rejects_records_missing_id(client):
    r = client.post("/jobs", json={
        "collection": "main",
        "records": [{"text": "no id"}],
    }, headers=AUTH)
    assert r.status_code == 400


def test_job_status_and_list(client):
    sub = client.post("/jobs", json={
        "collection": "main",
        "records": [{"id": "1", "text": "a"}],
    }, headers=AUTH).json()
    jid = sub["job_id"]

    r = client.get(f"/jobs/{jid}", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["status"] == "done"

    r = client.get("/jobs", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["jobs"][0]["collection"] == "main"


def test_job_status_unknown_404(client):
    r = client.get("/jobs/nope", headers=AUTH)
    assert r.status_code == 404


def test_cancel_job(client):
    # A job that's still running (we override spawn to leave it running here).
    cfg = types.SimpleNamespace(
        DEFAULT_MODEL="embeddinggemma", DEFAULT_DIM=None, CACHE_DIR="/cache",
        BATCH_SIZE=64, JOB_LIST_LIMIT=50, EXPORT_LIMIT_DEFAULT=500,
        EXPORT_LIMIT_MAX=5000, GPU="L4", API_TOKEN="testtoken",
    )
    jobs = MemoryJobStore()
    app = build_app(
        cfg=cfg, store=store, embed_fn=_fake_embed, get_spec_fn=get_spec,
        registry=enabled_registry(), read_job=jobs.read, write_job=jobs.write,
        list_job_docs=jobs.list,
        spawn_job=lambda *a: (jobs.write(a[0], {"status": "running", "total": 3,
                                                 "processed": 0}),
                              types.SimpleNamespace(object_id="c")),
    )
    c = TestClient(app)
    sub = c.post("/jobs", json={"collection": "main",
                                "records": [{"id": "1", "text": "a"}]}, headers=AUTH).json()
    r = c.post(f"/jobs/{sub['job_id']}/cancel", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["cancel_requested"] is True


def test_cancel_done_job_conflict(client):
    sub = client.post("/jobs", json={
        "collection": "main",
        "records": [{"id": "1", "text": "a"}],
    }, headers=AUTH).json()
    # spawn marks it done immediately in the `client` fixture.
    r = client.post(f"/jobs/{sub['job_id']}/cancel", headers=AUTH)
    assert r.status_code == 409


# sync
def test_sync_collections(client):
    client.post("/jobs", json={
        "collection": "main",
        "records": [{"id": "1", "text": "a"}],
    }, headers=AUTH)
    r = client.get("/sync/collections", headers=AUTH)
    assert r.status_code == 200
    cols = r.json()["collections"]
    assert any(c["collection"] == "main" for c in cols)


def test_sync_export_json_rows_carry_vector(client):
    client.post("/jobs", json={
        "collection": "main",
        "records": [{"id": "1", "text": "a"}],
    }, headers=AUTH)
    r = client.get("/sync/export", params={"collection": "main", "since": 0},
                   headers=AUTH)
    assert r.status_code == 200
    page = r.json()
    assert page["count"] >= 1
    for row in page["rows"]:
        assert "vector" in row and row["vector"] is not None
        assert row["model"] == "embeddinggemma"
        assert row["dim"] == 768
    assert page["done"] is True


def test_sync_export_arrow(client):
    client.post("/jobs", json={
        "collection": "main",
        "records": [{"id": "1", "text": "a"}],
    }, headers=AUTH)
    r = client.get("/sync/export", params={"collection": "main", "format": "arrow"},
                   headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/vnd.apache.arrow.stream")
    assert r.headers["X-Count"] == "1"
    # parseable
    import pyarrow as pa
    tbl = pa.ipc.open_stream(pa.BufferReader(r.content)).read_all()
    assert tbl.num_rows == 1
    assert "vector" in tbl.column_names


def test_sync_export_bad_format(client):
    r = client.get("/sync/export", params={"collection": "main", "format": "xml"},
                   headers=AUTH)
    assert r.status_code == 400


def test_sync_export_limit_clamped(client):
    # request a huge limit; should be clamped to EXPORT_LIMIT_MAX (5000), not 400
    r = client.get("/sync/export", params={"collection": "main", "limit": 999999},
                   headers=AUTH)
    assert r.status_code == 200


# stats
def test_stats(client):
    r = client.get("/stats", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["default_model"] == "embeddinggemma"
    assert body["gpu"] == "L4"
    assert "loaded_models" in body
    assert body["total_rows"] >= 0


def test_graph_import_and_export(client):
    payload = {
        "graph_id": "vault-main",
        "entities": [
            {
                "id": "note:1",
                "name": "30-resources/example.md",
                "type": "note",
                "aliases": "Example",
                "summary": "",
                "collection_ids": '["30-resources/example.md"]',
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
        "relations": [
            {
                "id": "edge:1",
                "from_entity_id": "note:1",
                "to_entity_id": "note:2",
                "relation_type": "forward",
                "fact": "example links to other",
                "fact_strength": 1.0,
                "source_entry_ids": '["30-resources/example.md"]',
                "valid_at": "2026-01-01T00:00:00Z",
                "expired_at": "",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
    }
    r = client.post("/graph/import", json=payload, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["entities"] == 1
    assert body["relations"] == 1

    r = client.get("/graph/export", params={"graph_id": "vault-main"}, headers=AUTH)
    assert r.status_code == 200
    exported = r.json()
    assert exported["graph_id"] == "vault-main"
    assert len(exported["entities"]) == 1
    assert len(exported["relations"]) == 1


def _make_graph_client():
    """Client with graph-artifact + manifest-job callbacks wired (Modal-free)."""
    cfg = types.SimpleNamespace(
        DEFAULT_MODEL="embeddinggemma",
        DEFAULT_DIM=None,
        CACHE_DIR="/cache",
        BATCH_SIZE=64,
        JOB_LIST_LIMIT=50,
        EXPORT_LIMIT_DEFAULT=500,
        EXPORT_LIMIT_MAX=5000,
        GPU="L4",
        API_TOKEN="testtoken",
    )
    jobs = MemoryJobStore()
    staged: list[dict] = []

    def stage_graph_artifact(family, kind, rows, sha256):
        import hashlib
        artifact_id = hashlib.sha256(rows.encode("utf-8")).hexdigest()
        staged.append({"artifact_id": artifact_id, "family": family, "kind": kind,
                       "bytes": len(rows.encode("utf-8")), "rows": len([l for l in rows.splitlines() if l.strip()]),
                       "sha256": artifact_id})
        return staged[-1]

    def create_graph_job(graph_id, mode, artifact_manifest=None):
        job_id = "job-1"
        doc = {
            "id": job_id, "kind": "graph-rebuild", "status": "queued", "phase": "queued",
            "graph_id": graph_id, "mode": mode, "source": "artifact" if artifact_manifest else "upload",
            "source_manifest": (artifact_manifest or {}).get("artifacts", []),
            "planned_entities": (artifact_manifest or {}).get("planned_entities", 0),
            "planned_relations": (artifact_manifest or {}).get("planned_relations", 0),
            "planned_batches": len((artifact_manifest or {}).get("artifacts", [])),
            "completed_batches": 0,
            "processed_entities": 0, "processed_relations": 0,
            "last_progress_at": "2026-01-01T00:00:00Z",
        }
        jobs.write(job_id, doc)
        return doc

    app = build_app(
        cfg=cfg,
        store=store,
        embed_fn=_fake_embed,
        get_spec_fn=get_spec,
        registry=enabled_registry(),
        read_job=jobs.read,
        write_job=jobs.write,
        list_job_docs=jobs.list,
        spawn_job=lambda *a, **k: types.SimpleNamespace(object_id="call-1"),
        create_graph_job=create_graph_job,
        stage_graph_artifact=stage_graph_artifact,
        read_graph_job=jobs.read,
        list_graph_jobs=lambda limit: [],
        loaded_models=lambda: ["embeddinggemma"],
    )
    return TestClient(app), staged


def test_graph_artifact_upload_returns_server_issued_id():
    client, _staged = _make_graph_client()
    rows = '{"id":"e1","name":"A"}\n'
    import hashlib
    sha = hashlib.sha256(rows.encode("utf-8")).hexdigest()
    r = client.post("/graph/artifacts", json={"family": "semantic", "kind": "entities", "rows": rows, "sha256": sha}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["artifact_id"] == sha
    assert body["kind"] == "entities"
    assert body["family"] == "semantic"
    assert body["rows"] == 1


def test_graph_artifact_upload_requires_auth():
    client, _ = _make_graph_client()
    r = client.post("/graph/artifacts", json={"family": "semantic", "kind": "entities", "rows": "x", "sha256": "y"})
    assert r.status_code == 401


def test_graph_job_create_with_artifact_manifest():
    client, _ = _make_graph_client()
    manifest = {
        "artifacts": [
            {"artifact_id": "a1", "family": "semantic", "kind": "entities", "bytes": 10, "rows": 5, "sha256": "x"},
        ],
        "planned_entities": 5,
        "planned_relations": 0,
    }
    r = client.post("/graph/jobs", json={"graph_id": "vault-main", "mode": "replace", "manifest": manifest}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "artifact"
    assert body["phase"] == "queued"
    assert body["planned_entities"] == 5
    assert body["planned_batches"] == 1