"""Tests for the file-backed job lister in app.py.

Guarded by ``importorskip("modal")`` because ``app`` imports the Modal SDK;
the test runs in the full image / `--with modal` envs and skips otherwise.
"""

from __future__ import annotations

import json
import os
import types

import pytest

pytest.importorskip("modal")

import app  # noqa: E402 - import after importorskip


def test_list_job_docs_orders_by_mtime_and_limits(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "_jobs"
    jobs_dir.mkdir()

    # Write 5 job docs with strictly increasing mtimes; job4 is newest.
    for i in range(5):
        p = jobs_dir / f"job{i}.json"
        p.write_text(json.dumps({"id": f"job{i}", "status": "done"}))
        os.utime(p, (1_000_000 + i * 10, 1_000_000 + i * 10))

    # A corrupt file that is newest of all — it must be skipped, not crash,
    # and must not consume a slot in the returned docs.
    bad = jobs_dir / "job_bad.json"
    bad.write_text("{not json")
    os.utime(bad, (2_000_000, 2_000_000))

    monkeypatch.setattr(app.config, "JOBS_DIR", str(jobs_dir))
    monkeypatch.setattr(app.vectors_volume, "reload", lambda *a, **k: None, raising=False)

    docs = app._list_job_docs(limit=3)

    # Newest-first by mtime, corrupt file skipped, limited to 3 valid docs.
    assert [d["id"] for d in docs] == ["job4", "job3", "job2"]


def test_list_job_docs_missing_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(app.config, "JOBS_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(app.vectors_volume, "reload", lambda *a, **k: None, raising=False)
    assert app._list_job_docs(limit=10) == []


def _stage(monkeypatch, tmp_path, family="semantic", kind="entities", rows="", sha256=""):
    monkeypatch.setattr(app.config, "GRAPH_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(app.vectors_volume, "commit", lambda *a, **k: None, raising=False)
    return app._stage_graph_artifact(family, kind, rows, sha256)


def test_stage_graph_artifact_issues_content_addressed_id(tmp_path, monkeypatch):
    rows = '{"id":"e1","name":"A"}\n{"id":"e2","name":"B"}\n'
    import hashlib
    expected_id = hashlib.sha256(rows.encode("utf-8")).hexdigest()
    out = _stage(monkeypatch, tmp_path, rows=rows, sha256=expected_id)
    assert out["artifact_id"] == expected_id
    assert out["kind"] == "entities"
    assert out["family"] == "semantic"
    assert out["rows"] == 2
    assert out["bytes"] == len(rows.encode("utf-8"))
    # Resolved under the fixed root, not a client path.
    assert (tmp_path / "artifacts" / f"{expected_id}.jsonl").exists()


def test_stage_graph_artifact_is_idempotent(tmp_path, monkeypatch):
    rows = '{"id":"e1","name":"A"}\n'
    import hashlib
    sha = hashlib.sha256(rows.encode("utf-8")).hexdigest()
    first = _stage(monkeypatch, tmp_path, rows=rows, sha256=sha)
    second = _stage(monkeypatch, tmp_path, rows=rows, sha256=sha)
    assert first["artifact_id"] == second["artifact_id"]


def test_stage_graph_artifact_rejects_sha_mismatch(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _stage(monkeypatch, tmp_path, rows='{"id":"e1"}\n', sha256="deadbeef")


def test_stage_graph_artifact_rejects_bad_kind(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="kind"):
        _stage(monkeypatch, tmp_path, kind="bogus", rows='{"id":"e1"}\n')


def test_create_graph_job_artifact_manifest_sets_phase_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app.config, "JOBS_DIR", str(tmp_path / "_jobs"))
    monkeypatch.setattr(app.vectors_volume, "commit", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app.vectors_volume, "reload", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(app, "rebuild_graph_job", types.SimpleNamespace(
        spawn=lambda job_id: types.SimpleNamespace(object_id=f"call-{job_id}")
    ))
    artifact_manifest = {
        "artifacts": [
            {"artifact_id": "a1", "family": "semantic", "kind": "entities", "bytes": 10, "rows": 5, "sha256": "x"},
            {"artifact_id": "a2", "family": "note-link", "kind": "relations", "bytes": 10, "rows": 3, "sha256": "y"},
        ],
        "planned_entities": 5,
        "planned_relations": 3,
    }
    doc = app._create_graph_job("vault-main", "replace", artifact_manifest)
    assert doc["source"] == "artifact"
    assert doc["phase"] == "queued"
    assert doc["planned_entities"] == 5
    assert doc["planned_relations"] == 3
    assert doc["planned_batches"] == 2
    assert doc["completed_batches"] == 0
    assert "last_progress_at" in doc


def test_graph_artifact_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(app.config, "GRAPH_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    with pytest.raises(ValueError, match="invalid graph artifact id"):
        app._graph_artifact_path("../../etc/passwd")
    with pytest.raises(ValueError, match="invalid graph artifact id"):
        app._graph_artifact_path("a1")
    # A valid 64-hex id resolves under the fixed root.
    valid = "a" * 64
    assert app._graph_artifact_path(valid) == f"{tmp_path}/artifacts/{valid}.jsonl"
