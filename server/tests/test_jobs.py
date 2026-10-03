"""Tests for the file-backed job lister in app.py.

Guarded by ``importorskip("modal")`` because ``app`` imports the Modal SDK;
the test runs in the full image / `--with modal` envs and skips otherwise.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("modal")

import app


def test_list_job_docs_orders_by_mtime_and_limits(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "_jobs"
    jobs_dir.mkdir()

    # Write 5 job docs with strictly increasing mtimes; job4 is newest.
    for i in range(5):
        p = jobs_dir / f"job{i}.json"
        p.write_text(json.dumps({"id": f"job{i}", "status": "done"}))
        os.utime(p, (1_000_000 + i * 10, 1_000_000 + i * 10))

    # A corrupt file that is newest of all: it must be skipped, not crash,
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
