"""Shared pytest config: put the ``server/`` dir on ``sys.path`` so the
modules (``config``, ``embedders``, ``store``, ``web``) import like they do
inside the Modal image, and point the store at a per-test temp Volume dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

# server/ is the parent of tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import pytest


@pytest.fixture(autouse=True)
def _tmp_vectors(tmp_path, monkeypatch):
    """Each test gets its own empty LanceDB dir so tests are isolated."""
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    monkeypatch.setattr(config, "VECTORS_DIR", str(vectors))
    # store reads config.VECTORS_DIR via VECTORS_DIR() indirection at call time.
    yield tmp_path