"""Tests for the vendored stdlib-only VictoriaMetrics writer (server/vm_metrics.py).

Subset of the canonical toolkit/tests/test_metrics.py, adapted for the vendored
copy: the env var is MODAL_EMBED_VM_URL and the module imports as a flat
server module (conftest puts server/ on sys.path).
"""

from __future__ import annotations

import ast
import http.server
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest

import vm_metrics  # vendored flat module; conftest puts server/ on sys.path


class _CaptureHandler(http.server.BaseHTTPRequestHandler):
    """Minimal VM stand-in: 204 for every POST, recording what arrived."""

    captured: ClassVar[list[tuple[str, bytes]]] = []
    headers_seen: ClassVar[list[dict]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).captured.append((self.path, body))
        type(self).headers_seen.append(dict(self.headers))
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args: object) -> None:  # keep test output quiet
        pass


@pytest.fixture()
def vm_server():
    """Spin a real stdlib HTTP server on a random port; yield (url, captured)."""
    _CaptureHandler.captured = []
    _CaptureHandler.headers_seen = []
    server = http.server.HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    yield url, _CaptureHandler.captured
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def reset_singleton():
    vm_metrics._writer = None
    yield
    vm_metrics._writer = None


def dead_port_url() -> str:
    """A URL guaranteed to have nothing listening (bind-then-close)."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = server.server_port
    server.server_close()
    return f"http://127.0.0.1:{port}"


def test_line_protocol_escapes_tag_values() -> None:
    line = vm_metrics.line_protocol("m", 1, tags={"a": "x,y", "b": "k=v", "c": "back\\slash", "d": "plain"}, ts=100)
    assert line == "m,a=x\\,y,b=k\\=v,c=back\\\\slash,d=plain 1 100"


def test_line_protocol_escapes_measurement_spaces() -> None:
    line = vm_metrics.line_protocol("embedding job done", 2, ts=50)
    assert line == "embedding_job_done 2 50"


def test_write_passes_explicit_ts_through(vm_server) -> None:
    url, captured = vm_server
    writer = vm_metrics.VMWriter(url=url)
    assert writer.write("embedding_request", 1, tags={"mode": "infer"}, ts=1699999999) is True
    assert len(captured) == 1
    path, body = captured[0]
    assert body == b"embedding_request,mode=infer 1 1699999999"
    assert path.startswith("/api/v2/write?")
    assert "precision=s" in path and "org=-" in path and "bucket=-" in path


def test_write_returns_false_and_never_raises_on_connection_failure(capsys) -> None:
    writer = vm_metrics.VMWriter(url=dead_port_url(), timeout=1.0)
    assert writer.write("embedding_probe", 1) is False
    assert "mtk metrics:" in capsys.readouterr().err


def test_singleton_honors_env_set_before_first_write(vm_server) -> None:
    url, captured = vm_server
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(vm_metrics.ENV_URL, url)
        vm_metrics._writer = None  # create-after-env-set ordering
        assert vm_metrics.write_metric("embedding_probe", 1) is True
    assert len(captured) == 1
    assert url in vm_metrics._writer.url


def test_writer_sends_basic_auth_header_from_init(vm_server) -> None:
    """VM servers fronted by raw -httpAuth.* flags need the Authorization
    header; the writer without auth sent none (silent 401s)."""
    url, _captured = vm_server
    import base64

    writer = vm_metrics.VMWriter(url=url, auth="op:seekrit")
    assert writer.write("embedding_probe", 1) is True
    seen = _CaptureHandler.headers_seen[-1]
    token = base64.b64encode(b"op:seekrit").decode("ascii")
    assert seen.get("Authorization") == f"Basic {token}"


def test_writer_auth_env_fallback_and_absence(vm_server, monkeypatch) -> None:
    url, _captured = vm_server
    monkeypatch.setenv(vm_metrics.ENV_AUTH, "envuser:envpass")
    vm_metrics.VMWriter(url=url).write("embedding_probe", 1)
    seen = _CaptureHandler.headers_seen[-1]
    assert "Authorization" in seen
    monkeypatch.delenv(vm_metrics.ENV_AUTH, raising=False)
    vm_metrics.VMWriter(url=url).write("embedding_probe", 1)
    assert "Authorization" not in _CaptureHandler.headers_seen[-1]


def test_effective_url_defaults_then_env(vm_server) -> None:
    url, _captured = vm_server
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv(vm_metrics.ENV_URL, raising=False)
        assert vm_metrics.effective_url() == (vm_metrics.DEFAULT_URL, "default")
        mp.setenv(vm_metrics.ENV_URL, url + "/")
        assert vm_metrics.effective_url() == (url, f"env {vm_metrics.ENV_URL}")


def test_module_top_level_imports_are_stdlib_only() -> None:
    src = Path(vm_metrics.__file__).read_text()
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    non_stdlib = imported - sys.stdlib_module_names
    assert non_stdlib == set(), f"non-stdlib imports at module top: {non_stdlib}"


def test_env_var_is_embed_namespaced() -> None:
    # Vendoring delta guard: the only behavioral divergences from the canonical
    # toolkit/metrics.py are the env var names.
    assert vm_metrics.ENV_URL == "MODAL_EMBED_VM_URL"
    assert vm_metrics.ENV_AUTH == "MODAL_EMBED_VM_AUTH"
    assert vm_metrics.DEFAULT_URL == "http://localhost:8428"
