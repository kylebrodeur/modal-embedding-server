"""Lifecycle tests for modal-embedding-server's hooks wiring.

The instance lives in ``web.py`` (module level; the fire sites read
``hooks.fire(...)`` next to the lifecycle points), so the route handlers fire
it directly and tests can register against the SAME shared instance. No
Modal SDK, no GPU: the store is real (temp-dir LanceDB from ``conftest``)
and the embed function is a fake, exactly like ``test_api.py`` - tests here
build their client through ``test_api._make_client`` and skip auth with its
bearer-token convention.

Guarantees under test:
- the shared instance declares the closed tag set (documented in web.py)
- every declared tag has a real fire site (whitespace-tolerant scan)
- an embed request fires request.pre -> embed.post -> request.post, in order
- job.pre fires at submit; job.post when the job reaches a terminal state
- hook errors are CONTAINED: a raising hook never breaks a request, healthy
  handlers on the same tag still run, and errors land in ``last_errors(tag)``

Isolation: the tests mutate the module-level instance's registry, so every
test registers through the ``registered(...)`` helper (unregisters on exit)
or the ``fired`` fixture (clears everything after).
"""

from __future__ import annotations

import re
import tempfile
import types
from pathlib import Path

import pytest
from test_api import _make_client

import web
from libs.hooks import registered
from web import hooks

AUTH = {"Authorization": "Bearer testtoken"}

EXPECTED_TAGS = (
    "request.pre",
    "request.post",
    "embed.post",
    "job.pre",
    "job.post",
)


@pytest.fixture()
def fired():
    """Record (tag, args) per fire; drop registrations + error memory after."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    for tag in hooks.tags:
        hooks.register(
            tag,
            lambda *args, _tag=tag, _calls=calls: _calls.append((_tag, args)),
        )
    yield calls
    hooks.clear()


class LaneExploded(RuntimeError):
    pass


def _boom(*_args) -> None:
    raise LaneExploded("lane exploded")


def test_instance_declares_the_closed_tag_set() -> None:
    assert hooks.name == "modal-embedding-server"
    assert hooks.tags == EXPECTED_TAGS
    # The module docstring documents the closed set, per the family contract.
    for tag in EXPECTED_TAGS:
        assert tag in web.__doc__, f"{tag} missing from web.py's docstring"


def test_fire_sources_reference_every_tag() -> None:
    """Every declared tag must have a fire site next to a real lifecycle point.

    ``job.post`` fires in the bulk worker (``app.py``); every other tag fires
    in ``web.py``. The scan is whitespace-tolerant: source formatting is not
    the contract, a fire at the boundary is.
    """
    repo = Path(web.__file__).resolve().parent
    sources = {
        "web": (repo / "web.py", tuple(t for t in EXPECTED_TAGS if t != "job.post")),
        "app": (repo / "app.py", ("job.pre", "job.post")),
    }
    for path, tags in sources.values():
        text = path.read_text(encoding="utf-8")
        for tag in tags:
            assert re.search(r'hooks\.fire\(\s*"' + re.escape(tag) + '"', text), (
                f"{tag} not fired in {path.name}"
            )


def test_embed_fires_in_family_order(fired) -> None:
    """An embed request fires request.pre -> embed.post -> request.post."""
    calls = fired
    c, _jobs = _make_client()
    r = c.post(
        "/embed", json={"texts": ["hello", "world"], "task": "query"}, headers=AUTH
    )
    assert r.status_code == 200
    assert [tag for tag, _args in calls] == [
        "request.pre",
        "embed.post",
        "request.post",
    ]
    assert calls[0][1] == ("POST", "/embed")  # request.pre payload
    assert calls[1][1] == ({"path": "/embed", "count": 2},)  # embed.post payload
    assert calls[2][1] == ("POST", "/embed", 200)  # request.post payload


def test_job_tags_fire_at_job_boundaries(fired, monkeypatch) -> None:
    """job.pre at accept; job.post when the job worker reaches a terminal state.

    The web route accepts the job (request.pre -> job.pre -> request.post);
    then this test runs the worker body in-process via ``embed_batch.fn``
    (the Modal shell stripped, the real worker logic) which fires job.post.
    """
    calls = fired
    c, _jobs = _make_client()
    r = c.post(
        "/jobs",
        json={
            "collection": "main",
            "records": [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}],
        },
        headers=AUTH,
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status = c.get(f"/jobs/{job_id}", headers=AUTH)
    assert status.status_code == 200

    fire_seq = [tag for tag, _args in calls]
    assert fire_seq == [
        "request.pre",
        "job.pre",
        "request.post",  # POST /jobs
        "request.pre",
        "request.post",  # GET /jobs/{job_id}
    ]
    assert [args[0]["job_id"] for tag, args in calls if tag == "job.pre"] == [job_id]

    # Now run the worker in-process: fires job.post(done) at its boundary.
    _run_embed_batch_in_process(job_id, monkeypatch)
    assert [args[0] for tag, args in calls if tag == "job.post"] == [
        {"job_id": job_id, "status": "done"},
    ]


def _fake_worker_embed(spec, texts, task, dim, cache_dir, batch_size=64):
    """Laptop-deterministic fake matching test_api._fake_embed's shape."""
    return [[(i + 1) / dim for i in range(dim)] for _ in texts]


def _run_embed_batch_in_process(job_id, monkeypatch) -> None:
    """Invoke the worker body directly, skipping the Modal shell.

    ``embed_batch._raw_f_`` is the Python callable Modal wraps into a remote
    function; calling it runs embed_batch's real body (job.post fires there)
    against the real (temp-dir) store + the package's job-dir helpers. The
    heavy ML import inside ``embedders.embed`` is swapped for the fake above.
    """
    import app as app_module

    assert app_module.hooks is hooks  # same shared instance, per the contract

    # embedders.embed would import sentence-transformers (laptop tests run
    # without the ML stack); monkeypatch the lazy import's attribute instead.
    import embedders

    monkeypatch.setattr(embedders, "embed", _fake_worker_embed)

    # The worker persists job docs under config.JOBS_DIR and commits the
    # vectors volume; point both at throwaway stand-ins for the test run.
    app_module.config.JOBS_DIR = tempfile.mkdtemp()
    app_module.vectors_volume = types.SimpleNamespace(
        commit=lambda: None, reload=lambda: None
    )
    app_module.embed_batch._raw_f_(
        job_id,
        "main",
        "embeddinggemma",
        None,
        {"collection": "main", "records": [{"id": "9", "text": "w"}]},
    )


def test_raising_hook_is_contained_and_request_succeeds() -> None:
    """A raising hook degrades that lane only; the route still responds 200."""
    with (
        registered(hooks, "request.pre", _boom),
        registered(hooks, "embed.post", _boom),
        registered(hooks, "request.post", _boom),
    ):
        c, _jobs = _make_client()
        r = c.post("/embed", json={"texts": ["hi"], "task": "query"}, headers=AUTH)
        assert r.status_code == 200
        assert r.json()["model"] == "embeddinggemma"
        for tag in ("request.pre", "embed.post", "request.post"):
            errs = hooks.last_errors(tag)
            assert len(errs) == 1, f"{tag} should record its lane's error"
            assert "LaneExploded: lane exploded" in errs[0]
    # After unregister, the same request is clean again.
    c, _jobs = _make_client()
    r = c.post("/embed", json={"texts": ["hi"], "task": "query"}, headers=AUTH)
    assert r.status_code == 200
    assert hooks.last_errors("request.pre") == []


def test_unknown_tag_refused_by_the_shared_instance() -> None:
    with pytest.raises(ValueError, match="modal-embedding-server: unknown hook tag"):
        hooks.register("embed.post.post", lambda payload: None)


def test_last_errors_mirror_the_latest_fire() -> None:
    """fire returns [error strings]; last_errors(tag) mirrors that fire."""
    calls: list[tuple] = []
    with (
        registered(hooks, "job.pre", _boom),
        registered(hooks, "job.pre", lambda payload: calls.append(payload)),
    ):
        errors = hooks.fire("job.pre", {"job_id": "abc"})
        assert len(errors) == 1
        assert "LaneExploded: lane exploded" in errors[0]
        assert hooks.last_errors("job.pre") == errors
        # The healthy hook on the same tag still ran.
        assert calls == [{"job_id": "abc"}]
    hooks.clear()
    assert hooks.last_errors("job.pre") == []
