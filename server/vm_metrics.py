"""App-level metrics: a stdlib-only VictoriaMetrics drop-in for the embedding server.

Vendored verbatim from ``toolkit/metrics.py`` in the modal-toolkit repo (the
canonical module; same API). Per-repo vendoring is deliberate: Modal deploy
trees are self-contained single-file trees and cannot import across repos.
The only delta from the canonical copy is the env var name:
``MODAL_EMBED_VM_URL`` (fallback ``http://localhost:8428``).

One writer pushes events, counters, and rates into **your own** VictoriaMetrics
(or InfluxDB; the line protocol is compatible with both).

Single-node VictoriaMetrics accepts PUT or POST with Influx line protocol at:

    {url}/api/v2/write?precision=s&org={org}&bucket={bucket}

``org``/``bucket`` are ignored by single-node VM (kept as Influx placeholders);
a 204 means the point landed. Timestamps are in seconds and mark the **reading**,
not the send: pass ``ts`` explicitly when the event happened earlier. 204 or
not, the writer never retries internally: a failed write returns ``False`` and
the *caller* decides the retry policy.

Naming convention for points: ``<package>_<event>``.
``embedding_request``, ``embedding_infer_seconds``, ``embedding_job_done``.

    >>> from vm_metrics import write_metric, MetricTimer, bump
    >>> bump("embedding_request")                       # stateless counter point
    >>> with MetricTimer("embedding_infer_seconds"):
    ...     ...  # writes elapsed seconds on exit

Only stdlib is imported at module top so the module works inside Modal
containers without extra image dependencies.
"""

# ruff: noqa: PYI034 -- vendored verbatim from modal-toolkit's canonical
# toolkit/metrics.py; the ``MetricTimer.__enter__`` return annotation stays
# ``MetricTimer`` (not ``Self``) to match the canonical copy exactly.
from __future__ import annotations

import os
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

ENV_URL = "MODAL_EMBED_VM_URL"
DEFAULT_URL = "http://localhost:8428"

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _escape_tag(value: str) -> str:
    """Escape a tag key or value for line protocol: backslash, comma, equals."""
    return value.replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=")


def _escape_measurement(name: str) -> str:
    """Measurement names keep it boring: spaces become underscores."""
    return name.replace(" ", "_")


def line_protocol(
    measurement: str,
    value: float,
    tags: dict[str, str] | None = None,
    ts: int | None = None,
) -> str:
    """Build one line: ``measurement,tag=v value timestamp_seconds``."""
    name = _escape_measurement(str(measurement))
    tag_part = ""
    if tags:
        tag_part = "," + ",".join(
            f"{_escape_tag(str(k))}={_escape_tag(str(v))}" for k, v in tags.items()
        )
    stamp = int(time.time()) if ts is None else int(ts)
    return f"{name}{tag_part} {value} {stamp}"


def effective_url() -> tuple[str, str]:
    """The VM URL in effect + where it came from (``env MODAL_EMBED_VM_URL`` or ``default``)."""
    env = os.environ.get(ENV_URL)
    if env:
        return env.rstrip("/"), f"env {ENV_URL}"
    return DEFAULT_URL, "default"


class VMWriter:
    """Writes single points to a VictoriaMetrics (or InfluxDB) write endpoint.

    Never raises and never retries: ``write`` returns True on 204, False on
    anything else (with a one-line warning on stderr). A metrics failure must
    never fail the host operation.
    """

    def __init__(
        self,
        url: str | None = None,
        org: str = "-",
        bucket: str = "-",
        timeout: float = 5.0,
    ) -> None:
        resolved = (url or os.environ.get(ENV_URL) or DEFAULT_URL).rstrip("/")
        self.url = resolved
        self.org = org
        self.bucket = bucket
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        query = urllib.parse.urlencode(
            {"precision": "s", "org": self.org, "bucket": self.bucket}
        )
        return f"{self.url}/api/v2/write?{query}"

    def write(
        self,
        measurement: str,
        value: float,
        tags: dict[str, str] | None = None,
        ts: int | None = None,
    ) -> bool:
        line = line_protocol(measurement, value, tags=tags, ts=ts)
        endpoint = self.endpoint
        req = urllib.request.Request(endpoint, data=line.encode("utf-8"), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 204:
                    return True
                print(
                    f"mtk metrics: write to {self.url} got HTTP {resp.status}",
                    file=sys.stderr,
                )
                return False
        except OSError as exc:
            # URLError subclasses OSError; this covers refused/DNS/timeout alike.
            print(f"mtk metrics: write to {self.url} failed: {exc}", file=sys.stderr)
            return False


_writer: VMWriter | None = None


def write_metric(
    measurement: str,
    value: float,
    tags: dict[str, str] | None = None,
    ts: int | None = None,
) -> bool:
    """Write one point via a lazily-created module singleton (reads the env once)."""
    global _writer
    if _writer is None:
        _writer = VMWriter()
    return _writer.write(measurement, value, tags=tags, ts=ts)


def bump(measurement: str, tags: dict[str, str] | None = None, n: float = 1) -> bool:
    """Stateless counter point: VM does the summation, the client keeps nothing."""
    return write_metric(measurement, n, tags=tags)


class MetricTimer:
    """Time a block (context manager) or a function (decorator), then write elapsed seconds.

    Usage::

        with MetricTimer("embedding_infer_seconds"):
            ...

        @MetricTimer("embedding_request_seconds")
        def job(): ...
    """

    def __init__(self, key: str, tags: dict[str, str] | None = None) -> None:
        self.key = key
        self.tags = tags
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> MetricTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        write_metric(self.key, self.elapsed, tags=self.tags)

    def __call__(self, fn: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with MetricTimer(self.key, self.tags):
                return fn(*args, **kwargs)

        return wrapper
