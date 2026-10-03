# Vendored from modal-shared-libs (libs/hooks.py) by `mtk libs sync`:
# family repo: modal-embedding-server, md5 2a27ae48220c6c4c7df2102d7783f885. Canonical source of truth; report
# fixes there, not here.
"""Lifecycle hooks: the family's extension seam.

The contract every package shares (the integration model lives in each
package's docs; this module is the mechanism):

- **Closed tag set.** A package declares ITS OWN tags at configure time
  (`configure(tags)`); registering an undeclared tag refuses. The tag
  list changes only in that package's releases - the contract is
  versioned like the wire.
- **Multiple registrations coexist** per tag (call order: registration
  order). Lanes build on other lanes' hooks.
- **Errors are contained and reported.** A hook failure degrades that
  lane, never the server; every fire records its errors per tag
  (`last_errors(tag)`).
- **Two registration forms**: the decorator `@on(tag)` and the dynamic
  `register(tag, fn)`; both share one registry.
- **Stdlib only.** No Modal, no framework; any package / overlay /
  plain Python consumer can use it anywhere.

A package fires at stable lifecycle points and documents its tags in
its own docs (families so far: boot.pre/boot.post/write.post,
request.pre/request.post, job.pre/job.post, stage boundaries).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

Registry = dict[str, list[Callable[..., Any]]]


class Hooks:
    """One hook space per consumer (package, overlay, or test)."""

    def __init__(self, tags: tuple[str, ...], name: str = "hooks") -> None:
        self._tags = tuple(tags)
        self.name = name
        self._registry: Registry = {tag: [] for tag in self._tags}
        self._last_errors: dict[str, list[str]] = {tag: [] for tag in self._tags}

    # -- registration ----------------------------------------------------------

    @property
    def tags(self) -> tuple[str, ...]:
        return self._tags

    def register(self, tag: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Add a handler for a declared tag; returns fn (doubles as a manual form)."""
        self._require_tag(tag)
        self._registry[tag].append(fn)
        return fn

    def unregister(self, tag: str, fn: Callable[..., Any]) -> None:
        """Remove a handler; a missing handler is a no-op (idempotent cleanup)."""
        self._require_tag(tag)
        with contextlib.suppress(ValueError):
            self._registry[tag].remove(fn)

    def on(self, tag: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form: `@hooks_instance.on("request.pre")`.

        Validates the tag EAGERLY (at decoration time, not apply time).
        """

        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            return self.register(tag, fn)

        self._require_tag(tag)
        return _wrap

    def _require_tag(self, tag: str) -> None:
        if tag not in self._registry:
            raise ValueError(f"{self.name}: unknown hook tag {tag!r}; tags: {sorted(self._registry)}")

    # -- firing ----------------------------------------------------------------

    def fire(self, tag: str, *args: Any) -> list[str]:
        """Run a tag's handlers in order; contain + record errors; return error strings."""
        self._require_tag(tag)
        errors: list[str] = []
        for fn in self._registry[tag]:
            try:
                fn(*args)
            except Exception as exc:  # contained: one lane's failure never blocks the host
                errors.append(f"{getattr(fn, '__name__', 'hook')}: {type(exc).__name__}: {str(exc)[:150]}")
        self._last_errors[tag] = errors
        return errors

    def last_errors(self, tag: str) -> list[str]:
        """The latest fire()'s errors for a tag (for health-style reporting)."""
        self._require_tag(tag)
        return list(self._last_errors[tag])

    def registrations(self, tag: str) -> list[Callable[..., Any]]:
        """Current handlers for a tag (copy; for introspection/health)."""
        self._require_tag(tag)
        return list(self._registry[tag])

    # -- test seam -------------------------------------------------------------

    def clear(self, tag: str | None = None) -> None:
        """Drop registrations (one tag or all) + error memory."""
        if tag is None:
            for known in self._tags:
                self._registry[known] = []
                self._last_errors[known] = []
            return
        self._require_tag(tag)
        self._registry[tag] = []
        self._last_errors[tag] = []


@contextlib.contextmanager
def registered(hooks: Hooks, tag: str, fn: Callable[..., Any]):
    """Scoped registration for tests/one-off wiring; unregisters on exit."""
    hooks.register(tag, fn)
    try:
        yield
    finally:
        hooks.unregister(tag, fn)
