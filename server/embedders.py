"""Pluggable, config-driven embedder registry.

EmbeddingGemma is the canonical model for this server, but the architecture
is deliberately multi-model: every vector is tagged with the model + dimension
that produced it, and the LanceDB tables on the Volume are namespaced by
``{model}__{dim}``. That lets us trial alternative embedders (Qwen3, BGE-M3,
Nomic, …) side by side without corrupting the canonical EmbeddingGemma space,
and it keeps the local sync honest — local only pulls tables whose model+dim
it can actually use.

Backends
--------
A model may be served by different *backends* that all produce the **same
model's** vectors (sanctioned compute set per decision log D7):

- ``sentence-transformers`` — load weights on the Modal GPU (self-host an
  open-weight model). Requires a GPU unless the model is tiny.
- ``ollama`` — proxy to a local or Ollama-Cloud host. No GPU work here.
- ``hf`` — Hugging Face Inference API / dedicated endpoint. No GPU work here.

GPU presence is config-driven (``config.GPU``); when the canonical model is
served via Ollama or HF, the service owns the store/bulk/sync and proxies
embeddings — the GPU/SentenceTransformer path stays available for
self-hosting open-weight models.

Config-driven registry
-----------------------
The built-in :data:`REGISTRY` is a set of defaults. An optional JSON registry
file (``MODAL_EMBED_REGISTRY_FILE``) is merged *over* those defaults at import time, so
adding a brand-new embedder requires only a config entry (+ an HF token if
gated) — **no code change**. See :func:`build_registry`.

This module stays importable without the heavy ML stack: ``torch`` /
``sentence-transformers`` / ``huggingface_hub`` are imported lazily inside the
backend functions, so tests run on a laptop.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

# Local import avoids a circular dependency at runtime (embedders ← config).
import config

Task = Literal["query", "document"]
Backend = Literal["sentence-transformers", "ollama", "hf", "openai-compatible"]


@dataclass(frozen=True)
class EmbedderSpec:
    """Static description of an embedding model.

    Frozen so registry entries are hashable / safe to share across threads.
    Use :func:`replace` (or the registry merge) to derive a variant.
    """

    key: str
    """Short stable id used in the API and as part of the table name."""

    hf_id: str
    """HuggingFace repo id (used by the ``sentence-transformers`` and ``hf``
    backends; ignored by ``ollama`` unless ``ollama_model`` is unset)."""

    native_dim: int
    """Full output dimension of the model."""

    backend: Backend = "sentence-transformers"
    """Which compute backend produces this model's vectors."""

    matryoshka_dims: tuple[int, ...] = ()
    """Dimensions the model can be truncated to without retraining. Empty if
    the model does not support Matryoshka Representation Learning."""

    # EmbeddingGemma (and several others) want asymmetric prompts for queries
    # vs documents. We format the raw text into these templates before
    # encoding. ``{text}`` is substituted. ``None`` means "encode as-is".
    query_prompt: str | None = None
    document_prompt: str | None = None

    gated: bool = False
    """Whether the HF repo requires accepting a license / an HF token."""

    trust_remote_code: bool = False
    """Pass ``trust_remote_code=True`` to the loader (e.g. Nomic)."""

    enabled: bool = True
    """Whether the model is exposed via /models and accepted by /embed + /jobs.
    The registry merge flips this to disable a built-in model from config."""

    # ── backend-specific knobs (all optional) ────────────────────────────────
    ollama_model: str | None = None
    """Ollama model name for the ``ollama`` backend. Defaults to ``key``."""

    hf_inference_endpoint: str | None = None
    """Per-model HF Inference endpoint URL override (else the global base)."""

    notes: str = ""

    def resolve_dim(self, dim: int | None) -> int:
        """Validate a requested output dimension against this model.

        ``None`` → the model's native dim. Otherwise the dim must be the native
        dim or one of the Matryoshka-truncatable dims.
        """
        if dim is None:
            return self.native_dim
        if dim == self.native_dim:
            return dim
        if dim in self.matryoshka_dims:
            return dim
        allowed = (self.native_dim, *self.matryoshka_dims)
        raise ValueError(
            f"Model '{self.key}' cannot produce {dim}-dim vectors. "
            f"Allowed: {sorted(set(allowed))}."
        )

    def format(self, text: str, task: Task) -> str:
        prompt = self.query_prompt if task == "query" else self.document_prompt
        return prompt.format(text=text) if prompt else text

    def to_public(self) -> dict:
        """JSON-safe description for the ``GET /models`` registry response."""
        return {
            "key": self.key,
            "hf_id": self.hf_id,
            "backend": self.backend,
            "native_dim": self.native_dim,
            "matryoshka_dims": list(self.matryoshka_dims),
            "query_prompt": self.query_prompt,
            "document_prompt": self.document_prompt,
            "gated": self.gated,
            "trust_remote_code": self.trust_remote_code,
            "enabled": self.enabled,
            "notes": self.notes,
        }


# ── Built-in registry defaults ───────────────────────────────────────────────
# Official EmbeddingGemma prompt templates (see the model card). Using explicit
# templates keeps us independent of any one sentence-transformers helper.

_BUILTIN: dict[str, EmbedderSpec] = {
    "embeddinggemma": EmbedderSpec(
        key="embeddinggemma",
        hf_id="google/embeddinggemma-300m",
        native_dim=768,
        matryoshka_dims=(512, 256, 128),
        query_prompt="task: search result | query: {text}",
        document_prompt="title: none | text: {text}",
        gated=True,
        notes="Canonical model. 300M params, multilingual, Matryoshka.",
    ),
    # ── Alternatives to explore (not the canonical space) ─────────────────────
    "qwen3-0.6b": EmbedderSpec(
        key="qwen3-0.6b",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        native_dim=1024,
        query_prompt=(
            "Instruct: Given a search query, retrieve relevant passages\n"
            "Query: {text}"
        ),
        notes="Strong MTEB scores; instruction-tuned query prompt.",
    ),
    "bge-m3": EmbedderSpec(
        key="bge-m3",
        hf_id="BAAI/bge-m3",
        native_dim=1024,
        notes="Multilingual, long-context (8k). No asymmetric prompt needed.",
    ),
    "nomic-1.5": EmbedderSpec(
        key="nomic-1.5",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        native_dim=768,
        matryoshka_dims=(512, 256, 128, 64),
        query_prompt="search_query: {text}",
        document_prompt="search_document: {text}",
        trust_remote_code=True,
        notes="Matryoshka; requires trust_remote_code.",
    ),
    "minilm-l6": EmbedderSpec(
        key="minilm-l6",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        native_dim=384,
        notes="Legacy parity with the local transformers.js fallback.",
    ),
}


def _coerce_dim(d: Any) -> int:
    try:
        return int(d)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"native_dim must be an int, got {d!r}") from exc


def _coerce_dims(dims: Any) -> tuple[int, ...]:
    if dims is None:
        return ()
    if isinstance(dims, (list, tuple)):
        return tuple(int(x) for x in dims)
    raise ValueError(f"matryoshka_dims must be a list, got {dims!r}")


def _merge_spec(base: EmbedderSpec, overrides: dict[str, Any]) -> EmbedderSpec:
    """Apply a config-dict override to a base spec, validating field types.

    Unknown keys are rejected so a typo in the registry file is loud rather
    than silently ignored.
    """
    fields = {f.name for f in base.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(overrides) - fields
    if unknown:
        raise ValueError(
            f"Registry override for '{base.key}' has unknown fields: {sorted(unknown)}"
        )

    kwargs: dict[str, Any] = {}
    if "native_dim" in overrides:
        kwargs["native_dim"] = _coerce_dim(overrides["native_dim"])
    if "matryoshka_dims" in overrides:
        kwargs["matryoshka_dims"] = _coerce_dims(overrides["matryoshka_dims"])
    if "backend" in overrides:
        b = overrides["backend"]
        if b not in ("sentence-transformers", "ollama", "hf"):
            raise ValueError(f"backend must be a known Backend, got {b!r}")
        kwargs["backend"] = b
    for f in ("query_prompt", "document_prompt", "notes", "ollama_model",
              "hf_inference_endpoint", "hf_id"):
        if f in overrides:
            kwargs[f] = overrides[f]
    for f in ("gated", "trust_remote_code", "enabled"):
        if f in overrides:
            kwargs[f] = bool(overrides[f])
    return replace(base, **kwargs)


def build_registry(
    builtin: dict[str, EmbedderSpec] | None = None,
    registry_file: str | Path | None = None,
    enabled_models: tuple[str, ...] | None = None,
) -> dict[str, EmbedderSpec]:
    """Build the effective registry by merging config over built-in defaults.

    Order of precedence (later wins):
      1. ``builtin`` defaults (the code).
      2. ``registry_file`` JSON ``{"models": {key: {…overrides…}}}`` — also
         adds entirely new models not present in the builtin set.
      3. ``enabled_models`` allow-list — anything not in the list is disabled.

    Pure function (no module globals) so tests can call it with arbitrary
    inputs and assert the merge semantics deterministically.
    """
    registry: dict[str, EmbedderSpec] = dict(builtin or _BUILTIN)

    # 2. merge / add from the registry file
    if registry_file:
        path = Path(registry_file)
        if path.exists():
            with path.open() as fh:
                data = json.load(fh)
            models = data.get("models", {}) if isinstance(data, dict) else {}
            for key, overrides in models.items():
                if not isinstance(overrides, dict):
                    raise TypeError(
                        f"Registry entry '{key}' must be an object, got {type(overrides)}"
                    )
                if key in registry:
                    registry[key] = _merge_spec(registry[key], overrides)
                else:
                    # New model: requires the minimal identifying fields.
                    if "hf_id" not in overrides or "native_dim" not in overrides:
                        raise ValueError(
                            f"New registry model '{key}' must define at least "
                            "'hf_id' and 'native_dim'"
                        )
                    spec = EmbedderSpec(
                        key=key,
                        hf_id=overrides["hf_id"],
                        native_dim=_coerce_dim(overrides["native_dim"]),
                    )
                    registry[key] = _merge_spec(spec, overrides)

    # 3. apply the enabled allow-list
    if enabled_models:
        allow = set(enabled_models)
        registry = {
            k: (v if k in allow else replace(v, enabled=False))
            for k, v in registry.items()
        }

    return registry


# The effective registry, resolved once at import from config.
REGISTRY: dict[str, EmbedderSpec] = build_registry(
    registry_file=config.REGISTRY_FILE or None,
    enabled_models=config.ENABLED_MODELS,
)


def enabled_registry() -> dict[str, EmbedderSpec]:
    """Return only enabled models (what /models lists and /embed accepts)."""
    return {k: v for k, v in REGISTRY.items() if v.enabled}


def get_spec(key: str, enabled_only: bool = True) -> EmbedderSpec:
    """Look up a model spec.

    By default only enabled models resolve (so a disabled built-in model
    can't be embedded via the API). Pass ``enabled_only=False`` for
    administrative lookups (e.g. /stats describing a disabled model).
    """
    pool = enabled_registry() if enabled_only else REGISTRY
    if key not in pool:
        if key in REGISTRY and not REGISTRY[key].enabled:
            raise KeyError(f"Model '{key}' is registered but disabled.")
        raise KeyError(
            f"Unknown embedder '{key}'. Known: {', '.join(sorted(REGISTRY))}."
        )
    return pool[key]


# ── Lazy, cached model loading (small LRU) ────────────────────────────────────
_LRU_MAX = int(os.environ.get("MODAL_EMBED_MODEL_LRU", "3"))
_loaded: OrderedDict[str, object] = OrderedDict()
_load_lock = threading.Lock()


def _evict_if_needed() -> None:
    """Keep at most ``_LRU_MAX`` loaded models to avoid GPU OOM with many
    experimental embedders co-resident."""
    while len(_loaded) > _LRU_MAX:
        _loaded.popitem(last=False)


def _load_sentence_transformers(spec: EmbedderSpec, cache_dir: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        spec.hf_id,
        cache_folder=cache_dir,
        trust_remote_code=spec.trust_remote_code,
    )


def load_model(spec: EmbedderSpec, cache_dir: str):
    """Load (and cache) the backend for ``spec``.

    Import is deferred so this module stays importable on the laptop that
    only needs the registry metadata (e.g. for tests), not the heavy ML stack.
    """
    if spec.key in _loaded:
        _loaded.move_to_end(spec.key)
        return _loaded[spec.key]

    with _load_lock:
        if spec.key in _loaded:  # double-checked
            _loaded.move_to_end(spec.key)
            return _loaded[spec.key]

        if spec.backend == "sentence-transformers":
            model = _load_sentence_transformers(spec, cache_dir)
        elif spec.backend == "ollama":
            # No heavy model object — the Ollama host owns the weights. We
            # cache a tiny handle carrying the host + model name.
            model = {"type": "ollama", "host": config.OLLAMA_HOST,
                     "model": spec.ollama_model or spec.key}
        elif spec.backend == "hf":
            model = {"type": "hf",
                     "endpoint": spec.hf_inference_endpoint
                     or f"{config.HF_INFERENCE_BASE_URL.rstrip('/')}/models/{spec.hf_id}"}
        elif spec.backend == "openai-compatible":
            model = {"type": "openai-compatible",
                     "host": spec.ollama_model or config.OLLAMA_HOST,
                     "model": spec.ollama_model or spec.key}
        else:  # pragma: no cover - exhaustiveness guard
            raise ValueError(f"Unknown backend '{spec.backend}' for '{spec.key}'")

        _loaded[spec.key] = model
        _evict_if_needed()
        return model


def unload_model(key: str) -> bool:
    """Drop a cached model (frees GPU memory). Returns True if it was loaded."""
    with _load_lock:
        if key in _loaded:
            _loaded.pop(key)
            return True
        return False


def _embed_st(spec: EmbedderSpec, model, formatted, out_dim, batch_size):
    kwargs = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    # sentence-transformers >= 3.x supports Matryoshka truncation natively.
    if out_dim != spec.native_dim:
        kwargs["truncate_dim"] = out_dim
    vectors = model.encode(formatted, **kwargs)
    return vectors.tolist()


def _embed_ollama(model: dict, formatted, out_dim):
    import urllib.error
    import urllib.request  # deferred; only used by this backend

    host = model["host"].rstrip("/")
    name = model["model"]

    def _finish(vecs):
        # Ollama returns the native dim; truncate + re-normalize for Matryoshka.
        out: list[list[float]] = []
        for vec in vecs:
            vec = vec[:out_dim] if out_dim < len(vec) else vec
            out.append(_normalize(vec))
        return out

    # Preferred: one batched request to the modern /api/embed (input: list),
    # so a whole bulk batch is a single round-trip and Ollama can parallelize.
    try:
        body = json.dumps({"model": name, "input": formatted}).encode()
        req = urllib.request.Request(f"{host}/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            embeddings = json.loads(resp.read())["embeddings"]
        return _finish(embeddings)
    except urllib.error.HTTPError as exc:
        # Only fall back when /api/embed is genuinely missing/unsupported
        # (404/405, i.e. an older server). Other HTTP errors (401, 5xx) re-raise
        # rather than replaying N slow per-text requests that would fail alike.
        if exc.code not in (404, 405):
            raise
    except KeyError:
        # 200 but no "embeddings" key (schema mismatch) -> try the legacy shape.
        pass

    out: list[list[float]] = []
    for text in formatted:
        body = json.dumps({"model": name, "prompt": text}).encode()
        req = urllib.request.Request(f"{host}/api/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            vec = json.loads(resp.read())["embedding"]
        vec = vec[:out_dim] if out_dim < len(vec) else vec
        out.append(_normalize(vec))
    return out


def _embed_openai_compatible(model: dict, formatted, out_dim):
    import urllib.request

    host = model["host"].rstrip("/")
    name = model["model"]
    api_key = model.get("api_key", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps({"model": name, "input": formatted}).encode()
    req = urllib.request.Request(
        f"{host}/v1/embeddings", data=body, headers=headers
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    out: list[list[float]] = []
    for item in data["data"]:
        vec = item["embedding"]
        vec = vec[:out_dim] if out_dim < len(vec) else vec
        out.append(_normalize(vec))
    return out


def _to_sentence_vector(raw) -> list[float]:
    """Coerce a feature_extraction result to a 1-D python float list.

    `feature_extraction` returns either a 1-D sentence vector ``[dim]`` or 2-D
    token embeddings ``[tokens, dim]`` (and occasionally ``[1, tokens, dim]``),
    depending on the endpoint. Mean-pool any token axis down to a single vector.
    """
    try:
        import numpy as np

        arr = np.asarray(raw, dtype=float)
        if arr.ndim <= 1:
            return arr.reshape(-1).tolist()
        # Collapse every leading axis (batch/tokens) onto the trailing dim axis.
        return arr.reshape(-1, arr.shape[-1]).mean(axis=0).tolist()
    except ImportError:  # numpy ships with the model deps; this keeps it optional
        seq = list(raw)
        # Squeeze leading single-element axes (e.g. a [1, tokens, dim] batch).
        while len(seq) == 1 and isinstance(seq[0], (list, tuple)):
            seq = list(seq[0])
        if seq and isinstance(seq[0], (list, tuple)):  # 2-D [tokens, dim] -> mean-pool
            dim = len(seq[0])
            return [sum(float(row[i]) for row in seq) / len(seq) for i in range(dim)]
        return [float(x) for x in seq]


def _embed_hf(model: dict, spec: EmbedderSpec, formatted, out_dim):
    # huggingface_hub's InferenceClient provides the sanctioned feature-
    # extraction path. Deferred import keeps the module laptop-importable.
    from huggingface_hub import InferenceClient

    token = os.environ.get("HF_TOKEN", "")
    client = InferenceClient(model=model["endpoint"], token=token or None)
    # feature_extraction is documented to take a *single* string; iterate rather
    # than passing the whole list (whose return shape is endpoint-dependent).
    out: list[list[float]] = []
    for text in formatted:
        vec = _to_sentence_vector(client.feature_extraction(text))
        vec = vec[:out_dim] if out_dim < len(vec) else vec
        out.append(_normalize(vec))
    return out


def _normalize(vec: list[float], eps: float = 1e-12) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm < eps:
        return vec
    return [x / norm for x in vec]


def embed(
    spec: EmbedderSpec,
    texts: list[str],
    task: Task,
    dim: int | None,
    cache_dir: str,
    batch_size: int = 64,
) -> list[list[float]]:
    """Encode ``texts`` and return L2-normalized vectors of length ``dim``.

    Dispatches on ``spec.backend``. For the ``ollama`` / ``hf`` backends no
    GPU is required — the weights live elsewhere and this container only
    owns the store / bulk orchestration / sync.
    """
    if not texts:
        return []
    out_dim = spec.resolve_dim(dim)
    model = load_model(spec, cache_dir)
    formatted = [spec.format(t, task) for t in texts]

    if spec.backend == "sentence-transformers":
        return _embed_st(spec, model, formatted, out_dim, batch_size)
    if spec.backend == "ollama":
        return _embed_ollama(model, formatted, out_dim)
    if spec.backend == "hf":
        return _embed_hf(model, spec, formatted, out_dim)
    if spec.backend == "openai-compatible":
        return _embed_openai_compatible(model, formatted, out_dim)
    raise ValueError(f"Unknown backend '{spec.backend}' for '{spec.key}'")  # pragma: no cover