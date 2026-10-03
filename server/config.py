"""Central configuration for the modal-embedding-server Modal app.

Everything that the bulk worker, the on-demand service, and the sync
endpoint need to agree on lives here so there is a single source of truth
for names, paths, and defaults.

Override any value at deploy time with an environment variable of the same
name (e.g. ``MODAL_EMBED_GPU=A10G uvx modal deploy server/app.py``).

Configurability contract
-------------------------
Per the Agent A brief, **none** of the following may be hard-coded in
business logic: they all resolve from config here:

- App/Volume/secret names, GPU type (incl. ``""`` = CPU), scaledown window,
  max concurrency, batch size, job page size, export limit caps.
- Default + canonical model, enabled model set, default output dim.
- Per-model HF id / prompts / Matryoshka dims / ``trust_remote_code``.
- Whether FTS / vector indexes are built.

Adding a brand-new embedder requires **only** a registry config entry +
(if gated) an HF token: no code change (see ``embedders.py``).
"""

from __future__ import annotations

import os

# ── Modal object names ────────────────────────────────────────────────────────
# Keep these stable: renaming a Volume or App orphans the data behind it.

APP_NAME = os.environ.get("MODAL_EMBED_APP_NAME", "modal-embedding-server")

# Persistent Volume that holds the LanceDB dataset (the server-side vector
# store). This is the thing we sync *down* to the local ``.lancedb``.
VECTORS_VOLUME_NAME = os.environ.get(
    "MODAL_EMBED_VECTORS_VOLUME", "modal-embedding-vectors"
)
VECTORS_DIR = "/vectors"
# VolumeFS backend version. LanceDB writes require hardlink support, which is
# only available on Modal Volume v2 (v1 lacks linkat → EPERM on commit). See
# lance-format/lance#5775. Must match an existing Volume when set.
VECTORS_VOLUME_VERSION = int(os.environ.get("MODAL_EMBED_VECTORS_VOLUME_VERSION", "2"))

# Separate Volume for cached HuggingFace model weights so cold starts don't
# re-download EmbeddingGemma every time.
CACHE_VOLUME_NAME = os.environ.get(
    "MODAL_EMBED_CACHE_VOLUME", "modal-embedding-hf-cache"
)
CACHE_DIR = "/cache"

# ── Secrets ───────────────────────────────────────────────────────────────────
# Create with:
#   modal secret create embedding-auth API_TOKEN=$(openssl rand -hex 32)
#   modal secret create huggingface-secret HF_TOKEN=hf_xxx
AUTH_SECRET_NAME = os.environ.get("MODAL_EMBED_AUTH_SECRET", "embedding-auth")
HF_SECRET_NAME = os.environ.get("MODAL_EMBED_HF_SECRET", "huggingface-secret")

# ── Compute ───────────────────────────────────────────────────────────────────
# GPU is OPTIONAL and config-driven. EmbeddingGemma-300m is small; an L4 is
# plenty and cheap. Set MODAL_EMBED_GPU="" (empty) to run on CPU: the service then does
# no GPU work and is free to proxy embedders served via Ollama / HF Inference
# (backends that own their own compute). Keep the GPU/SentenceTransformer path
# for self-hosting open-weight models on Modal.
_gpu_env = os.environ.get("MODAL_EMBED_GPU", "L4")
GPU = _gpu_env.strip() or None  # "" / "none" → CPU (None)

# How long an idle container stays warm before scaling to zero (seconds).
SCALEDOWN_WINDOW = int(os.environ.get("MODAL_EMBED_SCALEDOWN_WINDOW", "300"))

# Max concurrent requests a single on-demand container will accept. Embedding
# is batch-friendly, so allow a handful to share the GPU.
MAX_CONCURRENT_INPUTS = int(os.environ.get("MODAL_EMBED_MAX_CONCURRENT", "8"))

# ── Embedding defaults ────────────────────────────────────────────────────────
# The canonical model for the project (eval-confirmed: EmbeddingGemma @ 768).
# We standardized on EmbeddingGemma so that vectors produced on Modal are
# directly usable by the local LanceDB. **This is a config value, not a
# hard-coded constant in logic**: read it from here everywhere.
DEFAULT_MODEL = os.environ.get("MODAL_EMBED_DEFAULT_MODEL", "embeddinggemma")

# Default output dimension when a caller omits ``dim``. ``None`` resolves to
# the model's native dim; set an int to force a Matryoshka truncation globally
# (e.g. 512 to ship smaller vectors). Kept distinct from DEFAULT_MODEL so the
# canonical model can be changed without touching dim policy.
DEFAULT_DIM = os.environ.get("MODAL_EMBED_DEFAULT_DIM")
DEFAULT_DIM = int(DEFAULT_DIM) if DEFAULT_DIM and DEFAULT_DIM.strip() else None

# Optional allow-list of enabled model keys (comma-separated). When set, only
# those keys are exposed via /models and accepted by /embed + /jobs; unknown
# keys raise. Empty/unset = all registered models enabled.
_enabled = os.environ.get("MODAL_EMBED_ENABLED_MODELS", "")
ENABLED_MODELS: tuple[str, ...] | None = (
    tuple(k.strip() for k in _enabled.split(",") if k.strip())
    if _enabled.strip()
    else None
)

# Records-per-forward-pass when embedding in bulk.
BATCH_SIZE = int(os.environ.get("MODAL_EMBED_BATCH_SIZE", "64"))

# ── Bulk job bookkeeping ──────────────────────────────────────────────────────
# Job status docs are written here on the Volume so the (separate) web
# container can report progress after a ``volume.reload()``.
JOBS_DIR = f"{VECTORS_DIR}/_jobs"
GRAPH_JOBS_DIR = f"{VECTORS_DIR}/_graph_jobs"
# Immutable graph-row artifacts staged by the client for manifest-based rebuilds.
# Artifacts live under one fixed root; the worker resolves them by server-issued
# artifact_id only (never a client-supplied path) and re-verifies sha256/bytes/kind.
GRAPH_ARTIFACTS_DIR = f"{VECTORS_DIR}/_graph_artifacts"
GRAPH_ARTIFACT_MAX_BYTES = int(
    os.environ.get("MODAL_EMBED_GRAPH_ARTIFACT_MAX_BYTES", str(64 * 1024 * 1024))
)
# Limits for graph upload/rebuild flows.
GRAPH_UPLOAD_BATCH_MAX = int(
    os.environ.get("MODAL_EMBED_GRAPH_UPLOAD_BATCH_MAX", "256")
)
GRAPH_REBUILD_BATCH_SIZE = int(
    os.environ.get("MODAL_EMBED_GRAPH_REBUILD_BATCH_SIZE", "256")
)

# How many rows the worker applies between progress/status commits during a
# rebuild phase. Amortizes Volume commits so a full rebuild is not one commit
# per row/request.
GRAPH_PROGRESS_COMMIT_INTERVAL = int(
    os.environ.get("MODAL_EMBED_GRAPH_PROGRESS_COMMIT_INTERVAL", "2000")
)

# Page size for the GET /jobs list endpoint.
JOB_LIST_LIMIT = int(os.environ.get("MODAL_EMBED_JOB_LIST_LIMIT", "50"))
# ── Export / sync caps ────────────────────────────────────────────────────────
# Hard ceiling on ``limit`` for /sync/export so a client can't request the
# whole corpus in one page. The endpoint clamps to this.
EXPORT_LIMIT_MAX = int(os.environ.get("MODAL_EMBED_EXPORT_LIMIT_MAX", "5000"))
EXPORT_LIMIT_DEFAULT = int(os.environ.get("MODAL_EMBED_EXPORT_LIMIT_DEFAULT", "500"))

# ── Vector store options ──────────────────────────────────────────────────────
# Build a vector index (IVF_PQ) on newly created tables. Set MODAL_EMBED_VECTOR_INDEX=""
# to skip (small datasets / test runs where exact search is fine).
# Default OFF (2026-08-22): IVF_PQ creation on a Modal Volume leaves an open
# index.idx file that blocks later volume.reload(), breaking job poll + sync.
# Exact (brute-force) search is fine for this vault's size.
VECTOR_INDEX_ENABLED = os.environ.get("MODAL_EMBED_VECTOR_INDEX", "0") not in (
    "",
    "0",
    "false",
)

# How many rows a table needs before an IVF_PQ index is created. Below this,
# exact (brute-force) search is fine and avoids training a tiny index.
VECTOR_INDEX_TRAIN_THRESHOLD = int(
    os.environ.get("MODAL_EMBED_VECTOR_INDEX_TRAIN_THRESHOLD", "256")
)

# Build a full-text-search (TFTS) index on the ``text`` column. Useful for
# hybrid search; set MODAL_EMBED_FTS="" to skip.
FTS_ENABLED = os.environ.get("MODAL_EMBED_FTS", "1") not in ("", "0", "false")

# Compact tables after a bulk job if fragmentation exceeds this fraction of
# live rows. ``0`` disables auto-compaction.
COMPACTION_FRAGMENTATION_THRESHOLD = float(
    os.environ.get("MODAL_EMBED_COMPACTION_THRESHOLD", "0.5")
)

# ── Model registry file ──────────────────────────────────────────────────────
# Optional path to a JSON file (baked into the image or on a Volume) whose
# entries are merged *over* the built-in REGISTRY defaults in embedders.py.
# This is the "add an embedder with config only" hook. Example:
#
#   { "models": {
#       "qwen3-0.6b": { "enabled": true },
#       "custom-bge": { "hf_id": "BAAI/bge-large-en-v1.5", "native_dim": 1024,
#                       "backend": "sentence-transformers" } } }
#
# Set MODAL_EMBED_REGISTRY_FILE to a path present in the image / Volume.
REGISTRY_FILE = os.environ.get("MODAL_EMBED_REGISTRY_FILE", "")

# ── Backends ──────────────────────────────────────────────────────────────────
# Ollama proxy backend: base URL of an Ollama host reachable from the container
# (Ollama Cloud or a tunnel). When a model's ``backend`` is "ollama", embed
# calls go here instead of loading a SentenceTransformer. Empty = unused.
OLLAMA_HOST = os.environ.get("MODAL_EMBED_OLLAMA_HOST", "http://localhost:11434")

# HF Inference backend: when a model's ``backend`` is "hf", embeddings are
# fetched from the HF Inference API (serverless or a dedicated endpoint).
# ``hf_inference_endpoint`` on a model spec overrides this per-model.
HF_INFERENCE_BASE_URL = os.environ.get(
    "MODAL_EMBED_HF_INFERENCE_BASE_URL", "https://api-inference.huggingface.co"
)
