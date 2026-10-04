"""Embedding provider clients for the retrieval eval (raw REST, minimal deps).

Sanctioned providers ONLY (per project governance — see
docs/plans/embedding-open-questions.md, D10). No third-party SaaS embedding
vendors are introduced here:

    - ollama : local Ollama OR Ollama Cloud (same model both places)
    - hf     : Hugging Face Inference (serverless feature-extraction, or a
               dedicated TEI / Inference Endpoint via `host`)
    - modal  : our own Modal embedding service (modal/app.py /embed contract)

Every provider exposes the same call:

    embed(provider, model, texts, task, cfg) -> list[list[float]]

`task` is "query" or "document". API keys / hosts come from env:

    OLLAMA_HOST (default http://127.0.0.1:11434), OLLAMA_API_KEY (Ollama Cloud)
    HF_TOKEN
    MODAL_EMBED_MODAL_URL, MODAL_EMBED_API_TOKEN   (our Modal service)

Only `requests` + `numpy` are required; no vendor SDKs.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# EmbeddingGemma's official asymmetric prompts (applied client-side where the
# runtime does not template them for us).
GEMMA_QUERY = "task: search result | query: {text}"
GEMMA_DOC = "title: none | text: {text}"


def _chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _maybe_gemma(model: str, texts: list[str], task: str) -> list[str]:
    if "embeddinggemma" in model:
        tmpl = GEMMA_QUERY if task == "query" else GEMMA_DOC
        return [tmpl.format(text=t) for t in texts]
    return texts


# ── Ollama (local or Ollama Cloud) ────────────────────────────────────────────
def _ollama(model: str, texts: list[str], task: str, cfg: dict) -> list[list[float]]:
    host = cfg.get("host") or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"

    texts = _maybe_gemma(model, texts, task)
    out: list[list[float]] = []
    for batch in _chunks(texts, cfg.get("batch_size", 32)):
        resp = requests.post(
            f"{host.rstrip('/')}/api/embed",
            json={"model": model, "input": batch},
            headers=headers,
            timeout=cfg.get("timeout", 120),
        )
        if not resp.ok:
            # Surface the response body so the eval's SKIPPED line is actionable
            # (raise_for_status alone hides Ollama's error message).
            raise RuntimeError(
                f"ollama /api/embed {resp.status_code} for '{model}': {resp.text[:400]}"
            )
        data = resp.json()
        embs = data.get("embeddings")
        if not embs:
            raise RuntimeError(
                f"ollama /api/embed returned no 'embeddings' for '{model}'. "
                f"Got keys {list(data)}; body: {str(data)[:300]}"
            )
        out.extend(embs)
    return out


# ── Hugging Face Inference (serverless feature-extraction, or TEI endpoint) ────
def _hf(model: str, texts: list[str], task: str, cfg: dict) -> list[list[float]]:
    token = os.environ.get("HF_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Default to the serverless feature-extraction router; override `host` to
    # point at a dedicated Inference Endpoint / TEI server.
    base = cfg.get("host") or "https://router.huggingface.co/hf-inference/models"
    url = f"{base.rstrip('/')}/{model}/pipeline/feature-extraction"

    import numpy as np

    texts = _maybe_gemma(model, texts, task)
    out: list[list[float]] = []
    for batch in _chunks(texts, cfg.get("batch_size", 32)):
        resp = requests.post(
            url, json={"inputs": batch}, headers=headers, timeout=cfg.get("timeout", 120)
        )
        resp.raise_for_status()
        for item in resp.json():
            arr = np.asarray(item, dtype=np.float32)
            # sentence-transformers models return a vector; raw models may return
            # a (tokens, dim) matrix → mean-pool to a sentence embedding.
            vec = arr.mean(axis=0) if arr.ndim == 2 else arr
            out.append(vec.tolist())
    return out


# ── Our Modal embedding service ─────────────────────────────────────────────
#
# The Modal service runs sentence-transformers on an L4 GPU.  The GPU itself
# is fast (~3-15 s per 256-text batch), but sending batches *sequentially* over
# HTTP wastes wall-clock time waiting for each round-trip.  When ``parallel``
# is true (the default), batches are sent concurrently via a thread pool so
# the GPU stays saturated and Modal can spin up additional containers
# automatically as concurrency increases.
#
# Config keys (all optional, under the embedder entry in eval/config.json):
#   batch_size   – texts per HTTP request      (default 256)
#   parallel     – send batches concurrently    (default true)
#   max_workers  – concurrent HTTP requests     (default min(num_batches, 8))
#   timeout      – per-request timeout, seconds (default 300)
#
# Set ``parallel: false`` to fall back to sequential mode (useful for
# debugging or when the Modal service has a single-container concurrency cap).

def _modal_single_batch(
    base: str, token: str, model: str, task: str, batch: list[str],
    dim: int | None, timeout: int,
) -> list[list[float]]:
    """Send one POST /embed request and return its vectors."""
    body = {"texts": batch, "task": task, "model": model}
    if dim:
        body["dim"] = dim
    resp = requests.post(
        f"{base}/embed",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(
            f"modal /embed {resp.status_code} for '{model}': {resp.text[:400]}"
        )
    return resp.json()["vectors"]


def _modal(model: str, texts: list[str], task: str, cfg: dict) -> list[list[float]]:
    base = (cfg.get("host") or os.environ["MODAL_EMBED_MODAL_URL"]).rstrip("/")
    token = os.environ["MODAL_EMBED_API_TOKEN"]
    bs = cfg.get("batch_size", 256)
    timeout = cfg.get("timeout", 300)
    dim = cfg.get("dim")
    batches = list(_chunks(texts, bs))
    n = len(batches)

    # Sequential mode (debugging / single-container setups).
    if not cfg.get("parallel", True) or n <= 1:
        out: list[list[float]] = []
        for i, batch in enumerate(batches):
            out.extend(
                _modal_single_batch(base, token, model, task, batch, dim, timeout)
            )
        return out

    # Parallel mode — send all batches concurrently via a thread pool.
    max_workers = cfg.get("max_workers", min(n, 8))
    results: list[list[list[float]] | None] = [None] * n
    first_error: Exception | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(
                _modal_single_batch,
                base, token, model, task, batch, dim, timeout,
            ): i
            for i, batch in enumerate(batches)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            done += 1
            if done % 10 == 0 or done == n:
                print(f"    [modal] {done}/{n} batches done")

    if first_error is not None:
        raise first_error

    out: list[list[float]] = []
    for batch_vecs in results:
        if batch_vecs is not None:
            out.extend(batch_vecs)
    return out


_DISPATCH = {
    "ollama": _ollama,
    "hf": _hf,
    "modal": _modal,
}


def embed(provider: str, model: str, texts: list[str], task: str, cfg: dict):
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(
            f"Unknown/unsanctioned provider '{provider}'. Allowed: {list(_DISPATCH)}"
        )
    return fn(model, texts, task, cfg)


def timed_embed(provider, model, texts, task, cfg):
    """Return (vectors, seconds)."""
    t0 = time.perf_counter()
    vecs = embed(provider, model, texts, task, cfg)
    return vecs, time.perf_counter() - t0
