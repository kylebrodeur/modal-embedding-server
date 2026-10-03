# modal-embedding-server

A [Modal](https://modal.com) app that does the heavy embedding work for a
local vault: **bulk background jobs**, **low-latency on-demand embedding**,
and a **Volume-backed LanceDB vector store** that syncs down to the local
`.lancedb`.

> **Status:** scaffold. Deployable and self-contained, but not yet wired into
> the TypeScript extension's embedding provider. See
> [`docs/MODAL_EMBEDDING.md`](../docs/MODAL_EMBEDDING.md) for the full design,
> API contract, and sync protocol.

## Why

The local extension embeds with `@xenova/transformers` (MiniLM, 384-dim) or a
local Ollama `embeddinggemma`. That's great for privacy and small edits, but
slow for **bulk** re-indexing and limited by laptop hardware. This app lets you:

- Re-embed thousands of records in the background on a GPU.
- Get fast on-demand vectors for interactive search.
- Standardize on **EmbeddingGemma** (768-dim, Matryoshka-truncatable) so server
  and local vectors are directly compatible.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Names, paths, GPU, defaults (env-overridable). |
| `embedders.py` | Pluggable model registry (EmbeddingGemma + alternatives) + encode. |
| `store.py` | LanceDB-on-Volume vector store; incremental `export_since`. |
| `app.py` | The Modal App: bulk worker, on-demand + sync web service. |
| `datagen.py` | GPU-hosted, multi-model synthetic eval-dataset generator (vLLM). |
| `client_example.py` | Python smoke-test client. |

## Setup

Use [uv](https://docs.astral.sh/uv/). `uvx modal` runs the Modal CLI in an
ephemeral env (or `uv tool install modal` to keep it on PATH). The heavy ML deps
(torch, vLLM, sentence-transformers, lancedb) are installed in the **Modal image**,
not locally — locally you only need the `modal` CLI.

```bash
uvx modal token new        # one-time auth  (or: uv tool install modal && modal token new)

# 1. API token that gates the web endpoints
uvx modal secret create embedding-auth API_TOKEN=$(openssl rand -hex 32)

# 2. HuggingFace token (EmbeddingGemma is a gated repo — accept its license
#    on huggingface.co first, then create a read token)
uvx modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

## Deploy

Create the two secrets first (EmbeddingGemma is a gated HF repo — accept
its license on huggingface.co, then create a read token):

```bash
uvx modal secret create embedding-auth API_TOKEN=$(openssl rand -hex 32)
uvx modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

Then:

```bash
uvx modal deploy modal/app.py
```

> **The vectors Volume must be v2.** LanceDB's commit uses hardlink/rename,
> which Modal Volume **v1** does not support (`linkat` → `Operation not
> permitted`, crashing every bulk upsert). The app creates the vectors
> Volume as v2 by default (`MODAL_EMBED_VECTORS_VOLUME_VERSION=2`). If you have an
> existing v1 vectors Volume, delete it first (`uvx modal volume delete
> <name> -y`) so the deploy recreates it as v2. See
> [lance-format/lance#5775](https://github.com/lance-format/lance/issues/5775).

Modal prints the web URL, e.g.
`https://<workspace>--modal-embedding-server.modal.run`.

First-time deploy before the secrets exist? Stand the infra up, then add
the secrets and redeploy:

```bash
MODAL_EMBED_ATTACH_SECRETS=0 MODAL_EMBED_PREWARM=0 uvx modal deploy modal/app.py   # infra only
# (service fail-closes: protected routes 503 until the auth secret is added)
```

`MODAL_EMBED_PREWARM=0` skips the start-time model pre-warm (it runs in a background
thread by default, so it never blocks readiness); set it back to `1` (the
default) once a valid `HF_TOKEN` is in place so the first `/embed` is fast.

**Verified live against a real deployment:** `/health`, `/models`, `/stats`,
`/sync/collections`, `POST /jobs` (queued → running → error), `GET /jobs`
(list), and 401-without-auth all work against the deployment. The only
remaining step is a human one: replace the `huggingface-secret` placeholder
with a real `HF_TOKEN` after accepting the Gemma license, then `/embed` and
bulk jobs produce vectors.

Quick GPU smoke test (no web layer):

```bash
uvx modal run modal/app.py
```

## Use

```bash
export MODAL_EMBED_MODAL_URL="https://…modal.run"
export MODAL_EMBED_API_TOKEN="…"   # matches the embedding-auth secret
uv run modal/client_example.py   # deps come from the script's PEP 723 metadata
```

## HTTP API (summary)

All routes except `/health` and `/models` require `Authorization: Bearer <API_TOKEN>`.
A deploy without the token fails **closed** (`503`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + default model. |
| GET | `/models` | Registry (key, hf_id, backend, native_dim, matryoshka_dims, prompts, gated, enabled). |
| GET | `/stats` | Store + compute stats (rows per namespace, indexes, GPU, loaded models). |
| POST | `/embed` | On-demand: `{texts, model?, dim?, task}` → `{model, dim, vectors}`. |
| POST | `/jobs` | Bulk: `{collection, records\|records_file\|reembed_from, model?, dim?}` → `{job_id, call_id, total}`. |
| GET | `/jobs/{id}` | Job progress: `{status, processed, total, …}`. |
| GET | `/jobs` | List recent jobs (newest-first), `?limit`. |
| POST | `/jobs/{id}/cancel` | Cooperative cancel (worker stops after its current batch). |
| GET | `/sync/collections` | Tables on the Volume (collection, model, dim, rows, indexes). |
| GET | `/sync/export` | Incremental pull: `?collection&model&dim&since&limit&format=json\|arrow`. Rows always carry `vector`. |
| POST | `/graph/import` | Replace one remote graph snapshot: `{graph_id, entities[], relations[]}`. |
| GET | `/graph/export` | Export one remote graph snapshot: `?graph_id=<id>` → `{graph_id, entities[], relations[]}`. |

Bulk jobs take **exactly one** source: `records` (inline), `records_file`
(a JSONL path on the Volume), or `reembed_from` `{collection, model, dim}`
(re-embed an existing namespace into a new `model__dim` one — model
migration). Jobs are idempotent (keyed by `id`) and resumable.

Full contract in [`docs/MODAL_EMBEDDING.md`](../docs/MODAL_EMBEDDING.md).

Full contract in [`docs/MODAL_EMBEDDING.md`](../docs/MODAL_EMBEDDING.md).

## Configuration

Everything is env-overridable (`MODAL_EMBED_*`); see `modal/config.py`. Highlights:

- `MODAL_EMBED_DEFAULT_MODEL` / `MODAL_EMBED_DEFAULT_DIM` — canonical model + output dim.
- `MODAL_EMBED_ENABLED_MODELS` — comma allow-list; unset = all registered models.
- `MODAL_EMBED_REGISTRY_FILE` — JSON registry merged over the built-in defaults, so
  a new embedder is config-only (no code change):
  `{"models": {"custom-bge": {"hf_id": "BAAI/bge-large-en-v1.5", "native_dim": 1024}}}`.
- `MODAL_EMBED_GPU` — `L4` (default) / `A10G` / `` (CPU, for Ollama/HF-proxy mode).
- `MODAL_EMBED_OLLAMA_HOST` / `MODAL_EMBED_HF_INFERENCE_BASE_URL` — non-GPU backends.
- `MODAL_EMBED_BATCH_SIZE`, `MODAL_EMBED_MAX_CONCURRENT`, `MODAL_EMBED_SCALEDOWN_WINDOW`.
- `MODAL_EMBED_VECTOR_INDEX` / `MODAL_EMBED_FTS` / `MODAL_EMBED_VECTOR_INDEX_TRAIN_THRESHOLD` — index
  policy; `MODAL_EMBED_EXPORT_LIMIT_MAX` caps sync page size.

Each `EmbedderSpec` carries a `backend` (`sentence-transformers` | `ollama`
| `hf`); GPU is optional — when the canonical model is served via Ollama or
HF, the service proxies and owns the store/bulk/sync.

## Tests

The `pytest` suite runs **without a GPU** — the model layer is mocked. The
heavy ML stack (torch, sentence-transformers) is NOT a test dep; only
`lancedb` / `pyarrow` / `fastapi` / `httpx` are (see `modal/pyproject.toml`).

```bash
cd modal
# Keep the venv out of the repo so the repo-wide biome hook doesn't lint it.
UV_PROJECT_ENVIRONMENT=/tmp/modal-embedding-venv uv run --extra test pytest tests
```

Covers: `EmbedderSpec.resolve_dim` + prompt formatting + registry
merge/override + backend dispatch; `store` upsert/watermark/namespacing/
vector-less-reject/Arrow/stats/delete; the full HTTP contract via a
`TestClient` against a Modal-free `web.build_app()` factory.

## Dataset generation (`datagen.py`)

Generate retrieval-eval datasets at scale on Modal GPUs, across multiple
open-weight generator models. Output is the `{query, relevant_id}` JSONL that
`eval/run_eval.py` consumes (`queries.path`), plus `generator_model` + `kind`
per row — so you can diversify a benchmark or cross-check that retrieval results
aren't an artifact of one generator's phrasing.

```bash
# corpus can be a JSONL ({id, fact|text}) OR an Obsidian/markdown vault dir
uvx modal run modal/datagen.py \
  --corpus /path/to/your-vault \
  --models qwen3-8b,mistral-7b \
  --n-per-doc 5 \
  --out datasets/generated-queries.jsonl
```

Writes the dataset both locally (`--out`) and to the Volume under `_datasets/`.
Generator models live in the `GENERATORS` registry in `datagen.py` — **verify
the HuggingFace ids and accept any gated licenses** for your account before a
run. Governance: generation uses **open-weight** models on **our own Modal
inference** only (D7/D10) — no third-party SaaS LLM. Default GPU is
`A100-40GB` (override with `MODAL_EMBED_DATAGEN_GPU`); 7–8B models fit comfortably.

For a hand-written, variance-free benchmark instead, see
[`eval/datasets/`](../eval/datasets/).

## Cost notes

EmbeddingGemma-300m is small; an `L4` (the default) handles it comfortably and
scales to zero after `MODAL_EMBED_SCALEDOWN_WINDOW` seconds idle. Set `MODAL_EMBED_GPU=""` to
run on CPU for very light use. Uncomment `min_containers=1` in `app.py` to keep
one container warm and eliminate cold starts (at the cost of idle spend).
