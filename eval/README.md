# Embedding eval harness

Pick the canonical embedding model with **data from your own vault** instead of
a guess. Compares embedders on a retrieval task and reports **recall@k**,
**MRR@10**, **query latency**, and an **approximate cost**.

Providers supported (**sanctioned set only** — no third-party SaaS embedding
vendors; see `docs/plans/embedding-open-questions.md` D10):

- **Ollama** — local *and* Ollama Cloud (same model both places)
- **HF Inference** — serverless feature-extraction, or a dedicated TEI /
  Inference Endpoint (set `host`)
- **Modal** — our own embedding service (`modal/app.py`)

All via raw REST (only `numpy` + `requests` needed).

## Run it

This is a [uv](https://docs.astral.sh/uv/) script — dependencies (`numpy`,
`requests`) are declared as PEP 723 inline metadata in `run_eval.py`, so `uv run`
resolves them automatically. No venv, no `pip install`. Install uv once:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
cp eval/config.example.json eval/config.json   # edit models + corpus path
```

### Pull the embedding models (Ollama)

The candidate models for the canonical-model decision. Missing models are
**skipped automatically** (not fatal), so pull whichever you want to compare:

```bash
ollama pull embeddinggemma          # canonical default (768-dim, Matryoshka)
ollama pull bge-m3                  # 1024-dim, multilingual, long-context
ollama pull qwen3-embedding         # if your Ollama build/library has this tag
# optional extra open candidates:
ollama pull nomic-embed-text        # 768-dim, Matryoshka
ollama pull mxbai-embed-large       # 1024-dim
```

Check exact tags with `ollama list`. (For Ollama Cloud, point `OLLAMA_HOST` at
your cloud host and set `OLLAMA_API_KEY`.)

### Run the eval

For a **local-Ollama-only** run you need **no env vars and no tokens** —
`run_eval.py` defaults `OLLAMA_HOST` to `http://127.0.0.1:11434`:

```bash
uv run eval/run_eval.py --config eval/config.json
```

Only set these if you're testing those providers (each is optional; missing →
that embedder is skipped):

```bash
export OLLAMA_HOST=http://127.0.0.1:11434      # only if not the default / for Cloud
export OLLAMA_API_KEY=…                         # only for Ollama Cloud
export HF_TOKEN=…                               # only for HF Inference embedders
export PVM_MODAL_URL=… PVM_API_TOKEN=…          # only for the Modal service
```

Results print to the terminal and are written to `eval/results/<timestamp>.{md,csv}`.

### Running evals through the Modal GPU service

The eval can use our own Modal embedding service (`modal/app.py`) instead of
local Ollama. This is useful for:

- **Speed** — the Modal service runs `sentence-transformers` on an **L4 GPU**,
  which embeds 256-text batches in ~3–15 s vs minutes on local Ollama.
- **A clean comparison** — the same model (e.g. `embeddinggemma`) produces the
  same vectors whether served via Ollama or Modal's `sentence-transformers`
  backend, so switching providers doesn't confound the experiment.
- **Running chunked A/B with large corpora** — chunking 997 notes produces
  ~18 000 chunks; local Ollama is impractically slow, but Modal's GPU + parallel
  batching handles it in minutes.

#### Prerequisites

1. **Deploy the Modal service** (one-time):
   ```bash
   uvx modal deploy modal/app.py
   ```
   The deploy prints the web URL, e.g.
   `https://<workspace>--pi-vault-mind-embed-embeddingservice-fastapi-app.modal.run`.

2. **Set the auth token** — generate one and store it in the Modal secret:
   ```bash
   TOKEN=$(openssl rand -hex 32)
   uvx modal secret create pi-vault-mind-auth API_TOKEN=$TOKEN --force
   uvx modal deploy modal/app.py   # redeploy so containers pick up the new token
   ```

3. **Export env vars** before running the eval:
   ```bash
   export PVM_MODAL_URL="https://<workspace>--pi-vault-mind-embed-embeddingservice-fastapi-app.modal.run"
   export PVM_API_TOKEN="$TOKEN"
   ```

#### Config — Modal embedder entry

Add a `modal` provider embedder to `eval/config.json`:

```json
{
  "name": "embeddinggemma",
  "provider": "modal",
  "model": "embeddinggemma",
  "timeout": 300,
  "batch_size": 256,
  "parallel": true,
  "max_workers": 8
}
```

**Modal-specific config keys** (all optional):

| key | default | description |
|-----|---------|-------------|
| `batch_size` | 256 | Texts per HTTP request. The L4 GPU handles 256 comfortably. |
| `parallel` | `true` | Send batches concurrently via a thread pool. Set `false` for sequential (debugging). |
| `max_workers` | `min(num_batches, 8)` | Concurrent HTTP requests. Higher = more parallelism = Modal may spin up extra containers. |
| `timeout` | 300 | Per-request timeout in seconds. |

The parallel mode prints progress to stdout:
```
    [modal] 10/72 batches done
    [modal] 20/72 batches done
    ...
```

#### Performance (real numbers from the Q4 A/B run)

| metric | local Ollama (sequential) | Modal L4 GPU (parallel, 8 workers) |
|-------|--------------------------|-----------------------------------|
| corpus size | 996 notes (whole-note) | 18 279 chunks (chunked) |
| embedder | embeddinggemma (Ollama) | embeddinggemma (sentence-transformers) |
| wall time | ~25+ min (estimated) | ~15 min (72 batches × 256) |
| batch latency | ~5–30 s per 32-text batch | ~3–15 s per 256-text batch |

With the parallel provider, 72 batches of 256 texts completed in ~15 min.
At `max_workers: 8`, Modal auto-scaled to 2 containers to handle the
concurrency. Higher `max_workers` would spin up more containers and reduce
wall time further, at higher GPU cost.

#### Example: Q4 whole-note vs chunked A/B via Modal

```bash
# Use a config with only embeddinggemma via Modal (see example above)
export PVM_MODAL_URL="https://<workspace>--pi-vault-mind-embed-embeddingservice-fastapi-app.modal.run"
export PVM_API_TOKEN="<your token>"

# Pass 1 — whole-note
uv run eval/run_eval.py --config eval/config.json --chunk off \
  --queries eval/datasets/stage1-queries.jsonl

# Pass 2 — chunked (reuses the same frozen query set)
uv run eval/run_eval.py --config eval/config.json --chunk on \
  --queries eval/datasets/stage1-queries.jsonl
```

**Result** (committed 2026-06-20, `259e080`): chunking *hurt* every metric —
recall@1 0.7089 → 0.6533, MRR@10 0.7918 → 0.7524, with 18× index bloat
(996 → 18 279 vectors). Decision: **keep whole-note embedding**. See
`docs/plans/embedding-open-questions.md` Q4 for the full analysis.

**Skipped models:** if a model/key/host is missing or errors, the harness prints
`=== <name> … === SKIPPED — <reason>` and continues — the model just won't be in
the table. Check that line if a model you expected is absent (a cold model load
can transiently fail; re-run once it's warm).

**Unlabeled vault (Stage 1):** to compare models on a vault with no labeled query
set, set `queries.path` to `null` and the harness auto-generates queries with a
local Ollama chat model (`queries.auto_generate`). Bump `sample` to ~150–200 for
statistical power. On a larger corpus this de-saturates recall and actually
discriminates between models (a ~50-note corpus does not — see
`docs/plans/embedding-open-questions.md`).

## Whole-note vs chunked A/B (Q4) — run this

Use `--queries` (freeze one query set) + `--chunk on|off` (override the config) so
both passes are identical except for chunking:

```bash
# Pass 1 — whole-note. Auto-generates queries and CACHES them to the path.
uv run eval/run_eval.py --config eval/config.json --chunk off \
  --queries eval/datasets/stage1-queries.jsonl

# Pass 2 — chunked. REUSES the exact same cached queries (clean A/B).
uv run eval/run_eval.py --config eval/config.json --chunk on \
  --queries eval/datasets/stage1-queries.jsonl
```

Each run writes a self-labeled report: `eval/results/<ts>-wholenote.{md,csv}` and
`<ts>-chunked.{md,csv}`. Compare recall@k / MRR between them. (Without the shared
`--queries` cache, auto-gen would produce *different* queries each run and the
comparison would be confounded by query variance, not chunking.)

## How the score is computed

Standard **document → query** retrieval eval (BEIR-style):

1. **Corpus** — `{id, text}` from either source:
   - a **JSONL file** (e.g. `collections/main.jsonl`, `text_field` defaults to `fact`), or
   - an **Obsidian/markdown vault directory** — set `corpus.path` to the vault root
     (auto-detected, or force with `"type": "markdown"`). One note per row,
     `id` = path relative to the root, YAML frontmatter stripped, notes shorter
     than `min_chars` skipped, and `exclude_globs` (default `**/91 Templates/**`)
     dropped.
2. **Queries** — either a labeled set you supply (`{query, relevant_id}` JSONL),
   or **auto-generated**: several (`per_note`) varied, paraphrased queries per
   sampled note via an Ollama chat model (local or Cloud, e.g. `gemma4:31b-cloud`
   — `ollama signin` so the daemon proxies the cloud model). The prompt forces
   paraphrase + specificity (mix of questions and keyword phrases) so the set
   tests semantic retrieval, not keyword overlap; output is deduped, title-echoes
   and junk lines dropped, each tagged `kind` (natural|keyword). The source note
   is the relevant target.
3. For each embedder we embed the corpus (`document` task) and the queries
   (`query` task), rank by cosine similarity, and check the rank of the source
   doc → recall@k and MRR.

Notes:
- EmbeddingGemma prompts (`task: search result | query:` / `title: none | text:`)
  are applied client-side for Ollama so the comparison is fair.
- Cost is **approximate** (≈4 chars/token) and uses the `price_per_1m_tokens`
  in your config — **confirm current vendor pricing**; the example values are
  placeholders and model names/versions may have moved on since this was written.
- This measures retrieval quality, latency, and cost. Weigh those against your
  other constraints (offline needs, vendor lock-in) — see
  [`../docs/plans/embedding-open-questions.md`](../docs/plans/embedding-open-questions.md).

## Chunking A/B (Q4) — DECIDED: keep whole-note

By default each note is embedded whole. Set `corpus.chunk.enabled: true` to
instead header-split notes into breadcrumb-prefixed chunks
(`max_chars`/`overlap_chars`). Queries stay note-level and scoring **collapses
chunks back to their parent note** (note-level recall), so a chunked run and a
whole-note run are directly comparable.

**Result (2026-06-20):** chunking **hurts** — every retrieval metric regressed
(recall@1 0.709→0.653, MRR@10 0.792→0.752) with 18× index bloat (996→18 279
vectors). The corpus is predominantly short, atomic notes that don't benefit from
splitting. Decision: **keep whole-note embedding**. Full numbers and analysis in
[`docs/plans/embedding-open-questions.md`](../docs/plans/embedding-open-questions.md) Q4.

## Interpreting it

- **recall@5 / MRR@10** → retrieval quality (higher is better). This is the
  primary signal for the canonical-model choice (Q1).
- **q_latency_ms** → interactive search feel (on-demand path).
- **approx_cost_usd** → run-rate at scale (bulk re-index).

Pick the model that wins on recall within your latency/cost budget, confirm it
runs where you need it (Ollama for offline same-space fallback, if required),
then lock it in `docs/plans/embedding-open-questions.md` (Q1) and
`docs/MODAL_EMBEDDING.md`.
