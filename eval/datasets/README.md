# Eval query datasets

Labeled query sets for the retrieval eval (`eval/run_eval.py`). Each line is
`{query, relevant_id, ...}` — `relevant_id` is the corpus id of the note the
query should retrieve (for a markdown vault corpus, that's the note's path
relative to the vault root).

Point the eval at one of these with `queries.path` in `eval/config.json`:

```jsonc
{
  "corpus":  { "path": "/path/to/your-vault", "type": "markdown" },
  "queries": { "path": "eval/datasets/recycvape-queries.jsonl" }
}
```

A labeled set gives a **stable benchmark** — unlike `queries.auto_generate`,
the numbers don't move between runs because of LLM query-generation variance,
so model-to-model recall comparisons are apples-to-apples.

## Datasets

| File | Corpus | Notes |
|---|---|---|
| `recycvape-queries.jsonl` | Any markdown vault (pair it with a `corpus.path` per the schema above) | 29 hand-written queries over 17 distinct notes. Mix of `natural` (20) and `keyword` (9) phrasings so you can see the semantic-vs-lexical gap. Stub notes (`[Content to be added]`) are intentionally not targeted. |

`kind` (`natural` | `keyword`) is metadata for slicing results; the eval reads
only `query` and `relevant_id` and ignores extra fields.

## Adding more

- **By hand** — write `{query, relevant_id}` lines; keep each query's *best*
  match unambiguous (avoid targeting two near-duplicate notes).
- **At scale** — generate synthetic queries with the Modal generator
  (`modal/datagen.py`), which emits this exact format and tags each row with the
  `generator_model` so you can diversify across models. See `modal/README.md`.
