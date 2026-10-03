# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "requests>=2.31"]
# ///
"""Retrieval eval for embedding model selection.

Compares embedders on YOUR data and reports recall@k, MRR@10, query latency,
and an (approximate) cost so the canonical-model choice is data-driven rather
than a guess. Sanctioned providers only: Ollama (local + Cloud), HF Inference,
and our own Modal service. No third-party SaaS embedding vendors.

How it works (standard "document → query" retrieval eval, BEIR-style):
  1. Load a corpus of {id, text} — a JSONL file, or an Obsidian/markdown vault
     directory (one note per row, frontmatter stripped).
  2. Get a query set of {query, relevant_id}. Either provide one, or
     auto-generate one query per sampled doc with a local Ollama LLM.
  3. For each embedder: embed corpus (document) + queries (query), rank by
     cosine, and score whether the source doc is retrieved.

Usage (uv handles deps from the inline script metadata above — no venv/pip):
    cp eval/config.example.json eval/config.json   # edit models + paths
    uv run eval/run_eval.py --config eval/config.json

Set the relevant API keys in your environment (see providers.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import providers
import requests

# ── data loading ──────────────────────────────────────────────────────────────
_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _load_jsonl(path: str, id_field: str, text_field: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get(text_field):
                rows.append({"id": str(obj.get(id_field) or len(rows)), "text": obj[text_field]})
    return rows


def _load_markdown_dir(path: str, exclude_globs: list[str], min_chars: int) -> list[dict]:
    """Ingest an Obsidian/markdown vault: one note → one corpus row.

    id = path relative to the vault root; text = note body with YAML
    frontmatter stripped. Notes shorter than ``min_chars`` are skipped.
    """
    import pathlib

    root = pathlib.Path(path)
    excluded: set = set()
    for pat in exclude_globs or []:
        excluded.update(root.glob(pat))

    rows = []
    for p in sorted(root.glob("**/*.md")):
        if any(p == e or e in p.parents for e in excluded):
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        body = _FRONTMATTER.sub("", raw).strip()
        if len(body) < min_chars:
            continue
        rows.append({"id": str(p.relative_to(root)), "text": body})
    return rows


def load_corpus(corpus_cfg: dict) -> list[dict]:
    """Load a corpus from either a JSONL file or a markdown vault directory.

    Auto-detects: a directory (or ``type: "markdown"``) is treated as a vault;
    anything else is read as JSONL.
    """
    path = corpus_cfg["path"]
    ctype = corpus_cfg.get("type")
    is_dir = os.path.isdir(path)

    if ctype == "markdown" or (ctype is None and is_dir):
        rows = _load_markdown_dir(
            path,
            corpus_cfg.get("exclude_globs", ["**/91 Templates/**"]),
            corpus_cfg.get("min_chars", 80),
        )
    else:
        rows = _load_jsonl(
            path,
            corpus_cfg.get("id_field", "id"),
            corpus_cfg.get("text_field", "fact"),
        )

    limit = corpus_cfg.get("limit")
    return rows[:limit] if limit else rows


_QLINE = re.compile(r'^\s*(?:[-*•]|\d+[.)\]])?\s*["\']?(.*?)["\']?\s*$')


def _clean_query(line: str) -> str:
    m = _QLINE.match(line)
    return (m.group(1) if m else line).strip()


def _query_kind(q: str) -> str:
    head = q.lower().strip().split(" ", 1)[0]
    natural = {
        "what", "how", "why", "who", "where", "when", "which", "whose", "is", "are",
        "do", "does", "can", "should", "describe", "explain", "give", "list", "summarize",
    }
    return "natural" if q.strip().endswith("?") or head in natural else "keyword"


def _note_title(doc: dict) -> str:
    for line in doc["text"].splitlines():
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            return m.group(1).strip().lower()
    return doc["id"].rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()


def _gen_prompt(text: str, n: int) -> str:
    return (
        f"You are building a search-retrieval benchmark. Read the NOTE and write {n} "
        "distinct search queries a person who half-remembers this note would type to find "
        "it again.\nRequirements:\n"
        "- Each query must be specific enough that THIS note is the single best match in a "
        "large vault of similar notes.\n"
        "- Vary the style: mix full natural-language questions with short keyword phrases.\n"
        "- Paraphrase — express the underlying concept; do NOT copy whole sentences or the "
        "title verbatim (we are testing meaning, not keyword overlap).\n"
        "- One query per line. No numbering, bullets, or quotes.\n\n"
        f"NOTE:\n{text}"
    )


def load_or_make_queries(cfg: dict, corpus: list[dict]) -> list[dict]:
    """Return [{query, relevant_id, kind}]. Provided file wins; else auto-generate
    several varied, paraphrased queries per sampled note."""
    qcfg = cfg.get("queries", {})
    if qcfg.get("path"):
        with open(qcfg["path"]) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    import random

    ag = qcfg.get("auto_generate", {})
    n = ag.get("sample", 50)
    per_note = max(1, ag.get("per_note", 3))
    max_ctx = ag.get("max_context_chars", 2000)
    gen_model = ag.get("ollama_model", "gemma4:31b-cloud")
    # Generator host is independent of the embedder host so a CLOUD generator can
    # pair with LOCAL embedders (or vice versa). Precedence: auto_generate.host →
    # OLLAMA_GEN_HOST → OLLAMA_HOST → localhost. (A signed-in local daemon also
    # proxies Ollama Cloud models, in which case localhost is fine.)
    host = (
        ag.get("host")
        or os.environ.get("OLLAMA_GEN_HOST")
        or os.environ.get("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OLLAMA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"

    # Random (seeded) sample so queries span the whole vault, not just the first
    # N notes by path.
    sample = corpus if len(corpus) <= n else random.Random(0).sample(corpus, n)
    queries: list[dict] = []
    print(
        f"Auto-generating up to {len(sample) * per_note} queries "
        f"({per_note}/note × {len(sample)} notes) with '{gen_model}' via {host}…"
    )
    for doc in sample:
        resp = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": gen_model, "prompt": _gen_prompt(doc["text"][:max_ctx], per_note), "stream": False},
            headers=headers,
            timeout=180,
        )
        if resp.status_code == 404:
            raise SystemExit(
                f"Ollama returned 404 for model '{gen_model}' at {host}. "
                "The model isn't available there — pull it (`ollama pull <tag>`), "
                "fix the tag in config.json (queries.auto_generate.ollama_model), "
                "or for an Ollama Cloud model run `ollama signin` (local daemon "
                "proxies it) or set queries.auto_generate.host=https://ollama.com "
                "+ OLLAMA_API_KEY. Check available tags with `ollama list`."
            )
        resp.raise_for_status()
        title = _note_title(doc)
        seen: set = set()
        for line in resp.json().get("response", "").splitlines():
            q = _clean_query(line)
            ql = q.lower()
            if len(q) < 10 or ql in seen or ql == title:
                continue
            if ql.startswith(("here are", "note:", "queries", "query:", "sure", "okay")):
                continue
            seen.add(ql)
            queries.append({"query": q, "relevant_id": doc["id"], "kind": _query_kind(q)})
            if len(seen) >= per_note:
                break
    print(f"Generated {len(queries)} queries.")
    return queries


# ── markdown chunking (Q4 A/B) ────────────────────────────────────────────────
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def split_markdown(text: str, max_chars: int, overlap: int) -> list[str]:
    """Header-aware split: one chunk per markdown section, with the heading
    breadcrumb prepended so each chunk is self-contextualizing. Over-long
    sections are windowed with overlap. No headings → window the whole note.
    """
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    crumb: list[str] = []
    body: list[str] = []

    def flush():
        joined = "\n".join(body).strip()
        if joined:
            sections.append((list(crumb), joined))

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            body.clear()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            crumb = [t for _, t in stack]
        else:
            body.append(line)
    flush()

    chunks: list[str] = []
    for bc, sect in sections:
        prefix = " > ".join(bc)
        head = f"{prefix}\n" if prefix else ""
        if len(sect) <= max_chars:
            chunks.append(head + sect)
            continue
        start, step = 0, max(1, max_chars - overlap)
        while start < len(sect):
            chunks.append(head + sect[start : start + max_chars])
            if start + max_chars >= len(sect):
                break
            start += step
    return chunks


def chunk_corpus(rows: list[dict], chunk_cfg: dict) -> list[dict]:
    """Expand note rows into chunk rows: id = `<note>#<n>`, parent = note id.
    Scoring collapses chunks back to their parent note (note-level recall)."""
    max_chars = chunk_cfg.get("max_chars", 1500)
    overlap = chunk_cfg.get("overlap_chars", 200)
    out = []
    for r in rows:
        pieces = split_markdown(r["text"], max_chars, overlap) or [r["text"]]
        for i, piece in enumerate(pieces):
            out.append({"id": f"{r['id']}#{i}", "text": piece, "parent": r["id"]})
    return out


# ── metrics ─────────────────────────────────────────────────────────────────--
def normalize(mat):
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def score(query_vecs, corpus_vecs, parents, relevant_ids, ks):
    """Note-level retrieval scoring. `parents[i]` is the note id that corpus row
    i belongs to (== its own id in whole-note mode; the parent note in chunk
    mode). Rankings are collapsed to unique parent notes (first occurrence), so
    whole-note and chunked runs are directly comparable.
    """
    import numpy as np

    q = normalize(np.asarray(query_vecs, dtype=np.float32))
    c = normalize(np.asarray(corpus_vecs, dtype=np.float32))
    sims = q @ c.T  # (Q, C) cosine
    order = np.argsort(-sims, axis=1)  # best first
    parents = list(parents)
    max_k = max(ks)
    walk = max(10, max_k)

    recall = {k: 0 for k in ks}
    rr_sum = 0.0
    for qi, rel in enumerate(relevant_ids):
        seen: set = set()
        rank = None
        for idx in order[qi]:
            p = parents[idx]
            if p in seen:
                continue
            seen.add(p)
            if p == rel:
                rank = len(seen)
                break
            if len(seen) >= walk:
                break
        if rank is not None:
            for k in ks:
                if rank <= k:
                    recall[k] += 1
            if rank <= 10:
                rr_sum += 1.0 / rank
    n = len(relevant_ids)
    return {
        **{f"recall@{k}": recall[k] / n for k in ks},
        "mrr@10": rr_sum / n,
    }


def approx_tokens(texts: list[str]) -> int:
    return sum(max(1, len(t) // 4) for t in texts)


# ── main ────────────────────────────────────────────────────────────────────--
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eval/config.json")
    ap.add_argument(
        "--chunk",
        choices=["on", "off"],
        default=None,
        help="override corpus.chunk.enabled for an A/B run (on=chunked, off=whole-note)",
    )
    ap.add_argument(
        "--queries",
        default=None,
        help="query JSONL path: if it EXISTS, load it (freezes the set for a clean A/B); "
        "if it does NOT exist, auto-generate then SAVE here so the next run can reuse it",
    )
    args = ap.parse_args()

    with Path(args.config).open() as fh:
        cfg = json.load(fh)
    if args.chunk is not None:
        cfg.setdefault("corpus", {}).setdefault("chunk", {})["enabled"] = args.chunk == "on"
    # --queries: reuse a cached set if present (so both A/B passes share queries)
    if args.queries and os.path.exists(args.queries):
        cfg.setdefault("queries", {})["path"] = args.queries
    ks = cfg.get("recall_k", [1, 5, 10])

    corpus = load_corpus(cfg["corpus"])
    n_notes = len(corpus)
    # Queries are generated from whole notes (relevant_id = note id) BEFORE any
    # chunking, so the labeled/auto-gen ground truth stays note-level.
    queries = load_or_make_queries(cfg, corpus)
    # First run with --queries pointing at a new path: cache the generated set.
    if args.queries and not os.path.exists(args.queries):
        os.makedirs(os.path.dirname(args.queries) or ".", exist_ok=True)
        with open(args.queries, "w") as fh:
            fh.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in queries)
        print(f"Saved {len(queries)} queries → {args.queries} (reuse with --queries for a clean A/B)")

    chunk_cfg = cfg["corpus"].get("chunk", {})
    if chunk_cfg.get("enabled"):
        corpus = chunk_corpus(corpus, chunk_cfg)
        mc = f"{chunk_cfg.get('max_chars', 1500)}/{chunk_cfg.get('overlap_chars', 200)}"
        mode_slug = "chunked"
        mode_label = f"chunked · {n_notes} notes → {len(corpus)} chunks ({mc} chars)"
        print(f"Corpus: {n_notes} notes → {len(corpus)} chunks (chunking ON, {mc})")
    else:
        for d in corpus:
            d["parent"] = d["id"]  # whole-note: each row is its own "note"
        mode_slug = "wholenote"
        mode_label = f"whole-note · {n_notes} notes"
        print(f"Corpus: {n_notes} notes (whole-note, chunking OFF)")
    print(f"Queries: {len(queries)}")
    if not corpus or not queries:
        raise SystemExit("Need a non-empty corpus and query set.")

    corpus_texts = [d["text"] for d in corpus]
    parents = [d["parent"] for d in corpus]
    query_texts = [q["query"] for q in queries]
    relevant_ids = [str(q["relevant_id"]) for q in queries]

    results = []
    for m in cfg["embedders"]:
        name = m["name"]
        print(f"\n=== {name} ({m['provider']}:{m['model']}) ===")
        try:
            mcfg = {k: m[k] for k in ("dim", "host", "batch_size") if k in m}
            cvecs, _ = providers.timed_embed(
                m["provider"], m["model"], corpus_texts, "document", mcfg
            )
            qvecs, qsecs = providers.timed_embed(
                m["provider"], m["model"], query_texts, "query", mcfg
            )
            metrics = score(qvecs, cvecs, parents, relevant_ids, ks)
            ntok = approx_tokens(corpus_texts) + approx_tokens(query_texts)
            price = m.get("price_per_1m_tokens", 0.0)
            row = {
                "name": name,
                "provider": m["provider"],
                "model": m["model"],
                "dim": len(cvecs[0]) if cvecs else 0,
                **{k: round(v, 4) for k, v in metrics.items()},
                "q_latency_ms": round(1000 * qsecs / max(1, len(query_texts)), 1),
                "approx_cost_usd": round(ntok / 1_000_000 * price, 4),
            }
            results.append(row)
            print({k: row[k] for k in row if k.startswith(("recall", "mrr"))})
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIPPED — {type(exc).__name__}: {exc}")

    if not results:
        raise SystemExit("No embedders ran successfully (missing keys/models?).")

    # ── report ── (filename + title carry the chunk mode so A/B runs are distinct)
    os.makedirs("eval/results", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"eval/results/{stamp}-{mode_slug}"
    cols = list(results[0].keys())

    with open(f"{base}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(results, key=lambda x: -x.get("recall@5", 0)):
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    table = "\n".join(md)
    with open(f"{base}.md", "w") as fh:
        fh.write(f"# Embedding eval — {stamp} — {mode_label}\n\n{table}\n")

    print(f"\n{mode_label}\n{table}\n\nWrote {base}.md and {base}.csv")

    print("\n" + table)
    print(f"\nWrote eval/results/{stamp}.md and .csv")


if __name__ == "__main__":
    main()
