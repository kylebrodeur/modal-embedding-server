"""Synthetic eval-dataset generation on Modal.

Scales up the `queries.auto_generate` idea from the local eval harness: hosts
open-weight instruct LLMs on Modal GPUs (via vLLM) and generates `{query,
relevant_id}` benchmark rows from a corpus — fast, in bulk, and across
**multiple generator models** so you can diversify a benchmark (or cross-check
that retrieval quality isn't an artifact of one generator's phrasing).

Output is the exact format `eval/run_eval.py` consumes (`queries.path`), with two
extra fields per row: `generator_model` and `kind`.

Governance (D7/D10): generation runs on **our own Modal inference** using
**open-weight** instruct models — no third-party SaaS LLM is introduced.

> **Status:** scaffold. Deployable; verify the HuggingFace model ids in
> `GENERATORS` (and accept any gated licenses) for your account before a run.

Usage (uvx runs the Modal CLI in an ephemeral env):
    # corpus can be a JSONL ({id, fact|text}) or an Obsidian/markdown vault dir
    uvx modal run modal/datagen.py --corpus /path/to/Vault --models qwen3-8b,mistral-7b \
        --n-per-doc 5 --out datasets/generated.jsonl
"""

from __future__ import annotations

import json
import os
import pathlib
import re

import modal

import config

# ── Generator registry (open-weight instruct models) ─────────────────────────
# key → HuggingFace id. Override/extend at deploy time; gated repos need an HF
# token (huggingface-secret) and a one-time license acceptance.
GENERATORS: dict[str, str] = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",  # gated
    "gemma3-12b": "google/gemma-3-12b-it",  # gated
}
DEFAULT_GENERATOR = os.environ.get("PVM_DATAGEN_DEFAULT", "qwen3-8b")
DATAGEN_GPU = os.environ.get("PVM_DATAGEN_GPU", "A100-40GB")

DATASETS_DIR = f"{config.VECTORS_DIR}/_datasets"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.6.0", "transformers>=4.56.0", "huggingface_hub>=0.25.0")
    .env({"HF_HOME": config.CACHE_DIR})
    .add_local_python_source("config", "datagen")
)

app = modal.App(f"{config.APP_NAME}-datagen", image=image)

vectors_volume = modal.Volume.from_name(config.VECTORS_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(config.CACHE_VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.VECTORS_DIR: vectors_volume, config.CACHE_DIR: cache_volume}
hf_secret = modal.Secret.from_name(config.HF_SECRET_NAME)


# ── Prompt + parsing ──────────────────────────────────────────────────────────
def _build_messages(text: str, n: int, max_chars: int):
    note = text[:max_chars]
    system = (
        "You build retrieval benchmarks. Given a note, write search queries a "
        "user might type that THIS note answers better than any other note."
    )
    user = (
        f"Write {n} diverse queries for the note below. Vary the style: some as "
        "full natural-language questions, some as short keyword phrases. Make them "
        "specific enough that this note is the single best match. Output ONE query "
        "per line — no numbering, no bullets, no quotes.\n\n"
        f"NOTE:\n{note}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_CLEAN = re.compile(r'^\s*(?:[-*\d.)\]]+\s*)?["\']?(.*?)["\']?\s*$')


def _parse_queries(raw: str, n: int) -> list[str]:
    out = []
    for line in raw.splitlines():
        m = _CLEAN.match(line)
        q = (m.group(1) if m else line).strip()
        if len(q) >= 8 and not q.lower().startswith(("here are", "note:", "queries")):
            out.append(q)
    return out[:n]


def _kind(q: str) -> str:
    ql = q.lower().strip()
    if ql.endswith("?") or ql.split(" ", 1)[0] in {
        "what", "how", "why", "who", "where", "when", "which", "describe", "explain", "give",
    }:
        return "natural"
    return "keyword"


# ── Generator (one parametrized instance per model) ───────────────────────────
@app.cls(gpu=DATAGEN_GPU, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 60)
class Generator:
    model_key: str = modal.parameter(default=DEFAULT_GENERATOR)

    @modal.enter()
    def load(self):
        from vllm import LLM

        hf_id = GENERATORS.get(self.model_key, self.model_key)
        self.hf_id = hf_id
        self.llm = LLM(
            model=hf_id,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=8192,
        )

    @modal.method()
    def generate(
        self, docs: list[dict], n_per_doc: int = 5, max_chars: int = 4000, temperature: float = 0.8
    ) -> list[dict]:
        """docs: [{id, text}] → [{query, relevant_id, generator_model, kind}]."""
        from vllm import SamplingParams

        convos = [_build_messages(d["text"], n_per_doc, max_chars) for d in docs]
        sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=256)
        results = self.llm.chat(convos, sp)

        rows = []
        for doc, res in zip(docs, results):
            for q in _parse_queries(res.outputs[0].text, n_per_doc):
                rows.append(
                    {
                        "query": q,
                        "relevant_id": doc["id"],
                        "generator_model": self.model_key,
                        "kind": _kind(q),
                    }
                )
        return rows


def _write_volume_dataset(name: str, rows: list[dict]) -> str:
    os.makedirs(DATASETS_DIR, exist_ok=True)
    path = f"{DATASETS_DIR}/{name}"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    vectors_volume.commit()
    return path


# ── Corpus loading (local side) ───────────────────────────────────────────────
_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _load_corpus(path: str, limit: int | None) -> list[dict]:
    """JSONL ({id, fact|text}) or a markdown vault directory."""
    p = pathlib.Path(path)
    rows = []
    if p.is_dir():
        for f in sorted(p.glob("**/*.md")):
            if "91 Templates" in str(f):
                continue
            body = _FRONTMATTER.sub("", f.read_text(encoding="utf-8", errors="ignore")).strip()
            if len(body) >= 80:
                rows.append({"id": str(f.relative_to(p)), "text": body})
    else:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            text = o.get("text") or o.get("fact")
            if text:
                rows.append({"id": str(o.get("id") or len(rows)), "text": text})
    return rows[:limit] if limit else rows


# ── Entry point ───────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    corpus: str,
    models: str = DEFAULT_GENERATOR,
    n_per_doc: int = 5,
    limit: int = 0,
    out: str = "generated-queries.jsonl",
):
    """Generate a dataset locally-driven; embedding/LLM compute runs on Modal.

    `models` is comma-separated generator keys (diversify the benchmark).
    """
    docs = _load_corpus(corpus, limit or None)
    if not docs:
        raise SystemExit(f"No documents loaded from {corpus}")
    model_keys = [m.strip() for m in models.split(",") if m.strip()]
    print(f"Corpus: {len(docs)} docs · generators: {model_keys} · {n_per_doc}/doc")

    all_rows: list[dict] = []
    for key in model_keys:
        print(f"  generating with '{key}'…")
        rows = Generator(model_key=key).generate.remote(docs, n_per_doc=n_per_doc)
        print(f"    → {len(rows)} queries")
        all_rows.extend(rows)

    # Persist on the Volume (server-side) and to a local file (client-side).
    remote_path = save_to_volume.remote(out, all_rows)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} rows → {out} (local) and {remote_path} (Modal Volume)")


@app.function(volumes=VOLUMES)
def save_to_volume(name: str, rows: list[dict]) -> str:
    return _write_volume_dataset(name, rows)
