# AGENTS.md — modal-embedding-server

Agent instructions for THIS repo (v1.1.0 on `main`). The family-wide
rules live in the system workspace's AGENTS.md
(`modal-toolkit-system/AGENTS.md`); this file is the local copy of the
rules that matter when you are standing inside this repo.

## What this repo is

GPU embedding service with a monotonic sync protocol + model-selection evals. Part of the modal-* family
(https://github.com/kylebrodeur/modal-toolkit). Apache-2.0.

## NON-NEGOTIABLE: deploys never come from this checkout

**Agents MUST NOT run `modal deploy` / `modal run` from this repo.**
This repo is the SOURCE for operator overlays, not a deploy target.
Deploying from here uses this repo's fixed app name + shared
Volume/Secret names, which collide with live lanes and can WIPE their
state (this exact failure happened to the writing-duo vault lane on
2026-10-09).

- Operator/lane deploys run from an overlay dir
  (`deploys/<pkg>/`: `deploy.json` composes `<slug>-<pkg>` app names +
  secret NAMES; `deploy.sh` is the entry point).
- Before ANY account-touching command, run the preflight guard:

  ```bash
  # from the system workspace:
  tools/guards/deploy-provenance.sh <your-overlay-dir>
  ```

- Standalone PUBLIC users cloning this repo directly are the
  exception the design protects: they get honest docs in the README
  and accept the single-tenant default names. AGENTS (which work in
  Kyle's workspace with live lanes) do not get that exemption.
- NEVER reset/reinit/pull-force a checkout to "fix" version confusion;
  the upgrade unit is an upstream TAG:
  `git fetch --tags && git checkout <tag>` (or `mtk pull`).

## Version + upgrade discipline

- The tag on `main` is the release of record; local checkouts track
  tags, not heads. After a family release wave, upgrade by TAG and
  re-run the repo suite (below).
- History: public repos carry the one squashed release commit
  (`feat: initial public release`); iteration commits land normally
  between releases. Do NOT amend/force-push a release without the
  system workspace's release checklist
  (`modal-toolkit-system/tools/release/release-repo.sh`).

## Verification (run before claiming anything works)

```bash
uv run --project server pytest server/tests -q
```

- Lint/format: `uv run ruff check .` + `uv run ruff format --check .`
  (inside `server/` for the server-wd repos).
- Vendored-copy parity where the repo vendors shared modules:
  `mtk libs check` (from the modal-toolkit sibling; CI runs it).
- Never claim a deploy/health result without running the thing.

## The lifecycle hook seam

- `server/web.py` owns the hooks instance: `request.pre/post` (every authed route), `embed.post`, `job.pre/post` (bulk worker in `server/app.py`).
- `job.pre` fires BOTH at HTTP job acceptance (`server/web.py`, POST /jobs)
  and at worker start (`server/app.py`, `embed_batch` entry); `job.post`
  fires only at worker-terminal states and never on the HTTP request path.

## Built-in model registry

The enabled registry is FIVE built-in models (`server/embedders.py`,
`_BUILTIN`; `MODAL_EMBED_ENABLED_MODELS` can restrict them):

| API key | Native dim | Notes |
|---|---|---|
| `embeddinggemma` | 768 | Canonical model (Matryoshka: 512/256/128). |
| `qwen3-0.6b` | 1024 | Instruction-tuned query prompt. |
| `bge-m3` | 1024 | Multilingual, long-context (8k). |
| `nomic-1.5` | 768 | Matryoshka (512/256/128/64); needs `trust_remote_code`. |
| `minilm-l6` | 384 | Legacy parity with the local transformers.js fallback. |

Never document a different set as "the model set": verify against
`server/embedders.py` `_BUILTIN` before citing it anywhere.

## Prose rules (this is a public repo)

No personal identifiers, no em/en dashes in prose, no AI-slop words
(robust / leverage / seamlessly / utilize), "private-first" not
"local-first". README claims must be runnable back (numbers honest,
sourced from runbooks/eval results).
