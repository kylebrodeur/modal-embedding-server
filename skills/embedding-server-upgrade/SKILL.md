---
name: embedding-server-upgrade
description: Deploy-and-upgrade discipline for the modal-embedding-server checkout. Use when a task asks to upgrade, redeploy, or sync this repo, or before any modal command that could touch the account from inside this checkout.
license: Apache-2.0
metadata:
  author: kylebrodeur
  family: modal-toolkit
  repo: modal-embedding-server
---

# embedding-server: upgrade + deploy boundary (lane agents)

This repo is the SOURCE for overlays. Never a deploy target from a
lane. The full rules live in this repo's AGENTS.md; this skill is the
muscle-memory version.

## Before any modal command inside this checkout

1. Prove provenance (from the system workspace):

   ```bash
   tools/guards/deploy-provenance.sh <your-overlay-dir>
   ```

   It refuses plain-repo deploys + bad-shaped app names, fails closed,
   and exports `MTK_APP_SLUG` / `MTK_OVERLAY_ROOT` / `MTK_APP_NAME`.

2. If an overlay for this package does not exist yet, CREATE one
   (`deploys/embedding/`: `deploy.json` with `pkg: embedding`,
   `app_name: <slug>-embedding`, secret NAMES matching
   `server/secrets.toml`; `deploy.sh` runs the real deploy). Do not
   improvise deploys "just this once".

## Upgrading this checkout

The upgrade unit is an upstream TAG, never a head and never a reset:

```bash
git fetch --tags
git checkout <tag>          # e.g. v1.1.0
uv run --project server pytest server/tests -q   # suite must pass
```

After a tag checkout, re-run the overlay's deploy.sh if that lane
serves this package (its own app: `<slug>-embedding`).

## What this repo fires (for overlay lanes)

The lifecycle hooks live in `server/web.py` + `server/app.py`
(`request.pre/post`, `embed.post`, `job.pre/post`; closed tag set).
There is NO `boot.*` tag in this package's set: the
boot-provenance observer from
`modal-toolkit-system/tools/guards/boot-provenance.py` registers on a
`boot.post` seam that lives in other packages (e.g. the vault lane's), not
here, so an overlay cannot wire it through this hook seam. If a lane needs
boot provenance for an embedding deployment, do it outside this package
(its deploy guard is `tools/guards/deploy-provenance.sh`).

## Version + upgrade contract

- The tag on main = the release of record; history is the ONE squashed
  release commit. Do NOT amend/force-push a release outside the
  system workspace's release checklist.
- Vendored shared modules (`server/libs/`) are md5-pinned
  (`mtk libs check`); hand-editing them = drift = refused.
- Never reset/reinit a checkout to solve version confusion: stop and
  ask Kyle.

## Operating with mtk

Source of truth: the [quickstart](../../docs/quickstart.md) and
[how-to](../../docs/how-to.md) in this repo, and the toolkit's `mtk` skill.

```bash
mtk init <project-id>      # from the project directory; checks prerequisites and the account
mtk pull embedding             # clone this repo into the repos root
mtk deploy embedding           # preflight the overlay, confirm secrets, run deploys/embedding/deploy.sh
mtk doctor --pkg embedding     # health and drift
```

- Overlay: `deploys/embedding/deploy.json` must name the app `<project>-embedding`.
- Secrets this package declares: embedding-auth (API_TOKEN, generated) and huggingface-secret (HF_TOKEN, you supply). `mtk deploy` asks before creating any.
- Never run `modal deploy` inside this checkout.
