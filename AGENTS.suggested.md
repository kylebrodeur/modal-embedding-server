# AGENTS.md (suggested)

This file is a suggestion for projects that use **modal-embedding-server**. Copy it to your
project as `AGENTS.md`, or merge the parts you want into your existing one.
Nothing here overwrites a file automatically.

## What this is

GPU embeddings with a local LanceDB sync. Part of the Modal Toolkit family: https://github.com/kylebrodeur/modal-toolkit.

## Working in a project that uses this

- Deploy only through the project's `deploys/embedding/` overlay. Never run
  `modal deploy` inside a package checkout.
- Check state with `mtk doctor`. Pull with `mtk pull embedding`. Deploy with `mtk deploy embedding`.
- Secrets are confirmed before anything is created. Never write secret values
  into a repo or a chat.
- Read `docs/quickstart.md` and `docs/how-to.md` before changing deploy or
  secret behavior.

## Conventions

- Keep changes small and reviewable. Run the repo's CI commands before pushing.
- Public prose: no em or en dashes, no personal identifiers.
