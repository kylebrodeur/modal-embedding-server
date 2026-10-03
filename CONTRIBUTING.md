# Contributing

Thanks for your interest in contributing. This project is extracted from production work and kept intentionally lean, so contributions should follow the same spirit.

## Ground Rules

- **Keep it boring.** Prefer straightforward code over clever abstractions. This is a utility, not a framework.
- **No new heavyweight deps** unless absolutely required for a model or runtime. The local test env must stay fast to `uv sync`.
- **Config, not code.** New models belong in config (`models.json` or `embedders.py`'s registry), not hard-coded in the endpoint logic.
- **Env prefixes** follow `MODAL_EMBED_*` / `MODAL_VISION_*` / `MODAL_INFERENCE_*` respectively. Never reintroduce project-specific branding.
- **No AI slop.** Comments and docs should describe *why* the code exists, not restate what it does.

## Workflow

1. Fork and create a feature branch.
2. Use `uv sync --group dev` to set up a local dev environment.
3. Ensure `uv run pytest tests -q` passes for any code change.
4. Run `uv run ruff check .` before submitting.
5. Submit a pull request against `main` with a clear description of what and why.

The project is licensed under Apache 2.0. By contributing you agree that your contributions will be licensed under the same terms.