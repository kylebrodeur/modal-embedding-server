# How to operate modal-embedding-server

Current release: v1.2.0. Commands come from `server/mtk-commands.toml` at
that tag. Start with the [quickstart](./quickstart.md) if you have not
deployed yet.

## Sync vectors to your machine

Syncing is client-side. The server computes; your machine stores and
searches. `mtk embedding sync` prints guidance only; it does not run the sync.
The sync protocol pulls only new vectors into a local LanceDB. Follow the
Phase 4 flight path in the repo's README for the client steps.

## Reindex

`mtk embedding reindex` rebuilds the server-side vector index. It runs
`modal run server/app.py` inside the public checkout. That is the same risk
class the deploy boundary forbids for deploys, so run reindex through your
overlay directory (the one under `deploys/embedding/`), not from the
checkout.

## Model swaps

Bulk re-embed jobs handle a model change. Benchmark candidate models on your
own data with the eval harness before you switch. The
[embed-eval-on-your-vault](https://github.com/kylebrodeur/embed-eval-on-your-vault)
harness is the companion for that.

## Health

`GET /health` for liveness; `GET /stats` for the collection count and loaded
models. A cold start is normal on the first call. Retry once before you debug.

## Scale to zero

The app idles to zero. The first request after idle pays a cold start.

## Related

- Quickstart: [quickstart.md](./quickstart.md)
