# Quickstart: modal-embedding-server

GPU embeddings on Modal, with vectors synced down to a local LanceDB. Current
release: v1.2.0.

Start with the [toolkit quickstart](https://github.com/kylebrodeur/modal-toolkit/blob/main/docs/quickstart.md) if `mtk` is not
installed yet.

## 1. Create a project

```bash
mtk init embed-lab
cd embed-lab
```

## 2. Pull the package

```bash
mtk pull embedding
```

## 3. Write the overlay

`deploys/embedding/deploy.json` names the app `<project>-embed`:

```json
{ "pkg": "embedding", "app_name": "embed-lab-embed" }
```

`deploys/embedding/deploy.sh` runs `modal deploy` from the pulled checkout by
path. Keep it in your project.

## 4. Deploy

```bash
mtk deploy embedding
```

Two secrets are declared. `embedding-auth` (key `API_TOKEN`) is generated as hex32.
`huggingface-secret` (key `HF_TOKEN`) is one you supply: a token with read access
and the gated model licenses accepted. `mtk deploy` asks before creating either.

## 5. Verify

```bash
mtk doctor --pkg embedding
curl -s https://<your-workspace>--<app>-serve.modal.run/health
```

`/health` is liveness. `GET /stats` returns the collection count and loaded models,
with the bearer token.

## Next

- Syncing vectors, reindex, and the eval harness: [how-to](./how-to.md)
