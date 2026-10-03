"""Drop all stale tables on the vectors Volume.

A remote reindex can hit a LanceDB lock ("open files preventing the operation")
when a table holds an open vector index. The server Volume can also carry stale
e2e-* test tables. This drops all tables on the Volume so the next reindex
rebuilds from local JSONL cleanly. Canonical data is NOT here: it lives in
local JSONL collections.

Run:  uvx modal run server/cleanup.py
"""
import config
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("lancedb>=0.16.0", "pyarrow>=17.0.0")
    .add_local_python_source("config", "embedders", "store", "web", "schemas")
)

app = modal.App(f"{config.APP_NAME}-cleanup", image=image)
vectors_volume = modal.Volume.from_name(config.VECTORS_VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.VECTORS_DIR: vectors_volume}


@app.function(volumes=VOLUMES)
def drop_all() -> dict:
    from store import _connect, _tables
    vectors_volume.reload()
    db = _connect()
    tables = _tables(db)
    results = {}
    for t in tables:
        try:
            db.drop_table(t)
            results[t] = "dropped"
        except Exception as e:  # noqa: BLE001
            results[t] = f"error: {type(e).__name__}: {e}"
    vectors_volume.commit()
    return results


@app.local_entrypoint()
def main():
    print(drop_all.remote())
