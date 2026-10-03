"""Drop all stale tables on the pi-vault-mind vectors Volume.

The remote reindex hit a LanceDB lock ("open files preventing the operation") on
col_personal_publishing_plan_main__embeddinggemma__768, which held an open vector index. The server
Volume also carries stale e2e-* test tables. This drops all tables on the Volume so the next
reindex rebuilds from local JSONL cleanly. Canonical data is NOT here — it lives in the local
.vault-mind/collections/*.jsonl.

Run:  uvx modal run modal/cleanup.py
"""
import modal
import config

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("lancedb>=0.16.0", "pyarrow>=17.0.0")
    .add_local_python_source("config", "embedders", "store", "web", "schemas")
)

app = modal.App("pi-vault-mind-cleanup", image=image)
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
