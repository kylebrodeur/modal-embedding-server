"""Diagnose the vectors Volume after a table-drop cleanup. Reports dir listing + LanceDB open error."""

import os

import modal

import config

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("lancedb>=0.16.0", "pyarrow>=17.0.0")
    .add_local_python_source("config", "embedders", "store", "web", "schemas")
)
app = modal.App(f"{config.APP_NAME}-diag", image=image)
vectors_volume = modal.Volume.from_name(
    config.VECTORS_VOLUME_NAME, create_if_missing=True
)
VOLUMES = {config.VECTORS_DIR: vectors_volume}


@app.function(volumes=VOLUMES)
def diag() -> dict:
    import traceback

    from store import _connect, _tables

    vectors_volume.reload()
    tree = []
    for root, dirs, files in os.walk("/vectors"):
        depth = root[len("/vectors") :].count(os.sep)
        if depth > 2:
            dirs[:] = []
            continue
        tree.append(f"{root}/ [{len(files)} files]")
        for f in files[:5]:
            tree.append(f"  {f}")
    out = {"tree": tree}
    try:
        db = _connect()
        out["tables"] = _tables(db)
        out["connect_ok"] = True
    except Exception as e:  # noqa: BLE001 - diag must not crash
        out["connect_ok"] = False
        out["connect_error"] = f"{type(e).__name__}: {e}"
        out["tb"] = traceback.format_exc().splitlines()[-6:]
    # Try the stats() path that /stats uses
    try:
        from store import stats

        out["stats_ok"] = True
        out["stats"] = stats()
    except Exception as e:  # noqa: BLE001 - diag must not crash
        out["stats_ok"] = False
        out["stats_error"] = f"{type(e).__name__}: {e}"
        out["stats_tb"] = traceback.format_exc().splitlines()[-8:]
    vectors_volume.commit()
    return out


@app.local_entrypoint()
def main():
    print(diag.remote())
