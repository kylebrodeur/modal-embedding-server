"""Minimal on-demand embedding example against the deployed Modal Embedding Server.

Usage:
    export MODAL_EMBED_MODAL_URL="https://<workspace>--modal-embedding-server.modal.run"
    export MODAL_EMBED_API_TOKEN="<your auth token>"
    uv run examples/embed_example.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

BASE = os.environ.get("MODAL_EMBED_MODAL_URL", "").rstrip("/")
TOKEN = os.environ.get("MODAL_EMBED_API_TOKEN", "")
MODEL = os.environ.get("MODAL_EMBED_MODEL", "embeddinggemma")

if not BASE or not TOKEN:
    print("Set MODAL_EMBED_MODAL_URL and MODAL_EMBED_API_TOKEN first.", file=sys.stderr)
    sys.exit(1)

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings map text to a numeric vector space.",
]


def main() -> None:
    payload = {"texts": TEXTS, "model": MODEL, "task": "document"}
    req = urllib.request.Request(
        f"{BASE}/embed",
        data=str(payload).replace("'", '"').encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as response:
        body = response.read().decode("utf-8")

    import json

    rows = json.loads(body)
    for text, vec in zip(TEXTS, rows, strict=True):
        print(f"{text}\n  dim={len(vec['vector'])}, head={vec['vector'][:4]}")


if __name__ == "__main__":
    main()