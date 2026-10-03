# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///
"""Minimal HTTP client for the deployed modal-embedding-server service.

Minimal client reference, and
handy for smoke-testing a deployment from the command line.

Usage (uv handles deps from the inline metadata above):
    export MODAL_EMBED_URL="https://<workspace>--modal-embedding-service.modal.run"
    export MODAL_EMBED_API_TOKEN="<the API_TOKEN you put in the embedding-auth secret>"
    uv run server/client_example.py
"""

from __future__ import annotations

import os
import time

import requests

BASE = os.environ["MODAL_EMBED_URL"].rstrip("/")
TOKEN = os.environ["MODAL_EMBED_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
# Override the model for the whole smoke (default: the canonical embeddinggemma).
# Use a non-gated model (e.g. minilm-l6) to run end-to-end without an HF token.
MODEL = os.environ.get("MODAL_EMBED_MODEL", "embeddinggemma")


def embed(texts: list[str], task: str = "query", model: str = MODEL):
    r = requests.post(
        f"{BASE}/embed",
        json={"texts": texts, "task": task, "model": model},
        headers=HEADERS,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def submit_job(collection: str, records: list[dict], model: str = MODEL):
    r = requests.post(
        f"{BASE}/jobs",
        json={"collection": collection, "records": records, "model": model},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def wait_for_job(job_id: str, poll: float = 2.0):
    while True:
        r = requests.get(f"{BASE}/jobs/{job_id}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        doc = r.json()
        print(f"  [{doc['status']}] {doc.get('processed', 0)}/{doc.get('total', '?')}")
        if doc["status"] in ("done", "error"):
            return doc
        time.sleep(poll)


def export(collection: str, since: int = 0, model: str = MODEL):
    r = requests.get(
        f"{BASE}/sync/export",
        params={"collection": collection, "since": since, "model": model},
        headers=HEADERS,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print("health:", requests.get(f"{BASE}/health", timeout=30).json())

    print(f"\non-demand embed (model={MODEL}):")
    out = embed(["how long do tokens last?"])
    print(f"  model={out['model']} dim={out['dim']} vec[0][:4]={out['vectors'][0][:4]}")

    print("\nbulk job:")
    job = submit_job(
        "main",
        [
            {
                "id": "1",
                "text": "JWT tokens expire after one hour.",
                "metadata": {"tag": "auth"},
            },
            {
                "id": "2",
                "text": "Refresh tokens live for 30 days.",
                "metadata": {"tag": "auth"},
            },
        ],
    )
    print("  submitted:", job)
    wait_for_job(job["job_id"])

    print("\nsync export (since=0):")
    page = export("main")
    print(f"  pulled {page['count']} rows, next_watermark={page['next_watermark']}")
