"""Tests for the Volume-backed vector store: upsert, namespacing, the monotonic
seq watermark, vector-less rejection, stats, delete, and Arrow export.

Uses a real (temp-dir) LanceDB - no GPU, no Modal. lancedb + pyarrow must be
installed (see modal/pyproject.toml [test] extras)."""

from __future__ import annotations

import pyarrow as pa
import pytest
import store


def _vec(d, seed=0.0):
    return [seed + i / d for i in range(d)]


def test_table_name_namespacing():
    assert store.table_name("main", "embeddinggemma", 768) == "col_main__embeddinggemma__768"
    # No sanitization: hyphens survive so the name matches the TS mirror
    # (namespacedTableName in src/modal-config.ts) byte-for-byte.
    assert store.table_name("main", "minilm-l6", 384) == "col_main__minilm-l6__384"


def test_upsert_and_export_watermark():
    rows = [
        {"id": "1", "text": "JWT tokens expire.", "vector": _vec(768, 0.0), "metadata": {"tag": "auth"}},
        {"id": "2", "text": "Refresh tokens.", "vector": _vec(768, 0.1), "metadata": {"tag": "auth"}},
    ]
    n = store.upsert("main", "embeddinggemma", 768, rows)
    assert n == 2

    page = store.export_since("main", "embeddinggemma", 768, since=0, limit=500)
    assert page["count"] == 2
    assert page["done"] is True
    # every exported row carries a vector (Agent B contract requirement #3)
    for r in page["rows"]:
        assert "vector" in r and r["vector"] is not None
        assert len(r["vector"]) == 768
        assert r["model"] == "embeddinggemma"
        assert r["dim"] == 768
        assert r["seq"] > 0
    # watermark = max seq in the page
    assert page["next_watermark"] == max(r["seq"] for r in page["rows"])


def test_export_since_is_incremental():
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "1", "text": "a", "vector": _vec(768), "metadata": {}},
    ])
    page0 = store.export_since("main", "embeddinggemma", 768, since=0, limit=500)
    wm = page0["next_watermark"]

    # add a second row
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "2", "text": "b", "vector": _vec(768, 0.5), "metadata": {}},
    ])
    page1 = store.export_since("main", "embeddinggemma", 768, since=wm, limit=500)
    assert page1["count"] == 1
    assert page1["rows"][0]["id"] == "2"
    assert page1["next_watermark"] > wm


def test_export_unknown_table_is_empty_done():
    page = store.export_since("nope", "embeddinggemma", 768, since=0, limit=500)
    assert page == {"rows": [], "next_watermark": 0, "count": 0, "done": True}


def test_upsert_is_idempotent_merge():
    rows1 = [{"id": "1", "text": "v1", "vector": _vec(768), "metadata": {}}]
    store.upsert("main", "embeddinggemma", 768, rows1)
    # re-upsert same id with new text -> update, not duplicate
    rows2 = [{"id": "1", "text": "v1-updated", "vector": _vec(768, 0.2), "metadata": {"x": 1}}]
    store.upsert("main", "embeddinggemma", 768, rows2)

    page = store.export_since("main", "embeddinggemma", 768, since=0, limit=500)
    assert page["count"] == 1
    assert page["rows"][0]["text"] == "v1-updated"
    assert page["rows"][0]["metadata"] == {"x": 1}


def test_upsert_rejects_vector_less_rows():
    with pytest.raises(ValueError, match="no vector"):
        store.upsert("main", "embeddinggemma", 768, [
            {"id": "1", "text": "no vec", "vector": None, "metadata": {}},
        ])


def test_upsert_empty_is_noop():
    assert store.upsert("main", "embeddinggemma", 768, []) == 0


def test_namespacing_keeps_models_separate():
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "1", "text": "g", "vector": _vec(768), "metadata": {}}])
    store.upsert("main", "minilm-l6", 384, [
        {"id": "1", "text": "m", "vector": _vec(384), "metadata": {}}])
    cols = store.list_collections()
    names = {c["table"] for c in cols}
    assert "col_main__embeddinggemma__768" in names
    assert "col_main__minilm-l6__384" in names
    # each has exactly one row
    assert sum(c["rows"] for c in cols) == 2


def test_list_collections_and_stats():
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "1", "text": "a", "vector": _vec(768), "metadata": {}}])
    cols = store.list_collections()
    assert cols[0]["collection"] == "main"
    assert cols[0]["rows"] == 1
    st = store.stats()
    assert st["tables"] == 1
    assert st["total_rows"] == 1
    assert "indexing" in st


def test_delete_rows():
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "1", "text": "a", "vector": _vec(768), "metadata": {}},
        {"id": "2", "text": "b", "vector": _vec(768, 0.3), "metadata": {}},
    ])
    deleted = store.delete_rows("main", "embeddinggemma", 768, ["1"])
    assert deleted == 1
    page = store.export_since("main", "embeddinggemma", 768, since=0, limit=500)
    assert page["count"] == 1
    assert page["rows"][0]["id"] == "2"


def test_drop_table():
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "1", "text": "a", "vector": _vec(768), "metadata": {}}])
    assert store.drop("main", "embeddinggemma", 768) is True
    assert store.drop("main", "embeddinggemma", 768) is False  # already gone


def test_max_seq():
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "1", "text": "a", "vector": _vec(768), "metadata": {}}])
    wm = store.max_seq("main", "embeddinggemma", 768)
    assert wm > 0
    assert store.max_seq("nope", "embeddinggemma", 768) == 0


def test_export_arrow_returns_ipc_stream():
    store.upsert("main", "embeddinggemma", 768, [
        {"id": "1", "text": "a", "vector": _vec(768), "metadata": {"k": "v"}}])
    payload, _next_wm, done, count = store.export_since_arrow(
        "main", "embeddinggemma", 768, since=0, limit=500)
    assert count == 1
    assert done is True
    # parse the IPC stream back to an Arrow table
    reader = pa.ipc.open_stream(pa.BufferReader(payload))
    tbl = reader.read_all()
    assert tbl.num_rows == 1
    assert "vector" in tbl.column_names
    assert tbl.column("id").to_pylist() == ["1"]


def test_export_arrow_empty_table_is_parseable():
    payload, _next_wm, done, count = store.export_since_arrow(
        "nope", "embeddinggemma", 768, since=0, limit=500)
    assert count == 0
    assert done is True
    reader = pa.ipc.open_stream(pa.BufferReader(payload))
    tbl = reader.read_all()
    assert tbl.num_rows == 0
    assert "vector" in tbl.column_names

def test_export_pages_cover_every_row_when_limit_is_small():
    """Regression for the unordered-limit paging bug: with more matching rows
    than `limit`, draining pages by watermark must return EVERY row exactly
    once, in ascending seq order. (A naive `.limit()` without ORDER BY returns
    an arbitrary subset and silently skips rows.)"""
    n = 25
    store.upsert(
        "main", "embeddinggemma", 768,
        [{"id": str(i), "text": f"note {i}", "vector": _vec(768, i / 100.0), "metadata": {}} for i in range(n)],
    )

    seen: list[str] = []
    seqs: list[int] = []
    since, limit, pages = 0, 7, 0
    while True:
        page = store.export_since("main", "embeddinggemma", 768, since=since, limit=limit)
        pages += 1
        assert len(page["rows"]) <= limit
        for r in page["rows"]:
            seen.append(r["id"])
            seqs.append(r["seq"])
        since = page["next_watermark"]
        if page["done"]:
            break
        assert pages < 20, "paging did not terminate"

    assert pages >= 4, "small limit should force multiple pages"
    assert sorted(seen, key=int) == [str(i) for i in range(n)]  # every row once
    assert len(set(seen)) == n  # no duplicates, none skipped
    assert seqs == sorted(seqs)  # ascending seq across pages


def test_seq_is_strictly_monotonic_within_a_batch():
    """Two rows in one upsert must get distinct, increasing seq (else a page
    boundary could split equal-seq rows and skip one)."""
    store.upsert(
        "main", "embeddinggemma", 768,
        [{"id": str(i), "text": "x", "vector": _vec(768), "metadata": {}} for i in range(50)],
    )
    page = store.export_since("main", "embeddinggemma", 768, since=0, limit=1000)
    seqs = [r["seq"] for r in page["rows"]]
    assert len(set(seqs)) == len(seqs), "seq values must be unique"
    assert seqs == sorted(seqs)


def test_replace_and_export_graph_snapshot():
    entities = [
        {
            "id": "note:1",
            "name": "30-resources/example.md",
            "type": "note",
            "aliases": "Example",
            "summary": "",
            "collection_ids": '["30-resources/example.md"]',
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    ]
    relations = [
        {
            "id": "edge:1",
            "from_entity_id": "note:1",
            "to_entity_id": "note:2",
            "relation_type": "forward",
            "fact": "example links to other",
            "fact_strength": 1.0,
            "source_entry_ids": '["30-resources/example.md"]',
            "valid_at": "2026-01-01T00:00:00Z",
            "expired_at": "",
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]
    result = store.replace_graph_snapshot("vault-main", entities, relations)
    assert result == {"graph_id": "vault-main", "entities": 1, "relations": 1}
    exported = store.export_graph_snapshot("vault-main")
    assert exported["graph_id"] == "vault-main"
    assert len(exported["entities"]) == 1
    assert len(exported["relations"]) == 1
    assert exported["entities"][0]["name"] == "30-resources/example.md"
