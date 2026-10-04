"""Server-side vector store: LanceDB living on a Modal Volume.

We deliberately use the *same* engine as the local extension (LanceDB) so the
"sync down to local" story is a row export/import rather than a database
translation. Tables are namespaced ``col_{collection}__{model}__{dim}`` so a
single Volume can hold the canonical EmbeddingGemma space alongside any
experimental embedders without collisions.

Incremental sync contract
--------------------------
Every row carries a monotonic integer ``seq`` (epoch microseconds at write
time). A client pulls everything with ``seq > since`` and remembers the max
``seq`` it saw as its next watermark. Rows are keyed by ``id`` and applied with
merge-insert on the client, so re-fetching a boundary row is harmless
(idempotent upsert).

Export rows **always carry ``vector``** — vector-less rows can't be
vector-searched, so the sync importer would skip them. (See Agent B note #3.)

Deletes / tombstones
--------------------
Switching the canonical model is a **re-embed into a new namespace**, never a
mix (ADR). The old table is simply :func:`drop`'d once nothing references it;
deletes are an administrative operation (:func:`delete_rows`) and are **not**
propagated through the incremental ``seq`` stream — the append-only watermark
is for adds/updates only.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time

import lancedb
import pyarrow as pa

import config

# Strictly-monotonic seq (epoch microseconds that never repeat within a
# process). Two rows in the same upsert batch must not share a seq, or one can
# be skipped at a page boundary during incremental sync.
_seq_lock = threading.Lock()
_last_seq = 0


def _now_seq() -> int:
    global _last_seq
    with _seq_lock:
        s = max(time.time_ns() // 1000, _last_seq + 1)
        _last_seq = s
        return s


def table_name(collection: str, model: str, dim: int) -> str:
    """The ADR-3 namespaced table name: ``col_{collection}__{model}__{dim}``.

    **No sanitization** — this must match the local extension's
    ``namespacedTableName`` in ``src/modal-config.ts`` byte-for-byte so the
    sync-down target table is the same on both sides. Model keys in the
    registry may contain hyphens (``minilm-l6``, ``qwen3-0.6b``); collection
    names in pi-vault-mind are simple (``main``, …). If a future collection
    name contains ``__`` the parser would be ambiguous — validate collection
    names at ingest time rather than mangling them here.
    """
    return f"col_{collection}__{model}__{dim}"


def _parse_name(name: str) -> tuple[str, str, int] | None:
    m = re.match(r"col_(.+)__(.+)__(\d+)$", name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _tables(db) -> list[str]:
    """Return table names as a plain list across lancedb versions.

    lancedb >= 0.21 with the namespace client returns a ``ListTablesResponse``
    with a ``.tables`` attribute; older builds (and ``table_names()``) return a
    plain list. Normalize so membership/iteration works uniformly.
    """
    try:
        names = db.list_tables()
    except Exception:  # noqa: BLE001 - very old builds
        names = db.table_names()
    if hasattr(names, "tables"):
        return list(names.tables)
    return list(names)


def _connect():
    return lancedb.connect(VECTORS_DIR())


def VECTORS_DIR() -> str:
    """Indirection so tests can point the store at a temp dir via monkeypatch."""
    return config.VECTORS_DIR


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("metadata", pa.string()),  # JSON blob
            pa.field("model", pa.string()),
            pa.field("dim", pa.int32()),
            pa.field("seq", pa.int64()),
            pa.field("created_at", pa.string()),
        ]
    )


def _open_or_create(db, collection: str, model: str, dim: int):
    name = table_name(collection, model, dim)
    if name in _tables(db):
        return db.open_table(name)
    return db.create_table(name, schema=_schema(dim))


def _ensure_indexes(tbl, dim: int) -> None:
    """Create a vector index + optional FTS on a table, idempotently.

    Best-effort: wrapped so a failure to train an index (too few rows, the
    index already exists, or FTS not available) never breaks an upsert.
    """
    try:
        names = {idx.name for idx in tbl.list_indices()}
    except Exception:  # noqa: BLE001 - older lancedb shapes
        names = set()

    if config.VECTOR_INDEX_ENABLED and "vector_idx" not in names:
        try:
            rows = tbl.count_rows()
        except Exception:  # noqa: BLE001
            rows = 0
        if rows >= config.VECTOR_INDEX_TRAIN_THRESHOLD:
            try:
                tbl.create_index(
                    index_type="IVF_PQ",
                    metric="cosine",
                    vector_column_name="vector",
                    num_partitions=max(8, rows // 256),
                    name="vector_idx",
                )
            except Exception:  # noqa: BLE001, S110 - index exists or too few rows, safe no-op
                pass

    if config.FTS_ENABLED and "text_fts" not in names:
        try:
            tbl.create_fts_index("text", name="text_fts")
        except Exception:  # noqa: BLE001, S110 - FTS unavailable / exists
            pass


def upsert(
    collection: str,
    model: str,
    dim: int,
    rows: list[dict],
) -> int:
    """Insert/update embedded rows. ``rows`` items: id, text, vector, metadata.

    Returns the number of rows written. Rows **must** carry a ``vector`` —
    vector-less rows are rejected (they can't be searched and would be skipped
    by the sync importer).
    """
    if not rows:
        return 0
    db = _connect()
    tbl = _open_or_create(db, collection, model, dim)

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = []
    for r in rows:
        vec = r.get("vector")
        if vec is None:
            raise ValueError(
                f"Row '{r.get('id')}' has no vector — vector-less upserts are "
                "rejected. (Re-embed or drop the row instead.)"
            )
        seq = _now_seq()
        meta = r.get("metadata") or {}
        payload.append(
            {
                "id": str(r["id"]),
                "text": r.get("text", ""),
                "vector": vec,
                "metadata": json.dumps(meta, ensure_ascii=False)
                if not isinstance(meta, str)
                else meta,
                "model": model,
                "dim": dim,
                "seq": seq,
                "created_at": r.get("created_at", now_iso),
            }
        )

    (
        tbl.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(payload)
    )
    _ensure_indexes(tbl, dim)
    return len(payload)


def export_since(
    collection: str,
    model: str,
    dim: int,
    since: int,
    limit: int,
    include_vectors: bool = True,
) -> dict:
    """Return rows with ``seq > since`` for incremental local sync.

    Rows **always** include ``vector`` for the sync path (``include_vectors``
    is kept only for internal/administrative reads). ``next_watermark`` is the
    max ``seq`` in the page (or ``since`` if empty), and ``done`` is False when
    the page was full so the client should pull again.
    """
    db = _connect()
    name = table_name(collection, model, dim)
    if name not in _tables(db):
        return {"rows": [], "next_watermark": since, "count": 0, "done": True}

    tbl = db.open_table(name)
    cols = ["id", "text", "vector", "metadata", "model", "dim", "seq", "created_at"]
    if not include_vectors:
        cols.remove("vector")

    # Fetch rows past the watermark, order by seq, then take ONE page.
    # We sort + slice in Python rather than chaining `.limit()` on the query:
    # LanceDB applies `.limit()` WITHOUT an implicit ORDER BY, so a full page
    # would return an arbitrary subset and `next_watermark` (= max seq of that
    # subset) would skip the unreturned lower-seq rows — silent data loss on
    # multi-page sync. Sorting here is correct and version-safe across lancedb
    # releases. For very large stores, swap in a DB-side ordered limit once the
    # lancedb ordering API is pinned
    # (order_by([ColumnOrdering(column_name="seq", order="ascending")])).
    matching = (
        tbl.search().where(f"seq > {int(since)}").select(cols).to_arrow().to_pylist()
    )
    matching.sort(key=lambda r: r["seq"])
    rows = matching[:limit]
    for r in rows:
        if isinstance(r.get("metadata"), str):
            try:
                r["metadata"] = json.loads(r["metadata"])
            except json.JSONDecodeError:
                pass
        # Ensure vector is a plain python list (pyarrow gives lists already,
        # but be defensive for the JSON path).
        if "vector" in r and r["vector"] is not None and not isinstance(r["vector"], list):
            r["vector"] = list(r["vector"])

    next_watermark = rows[-1]["seq"] if rows else since
    return {
        "rows": rows,
        "next_watermark": next_watermark,
        "count": len(rows),
        "done": len(matching) <= limit,
    }


def export_since_arrow(
    collection: str,
    model: str,
    dim: int,
    since: int,
    limit: int,
) -> tuple[bytes, int, bool, int]:
    """Same as :func:`export_since` but returns an Arrow IPC **stream** payload.

    The vector column is always included (no ``include_vectors`` flag) so the
    local importer can merge-insert directly without re-embedding.

    Returns ``(payload_bytes, next_watermark, done, count)``; the app attaches
    the watermark/done/count as response headers since the body is binary.
    """
    page = export_since(collection, model, dim, since, limit, include_vectors=True)
    rows = page["rows"]
    if not rows:
        # Empty → an empty IPC stream. Easiest: build a zero-row table from the
        # schema so the client still gets a parseable payload + the watermark.
        schema = _schema(dim)
        table = pa.table({f: pa.array([], type=t) for f, t in zip(schema.names, schema.types)})
    else:
        # from_pylist requires values matching the schema. ``metadata`` was
        # decoded to a dict for the JSON path; re-serialize to the JSON string
        # the schema expects. Vectors must be float32 lists.
        arrow_rows = []
        for r in rows:
            meta = r.get("metadata")
            if not isinstance(meta, str):
                meta = json.dumps(meta, ensure_ascii=False)
            arrow_rows.append({**r, "metadata": meta,
                                "vector": [float(x) for x in r["vector"]]})
        table = pa.Table.from_pylist(arrow_rows, schema=_schema(dim))
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    payload = sink.getvalue()
    return payload, page["next_watermark"], page["done"], page["count"]


def list_collections() -> list[dict]:
    """Summarize every table on the Volume: collection, model, dim, row count."""
    db = _connect()
    out = []
    for name in _tables(db):
        parsed = _parse_name(name)
        if not parsed:
            continue
        collection, model, dim = parsed
        tbl = db.open_table(name)
        try:
            idx_names = [i.name for i in tbl.list_indices()]
        except Exception:  # noqa: BLE001
            idx_names = []
        out.append(
            {
                "collection": collection,
                "model": model,
                "dim": dim,
                "rows": tbl.count_rows(),
                "table": name,
                "has_vector_index": "vector_idx" in idx_names,
                "has_fts": "text_fts" in idx_names,
            }
        )
    return out


def graph_entity_table_name(graph_id: str) -> str:
    return f"graph_entities__{graph_id}"


def graph_relation_table_name(graph_id: str) -> str:
    return f"graph_relations__{graph_id}"


def _graph_entity_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("type", pa.string()),
            pa.field("aliases", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("collection_ids", pa.string()),
            pa.field("created_at", pa.string()),
            pa.field("updated_at", pa.string()),
        ]
    )


def _graph_relation_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("from_entity_id", pa.string()),
            pa.field("to_entity_id", pa.string()),
            pa.field("relation_type", pa.string()),
            pa.field("fact", pa.string()),
            pa.field("fact_strength", pa.float32()),
            pa.field("source_entry_ids", pa.string()),
            pa.field("valid_at", pa.string()),
            pa.field("expired_at", pa.string()),
            pa.field("created_at", pa.string()),
        ]
    )


def replace_graph_snapshot(graph_id: str, entities: list[dict], relations: list[dict]) -> dict:
    db = _connect()
    entity_table = graph_entity_table_name(graph_id)
    relation_table = graph_relation_table_name(graph_id)
    for name in (entity_table, relation_table):
        if name in _tables(db):
            db.drop_table(name)
    if entities:
        db.create_table(entity_table, data=pa.Table.from_pylist(entities, schema=_graph_entity_schema()))
    if relations:
        db.create_table(
            relation_table,
            data=pa.Table.from_pylist(relations, schema=_graph_relation_schema()),
        )
    return {"graph_id": graph_id, "entities": len(entities), "relations": len(relations)}


def _dedupe_graph_rows(rows: list[dict], provenance_field: str) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in rows:
        rid = str(row["id"])
        existing = merged.get(rid)
        if existing is None:
            merged[rid] = dict(row)
            continue
        next_row = dict(existing)
        next_row.update(row)
        next_row[provenance_field] = json.dumps(
            sorted(_json_list_values(existing.get(provenance_field)) | _json_list_values(row.get(provenance_field))),
            ensure_ascii=False,
        )
        merged[rid] = next_row
    return list(merged.values())


def _dedupe_graph_entities(rows: list[dict]) -> list[dict]:
    return _dedupe_graph_rows(rows, "collection_ids")


def _dedupe_graph_relations(rows: list[dict]) -> list[dict]:
    return _dedupe_graph_rows(rows, "source_entry_ids")

def _json_list_values(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value if str(item)}
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return set()
        if isinstance(parsed, list):
            return {str(item) for item in parsed if str(item)}
    return set()


def _quote_ids_sql(ids: list[str]) -> str:
    return ",".join("'" + str(item).replace("'", "''") + "'" for item in ids)


def _load_existing_rows_by_id(tbl, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    rows: dict[str, dict] = {}
    for start in range(0, len(ids), 128):
        batch_ids = ids[start : start + 128]
        ids_sql = _quote_ids_sql(batch_ids)
        batch_rows = tbl.search().where(f"id IN ({ids_sql})").to_arrow().to_pylist()
        for row in batch_rows:
            rows[str(row["id"])] = row
    return rows


def _merge_graph_entity_row(existing: dict | None, incoming: dict) -> dict:
    if existing is None:
        return incoming
    merged = dict(existing)
    merged.update(incoming)
    merged["collection_ids"] = json.dumps(
        sorted(_json_list_values(existing.get("collection_ids")) | _json_list_values(incoming.get("collection_ids"))),
        ensure_ascii=False,
    )
    return merged


def _merge_graph_relation_row(existing: dict | None, incoming: dict) -> dict:
    if existing is None:
        return incoming
    merged = dict(existing)
    merged.update(incoming)
    merged["source_entry_ids"] = json.dumps(
        sorted(_json_list_values(existing.get("source_entry_ids")) | _json_list_values(incoming.get("source_entry_ids"))),
        ensure_ascii=False,
    )
    return merged


def upsert_graph_snapshot(graph_id: str, entities: list[dict], relations: list[dict]) -> dict:
    db = _connect()
    entity_table = graph_entity_table_name(graph_id)
    relation_table = graph_relation_table_name(graph_id)
    deduped_entities = _dedupe_graph_entities(entities)
    deduped_relations = _dedupe_graph_relations(relations)
    if deduped_entities:
        entity_tbl = db.open_table(entity_table) if entity_table in _tables(db) else db.create_table(
            entity_table, schema=_graph_entity_schema()
        )
        existing_entities = _load_existing_rows_by_id(entity_tbl, [str(item["id"]) for item in deduped_entities])
        merged_entities = [
            _merge_graph_entity_row(existing_entities.get(str(row["id"])), row) for row in deduped_entities
        ]
        (
            entity_tbl.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(merged_entities)
        )
    if deduped_relations:
        relation_tbl = db.open_table(relation_table) if relation_table in _tables(db) else db.create_table(
            relation_table, schema=_graph_relation_schema()
        )
        existing_relations = _load_existing_rows_by_id(relation_tbl, [str(item["id"]) for item in deduped_relations])
        merged_relations = [
            _merge_graph_relation_row(existing_relations.get(str(row["id"])), row) for row in deduped_relations
        ]
        (
            relation_tbl.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(merged_relations)
        )
    return {"graph_id": graph_id, "entities": len(deduped_entities), "relations": len(deduped_relations)}


def export_graph_page(graph_id: str, kind: str, offset: int, limit: int) -> dict:
    db = _connect()
    table_name = graph_entity_table_name(graph_id) if kind == "entities" else graph_relation_table_name(graph_id)
    if table_name not in _tables(db):
        return {"graph_id": graph_id, "kind": kind, "rows": [], "offset": offset, "count": 0, "done": True}
    tbl = db.open_table(table_name)
    rows = tbl.search().to_arrow().to_pylist()
    page = rows[offset : offset + limit]
    return {
        "graph_id": graph_id,
        "kind": kind,
        "rows": page,
        "offset": offset + len(page),
        "count": len(page),
        "done": offset + len(page) >= len(rows),
    }


def export_graph_snapshot(graph_id: str) -> dict:
    db = _connect()
    entity_table = graph_entity_table_name(graph_id)
    relation_table = graph_relation_table_name(graph_id)
    entities = db.open_table(entity_table).search().to_arrow().to_pylist() if entity_table in _tables(db) else []
    relations = db.open_table(relation_table).search().to_arrow().to_pylist() if relation_table in _tables(db) else []
    return {"graph_id": graph_id, "entities": entities, "relations": relations}


def stats() -> dict:
    """Aggregate stats across all tables: rows per namespace, index presence."""
    cols = list_collections()
    total_rows = sum(c["rows"] for c in cols)
    return {
        "tables": len(cols),
        "total_rows": total_rows,
        "vector_dir": VECTORS_DIR(),
        "collections": cols,
        "indexing": {
            "vector_index_enabled": config.VECTOR_INDEX_ENABLED,
            "fts_enabled": config.FTS_ENABLED,
            "train_threshold": config.VECTOR_INDEX_TRAIN_THRESHOLD,
        },
    }


def delete_rows(collection: str, model: str, dim: int, ids: list[str]) -> int:
    """Administrative delete: remove rows by ``id`` from a namespace.

    Not propagated via the incremental ``seq`` stream (the watermark is
    append-only). Use this to clean a namespace before a re-embed, or to drop
    stale rows from a table you intend to :func:`drop` afterwards.
    """
    if not ids:
        return 0
    db = _connect()
    name = table_name(collection, model, dim)
    if name not in _tables(db):
        return 0
    tbl = db.open_table(name)
    ids_sql = ",".join("'" + str(i).replace("'", "''") + "'" for i in ids)
    before = tbl.count_rows()
    tbl.delete(f"id IN ({ids_sql})")
    return before - tbl.count_rows()


def drop(collection: str, model: str, dim: int) -> bool:
    """Drop an entire namespace table. Returns True if it existed."""
    db = _connect()
    name = table_name(collection, model, dim)
    if name not in _tables(db):
        return False
    db.drop_table(name)
    return True


def compact(collection: str, model: str, dim: int) -> dict:
    """Compact a table's files to reduce fragmentation after bulk writes.

    Returns a small status dict. No-op if the table doesn't exist.
    """
    db = _connect()
    name = table_name(collection, model, dim)
    if name not in _tables(db):
        return {"table": name, "compacted": False, "reason": "missing"}
    tbl = db.open_table(name)
    try:
        # lancedb >= 0.16 compaction API. Some builds expose compact_files()
        # with finer control; optimize() is the portable one.
        tbl.optimize()
    except Exception as exc:  # noqa: BLE001
        return {"table": name, "compacted": False, "error": str(exc)}
    return {"table": name, "compacted": True, "rows": tbl.count_rows()}


def _fragmentation(tbl) -> int:  # pragma: no cover - best-effort probe
    """Best-effort count of fragments (unavailable -> -1). Kept for future use."""
    try:
        return len(tbl.list_lance_fragments())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return -1


def max_seq(collection: str, model: str, dim: int) -> int:
    """The current high-watermark ``seq`` for a namespace (0 if absent)."""
    db = _connect()
    name = table_name(collection, model, dim)
    if name not in _tables(db):
        return 0
    tbl = db.open_table(name)
    # Stream the seq column in batches and track the running max so we never
    # materialize the whole column in Python (tables can be large). seq values
    # are strictly positive microsecond stamps, so 0 doubles as "empty".
    import pyarrow.compute as pc

    mx = 0
    try:
        for b in tbl.search().select(["seq"]).to_batches():
            if b.num_rows:
                v = pc.max(b.column("seq")).as_py()
                if v is not None and v > mx:
                    mx = v
    except Exception:  # noqa: BLE001
        return 0
    return int(mx)