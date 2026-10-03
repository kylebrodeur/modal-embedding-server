"""Pydantic request models for the embedding service HTTP contract.

Imported **only** inside ``web.build_app`` (and so only in the container / test
venv, where pydantic is installed). Keeping these out of ``web.py``'s top
level means ``modal deploy`` can import ``web.py`` locally to serialize the
Modal App without needing pydantic/fastapi on the laptop.

These define the request/response bodies for the HTTP contract (see README).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReembedSource(BaseModel):
    collection: str
    model: str
    dim: int


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str | None = None
    dim: int | None = None
    task: str = "query"  # "query" | "document"


class V1EmbeddingRequest(BaseModel):
    """OpenAI-compatible /v1/embeddings request."""

    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] | None = "float"


class JobRequest(BaseModel):
    collection: str
    records: list[dict] | None = Field(
        default=None, description="[{id, text, metadata?}]: inline path"
    )
    records_file: str | None = Field(
        default=None,
        description="Path to a JSONL file on the Volume: large-corpus path",
    )
    reembed_from: ReembedSource | None = Field(
        default=None, description="Re-embed an existing namespace into this one"
    )
    model: str | None = None
    dim: int | None = None


class GraphImportRequest(BaseModel):
    graph_id: str
    entities: list[dict] = Field(default_factory=list)
    relations: list[dict] = Field(default_factory=list)


class GraphChunkRequest(BaseModel):
    graph_id: str
    entities: list[dict] = Field(default_factory=list)
    relations: list[dict] = Field(default_factory=list)


class GraphArtifactUploadRequest(BaseModel):
    """One immutable graph-row artifact staged by the client.

    ``family`` is the derivation source (semantic rows from JSONL facts, or
    note-link rows from vault markdown); ``kind`` is the row family. ``rows`` is
    the NDJSON payload (one JSON object per line). The server issues the
    ``artifact_id`` and re-verifies ``sha256``/``bytes``/``kind`` before the
    worker streams anything.
    """

    family: Literal["semantic", "note-link"]
    kind: Literal["entities", "relations"]
    rows: str = Field(default="", description="NDJSON payload (one row per line)")
    sha256: str = Field(default="", description="hex sha256 of the raw rows payload")


class GraphArtifactManifestEntry(BaseModel):
    """A server-issued artifact reference in a rebuild manifest."""

    artifact_id: str
    family: Literal["semantic", "note-link"]
    kind: Literal["entities", "relations"]
    bytes: int = 0
    rows: int = 0
    sha256: str = ""


class GraphJobManifest(BaseModel):
    """Aggregate manifest for a manifest-based full graph rebuild."""

    artifacts: list[GraphArtifactManifestEntry] = Field(default_factory=list)
    planned_entities: int = 0
    planned_relations: int = 0


class GraphJobRequest(BaseModel):
    graph_id: str
    mode: Literal["replace", "upsert"] = "replace"
    manifest: GraphJobManifest | None = None


class GraphJobBatchRequest(BaseModel):
    batch_key: str | None = None
    entities: list[dict] = Field(default_factory=list)
    relations: list[dict] = Field(default_factory=list)
