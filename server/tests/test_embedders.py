"""Tests for the embedder registry: spec validation, prompt formatting, and the
config-driven registry merge/override. No GPU / no torch — pure data tests."""

from __future__ import annotations

import json

import pytest

from embedders import _BUILTIN, EmbedderSpec, build_registry


# resolve_dim
def test_resolve_dim_none_returns_native():
    spec = _BUILTIN["embeddinggemma"]
    assert spec.resolve_dim(None) == 768


def test_resolve_dim_native_allowed():
    spec = _BUILTIN["embeddinggemma"]
    assert spec.resolve_dim(768) == 768


def test_resolve_dim_matryoshka_allowed():
    spec = _BUILTIN["embeddinggemma"]
    assert spec.resolve_dim(512) == 512
    assert spec.resolve_dim(128) == 128


def test_resolve_dim_rejects_invalid():
    spec = _BUILTIN["embeddinggemma"]
    with pytest.raises(ValueError, match="cannot produce 999-dim"):
        spec.resolve_dim(999)


def test_resolve_dim_non_matryoshka_model():
    spec = _BUILTIN["bge-m3"]  # no matryoshka_dims
    assert spec.resolve_dim(None) == 1024
    with pytest.raises(ValueError):
        spec.resolve_dim(512)


# prompt formatting
def test_format_query_and_document():
    spec = _BUILTIN["embeddinggemma"]
    assert spec.format("hello", "query") == "task: search result | query: hello"
    assert spec.format("hello", "document") == "title: none | text: hello"


def test_format_passthrough_when_no_prompt():
    spec = _BUILTIN["bge-m3"]  # no prompts
    assert spec.format("raw text", "query") == "raw text"
    assert spec.format("raw text", "document") == "raw text"


def test_to_public_includes_native_dim():
    pub = _BUILTIN["embeddinggemma"].to_public()
    assert pub["native_dim"] == 768
    assert pub["backend"] == "sentence-transformers"
    assert pub["matryoshka_dims"] == [512, 256, 128]
    assert pub["enabled"] is True


# registry merge / override
def test_build_registry_defaults_unchanged():
    reg = build_registry()
    assert "embeddinggemma" in reg
    assert reg["embeddinggemma"].native_dim == 768
    assert reg["embeddinggemma"].enabled is True


def test_build_registry_merge_override(tmp_path):
    f = tmp_path / "registry.json"
    f.write_text(json.dumps({
        "models": {
            "embeddinggemma": {"enabled": False, "native_dim": 512},
            "bge-m3": {"backend": "ollama", "ollama_model": "bge-m3:latest"},
        }
    }))
    reg = build_registry(registry_file=f)
    # override existing
    assert reg["embeddinggemma"].enabled is False
    assert reg["embeddinggemma"].native_dim == 512
    assert reg["bge-m3"].backend == "ollama"
    assert reg["bge-m3"].ollama_model == "bge-m3:latest"


def test_build_registry_adds_new_model(tmp_path):
    f = tmp_path / "registry.json"
    f.write_text(json.dumps({
        "models": {
            "custom-bge": {
                "hf_id": "BAAI/bge-large-en-v1.5",
                "native_dim": 1024,
                "backend": "sentence-transformers",
            }
        }
    }))
    reg = build_registry(registry_file=f)
    assert "custom-bge" in reg
    assert reg["custom-bge"].native_dim == 1024
    assert reg["custom-bge"].hf_id == "BAAI/bge-large-en-v1.5"


def test_build_registry_new_model_requires_min_fields(tmp_path):
    f = tmp_path / "registry.json"
    f.write_text(json.dumps({"models": {"incomplete": {"native_dim": 768}}}))
    with pytest.raises(ValueError, match="hf_id"):
        build_registry(registry_file=f)


def test_build_registry_rejects_unknown_field(tmp_path):
    f = tmp_path / "registry.json"
    f.write_text(json.dumps({"models": {"embeddinggemma": {"bogus": 1}}}))
    with pytest.raises(ValueError, match="unknown fields"):
        build_registry(registry_file=f)


def test_build_registry_enabled_filter():
    reg = build_registry(enabled_models=("embeddinggemma",))
    assert reg["embeddinggemma"].enabled is True
    # everything else disabled
    assert reg["bge-m3"].enabled is False
    assert reg["qwen3-0.6b"].enabled is False


def test_get_spec_respects_enabled():
    from embedders import get_spec
    # disabled model from the previous test's filter is independent here;
    # the real module-level REGISTRY is unfiltered (no ENABLED_MODELS in tests).
    spec = get_spec("embeddinggemma")
    assert spec.key == "embeddinggemma"
    with pytest.raises(KeyError):
        get_spec("does-not-exist")


# backend dispatch (mocked, no ML stack)
def test_embed_dispatches_to_backend(monkeypatch):
    import embedders

    calls = {}

    def fake_load(spec, cache_dir):
        return {"type": "ollama", "host": "http://x", "model": spec.key}

    monkeypatch.setattr(embedders, "load_model", fake_load)

    def fake_ollama(model, formatted, out_dim):
        calls["ollama"] = (formatted, out_dim)
        return [[0.1] * out_dim for _ in formatted]

    monkeypatch.setattr(embedders, "_embed_ollama", fake_ollama)

    spec = EmbedderSpec(key="m", hf_id="m", native_dim=4, backend="ollama")
    out = embedders.embed(spec, ["a", "b"], task="query", dim=None, cache_dir="/c")
    assert len(out) == 2
    assert all(len(v) == 4 for v in out)
    # no prompt on this spec -> passthrough
    assert calls["ollama"][0] == ["a", "b"]


def test_embed_empty_texts_returns_empty():
    import embedders

    spec = EmbedderSpec(key="m", hf_id="m", native_dim=4)
    assert embedders.embed(spec, [], task="query", dim=None, cache_dir="/c") == []


# Ollama backend — batched /api/embed with legacy fallback


class _FakeResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def test_embed_ollama_batches_via_api_embed(monkeypatch):
    import urllib.request

    import embedders

    seen = []

    def fake_urlopen(req, timeout=120):
        seen.append((req.full_url, json.loads(req.data)))
        return _FakeResp({"embeddings": [[3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = embedders._embed_ollama({"host": "http://x", "model": "bge"}, ["a", "b"], out_dim=4)

    # one batched call to /api/embed carrying the whole input list
    assert len(seen) == 1
    assert seen[0][0].endswith("/api/embed")
    assert seen[0][1]["input"] == ["a", "b"]
    # vectors are L2-normalized
    assert len(out) == 2
    assert abs(sum(c * c for c in out[0]) - 1.0) < 1e-6


def test_embed_ollama_falls_back_to_per_text_on_404(monkeypatch):
    import urllib.error
    import urllib.request

    import embedders

    urls = []

    def fake_urlopen(req, timeout=120):
        urls.append(req.full_url)
        if req.full_url.endswith("/api/embed"):
            raise urllib.error.HTTPError(req.full_url, 404, "nf", None, None)
        return _FakeResp({"embedding": [1.0, 0.0, 0.0, 0.0]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = embedders._embed_ollama({"host": "http://x", "model": "bge"}, ["a", "b"], out_dim=4)

    assert urls[0].endswith("/api/embed")  # tried the batch path first
    assert sum(1 for u in urls if u.endswith("/api/embeddings")) == 2  # then one per text
    assert len(out) == 2


def test_embed_ollama_reraises_non_404(monkeypatch):
    """A persistent 5xx/auth error must NOT trigger the per-text fallback."""
    import urllib.error
    import urllib.request

    import embedders

    def fake_urlopen(req, timeout=120):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        embedders._embed_ollama({"host": "http://x", "model": "m"}, ["a"], out_dim=4)


# HF backend — per-text feature_extraction with shape coercion


def test_embed_hf_iterates_per_text_and_mean_pools(monkeypatch):
    import sys
    import types

    import embedders

    seen = []

    class _FakeClient:
        def __init__(self, model=None, token=None):
            pass

        def feature_extraction(self, text):
            seen.append(text)
            # 2-D token embeddings [tokens, dim] -> exercises mean-pool
            return [[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.InferenceClient = _FakeClient
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    spec = EmbedderSpec(key="m", hf_id="m", native_dim=4, backend="hf")
    out = embedders._embed_hf({"endpoint": "http://hf"}, spec, ["a", "b"], out_dim=4)

    assert seen == ["a", "b"]  # one call per text, not a single list call
    assert len(out) == 2
    # mean-pool([[2,0,0,0],[0,0,0,0]]) = [1,0,0,0] -> normalized [1,0,0,0]
    assert abs(out[0][0] - 1.0) < 1e-6 and abs(sum(c * c for c in out[0]) - 1.0) < 1e-6


def test_to_sentence_vector_pure_python_handles_shapes(monkeypatch):
    """With numpy unavailable, coerce 1-D, 2-D, and squeezable 3-D shapes."""
    import sys

    import embedders

    monkeypatch.setitem(sys.modules, "numpy", None)  # force the ImportError path

    assert embedders._to_sentence_vector([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    # 2-D [tokens, dim] -> mean-pool
    assert embedders._to_sentence_vector([[2.0, 0.0], [0.0, 0.0]]) == [1.0, 0.0]
    # 3-D [1, tokens, dim] -> squeeze batch then mean-pool (no TypeError)
    assert embedders._to_sentence_vector([[[2.0, 0.0], [0.0, 0.0]]]) == [1.0, 0.0]