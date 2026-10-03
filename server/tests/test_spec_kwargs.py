"""The spec-kwargs contract (issue modal-embedding-server#8):

`model_kwargs` + `config_kwargs` let a registry entry convey loader kwargs
to the sentence-transformers loader (bf16 dtype safety; the selective-
encoder text-only load) with zero behavior change for every entry that
leaves them empty. The loader passes them and merges them ahead of the
spec's own explicit args.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import embedders
from embedders import EmbedderSpec, build_registry


class TestSpecKwargs:
    def test_defaults_are_empty(self):
        spec = EmbedderSpec(key="m", hf_id="m", native_dim=4)
        assert spec.model_kwargs == {}
        assert spec.config_kwargs == {}

    def test_defaults_are_explicit_and_safe(self):
        # fresh dicts per instance (no shared-mutable-default bug):
        a = EmbedderSpec(key="a", hf_id="a", native_dim=4)
        b = EmbedderSpec(key="b", hf_id="b", native_dim=4)
        a.model_kwargs["x"] = 1
        assert b.model_kwargs == {}

    def test_constructor_overrides_flatten_to_kwargs(self):
        spec = EmbedderSpec(
            key="v2",
            hf_id="google/embeddinggemma-2",
            native_dim=768,
            model_kwargs={"torch_dtype": "bf16"},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        assert spec.model_kwargs["torch_dtype"] == "bf16"
        assert spec.config_kwargs["vision_config"] is None

    def test_registry_merge_carries_kwargs(self):
        spec = EmbedderSpec(key="m", hf_id="m", native_dim=4)
        merged = embedders._merge_spec(
            spec,
            {"model_kwargs": {"torch_dtype": "bf16"}, "config_kwargs": {"vision_config": None}},
        )
        assert merged.model_kwargs == {"torch_dtype": "bf16"}
        assert merged.config_kwargs == {"vision_config": None}

    def test_new_registry_entry_carries_kwargs(self, tmp_path):
        reg = tmp_path / "registry.json"
        reg.write_text(
            '{"models": {"embeddinggemma-2": {"hf_id": "google/embeddinggemma-2",'
            ' "native_dim": 768, "model_kwargs": {"torch_dtype": "bf16"}}}}'
        )
        out = build_registry(builtin={}, registry_file=reg)
        assert out["embeddinggemma-2"].model_kwargs == {"torch_dtype": "bf16"}

    def test_loader_passes_kwargs(self, monkeypatch):
        seen = {}

        class _FakeST:
            def __init__(self, hf_id, **kwargs):
                seen.update(kwargs)

        # _load_sentence_transformers does a deferred `from
        # sentence_transformers import SentenceTransformer`: inject a fake
        # module whose attribute carries the constructor:
        import types

        fake = types.SimpleNamespace(SentenceTransformer=_FakeST)
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
        spec = EmbedderSpec(
            key="v2",
            hf_id="google/embeddinggemma-2",
            native_dim=768,
            model_kwargs={"torch_dtype": "bf16"},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        embedders._load_sentence_transformers(spec, "/cache")
        assert seen["model_kwargs"] == {"torch_dtype": "bf16"}
        assert seen["config_kwargs"] == {"vision_config": None, "audio_config": None}
        assert seen["trust_remote_code"] is False

    def test_merge_rejects_non_dict_kwargs(self):
        spec = EmbedderSpec(key="m", hf_id="m", native_dim=4)
        with pytest.raises(ValueError, match="model_kwargs"):
            embedders._merge_spec(spec, {"model_kwargs": ["bad"]})
        with pytest.raises(ValueError, match="config_kwargs"):
            embedders._merge_spec(spec, {"config_kwargs": "bad"})

    def test_to_public_includes_kwargs_keys_shape(self):
        spec = EmbedderSpec(key="m", hf_id="m", native_dim=4, model_kwargs={"torch_dtype": "bf16"})
        public = spec.to_public()
        assert public["model_kwargs"] is not None
