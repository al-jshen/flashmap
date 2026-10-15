"""The sglang plugin registers with an installed sglang (flashmap[sglang]), and sglang
reads a release folder's config.json as a DSPARK draft."""

import importlib.metadata
import json

import pytest

from flashmap.sglang import SUPPORTED_SGLANG, _sglang_version, register

pytest.importorskip("sglang")
if _sglang_version() != SUPPORTED_SGLANG:
    pytest.skip(f"sglang {_sglang_version()} is not supported by the plugin", allow_module_level=True)

ARCHITECTURE = "FlashMapDraftModel"


def release_config():
    """The config.json of a release folder (FlashMap-Qwen3-8B's)."""
    return {
        "architectures": [ARCHITECTURE],
        "model_type": "flashmap",
        "block_size": 16,
        "cond_dim": 256,
        "draft_vocab_size": 65536,
        "dspark_config": {"markov_head_type": "vanilla", "markov_rank": 1, "mask_token_id": 151669},
        "dtype": "bfloat16",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "jump_grids": {"1": [0.0, 1.0], "2": [0.0, 0.8172569274902344, 1.0]},
        "max_position_embeddings": 131072,
        "num_attention_heads": 16,
        "num_hidden_layers": 6,
        "target_hidden_size": 4096,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "target_model": "Qwen/Qwen3-8B",
        "vocab_size": 151936,
    }


def test_entry_point():
    plugins = importlib.metadata.entry_points(group="sglang.srt.plugins")
    assert any(p.name == "flashmap" and p.value == "flashmap.sglang:register" for p in plugins)


def test_register_names_the_package(monkeypatch):
    monkeypatch.delenv("SGLANG_EXTERNAL_MODEL_PACKAGE", raising=False)
    register()
    import os

    assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "flashmap.sglang"
    register()  # idempotent
    assert os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] == "flashmap.sglang"


def test_release_folder_reads_as_dspark_draft(tmp_path):
    from sglang.srt.speculative.dspark_components.dspark_config import parse_dspark_draft_config
    from sglang.srt.utils.hf_transformers.common import get_context_length
    from sglang.srt.utils.hf_transformers_utils import get_config

    from flashmap import FlashMapConfig

    (tmp_path / "config.json").write_text(json.dumps(release_config()))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, FlashMapConfig) and config.architectures == [ARCHITECTURE]
    draft = parse_dspark_draft_config(draft_hf_config=config)
    assert draft.gamma == 16 and draft.target_layer_ids == [1, 9, 17, 25, 33]
    assert draft.markov_rank == 1 and draft.markov_head_type == "vanilla" and draft.mask_token_id == 151669
    assert draft.num_hidden_layers == 6
    assert get_context_length(config) == 131072


def test_model_registry_resolves_the_model():
    """However the registry was first imported, after register() it serves the model."""
    from sglang.srt.models.registry import ModelRegistry

    register()
    from flashmap.sglang.model import FlashMapDraftModel

    model_cls, arch = ModelRegistry.resolve_model_cls([ARCHITECTURE])
    assert model_cls is FlashMapDraftModel and arch == ARCHITECTURE
