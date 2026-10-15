"""The vLLM plugin registers with an installed vLLM (flashmap[vllm])."""

import importlib.metadata

import pytest

from flashmap.vllm import FLASHMAP_ARCHITECTURE, SUPPORTED_VLLM, _vllm_version, register, update_flashmap

vllm = pytest.importorskip("vllm")
algos = pytest.importorskip("vllm.transformers_utils.configs.speculators.algos")
base = pytest.importorskip("vllm.transformers_utils.configs.speculators.base")


def test_entry_point():
    plugins = importlib.metadata.entry_points(group="vllm.general_plugins")
    assert any(p.name == "flashmap" and p.value == "flashmap.vllm:register" for p in plugins)


def test_register_is_idempotent_and_hooks_vllm():
    if _vllm_version() not in SUPPORTED_VLLM:
        pytest.skip(f"vLLM {vllm.__version__} is not supported by the plugin")
    from vllm.config.speculative import SpeculativeConfig

    register()
    post_init = SpeculativeConfig.__post_init__
    register()
    assert SpeculativeConfig.__post_init__ is post_init and post_init.flashmap
    assert algos.SUPPORTED_SPECULATORS_TYPES["flashmap"] is update_flashmap
    assert FLASHMAP_ARCHITECTURE in vllm.ModelRegistry.get_supported_archs()
    config = {
        "speculators_model_type": "flashmap",
        "speculators_config": {"proposal_methods": [{"speculative_tokens": 16}]},
    }
    assert base.SpeculatorsConfig.build_vllm_speculative_config(config) == {
        "method": "dflash",
        "num_speculative_tokens": 16,
    }
