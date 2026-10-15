"""The vLLM plugin's reading of a release config.json (no vLLM or GPU needed)."""

import importlib.util
import json
from pathlib import Path

from flashmap import FlashMapConfig
from flashmap.vllm import release_config_fields, update_flashmap

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_checkpoint.py"
spec = importlib.util.spec_from_file_location("convert_checkpoint", SCRIPT)
convert_checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_checkpoint)


def release_config(**overrides) -> dict:
    config = FlashMapConfig(target_model="Qwen/Qwen3-8B", **overrides).to_dict()
    config.update(convert_checkpoint.vllm_fields(config["block_size"], "Qwen/Qwen3-8B", ["Qwen3ForCausalLM"], 65536))
    return json.loads(json.dumps(config))


def test_release_config_maps_onto_vllm_draft_config():
    config = release_config()
    # what vLLM does with it: start from transformer_layer_config, apply the updater
    draft = dict(config["transformer_layer_config"])
    update_flashmap(config_dict=config, pre_trained_config=draft)
    assert draft["architectures"] == ["FlashMapDraftModel"]
    assert (draft["hidden_size"], draft["num_hidden_layers"], draft["num_attention_heads"]) == (2048, 6, 16)
    assert (draft["num_key_value_heads"], draft["head_dim"], draft["intermediate_size"]) == (16, 128, 8192)
    assert draft["vocab_size"] == 151936 and draft["draft_vocab_size"] == 65536
    # vLLM captures the input of the decoder layer after each of target_layer_ids
    assert draft["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]
    assert draft["dflash_config"]["target_layer_ids"] == [1, 9, 17, 25, 33]
    assert draft["rope_parameters"]["rope_theta"] == 10000.0
    assert draft["timestep_embed_dim"] == 256 and draft["timestep_frequency_dim"] == 256
    assert draft["block_size"] == 16 and draft["num_steps"] == 1 and draft["time_scale"] == 1000.0
    assert draft["time_grids"] == [config["jump_grids"][str(n)] for n in range(1, 5)]
    assert release_config_fields(config)["adaln_eps"] == config["layer_norm_eps"]


def test_vllm_speculative_config_from_release():
    config = release_config(block_size=8)
    method = config["speculators_config"]["proposal_methods"][0]
    assert config["speculators_model_type"] == "flashmap" and method["speculative_tokens"] == 8
    assert config["speculators_config"]["verifier"]["name_or_path"] == "Qwen/Qwen3-8B"



def test_method_and_block_size_are_filled_in(tmp_path):
    """--speculative-config '{"model": <FlashMap>}' needs no method or token count."""
    from types import SimpleNamespace

    from flashmap.vllm import _flashmap_config, _post_init

    folder = tmp_path / "FlashMap"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(release_config(block_size=8)))
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert _flashmap_config(str(folder), None)["block_size"] == 8
    assert _flashmap_config(str(other), None) is None
    assert _flashmap_config("ngram", None) is None

    seen = []
    post_init = _post_init(seen.append)
    def spec(model, **kwargs):
        fields = {"method": None, "num_speculative_tokens": None, "revision": None, **kwargs}
        return SimpleNamespace(model=model, _is_custom_proposer_path=lambda m: False, **fields)

    flashmap, given, explicit, plain = (
        spec(str(folder)),
        spec(str(folder), num_speculative_tokens=4),
        spec(str(folder), method="eagle3"),
        spec(str(other)),
    )
    for config in (flashmap, given, explicit, plain):
        post_init(config)
    assert seen == [flashmap, given, explicit, plain]  # vLLM's own __post_init__ always runs
    assert (flashmap.method, flashmap.num_speculative_tokens) == ("dflash", 8)
    assert (given.method, given.num_speculative_tokens) == ("dflash", 4)
    assert explicit.method == "eagle3" and plain.method is None
