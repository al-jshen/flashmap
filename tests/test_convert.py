"""scripts/convert_checkpoint.py on toy DeepSpec, speculators and SpecForge checkpoints."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModel

from flashmap import FlashMapConfig, FlashMapDraftModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_checkpoint.py"
spec = importlib.util.spec_from_file_location("convert_checkpoint", SCRIPT)
convert_checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_checkpoint)

TARGET_LAYER_IDS = [0, 2]


def toy_weights():
    config = FlashMapConfig(
        target_model="toy/target",
        target_hidden_size=16,
        vocab_size=64,
        target_layer_ids=TARGET_LAYER_IDS,
        block_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        cond_dim=16,
        time_embed_dim=8,
    )
    torch.manual_seed(0)
    weights = FlashMapDraftModel(config).state_dict()
    for name, tensor in weights.items():
        if name.endswith("norm.weight"):  # 1 + scale, exact in float32 both ways
            weights[name] = 1 + torch.randint(-20, 20, tensor.shape) / 256
        else:
            weights[name] = torch.randn(tensor.shape)
    return config, weights


TOP_LEVEL = {
    "time_embed.linear_1.": "denoiser.temb.mlp.0.",
    "time_embed.linear_2.": "denoiser.temb.mlp.2.",
    "jump_embed.linear_1.": "denoiser.temb_jump.mlp.0.",
    "jump_embed.linear_2.": "denoiser.temb_jump.mlp.2.",
    "final_adaln.linear.": "denoiser.final_norm.cond_proj.1.",
    "x_embed.": "denoiser.x_embed.",
    "head.": "denoiser.head.",
}
IN_LAYER = {
    "self_attn_adaln.linear.": "adaln_self.cond_proj.1.",
    "cross_attn_adaln.linear.": "adaln_cross.cond_proj.1.",
    "mlp_adaln.linear.": "adaln_mlp.cond_proj.1.",
    "mlp.fc1.": "mlp.0.",
    "mlp.fc2.": "mlp.3.",
    "q_proj.": "wq.",
    "o_proj.": "wo.",
    "q_norm.": "qnorm.",
    "k_norm.": "knorm.",
}


def to_deepspec(weights):
    """FlashMap weights under DeepSpec's names, with the target's embedding and lm_head."""
    out = {"embed_tokens.weight": torch.randn(64, 16), "lm_head.weight": torch.randn(64, 16)}
    for name, tensor in weights.items():
        if name.endswith("v_proj.weight"):
            continue
        if name.startswith("layers."):
            new = "denoiser.blocks." + name[len("layers."):]
            for old, deepspec in IN_LAYER.items():
                new = new.replace(old, deepspec)
            if name.endswith("k_proj.weight"):  # keys and values in one projection
                new = new.replace("k_proj.", "wkv.")
                tensor = torch.cat([tensor, weights[name.replace("k_proj.", "v_proj.")]])
        else:
            new = next((d + name[len(o):] for o, d in TOP_LEVEL.items() if name.startswith(o)), name)
        if name.endswith("norm.weight"):
            new, tensor = new[: -len("weight")] + "scale", tensor - 1
        out[new] = tensor
    return out


def write(path: Path, config: dict, weights: dict) -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    save_file(weights, str(path / "model.safetensors"))
    return path


def check_release(path: Path, weights: dict, **fields):
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.float32)
    assert type(model).__name__ == "FlashMapDraftModel"
    for key, value in fields.items():
        assert getattr(model.config, key) == value, key
    loaded = model.state_dict()
    assert loaded.keys() == weights.keys()
    for name, tensor in weights.items():
        assert torch.equal(loaded[name], tensor), name


def test_deepspec(tmp_path):
    config, weights = toy_weights()
    source = {
        "hidden_size": 16,
        "vocab_size": 64,
        "target_layer_ids": TARGET_LAYER_IDS,
        "block_size": 4,
        "dit_embed_dim": 32,
        "dit_cond_dim": 16,
        "dit_num_heads": 2,
        "num_hidden_layers": 2,
        "state_featurization": "posterior_noise",
        "noise_gate": "jump",
        "time_schedule": "decoding_error",
        "rope_theta": 1000000,  # the target's, not the DiT's
    }
    checkpoint = write(tmp_path / "deepspec", source, to_deepspec(weights))
    with pytest.raises(ValueError, match="target-model"):
        convert_checkpoint.convert(checkpoint, tmp_path / "out0", dtype=torch.float32)
    convert_checkpoint.convert(checkpoint, tmp_path / "out", target_model="toy/target", dtype=torch.float32)
    check_release(tmp_path / "out", weights, target_layer_ids=TARGET_LAYER_IDS, rope_theta=10000.0, time_embed_dim=8)


def test_speculators(tmp_path):
    config, weights = toy_weights()
    source = {
        "speculators_model_type": "flashmap",
        "architectures": ["FlashMapDraftModel"],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "vocab_size": 64,
        },
        "aux_hidden_state_layer_ids": [i + 1 for i in TARGET_LAYER_IDS],
        "block_size": 4,
        "target_hidden_size": 16,
        "draft_vocab_size": 32,
        "timestep_embed_dim": 16,
        "adaln_eps": 1e-6,
        "noise_gate": "jump",
        "num_steps": 1,
        "time_grids": [[0.0, 1.0], [0.0, 0.5, 1.0]],
        "speculators_config": {"algorithm": "flashmap", "verifier": {"name_or_path": "toy/target"}},
    }
    vocab_maps = {"d2t": torch.arange(32), "t2d": torch.ones(64, dtype=torch.bool)}
    checkpoint = write(tmp_path / "speculators", source, {**weights, **vocab_maps})
    convert_checkpoint.convert(checkpoint, tmp_path / "out", dtype=torch.float32)
    check_release(
        tmp_path / "out", weights,
        target_layer_ids=TARGET_LAYER_IDS, target_model="toy/target", jump_grids={"1": [0.0, 1.0], "2": [0.0, 0.5, 1.0]},
    )
    flow = write(tmp_path / "flow", {**source, "flow_map": False}, weights)
    with pytest.raises(ValueError, match="stage 1"):
        convert_checkpoint.convert(flow, tmp_path / "out2", dtype=torch.float32)


def test_specforge_with_draft_vocab(tmp_path):
    config, weights = toy_weights()
    source = {**config.to_dict(), "flow_map": True, "draft_vocab_size": 64}
    source.pop("auto_map", None)
    checkpoint = write(tmp_path / "specforge", source, weights)
    ranking = {"ids": torch.randperm(64), "counts": torch.arange(64, 0, -1)}
    save_file(ranking, str(tmp_path / "ranking.safetensors"), {"samples": "50", "cache": "/private/path"})
    convert_checkpoint.convert(
        checkpoint, tmp_path / "out", draft_vocab=tmp_path / "ranking.safetensors", dtype=torch.float32
    )
    check_release(tmp_path / "out", weights, target_layer_ids=TARGET_LAYER_IDS, jump_grids=config.jump_grids)
    shipped = load_file(str(tmp_path / "out" / "draft_vocab.safetensors"))
    assert all(torch.equal(shipped[k], ranking[k]) for k in ranking)
    with safe_open(str(tmp_path / "out" / "draft_vocab.safetensors"), "pt") as handle:
        assert "cache" not in handle.metadata() and handle.metadata()["samples"] == "50"

    # vLLM's fields: a speculators-style header and the restricted draft vocabulary
    released = json.loads((tmp_path / "out" / "config.json").read_text())
    assert released["speculators_model_type"] == "flashmap"
    assert released["speculators_config"]["verifier"]["name_or_path"] == "toy/target"
    assert released["speculators_config"]["proposal_methods"][0]["speculative_tokens"] == 4
    assert released["transformer_layer_config"] == {} and released["draft_vocab_size"] == 64
    # a release converts again as itself, not as a speculators checkpoint
    convert_checkpoint.convert(tmp_path / "out", tmp_path / "again", dtype=torch.float32)
    check_release(tmp_path / "again", weights, target_layer_ids=TARGET_LAYER_IDS, jump_grids=config.jump_grids)
    assert "draft_vocab_size" not in json.loads((tmp_path / "again" / "config.json").read_text())


def test_wrong_weights_are_caught(tmp_path):
    config, weights = toy_weights()
    del weights["noise_gate"]
    source = {**config.to_dict(), "flow_map": True}
    checkpoint = write(tmp_path / "specforge", source, weights)
    with pytest.raises(ValueError, match="noise_gate"):
        convert_checkpoint.convert(checkpoint, tmp_path / "out", dtype=torch.float32)
