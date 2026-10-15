"""Build a standalone FlashMap release folder from a trained checkpoint.

Three kinds of checkpoint are recognized by their config.json:

* DeepSpec format: checkpoints of the original research code (not public), built on
  DeepSpec, where the released drafters were trained. They have their own weight names,
  RMSNorm scales stored as ``scale`` for ``1 + scale``, fused key/value projections, and
  frozen copies of the target's embedding and lm_head.
* speculators (vLLM's training library): FlashMap's weight names, a speculators
  config (``speculators_model_type: "flashmap"``), and optionally ``d2t``/``t2d``
  draft-vocabulary maps, which the standalone model does not use.
* SpecForge (SGLang's training library), exported with ``specforge export --to hf``:
  FlashMap's weight names and config, without the model code.

    python scripts/convert_checkpoint.py --checkpoint <checkpoint> --output FlashMap-Qwen3-8B \
        --target-model Qwen/Qwen3-8B --model-card model_cards/FlashMap-Qwen3-8B.md \
        --draft-vocab <ranking>.safetensors

The output folder holds config.json, model.safetensors (bf16, without the target's
embedding and lm_head), the model code (so that `AutoModel.from_pretrained(...,
trust_remote_code=True)` works) and, optionally, the model card as README.md and
draft_vocab.safetensors, the target's token-frequency ranking that serving
integrations use to restrict the draft vocabulary. The saved weights are checked
against the source and against the names and shapes the model expects.

config.json also carries the fields with which vLLM and sglang serve the folder as
it is (with `pip install flashmap[vllm]` or `flashmap[sglang]`, see `vllm_fields`
and `sglang_fields`).
"""

import argparse
import inspect
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flashmap import FlashMapConfig, FlashMapDraftModel  # noqa: E402

REMOTE_CODE = ("configuration_flashmap.py", "modeling_flashmap.py")
# Weights the target provides at runtime, and maps (d2t, t2d) that only serving uses.
DROPPED = ("embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight", "d2t", "t2d")

# ---- DeepSpec ---------------------------------------------------------------------

# DeepSpec RMSNorms scale by (1 + scale); FlashMap's by weight = 1 + scale.
NORMS = {
    "hidden_norm.scale": "hidden_norm.weight",
    "anchor_norm.scale": "anchor_norm.weight",
    "state_norm.scale": "state_norm.weight",
    "noise_norm.scale": "noise_norm.weight",
}


def rename(name: str) -> str | None:
    """The FlashMap name of a DeepSpec weight (None: split or dropped)."""
    if name in NORMS:
        return NORMS[name]
    if name.startswith(("fc.", "anchor_proj.")) or name == "noise_gate":
        return name
    if not name.startswith("denoiser."):
        return None
    name = name[len("denoiser."):]
    for old, new in (
        ("x_embed.", "x_embed."),
        ("temb.mlp.0.", "time_embed.linear_1."),
        ("temb.mlp.2.", "time_embed.linear_2."),
        ("temb_jump.mlp.0.", "jump_embed.linear_1."),
        ("temb_jump.mlp.2.", "jump_embed.linear_2."),
        ("final_norm.cond_proj.1.", "final_adaln.linear."),
        ("head.", "head."),
    ):
        if name.startswith(old):
            return new + name[len(old):]
    if not name.startswith("blocks."):
        return None
    _, layer, rest = name.split(".", 2)
    prefix = f"layers.{layer}."
    for old, new in (
        ("adaln_self.cond_proj.1.", "self_attn_adaln.linear."),
        ("adaln_cross.cond_proj.1.", "cross_attn_adaln.linear."),
        ("adaln_mlp.cond_proj.1.", "mlp_adaln.linear."),
        ("mlp.0.", "mlp.fc1."),
        ("mlp.3.", "mlp.fc2."),
    ):
        if rest.startswith(old):
            return prefix + new + rest[len(old):]
    for attn in ("self_attn", "cross_attn"):
        for old, new in (
            ("wq.weight", "q_proj.weight"),
            ("wo.weight", "o_proj.weight"),
            ("qnorm.scale", "q_norm.weight"),
            ("knorm.scale", "k_norm.weight"),
        ):
            if rest == f"{attn}.{old}":
                return f"{prefix}{attn}.{new}"
    return None


def deepspec_weights(source: dict) -> dict:
    out = {}
    for name, tensor in source.items():
        if name in DROPPED:
            continue
        if name.endswith(".wkv.weight"):
            # keys and values are one projection in DeepSpec, keys first
            stem = rename(name.replace(".wkv.weight", ".wq.weight")).rsplit(".", 2)[0]
            keys, values = tensor.chunk(2, dim=0)
            out[f"{stem}.k_proj.weight"] = keys.contiguous()
            out[f"{stem}.v_proj.weight"] = values.contiguous()
            continue
        new = rename(name)
        if new is None:
            raise KeyError(f"no FlashMap name for DeepSpec weight {name!r}")
        # 1 + scale in the weight's own dtype, exactly as DeepSpec evaluates it
        out[new] = (1 + tensor) if name.endswith(".scale") else tensor
    return out


def decoding_error_grids(vocab_size: int, max_steps: int = 4) -> dict:
    """Jump grids of DeepSpec's decoding_error time schedule (Flow Map Language
    Models, eq. 25): t_n = t(n / N), ends set to exactly 0 and 1."""
    t = torch.linspace(0.0, 1.0, 4001, dtype=torch.float64)
    z = torch.linspace(-10.0, 10.0, 2001, dtype=torch.float64)
    log_phi = -0.5 * z**2 - 0.5 * math.log(2 * math.pi)
    c = t / (1.0 - t).clamp_min(torch.finfo(torch.float64).tiny)
    c = torch.where(t >= 1.0, torch.full_like(c, float("inf")), c)
    log_integrand = (vocab_size - 1) * torch.special.log_ndtr(z[None] + c[:, None]) + log_phi[None]
    error = (1.0 - torch.trapezoid(log_integrand.exp(), z, dim=-1)).clamp(0.0, 1.0)
    error0 = 1.0 - 1.0 / vocab_size
    tau = ((error0 - error) / error0).clamp(0.0, 1.0)
    tau[0], tau[-1] = 0.0, 1.0
    tau = torch.cummax(tau, dim=0).values
    grids = {}
    for steps in range(1, max_steps + 1):
        x = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
        idx = torch.searchsorted(tau, x, right=True).clamp(1, tau.numel() - 1)
        x0, x1, y0, y1 = tau[idx - 1], tau[idx], t[idx - 1], t[idx]
        w = torch.where(x1 > x0, (x - x0) / (x1 - x0), torch.zeros_like(x))
        grid = (y0 + w * (y1 - y0)).float()
        grid[0], grid[-1] = 0.0, 1.0
        grids[str(steps)] = grid.tolist()
    return grids


def from_deepspec(source: dict, target_model: str | None) -> dict:
    if source.get("state_featurization") != "posterior_noise" or source.get("time_schedule") != "decoding_error":
        raise ValueError("only DeepSpec flow maps (posterior_noise featurization, decoding_error schedule) convert")
    if not target_model:
        raise ValueError("a DeepSpec config does not name its target: pass --target-model")
    return dict(
        target_model=target_model,
        vocab_size=int(source["vocab_size"]),
        target_layer_ids=[int(i) for i in source["target_layer_ids"]],
        block_size=int(source["block_size"]),
        time_scale=float(source.get("dit_time_scale", 1000.0)),
        rope_theta=float(source.get("dit_rope_base", 10000.0)),  # its rope_theta is the target's
        noise_gate=source.get("noise_gate", "constant"),
        jump_grids=decoding_error_grids(int(source["vocab_size"])),
    )


# ---- speculators and SpecForge ----------------------------------------------------


def from_speculators(source: dict, target_model: str | None) -> dict:
    if not source.get("flow_map", True):
        raise ValueError("this is a FlashMap flow (stage 1), not the distilled map that drafts")
    layer_config = source["transformer_layer_config"]
    rope = layer_config.get("rope_parameters") or {}
    verifier = (source.get("speculators_config") or {}).get("verifier") or {}
    return dict(
        target_model=target_model or verifier.get("name_or_path"),
        vocab_size=int(layer_config["vocab_size"]),
        # speculators counts the decoder layer whose input it captures: one later
        target_layer_ids=[int(i) - 1 for i in source["aux_hidden_state_layer_ids"]],
        block_size=int(source["block_size"]),
        time_scale=float(source.get("time_scale", 1000.0)),
        rope_theta=float(rope.get("rope_theta", layer_config.get("rope_theta", 10000.0))),
        rms_norm_eps=float(layer_config.get("rms_norm_eps", 1e-6)),
        qk_norm_eps=float(source.get("qk_norm_eps", 1e-5)),
        layer_norm_eps=float(source.get("adaln_eps", 1e-6)),
        noise_gate=source.get("noise_gate", "jump"),
        time_schedule=source.get("time_schedule", "decoding_error"),
        num_steps=int(source.get("num_steps", 1)),
        jump_grids={str(i): list(grid) for i, grid in enumerate(source["time_grids"], start=1)},
    )


def from_flashmap(source: dict, target_model: str | None) -> dict:
    if not source.get("flow_map", True):
        raise ValueError("this is a FlashMap flow (stage 1), not the distilled map that drafts")
    names = inspect.signature(FlashMapConfig.__init__).parameters.keys() - {"self", "kwargs"}
    fields = {key: source[key] for key in names if key in source}
    if target_model:
        fields["target_model"] = target_model
    return fields


# ---- serving ------------------------------------------------------------------------

# vLLM drafts over the first DRAFT_VOCAB_SIZE ids of draft_vocab.safetensors, when shipped.
DRAFT_VOCAB_SIZE = 65536


def vllm_fields(block_size: int, target_model: str, target_architectures: list, draft_vocab_size: int | None) -> dict:
    """config.json fields for vLLM: a speculators-style header (vLLM's format for draft
    models, which flashmap's vLLM plugin maps onto the fields above; the DiT's dimensions
    stay where they are, so its `transformer_layer_config` is empty) and the size of the
    restricted draft vocabulary, if any."""
    fields = {
        "speculators_model_type": "flashmap",
        "speculators_config": {
            "algorithm": "flashmap",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {"proposal_type": "greedy", "speculative_tokens": block_size, "accept_tolerance": 0.0, "verifier_accept_k": 1}
            ],
            "verifier": {"name_or_path": target_model, "architectures": list(target_architectures)},
        },
        "transformer_layer_config": {},
    }
    if draft_vocab_size:
        fields["draft_vocab_size"] = draft_vocab_size
    return fields


# Positions a served draft accepts: Qwen3's context with YaRN (and vLLM's default).
MAX_POSITION_EMBEDDINGS = 131072


def sglang_fields(target_model: str, vocab_size: int) -> dict:
    """config.json fields for sglang's DSPARK worker, which drafts with flashmap.sglang:
    the longest context it may serve (sglang refuses a target context longer than the
    draft's) and, under `dspark_config`, DSPARK's own fields: a rank-1 "vanilla" Markov
    head, which flashmap.sglang replaces by an identity (FlashMap has none), and the
    token DSPARK fills the block's slots with, which FlashMap never reads."""
    return {
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "dspark_config": {
            "markov_rank": 1,
            "markov_head_type": "vanilla",
            "mask_token_id": mask_token_id(target_model, vocab_size),
        },
    }


def mask_token_id(target_model: str, vocab_size: int) -> int:
    """A token id outside the target tokenizer's vocabulary (151669 for Qwen3): the
    first past it, or the last embedding row if the tokenizer is not reachable."""
    try:
        from transformers import AutoTokenizer

        tokens = len(AutoTokenizer.from_pretrained(target_model))
        if tokens < vocab_size:
            return tokens
    except Exception:  # noqa: BLE001
        pass
    return vocab_size - 1


def target_architectures(target_model: str) -> list:
    """The target's architectures from its config.json, or [] if it is not reachable
    (vLLM reads only the target's name)."""
    try:
        from transformers import PretrainedConfig

        return PretrainedConfig.get_config_dict(target_model)[0].get("architectures") or []
    except Exception:  # noqa: BLE001
        return []


# ---- all ----------------------------------------------------------------------------


def architecture(weights: dict) -> dict:
    """The sizes of the network, read off its weights."""
    hidden_size, target_hidden_size = weights["x_embed.weight"].shape
    cond_dim, time_embed_dim = weights["time_embed.linear_1.weight"].shape
    return dict(
        target_hidden_size=target_hidden_size,
        hidden_size=hidden_size,
        num_hidden_layers=1 + max(int(name.split(".")[1]) for name in weights if name.startswith("layers.")),
        num_attention_heads=hidden_size // weights["layers.0.self_attn.q_norm.weight"].shape[0],
        intermediate_size=weights["layers.0.mlp.fc1.weight"].shape[0],
        cond_dim=cond_dim,
        time_embed_dim=time_embed_dim,
    )


def check_weights(config: FlashMapConfig, weights: dict) -> None:
    """The weights are exactly the model's: the same names and shapes."""
    with torch.device("meta"):
        expected = {name: tuple(t.shape) for name, t in FlashMapDraftModel(config).state_dict().items()}
    got = {name: tuple(t.shape) for name, t in weights.items()}
    if got != expected:
        missing, unexpected = sorted(expected.keys() - got.keys()), sorted(got.keys() - expected.keys())
        wrong = sorted(name for name in expected.keys() & got.keys() if expected[name] != got[name])
        raise ValueError(f"weights do not fit the model: missing {missing}, unexpected {unexpected}, shape {wrong}")


def save_draft_vocab(path, output: Path, target_model: str) -> None:
    """The ranking's tensors as they are; its metadata without local paths."""
    ranking = load_file(path)
    with safe_open(path, "pt") as handle:
        samples = (handle.metadata() or {}).get("samples")
    metadata = {
        "description": "target response-token frequency ranking: ids = token ids, most frequent first; "
        "counts[i] = occurrences of ids[i]",
        "target": target_model,
        **({"samples": samples} if samples else {}),
    }
    save_file({name: ranking[name] for name in ("counts", "ids")}, str(output / "draft_vocab.safetensors"), metadata)


def convert(checkpoint, output, target_model=None, model_card=None, draft_vocab=None, dtype=torch.bfloat16):
    checkpoint, output = Path(checkpoint), Path(output)
    source_config = json.loads((checkpoint / "config.json").read_text())
    source = load_file(str(checkpoint / "model.safetensors"))
    if source_config.get("model_type") == "flashmap":  # before speculators: a release has vllm_fields
        kind, fields, weights = "SpecForge/FlashMap", from_flashmap(source_config, target_model), source
    elif source_config.get("speculators_model_type") == "flashmap":
        kind, fields, weights = "speculators", from_speculators(source_config, target_model), source
    elif "dit_embed_dim" in source_config:
        kind, fields, weights = "DeepSpec", from_deepspec(source_config, target_model), deepspec_weights(source)
    else:
        raise ValueError(f"{checkpoint} is not a DeepSpec, speculators or SpecForge FlashMap checkpoint")
    weights = {k: t.to(dtype) if t.is_floating_point() else t for k, t in weights.items() if k not in DROPPED}
    config = FlashMapConfig(**{**fields, **architecture(weights)}, dtype=str(dtype).removeprefix("torch."))
    config.architectures = ["FlashMapDraftModel"]
    config.auto_map = {
        "AutoConfig": "configuration_flashmap.FlashMapConfig",
        "AutoModel": "modeling_flashmap.FlashMapDraftModel",
    }
    check_weights(config, weights)
    vocab_size = None
    if draft_vocab:
        with safe_open(draft_vocab, "pt") as handle:
            vocab_size = min(DRAFT_VOCAB_SIZE, handle.get_slice("ids").get_shape()[0])
    for key, value in vllm_fields(
        config.block_size, config.target_model, target_architectures(config.target_model), vocab_size
    ).items():
        setattr(config, key, value)
    for key, value in sglang_fields(config.target_model, config.vocab_size).items():
        setattr(config, key, value)

    output.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output / "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(output)
    for name in REMOTE_CODE:
        shutil.copy(REPO / "flashmap" / name, output / name)
    if model_card:
        shutil.copy(model_card, output / "README.md")
    if draft_vocab:
        save_draft_vocab(draft_vocab, output, config.target_model)

    # what was written is what was converted, and a DeepSpec conversion is its source renamed
    saved = load_file(str(output / "model.safetensors"))
    assert saved.keys() == weights.keys() and all(torch.equal(saved[k], weights[k]) for k in weights)
    if kind == "DeepSpec" and all(t.dtype == dtype for t in source.values()):
        verify_deepspec(source, saved)
    print(f"{kind} checkpoint -> {output}: {len(weights)} tensors, {sum(t.numel() for t in weights.values()) / 1e6:.1f}M")
    return config


def verify_deepspec(source: dict, converted: dict) -> None:
    """Every converted tensor is the DeepSpec tensor it came from, bit for bit."""
    for name, tensor in source.items():
        if name in DROPPED:
            continue
        if name.endswith(".wkv.weight"):
            stem = rename(name.replace(".wkv.weight", ".wq.weight")).rsplit(".", 2)[0]
            keys, values = tensor.chunk(2, dim=0)
            pairs = [(f"{stem}.k_proj.weight", keys), (f"{stem}.v_proj.weight", values)]
        elif name.endswith(".scale"):
            pairs = [(rename(name), (1.0 + tensor.float()).to(tensor.dtype))]
        else:
            pairs = [(rename(name), tensor)]
        for new, expected in pairs:
            assert converted[new].dtype == expected.dtype and torch.equal(converted[new], expected), new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="folder with config.json and model.safetensors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model", default=None, help="hub id of the target (required for DeepSpec)")
    parser.add_argument("--model-card", default=None, help="markdown file to write as README.md")
    parser.add_argument("--draft-vocab", default=None, help="token-frequency ranking to ship (ids, counts)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    convert(args.checkpoint, args.output, args.target_model, args.model_card, args.draft_vocab, dtype)


if __name__ == "__main__":
    main()
