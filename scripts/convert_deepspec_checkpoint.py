"""Convert a DeepSpec flow-map checkpoint into a FlashMap release folder.

FlashMap was trained under the name "DFlashier flow map" in a fork of DeepSpec
(github.com/al-jshen/DeepSpec; upstream github.com/deepseek-ai/DeepSpec), which saves
the drafter together with frozen copies of the target's embedding and lm_head. The
release keeps only the drafter's own weights, renamed to the FlashMap modules, and
loads the rest from the target at runtime:

    python scripts/convert_deepspec_checkpoint.py \
        --checkpoint <deepspec run>/step_23268 --target-model Qwen/Qwen3-8B \
        --output FlashMap-Qwen3-8B --model-card model_cards/FlashMap-Qwen3-8B.md

The output folder holds config.json, model.safetensors (bf16), the remote-code
files (so `AutoModel.from_pretrained(..., trust_remote_code=True)` works) and,
optionally, the model card as README.md. Every converted tensor is checked
against the source after writing.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flashmap.configuration_flashmap import FlashMapConfig  # noqa: E402

REMOTE_CODE = ("configuration_flashmap.py", "modeling_flashmap.py")
# The target's own weights, loaded from the target at runtime.
TARGET_WEIGHTS = ("embed_tokens.weight", "lm_head.weight")
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


def convert_state_dict(source: dict) -> dict:
    out = {}
    for name, tensor in source.items():
        if name in TARGET_WEIGHTS:
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
        is_norm = name.endswith(".scale")
        # 1 + scale in the weight's own dtype, exactly as DeepSpec evaluates it
        out[new] = (1 + tensor) if is_norm else tensor
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


def build_config(source_config: dict, target_model: str) -> FlashMapConfig:
    expected = {
        "state_featurization": "posterior_noise",
        "noise_gate": "jump",
        "time_schedule": "decoding_error",
    }
    for key, value in expected.items():
        if source_config.get(key) != value:
            raise ValueError(f"only {key}={value!r} checkpoints convert, got {source_config.get(key)!r}")
    embed_dim = int(source_config["dit_embed_dim"])
    config = FlashMapConfig(
        target_model=target_model,
        target_hidden_size=int(source_config["hidden_size"]),
        vocab_size=int(source_config["vocab_size"]),
        target_layer_ids=[int(i) for i in source_config["target_layer_ids"]],
        block_size=int(source_config["block_size"]),
        hidden_size=embed_dim,
        num_hidden_layers=int(source_config["num_hidden_layers"]),
        num_attention_heads=int(source_config["dit_num_heads"]),
        intermediate_size=int(float(source_config.get("dit_mlp_ratio", 4.0)) * embed_dim),
        cond_dim=int(source_config["dit_cond_dim"]),
        time_scale=float(source_config.get("dit_time_scale", 1000.0)),
        rope_theta=float(source_config.get("dit_rope_base", 10000.0)),
        num_steps=1,
        jump_grids=decoding_error_grids(int(source_config["vocab_size"])),
        dtype="bfloat16",
    )
    config.architectures = ["FlashMapDraftModel"]
    config.auto_map = {
        "AutoConfig": "configuration_flashmap.FlashMapConfig",
        "AutoModel": "modeling_flashmap.FlashMapDraftModel",
    }
    return config


def verify(source: dict, output_dir: Path) -> None:
    """Every released tensor is the DeepSpec tensor it came from, bit for bit."""
    converted = load_file(str(output_dir / "model.safetensors"))
    seen = set()
    for name, tensor in source.items():
        if name in TARGET_WEIGHTS:
            continue
        if name.endswith(".wkv.weight"):
            stem = rename(name.replace(".wkv.weight", ".wq.weight")).rsplit(".", 2)[0]
            keys, values = tensor.chunk(2, dim=0)
            pairs = [(f"{stem}.k_proj.weight", keys), (f"{stem}.v_proj.weight", values)]
        elif name.endswith(".scale"):
            expected = (1.0 + tensor.float()).to(tensor.dtype)
            assert torch.equal(1 + tensor, expected)
            pairs = [(rename(name), expected)]
        else:
            pairs = [(rename(name), tensor)]
        for new, expected in pairs:
            got = converted[new]
            assert got.dtype == expected.dtype and torch.equal(got, expected), new
            seen.add(new)
    assert seen == set(converted), sorted(set(converted) ^ seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="DeepSpec step directory")
    parser.add_argument("--target-model", required=True, help="hub id of the target, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-card", default=None, help="markdown file to write as README.md")
    args = parser.parse_args()

    checkpoint, output = Path(args.checkpoint), Path(args.output)
    source_config = json.loads((checkpoint / "config.json").read_text())
    source = load_file(str(checkpoint / "model.safetensors"))
    output.mkdir(parents=True, exist_ok=True)

    config = build_config(source_config, args.target_model)
    state = convert_state_dict(source)
    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(output)
    for name in REMOTE_CODE:
        shutil.copy(REPO / "flashmap" / name, output / name)
    if args.model_card:
        shutil.copy(args.model_card, output / "README.md")

    verify(source, output)
    params = sum(t.numel() for t in state.values())
    print(f"wrote {output}: {len(state)} tensors, {params / 1e6:.1f}M parameters, verified")


if __name__ == "__main__":
    main()
