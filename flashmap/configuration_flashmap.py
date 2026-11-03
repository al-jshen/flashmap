"""Configuration of a FlashMap draft model."""

from transformers import PretrainedConfig


# Jump grids of the "decoding_error" time schedule (Flow Map Language Models,
# eq. 25) at the Qwen3 vocabulary size, with the endpoints set to exactly 0 and 1.
# The schedule depends on the size of the vocabulary the flow lives on, so a
# drafter with a ``state_vocab_size`` carries its own grids.
# A draft with ``n`` steps jumps from grid[i] to grid[i + 1] for i < n.
DEFAULT_JUMP_GRIDS = {
    "1": [0.0, 1.0],
    "2": [0.0, 0.8172569274902344, 1.0],
    "3": [0.0, 0.8010796308517456, 0.8310279846191406, 1.0],
    "4": [0.0, 0.7906050682067871, 0.8172569274902344, 0.8379518389701843, 1.0],
}
MLP_TYPES = ("gelu", "swiglu")
# Settings a converted drafter's config may name; this package serves only these values.
FIXED_SETTINGS = {"sampler": "flow_map", "state_featurization": "posterior_noise"}


def jump_times(grid) -> list:
    """Per jump of a draft over ``grid`` (``0 = t_0 < ... < t_n = 1``): ``(s, t,
    next)``, the jump's network evaluation ``delta_{s,t}`` of the state at ``s``,
    and the time ``next`` the state moves to with it, ``x <- ((1 - next) x + (next
    - s) softmax(logits)) / (1 - s)`` (the last jump's logits are the draft's). As
    in Flow Map Language Models (Algorithm 8), each jump lands where the next starts:
    ``t = next``."""
    grid = [float(t) for t in grid]
    return [(s, nxt, nxt) for s, nxt in zip(grid[:-1], grid[1:])]


class FlashMapConfig(PretrainedConfig):
    r"""A FlashMap drafter for speculative decoding.

    FlashMap drafts a block of ``block_size`` tokens after the last verified one
    (the anchor) with a flow map: a DiT that jumps from Gaussian noise over the
    target's vocabulary to the block's per-slot token distributions, conditioned on
    the target model's hidden states through cross-attention.

    The drafter borrows the target's token embedding and lm_head, which are not part
    of its weights.

    Args:
        target_model (`str`): Hub id of the target model the drafter was trained for.
        target_hidden_size (`int`): Hidden size of the target model.
        vocab_size (`int`): Vocabulary size of the target model.
        state_vocab_size (`int`, *optional*): Size of the vocabulary the flow lives on
            (its noise, the posterior it embeds and the distributions it predicts).
            `None`: the target's whole vocabulary. Otherwise the drafter was trained
            on the target's `state_vocab_size` most frequent tokens, listed in id
            order by its `draft_ids` buffer (part of its weights); its drafts give
            every other token probability 0.
        target_layer_ids (`list[int]`): Target decoder layers whose outputs (that is,
            `hidden_states[i + 1]` of a transformers forward) are the drafter's context.
        block_size (`int`): Number of draft tokens per block.
        hidden_size (`int`): Width of the DiT.
        num_hidden_layers (`int`): Number of DiT blocks.
        num_attention_heads (`int`): Attention heads of the DiT.
        intermediate_size (`int`): MLP width of the DiT.
        mlp_type (`str`): The DiT's MLP: "gelu" (`fc2(gelu(fc1(x)))`, with biases) or
            "swiglu" (`down_proj(silu(gate) * up)` with `gate, up` the two halves of
            `gate_up_proj(x)`, without biases).
        cond_dim (`int`): Width of the time conditioning.
        time_embed_dim (`int`): Size of the sinusoidal time features.
        time_scale (`float`): Times in [0, 1] are multiplied by this before their
            sinusoidal embedding.
        rope_theta (`float`): RoPE base. RoPE rotates interleaved pairs of channels.
        rms_norm_eps (`float`): Epsilon of the context, anchor and state RMSNorms.
        qk_norm_eps (`float`): Epsilon of the query/key RMSNorms.
        layer_norm_eps (`float`): Epsilon of the adaptive LayerNorms.
        noise_gate (`str`): How the noise term of the state embedding is weighted:
            "jump" by `noise_gate * (t - s)` for a jump from `s` to `t`, "constant" by
            `noise_gate` alone.
        time_schedule (`str`): The schedule the jump grids come from.
        num_steps (`int`): Default number of jumps per draft.
        jump_grids (`dict[str, list[float]]`): Jump times for each number of steps.
    """

    model_type = "flashmap"

    def __init__(
        self,
        target_model: str = "Qwen/Qwen3-8B",
        target_hidden_size: int = 4096,
        vocab_size: int = 151936,
        state_vocab_size=None,
        target_layer_ids=(1, 9, 17, 25, 33),
        block_size: int = 16,
        hidden_size: int = 2048,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        intermediate_size: int = 8192,
        mlp_type: str = "gelu",
        cond_dim: int = 256,
        time_embed_dim: int = 256,
        time_scale: float = 1000.0,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        qk_norm_eps: float = 1e-5,
        layer_norm_eps: float = 1e-6,
        noise_gate: str = "jump",
        time_schedule: str = "decoding_error",
        num_steps: int = 1,
        jump_grids=None,
        **kwargs,
    ):
        self.target_model = target_model
        self.target_hidden_size = target_hidden_size
        self.vocab_size = vocab_size
        if state_vocab_size is not None and not 0 < int(state_vocab_size) <= vocab_size:
            raise ValueError(f"state_vocab_size must be in (0, {vocab_size}], got {state_vocab_size}")
        self.state_vocab_size = None if state_vocab_size is None else int(state_vocab_size)
        self.target_layer_ids = list(target_layer_ids)
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        if mlp_type not in MLP_TYPES:
            raise ValueError(f"mlp_type must be one of {MLP_TYPES}, got {mlp_type!r}")
        self.mlp_type = mlp_type
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.time_scale = time_scale
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm_eps = qk_norm_eps
        self.layer_norm_eps = layer_norm_eps
        self.noise_gate = noise_gate
        self.time_schedule = time_schedule
        self.num_steps = num_steps
        self.jump_grids = dict(DEFAULT_JUMP_GRIDS if jump_grids is None else jump_grids)
        for name, value in FIXED_SETTINGS.items():
            if kwargs.pop(name, value) != value:
                raise ValueError(f"this version of flashmap serves only {name}={value!r}")
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


__all__ = ["FlashMapConfig", "jump_times"]
