"""Configuration of a FlashMap draft model."""

from transformers import PretrainedConfig


# Jump grids of the "decoding_error" time schedule (Flow Map Language Models,
# eq. 25) at the Qwen3 vocabulary size, with the endpoints set to exactly 0 and 1.
# A draft with ``n`` steps jumps from grid[i] to grid[i + 1] for i < n.
DEFAULT_JUMP_GRIDS = {
    "1": [0.0, 1.0],
    "2": [0.0, 0.8172569274902344, 1.0],
    "3": [0.0, 0.8010796308517456, 0.8310279846191406, 1.0],
    "4": [0.0, 0.7906050682067871, 0.8172569274902344, 0.8379518389701843, 1.0],
}


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
        target_layer_ids (`list[int]`): Target decoder layers whose outputs (that is,
            `hidden_states[i + 1]` of a transformers forward) are the drafter's context.
        block_size (`int`): Number of draft tokens per block.
        hidden_size (`int`): Width of the DiT.
        num_hidden_layers (`int`): Number of DiT blocks.
        num_attention_heads (`int`): Attention heads of the DiT.
        intermediate_size (`int`): MLP width of the DiT.
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
        target_layer_ids=(1, 9, 17, 25, 33),
        block_size: int = 16,
        hidden_size: int = 2048,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        intermediate_size: int = 8192,
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
        self.target_layer_ids = list(target_layer_ids)
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
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
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


__all__ = ["FlashMapConfig"]
