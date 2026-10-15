"""FlashMap drafting in vLLM, as a vLLM general plugin (``pip install flashmap[vllm]``).

Installing flashmap registers ``register`` under the ``vllm.general_plugins`` entry
point, which vLLM calls in every process before it parses a model. It teaches vLLM,
without patching its source:

* to read a FlashMap release folder (``model_type: "flashmap"`` with
  ``speculators_model_type: "flashmap"`` in its config.json) as a draft model on
  vLLM's DFlash runtime (``method="dflash"``) with the ``FlashMapDraftModel``
  architecture; speculators-format FlashMap checkpoints load the same way;
* the ``FlashMapDraftModel`` architecture (model.py);
* the speculator that drives it on the V2 model runner (speculator.py), chosen by
  architecture the way vLLM chooses its DFlash2 speculator.

Every hook delegates to vLLM's own code unless a FlashMap checkpoint is in use. The
plugin supports vLLM 0.31.x; on other versions, or if registering fails, it logs a
warning and leaves vLLM untouched.

Serve with ``vllm serve <target> --speculative-config '{"model": "<FlashMap folder or
Hub id>", "num_speculative_tokens": 16}'``; the plugin fills in ``"method": "dflash"``.

The same plugin lives in the speculators fork (``speculators.models.flashmap.vllm``),
which also trains FlashMap drafters and is where it would be upstreamed to vLLM from.
"""

import logging
import sys

__all__ = ["FLASHMAP_ARCHITECTURE", "SUPPORTED_VLLM", "register", "update_flashmap"]

logger = logging.getLogger(__name__)

FLASHMAP_ARCHITECTURE = "FlashMapDraftModel"
# vLLM (major, minor) versions whose internals the plugin hooks into.
SUPPORTED_VLLM = ((0, 31),)

# Speculators config fields the vLLM draft config carries over.
_FLASHMAP_FIELDS = (
    "target_hidden_size",
    "draft_vocab_size",
    "block_size",
    "timestep_embed_dim",
    "timestep_frequency_dim",
    "time_scale",
    "qk_norm_eps",
    "adaln_eps",
    "noise_gate",
    "time_schedule",
    "num_steps",
    "time_grids",
)


# A released FlashMap config holds the DiT's dimensions at its top level; vLLM's
# draft config reads them under the transformer_layer_config names.
_RELEASE_DIT_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "rms_norm_eps",
)


def release_config_fields(config_dict: dict) -> dict:
    """The speculators-config view of a released FlashMap config.json
    (``model_type: "flashmap"``, served as it is): its DiT dimensions, the aux
    layers vLLM captures (DeepSpec's ``target_layer_ids`` + 1) and its field
    names mapped onto this plugin's."""
    dim, heads = config_dict["hidden_size"], config_dict["num_attention_heads"]
    grids = config_dict.get("jump_grids") or {}
    fields = {key: config_dict[key] for key in _RELEASE_DIT_FIELDS}
    fields.update(
        num_key_value_heads=heads,
        head_dim=dim // heads,
        hidden_act="gelu",
        tie_word_embeddings=False,
        max_position_embeddings=config_dict.get("max_position_embeddings", 131072),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(config_dict.get("rope_theta", 10000.0)),
        },
        aux_hidden_state_layer_ids=[i + 1 for i in config_dict["target_layer_ids"]],
        timestep_embed_dim=config_dict.get("cond_dim", 256),
        timestep_frequency_dim=config_dict.get("time_embed_dim"),
        adaln_eps=config_dict.get("layer_norm_eps", 1e-6),
        time_grids=[grids[str(n)] for n in range(1, len(grids) + 1)] or None,
    )
    return fields


def update_flashmap(config_dict: dict, pre_trained_config: dict) -> None:
    """Map a ``flashmap`` speculators config, or a released FlashMap config
    with ``speculators_model_type: "flashmap"``, onto the dict vLLM builds the
    draft model's config from (cf. vLLM's ``update_dflash``)."""
    if "aux_hidden_state_layer_ids" not in config_dict:
        released = release_config_fields(config_dict)
        config_dict = {**config_dict, **released}
        pre_trained_config.update(
            {
                key: released[key]
                for key in (
                    *_RELEASE_DIT_FIELDS,
                    "num_key_value_heads",
                    "head_dim",
                    "hidden_act",
                    "tie_word_embeddings",
                    "max_position_embeddings",
                    "rope_parameters",
                )
            }
        )
    if config_dict.get("flow_map") is False:
        raise ValueError(
            "This FlashMap checkpoint is a stage-1 flow (flow_map: false), which "
            "drafts only by integrating over many steps; serve the flow map "
            "distilled from it."
        )
    pre_trained_config["architectures"] = [FLASHMAP_ARCHITECTURE]
    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    pre_trained_config["dflash_config"] = {
        # FlashMap's slots are not embedded token ids; any valid id fills them.
        "mask_token_id": config_dict.get("mask_token_id") or 0,
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
        "causal": False,
    }
    for key in _FLASHMAP_FIELDS:
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]


def _init_speculator(init_speculator):
    def init_flashmap_or_other(vllm_config, device):
        spec = vllm_config.speculative_config
        if (
            spec is not None
            and spec.method == "dflash"
            and FLASHMAP_ARCHITECTURE in (spec.draft_model_config.architectures or [])
        ):
            from flashmap.vllm.speculator import (  # noqa: PLC0415
                FlashMapSpeculator,
            )

            return FlashMapSpeculator(vllm_config, device)
        return init_speculator(vllm_config, device)

    init_flashmap_or_other.flashmap = True  # type: ignore[attr-defined]
    return init_flashmap_or_other


def _flashmap_config(model: object, revision: str | None) -> dict | None:
    """The config.json of ``model`` if it is a FlashMap drafter vLLM can serve."""
    if not isinstance(model, str) or model in ("ngram", "[ngram]"):
        return None
    try:
        from transformers import PretrainedConfig  # noqa: PLC0415

        config_dict, _ = PretrainedConfig.get_config_dict(model, revision=revision)
    except Exception:  # noqa: BLE001
        return None
    if config_dict.get("speculators_model_type") != "flashmap":
        return None
    return config_dict


def _post_init(post_init):
    """Let ``--speculative-config '{"model": <FlashMap>}'`` leave out ``method``
    (FlashMap runs on vLLM's DFlash runtime) and ``num_speculative_tokens`` (the
    drafter's block size)."""

    def post_init_flashmap(self):
        if self.method is None and not self._is_custom_proposer_path(self.model):
            config_dict = _flashmap_config(self.model, self.revision)
            if config_dict is not None:
                self.method = "dflash"
                if self.num_speculative_tokens is None:
                    self.num_speculative_tokens = int(config_dict["block_size"])
        post_init(self)

    post_init_flashmap.flashmap = True  # type: ignore[attr-defined]
    return post_init_flashmap


def _vllm_version() -> tuple[int, int] | None:
    from vllm import __version__  # noqa: PLC0415

    try:
        major, minor = __version__.split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def register() -> None:
    """vLLM general plugin entry point (idempotent, never raises)."""
    try:
        version = _vllm_version()
        if version not in SUPPORTED_VLLM:
            logger.warning(
                "FlashMap's vLLM plugin supports vLLM %s, not %s; FlashMap "
                "checkpoints will not load.",
                " / ".join(f"{a}.{b}.x" for a, b in SUPPORTED_VLLM),
                version,
            )
            return
        _register()
    except Exception:  # noqa: BLE001
        logger.warning(
            "FlashMap's vLLM plugin failed to register; FlashMap checkpoints will "
            "not load.",
            exc_info=True,
        )


def _register() -> None:
    from vllm import ModelRegistry  # noqa: PLC0415
    from vllm.transformers_utils.configs.speculators import (  # noqa: PLC0415
        algos,
        base,
    )

    algos.SUPPORTED_SPECULATORS_TYPES["flashmap"] = update_flashmap

    from vllm.transformers_utils import config as config_utils  # noqa: PLC0415

    # vLLM reads a config.json as a speculators checkpoint only when it has no
    # model_type; a release folder's is "flashmap" (for transformers), so route
    # that model_type the same way: SpeculatorsConfig, through update_flashmap.
    config_utils._CONFIG_REGISTRY.setdefault("flashmap", "SpeculatorsConfig")  # noqa: SLF001
    config_utils._SPECULATIVE_DECODING_CONFIGS.add("flashmap")  # noqa: SLF001

    build = base.SpeculatorsConfig.build_vllm_speculative_config
    if not getattr(build, "flashmap", False):
        # FlashMap runs on vLLM's DFlash runtime.
        def build_vllm_speculative_config(cls, config_dict):
            result = build.__func__(cls, config_dict)
            if result["method"] == "flashmap":
                result["method"] = "dflash"
            return result

        build_vllm_speculative_config.flashmap = True  # type: ignore[attr-defined]
        base.SpeculatorsConfig.build_vllm_speculative_config = classmethod(  # type: ignore[method-assign]
            build_vllm_speculative_config
        )

    from vllm.config.speculative import SpeculativeConfig  # noqa: PLC0415

    if not getattr(SpeculativeConfig.__post_init__, "flashmap", False):
        SpeculativeConfig.__post_init__ = _post_init(  # type: ignore[method-assign]
            SpeculativeConfig.__post_init__
        )

    if FLASHMAP_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            FLASHMAP_ARCHITECTURE,
            "flashmap.vllm.model:FlashMapDraftModel",
        )

    from vllm.v1.worker.gpu import spec_decode  # noqa: PLC0415

    if not getattr(spec_decode.init_speculator, "flashmap", False):
        spec_decode.init_speculator = _init_speculator(spec_decode.init_speculator)
    # The model runner binds init_speculator at import.
    runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if runner is not None and not getattr(runner.init_speculator, "flashmap", False):
        runner.init_speculator = spec_decode.init_speculator
