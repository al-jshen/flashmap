from transformers import AutoConfig, AutoModel

from .configuration_flashmap import FlashMapConfig
from .modeling_flashmap import FlashMapCache, FlashMapDraftModel, FlashMapGenerateOutput

# With the package imported, AutoModel loads FlashMap folders without trust_remote_code,
# and save_pretrained writes the model code next to the weights, so that the folder
# loads with trust_remote_code=True alone.
AutoConfig.register(FlashMapConfig.model_type, FlashMapConfig, exist_ok=True)
AutoModel.register(FlashMapConfig, FlashMapDraftModel, exist_ok=True)
FlashMapConfig.register_for_auto_class()
FlashMapDraftModel.register_for_auto_class("AutoModel")

__all__ = ["FlashMapCache", "FlashMapConfig", "FlashMapDraftModel", "FlashMapGenerateOutput"]
