"""Serve FlashMap drafts in sglang (``pip install "flashmap[sglang]"``).

sglang's DSPARK speculative worker drafts with :class:`.model.FlashMapDraftModel`,
which serves a FlashMap release folder as it is:

    sglang serve --model-path Qwen/Qwen3-8B \\
        --speculative-algorithm DSPARK \\
        --speculative-draft-model-path j-shen/FlashMap-Qwen3-8B

Installing the package registers :func:`register` as an sglang plugin (entry point
group ``sglang.srt.plugins``), which sglang runs at startup in every server
process. It registers the ``flashmap`` config with transformers and the model
with sglang's model registry, by naming this package in
``SGLANG_EXTERNAL_MODEL_PACKAGE`` (sglang then imports every module here and
registers its ``EntryClass``). Setting ``SGLANG_EXTERNAL_MODEL_PACKAGE=flashmap.sglang``
by hand does the same where sglang's plugins do not run.

The model relies on internals of one sglang release (``SUPPORTED_SGLANG``). On
another release the plugin leaves sglang untouched and logs why;
``FLASHMAP_SGLANG_ANY_VERSION=1`` tries anyway.
"""

import logging
import os
import sys

SUPPORTED_SGLANG = "0.5.20"

logger = logging.getLogger(__name__)


def _sglang_version() -> str:
    from importlib.metadata import version

    return version("sglang")


def check_sglang_version() -> None:
    """Raise unless the installed sglang is the release this package targets
    (``FLASHMAP_SGLANG_ANY_VERSION=1`` skips the check)."""
    found = _sglang_version()
    if found == SUPPORTED_SGLANG or os.environ.get("FLASHMAP_SGLANG_ANY_VERSION") == "1":
        return
    raise RuntimeError(
        f"flashmap.sglang needs sglang=={SUPPORTED_SGLANG} (it uses internals of its "
        f"DSPARK worker), found {found}: pip install sglang=={SUPPORTED_SGLANG}"
    )


def register() -> None:
    """The sglang plugin: makes FlashMap release folders servable as draft models.

    Idempotent. Loading it imported ``flashmap``, which registers the
    ``flashmap`` config with transformers (so sglang parses a release folder's
    config.json without ``--trust-remote-code``). It names this package in
    ``SGLANG_EXTERNAL_MODEL_PACKAGE``, which the server's processes inherit and
    sglang's model registry reads when it is first imported; a registry
    imported already, or one told to load another package, gets the model
    here."""
    try:
        check_sglang_version()
    except RuntimeError as error:
        logger.warning("FlashMap drafts are not available: %s", error)
        return
    current = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    if not current:
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = __name__
    if (current and current != __name__) or "sglang.srt.models.registry" in sys.modules:
        from sglang.srt.models.registry import ModelRegistry

        ModelRegistry.register(__name__, overwrite=True)


# sglang imports this package when SGLANG_EXTERNAL_MODEL_PACKAGE names it; a
# version mismatch then stops the server with the reason (sglang would otherwise
# log and skip the model module, and fail later on an unsupported architecture).
if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == __name__:
    check_sglang_version()
