"""Fail-closed local chat-template defaults for the Qwen3.5 five-day run.

Python imports ``sitecustomize`` automatically when this directory is placed at
the front of ``PYTHONPATH``. The patch is deliberately gated by an environment
variable so normal repository jobs remain unchanged.
"""

from __future__ import annotations

import os


def _enabled() -> bool:
    return os.environ.get("STACKPILOT_QWEN35_NO_THINK", "0") == "1"


if _enabled():
    from transformers import AutoModelForCausalLM
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not getattr(PreTrainedTokenizerBase.apply_chat_template, "_stackpilot_no_think", False):
        _original_apply_chat_template = PreTrainedTokenizerBase.apply_chat_template

        def _apply_chat_template_no_think(self, conversation, *args, **kwargs):
            # Qwen3.5 uses the chat-template boolean rather than a prompt suffix.
            # Reject an explicit request for thinking instead of silently
            # producing mixed-mode prompts inside the scientific run.
            requested = kwargs.get("enable_thinking")
            if requested not in (None, False):
                raise RuntimeError(
                    "Qwen3.5 five-day experiment forbids enable_thinking=True"
                )
            kwargs["enable_thinking"] = False
            return _original_apply_chat_template(self, conversation, *args, **kwargs)

        _apply_chat_template_no_think._stackpilot_no_think = True
        PreTrainedTokenizerBase.apply_chat_template = _apply_chat_template_no_think

    # Transformers 5 renamed the preferred loading argument to ``dtype``.
    # Existing shared StackPilot code still passes ``torch_dtype``; translate
    # it only inside this isolated environment.
    if not getattr(AutoModelForCausalLM.from_pretrained, "_stackpilot_dtype_bridge", False):
        _original_from_pretrained = AutoModelForCausalLM.from_pretrained

        def _from_pretrained_dtype_bridge(*args, **kwargs):
            if "torch_dtype" in kwargs and "dtype" not in kwargs:
                kwargs["dtype"] = kwargs.pop("torch_dtype")
            kwargs.setdefault("low_cpu_mem_usage", True)
            return _original_from_pretrained(*args, **kwargs)

        _from_pretrained_dtype_bridge._stackpilot_dtype_bridge = True
        AutoModelForCausalLM.from_pretrained = staticmethod(
            _from_pretrained_dtype_bridge
        )
