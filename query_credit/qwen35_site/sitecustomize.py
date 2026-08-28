"""Fail-closed local chat-template defaults for the Qwen3.5 weekend run.

Python imports ``sitecustomize`` automatically when this directory is placed at
front of ``PYTHONPATH``. The patch is deliberately gated by an environment
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
            # Qwen3.5 does not support the old /nothink soft switch. Its local
            # tokenizer template reads this boolean directly.
            kwargs.setdefault("enable_thinking", False)
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
