from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def configure_cuda_attention() -> None:
    """Avoid the cuDNN SDPA path that fails on the target CUDA 12.9 stack."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def load_e5_with_eager_attention(model_path: str, use_fp16: bool = False):
    """Match Search-R1's loader while forcing deterministic eager attention."""
    configure_cuda_attention()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    return model, tokenizer
