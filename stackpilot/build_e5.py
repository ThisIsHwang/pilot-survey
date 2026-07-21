from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _load_model_with_eager_attention(model_path: str, use_fp16: bool = False):
    """Match Search-R1's loader while avoiding cuDNN-backed SDPA."""
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    search_r1_root = project_root / "upstream" / "Search-R1"
    if not search_r1_root.is_dir():
        raise SystemExit(f"Search-R1 checkout is missing: {search_r1_root}")

    sys.path.insert(0, str(search_r1_root))
    from search_r1.search import index_builder  # noqa: PLC0415

    # The model is loaded with eager attention below. Disabling cuDNN SDPA as
    # well prevents a future upstream model change from selecting the failing
    # backend implicitly.
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    index_builder.load_model = _load_model_with_eager_attention
    print("E5 attention backend: eager (cuDNN SDPA disabled)")
    index_builder.main()


if __name__ == "__main__":
    main()
