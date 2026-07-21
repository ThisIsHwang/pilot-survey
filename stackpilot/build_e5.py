from __future__ import annotations

import sys
from pathlib import Path

from stackpilot.cuda_compat import (
    configure_cuda_attention,
    load_e5_with_eager_attention,
)


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
    configure_cuda_attention()
    index_builder.load_model = load_e5_with_eager_attention
    print("E5 attention backend: eager (cuDNN SDPA disabled)")
    index_builder.main()


if __name__ == "__main__":
    main()
