#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

SEARCH_R1_PYTHON=$ROOT/.venv-searchr1/bin/python
SEARCH_R1=${SEARCH_R1_ROOT:-$ROOT/upstream/Search-R1}
SEARCH_R1_COMMIT=${SEARCH_R1_COMMIT:-598e61bd1d36895726d28a8d06b3a15bed19f5d3}

[[ -x "$SEARCH_R1_PYTHON" ]] || {
  echo "Missing .venv-searchr1; run bash scripts/bootstrap_searchr1.sh" >&2
  exit 1
}
[[ -e "$SEARCH_R1/.git" ]] || {
  echo "Missing pinned Search-R1 checkout: $SEARCH_R1" >&2
  exit 1
}
if [[ "$(git -C "$SEARCH_R1" rev-parse HEAD)" != "$SEARCH_R1_COMMIT" ]]; then
  echo "Search-R1 is not at the required commit $SEARCH_R1_COMMIT" >&2
  exit 1
fi
RUNTIME_PATCH=$ROOT/searchr1_stage2/searchr1-runtime.patch
if ! git -C "$SEARCH_R1" apply --unidiff-zero --reverse --check "$RUNTIME_PATCH" \
  >/dev/null 2>&1; then
  echo "Search-R1 retrieval-timeout patch is missing; rerun bootstrap_searchr1.sh" >&2
  exit 1
fi
if ! nvcc --version | grep -Eq 'release 12\.9([, ]|$)'; then
  echo "Expected host CUDA toolkit 12.9." >&2
  nvcc --version >&2 || true
  exit 1
fi

min_shm_gib=${STAGE2_MIN_SHM_GIB:-8}
if [[ ! "$min_shm_gib" =~ ^[1-9][0-9]*$ ]]; then
  echo "STAGE2_MIN_SHM_GIB must be a positive integer; got '$min_shm_gib'." >&2
  exit 2
fi
available_shm_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
if (( available_shm_kib < min_shm_gib * 1024 * 1024 )); then
  echo "Stage 2 requires at least ${min_shm_gib} GiB free in /dev/shm; found $((available_shm_kib / 1024 / 1024)) GiB." >&2
  exit 1
fi

min_disk_gib=${STAGE2_MIN_DISK_GIB:-60}
if [[ ! "$min_disk_gib" =~ ^[1-9][0-9]*$ ]]; then
  echo "STAGE2_MIN_DISK_GIB must be a positive integer; got '$min_disk_gib'." >&2
  exit 2
fi
available_disk_kib=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
if (( available_disk_kib < min_disk_gib * 1024 * 1024 )); then
  echo "Stage 2 requires at least ${min_disk_gib} GiB free under $ROOT; found $((available_disk_kib / 1024 / 1024)) GiB." >&2
  exit 1
fi

"$SEARCH_R1_PYTHON" - <<'PY'
import importlib.metadata
import torch
import vllm._C  # noqa: F401
from flash_attn import flash_attn_func

expected = {
    "torch": "2.4.0",
    "vllm": "0.6.3",
    "transformers": "4.47.1",
    "tensordict": "0.5.0",
    "ray": "2.42.1",
    "flash-attn": "2.8.3",
}
for package, version in expected.items():
    actual = importlib.metadata.version(package).split("+")[0]
    if actual != version:
        raise SystemExit(f"Expected {package} {version}, found {actual}")
if torch.version.cuda != "12.1":
    raise SystemExit(f"Expected isolated Search-R1 cu121 wheel, found CUDA {torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"Expected eight visible H100s, found {torch.cuda.device_count()}")
for index in range(8):
    props = torch.cuda.get_device_properties(index)
    if "H100" not in props.name or props.total_memory < 70 * 1024**3:
        raise SystemExit(
            f"GPU {index} must be a full H100; found {props.name} "
            f"with {props.total_memory / 1024**3:.1f} GiB"
        )
q = torch.randn(
    1, 16, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
)
out = flash_attn_func(q, q, q, dropout_p=0.0, causal=True)
if out.shape != q.shape or not torch.isfinite(out).all():
    raise SystemExit("flash-attn forward H100 warm-up failed")
out.float().sum().backward()
if q.grad is None or not torch.isfinite(q.grad).all():
    raise SystemExit("flash-attn forward/backward H100 warm-up failed")
print(
    f"Search-R1 preflight passed: torch {torch.__version__} compatibility wheel; "
    f"{torch.cuda.device_count()} full H100s."
)
PY

echo "Host toolkit: CUDA 12.9; isolated legacy Search-R1 wheel runtime: CUDA 12.1 (expected)."
echo "Free /dev/shm: $((available_shm_kib / 1024 / 1024)) GiB; free disk: $((available_disk_kib / 1024 / 1024)) GiB."
