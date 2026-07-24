from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

PAGED_FLAT_LOAD_MODE = "paged-fp32-flat"
PAGED_FLAT_STORAGE_DTYPE = "float32"


@dataclass
class PagedFlatGpuLoad:
    temp_memory_mib: int
    resources: list[Any] = field(default_factory=list)
    documents: int | None = None
    index_bytes: int | None = None
    dimension: int | None = None

    @property
    def mode(self) -> str:
        return PAGED_FLAT_LOAD_MODE

    @property
    def storage_dtype(self) -> str:
        return PAGED_FLAT_STORAGE_DTYPE


def _paged_flat_index_to_gpu(
    faiss: Any,
    cpu_index: Any,
    clone_options: Any,
    state: PagedFlatGpuLoad,
) -> Any:
    if int(faiss.get_num_gpus()) != 1:
        raise RuntimeError(
            "The paged FAISS loader requires exactly one visible GPU; "
            f"found {faiss.get_num_gpus()}"
        )
    if clone_options is None:
        raise RuntimeError("The paged FAISS loader requires clone options")
    if not bool(getattr(clone_options, "shard", False)):
        raise RuntimeError("The paged FAISS loader requires shard=True")

    index = (
        faiss.downcast_index(cpu_index)
        if hasattr(faiss, "downcast_index")
        else cpu_index
    )
    flat_types = tuple(
        index_type
        for index_type in (
            getattr(faiss, "IndexFlatIP", None),
            getattr(faiss, "IndexFlatL2", None),
        )
        if index_type is not None
    )
    if flat_types and not isinstance(index, flat_types):
        raise TypeError(
            "The paged FAISS loader supports only IndexFlatIP/IndexFlatL2; "
            f"got {type(index).__name__}"
        )
    if not hasattr(index, "get_xb"):
        raise TypeError(
            "The paged FAISS loader requires a flat CPU index with get_xb()"
        )

    metric = int(index.metric_type)
    if metric == int(faiss.METRIC_INNER_PRODUCT):
        gpu_index_type = faiss.GpuIndexFlatIP
    elif metric == int(faiss.METRIC_L2):
        gpu_index_type = faiss.GpuIndexFlatL2
    else:
        raise TypeError(
            "The paged FAISS loader supports only inner-product or L2 metrics; "
            f"got metric {metric}"
        )

    resources = faiss.StandardGpuResources()
    resources.setTempMemory(state.temp_memory_mib * 1024 * 1024)
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    # Search-R1 requests FP16 in its generic FAISS clone path. The Hard-RQ0
    # backend deliberately overrides it: one H100 80GB can hold the 60.1 GiB
    # wiki-18 flat index in FP32, avoiding half-storage near-tie rank swaps.
    config.useFloat16 = False
    if hasattr(config, "use_cuvs"):
        # The native GpuIndex::add path pages large host inputs. cuVS bypasses
        # that path in FAISS 1.14.x and recreates the full-transfer OOM.
        config.use_cuvs = False

    documents = int(index.ntotal)
    dimension = int(index.d)
    state.resources.append(resources)
    gpu_index = gpu_index_type(resources, dimension, config)
    if not hasattr(gpu_index, "add_c"):
        raise RuntimeError(
            "This FAISS build does not expose the native add_c API required "
            "for memory-safe paged loading"
        )

    print(
        "Loading FAISS IndexFlat on the single visible GPU with native paged "
        f"FP32 add: documents={documents:,}, dimension={dimension}, "
        f"scratch={state.temp_memory_mib} MiB"
    )
    # One call is intentional. GpuIndexFlat::add first reserves the complete
    # FP32 destination and then GpuIndex::add pages the 60 GiB host matrix.
    # Repeating small Python add calls would cause destination reallocations.
    gpu_index.add_c(documents, index.get_xb())
    if hasattr(resources, "syncDefaultStreamCurrentDevice"):
        resources.syncDefaultStreamCurrentDevice()
    if int(gpu_index.ntotal) != documents:
        raise RuntimeError(
            "Paged FAISS load produced an incomplete GPU index: "
            f"{gpu_index.ntotal:,} != {documents:,}"
        )

    state.documents = documents
    state.dimension = dimension
    state.index_bytes = documents * dimension * 4
    print(
        "Paged FAISS GPU load complete: "
        f"{documents:,} vectors, {state.index_bytes / (1024**3):.2f} GiB FP32"
    )
    return gpu_index


@contextmanager
def paged_flat_gpu_loader(
    faiss: Any, *, temp_memory_mib: int = 512
) -> Iterator[PagedFlatGpuLoad]:
    if temp_memory_mib <= 0:
        raise ValueError("FAISS temporary memory must be a positive MiB value")
    if int(faiss.get_num_gpus()) != 1:
        raise RuntimeError(
            "The paged FAISS loader requires exactly one visible GPU; "
            f"found {faiss.get_num_gpus()}"
        )

    original = faiss.index_cpu_to_all_gpus
    state = PagedFlatGpuLoad(temp_memory_mib=temp_memory_mib)

    def load(index: Any, co: Any = None, ngpu: int = -1) -> Any:
        if int(ngpu) not in {-1, 1}:
            raise RuntimeError(
                "The paged FAISS loader accepts only the single visible GPU; "
                f"ngpu={ngpu}"
            )
        return _paged_flat_index_to_gpu(faiss, index, co, state)

    faiss.index_cpu_to_all_gpus = load
    try:
        yield state
    finally:
        faiss.index_cpu_to_all_gpus = original
