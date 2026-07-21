from __future__ import annotations

import sys
from types import ModuleType


def install_langchain_retriever_compat() -> None:
    """Restore the pre-1.0 LangChain path imported by RAGatouille 0.0.9."""
    try:
        from langchain.retrievers.document_compressors.base import (  # noqa: F401
            BaseDocumentCompressor,
        )

        return
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("langchain.retrievers"):
            raise

    import langchain
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from langchain_core.retrievers import BaseRetriever

    retrievers = ModuleType("langchain.retrievers")
    retrievers.__path__ = []  # type: ignore[attr-defined]
    retrievers.BaseRetriever = BaseRetriever  # type: ignore[attr-defined]

    compressors = ModuleType("langchain.retrievers.document_compressors")
    compressors.__path__ = []  # type: ignore[attr-defined]

    base = ModuleType("langchain.retrievers.document_compressors.base")
    base.BaseDocumentCompressor = BaseDocumentCompressor  # type: ignore[attr-defined]

    retrievers.document_compressors = compressors  # type: ignore[attr-defined]
    compressors.base = base  # type: ignore[attr-defined]
    langchain.retrievers = retrievers  # type: ignore[attr-defined]
    sys.modules[retrievers.__name__] = retrievers
    sys.modules[compressors.__name__] = compressors
    sys.modules[base.__name__] = base
