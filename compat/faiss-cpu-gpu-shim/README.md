# FAISS GPU compatibility shim

RAGatouille declares a hard dependency on the `faiss-cpu` distribution even
though it only imports the common `faiss` Python module. Installing both the
CPU and GPU wheels would make them overwrite the same module files.

This metadata-only local distribution satisfies RAGatouille's dependency and
depends on `faiss-gpu`. It contains no Python modules; the real `faiss` module
is supplied entirely by the GPU wheel.
