# LlamaIndex core compatibility shim

RAGatouille declares the `llama-index` meta-distribution, but only imports
`llama_index.core.Document` and `llama_index.core.text_splitter.SentenceSplitter`.
The meta-distribution installs unrelated readers, agents, embeddings, and LLM
plugins.

This metadata-only local distribution satisfies RAGatouille's dependency and
depends on the pinned `llama-index-core` package. It contains no Python modules.
