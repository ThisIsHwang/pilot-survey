from __future__ import annotations

import argparse
import json
from pathlib import Path

from stackpilot.ragatouille_compat import install_langchain_retriever_compat


def main() -> None:
    install_langchain_retriever_compat()
    from ragatouille import RAGPretrainedModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index-name", default="hotpot_pilot_colbert")
    parser.add_argument("--model", default="colbert-ir/colbertv2.0")
    parser.add_argument("--index-root", required=True)
    args = parser.parse_args()

    documents, ids, metadata = [], [], []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            documents.append(row["contents"])
            ids.append(row["id"])
            metadata.append({"title": row["title"]})

    Path(args.index_root).mkdir(parents=True, exist_ok=True)
    rag = RAGPretrainedModel.from_pretrained(args.model, index_root=args.index_root)
    path = rag.index(
        index_name=args.index_name,
        collection=documents,
        document_ids=ids,
        document_metadatas=metadata,
        split_documents=False,
        overwrite_index=True,
    )
    print(path)


if __name__ == "__main__":
    main()
