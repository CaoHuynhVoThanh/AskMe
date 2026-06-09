from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from askme.config import get_settings


@lru_cache
def build_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(
        settings.hf_reranker_model,
        max_length=settings.reranker_max_length,
        device=settings.hf_device,
    )


def rerank_documents(
    documents: list[Document], question: str, top_k: int = 1
) -> list[Document]:
    """Rerank documents using CrossEncoder for better relevance."""
    if len(documents) <= top_k:
        return documents

    reranker = build_reranker()

    # Score all documents
    scores = reranker.predict(
        [[question, doc.page_content] for doc in documents]
    )

    # Sort by score and return top_k
    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
