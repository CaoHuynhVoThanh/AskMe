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
    settings = get_settings()
    if len(documents) <= top_k:
        return documents

    if not settings.enable_reranker:
        if settings.debug_reranker:
            print("[debug] reranker_disabled: using retriever order")
        return documents[:top_k]

    try:
        reranker = build_reranker()
        scores = reranker.predict(
            [[question, doc.page_content] for doc in documents]
        )
    except Exception as exc:
        build_reranker.cache_clear()
        if settings.debug_reranker:
            print(
                "[debug] reranker_failed:",
                {
                    "error": str(exc),
                    "fallback": f"using first {top_k} retriever documents",
                },
            )
        return documents[:top_k]

    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in ranked[:top_k]]
    if settings.debug_reranker:
        print(
            "[debug] reranker_loaded:",
            {
                "candidate_count": len(documents),
                "top_k": top_k,
                "model": settings.hf_reranker_model,
            },
        )
    return reranked
