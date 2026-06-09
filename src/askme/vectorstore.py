from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from askme.config import get_settings
from askme.embeddings import build_embeddings


def build_vectorstore() -> QdrantVectorStore:
    settings = get_settings()
    embeddings = build_embeddings()
    _ensure_collection(embeddings)

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
    )


def build_retriever():
    settings = get_settings()
    return build_vectorstore().as_retriever(
        search_kwargs={"k": settings.retriever_top_k},
    )


def _ensure_collection(embeddings) -> None:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    existing = {collection.name for collection in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        return

    vector_size = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
