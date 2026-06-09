from qdrant_client import QdrantClient

from askme.config import get_settings


def main() -> None:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    
    try:
        client.delete_collection(collection_name=settings.qdrant_collection)
        print(f"✅ Deleted collection: {settings.qdrant_collection}")
    except Exception as e:
        print(f"⚠️ Collection not found or error: {e}")


if __name__ == "__main__":
    main()
