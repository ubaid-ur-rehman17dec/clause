import os
from app.core.config import get_settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance


def main():
    settings = get_settings()
    client = QdrantClient(path=settings.qdrant_data)
    if client.collection_exists(collection_name=settings.collection):
        print(f"Collection {settings.collection} already exists")
    else:
        client.recreate_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(size=settings.embeded_dim, distance=Distance.COSINE),
        )
        print(f"Created collection {settings.collection}")

if __name__ == "__main__":
    main()
from app.core.config import get_settings
from qdrant_client import QdrantClient


def main():
    settings = get_settings()
    client = QdrantClient(path=settings.qdrant_data)
    if client.collection_exists(collection_name=settings.collection):
        print(f"Collection {settings.collection} already exists")
    else:
        # Create collection with vector size and distance
        client.recreate_collection(
            collection_name=settings.collection,
            vectors_config={"size": settings.embeded_dim, "distance": "Cosine"},
        )
        print(f"Created collection {settings.collection}")

if __name__ == "__main__":
    main()
