from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings, get_embeddings, get_qdrant_client


def load(path: str) -> list[Document]:
     return PyPDFLoader(file_path=str(path)).load()

def ingest():
    settings = get_settings()

    # 1. Load
    file = Path(settings.data_path)
    docs = load(path=file)

    # 2. Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents=docs)

    # 3. Embed
    embed = get_embeddings(settings)

    # 4. Qdrant

    client = get_qdrant_client(settings)

    try:
        if client.collection_exists(collection_name=settings.collection):
            client.delete_collection(collection_name=settings.collection)

        client.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(
                size=settings.embeded_dim,
                distance=Distance.COSINE
            )
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection,
            embedding=embed
        )

        vector_store.add_documents(chunks)

    finally:
        client.close()

    return len(chunks)


if __name__ == "__main__":
    print(ingest(), "chunk indexed!")
