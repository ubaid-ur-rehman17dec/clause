import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()


# api/.env
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR/".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=ENV_FILE,
        extra="ignore"
    )

    # App
    app_name: str = "Clause"
    data_path: str = str(
        BASE_DIR / "files" / "shop_handbook.pdf"
    )

    # Chat
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "llama3.2:3b")
    chat_provider: str = os.getenv("RAG_CHAT_PROVIDER", "ollama")
    chat_temperature: float = float(os.getenv("RAG_CHAT_TEMPERATURE", 0.1))

    # Embeddings
    embeded_provider: str = os.getenv("RAG_EMBED_PROVIDER", "ollama")
    embeded_model: str = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text:latest")
    embeded_dim: int = int(os.getenv("RAG_EMBED_DIM", 768))

    # Indexing
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", 100))

    # Retrieval
    top_k: int = int(os.getenv("RAG_TOP_K", 3))
    threshold_score: float = float(os.getenv("RAG_SCORE_THRESHOLD", 0.5))

    # Qdrant
    collection: str = "shop_handbook"
    qdrant_data: str = str(BASE_DIR/"vectors_store")
    qdrant_url: str | None = None

    # Prompts
    system: str = (
        "You are a helpful AI Assistant. "
        "Answer the user's question using ONLY the provided context. "
        "Answer all parts clearly and directly. "
        "Do not repeat information. "
        "If the context does not contain the answer, "
        "say that you don't know."
    )


@lru_cache
def get_settings():
    return Settings()


def get_embeddings(settings: Settings) -> Embeddings:
    return init_embeddings(
        model=settings.embeded_model,
        provider=settings.embeded_provider
    )


def get_qdrant_client(settings: Settings):
    return QdrantClient(
        path=settings.qdrant_data
    )
