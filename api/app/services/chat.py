from langchain.chat_models import init_chat_model
# from langchain.embeddings import init_embeddings
# from langchain.agents import create_agent
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from app.core.config import get_settings, get_embeddings, get_qdrant_client

No_ANSWER = "I don't know. The clause shop handbook does not contain any relevant information to answer your question. Please ask a different question or provide more context."

class Source(BaseModel):
    source: str
    page: int | None = None
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]

class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_qdrant_client(self.settings)
        if not self.client.collection_exists(collection_name=self.settings.collection):
            raise RuntimeError(f"Collection {self.settings.collection} not found")
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.settings.collection,
            embedding=get_embeddings(self.settings)
        )

        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            temperature=self.settings.chat_temperature,
        )

    def ask(self, question: str) -> Answer:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        results = self.store.similarity_search_with_score(
            query=question,
            k=self.settings.top_k,
            score_threshold=self.settings.threshold_score

        )

        if not results:
            return Answer(answer=No_ANSWER, sources=[])

        context = "\n\n".join([doc.page_content for doc, _ in results])

        messages = [
            {
                "role": "system",
                "content": self.settings.system
            },
            {
                "role": "user",
                "content": f"Context:{context}\nQuestion: {question}"
            }
        ]

        return Answer(
            answer=self.llm.invoke(messages).content,
            sources=[
                Source(
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page", None),
                    score=round(score, 4)
                )
                for doc, score in results
            ]
        )


    def close(self) -> None:
        self.client.close()
