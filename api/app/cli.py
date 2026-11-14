from app.services.ingestion import ingest
from app.services.chat import RAGService

def main():
    rag_service = RAGService()
    try:
        query = input("You: ").strip()
        result = rag_service.ask(query)
        print(f"AI: {result.answer}")
        if result.sources:
            pass
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rag_service.close()

if __name__ == "__main__":
    ingest()
    main()
