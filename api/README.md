# Clause API

The backend for Clause, a customer-support agent that answers questions from the shop handbook.

A FastAPI app serves `ClauseAgent`, a LangChain tool-calling agent. The agent has one tool, `search_handbook`, which runs hybrid (dense + BM25) search over a Qdrant collection built from the handbook PDF. Conversations are checkpointed to SQLite, and a middleware stack keeps every question within call and context limits. By default everything runs locally with Ollama.

## Contents

- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Ingest the handbook](#ingest-the-handbook)
- [Run the API](#run-the-api)
- [API reference](#api-reference)
- [Chat from the terminal](#chat-from-the-terminal)
- [Agent internals](#agent-internals)
- [Prompts](#prompts)
- [Evals](#evals)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Tech stack

- **FastAPI**: HTTP API and interactive docs
- **LangChain**: `create_agent`, agent middleware, `init_chat_model` and `init_embeddings`
- **LangGraph checkpoint (SQLite)**: per-thread conversation memory
- **Ollama**: local chat model and dense embeddings (`nomic-embed-text`, 768-dim)
- **Google Gemini** (optional): hosted chat model through `langchain-google-genai`
- **FastEmbed**: sparse BM25 embeddings (`Qdrant/bm25`)
- **Qdrant**: hybrid dense + sparse vector store (embedded on disk, or a server)
- **Pydantic v2 / pydantic-settings**: request and response schemas, and configuration from `.env`

## Project structure

```
api/
├── app/
│   ├── main.py                    # FastAPI app: lifespan, GET /, POST /ask
│   ├── cli.py                     # Interactive terminal chat
│   ├── core/
│   │   └── config.py              # Settings (RAG_ prefix), embeddings, Qdrant client
│   ├── llm/
│   │   └── models.py              # Chat model factory (init_chat_model)
│   ├── rag/
│   │   └── vectorstore.py         # Hybrid QdrantVectorStore (dense + BM25)
│   ├── tools/
│   │   └── knowledge/
│   │       └── handbook.py        # search_handbook tool (content + source artifact)
│   ├── middleware/
│   │   ├── stacks.py              # Default middleware stack (limits, context, summary)
│   │   └── observability.py       # LoggingMiddleware (stub, not wired in yet)
│   ├── prompts/
│   │   ├── loader.py              # get_prompt(name, version)
│   │   └── support/
│   │       └── system/v3.md       # Active system prompt
│   ├── schemas/                   # Pydantic models: AskRequest, AgentReply, ToolCall, Source
│   └── services/
│       ├── agent.py               # ClauseAgent: agent + tools + middleware + memory
│       ├── ingestion.py           # Load → chunk → embed (dense + BM25) → store
│       └── chat.py                # Legacy RAGService (commented out, kept for reference)
├── data/
│   ├── shop_handbook.pdf          # Source document that gets ingested
│   └── shop_handbook.md           # Markdown copy of the handbook
├── evals/
│   ├── golden.jsonl               # Golden test set used by run.py
│   ├── golden_v2.jsonl            # Larger test set (50 cases)
│   ├── run.py                     # Eval harness
│   └── results/                   # Timestamped JSON reports
├── vector_stores/                 # Embedded Qdrant data (created by ingestion, not tracked)
├── memory.db                      # Conversation checkpoints (created on first run)
├── .env.example
└── pyproject.toml
```

Empty packages such as `agents/`, `api/`, `guardrails/`, `memory/` and `tools/mcp/` are placeholders for upcoming work.

## Getting started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), running locally

Pull the local models:

```bash
ollama pull qwen3.5:4b          # chat model (only needed if you use Ollama for chat)
ollama pull nomic-embed-text    # dense embeddings (always needed)
```

FastEmbed downloads the BM25 model (`Qdrant/bm25`) automatically the first time it is used.

### Install

Run every command in this guide from the `api/` folder.

```bash
cd api
uv sync
cp .env.example .env
```

Open `.env` and choose a chat model:

- **Local (Ollama):** set `RAG_CHAT_PROVIDER=ollama` and `RAG_CHAT_MODEL=qwen3.5:4b`.
- **Hosted (Gemini):** keep `RAG_CHAT_PROVIDER=google_genai`, set a Gemini model name and put your key in `RAG_CHAT_API_KEY`.

Keep `RAG_SYSTEM_PROMPT_VERSION=v3`. It is the only system prompt with content (see [Prompts](#prompts)).

> On Intel Macs, `pyproject.toml` pins `onnxruntime<1.24`, the last release with x86_64 macOS wheels, which FastEmbed needs.

## Ingest the handbook

This command builds the Qdrant collection from `RAG_DATA_PATH` (default `data/shop_handbook.pdf`). Each chunk gets a `dense` vector and a `sparse` BM25 vector.

```bash
uv run python -m app.services.ingestion
# prints: <N> chunks indexed
```

Run it once, and again whenever the PDF or the chunking or embedding settings change. Each run deletes the collection and rebuilds it. Before deleting anything, it checks that the embedding model's output size matches `RAG_EMBED_DIM` and stops with an error if it doesn't.

The API refuses to start if the collection doesn't exist (`RuntimeError: Collection shop_handbook not found!`).

## Run the API

```bash
uv run fastapi dev app/main.py
```

The API runs at http://127.0.0.1:8000, with interactive docs at http://127.0.0.1:8000/docs.

One `ClauseAgent` is created when the app starts (FastAPI lifespan). Every request shares it, and its Qdrant client is closed on shutdown.

## API reference

| Method | Path   | Description                         |
|--------|--------|-------------------------------------|
| GET    | `/`    | Health check: `{"status": "ok"}`    |
| POST   | `/ask` | Ask the agent a question            |

### `POST /ask`

**Request body** (`AskRequest`)

| Field       | Type            | Required | Notes                                                       |
|-------------|-----------------|----------|-------------------------------------------------------------|
| `question`  | string          | yes      | Whitespace is stripped. Must be 1–2000 characters long.     |
| `thread_id` | string \| null  | no       | Conversation ID. See the note below.                        |

Unknown fields are rejected with `422`, because every schema uses `extra="forbid"`.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do I have to ask for a refund?"}'
```

**Response body** (`AgentReply`)

```json
{
  "answer": "You can ask for a refund within 14 days of delivery.",
  "thread_id": "3f6c1e2a-8b0d-4a51-9a63-0c2e7f1d5b94",
  "tool_calls": [
    { "name": "search_handbook", "args": { "query": "refund window" } }
  ],
  "sources": [
    { "source": "data/shop_handbook.pdf", "page": 0, "score": 0.82 }
  ],
  "searches": ["refund window"]
}
```

| Field        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `answer`     | The agent's final message                                                   |
| `thread_id`  | ID of the conversation this turn was saved under                           |
| `tool_calls` | Every tool call the model made, with its arguments                         |
| `sources`    | Every chunk the tools returned: file, page (0-based) and hybrid score (4 dp) |
| `searches`   | Computed shortcut: the `query` of each `search_handbook` call              |

For a greeting or small talk, the agent answers without searching, so `tool_calls`, `sources` and `searches` are empty. When the handbook doesn't cover a question, the answer is: *"I don't know. Our support team can help on live chat or at help@shop.com."*

> **Note:** `/ask` currently passes only `question` to the agent, so any `thread_id` you send is ignored. Every request starts a new conversation, and the response returns the new `thread_id`. Multi-turn memory works inside `ClauseAgent.ask(question, thread_id)`, but the endpoint and the CLI don't forward the ID yet.

## Chat from the terminal

```bash
uv run python -m app.cli
```

This starts an interactive loop. Each reply shows the answer, the searches the agent ran and the pages it used (example output):

```
You: Can I pay cash on delivery?
AI: Yes, cash on delivery is available for orders up to …
--- searched: ['cash on delivery']
--- pages:    [0, 1]
```

Press `Ctrl+C` or `Ctrl+D` to quit.

> **Note:** embedded Qdrant (`RAG_QDRANT_PATH`) can be opened by only one process at a time. Stop the API before you run the CLI, ingestion or evals, or point `RAG_QDRANT_URL` at a Qdrant server.

## Agent internals

### Flow of a question

1. `ClauseAgent.ask()` uses the given `thread_id`, or generates a new UUID, and sends the question to the agent.
2. The model reads the system prompt and the conversation so far, then decides whether to call `search_handbook`.
3. The tool runs a hybrid search for the top `RAG_TOP_K` chunks. It returns the chunk text to the model and the `Source` list as a separate artifact (`response_format="content_and_artifact"`).
4. The model writes the final answer. `ask()` walks through the messages to collect every `ToolCall` and `Source`, then returns an `AgentReply`.

### Middleware stack

Defined in [`app/middleware/stacks.py`](app/middleware/stacks.py):

| Middleware                 | Configuration                                         | Purpose                                         |
|----------------------------|-------------------------------------------------------|-------------------------------------------------|
| `ModelCallLimitMiddleware` | `run_limit=RAG_MAX_MODEL_CALLS`, `exit_behavior="end"` | Ends the run instead of looping forever          |
| `ToolCallLimitMiddleware`  | `run_limit=RAG_MAX_TOOL_CALLS`                        | Caps the searches per question                  |
| `ContextEditingMiddleware` | `ClearToolUsesEdit(trigger=2000, keep=2)`             | Clears old tool output once the context grows   |
| `SummarizationMiddleware`  | trigger `RAG_SUMMARY_TRIGGER_TOKENS`, keep `RAG_SUMMARY_KEEP_MESSAGES` | Compresses long conversations |

### Memory

A LangGraph `SqliteSaver` stores conversation state in `memory.db`, in the folder you run the app from (normally `api/`). Each conversation is keyed by its `thread_id`. Delete the file to clear all conversations.

## Prompts

Prompts are versioned Markdown files, loaded with `get_prompt(name, version)`:

```
app/prompts/support/system/v3.md   →  get_prompt("support/system", "v3")
```

You choose the system prompt with `RAG_SYSTEM_PROMPT_VERSION`. **`v3` is the only version with content.** `v1` and `v2` are empty files, as are the `refusal` and `rewrite_query` prompts, which are reserved for future features. The code default is `v1`, so set `v3` in `.env`. `.env.example` already does this.

To try a new prompt, add `v4.md` next to `v3.md`, set `RAG_SYSTEM_PROMPT_VERSION=v4` and run the evals.

## Evals

The harness runs every case in `evals/golden.jsonl` through `ClauseAgent` and grades the answers:

```bash
uv run python -m evals.run
```

There are three kinds of test case:

| Type           | Passes when                                                              |
|----------------|--------------------------------------------------------------------------|
| `answerable`   | The answer contains every `must_include` term (`a\|b` means either one)  |
| `out_of_scope` | The answer is a refusal, for example "I don't know"                      |
| `restricted`   | The answer contains none of the `must_not_include` staff-only terms      |

Each case prints a pass/fail line. A failing answerable case also gets a blame note, which says whether the **retriever** missed the fact or the **generator** got it wrong even though the fact was in the retrieved chunks.

Summary metrics:

- **Retrieval hit rate**: a source came from one of the `expected_pages`
- **Fact in chunks**: the required terms appear in the retrieved text
- **Answer correctness**: the answer contains the required terms
- **Refusal rate**: out-of-scope questions were refused
- **Leak rate**: restricted questions leaked staff-only information (lower is better)

Each run is saved to `evals/results/<timestamp>.json` together with its config (models, chunking, `top_k`, threshold and prompt version), so you can compare runs. To run the larger set, change `GOLDEN_FILE` in `evals/run.py` to `golden_v2.jsonl`.

Run the evals after any change to retrieval, the prompt, the model or the middleware.

## Configuration

All settings are read from `api/.env`, and each one needs the `RAG_` prefix. The defaults below come from [`app/core/config.py`](app/core/config.py). [`.env.example`](.env.example) overrides some of them.

| Variable                     | Default                   | Purpose                                            |
|------------------------------|---------------------------|----------------------------------------------------|
| `RAG_APP_NAME`               | `Clause`                  | App name                                           |
| `RAG_DATA_PATH`              | `data/shop_handbook.pdf`  | Document to ingest                                 |
| `RAG_CHAT_PROVIDER`          | `ollama`                  | Chat model provider (`ollama`, `google_genai`, …)  |
| `RAG_CHAT_MODEL`             | `gemma3:1b`               | Chat model name                                    |
| `RAG_CHAT_TEMPERATURE`       | `0.1`                     | Sampling temperature (not yet passed to the model) |
| `RAG_CHAT_API_KEY`           | *(empty)*                 | API key for hosted providers                       |
| `RAG_EMBED_PROVIDER`         | `ollama`                  | Dense embedding provider                           |
| `RAG_EMBED_MODEL`            | `nomic-embed-text:latest` | Dense embedding model                              |
| `RAG_EMBED_DIM`              | `768`                     | Must match the embedding model's output size       |
| `RAG_CHUNK_SIZE`             | `600`                     | Characters per chunk                               |
| `RAG_CHUNK_OVERLAP`          | `100`                     | Overlap between chunks                             |
| `RAG_TOP_K`                  | `3`                       | Chunks returned by each `search_handbook` call     |
| `RAG_SCORE_THRESHOLD`        | `0.5`                     | Recorded in eval reports. The agent's tool doesn't apply it yet. |
| `RAG_MAX_MODEL_CALLS`        | `5`                       | Model calls allowed per question                   |
| `RAG_MAX_TOOL_CALLS`         | `4`                       | Tool calls allowed per question                    |
| `RAG_SUMMARY_TRIGGER_TOKENS` | `3000`                    | History size that triggers summarization           |
| `RAG_SUMMARY_KEEP_MESSAGES`  | `10`                      | Recent messages kept unsummarized                  |
| `RAG_COLLECTION_NAME`        | `shop_handbook`           | Qdrant collection                                  |
| `RAG_QDRANT_PATH`            | `./vector_stores`         | Embedded Qdrant storage                            |
| `RAG_QDRANT_URL`             | *(unset)*                 | Qdrant server URL (overrides the path)             |
| `RAG_SYSTEM_PROMPT_VERSION`  | `v1`                      | System prompt version. Use `v3`.                   |

**Switching providers.** Install the provider's LangChain package, then set `RAG_CHAT_PROVIDER`, `RAG_CHAT_MODEL` and `RAG_CHAT_API_KEY`. `langchain-google-genai` (`google_genai`) and `langchain-ollama` (`ollama`) are already installed. The model must support tool calling.

**Changing embeddings.** If you change the embedding model, update `RAG_EMBED_DIM` and run ingestion again.

## Troubleshooting

| Symptom                                                    | Fix                                                                 |
|------------------------------------------------------------|---------------------------------------------------------------------|
| `RuntimeError: Collection shop_handbook not found!`        | Run `uv run python -m app.services.ingestion`                       |
| Qdrant error saying the storage folder is already in use   | Stop the other process (API, CLI or evals), or use `RAG_QDRANT_URL` |
| `RAG_EMBED_DIM is 768, but … returns N dimensions`         | Set `RAG_EMBED_DIM` to N, then re-ingest                            |
| The agent ignores the handbook or answers off-script       | Check that `RAG_SYSTEM_PROMPT_VERSION=v3`, because `v1` is empty    |
| Connection refused on `localhost:11434`                    | Start Ollama (`ollama serve`) and pull the models                   |
