# RAG Tutor 🎓

A personal RAG tutor for your study materials. The application indexes PDF textbooks using **semantic chunking**, generates high-quality embeddings via the Gemini API, stores them locally in a Chroma DB vector database, and lets you ask questions via a web chat interface with **conversational memory** and **feedback tracking**.

This application is modularized, fully tested, and supports multiple subjects.

---

## Project Structure

- **`rag_tutor/`**: Core application package:
  - **`__main__.py`**: Unified entrypoint CLI to run commands (`run`, `ingest`).
  - **`config.py`**: Configuration constants and environment variable loaders.
  - **`documents/`**: PDF loaders (PyMuPDF), **semantic text chunking**, document models, and file-hashing registry trackers.
  - **`ingestion/`**: Ingestion pipeline orchestration with incremental updates.
  - **`llm/`**: Multi-provider LLM abstraction (Gemini, OpenAI, DeepSeek), exponential backoff batch embeddings, and generation stream management with backup fallback models.
  - **`retrieval/`**: Vector store interface, prompt constructors, **conversational memory**, query acronym expansion, and retrieval service logic.
  - **`ui/`**: Streamlit pages, styling sheets, sidebars, message views, and **feedback widgets**.
  - **`feedback.py`**: Local JSONL persistence for thumbs up/down ratings.
- **`tests/`**: Unit and integration test suite.
- **`pyproject.toml`**: Modern Python packaging with pinned dependencies.
- **`requirements.txt`**: Legacy pinned requirements file.
- **`.env.template`**: Configuration template for API keys and database folders.

---

## Supported LLM Providers

RAG Tutor supports multiple LLM backends via a unified provider abstraction:

| Provider | Chat / Generation | Embeddings | Notes |
|----------|-------------------|------------|-------|
| **Gemini** (Google) | ✅ | ✅ | Default. Native support. |
| **OpenAI** | ✅ | ✅ | GPT-4o-mini, text-embedding-3-small |
| **DeepSeek** | ✅ | ❌ | OpenAI-compatible API. No embedding API yet. |

> **Hybrid mode**: Use `LLM_PROVIDER=deepseek` + `EMBEDDING_PROVIDER=openai` to get DeepSeek's cheap reasoning with OpenAI embeddings.

---

## Setup Instructions

### 1. Configure the Environment

1. Copy `.env.template` to `.env`:
   ```bash
   cp .env.template .env
   ```
2. Open `.env` and configure your provider:

   **Gemini (default):**
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   **OpenAI:**
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   **DeepSeek (hybrid with OpenAI embeddings):**
   ```env
   LLM_PROVIDER=deepseek
   EMBEDDING_PROVIDER=openai
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. *(Optional)* Customize the database and material directories:
   ```env
   CHROMA_DB_PATH=chroma_db
   STUDY_MATERIALS_DIR=study_materials
   ```

### 2. Install Dependencies

**Recommended (modern tooling with `pyproject.toml`):**
```bash
source venv/bin/activate
pip install -e .
```

**Alternative (legacy `requirements.txt`):**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Running Ingestion

You can ingest textbook PDFs in one of two ways:

- **Via Command Line**:
  ```bash
  python -m rag_tutor ingest --subject "Operating System"
  ```
- **Via the Streamlit UI**:
  Start the app, select or create your subject, upload documents in the sidebar, and click **"🚀 Ingest / Update Subject"**.

> **Note**: Documents are now split into **semantic chunks** (default ~1000 chars with 200 char overlap) using `RecursiveCharacterTextSplitter` before embedding. This improves retrieval accuracy by keeping related concepts together while avoiding oversized context windows.

### 4. Run the Streamlit Application

Start the web application:
```bash
python -m rag_tutor run
```

---

## Running the Test Suite

The project includes unit and integration tests covering hashing, legacy registry migrations, acronym expansions, prompt construction, vector filtering, **semantic chunking**, **chat history formatting**, **query disambiguation**, **feedback persistence**, and **multi-provider LLM abstraction**.

To run the test suite:
```bash
PYTHONPATH=. ./venv/bin/pytest
```

---

## Key Features

- **Multi-Provider LLM Support**: Switch between Gemini, OpenAI, and DeepSeek (OpenAI-compatible). Supports hybrid setups (e.g., DeepSeek chat + OpenAI embeddings) via `LLM_PROVIDER` and `EMBEDDING_PROVIDER` env vars.
- **Multi-Subject Workspace**: Create individual subjects and manage materials separately. Search queries are isolated to the active subject.
- **Semantic Chunking**: PDF pages are split into overlapping semantic chunks using `RecursiveCharacterTextSplitter` for better retrieval accuracy.
- **Acronym Query Expansion**: Automatically expands common technical abbreviations (e.g. PCB to Process Control Block) dynamically before searching.
- **Conversational Memory**: The LLM prompt includes the last 3 conversation turns, and follow-up queries like *"Explain more"* are automatically disambiguated using prior assistant context.
- **Transient Error Retry & Model Fallback**: Automatically retries on rate limits (429/503) and falls back to a secondary model (`gemini-1.5-flash`) if high demand persists.
- **Unified Citations**: Answers citation pages cleanly referencing PDFs, with detailed excerpts rendered in premium dark UI cards.
- **Incremental Ingestion**: Hashes files, skipping duplicates and cleaning up old chunks when a document changes.
- **Feedback Tracking**: Rate each assistant response with 👍 / 👎. Feedback is persisted locally to `chroma_db/feedback.jsonl` for future evaluation and iteration.

---

## License

This project is licensed under the [MIT License](LICENSE).
