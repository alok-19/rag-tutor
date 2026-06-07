# Study RAG Assistant 🎓

An interactive, AI-powered Retrieval-Augmented Generation (RAG) study assistant. The application indexes PDF textbooks, generates high-quality embeddings using the Gemini API, stores them locally in a Chroma DB vector database, and lets you ask questions via a web chat interface.

This application is modularized, fully tested, and supports multiple subjects.

---

## Project Structure

- **`app.py`**: Streamlit application UI entrypoint launcher.
- **`ingest.py`**: CLI document ingestion launcher.
- **`study_rag/`**: Core application package:
  - **`config.py`**: Configuration constants and environment variable loaders.
  - **`documents/`**: PDF loaders, document models, and file-hashing registry trackers.
  - **`ingestion/`**: Ingestion pipeline orchestration.
  - **`llm/`**: Gemini client instantiation, exponential backoff batch embeddings, and generation stream management with backup fallback models.
  - **`retrieval/`**: Vector store interface, prompt constructors, query acronym expansion, and retrieval service logic.
  - **`ui/`**: Streamlit pages, styling sheets, sidebars, and message view scripts.
- **`tests/`**: Unit and integration test suite.
- **`requirements.txt`**: Package dependencies (Streamlit, Google GenAI, ChromaDB, PyPDF, etc.).
- **`.env.template`**: Configuration template for API keys and database folders.

---

## Setup Instructions

### 1. Configure the Environment

1. Copy `.env.template` to `.env`:
   ```bash
   cp .env.template .env
   ```
2. Open `.env` and fill in your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
3. *(Optional)* Customize the database and material directories:
   ```env
   CHROMA_DB_PATH=chroma_db
   STUDY_MATERIALS_DIR=study_materials
   ```

### 2. Install Dependencies

Activate the virtual environment and install the required packages:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Running Ingestion

You can ingest textbook PDFs in one of two ways:

- **Via Command Line**:
  ```bash
  python ingest.py
  ```
- **Via the Streamlit UI**:
  Start the app, select or create your subject, upload documents in the sidebar, and click **"🚀 Ingest / Update Subject"**.

### 4. Run the Streamlit Application

Start the development server:
```bash
streamlit run app.py
```

---

## Running the Test Suite

The project includes unit and integration tests covering hashing, legacy registry migrations, acronym expansions, prompt construction, vector filtering, and insertion changes.

To run the test suite:
```bash
PYTHONPATH=. ./venv/bin/pytest
```

---

## Key Features

- **Multi-Subject Workspace**: Create individual subjects and manage materials separately. Search queries are isolated to the active subject.
- **Acronym Query Expansion**: Automatically expands common technical abbreviations (e.g. PCB to Process Control Block) dynamically before searching.
- **Transient Error Retry & Model Fallback**: Automatically retries on rate limits (429/503) and falls back to a secondary model (`gemini-1.5-flash`) if high demand persists.
- **Unified Citations**: Answers citation pages cleanly referencing PDFs, with detailed excerpts rendered in premium dark UI cards.
- **Incremental Ingestion**: Hashes files, skipping duplicates and cleaning up old chunks when a document changes.
