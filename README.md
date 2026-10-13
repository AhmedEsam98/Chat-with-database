# Database AI Agent — Multi-Database Architecture (Talk2Tables)

Text-to-SQL chatbot (Azure OpenAI) supporting **ANY SQL database** (not just PostgreSQL), including:
- **PostgreSQL** (`asyncpg`)
- **MySQL / MariaDB** (`aiomysql`)
- **SQLite** (`aiosqlite`) — Local or in-memory, includes a built-in demo database at `data/sample_ecommerce.db`.
- **Microsoft SQL Server** (`aioodbc`)
- **Custom / Generic Connection URIs** — Any SQLAlchemy-compatible async connection string.

Features async I/O, dynamic in-chat credential management, universal schema introspection, SQL guardrails, pure Semantic Vector Caching (Redis + Azure OpenAI embeddings), dynamic Streamlit UI controls, and cloud schema backups via Azure Blob Storage.

---

## 🗄️ Multi-Database Support & In-Chat Credentials

You can connect to and query any database directly from the chat interface:

1. **Credentials Form Mode**:
   - Select database engine (**PostgreSQL**, **MySQL**, **SQLite**, or **Microsoft SQL Server**).
   - Enter Host, Port, Database name, Username, Password, and SSL Mode.
   - For SQLite: enter file path or click **"📦 Use Built-in Demo E-Commerce Database"** for 1-click testing.
2. **Direct Connection URI Mode**:
   - Paste any async SQLAlchemy connection string (e.g. `postgresql+asyncpg://...`, `sqlite+aiosqlite:///...`, `mysql+aiomysql://...`).
3. **One-Click Actions**:
   - **🔌 Test Connection**: Verifies database reachability and authentication without applying.
   - **💾 Connect DB**: Switches the active database, re-introspects the schema, updates the system prompt, and resets chat history for the new target.
   - **🔄 Reset to Default (.env)**: Reverts back to the default database configured in `.env`.
   - **📋 Introspected Tables Expander**: Real-time view of discovered tables and columns.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```bash
cp .env.example .env   # then edit .env with your values
```

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | API version (default: `2024-12-01-preview`) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Primary chat model deployment (e.g. `gpt-4.1-mini`) |
| `AZURE_OPENAI_NANO_DEPLOYMENT_NAME` | Secondary chat model deployment (e.g. `gpt-4.1-nano`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment (e.g. `text-embedding-3-small`) |
| `DATABASE_URL` | Default database connection string (e.g. `postgresql+asyncpg://...` or `sqlite+aiosqlite:///...`) |
| `REDIS_HOST` | Redis hostname (`localhost` for host, `redis` in Docker) |
| `REDIS_PORT` | Redis port (default: `6379`) |
| `AZURE_STORAGE_CONNECTION_STRING` | (Optional) Azure Blob Storage connection string |
| `AZURE_STORAGE_CONTAINER_NAME` | Azure Blob container name (default: `chatbot-data`) |

### 3. Run Applications

#### Streamlit Web UI (Recommended)
```bash
streamlit run app.py
```
Opens the web interface at http://localhost:8501.

#### FastAPI REST API Server
```bash
uvicorn main:app --reload --port 8000
```
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Endpoints for querying (`/query`), connection testing (`/connection/test`), and database switching (`/connection/switch`).

#### Interactive CLI Chat
```bash
python cli.py
```
Commands:
- `/sample` — Switch to built-in demo SQLite database
- `/db` — Show active database and discovered tables
- `/connect <url>` — Connect to any database URL
- `/model <name>` — Switch model
- `/quit` — Exit

---

## ⚡ Pure Semantic Vector Cache

The application uses a **pure Semantic Vector Cache** powered by Redis and Azure OpenAI embeddings:
- Every user query is vectorized via Azure OpenAI embeddings (`text-embedding-3-small`).
- Compares question vectors stored in Redis hashes using **cosine similarity**.
- Matches paraphrased questions with identical semantic intent.
- Configurable similarity threshold in Streamlit UI (from `0.70` to `0.99`).

---

## 🐳 Docker Deployment

Run the full stack (FastAPI backend + Streamlit web UI + Redis):
```bash
docker compose up -d --build
```
