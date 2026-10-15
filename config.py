"""
Configuration module for the Multi-Database AI Agent (Talk2Tables).

Centralizes environment variables, model choices, Redis settings,
caching thresholds, multi-database drivers, and generation parameters with safe defaults.
"""

import os
import urllib.parse
from dotenv import load_dotenv

# Load environment variables (.env in project root, with container env taking precedence)
load_dotenv(override=False)


def reload_env() -> str | None:
    """Reload the .env file with override=True to capture any runtime changes."""
    global DATABASE_URL, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
    load_dotenv(override=True)
    DATABASE_URL = os.getenv("DATABASE_URL")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://gbg-ai-2.openai.azure.com/")
    return DATABASE_URL


def get_default_db_url() -> str:
    """Return the default database URL from .env, refreshing if needed."""
    url = os.getenv("DATABASE_URL")
    if not url:
        reload_env()
        url = os.getenv("DATABASE_URL")
    return url or "sqlite+aiosqlite:///:memory:"


# ---------------------------------------------------------------------------
# Azure OpenAI Settings
# ---------------------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://gbg-ai-2.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# Model 2 (Nano) Optional Overrides
AZURE_OPENAI_NANO_API_KEY = os.getenv("AZURE_OPENAI_NANO_API_KEY")
AZURE_OPENAI_NANO_ENDPOINT = os.getenv("AZURE_OPENAI_NANO_ENDPOINT")
AZURE_OPENAI_NANO_API_VERSION = os.getenv("AZURE_OPENAI_NANO_API_VERSION")
AZURE_OPENAI_NANO_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_NANO_DEPLOYMENT_NAME", "gpt-4.1-nano")

MODELS = {
    "model_1": f"Model 1 ({AZURE_OPENAI_DEPLOYMENT_NAME})",
    "model_2": f"Model 2 ({AZURE_OPENAI_NANO_DEPLOYMENT_NAME})",
}


# ---------------------------------------------------------------------------
# Database Settings & Multi-Engine Helpers
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))

SUPPORTED_ENGINES = {
    "postgresql": "PostgreSQL",
    "mysql": "MySQL / MariaDB",
    "sqlite": "SQLite",
    "mssql": "Microsoft SQL Server",
    "custom": "Custom Connection URI",
}

DEFAULT_PORTS = {
    "postgresql": 5432,
    "mysql": 3306,
    "mssql": 1433,
}


def normalize_db_url(raw_url: str) -> tuple[str, str]:
    """Normalize any database connection string to its async SQLAlchemy equivalent.

    Returns (normalized_async_url, dialect_name).
    """
    url = raw_url.strip()
    u_lower = url.lower()

    if u_lower.startswith("sqlite"):
        dialect = "sqlite"
        if not u_lower.startswith("sqlite+aiosqlite"):
            url = "sqlite+aiosqlite" + url[len("sqlite"):]
    elif u_lower.startswith("postgresql") or u_lower.startswith("postgres"):
        dialect = "postgresql"
        if u_lower.startswith("postgres://"):
            url = "postgresql+asyncpg://" + url[len("postgres://"):]
        elif u_lower.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    elif u_lower.startswith("mysql") or u_lower.startswith("mariadb"):
        dialect = "mysql"
        if u_lower.startswith("mysql://"):
            url = "mysql+aiomysql://" + url[len("mysql://"):]
        elif u_lower.startswith("mariadb://"):
            url = "mysql+aiomysql://" + url[len("mariadb://"):]
    elif u_lower.startswith("mssql"):
        dialect = "mssql"
        if u_lower.startswith("mssql://"):
            url = "mssql+aioodbc://" + url[len("mssql://"):]
    else:
        dialect = "custom"

    return url, dialect


def build_db_url(
    engine: str,
    host: str = "localhost",
    port: int | None = None,
    database: str = "",
    username: str = "",
    password: str = "",
    ssl_mode: str | None = None,
    sqlite_path: str = "",
) -> tuple[str, str]:
    """Build an async SQLAlchemy database URL from explicit credentials.

    Returns (connection_url, display_label).
    """
    engine = engine.lower()

    if engine == "sqlite":
        path = sqlite_path.strip()
        if not path or path == ":memory:":
            return "sqlite+aiosqlite:///:memory:", "SQLite (in-memory)"
        clean_path = path.replace("\\", "/")
        return f"sqlite+aiosqlite:///{clean_path}", f"SQLite ({path})"

    user_enc = urllib.parse.quote_plus(username) if username else ""
    pass_enc = urllib.parse.quote_plus(password) if password else ""
    auth = ""
    if user_enc:
        auth = f"{user_enc}:{pass_enc}@" if pass_enc else f"{user_enc}@"

    port = port or DEFAULT_PORTS.get(engine, 5432)
    db_clean = database.lstrip("/")

    query_params = []
    if ssl_mode and ssl_mode not in ("disable", "none"):
        query_params.append(f"ssl={ssl_mode}")
    q_str = f"?{'&'.join(query_params)}" if query_params else ""

    if engine == "postgresql":
        driver = "postgresql+asyncpg"
        label = f"PostgreSQL ({host}:{port}/{db_clean})"
    elif engine == "mysql":
        driver = "mysql+aiomysql"
        label = f"MySQL ({host}:{port}/{db_clean})"
    elif engine == "mssql":
        driver = "mssql+aioodbc"
        label = f"MSSQL ({host}:{port}/{db_clean})"
    else:
        driver = "postgresql+asyncpg"
        label = f"{engine.capitalize()} ({host}:{port}/{db_clean})"

    return f"{driver}://{auth}{host}:{port}/{db_clean}{q_str}", label


# ---------------------------------------------------------------------------
# Redis & Caching Settings
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", os.getenv("REDIS_CACHE_TTL", "3600")))

# Semantic Cache
SEMANTIC_SIMILARITY_THRESHOLD = float(
    os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
)
SEMANTIC_CACHE_KEY_PREFIX = os.getenv("SEMANTIC_CACHE_KEY_PREFIX", "semantic_cache")
SEMANTIC_MAX_ENTRIES = int(os.getenv("SEMANTIC_MAX_ENTRIES", "500"))

# Feature toggles
ENABLE_SEMANTIC_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() in ("true", "1", "yes")
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Generation & Pipeline Parameters
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_SQL_ATTEMPTS = int(os.getenv("MAX_SQL_ATTEMPTS", "4"))
MAX_SAMPLE_ROWS = int(os.getenv("MAX_SAMPLE_ROWS", "100"))
FAILURE_MESSAGE = os.getenv("FAILURE_MESSAGE", "Sorry, I couldn't answer that question.")


# ---------------------------------------------------------------------------
# Azure Blob Storage (Cloud Backup)
# ---------------------------------------------------------------------------
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "chatbot-data")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _mask_secret(secret: str | None, visible_chars: int = 4) -> str:
    """Mask sensitive keys for safe display in UI."""
    if not secret:
        return "Not Set"
    if len(secret) <= visible_chars * 2:
        return "****"
    return f"{secret[:visible_chars]}...{secret[-visible_chars:]}"


def get_config_summary() -> dict:
    """Return a sanitized dictionary of current configuration parameters."""
    return {
        "Azure OpenAI Endpoint": AZURE_OPENAI_ENDPOINT or "Not Configured",
        "Azure OpenAI Model 1": AZURE_OPENAI_DEPLOYMENT_NAME,
        "Azure OpenAI Embedding Model": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "Azure OpenAI API Key": _mask_secret(AZURE_OPENAI_API_KEY),
        "Redis Host & Port": f"{REDIS_HOST}:{REDIS_PORT}",
        "Cache TTL": f"{CACHE_TTL_SECONDS} seconds",
        "Semantic Cache Threshold": f"{SEMANTIC_SIMILARITY_THRESHOLD:.2f}",
        "Default Temperature": f"{DEFAULT_TEMPERATURE:.2f}",
        "Max SQL Retries": MAX_SQL_ATTEMPTS,
        "Database Configured": "Yes" if DATABASE_URL else "No",
        "Azure Storage Container": AZURE_STORAGE_CONTAINER_NAME if AZURE_STORAGE_CONNECTION_STRING else "Not Configured",
    }
