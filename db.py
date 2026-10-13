"""
Database connection pool, query execution, and schema introspection module.

Supports ANY SQL database (PostgreSQL, MySQL, SQLite, MSSQL, custom URIs) with:
- Dynamic in-chat credential configuration and connection switching.
- Universal dialect-agnostic introspection using SQLAlchemy inspect().
- Strict read-only query guardrails and safe value sampling.

Usage:
    from db import (
        execute_sql,
        is_safe_question,
        is_safe_query,
        fetch_schema_info,
        fetch_foreign_keys,
        fetch_distinct_values,
        test_db_connection,
        set_database_connection,
        get_active_dialect,
        get_active_connection_info,
    )
"""

import asyncio
import os
import re
import config
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()

# Active connection state
_current_url = config.DATABASE_URL or os.getenv("DATABASE_URL") or "sqlite+aiosqlite:///:memory:"
_current_url, _current_dialect = config.normalize_db_url(_current_url)
_current_label = f"{_current_dialect.upper()} Database"

_engines: dict[asyncio.AbstractEventLoop | None, AsyncEngine] = {}


def get_active_dialect() -> str:
    """Return the active database dialect name (postgresql, mysql, sqlite, mssql)."""
    return _current_dialect


def get_active_connection_info() -> dict:
    """Return sanitized connection info for UI display."""
    url = _current_url
    masked = url
    if "@" in url and "://" in url:
        prefix, rest = url.split("://", 1)
        user_pass, host_db = rest.split("@", 1)
        if ":" in user_pass:
            user = user_pass.split(":", 1)[0]
            masked = f"{prefix}://{user}:****@{host_db}"
        else:
            masked = f"{prefix}://****@{host_db}"

    return {
        "url": masked,
        "dialect": _current_dialect,
        "label": _current_label,
    }


def set_database_connection(url: str, label: str | None = None) -> None:
    """Switch active database connection and dispose previous engine instances."""
    global _current_url, _current_dialect, _current_label
    normalized_url, dialect = config.normalize_db_url(url)
    _current_url = normalized_url
    _current_dialect = dialect
    _current_label = label or f"{dialect.upper()} Connection"

    # Dispose old engines
    for eng in _engines.values():
        try:
            eng.sync_engine.dispose()
        except Exception:
            pass
    _engines.clear()

    # Invalidate schema prompt cache if schema module is loaded
    try:
        import schema
        schema.invalidate_cache()
    except Exception:
        pass


def get_engine() -> AsyncEngine:
    """Return an async engine bound to the current running event loop using NullPool."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    eng = _engines.get(loop)
    if eng is None:
        eng = create_async_engine(
            _current_url,
            poolclass=NullPool,
            pool_pre_ping=True,
        )
        _engines[loop] = eng
    return eng


class _EngineProxy:
    """Proxy object ensuring operations always use the active engine."""

    def connect(self):
        return get_engine().connect()

    def begin(self):
        return get_engine().begin()

    def __getattr__(self, name):
        return getattr(get_engine(), name)


engine = _EngineProxy()


async def test_db_connection(target_url: str | None = None) -> tuple[bool, str]:
    """Test connectivity for either the active database or a specified target URL."""
    url_to_test = target_url or _current_url
    if not url_to_test:
        return False, "Database URL not configured."

    norm_url, _ = config.normalize_db_url(url_to_test)
    temp_eng = create_async_engine(norm_url, poolclass=NullPool, pool_pre_ping=True)
    try:
        async with temp_eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, "Connected"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            temp_eng.sync_engine.dispose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Safety Guardrails
# ---------------------------------------------------------------------------

FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|CREATE|ATTACH|DETACH|PRAGMA|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


def is_safe_question(question: str) -> tuple[bool, str | None]:
    """Check if the user question is attempting to perform a destructive or mutating operation."""
    q = question.strip().lower()
    destructive_keywords = ("delete", "drop", "truncate", "wipe", "destroy")
    if any(re.search(rf"\b{kw}\b", q) for kw in destructive_keywords):
        return False, "Query rejected: only single SELECT statements are allowed."

    mutating_actions = ("alter", "insert", "update", "create", "remove")
    if (
        re.search(
            r"\b(alter|insert|update|create|remove)\b\s+.*?\b(table|database|record|row|into|from|schema|column)\b",
            q,
        )
        or q.startswith(destructive_keywords + mutating_actions)
    ):
        return False, "Query rejected: only single SELECT statements are allowed."

    return True, None


def is_safe_query(sql: str) -> bool:
    """Only allow single, read-only SELECT statements."""
    clean_sql = sql.strip().rstrip(";")
    if ";" in clean_sql:
        return False  # blocks stacked/multiple statements

    upper_sql = clean_sql.upper()
    is_select = upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")
    return is_select and not FORBIDDEN.search(clean_sql)


def quote_identifier(name: str) -> str:
    """Quote table or column name according to active database dialect conventions."""
    clean = name.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
    if _current_dialect == "mysql":
        return f"`{clean}`"
    elif _current_dialect == "mssql":
        return f"[{clean}]"
    else:
        # PostgreSQL, SQLite, and ANSI default
        return f'"{clean}"'


def is_system_table(table_name: str) -> bool:
    """Exclude internal database system tables from prompts."""
    t = table_name.lower()
    if _current_dialect == "sqlite":
        return t.startswith("sqlite_") or t.startswith("_litestream_")
    if _current_dialect == "postgresql":
        return t.startswith("pg_") or t.startswith("sql_")
    if _current_dialect == "mysql":
        return t in ("sys_config", "general_log", "slow_log")
    if _current_dialect == "mssql":
        return t.startswith("sys") or t.startswith("spt_") or t == "msreplication_options"
    return False


async def execute_sql(query: str, row_limit: int | None = None) -> tuple[list | None, str | None]:
    """Execute a validated SQL query and return (rows, error)."""
    clean_query = query.strip().rstrip(";")
    if not is_safe_query(clean_query):
        return None, "Query rejected: only single SELECT statements are allowed."

    if row_limit is not None and _current_dialect != "mssql":
        sql_to_run = f"SELECT * FROM ({clean_query}) AS sub LIMIT {row_limit}"
    else:
        sql_to_run = clean_query

    try:
        eng = get_engine()
        async with eng.connect() as conn:
            result = await conn.execute(text(sql_to_run))
            rows = [dict(row._mapping) for row in result.fetchall()]
            return rows, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Universal Dialect-Agnostic Schema Introspection
# ---------------------------------------------------------------------------

async def fetch_schema_info() -> list[dict]:
    """Fetch all tables and columns from the active database using SQLAlchemy inspect()."""
    def _inspect_sync(sync_conn):
        insp = inspect(sync_conn)
        tables = insp.get_table_names()
        cols_info = []
        for tbl in tables:
            if is_system_table(tbl):
                continue
            for col in insp.get_columns(tbl):
                cols_info.append({
                    "table_name": tbl,
                    "column_name": col["name"],
                    "data_type": str(col["type"]),
                })
        return cols_info

    try:
        eng = get_engine()
        async with eng.connect() as conn:
            return await conn.run_sync(_inspect_sync)
    except Exception:
        return []


async def fetch_foreign_keys() -> list[dict]:
    """Fetch foreign key relationships from the active database using SQLAlchemy inspect()."""
    def _inspect_sync(sync_conn):
        insp = inspect(sync_conn)
        tables = insp.get_table_names()
        fks_info = []
        for tbl in tables:
            if is_system_table(tbl):
                continue
            for fk in insp.get_foreign_keys(tbl):
                ref_tbl = fk.get("referred_table")
                c_cols = fk.get("constrained_columns", [])
                r_cols = fk.get("referred_columns", [])
                for c_col, r_col in zip(c_cols, r_cols):
                    fks_info.append({
                        "table_name": tbl,
                        "column_name": c_col,
                        "foreign_table_name": ref_tbl,
                        "foreign_column_name": r_col,
                    })
        return fks_info

    try:
        eng = get_engine()
        async with eng.connect() as conn:
            return await conn.run_sync(_inspect_sync)
    except Exception:
        return []


async def fetch_distinct_values(table: str, column: str, limit: int = 25) -> list[str]:
    """Fetch distinct non-null values for a column using dialect-safe quoting."""
    quoted_table = quote_identifier(table)
    quoted_column = quote_identifier(column)

    q = text(
        f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
        f"WHERE {quoted_column} IS NOT NULL "
        f"ORDER BY {quoted_column} LIMIT :lim"
    )
    try:
        eng = get_engine()
        async with eng.connect() as conn:
            result = await conn.execute(q, {"lim": limit})
            return [str(row[0]) for row in result.fetchall()]
    except Exception:
        return []
