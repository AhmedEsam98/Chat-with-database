"""
Dynamic schema builder — introspects table structures, relationships,
and column sample values for ANY connected database to construct a rich, value-aware system prompt for LLMs.

Supports PostgreSQL, MySQL, SQLite, MSSQL, and custom SQL dialects.

Usage:
    from schema import get_schema_prompt, invalidate_cache
    schema_prompt = await get_schema_prompt()
"""

import asyncio
import time
from collections import defaultdict

# In-memory cache so we don't hit the DB on every request
_cache = {"prompt": None, "ts": 0}
SCHEMA_CACHE_TTL = 3600  # 1 hour

_SKIP_SUFFIXES = ("Id", "id")
_SKIP_NAMES = {
    "Fax", "Phone", "PostalCode", "Email", "Address",
    "BillingAddress", "BillingPostalCode", "Password", "PasswordHash",
}


def _should_sample(column_name: str, data_type: str) -> bool:
    """Decide whether to show sample values for this column."""
    dt_lower = data_type.lower()
    text_indicators = ("char", "text", "varchar", "string", "nvarchar", "user-defined")
    if not any(ti in dt_lower for ti in text_indicators):
        return False
    if column_name.endswith(_SKIP_SUFFIXES):
        return False
    if column_name in _SKIP_NAMES:
        return False
    return True


async def get_schema_prompt() -> str:
    """Return the full system prompt with dialect-specific schema + sample values (cached)."""
    now = time.time()
    if _cache["prompt"] and (now - _cache["ts"]) < SCHEMA_CACHE_TTL:
        return _cache["prompt"]

    # Lazy import to avoid circular dependency
    from db import (
        fetch_schema_info,
        fetch_foreign_keys,
        fetch_distinct_values,
        get_active_dialect,
    )

    columns_info = await fetch_schema_info()
    fk_info = await fetch_foreign_keys()
    dialect = get_active_dialect()

    # Group columns by table
    tables = defaultdict(list)
    for col in columns_info:
        tables[col["table_name"]].append(col)

    # Fetch sample values for text columns concurrently
    sample_tasks = []
    task_keys = []
    for table_name, cols in tables.items():
        for col in cols:
            if _should_sample(col["column_name"], col["data_type"]):
                sample_tasks.append(fetch_distinct_values(table_name, col["column_name"], limit=20))
                task_keys.append((table_name, col["column_name"]))

    results = await asyncio.gather(*sample_tasks, return_exceptions=True)
    sample_values = {}
    for key, vals in zip(task_keys, results):
        if isinstance(vals, list) and vals:
            sample_values[key] = vals

    prompt = _build_prompt(tables, fk_info, sample_values, dialect)
    _cache["prompt"] = prompt
    _cache["ts"] = now
    return prompt


def _infer_relationships(tables, fk_info):
    """Combine database foreign keys with naming-convention inferences."""
    known_fks = [
        {
            "table_name": fk["table_name"],
            "column_name": fk["column_name"],
            "foreign_table_name": fk["foreign_table_name"],
            "foreign_column_name": fk["foreign_column_name"],
        }
        for fk in fk_info
    ]
    existing_pairs = {(f["table_name"], f["column_name"]) for f in known_fks}

    table_pk_map = {}
    for tbl, cols in tables.items():
        col_names = {c["column_name"] for c in cols}
        if f"{tbl}Id" in col_names:
            table_pk_map[tbl] = f"{tbl}Id"
        elif "Id" in col_names:
            table_pk_map[tbl] = "Id"

    for tbl, cols in tables.items():
        for col in cols:
            col_name = col["column_name"]
            pair = (tbl, col_name)
            if pair in existing_pairs:
                continue

            if col_name.endswith("Id") and col_name != table_pk_map.get(tbl):
                target_table = col_name[:-2]
                if target_table in table_pk_map:
                    known_fks.append({
                        "table_name": tbl,
                        "column_name": col_name,
                        "foreign_table_name": target_table,
                        "foreign_column_name": table_pk_map[target_table],
                    })
                    existing_pairs.add(pair)

    return sorted(known_fks, key=lambda x: (x["table_name"], x["column_name"]))


def _build_prompt(tables, fk_info, sample_values, dialect: str) -> str:
    """Assemble the system prompt customized for the active dialect."""
    dialect_upper = dialect.upper()
    if "POSTGRES" in dialect_upper:
        dialect_name = "PostgreSQL"
        quote_rule = 'Wrap table names and column names in double quotes if they contain uppercase or spaces (e.g. SELECT "Title" FROM "Album").'
    elif "MYSQL" in dialect_upper or "MARIADB" in dialect_upper:
        dialect_name = "MySQL"
        quote_rule = 'Wrap table names and column names in backticks if needed (e.g. SELECT `Title` FROM `Album`).'
    elif "SQLITE" in dialect_upper:
        dialect_name = "SQLite"
        quote_rule = 'Wrap table names and column names in double quotes if needed (e.g. SELECT "Title" FROM "Album").'
    elif "MSSQL" in dialect_upper:
        dialect_name = "Microsoft SQL Server (T-SQL)"
        quote_rule = 'Wrap table names and column names in square brackets if needed (e.g. SELECT [Title] FROM [Album]).'
    else:
        dialect_name = dialect.capitalize()
        quote_rule = 'Wrap table names and column names in double quotes if needed.'

    lines = [
        f"You are a {dialect_name} database expert. Your database contains the following tables:"
    ]

    if not tables:
        lines.append("- (No user tables discovered or database is empty)")

    for table_name in sorted(tables.keys()):
        cols = tables[table_name]
        col_names = ", ".join(c["column_name"] for c in cols)
        lines.append(f"- {table_name} ({col_names})")

        # Show sample values for relevant text columns
        for col in cols:
            key = (table_name, col["column_name"])
            if key in sample_values:
                vals = sample_values[key]
                shown = vals[:15]
                val_str = ", ".join(f'"{v}"' for v in shown)
                if len(vals) > 15:
                    val_str += ", ..."
                lines.append(f"    -> {col['column_name']} sample values include: {val_str}")

    fks = _infer_relationships(tables, fk_info)
    lines.append("")
    lines.append("RELATIONSHIPS (for JOINs):")
    for fk in fks:
        lines.append(
            f"- {fk['table_name']}.{fk['column_name']} -> "
            f"{fk['foreign_table_name']}.{fk['foreign_column_name']}"
        )

    lines.append("")
    lines.append("IMPORTANT RULES:")
    lines.append(f"1. Return ONLY a valid {dialect_name} SQL query.")
    lines.append("2. Do NOT add any formatting like ```sql or ```.")
    lines.append("3. Do NOT add any explanations. Just the raw SQL string.")
    lines.append(f"4. {quote_rule}")
    lines.append(
        "5. Only ever generate SELECT statements. Never generate "
        "INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or GRANT. "
        "If the user asks to delete, drop, truncate, or modify any table or data, "
        "return ONLY the single word: REJECTED"
    )
    lines.append(
        "6. If a question is ambiguous or unanswerable from this schema, "
        "still return your best-effort SELECT query."
    )
    lines.append(
        "7. When the user mentions a value (country, name, genre, etc.), "
        "match it against the sample values listed above."
    )

    return "\n".join(lines)


def invalidate_cache():
    """Force the next call to get_schema_prompt() to re-fetch from the DB."""
    _cache["prompt"] = None
    _cache["ts"] = 0
