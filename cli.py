"""
Interactive CLI for chatting with ANY database using the Talk2Tables AI agent.

Supports PostgreSQL, MySQL, SQLite, MSSQL, and custom database URIs.

Run with:
    python cli.py

Interactive commands:
    /db                   View active database connection & tables
    /sample               Connect to built-in sample SQLite database
    /connect <url>        Connect to any database URL
    /model <model_name>   Switch between available models
    /quit or /exit        Exit the CLI session
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

import config
from db import (
    execute_sql,
    is_safe_question,
    test_db_connection,
    set_database_connection,
    get_active_dialect,
    get_active_connection_info,
    fetch_schema_info,
)
from llm import generate_sql, generate_answer, rewrite_question, explain_sql
from cache import get_semantic_cached_answer, set_semantic_cached_answer

MODELS = config.MODELS
FAILURE_MESSAGE = config.FAILURE_MESSAGE
MAX_SQL_ATTEMPTS = config.MAX_SQL_ATTEMPTS


async def chat(question: str, model: str) -> tuple[str, str | None, str | None, bool, list | None]:
    """Run the full text-to-SQL pipeline and return (answer, sql, sql_explanation, cached, rows)."""
    # 0. Safety guardrail
    is_safe, error_msg = is_safe_question(question)
    if not is_safe:
        return error_msg, None, None, False, None

    # 1. Semantic cache
    if config.ENABLE_SEMANTIC_CACHE:
        cached = await get_semantic_cached_answer(question, model=model)
        if cached:
            return cached, None, None, True, None

    # 2. Rewrite question
    if config.ENABLE_QUERY_REWRITE:
        rewritten = await rewrite_question(question, model=model)
    else:
        rewritten = question

    # 3. Text-to-SQL with self-correction
    last_sql = None
    history = []
    for _ in range(MAX_SQL_ATTEMPTS):
        sql = await generate_sql(rewritten, history=history, model=model)
        last_sql = sql
        if sql.strip().upper() == "REJECTED":
            return "Query rejected: only single SELECT statements are allowed.", None, None, False, None

        rows, error = await execute_sql(sql)
        if error and "Query rejected" in error:
            return error, sql, None, False, None
        if not error:
            answer, sql_explanation = await asyncio.gather(
                generate_answer(question, rows, model=model),
                explain_sql(sql, question=question, model=model),
            )
            if config.ENABLE_SEMANTIC_CACHE:
                await set_semantic_cached_answer(question, answer, model=model)
            return answer, sql, sql_explanation, False, rows
        history.append((sql, error))

    return FAILURE_MESSAGE, last_sql, None, False, None


async def run_cli():
    model = "model_1"
    print("")
    print("=" * 60)
    print("       Talk2Tables — Multi-Database AI Agent CLI")
    print("=" * 60)
    info = get_active_connection_info()
    print(f"  Active DB: {info['label']} ({info['dialect']})")
    print(f"  Model:     {model}")
    print("  Commands:")
    print("    /db              View active database connection & tables")
    print("    /sample          Connect to built-in sample SQLite database")
    print("    /connect <url>   Connect to any database URL")
    print("    /model <name>    Switch active model (model_1 / model_2)")
    print("    /quit            Exit CLI session")
    print("")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        if line.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if line.lower() == "/db":
            ok, msg = await test_db_connection()
            info = get_active_connection_info()
            cols = await fetch_schema_info()
            from collections import defaultdict
            tables_map = defaultdict(list)
            for c in cols:
                tables_map[c["table_name"]].append(c["column_name"])

            print(f"\n[Database Status]")
            print(f"  Connected: {ok} ({msg})")
            print(f"  Dialect:   {info['dialect']}")
            print(f"  Label:     {info['label']}")
            print(f"  Tables:    {len(tables_map)}")
            if tables_map:
                print(f"  Table list: {', '.join(sorted(tables_map.keys()))}")
            print("")
            continue

        if line.lower() == "/sample":
            sample_path = os.path.join("data", "sample_ecommerce.db")
            if os.path.exists(sample_path):
                sample_url, sample_label = config.build_db_url("sqlite", sqlite_path=sample_path)
                set_database_connection(sample_url, sample_label)
                print(f"\nSwitched to sample database: {sample_label}\n")
            else:
                print(f"\nSample database file not found at {sample_path}\n")
            continue

        if line.lower().startswith("/connect"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /connect <database_url>")
                continue
            raw_url = parts[1]
            ok, msg = await test_db_connection(raw_url)
            if ok:
                norm_url, dialect = config.normalize_db_url(raw_url)
                set_database_connection(norm_url, f"{dialect.upper()} Database")
                print(f"\nConnected successfully to: {norm_url}\n")
            else:
                print(f"\nConnection failed: {msg}\n")
            continue

        if line.lower().startswith("/model"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Available models: {', '.join(MODELS.keys())}")
            elif parts[1] in MODELS:
                model = parts[1]
                print(f"Switched to {MODELS[model]}")
            else:
                print(f"Unknown model. Available: {', '.join(MODELS.keys())}")
            continue

        # Run pipeline
        answer, sql, sql_explanation, cached, rows = await chat(line, model)
        cached_tag = " (served from semantic cache)" if cached else ""
        count_tag = f" [{len(rows)} records retrieved]" if rows is not None else ""
        print(f"\nBot: {answer}{cached_tag}{count_tag}")
        if sql:
            print(f"\n[Generated SQL]\n{sql}")
        if sql_explanation:
            print(f"\n[Query Explanation]\n{sql_explanation}")
        print("")


def main():
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
