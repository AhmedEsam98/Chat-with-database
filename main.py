"""
FastAPI REST API backend for the Database AI Agent (Talk2Tables).

Provides HTTP endpoints for text-to-SQL question answering, multi-database
credential management, Redis cache management, health monitoring, and Azure Blob Storage schema backups.

Run with:
    uvicorn main:app --reload --port 8000

Or run via Docker:
    docker-compose up api

API Documentation:
    Swagger UI: http://localhost:8000/docs
    ReDoc:      http://localhost:8000/redoc
"""

import asyncio
from enum import Enum
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
from schema import get_schema_prompt
from storage import is_storage_configured, backup_schema_to_blob, list_blobs, CONTAINER_NAME
from cache import (
    get_semantic_cached_answer,
    set_semantic_cached_answer,
    get_all_semantic_cached_questions,
)

app = FastAPI(
    title="Talk2Tables Multi-Database AI Agent API",
    description="Chat with ANY database (PostgreSQL, MySQL, SQLite, MSSQL) using Azure OpenAI (text-to-SQL).",
    version="2.0",
)

FAILURE_MESSAGE = config.FAILURE_MESSAGE
MAX_SQL_ATTEMPTS = config.MAX_SQL_ATTEMPTS


class ModelChoice(str, Enum):
    model_1 = "model_1"
    model_2 = "model_2"


class QueryRequest(BaseModel):
    question: str
    model: ModelChoice = ModelChoice.model_1


class QueryResponse(BaseModel):
    answer: str
    cached: bool = False
    model_used: str = "model_1"
    sql: str | None = None
    sql_explanation: str | None = None
    row_count: int | None = None
    data: list | None = None


class ConnectionTestRequest(BaseModel):
    engine: str | None = None
    connection_url: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: str | None = None
    ssl_mode: str | None = None
    sqlite_path: str | None = None


@app.get("/")
async def root():
    return {
        "message": "Talk2Tables Multi-Database AI Agent API is running.",
        "active_database": get_active_connection_info(),
        "supported_engines": list(config.SUPPORTED_ENGINES.keys()),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/connection/status")
async def connection_status():
    ok, msg = await test_db_connection()
    cols = await fetch_schema_info()
    tables = {c["table_name"] for c in cols}
    info = get_active_connection_info()
    return {
        "connected": ok,
        "message": msg,
        "dialect": info["dialect"],
        "label": info["label"],
        "url": info["url"],
        "tables_count": len(tables),
    }


@app.post("/connection/test")
async def test_connection_endpoint(req: ConnectionTestRequest):
    if req.connection_url:
        target_url = req.connection_url
    elif req.engine:
        target_url, _ = config.build_db_url(
            engine=req.engine,
            host=req.host or "localhost",
            port=req.port,
            database=req.database or "",
            username=req.username or "",
            password=req.password or "",
            ssl_mode=req.ssl_mode,
            sqlite_path=req.sqlite_path or "",
        )
    else:
        target_url = None

    ok, msg = await test_db_connection(target_url)
    return {"connected": ok, "message": msg}


@app.post("/connection/switch")
async def switch_connection_endpoint(req: ConnectionTestRequest):
    if req.connection_url:
        target_url, dialect = config.normalize_db_url(req.connection_url)
        target_label = f"{dialect.upper()} Custom Connection"
    elif req.engine:
        target_url, target_label = config.build_db_url(
            engine=req.engine,
            host=req.host or "localhost",
            port=req.port,
            database=req.database or "",
            username=req.username or "",
            password=req.password or "",
            ssl_mode=req.ssl_mode,
            sqlite_path=req.sqlite_path or "",
        )
    else:
        raise HTTPException(status_code=400, detail="Must provide either connection_url or engine credentials.")

    ok, msg = await test_db_connection(target_url)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Cannot connect to database: {msg}")

    set_database_connection(target_url, target_label)
    return {
        "success": True,
        "label": target_label,
        "dialect": get_active_dialect(),
    }


@app.get("/cached-questions")
async def cached_questions(model: ModelChoice = ModelChoice.model_1):
    return await get_all_semantic_cached_questions(model=model.value)


@app.get("/storage/status")
async def storage_status():
    configured = is_storage_configured()
    blobs = await list_blobs() if configured else []
    return {
        "configured": configured,
        "container": CONTAINER_NAME,
        "blob_count": len(blobs),
    }


@app.post("/storage/backup-schema")
async def backup_schema_endpoint():
    if not is_storage_configured():
        raise HTTPException(
            status_code=400,
            detail="Azure Blob Storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING in .env",
        )
    schema_prompt = await get_schema_prompt()
    url = await backup_schema_to_blob(schema_prompt)
    if not url:
        raise HTTPException(status_code=500, detail="Failed to upload schema backup to Azure Blob Storage.")
    return {"status": "success", "blob_url": url}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    # 0. Safety guardrail
    is_safe, error_msg = is_safe_question(req.question)
    if not is_safe:
        return QueryResponse(answer=error_msg, model_used=req.model.value)

    # 1. Semantic cache lookup
    if config.ENABLE_SEMANTIC_CACHE:
        cached = await get_semantic_cached_answer(
            req.question, model=req.model.value, threshold=config.SEMANTIC_SIMILARITY_THRESHOLD
        )
        if cached:
            return QueryResponse(answer=cached, cached=True, model_used=req.model.value)

    # 2. Rewrite query
    if config.ENABLE_QUERY_REWRITE:
        rewritten = await rewrite_question(req.question, model=req.model.value)
    else:
        rewritten = req.question

    # 3. Text-to-SQL self-correction loop
    last_sql = None
    history = []
    for _ in range(MAX_SQL_ATTEMPTS):
        sql = await generate_sql(rewritten, history=history, model=req.model.value)
        last_sql = sql
        if sql.strip().upper() == "REJECTED":
            return QueryResponse(
                answer="Query rejected: only single SELECT statements are allowed.",
                sql=sql,
                model_used=req.model.value,
            )

        rows, error = await execute_sql(sql)
        if error and "Query rejected" in error:
            return QueryResponse(answer=error, sql=sql, model_used=req.model.value)

        if not error:
            answer, sql_explanation = await asyncio.gather(
                generate_answer(req.question, rows, model=req.model.value),
                explain_sql(sql, question=req.question, model=req.model.value),
            )
            if config.ENABLE_SEMANTIC_CACHE:
                await set_semantic_cached_answer(req.question, answer, model=req.model.value)
            return QueryResponse(
                answer=answer,
                model_used=req.model.value,
                sql=sql,
                sql_explanation=sql_explanation,
                row_count=len(rows),
                data=rows,
            )
        history.append((sql, error))

    return QueryResponse(answer=FAILURE_MESSAGE, sql=last_sql, model_used=req.model.value)