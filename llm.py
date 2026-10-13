"""
Azure OpenAI LLM integration module for natural language processing and text-to-SQL.

Handles multi-model routing (gpt-4.1-mini & gpt-4.1-nano), query rewriting for schema
alignment, self-correcting SQL generation, and natural-language answer synthesis.

Usage:
    from llm import rewrite_question, generate_sql, generate_answer, explain_sql
"""

import os
import config
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from schema import get_schema_prompt

load_dotenv()

# --- Model 1 (default: gpt-4.1-mini) ---
client = AsyncAzureOpenAI(
    api_version=config.AZURE_OPENAI_API_VERSION,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_key=config.AZURE_OPENAI_API_KEY,
)
DEPLOYMENT = config.AZURE_OPENAI_DEPLOYMENT_NAME

# --- Model 2 (nano) ---
client_nano = None
if config.AZURE_OPENAI_NANO_API_KEY and config.AZURE_OPENAI_NANO_ENDPOINT:
    client_nano = AsyncAzureOpenAI(
        api_version=config.AZURE_OPENAI_NANO_API_VERSION or config.AZURE_OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_NANO_ENDPOINT,
        api_key=config.AZURE_OPENAI_NANO_API_KEY,
    )
DEPLOYMENT_NANO = config.AZURE_OPENAI_NANO_DEPLOYMENT_NAME or DEPLOYMENT


def _resolve(model: str):
    """Map the requested model name to its client + deployment."""
    if model == "model_2" and client_nano is not None:
        return client_nano, DEPLOYMENT_NANO
    return client, DEPLOYMENT


async def rewrite_question(
    question: str,
    model: str = "model_1",
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> str:
    """Rewrite the user's question for better SQL generation."""
    from db import is_safe_question
    is_safe, _ = is_safe_question(question)
    if not is_safe:
        return question

    active_client, active_deployment = _resolve(model)
    schema_context = await get_schema_prompt()

    response = await active_client.chat.completions.create(
        model=active_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query preprocessor for a text-to-SQL system. "
                    "Rewrite the user's natural-language question to be clearer "
                    "and more precise for SQL generation.\n\n"
                    "Rules:\n"
                    "- Expand common abbreviations and map to actual database "
                    "values (e.g. US -> USA, UK -> United Kingdom)\n"
                    "- Fix obvious typos\n"
                    "- If the user refers to a value, match it to the actual database values shown below\n"
                    "- Keep the original meaning - do NOT change what is being asked\n"
                    "- Return ONLY the rewritten question, nothing else\n"
                    "- If the question is already clear, return it unchanged\n\n"
                    f"DATABASE CONTEXT:\n{schema_context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


async def generate_sql(
    question: str,
    history: list = None,
    model: str = "model_1",
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> str:
    """Ask the LLM to translate a natural-language question into SQL."""
    from db import is_safe_question
    is_safe, _ = is_safe_question(question)
    if not is_safe:
        return "REJECTED"

    active_client, active_deployment = _resolve(model)
    schema_prompt = await get_schema_prompt()

    messages = [
        {"role": "system", "content": schema_prompt},
        {"role": "user", "content": f"Write a SQL query to answer this question: {question}"},
    ]

    if history:
        for sql, error in history:
            messages.append({"role": "assistant", "content": sql})
            messages.append({
                "role": "user",
                "content": (
                    f"That query failed with error: {error}. "
                    "Fix it and return ONLY the corrected raw SQL query."
                ),
            })

    response = await active_client.chat.completions.create(
        model=active_deployment,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


async def generate_answer(
    question: str,
    db_result,
    model: str = "model_1",
    max_sample_rows: int = config.MAX_SAMPLE_ROWS,
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> str:
    """Turn raw DB rows into a natural-language answer."""
    active_client, active_deployment = _resolve(model)

    if isinstance(db_result, list):
        total_rows = len(db_result)
        if total_rows > max_sample_rows:
            sample = db_result[:max_sample_rows]
            db_context = (
                f"Total rows retrieved from database: {total_rows}.\n"
                f"Showing a partial sample of the first {max_sample_rows} rows:\n{sample}"
            )
        else:
            db_context = f"Total rows: {total_rows}.\n{db_result}"
    else:
        db_context = str(db_result)

    response = await active_client.chat.completions.create(
        model=active_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly, read-only database assistant. Answer the user's question "
                    "in natural language using the provided database result.\n\n"
                    "Rules for presenting data:\n"
                    "- If the database result is large and only a partial sample is provided, clearly "
                    "state the total count of records found, highlight or summarize key entries from the sample, "
                    "and note that full records are available in their table view.\n"
                    "- Never explain, write, or suggest commands that modify or delete data."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nDatabase Result:\n{db_context}",
            },
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content


async def explain_sql(
    sql: str,
    question: str = "",
    model: str = "model_1",
    temperature: float = config.DEFAULT_TEMPERATURE,
) -> str:
    """Generate a clear, natural-language explanation of what the SQL query does."""
    if not sql or sql.strip().upper() == "REJECTED":
        return ""

    try:
        active_client, active_deployment = _resolve(model)

        user_prompt = f"SQL Query:\n```sql\n{sql}\n```"
        if question:
            user_prompt = f"User Question: {question}\n\n{user_prompt}"

        response = await active_client.chat.completions.create(
            model=active_deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a database and SQL expert. Explain the provided SQL query clearly and concisely. "
                        "Break down:\n"
                        "- What tables and columns are queried\n"
                        "- Any joins, relationships, filter conditions, groupings, or sorting applied\n"
                        "- How this query retrieves the answer to the user's question\n"
                        "Keep the explanation brief, well-structured (using bullet points), and easy to understand."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate query explanation: {e}"
