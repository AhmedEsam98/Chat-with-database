"""
Vector embeddings generation module using Azure OpenAI.

Generates dense vector embeddings for input text strings, used primarily by the
semantic caching layer to identify semantically equivalent questions.

Usage:
    Imported by cache.py:
        from embeddings import get_embedding
        vector = await get_embedding("Total revenue in 2023")
"""

import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

embedding_client = AsyncAzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


async def get_embedding(text: str) -> list[float]:
    response = await embedding_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text.strip().lower(),
    )
    return response.data[0].embedding
