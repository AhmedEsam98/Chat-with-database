"""
Semantic Redis caching system for text-to-SQL responses.

Implements semantic vector caching (cosine similarity on Azure OpenAI embeddings)
to return instant answers for repeated or paraphrased database queries.

Usage:
    from cache import (
        get_semantic_cached_answer,
        set_semantic_cached_answer,
        clear_cache,
        is_redis_available,
        get_all_semantic_cached_questions,
    )
"""

import asyncio
import os
import json
import hashlib
import numpy as np
import redis.asyncio as redis

import config
from embeddings import get_embedding

_redis_clients: dict[asyncio.AbstractEventLoop | None, redis.Redis] = {}


def get_redis_client() -> redis.Redis:
    """Return a Redis client bound to the current running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    client = _redis_clients.get(loop)
    if client is None:
        client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
        )
        _redis_clients[loop] = client
    return client


class _RedisProxy:
    """Proxy object ensuring Redis operations always use the client bound to the active event loop."""

    def __getattr__(self, name):
        return getattr(get_redis_client(), name)


redis_client = _RedisProxy()


async def is_redis_available() -> bool:
    """Check if Redis connection can be pinged."""
    try:
        r = get_redis_client()
        return bool(await r.ping())
    except Exception:
        return False


async def get_semantic_cached_answer(
    question: str,
    model: str = "model_1",
    threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD,
) -> str | None:
    """Find the closest semantically equivalent cached question above threshold."""
    try:
        r = get_redis_client()
        prefix = f"{config.SEMANTIC_CACHE_KEY_PREFIX}:{model}:"
        keys = await r.keys(f"{prefix}*")
        if not keys:
            return None

        query_vec = np.array(await get_embedding(question), dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None

        best_similarity = -1.0
        best_answer = None

        for key in keys:
            raw_data = await r.get(key)
            if not raw_data:
                continue
            try:
                entry = json.loads(raw_data)
                cached_vec = np.array(entry["embedding"], dtype=np.float32)
                cached_norm = np.linalg.norm(cached_vec)
                if cached_norm == 0:
                    continue
                similarity = float(np.dot(query_vec, cached_vec) / (query_norm * cached_norm))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_answer = entry.get("answer")
            except Exception:
                continue

        if best_similarity >= threshold and best_answer:
            return best_answer
        return None
    except Exception:
        return None


async def set_semantic_cached_answer(
    question: str,
    answer: str,
    model: str = "model_1",
) -> bool:
    """Save question embedding and answer to Redis."""
    try:
        r = get_redis_client()
        embedding = await get_embedding(question)
        data = {
            "question": question,
            "answer": answer,
            "embedding": embedding,
        }
        q_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()
        key = f"{config.SEMANTIC_CACHE_KEY_PREFIX}:{model}:{q_hash}"
        await r.setex(key, config.CACHE_TTL_SECONDS, json.dumps(data))
        return True
    except Exception:
        return False


async def clear_cache() -> bool:
    """Clear all semantic cache keys in Redis."""
    try:
        r = get_redis_client()
        keys = await r.keys(f"{config.SEMANTIC_CACHE_KEY_PREFIX}:*")
        if keys:
            await r.delete(*keys)
        return True
    except Exception:
        return False


async def get_all_semantic_cached_questions(model: str = "model_1") -> list[str]:
    """Return a list of all question texts currently stored in the cache."""
    try:
        r = get_redis_client()
        prefix = f"{config.SEMANTIC_CACHE_KEY_PREFIX}:{model}:"
        keys = await r.keys(f"{prefix}*")
        questions = []
        for key in keys:
            raw = await r.get(key)
            if raw:
                try:
                    entry = json.loads(raw)
                    if "question" in entry:
                        questions.append(entry["question"])
                except Exception:
                    continue
        return questions
    except Exception:
        return []
