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
import time
import json
import hashlib
import numpy as np
import redis.asyncio as redis

import config
from embeddings import get_embedding

_redis_clients: dict[asyncio.AbstractEventLoop | None, redis.Redis] = {}
_memory_cache: dict[str, dict] = {}

# Circuit breaker for Redis availability
_redis_last_check = 0.0
_redis_online = None


def get_redis_client() -> redis.Redis:
    """Return a Redis client bound to the current running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    client = _redis_clients.get(loop)
    if client is None:
        host = "127.0.0.1" if config.REDIS_HOST in ("localhost", "127.0.0.1") else config.REDIS_HOST
        client = redis.Redis(
            host=host,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=0.3,
            socket_timeout=0.3,
        )
        _redis_clients[loop] = client
    return client


class _RedisProxy:
    """Proxy object ensuring Redis operations always use the client bound to the active event loop."""

    def __getattr__(self, name):
        return getattr(get_redis_client(), name)


redis_client = _RedisProxy()


async def is_redis_available() -> bool:
    """Check if Redis connection can be pinged with quick timeout and circuit breaker."""
    global _redis_last_check, _redis_online
    now = time.time()
    if _redis_online is not None and (now - _redis_last_check) < 15.0:
        return _redis_online

    try:
        r = get_redis_client()
        ok = await asyncio.wait_for(r.ping(), timeout=0.3)
        _redis_online = bool(ok)
    except Exception:
        _redis_online = False
    _redis_last_check = now
    return _redis_online


async def get_semantic_cached_answer(
    question: str,
    model: str = "model_1",
    threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD,
) -> str | None:
    """Find the closest semantically equivalent cached question above threshold."""
    now = time.time()
    q_norm_text = question.strip().lower()
    q_hash = hashlib.md5(q_norm_text.encode()).hexdigest()
    exact_key = f"{config.SEMANTIC_CACHE_KEY_PREFIX}:{model}:{q_hash}"

    # 1. Instant exact match in memory (<0.1ms)
    if exact_key in _memory_cache:
        item = _memory_cache[exact_key]
        if (now - item.get("ts", 0)) < config.CACHE_TTL_SECONDS:
            return item.get("answer")
        else:
            del _memory_cache[exact_key]

    # 2. Extract active in-memory entries for semantic matching
    active_memory = [
        item for item in _memory_cache.values()
        if item.get("model") == model and (now - item.get("ts", 0)) < config.CACHE_TTL_SECONDS
    ]

    # 3. Check Redis keys only if Redis is confirmed online
    redis_available = await is_redis_available()
    redis_keys = []
    r = None
    if redis_available:
        try:
            r = get_redis_client()
            prefix = f"{config.SEMANTIC_CACHE_KEY_PREFIX}:{model}:"
            redis_keys = await asyncio.wait_for(r.keys(f"{prefix}*"), timeout=0.4)
        except Exception:
            redis_keys = []

    # If cache is completely empty, exit early without calling OpenAI embedding API
    if not active_memory and not redis_keys:
        return None

    # Compute question embedding for similarity comparison
    try:
        query_vec = np.array(await get_embedding(question), dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None
    except Exception:
        return None

    best_similarity = -1.0
    best_answer = None

    # Compare against in-memory embeddings
    for item in active_memory:
        cached_vec = np.array(item["embedding"], dtype=np.float32)
        cached_norm = np.linalg.norm(cached_vec)
        if cached_norm == 0:
            continue
        sim = float(np.dot(query_vec, cached_vec) / (query_norm * cached_norm))
        if sim > best_similarity:
            best_similarity = sim
            best_answer = item["answer"]

    # Compare against Redis embeddings if available
    if r and redis_keys:
        for key in redis_keys:
            try:
                raw_data = await asyncio.wait_for(r.get(key), timeout=0.3)
                if not raw_data:
                    continue
                entry = json.loads(raw_data)
                cached_vec = np.array(entry["embedding"], dtype=np.float32)
                cached_norm = np.linalg.norm(cached_vec)
                if cached_norm == 0:
                    continue
                sim = float(np.dot(query_vec, cached_vec) / (query_norm * cached_norm))
                if sim > best_similarity:
                    best_similarity = sim
                    best_answer = entry.get("answer")
            except Exception:
                continue

    if best_similarity >= threshold and best_answer:
        return best_answer
    return None


async def set_semantic_cached_answer(
    question: str,
    answer: str,
    model: str = "model_1",
) -> bool:
    """Save question embedding and answer to both in-memory cache and Redis."""
    try:
        now = time.time()
        embedding = await get_embedding(question)
        data = {
            "question": question,
            "answer": answer,
            "embedding": embedding,
            "model": model,
            "ts": now,
        }
        q_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()
        key = f"{config.SEMANTIC_CACHE_KEY_PREFIX}:{model}:{q_hash}"

        # 1. Always store in memory (instant)
        _memory_cache[key] = data

        # 2. Also save to Redis if available
        if await is_redis_available():
            try:
                r = get_redis_client()
                await asyncio.wait_for(
                    r.setex(key, config.CACHE_TTL_SECONDS, json.dumps(data)),
                    timeout=0.5,
                )
            except Exception:
                pass
        return True
    except Exception:
        return False


async def clear_cache() -> bool:
    """Clear all semantic cache keys in memory and Redis."""
    _memory_cache.clear()
    try:
        if await is_redis_available():
            r = get_redis_client()
            keys = await r.keys(f"{config.SEMANTIC_CACHE_KEY_PREFIX}:*")
            if keys:
                await r.delete(*keys)
        return True
    except Exception:
        return False


async def get_all_semantic_cached_questions(model: str = "model_1") -> list[str]:
    """Return a list of all question texts currently stored in the cache."""
    now = time.time()
    questions = [
        item["question"] for item in _memory_cache.values()
        if item.get("model") == model and (now - item.get("ts", 0)) < config.CACHE_TTL_SECONDS and "question" in item
    ]
    if await is_redis_available():
        try:
            r = get_redis_client()
            prefix = f"{config.SEMANTIC_CACHE_KEY_PREFIX}:{model}:"
            keys = await r.keys(f"{prefix}*")
            for key in keys:
                raw = await r.get(key)
                if raw:
                    entry = json.loads(raw)
                    q = entry.get("question")
                    if q and q not in questions:
                        questions.append(q)
        except Exception:
            pass
    return questions
