
import json
import hashlib
import logging

import numpy as np
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import REDIS_URL


logger = logging.getLogger(__name__)

# Redis connection pool
redis_client = Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    health_check_interval=30,
)


def cosine_similarity(
    a: list[float],
    b: list[float]
) -> float:
    """Calculate cosine similarity between two embeddings."""

    vector_a = np.asarray(a, dtype=np.float32)
    vector_b = np.asarray(b, dtype=np.float32)

    if vector_a.shape != vector_b.shape:
        raise ValueError("Embedding dimensions must match.")

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

    return float(similarity)


async def get_semantic_cached_answer(
    query_embedding: list[float],
    threshold: float = 0.88
) -> dict | None:
    """
    Search Redis for semantically similar cached queries.

    Returns cached answer if similarity >= threshold.
    """

    try:
        async for key in redis_client.scan_iter(
            match="query_cache:*",
            count=100
        ):
            cached_value = await redis_client.get(key)

            if not cached_value:
                continue

            try:
                cached_data = json.loads(cached_value)

                cached_embedding = cached_data["embedding"]

                similarity = cosine_similarity(
                    query_embedding,
                    cached_embedding
                )

                if similarity >= threshold:
                    logger.info(
                        "Semantic cache hit. Similarity: %.4f",
                        similarity
                    )

                    return {
                        "cached": True,
                        "similarity": similarity,
                        "answer": cached_data["answer"],
                        "sources": cached_data["sources"]
                    }

            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                logger.warning("Skipping invalid cache entry: %s", key)
                continue

        logger.info("Semantic cache miss.")
        return None

    except RedisError:
        logger.exception("Redis semantic cache lookup failed.")
        return None


async def set_semantic_cache(
    query_text: str,
    query_embedding: list[float],
    answer: str,
    sources: list[str],
    ttl: int = 3600
) -> bool:
    """
    Store query, embedding and generated answer in Redis.

    Default TTL: 1 hour.
    """

    try:
        normalized_query = query_text.strip().lower()

        # Stable cache key using SHA-256
        query_hash = hashlib.sha256(
            normalized_query.encode("utf-8")
        ).hexdigest()

        cache_key = f"query_cache:{query_hash}"

        cache_data = {
            "query": query_text,
            "embedding": query_embedding,
            "answer": answer,
            "sources": sources
        }

        await redis_client.set(
            cache_key,
            json.dumps(cache_data),
            ex=ttl
        )

        logger.info("Semantic cache entry stored: %s", cache_key)

        return True

    except (RedisError, TypeError, ValueError):
        logger.exception("Failed to store semantic cache entry.")
        return False


async def close_redis_connection():
    """Close Redis connection gracefully."""

    await redis_client.aclose()
