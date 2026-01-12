import asyncio
import logging

import asyncpg

from app.config import settings

_pool: asyncpg.Pool | None = None
_logger = logging.getLogger(__name__)


async def init_db() -> None:
    global _pool
    if _pool:
        return
    last_error: Exception | None = None
    delay = 0.5
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            _pool = await asyncpg.create_pool(dsn=settings.database_url, min_size=1, max_size=10)
            return
        except Exception as exc:
            last_error = exc
            _logger.warning(
                "Database not ready, retrying in %.1fs (%d/%d)",
                delay,
                attempt,
                max_attempts,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
    if last_error:
        raise last_error


async def close_db() -> None:
    if _pool:
        await _pool.close()


async def fetch(query: str, *args):
    if not _pool:
        raise RuntimeError("Database pool is not initialized")
    async with _pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args):
    if not _pool:
        raise RuntimeError("Database pool is not initialized")
    async with _pool.acquire() as conn:
        return await conn.fetchrow(query, *args)
