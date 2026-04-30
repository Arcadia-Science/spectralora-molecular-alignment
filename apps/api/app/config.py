import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_cluster_nodes: str
    model_url: str
    parquet_dir: str
    cache_ttl_seconds: int
    database_min_pool_size: int
    database_max_pool_size: int
    model_timeout_seconds: float
    model_max_connections: int
    model_max_keepalive_connections: int
    model_max_concurrency: int
    model_retry_attempts: int
    model_retry_base_delay_seconds: float
    model_retry_max_delay_seconds: float


settings = Settings(
    database_url=os.getenv("DATABASE_URL", "postgresql://detanet:detanet@localhost:5432/detanet"),
    redis_cluster_nodes=os.getenv("REDIS_CLUSTER_NODES", "localhost:6379"),
    model_url=os.getenv("MODEL_URL", "http://localhost:8001"),
    parquet_dir=os.getenv("PARQUET_DIR", "/data/processed"),
    cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
    database_min_pool_size=int(os.getenv("DATABASE_MIN_POOL_SIZE", "1")),
    database_max_pool_size=int(os.getenv("DATABASE_MAX_POOL_SIZE", "10")),
    model_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "120")),
    model_max_connections=int(os.getenv("MODEL_MAX_CONNECTIONS", "20")),
    model_max_keepalive_connections=int(os.getenv("MODEL_MAX_KEEPALIVE_CONNECTIONS", "10")),
    model_max_concurrency=int(os.getenv("MODEL_MAX_CONCURRENCY", "8")),
    model_retry_attempts=int(os.getenv("MODEL_RETRY_ATTEMPTS", "6")),
    model_retry_base_delay_seconds=float(os.getenv("MODEL_RETRY_BASE_DELAY_SECONDS", "0.5")),
    model_retry_max_delay_seconds=float(os.getenv("MODEL_RETRY_MAX_DELAY_SECONDS", "5")),
)
