import hashlib
import json

from redis.asyncio.cluster import ClusterNode, RedisCluster

_client: RedisCluster | None = None


def _parse_nodes(nodes_str: str) -> list[ClusterNode]:
    nodes = []
    for part in nodes_str.split(","):
        host, port = part.strip().split(":")
        nodes.append(ClusterNode(host, int(port)))
    return nodes


async def init_cache(cluster_nodes_str: str) -> None:
    global _client
    _client = RedisCluster(
        startup_nodes=_parse_nodes(cluster_nodes_str),
        decode_responses=True,
    )


async def close_cache() -> None:
    if _client is not None:
        await _client.aclose()


async def get_json(key: str):
    if _client is None:
        return None
    raw = await _client.get(key)
    return json.loads(raw) if raw is not None else None


async def set_json(key: str, value, ttl_seconds: int) -> None:
    if _client is None:
        return
    await _client.setex(key, ttl_seconds, json.dumps(value))


def smiles_key(smiles: str) -> str:
    digest = hashlib.sha256(smiles.encode()).hexdigest()
    return f"pred:raman:smiles:{digest}"
