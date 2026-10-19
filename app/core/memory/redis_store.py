"""
Redis 对话缓存 - 存储最近对话历史，提供快速访问
"""
import asyncio
import json
import logging
from typing import List, Dict, Optional
import redis.asyncio as redis

from app.config.settings import settings

logger = logging.getLogger(__name__)

# LLM 派生结果(查询改写等)的缓存时长。比会话缓存长得多: 评测要能跨小时重跑得到
# 同样的输入; 而缓存失效的代价只是"重新问模型一次", 不存在正确性风险, 所以可以放心长。
CACHE_TTL_SECONDS = 7 * 24 * 3600


class RedisStore:
    """Redis 对话缓存管理器"""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._loop = None  # 客户端所属的事件循环
        self._ttl = 3600  # 缓存过期时间：1小时

    async def get_client(self) -> redis.Redis:
        """懒加载 Redis 连接。

        必须**跟着当前事件循环重建**: redis.asyncio 的连接池绑在创建它的循环上,
        换循环再用来会直接抛 `Event loop is closed`。这个坑是静默的 —— 缓存写失败
        被 cache_set 的吞异常逻辑吃掉, 表现就是"缓存看着接上了却永远不命中"。
        生产(uvicorn 单一长循环)碰不到, 但评测脚本、探针、任何多次 asyncio.run 的
        进程都会踩, 而缓存的价值恰恰要在评测里才用得上。
        """
        loop = asyncio.get_running_loop()
        if self._redis is not None and self._loop is not loop:
            stale, self._redis = self._redis, None
            try:
                await stale.aclose()
            except Exception:  # noqa: BLE001 - 旧连接关掉失败不影响重建
                pass
        if self._redis is None:
            self._redis = redis.from_url(
                settings.redis_url,
                decode_responses=True,
            )
            self._loop = loop
        return self._redis

    def _key(self, conversation_id: str) -> str:
        """生成 Redis key"""
        return f"rag:conversation:{conversation_id}:messages"

    async def save_messages(
        self, conversation_id: str, messages: List[Dict[str, str]]
    ):
        """
        保存对话消息到 Redis
        Args:
            conversation_id: 会话ID
            messages: 消息列表
        """
        client = await self.get_client()
        key = self._key(conversation_id)
        await client.setex(key, self._ttl, json.dumps(messages, ensure_ascii=False))

    async def get_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        从 Redis 获取对话历史
        Args:
            conversation_id: 会话ID
        Returns:
            消息列表
        """
        client = await self.get_client()
        key = self._key(conversation_id)
        data = await client.get(key)
        if data:
            return json.loads(data)
        return []

    async def append_message(
        self, conversation_id: str, message: Dict[str, str]
    ):
        """
        追加单条消息
        Args:
            conversation_id: 会话ID
            message: 消息字典
        """
        messages = await self.get_messages(conversation_id)
        messages.append(message)
        await self.save_messages(conversation_id, messages)

    async def delete_messages(self, conversation_id: str):
        """删除会话缓存"""
        client = await self.get_client()
        key = self._key(conversation_id)
        await client.delete(key)

    # ===== 通用缓存(查询改写等 LLM 派生结果) =====
    #
    # 与上面会话方法最大的区别:**任何异常都吞掉、当成未命中**。
    # 缓存是加速件不是数据源, Redis 挂了顶多多问模型一次;
    # 要是让它把整次问答带崩, 那是把加速件升级成了单点故障。
    def _cache_key(self, key: str) -> str:
        return f"rag:cache:{key}"

    async def cache_get(self, key: str) -> Optional[dict]:
        try:
            client = await self.get_client()
            data = await client.get(self._cache_key(key))
            return json.loads(data) if data else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RedisStore] 读缓存失败(按未命中处理): %s", exc)
            return None

    async def cache_set(self, key: str, value: dict, ttl: int = CACHE_TTL_SECONDS) -> None:
        try:
            client = await self.get_client()
            await client.setex(self._cache_key(key), ttl, json.dumps(value, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RedisStore] 写缓存失败(忽略, 不影响本次回答): %s", exc)

    async def ping(self) -> bool:
        """检查 Redis 连接"""
        try:
            client = await self.get_client()
            return await client.ping()
        except Exception:
            return False


# 全局实例
redis_store = RedisStore()
